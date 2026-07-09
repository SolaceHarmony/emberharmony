//! PROTOTYPE / REFERENCE RUNG — NOT the live engine. This Rust-side kcoro chassis was
//! the E1 proof that the parked descriptor-flow model works and stays bit-exact; the
//! LIVE engine is the resident native stage machine (native/src/engine/flashkern_engine.cpp via
//! [`super::native_engine`]), which the model hot path reaches through
//! `process_engine()`. Keep this module for its tests (channel-dispatch parity vs the
//! direct kernels) and as the readable specification of the dispatch rules — do not
//! optimize it, and do not mount new passes here.
//!
//! The original chassis notes: persistent micro-kernel workers consume tile-job
//! DESCRIPTORS from channels and park when dry — no fork/join scopes, no spin
//! barriers, no payload copies; jobs carry addresses, results land in place.
//!
//! kcoro's runtime contract (kc_chan.c): channel send/recv REQUIRE coroutine context —
//! they park the calling coroutine, never a thread. So the split is exact:
//!   * inside kcoro — worker coroutines (recv → kernel → account) and a per-pass feeder
//!     coroutine (sends the tile descriptors);
//!   * outside kcoro — the Rust caller's whole surface is `kc_dispatcher_spawn_co`
//!     (mutex-guarded, external-thread-safe: kc_sched.c) plus ONE Condvar block until
//!     the pass completes. Zero spin on both sides of the boundary.
//!
//! Prototype mounted so far:
//!   * row-band GEMV tiles (the E1 smoke: linkage, dispatch model, descriptor handoff,
//!     pass-boundary handback, parity + throughput gates);
//!   * the fused-MLP decode block ([`TileEngine::fused_mlp`]) — the first token-pass
//!     stage. The threadgroup port's three spin-barriers become the kcoro-native
//!     pattern: a coordinator coroutine publishes each stage's tiles and PARKS on a
//!     completion channel between stages; workers never barrier at all. Stage math is
//!     verbatim [`super::decode::fused_mlp_decode`], so the result is bit-identical.

#![cfg(all(
    has_kcoro,
    any(
        all(target_arch = "aarch64", has_flashkern_neon),
        all(target_arch = "x86_64", has_flashkern_x86)
    )
))]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};

// ---- kcoro FFI (include/kcoro.h, kcoro_dispatch.h, kcoro_sched.h) --------------------
#[allow(non_camel_case_types)]
type kc_chan_t = c_void;
#[allow(non_camel_case_types)]
type kc_dispatcher_t = c_void;

extern "C" {
    fn kc_chan_make(out: *mut *mut kc_chan_t, kind: i32, elem_sz: usize, capacity: usize) -> i32;
    fn kc_chan_destroy(ch: *mut kc_chan_t);
    fn kc_chan_send(ch: *mut kc_chan_t, msg: *const c_void, timeout_ms: i64) -> i32;
    fn kc_chan_recv(ch: *mut kc_chan_t, out: *mut c_void, timeout_ms: i64) -> i32;
    fn kc_chan_close(ch: *mut kc_chan_t);
    fn kc_dispatcher_new(workers: i32) -> *mut kc_dispatcher_t;
    fn kc_dispatcher_release(dispatcher: *mut kc_dispatcher_t);
    fn kc_dispatcher_spawn_co(
        dispatcher: *mut kc_dispatcher_t,
        f: extern "C" fn(*mut c_void),
        arg: *mut c_void,
        stack_size: usize,
        out_co: *mut *mut c_void,
    ) -> i32;
}

/// KC_RENDEZVOUS, and deliberately so: in this kcoro build the rendezvous paths are the
/// ones that truly PARK on infinite timeouts (waiter token + `kcoro_park`, woken via
/// `kc_sched_enqueue_ready` — kc_chan.c:1046/1263), while the buffered infinite paths
/// yield-retry, i.e. cooperatively spin. Rendezvous is also the truer semantics for tile
/// flow: direct descriptor handoff, feeder parks when every worker is busy
/// (backpressure), workers park when the queue is dry. Zero spin on both sides.
const KC_RENDEZVOUS: i32 = 0;
/// kcoro's default coroutine stack is 64 KiB (mmap + guard page). The kernels are
/// register machines with byte-scale locals; double it for headroom and forget it.
const CO_STACK: usize = 128 * 1024;

// Job kinds. GEMV carries its operands inline; the MLP stages carry a pass-context
// pointer and a tile/band range — payloads never ride the channel, only descriptors.
const JK_GEMV: u32 = 1;
const JK_MLP_SUMSQ: u32 = 2; // grid-stride Σx² partial for tile r0
const JK_MLP_NORM: u32 = 3; // grid-stride rms-norm apply for tile r0
const JK_MLP_GATEUP: u32 = 4; // gate/up rows [r0,r1): two nt dots + the silu ladder
const JK_MLP_DOWN: u32 = 5; // down rows [r0,r1): nt dot + residual, straight to out

/// A tile job: a POD descriptor — the addresses point into the weight mmap / arena
/// (usize so the type stays plain data through the channel's elem_sz memcpy; this
/// descriptor is the ONLY thing kcoro moves).
#[repr(C)]
#[derive(Clone, Copy)]
struct TileJob {
    kind: u32,
    r0: u32, // row band [r0, r1); for grid-stride stages r0 = tile index
    r1: u32,
    k: u32,      // GEMV reduction length
    a: usize,    // GEMV: x bits [k].       MLP: *const MlpPass
    b: usize,    // GEMV: w bits [n,k].     MLP: unused
    c: usize,    // GEMV: out f32 [n].      MLP: unused
    done: usize, // *mut kc_chan_t — per-pass completion channel; worker sends one token per job
}

/// Per-pass completion state. Lives on the caller's stack for the duration of the pass;
/// the pass COORDINATOR flips the flag under the mutex after draining every completion
/// token, which is the release edge the blocked caller wakes on (the standard condvar
/// handshake — safe to pop immediately).
struct PassSync {
    mu: Mutex<bool>,
    cv: Condvar,
}

/// GEMV coordinator context. Caller-stack-lived for the duration of the pass.
struct GemvCoord {
    jobs: *mut kc_chan_t,
    done: *mut kc_chan_t,
    window: usize,
    x: usize,
    w: usize,
    out: usize,
    n: usize,
    k: usize,
    band: usize,
    sync: *const PassSync,
}

/// Fused-MLP pass context: every pointer the stage kernels need. Caller-stack-lived;
/// scratch buffers are owned by the caller for the duration of the pass (arena later).
struct MlpPass {
    x: usize,      // bf16 bits [h]
    norm_w: usize, // bf16 bits [h]
    w1: usize,     // bf16 bits [i,h]
    w3: usize,     // bf16 bits [i,h]
    w2: usize,     // bf16 bits [h,i]
    h: usize,
    i: usize,
    eps: f32,
    tiles: usize,    // FIXED tile count — the deterministic partial/fold order
    partials: usize, // *mut f32 [tiles]
    xn: usize,       // *mut u16 [h]
    gu: usize,       // *mut f32 [2i] — g in [0,i), u in [i,2i)
    t: usize,        // *mut u16 [i]
    out: usize,      // *mut u16 [h]
    rs: AtomicU32,   // f32 bits of 1/sqrt(mean+eps); coordinator-published before NORM
}

/// Fused-MLP coordinator context.
struct MlpCoord {
    jobs: *mut kc_chan_t,
    done: *mut kc_chan_t,
    window: usize,
    pass: *const MlpPass,
    sync: *const PassSync,
}

extern "C" fn worker_main(arg: *mut c_void) {
    // One persistent micro-kernel coroutine: recv parks it when the queue is dry, a
    // published descriptor wakes it, channel close (KC_EPIPE) retires it. After every
    // job it sends one token on the pass's done channel — the coordinator's stage
    // accounting. Stage math is verbatim decode.rs::fused_mlp_decode (bit-identical).
    let jobs = arg as *mut kc_chan_t;
    loop {
        let mut job = std::mem::MaybeUninit::<TileJob>::uninit();
        // SAFETY: POD out-param recv; -1 = park until a job or close.
        let rc = unsafe { kc_chan_recv(jobs, job.as_mut_ptr() as *mut c_void, -1) };
        if rc != 0 {
            return; // closed
        }
        // SAFETY: rc == 0 ⇒ kcoro copied a full descriptor in.
        let job = unsafe { job.assume_init() };
        match job.kind {
            JK_GEMV => {
                let n = (job.r1 - job.r0) as usize;
                // SAFETY: descriptor produced by TileEngine from live slices; rows
                // r0..r1 in bounds by construction; buffers held until pass completion.
                unsafe {
                    super::decode::nt_rows(
                        job.a as *const u16,
                        (job.b as *const u16).add(job.r0 as usize * job.k as usize),
                        (job.c as *mut f32).add(job.r0 as usize),
                        n,
                        job.k as usize,
                    );
                }
            }
            JK_MLP_SUMSQ => {
                // SAFETY: pass outlives the block (caller parks until completion).
                let p = unsafe { &*(job.a as *const MlpPass) };
                let x = p.x as *const u16;
                let tile = job.r0 as usize;
                let mut sum = 0f32;
                let mut idx = tile;
                while idx < p.h {
                    // SAFETY: idx < h.
                    let v = super::decode::bf16_f32(unsafe { *x.add(idx) });
                    sum += v * v;
                    idx += p.tiles;
                }
                // SAFETY: partials[tile] is this tile's private slot.
                unsafe { *(p.partials as *mut f32).add(tile) = sum };
            }
            JK_MLP_NORM => {
                let p = unsafe { &*(job.a as *const MlpPass) };
                let x = p.x as *const u16;
                let nw = p.norm_w as *const u16;
                let xn = p.xn as *mut u16;
                let rs = f32::from_bits(p.rs.load(Ordering::Acquire));
                let tile = job.r0 as usize;
                let mut idx = tile;
                while idx < p.h {
                    // SAFETY: grid-stride cell idx is this tile's private slot.
                    unsafe {
                        let v = super::decode::bf16_f32(*x.add(idx))
                            * rs
                            * super::decode::bf16_f32(*nw.add(idx));
                        *xn.add(idx) = super::decode::rb_bits(v);
                    }
                    idx += p.tiles;
                }
            }
            JK_MLP_GATEUP => {
                let p = unsafe { &*(job.a as *const MlpPass) };
                let (r0, r1) = (job.r0 as usize, job.r1 as usize);
                let n = r1 - r0;
                // SAFETY: xn is stage-complete (coordinator ordering); g/u row ranges
                // are tile-private; w1/w3 row slices in-bounds by the entry asserts.
                unsafe {
                    let gu = p.gu as *mut f32;
                    let t = p.t as *mut u16;
                    super::decode::nt_rows(
                        p.xn as *const u16,
                        (p.w1 as *const u16).add(r0 * p.h),
                        gu.add(r0),
                        n,
                        p.h,
                    );
                    super::decode::nt_rows(
                        p.xn as *const u16,
                        (p.w3 as *const u16).add(r0 * p.h),
                        gu.add(p.i + r0),
                        n,
                        p.h,
                    );
                    for r in r0..r1 {
                        let g = super::decode::bf16_f32(super::decode::rb_bits(*gu.add(r)));
                        let sg = super::decode::rb_bits(g / (1.0 + (-g).exp()));
                        let u = super::decode::rb_bits(*gu.add(p.i + r));
                        *t.add(r) = super::decode::rb_bits(
                            super::decode::bf16_f32(sg) * super::decode::bf16_f32(u),
                        );
                    }
                }
            }
            JK_MLP_DOWN => {
                let p = unsafe { &*(job.a as *const MlpPass) };
                let (r0, r1) = (job.r0 as usize, job.r1 as usize);
                let n = r1 - r0;
                let mut y = vec![0f32; n]; // tile-private accumulator
                                           // SAFETY: t is stage-complete; w2 row slice in-bounds; out rows private.
                unsafe {
                    let x = p.x as *const u16;
                    let out = p.out as *mut u16;
                    super::decode::nt_rows(
                        p.t as *const u16,
                        (p.w2 as *const u16).add(r0 * p.i),
                        y.as_mut_ptr(),
                        n,
                        p.i,
                    );
                    for (j, &yv) in y.iter().enumerate() {
                        let d = super::decode::bf16_f32(super::decode::rb_bits(yv));
                        let r = super::decode::rb_bits(d + super::decode::bf16_f32(*x.add(r0 + j)));
                        *out.add(r0 + j) = r;
                    }
                }
            }
            _ => {}
        }
        // Publish completion: one token per job on the pass's done channel. Rendezvous
        // send parks this worker until the coordinator takes the handoff — zero-spin.
        let token = 1u8;
        // SAFETY: the done channel outlives the pass (caller-owned).
        unsafe {
            kc_chan_send(
                job.done as *mut kc_chan_t,
                &token as *const u8 as *const c_void,
                -1,
            )
        };
    }
}

// Coordinator-side helpers: publish one job / drain n completion tokens. Both run in
// coroutine context only.
unsafe fn co_send(jobs: *mut kc_chan_t, job: &TileJob) {
    let rc = kc_chan_send(jobs, job as *const TileJob as *const c_void, -1);
    debug_assert_eq!(rc, 0, "engine: job send failed rc={rc}");
}
unsafe fn co_drain(done: *mut kc_chan_t, n: usize) {
    for _ in 0..n {
        let mut token = 0u8;
        let rc = kc_chan_recv(done, &mut token as *mut u8 as *mut c_void, -1);
        debug_assert_eq!(rc, 0, "engine: done recv failed rc={rc}");
    }
}
/// Windowed publish — the completion-backpressure rule. Both channels are rendezvous,
/// and a worker parks on its done-token send until the coordinator drains it; so once
/// `window` (= worker count) jobs are outstanding, every further publish must drain one
/// completion first, unparking exactly one worker to rendezvous with the next job.
/// Publishing more than `window` without draining deadlocks: all workers parked on
/// done-sends, coordinator parked on a jobs-send no one can take.
unsafe fn co_send_windowed(
    jobs: *mut kc_chan_t,
    done: *mut kc_chan_t,
    job: &TileJob,
    outstanding: &mut usize,
    window: usize,
) {
    if *outstanding >= window {
        co_drain(done, 1);
        *outstanding -= 1;
    }
    co_send(jobs, job);
    *outstanding += 1;
}
/// Flip the pass condvar — the Rust caller's wake at the pass boundary.
fn pass_complete(sync: *const PassSync) {
    // SAFETY: sync outlives the pass; standard condvar handshake.
    let sync = unsafe { &*sync };
    let mut done = sync.mu.lock().unwrap();
    *done = true;
    sync.cv.notify_all();
}

extern "C" fn gemv_coord_main(arg: *mut c_void) {
    // GEMV pass coordinator: publish the band tiles, drain their completions, flip the
    // caller's condvar. Parks (never spins) at both channel edges.
    // SAFETY: ctx is the caller-stack GemvCoord, alive until the pass completes.
    let ctx = unsafe { &*(arg as *const GemvCoord) };
    let mut outstanding = 0usize;
    let mut r0 = 0usize;
    while r0 < ctx.n {
        let r1 = (r0 + ctx.band).min(ctx.n);
        let job = TileJob {
            kind: JK_GEMV,
            r0: r0 as u32,
            r1: r1 as u32,
            k: ctx.k as u32,
            a: ctx.x,
            b: ctx.w,
            c: ctx.out,
            done: ctx.done as usize,
        };
        // SAFETY: coroutine context; windowed against completion backpressure.
        unsafe { co_send_windowed(ctx.jobs, ctx.done, &job, &mut outstanding, ctx.window) };
        r0 = r1;
    }
    unsafe { co_drain(ctx.done, outstanding) };
    pass_complete(ctx.sync);
}

extern "C" fn mlp_coord_main(arg: *mut c_void) {
    // Fused-MLP block coordinator — the kcoro-native form of the threadgroup port's
    // three spin-barriers: publish a stage's tiles, PARK on the done channel until the
    // stage drains, then publish the next. The serial partial-fold between SUMSQ and
    // NORM runs here, in fixed tile order — the same deterministic order as
    // decode.rs::fused_mlp_decode's in-lane fold, so the pass is bit-identical.
    // SAFETY: ctx/pass are caller-stack, alive until the pass completes.
    let ctx = unsafe { &*(arg as *const MlpCoord) };
    let p = unsafe { &*ctx.pass };
    let tiles = p.tiles;
    let job0 = TileJob {
        kind: 0,
        r0: 0,
        r1: 0,
        k: 0,
        a: ctx.pass as usize,
        b: 0,
        c: 0,
        done: ctx.done as usize,
    };

    // Stage 1a: Σx² partials, one grid-stride tile each.
    let mut outstanding = 0usize;
    for l in 0..tiles {
        let job = TileJob {
            kind: JK_MLP_SUMSQ,
            r0: l as u32,
            ..job0
        };
        unsafe { co_send_windowed(ctx.jobs, ctx.done, &job, &mut outstanding, ctx.window) };
    }
    unsafe { co_drain(ctx.done, outstanding) };
    outstanding = 0;

    // Stage 1b (serial, deterministic): fold partials in tile order, publish rs.
    let mut total = 0f32;
    for l in 0..tiles {
        // SAFETY: stage-complete partials, read-only here.
        total += unsafe { *(p.partials as *const f32).add(l) };
    }
    let rs = 1.0f32 / (total / p.h as f32 + p.eps).sqrt();
    p.rs.store(rs.to_bits(), Ordering::Release);

    // Stage 1c: apply the norm, grid-stride tiles.
    for l in 0..tiles {
        let job = TileJob {
            kind: JK_MLP_NORM,
            r0: l as u32,
            ..job0
        };
        unsafe { co_send_windowed(ctx.jobs, ctx.done, &job, &mut outstanding, ctx.window) };
    }
    unsafe { co_drain(ctx.done, outstanding) };
    outstanding = 0;

    // Stage 2: gate/up row bands (contiguous, so nt streams contiguous weight rows).
    let i_chunk = p.i.div_ceil(tiles);
    for l in 0..tiles {
        let r0 = (l * i_chunk).min(p.i);
        let r1 = ((l + 1) * i_chunk).min(p.i);
        if r1 > r0 {
            let job = TileJob {
                kind: JK_MLP_GATEUP,
                r0: r0 as u32,
                r1: r1 as u32,
                ..job0
            };
            unsafe { co_send_windowed(ctx.jobs, ctx.done, &job, &mut outstanding, ctx.window) };
        }
    }
    unsafe { co_drain(ctx.done, outstanding) };
    outstanding = 0;

    // Stage 3: down rows + residual.
    let h_chunk = p.h.div_ceil(tiles);
    for l in 0..tiles {
        let r0 = (l * h_chunk).min(p.h);
        let r1 = ((l + 1) * h_chunk).min(p.h);
        if r1 > r0 {
            let job = TileJob {
                kind: JK_MLP_DOWN,
                r0: r0 as u32,
                r1: r1 as u32,
                ..job0
            };
            unsafe { co_send_windowed(ctx.jobs, ctx.done, &job, &mut outstanding, ctx.window) };
        }
    }
    unsafe { co_drain(ctx.done, outstanding) };

    pass_complete(ctx.sync);
}

/// The persistent tile engine: `workers` kcoro micro-kernel coroutines living for the
/// engine's lifetime, fed by one buffered descriptor channel. v0 exposes the row-band
/// GEMV pass; the token/frame passes mount here as further job kinds.
pub struct TileEngine {
    dispatcher: *mut kc_dispatcher_t,
    jobs: *mut kc_chan_t,
    workers: usize,
}

// SAFETY: the dispatcher and channel are kcoro's, thread-safe by the runtime's contract.
unsafe impl Send for TileEngine {}
unsafe impl Sync for TileEngine {}

impl TileEngine {
    pub fn new(workers: usize) -> Option<Self> {
        let workers = workers.clamp(1, 16);
        let mut jobs: *mut kc_chan_t = std::ptr::null_mut();
        // SAFETY: out-param construction; buffered channel of POD descriptors.
        let rc =
            unsafe { kc_chan_make(&mut jobs, KC_RENDEZVOUS, std::mem::size_of::<TileJob>(), 0) };
        if rc != 0 || jobs.is_null() {
            return None;
        }
        // SAFETY: kcoro dispatcher over `workers` OS worker threads.
        let dispatcher = unsafe { kc_dispatcher_new(workers as i32) };
        if dispatcher.is_null() {
            unsafe { kc_chan_destroy(jobs) };
            return None;
        }
        for _ in 0..workers {
            // SAFETY: spawn_co is external-thread-safe; the channel pointer outlives
            // the coroutines (Drop closes before destroying).
            let rc = unsafe {
                kc_dispatcher_spawn_co(
                    dispatcher,
                    worker_main,
                    jobs,
                    CO_STACK,
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                unsafe {
                    kc_chan_close(jobs);
                    kc_dispatcher_release(dispatcher);
                    kc_chan_destroy(jobs);
                }
                return None;
            }
        }
        Some(Self {
            dispatcher,
            jobs,
            workers,
        })
    }

    /// A fresh per-pass completion channel (rendezvous: worker sends park until the
    /// coordinator drains them — zero-spin on both sides).
    fn make_done(&self) -> *mut kc_chan_t {
        let mut done: *mut kc_chan_t = std::ptr::null_mut();
        // SAFETY: out-param construction; 1-byte tokens.
        let rc = unsafe { kc_chan_make(&mut done, KC_RENDEZVOUS, 1, 0) };
        assert!(
            rc == 0 && !done.is_null(),
            "engine: done channel create failed rc={rc}"
        );
        done
    }

    /// Park the caller until the coordinator flips the pass condvar, then free the
    /// pass's done channel.
    fn wait_pass(&self, sync: &PassSync, done: *mut kc_chan_t) {
        let mut finished = sync.mu.lock().unwrap();
        while !*finished {
            finished = sync.cv.wait(finished).unwrap();
        }
        drop(finished);
        // SAFETY: pass complete ⇒ no worker or coordinator still holds the channel.
        unsafe {
            kc_chan_close(done);
            kc_chan_destroy(done);
        }
    }

    /// One GEMV pass through the engine: spawn the coordinator (the single handoff),
    /// park on the pass condvar until the last tile lands. out = W[n,k]·x, bit-exact
    /// per row vs the direct kernel (band split never changes a row's reduction).
    pub fn gemv_nt(&self, x: &[u16], w_nk: &[u16], out: &mut [f32], n: usize, k: usize) {
        assert_eq!(x.len(), k, "engine gemv: x.len() != k");
        assert_eq!(w_nk.len(), n * k, "engine gemv: w.len() != n*k");
        assert_eq!(out.len(), n, "engine gemv: out.len() != n");
        if n == 0 {
            return;
        }
        // Small over-decomposition (4 bands per worker) so fast workers keep flowing
        // instead of waiting on the slowest — tile flow, not stage lockstep.
        let bands = (self.workers * 4).min(n);
        let band = n.div_ceil(bands);
        let sync = PassSync {
            mu: Mutex::new(false),
            cv: Condvar::new(),
        };
        let done = self.make_done();
        let ctx = GemvCoord {
            jobs: self.jobs,
            done,
            window: self.workers,
            x: x.as_ptr() as usize,
            w: w_nk.as_ptr() as usize,
            out: out.as_mut_ptr() as usize,
            n,
            k,
            band,
            sync: &sync,
        };
        // SAFETY: ctx and sync outlive the pass — this thread parks in wait_pass until
        // the coordinator flips the flag after draining every completion.
        let rc = unsafe {
            kc_dispatcher_spawn_co(
                self.dispatcher,
                gemv_coord_main,
                &ctx as *const GemvCoord as *mut c_void,
                CO_STACK,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, 0, "engine gemv: coordinator spawn failed rc={rc}");
        self.wait_pass(&sync, done);
    }

    /// One fused-MLP decode block through the engine — same contract and bit-identical
    /// result as [`super::decode::fused_mlp_decode`] at the same `lanes`, but dispatched
    /// as parked tile flow instead of a rayon fork/join with spin barriers.
    pub fn fused_mlp(
        &self,
        x: &[u16],
        w: &super::decode::FusedMlpWeights,
        out: &mut [u16],
        lanes: usize,
    ) {
        let h = x.len();
        let i = w.w1.len() / h;
        assert!(h > 0 && i > 0, "engine fused_mlp: empty dims");
        assert_eq!(w.norm_w.len(), h, "engine fused_mlp: norm_w.len() != H");
        assert_eq!(w.w1.len(), i * h, "engine fused_mlp: w1.len() != I·H");
        assert_eq!(w.w3.len(), i * h, "engine fused_mlp: w3.len() != I·H");
        assert_eq!(w.w2.len(), h * i, "engine fused_mlp: w2.len() != H·I");
        assert_eq!(out.len(), h, "engine fused_mlp: out.len() != H");
        let tiles = lanes.clamp(1, h.min(i));

        // Pass scratch: caller-frame Vecs for now (arena slots later). Same shapes as
        // the threadgroup port's shared scratch.
        let mut partials = vec![0f32; tiles];
        let mut xn = vec![0u16; h];
        let mut gu = vec![0f32; 2 * i];
        let mut t = vec![0u16; i];

        let sync = PassSync {
            mu: Mutex::new(false),
            cv: Condvar::new(),
        };
        let done = self.make_done();
        let pass = MlpPass {
            x: x.as_ptr() as usize,
            norm_w: w.norm_w.as_ptr() as usize,
            w1: w.w1.as_ptr() as usize,
            w3: w.w3.as_ptr() as usize,
            w2: w.w2.as_ptr() as usize,
            h,
            i,
            eps: w.eps,
            tiles,
            partials: partials.as_mut_ptr() as usize,
            xn: xn.as_mut_ptr() as usize,
            gu: gu.as_mut_ptr() as usize,
            t: t.as_mut_ptr() as usize,
            out: out.as_mut_ptr() as usize,
            rs: AtomicU32::new(0),
        };
        let ctx = MlpCoord {
            jobs: self.jobs,
            done,
            window: self.workers,
            pass: &pass,
            sync: &sync,
        };
        // SAFETY: pass/ctx/sync and the scratch Vecs outlive the pass — this thread
        // parks in wait_pass until the coordinator flips the flag after stage 3 drains.
        let rc = unsafe {
            kc_dispatcher_spawn_co(
                self.dispatcher,
                mlp_coord_main,
                &ctx as *const MlpCoord as *mut c_void,
                CO_STACK,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, 0, "engine fused_mlp: coordinator spawn failed rc={rc}");
        self.wait_pass(&sync, done);
    }
}

impl Drop for TileEngine {
    fn drop(&mut self) {
        // Close wakes every parked worker with KC_EPIPE → coroutines return; release
        // shuts the scheduler down (joins worker threads, destroys any never-resumed
        // ready coroutines); only then is the channel freed.
        unsafe {
            kc_chan_close(self.jobs);
            kc_dispatcher_release(self.dispatcher);
            kc_chan_destroy(self.jobs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_fused_mlp_bit_parity_with_threadgroup_port() {
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused mlp kernel unavailable — skipping");
            return;
        }
        let Some(engine) = TileEngine::new(8) else {
            eprintln!("kcoro engine init failed — skipping");
            return;
        };
        let rnd = |i: usize, seed: usize| -> u16 {
            bf16::from_f32(
                (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 2000) as f32 / 1000.0) - 1.0,
            )
            .to_bits()
        };
        for &(h, i) in &[(64usize, 96usize), (256, 512), (1024, 2048)] {
            let x: Vec<u16> = (0..h).map(|j| rnd(j, 1)).collect();
            let w = crate::flashkern::decode::FusedMlpWeights {
                norm_w: &(0..h).map(|j| rnd(j, 2)).collect::<Vec<_>>(),
                w1: &(0..i * h).map(|j| rnd(j, 3)).collect::<Vec<_>>(),
                w3: &(0..i * h).map(|j| rnd(j, 4)).collect::<Vec<_>>(),
                w2: &(0..h * i).map(|j| rnd(j, 5)).collect::<Vec<_>>(),
                eps: 1e-5,
            };
            for lanes in [1usize, 3, 8] {
                let mut want = vec![0u16; h];
                crate::flashkern::decode::fused_mlp_decode(&x, &w, &mut want, lanes);
                let mut got = vec![0u16; h];
                engine.fused_mlp(&x, &w, &mut got, lanes);
                assert_eq!(got, want, "H={h} I={i} lanes={lanes}");
            }
        }

        // Timing signal (printed, not gated) at the real decode shape.
        let (h, i) = (1024usize, 4096usize);
        let x: Vec<u16> = (0..h).map(|j| rnd(j, 1)).collect();
        let norm_w: Vec<u16> = (0..h).map(|j| rnd(j, 2)).collect();
        let w1: Vec<u16> = (0..i * h).map(|j| rnd(j, 3)).collect();
        let w3: Vec<u16> = (0..i * h).map(|j| rnd(j, 4)).collect();
        let w2: Vec<u16> = (0..h * i).map(|j| rnd(j, 5)).collect();
        let w = crate::flashkern::decode::FusedMlpWeights {
            norm_w: &norm_w,
            w1: &w1,
            w3: &w3,
            w2: &w2,
            eps: 1e-5,
        };
        let mut out = vec![0u16; h];
        let lanes = 8;
        let t = std::time::Instant::now();
        for _ in 0..50 {
            engine.fused_mlp(&x, &w, &mut out, lanes);
        }
        let eng_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
        let t = std::time::Instant::now();
        for _ in 0..50 {
            crate::flashkern::decode::fused_mlp_decode(&x, &w, &mut out, lanes);
        }
        let tg_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
        eprintln!(
            "engine fused_mlp {eng_ms:.3} ms vs threadgroup+spin {tg_ms:.3} ms (H=1024 I=4096, lanes=8)"
        );
    }

    #[test]
    fn engine_gemv_parity_and_flow() {
        use half::bf16;
        if !crate::bf16_gemm::bf16_gemm_nt_available() {
            eprintln!("nt kernel unavailable — skipping");
            return;
        }
        let Some(engine) = TileEngine::new(8) else {
            eprintln!("kcoro engine init failed — skipping");
            return;
        };
        let (n, k) = (2048usize, 2048usize);
        let x: Vec<u16> = (0..k)
            .map(|i| bf16::from_f32(((i * 37 % 23) as f32 / 23.0) - 0.5).to_bits())
            .collect();
        let w: Vec<u16> = (0..n * k)
            .map(|i| bf16::from_f32(((i * 31 % 19) as f32 / 19.0) - 0.5).to_bits())
            .collect();

        let mut want = vec![0f32; n];
        crate::flashkern::neon_or_x86_gemv(&x, &w, &mut want, n, k);

        let mut got = vec![0f32; n];
        engine.gemv_nt(&x, &w, &mut got, n, k);
        // Same kernel per row → bit-exact regardless of the band split.
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), r.to_bits(), "row {i}");
        }

        // Throughput signal (printed, not gated): engine flow vs the rayon fan-out.
        let t = std::time::Instant::now();
        for _ in 0..50 {
            engine.gemv_nt(&x, &w, &mut got, n, k);
        }
        let eng_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
        let t = std::time::Instant::now();
        for _ in 0..50 {
            crate::flashkern::neon_or_x86_gemv(&x, &w, &mut got, n, k);
        }
        let ray_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
        eprintln!("engine gemv {eng_ms:.3} ms vs rayon fan-out {ray_ms:.3} ms (N=K=2048)");
    }
}
