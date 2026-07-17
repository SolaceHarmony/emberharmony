//! Synthetic gates for the native sliding-context state machine. These do not
//! need a checkpoint: they exercise the exact helper used by LfmConversation,
//! the overlap-safe KV compaction, and the architecture RoPE range leaf.

use liquid_audio::NativeVoiceSampling;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct Window {
    capacity: u64,
    runway: u64,
    position: u64,
    start: u64,
    cursor: u64,
    rope_base: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct Move {
    dropped: u64,
    source: u64,
    retained: u64,
    compact: u32,
    reserved: u32,
}

unsafe extern "C" {
    fn lfm_context_window_admit(window: *const Window, needed: usize) -> i32;
    fn lfm_context_window_prefill_chunk(
        window: *const Window,
        remaining: usize,
        max_rows: usize,
        out_rows: *mut usize,
    ) -> i32;
    fn lfm_context_window_reserve(window: *mut Window, needed: usize, movement: *mut Move) -> i32;
    fn lfm_context_window_commit(window: *mut Window) -> i32;
    fn lfm_context_compact_bf16(
        plane: *mut u16,
        heads: usize,
        head_stride: usize,
        head_dim: usize,
        source_row: usize,
        retained_rows: usize,
    ) -> i32;
    fn lfm_rope_table_f32(
        positions: usize,
        head_dim: usize,
        theta: f32,
        cosine: *mut f32,
        sine: *mut f32,
    ) -> i32;
    fn lfm_rope_range_f32(
        first_position: u64,
        positions: usize,
        head_dim: usize,
        theta: f32,
        cosine: *mut f32,
        sine: *mut f32,
    ) -> i32;
}

fn causal_lengths(mut window: Window, rows: usize, max_rows: usize) -> (Vec<u64>, Window) {
    assert_eq!(unsafe { lfm_context_window_admit(&window, rows) }, 0);
    let mut lengths = Vec::with_capacity(rows);
    let mut completed = 0;
    while completed < rows {
        let mut chunk = 0;
        assert_eq!(
            unsafe {
                lfm_context_window_prefill_chunk(&window, rows - completed, max_rows, &mut chunk)
            },
            0
        );
        assert!(chunk > 0 && chunk <= rows - completed);
        let prior = window.position;
        let movement = reserve(&mut window, chunk);
        for row in 0..chunk {
            lengths.push(window.position + row as u64 + 1);
        }
        for _ in 0..chunk {
            assert_eq!(unsafe { lfm_context_window_commit(&mut window) }, 0);
        }
        assert_eq!(
            movement.dropped,
            (prior + chunk as u64).saturating_sub(window.capacity)
        );
        completed += chunk;
    }
    (lengths, window)
}

fn reserve(window: &mut Window, needed: usize) -> Move {
    /* Keep the crate rlib (and therefore its native static archives) in this
     * integration test even though the probe ABI itself is deliberately private. */
    std::hint::black_box(std::mem::size_of::<NativeVoiceSampling>());
    let mut movement = Move::default();
    assert_eq!(
        unsafe { lfm_context_window_reserve(window, needed, &mut movement) },
        0
    );
    movement
}

fn append(window: &mut Window, plane: &mut [u16], stride: usize, token: u16) {
    let row = window.start as usize + window.position as usize;
    plane[row] = token;
    plane[stride + row] = 1000 + token;
    assert_eq!(unsafe { lfm_context_window_commit(window) }, 0);
}

#[test]
fn rollover_retains_the_exact_latest_window_and_monotonic_cursor() {
    const CAPACITY: usize = 4;
    const RUNWAY: usize = 2;
    const STRIDE: usize = CAPACITY + RUNWAY;
    let mut window = Window {
        capacity: CAPACITY as u64,
        runway: RUNWAY as u64,
        ..Window::default()
    };
    let mut keys = [0u16; STRIDE * 2];
    let convolution = [71u16, 72, 73, 74];

    for token in 0..CAPACITY as u16 {
        assert_eq!(reserve(&mut window, 1).dropped, 0);
        append(&mut window, &mut keys, STRIDE, token);
    }
    for token in 4..=5u16 {
        let movement = reserve(&mut window, 1);
        assert_eq!((movement.dropped, movement.compact), (1, 0));
        append(&mut window, &mut keys, STRIDE, token);
    }

    let movement = reserve(&mut window, 1);
    assert_eq!(
        (
            movement.dropped,
            movement.source,
            movement.retained,
            movement.compact
        ),
        (1, 3, 3, 1)
    );
    assert_eq!(
        unsafe {
            lfm_context_compact_bf16(
                keys.as_mut_ptr(),
                2,
                STRIDE,
                1,
                movement.source as usize,
                movement.retained as usize,
            )
        },
        0
    );
    append(&mut window, &mut keys, STRIDE, 6);

    assert_eq!(&keys[..CAPACITY], &[3, 4, 5, 6]);
    assert_eq!(&keys[STRIDE..STRIDE + CAPACITY], &[1003, 1004, 1005, 1006]);
    assert_eq!(
        convolution,
        [71, 72, 73, 74],
        "KV rollover must not reset short-conv carry"
    );
    assert_eq!(window.position, CAPACITY as u64);
    assert_eq!(window.start, 0);
    assert_eq!(window.cursor, 7);
    assert_eq!(window.rope_base, 3);
    assert_eq!(window.rope_base + window.position, window.cursor);
}

#[test]
fn whole_action_admission_does_not_evict_before_the_first_pass() {
    let window = Window {
        capacity: 4,
        runway: 2,
        position: 4,
        cursor: 9,
        rope_base: 5,
        ..Window::default()
    };
    let before = window;
    assert_eq!(unsafe { lfm_context_window_admit(&window, 3) }, 0);
    assert_eq!(
        (
            window.position,
            window.start,
            window.cursor,
            window.rope_base,
        ),
        (
            before.position,
            before.start,
            before.cursor,
            before.rope_base,
        ),
        "whole-action validation must not discard causal history"
    );
    assert_eq!(unsafe { lfm_context_window_admit(&window, 5) }, -28);
    assert_eq!(
        (
            window.position,
            window.start,
            window.cursor,
            window.rope_base
        ),
        (
            before.position,
            before.start,
            before.cursor,
            before.rope_base
        ),
        "rejected whole-action admission must not mutate context state"
    );
}

#[test]
fn batched_rollover_matches_sequential_causal_window_lengths() {
    let full = Window {
        capacity: 4,
        runway: 4,
        position: 4,
        cursor: 4,
        ..Window::default()
    };
    let (batched, batched_window) = causal_lengths(full, 3, 4);
    let (sequential, sequential_window) = causal_lengths(full, 3, 1);
    assert_eq!(batched, sequential);
    assert_eq!(batched, [4, 4, 4]);
    assert_eq!(
        (
            batched_window.position,
            batched_window.start,
            batched_window.cursor,
            batched_window.rope_base,
        ),
        (
            sequential_window.position,
            sequential_window.start,
            sequential_window.cursor,
            sequential_window.rope_base,
        )
    );
    assert_eq!(batched_window.cursor, 7);
    assert_eq!(batched_window.rope_base, 3);
}

#[test]
fn cursor_overflow_is_rejected_without_mutating_the_window() {
    let mut window = Window {
        capacity: 4,
        runway: 2,
        position: 3,
        cursor: u64::MAX - 1,
        rope_base: u64::MAX - 4,
        ..Window::default()
    };
    let before = window;
    let mut movement = Move::default();
    assert_eq!(
        unsafe { lfm_context_window_reserve(&mut window, 2, &mut movement) },
        -libc::EOVERFLOW
    );
    assert_eq!(
        (
            window.position,
            window.start,
            window.cursor,
            window.rope_base,
        ),
        (
            before.position,
            before.start,
            before.cursor,
            before.rope_base,
        )
    );
    assert_eq!(movement.dropped, 0);
}

#[test]
fn absolute_rope_range_is_bit_exact_with_the_zero_based_table() {
    const POSITIONS: usize = 13;
    const HEAD: usize = 8;
    const HALF: usize = HEAD / 2;
    let mut all_cos = [0.0f32; POSITIONS * HALF];
    let mut all_sin = [0.0f32; POSITIONS * HALF];
    let mut range_cos = [0.0f32; 3 * HALF];
    let mut range_sin = [0.0f32; 3 * HALF];
    assert_eq!(
        unsafe {
            lfm_rope_table_f32(
                POSITIONS,
                HEAD,
                1_000_000.0,
                all_cos.as_mut_ptr(),
                all_sin.as_mut_ptr(),
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            lfm_rope_range_f32(
                7,
                3,
                HEAD,
                1_000_000.0,
                range_cos.as_mut_ptr(),
                range_sin.as_mut_ptr(),
            )
        },
        0
    );
    for index in 0..3 * HALF {
        assert_eq!(
            range_cos[index].to_bits(),
            all_cos[7 * HALF + index].to_bits()
        );
        assert_eq!(
            range_sin[index].to_bits(),
            all_sin[7 * HALF + index].to_bits()
        );
    }
}
