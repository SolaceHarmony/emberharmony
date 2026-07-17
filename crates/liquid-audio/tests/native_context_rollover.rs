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
fn whole_action_reservation_evicts_before_any_new_pass() {
    let mut window = Window {
        capacity: 4,
        runway: 2,
        position: 4,
        cursor: 9,
        rope_base: 5,
        ..Window::default()
    };
    let movement = reserve(&mut window, 3);
    assert_eq!(
        (movement.dropped, movement.source, movement.retained),
        (3, 3, 1)
    );
    assert_eq!(movement.compact, 1);
    assert_eq!(
        (window.position, window.cursor, window.rope_base),
        (1, 9, 8)
    );
    let before = window;
    assert_eq!(
        unsafe { lfm_context_window_reserve(&mut window, 5, &mut Move::default()) },
        -28
    );
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
