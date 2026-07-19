#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use liquid_audio as _;
use std::ffi::{c_char, CStr, CString};
use std::mem::size_of;

#[repr(C)]
struct MimiWeight {
    name: *const c_char,
    bytes: *const u8,
    shape: *const u64,
    ndim: u32,
    len: u64,
}

#[repr(C)]
struct MimiWeightTable {
    entries: *const MimiWeight,
    count: u32,
    bound: *mut u8,
}

#[repr(C)]
struct MimiArena {
    base: *mut u8,
    size: usize,
    used: usize,
    derived: *mut std::ffi::c_void,
    derived_cursor: usize,
}

#[repr(C)]
struct MimiConvTrState {
    _private: [u8; 0],
}

#[repr(C)]
struct MimiConvState {
    _private: [u8; 0],
}

#[repr(C)]
struct MimiUpsampleState {
    _private: [u8; 0],
}

extern "C" {
    fn mimi_conv_init(
        state: *mut *mut MimiConvState,
        weights: *const MimiWeightTable,
        prefix: *const c_char,
        input_channels: i32,
        output_channels: i32,
        kernel: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        causal: i32,
        matrix_workspace: *mut f32,
        matrix_workspace_floats: usize,
        arena: *mut MimiArena,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn mimi_conv_step(
        state: *mut MimiConvState,
        input: *const f32,
        input_frames: i32,
        output: *mut f32,
    ) -> i32;
    fn mimi_conv_reset(state: *mut MimiConvState);
    fn mimi_conv1d_carry_copy_bytes_saved() -> u64;
    fn mimi_conv_matrix_workspace_bytes_saved() -> u64;
    fn mimi_convtr_init(
        state: *mut *mut MimiConvTrState,
        weights: *const MimiWeightTable,
        prefix: *const c_char,
        input_channels: i32,
        output_channels: i32,
        kernel: i32,
        stride: i32,
        causal: i32,
        matrix_workspace: *mut f32,
        matrix_workspace_floats: usize,
        arena: *mut MimiArena,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn mimi_convtr_step(
        state: *mut MimiConvTrState,
        input: *const f32,
        input_frames: i32,
        output: *mut f32,
    ) -> i32;
    fn mimi_convtr_reset(state: *mut MimiConvTrState);
    fn mimi_conv_carry_copy_bytes_saved() -> u64;
    fn mimi_upsample_init(
        state: *mut *mut MimiUpsampleState,
        weights: *const MimiWeightTable,
        arena: *mut MimiArena,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn mimi_upsample_step(
        state: *mut MimiUpsampleState,
        input: *const f32,
        input_frames: i32,
        output: *mut f32,
    ) -> i32;
    fn mimi_upsample_reset(state: *mut MimiUpsampleState);
}

#[test]
fn carry_commit_rotates_dead_banks_and_accounts_removed_copy_traffic() {
    const SEANET_BYTES: u64 = (512 * 8 + 256 * 6 + 128 * 5 + 64 * 4) * 4;
    const UPSAMPLE_BYTES: u64 = 512 * 2 * 4;
    const CONV1D_BYTES: u64 = (512 * 6 + (512 + 256 + 128 + 64) * 2 + 64 * 2) * 4;
    const TRANSPOSED_CALLS: u64 = 512 + 256 + 128 + 64 + 1;
    const CONV1D_CALLS: u64 = 512 + 512 + 256 + 128 + 64 + 64;
    const CONV_MATRIX_BYTES: u64 = 10_485_760;
    const CONVTR_MATRIX_BYTES: u64 = 26_738_688;
    const MATRIX_BYTES: u64 = CONV_MATRIX_BYTES + CONVTR_MATRIX_BYTES;
    const SHARED_B2_BYTES: u64 = 245_760 * 4;

    assert_eq!(unsafe { mimi_conv_carry_copy_bytes_saved() }, 30_208);
    assert_eq!(unsafe { mimi_conv1d_carry_copy_bytes_saved() }, 20_480);
    assert_eq!(
        unsafe { mimi_conv_matrix_workspace_bytes_saved() },
        MATRIX_BYTES
    );
    assert_eq!(SEANET_BYTES + UPSAMPLE_BYTES, 30_208);
    assert_eq!(CONV1D_BYTES, 20_480);
    assert_eq!(CONV_MATRIX_BYTES, 10 * 1024 * 1024);
    assert_eq!(CONVTR_MATRIX_BYTES, 25 * 1024 * 1024 + 512 * 1024);
    assert_eq!(MATRIX_BYTES, 37_224_448);
    assert_eq!(MATRIX_BYTES - SHARED_B2_BYTES, 36_241_408);
    assert_eq!(TRANSPOSED_CALLS, 961);
    assert_eq!(CONV1D_CALLS, 1_536);

    let source = include_str!("../native/src/mimi/mimi_conv.cpp");
    assert_eq!(source.matches("publish(s,").count(), 3);
    assert_eq!(
        source.matches("flip(s->prev, s->carry_scratch);").count(),
        2
    );
    assert!(!source.contains("memcpy(s->prev + (size_t)c * s->carry_cap"));
    assert!(!source.contains("memcpy(s->prev, s->carry_scratch"));
    assert!(!source.contains("memcpy(s->prev + (size_t)oc * invalid"));
    assert!(!source.contains("MIMI_CONV_GEMM_MAX_N"));
    assert!(!source.contains("im2col"));
    assert!(!source.contains("g_gemm"));
}

struct Arena {
    _storage: Vec<u8>,
    raw: MimiArena,
}

impl Arena {
    fn new(size: usize) -> Self {
        let mut storage = vec![0u8; size + 63];
        let address = storage.as_mut_ptr() as usize;
        let offset = (64 - address % 64) % 64;
        let raw = MimiArena {
            base: unsafe { storage.as_mut_ptr().add(offset) },
            size,
            used: 0,
            derived: std::ptr::null_mut(),
            derived_cursor: 0,
        };
        Self {
            _storage: storage,
            raw,
        }
    }
}

fn resident(values: &[f32], skew: usize) -> (Vec<u8>, usize) {
    let mut storage = vec![0xa5; values.len() * size_of::<f32>() + 8];
    let address = storage.as_ptr() as usize;
    let aligned = (size_of::<f32>() - address % size_of::<f32>()) % size_of::<f32>();
    let offset = aligned + skew;
    for (index, value) in values.iter().enumerate() {
        let start = offset + index * size_of::<f32>();
        storage[start..start + size_of::<f32>()].copy_from_slice(&value.to_le_bytes());
    }
    (storage, offset)
}

fn message(error: &[c_char]) -> String {
    unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn check_conv(
    channels: usize,
    kernel: usize,
    stride: usize,
    values: &[f32],
    cases: &[(Vec<f32>, Vec<f32>)],
    reset: usize,
    size: usize,
) {
    for skew in [0usize, 1] {
        let name = CString::new("probe.conv.conv.weight").unwrap();
        let bias_name = CString::new("probe.conv.conv.bias").unwrap();
        let shape = [1u64, channels as u64, kernel as u64];
        let bias_shape = [1u64];
        let (weights, weight_offset) = resident(values, skew);
        let (bias, bias_offset) = resident(&[1.0], skew);
        let entries = [
            MimiWeight {
                name: name.as_ptr(),
                bytes: unsafe { weights.as_ptr().add(weight_offset) },
                shape: shape.as_ptr(),
                ndim: 3,
                len: values.len() as u64,
            },
            MimiWeight {
                name: bias_name.as_ptr(),
                bytes: unsafe { bias.as_ptr().add(bias_offset) },
                shape: bias_shape.as_ptr(),
                ndim: 1,
                len: 1,
            },
        ];
        let table = MimiWeightTable {
            entries: entries.as_ptr(),
            count: entries.len() as u32,
            bound: std::ptr::null_mut(),
        };
        let prefix = CString::new("probe").unwrap();
        let mut arena = Arena::new(size);
        let frames = cases
            .iter()
            .map(|(input, _)| input.len() / channels)
            .max()
            .unwrap_or(1)
            .max(1);
        let capacity = channels * kernel * frames;
        let canary = f32::from_bits(0x7fc0_51a7);
        let mut matrix = vec![canary; capacity + 2];
        let mut error = [0i8; 256];
        let mut state = std::ptr::null_mut();
        let status = unsafe {
            mimi_conv_init(
                &mut state,
                &table,
                prefix.as_ptr(),
                channels as i32,
                1,
                kernel as i32,
                stride as i32,
                1,
                1,
                1,
                matrix.as_mut_ptr().add(1),
                capacity,
                &mut arena.raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(status, 0, "{}", message(&error));

        for (index, (input, expected)) in cases.iter().enumerate() {
            if index == reset {
                unsafe { mimi_conv_reset(state) };
            }
            let frames = input.len() / channels;
            let mut output = vec![f32::NAN; input.len() * stride + kernel];
            let count = unsafe {
                mimi_conv_step(state, input.as_ptr(), frames as i32, output.as_mut_ptr())
            };
            assert_eq!(count, expected.len() as i32, "case {index}, skew {skew}");
            assert_eq!(
                output[..expected.len()]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "case {index}, skew {skew}"
            );
            assert_eq!(matrix[0].to_bits(), canary.to_bits());
            assert_eq!(matrix[capacity + 1].to_bits(), canary.to_bits());
        }
    }
}

#[test]
fn conv1d_context_matches_copy_commit_goldens_across_routes_priming_and_reset() {
    let cases = [
        (vec![1.0], vec![4.0]),
        (vec![2.0], vec![9.0]),
        (vec![-1.0], vec![3.0]),
        (vec![2.0], vec![7.0]),
        (vec![0.0], vec![5.0]),
    ];
    check_conv(1, 3, 1, &[1.0, 2.0, 3.0], &cases, 3, 32 * 1024);

    let mut weights = vec![0.0f32; 512 * 3];
    weights[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
    let wide = cases
        .iter()
        .map(|(input, expected)| {
            let mut frame = vec![0.0f32; 512];
            frame[0] = input[0];
            (frame, expected.clone())
        })
        .collect::<Vec<_>>();
    check_conv(512, 3, 1, &weights, &wide, 3, 64 * 1024);

    let priming = [
        (vec![1.0], vec![]),
        (vec![2.0], vec![9.0]),
        (vec![-1.0], vec![]),
        (vec![3.0], vec![10.0]),
        (vec![2.0], vec![]),
        (vec![0.0], vec![5.0]),
    ];
    check_conv(1, 3, 2, &[1.0, 2.0, 3.0], &priming, 4, 32 * 1024);
}

#[test]
fn convtranspose_carry_matches_the_copy_commit_golden_across_reset() {
    for skew in [0usize, 1] {
        let name = CString::new("probe.convtr.convtr.weight").unwrap();
        let bias_name = CString::new("probe.convtr.convtr.bias").unwrap();
        let shape = [1u64, 1, 4];
        let bias_shape = [1u64];
        let (weights, weight_offset) = resident(&[1.0, 2.0, 3.0, 4.0], skew);
        let (bias, bias_offset) = resident(&[1.0], skew);
        let entries = [
            MimiWeight {
                name: name.as_ptr(),
                bytes: unsafe { weights.as_ptr().add(weight_offset) },
                shape: shape.as_ptr(),
                ndim: 3,
                len: 4,
            },
            MimiWeight {
                name: bias_name.as_ptr(),
                bytes: unsafe { bias.as_ptr().add(bias_offset) },
                shape: bias_shape.as_ptr(),
                ndim: 1,
                len: 1,
            },
        ];
        let table = MimiWeightTable {
            entries: entries.as_ptr(),
            count: entries.len() as u32,
            bound: std::ptr::null_mut(),
        };
        let prefix = CString::new("probe").unwrap();
        let mut arena = Arena::new(32 * 1024);
        let mut matrix = [f32::from_bits(0x7fc0_51a7)];
        let mut error = [0i8; 256];
        let mut state = std::ptr::null_mut();
        let status = unsafe {
            mimi_convtr_init(
                &mut state,
                &table,
                prefix.as_ptr(),
                1,
                1,
                4,
                2,
                1,
                matrix.as_mut_ptr(),
                matrix.len(),
                &mut arena.raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(status, 0, "{}", message(&error));

        for (input, expected) in [
            (1.0f32, [2.0f32, 3.0]),
            (2.0, [6.0, 9.0]),
            (-1.0, [6.0, 7.0]),
        ] {
            let mut output = [f32::NAN; 2];
            assert_eq!(
                unsafe { mimi_convtr_step(state, &input, 1, output.as_mut_ptr()) },
                2
            );
            assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
        }

        unsafe { mimi_convtr_reset(state) };
        for (input, expected) in [(2.0f32, [3.0f32, 5.0]), (0.0, [7.0, 9.0])] {
            let mut output = [f32::NAN; 2];
            assert_eq!(
                unsafe { mimi_convtr_step(state, &input, 1, output.as_mut_ptr()) },
                2
            );
            assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
}

#[test]
fn l0_residual_matrix_uses_b2_suffix_without_touching_the_live_output_prefix() {
    const INPUTS: usize = 512;
    const OUTPUTS: usize = 256;
    const KERNEL: usize = 3;
    const BASE_FRAMES: usize = 4;
    const FRAMES: usize = BASE_FRAMES * 8;
    const B2: usize = 245_760;
    const LIVE: usize = OUTPUTS * FRAMES;
    const REQUIRED: usize = INPUTS * KERNEL * FRAMES;
    const CANARY: u32 = 0x7fc0_51a7;

    assert_eq!(LIVE, 8_192);
    assert_eq!(REQUIRED, 49_152);
    assert!(LIVE + REQUIRED < B2);

    let name = CString::new("probe.conv.conv.weight").unwrap();
    let shape = [OUTPUTS as u64, INPUTS as u64, KERNEL as u64];
    let mut values = vec![0.0f32; OUTPUTS * INPUTS * KERNEL];
    values[2] = 1.0;
    let (weights, offset) = resident(&values, 1);
    let entries = [MimiWeight {
        name: name.as_ptr(),
        bytes: unsafe { weights.as_ptr().add(offset) },
        shape: shape.as_ptr(),
        ndim: 3,
        len: values.len() as u64,
    }];
    let table = MimiWeightTable {
        entries: entries.as_ptr(),
        count: 1,
        bound: std::ptr::null_mut(),
    };
    let prefix = CString::new("probe").unwrap();
    let mut arena = Arena::new(64 * 1024);
    let mut b2 = vec![f32::from_bits(CANARY); B2 + 2];
    let mut error = [0i8; 256];
    let mut state = std::ptr::null_mut();
    let status = unsafe {
        mimi_conv_init(
            &mut state,
            &table,
            prefix.as_ptr(),
            INPUTS as i32,
            OUTPUTS as i32,
            KERNEL as i32,
            1,
            1,
            1,
            1,
            b2.as_mut_ptr().add(1 + LIVE),
            B2 - LIVE,
            &mut arena.raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    assert_eq!(status, 0, "{}", message(&error));
    assert!(arena.raw.used < 64 * 1024);

    let mut input = vec![0.0f32; INPUTS * FRAMES];
    for (frame, value) in input[..FRAMES].iter_mut().enumerate() {
        *value = frame as f32 + 1.0;
    }
    assert_eq!(
        unsafe { mimi_conv_step(state, input.as_ptr(), FRAMES as i32, b2.as_mut_ptr().add(1),) },
        FRAMES as i32
    );

    assert_eq!(b2[0].to_bits(), CANARY);
    assert_eq!(b2[B2 + 1].to_bits(), CANARY);
    for frame in 0..FRAMES {
        assert_eq!(b2[1 + frame].to_bits(), input[frame].to_bits());
    }
    for value in &b2[1 + FRAMES..1 + LIVE] {
        assert_eq!(value.to_bits(), 0.0f32.to_bits());
    }
    assert_ne!(b2[1 + LIVE].to_bits(), CANARY);
    assert_eq!(b2[1 + LIVE + REQUIRED].to_bits(), CANARY);
}

#[test]
fn deepest_convtranspose_exact_n2_workspace_and_n4_fallback_are_bounded() {
    const INPUTS: usize = 128;
    const OUTPUTS: usize = 64;
    const KERNEL: usize = 8;
    const STRIDE: usize = 4;
    const B2: usize = 245_760;
    const N2_FRAMES: usize = 2 * 8 * 6 * 5;
    const N4_FRAMES: usize = 4 * 8 * 6 * 5;
    const CANARY: u32 = 0x7fc0_51a7;

    assert_eq!(N2_FRAMES, 480);
    assert_eq!(KERNEL * OUTPUTS * N2_FRAMES, B2);
    assert_eq!(N4_FRAMES, 960);
    assert_eq!(KERNEL * OUTPUTS * N4_FRAMES, 2 * B2);

    let name = CString::new("probe.convtr.convtr.weight").unwrap();
    let shape = [INPUTS as u64, OUTPUTS as u64, KERNEL as u64];
    let mut values = vec![0.0f32; INPUTS * OUTPUTS * KERNEL];
    values[0] = 1.0;
    let (weights, offset) = resident(&values, 1);
    let entries = [MimiWeight {
        name: name.as_ptr(),
        bytes: unsafe { weights.as_ptr().add(offset) },
        shape: shape.as_ptr(),
        ndim: 3,
        len: values.len() as u64,
    }];
    let table = MimiWeightTable {
        entries: entries.as_ptr(),
        count: 1,
        bound: std::ptr::null_mut(),
    };
    let prefix = CString::new("probe").unwrap();

    let run = |frames: usize, expect_matrix: bool| {
        let mut arena = Arena::new(64 * 1024);
        let mut matrix = vec![f32::from_bits(CANARY); B2 + 2];
        let mut error = [0i8; 256];
        let mut state = std::ptr::null_mut();
        let status = unsafe {
            mimi_convtr_init(
                &mut state,
                &table,
                prefix.as_ptr(),
                INPUTS as i32,
                OUTPUTS as i32,
                KERNEL as i32,
                STRIDE as i32,
                1,
                matrix.as_mut_ptr().add(1),
                B2,
                &mut arena.raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(status, 0, "{}", message(&error));
        assert!(arena.raw.used < 64 * 1024);

        let mut input = vec![0.0f32; INPUTS * frames];
        for (frame, value) in input[..frames].iter_mut().enumerate() {
            *value = frame as f32 + 1.0;
        }
        let emit = frames * STRIDE;
        let mut output = vec![f32::NAN; OUTPUTS * emit];
        assert_eq!(
            unsafe { mimi_convtr_step(state, input.as_ptr(), frames as i32, output.as_mut_ptr(),) },
            emit as i32
        );

        assert_eq!(matrix[0].to_bits(), CANARY);
        assert_eq!(matrix[B2 + 1].to_bits(), CANARY);
        if expect_matrix {
            assert_ne!(matrix[1].to_bits(), CANARY);
            assert_ne!(matrix[B2].to_bits(), CANARY);
        } else {
            assert!(matrix[1..=B2].iter().all(|value| value.to_bits() == CANARY));
        }
        for frame in 0..frames {
            assert_eq!(output[frame * STRIDE].to_bits(), input[frame].to_bits());
            for tap in 1..STRIDE {
                assert_eq!(output[frame * STRIDE + tap].to_bits(), 0.0f32.to_bits());
            }
        }
        assert!(output[emit..]
            .iter()
            .all(|value| value.to_bits() == 0.0f32.to_bits()));
        arena.raw.used
    };

    let exact = run(N2_FRAMES, true);
    let fallback = run(N4_FRAMES, false);
    assert_eq!(exact, fallback);
}

#[test]
fn depthwise_upsample_carry_matches_the_copy_commit_golden_across_reset() {
    const DIM: usize = 512;
    for skew in [0usize, 1] {
        let name = CString::new("upsample.convtr.convtr.convtr.weight").unwrap();
        let shape = [DIM as u64, 1, 4];
        let values = (0..DIM)
            .flat_map(|_| [1.0f32, 2.0, 3.0, 4.0])
            .collect::<Vec<_>>();
        let (weights, offset) = resident(&values, skew);
        let entries = [MimiWeight {
            name: name.as_ptr(),
            bytes: unsafe { weights.as_ptr().add(offset) },
            shape: shape.as_ptr(),
            ndim: 3,
            len: values.len() as u64,
        }];
        let table = MimiWeightTable {
            entries: entries.as_ptr(),
            count: 1,
            bound: std::ptr::null_mut(),
        };
        let mut arena = Arena::new(64 * 1024);
        let mut error = [0i8; 256];
        let mut state = std::ptr::null_mut();
        let status = unsafe {
            mimi_upsample_init(
                &mut state,
                &table,
                &mut arena.raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(status, 0, "{}", message(&error));

        for (input, expected) in [(1.0f32, [1.0f32, 2.0]), (2.0, [5.0, 8.0])] {
            let input = [input; DIM];
            let mut output = [f32::NAN; 2 * DIM];
            assert_eq!(
                unsafe { mimi_upsample_step(state, input.as_ptr(), 1, output.as_mut_ptr()) },
                2
            );
            for pair in output.chunks_exact(2) {
                assert_eq!(pair[0].to_bits(), expected[0].to_bits());
                assert_eq!(pair[1].to_bits(), expected[1].to_bits());
            }
        }

        unsafe { mimi_upsample_reset(state) };
        let input = [2.0f32; DIM];
        let mut output = [f32::NAN; 2 * DIM];
        assert_eq!(
            unsafe { mimi_upsample_step(state, input.as_ptr(), 1, output.as_mut_ptr()) },
            2
        );
        for pair in output.chunks_exact(2) {
            assert_eq!(pair[0].to_bits(), 2.0f32.to_bits());
            assert_eq!(pair[1].to_bits(), 4.0f32.to_bits());
        }
    }
}
