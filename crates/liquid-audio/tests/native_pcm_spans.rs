#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use liquid_audio as _;
use std::ffi::c_int;
use std::ptr;

const FRONTEND_ABI: u32 = 1;
const VALID_ONLY: u32 = 1;
const BF16_OUTPUT: u32 = 2;

#[repr(C)]
struct Frontend {
    _private: [u8; 0],
}

#[repr(C)]
struct FrontendWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
struct Resampler {
    _private: [u8; 0],
}

#[repr(C)]
struct ResamplerWorkspace {
    _private: [u8; 0],
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Span {
    data: *const f32,
    length: u64,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Chain {
    count: u32,
    reserved0: u32,
    length: u64,
    spans: [Span; 2],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct FrontendConfig {
    size: u32,
    abi_version: u32,
    sample_rate: u32,
    n_window_size: u32,
    n_window_stride: u32,
    n_fft: u32,
    nfilt: u32,
    exact_pad: u32,
    pad_to: u32,
    reserved0: u32,
    preemph: f64,
    log_zero_guard_value: f64,
    mag_power: f64,
    reserved: [u64; 4],
}

unsafe extern "C" {
    fn lfm_f32_span_chain_init(spans: *const Span, span_count: u32, out: *mut Chain) -> c_int;
    fn lfm_frontend_create(config: *const FrontendConfig, out: *mut *mut Frontend) -> c_int;
    fn lfm_frontend_destroy(frontend: *mut Frontend) -> c_int;
    fn lfm_frontend_workspace_create(out: *mut *mut FrontendWorkspace) -> c_int;
    fn lfm_frontend_workspace_destroy(workspace: *mut FrontendWorkspace) -> c_int;
    fn lfm_frontend_workspace_reserve(
        frontend: *const Frontend,
        workspace: *mut FrontendWorkspace,
        max_sample_count: u64,
        flags: u32,
    ) -> c_int;
    fn lfm_frontend_seq_len(frontend: *const Frontend, sample_count: u64) -> u64;
    fn lfm_frontend_forward_bf16_workspace(
        frontend: *const Frontend,
        workspace: *mut FrontendWorkspace,
        pcm: *const f32,
        sample_count: u64,
        out_mel: *mut u16,
        out_capacity_values: u64,
    ) -> c_int;
    fn lfm_frontend_forward_bf16_spans_workspace(
        frontend: *const Frontend,
        workspace: *mut FrontendWorkspace,
        pcm: *const Chain,
        out_mel: *mut u16,
        out_capacity_values: u64,
    ) -> c_int;
    fn lfm_resampler_create(orig_freq: u32, new_freq: u32, out: *mut *mut Resampler) -> c_int;
    fn lfm_resampler_destroy(resampler: *mut Resampler) -> c_int;
    fn lfm_resampler_out_length(
        resampler: *const Resampler,
        sample_count: u64,
        out_length: *mut u64,
    ) -> c_int;
    fn lfm_resampler_workspace_create(out: *mut *mut ResamplerWorkspace) -> c_int;
    fn lfm_resampler_workspace_destroy(workspace: *mut ResamplerWorkspace) -> c_int;
    fn lfm_resampler_workspace_reserve(
        resampler: *const Resampler,
        workspace: *mut ResamplerWorkspace,
        max_sample_count: u64,
    ) -> c_int;
    fn lfm_resampler_process(
        resampler: *const Resampler,
        workspace: *mut ResamplerWorkspace,
        input: *const f32,
        sample_count: u64,
        destination: *mut f32,
        destination_capacity: u64,
        result: *mut Span,
    ) -> c_int;
    fn lfm_resampler_process_spans(
        resampler: *const Resampler,
        workspace: *mut ResamplerWorkspace,
        input: *const Chain,
        destination: *mut f32,
        destination_capacity: u64,
        result: *mut Chain,
    ) -> c_int;
}

fn input(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let phase = index as f32 * 0.071_125;
            phase.sin() * 0.61 + (phase * 0.37).cos() * 0.19
        })
        .collect()
}

fn chain(input: &[f32], split: usize) -> Chain {
    assert!(split > 0 && split < input.len());
    let spans = [
        Span {
            data: input.as_ptr(),
            length: split as u64,
        },
        Span {
            data: unsafe { input.as_ptr().add(split) },
            length: (input.len() - split) as u64,
        },
    ];
    let mut chain = Chain::default();
    assert_eq!(
        unsafe { lfm_f32_span_chain_init(spans.as_ptr(), 2, &mut chain) },
        0
    );
    chain
}

#[test]
fn frontend_complete_output_is_invariant_across_pcm_span_boundaries() {
    let input = input(2_049);
    for preemph in [0.0, 0.97] {
        let config = FrontendConfig {
            size: size_of::<FrontendConfig>() as u32,
            abi_version: FRONTEND_ABI,
            sample_rate: 24_000,
            n_window_size: 32,
            n_window_stride: 8,
            n_fft: 32,
            nfilt: 8,
            exact_pad: 1,
            pad_to: 0,
            reserved0: 0,
            preemph,
            log_zero_guard_value: 2.0f64.powi(-24),
            mag_power: 2.0,
            reserved: [0; 4],
        };
        let mut frontend = ptr::null_mut();
        let mut workspace = ptr::null_mut();
        assert_eq!(unsafe { lfm_frontend_create(&config, &mut frontend) }, 0);
        assert_eq!(unsafe { lfm_frontend_workspace_create(&mut workspace) }, 0);
        assert_eq!(
            unsafe {
                lfm_frontend_workspace_reserve(
                    frontend,
                    workspace,
                    input.len() as u64,
                    VALID_ONLY | BF16_OUTPUT,
                )
            },
            0
        );
        let frames = unsafe { lfm_frontend_seq_len(frontend, input.len() as u64) };
        let values = frames as usize * config.nfilt as usize;
        let mut expected = vec![0u16; values];
        assert_eq!(
            unsafe {
                lfm_frontend_forward_bf16_workspace(
                    frontend,
                    workspace,
                    input.as_ptr(),
                    input.len() as u64,
                    expected.as_mut_ptr(),
                    expected.len() as u64,
                )
            },
            0
        );

        // First/last-sample boundaries cover the preemphasis carry. Values
        // around hop/window edges cover frame gather on both sides of a span.
        for split in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1_024, 2_048] {
            let input = chain(&input, split);
            let mut actual = vec![0u16; values];
            assert_eq!(
                unsafe {
                    lfm_frontend_forward_bf16_spans_workspace(
                        frontend,
                        workspace,
                        &input,
                        actual.as_mut_ptr(),
                        actual.len() as u64,
                    )
                },
                0,
                "preemph={preemph} split={split}"
            );
            assert_eq!(actual, expected, "preemph={preemph} split={split}");
        }
        assert_eq!(unsafe { lfm_frontend_workspace_destroy(workspace) }, 0);
        assert_eq!(unsafe { lfm_frontend_destroy(frontend) }, 0);
    }
}

#[test]
fn sinc_resampler_reads_two_pcm_spans_without_changing_any_output_bit() {
    let input = input(4_097);
    let mut resampler = ptr::null_mut();
    let mut workspace = ptr::null_mut();
    assert_eq!(
        unsafe { lfm_resampler_create(48_000, 24_000, &mut resampler) },
        0
    );
    assert_eq!(unsafe { lfm_resampler_workspace_create(&mut workspace) }, 0);
    assert_eq!(
        unsafe { lfm_resampler_workspace_reserve(resampler, workspace, input.len() as u64) },
        0
    );
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_resampler_out_length(resampler, input.len() as u64, &mut count) },
        0
    );
    let mut expected = vec![0.0f32; count as usize];
    let mut result = Span::default();
    assert_eq!(
        unsafe {
            lfm_resampler_process(
                resampler,
                workspace,
                input.as_ptr(),
                input.len() as u64,
                expected.as_mut_ptr(),
                expected.len() as u64,
                &mut result,
            )
        },
        0
    );
    assert_eq!(result.data, expected.as_ptr());
    assert_eq!(result.length, count);

    // The 48k→24k sinc plan has width 13. Straddle both sides of that window,
    // its full 28-tap kernel, the input midpoint, and the final sample.
    for split in [1, 11, 12, 13, 14, 27, 28, 29, 2_048, 4_096] {
        let input = chain(&input, split);
        let mut actual = vec![0.0f32; count as usize];
        let mut output = Chain::default();
        assert_eq!(
            unsafe {
                lfm_resampler_process_spans(
                    resampler,
                    workspace,
                    &input,
                    actual.as_mut_ptr(),
                    actual.len() as u64,
                    &mut output,
                )
            },
            0,
            "split={split}"
        );
        assert_eq!(output.count, 1);
        assert_eq!(output.length, count);
        assert_eq!(output.spans[0].data, actual.as_ptr());
        assert_eq!(actual, expected, "split={split}");
    }
    assert_eq!(unsafe { lfm_resampler_workspace_destroy(workspace) }, 0);
    assert_eq!(unsafe { lfm_resampler_destroy(resampler) }, 0);
}

#[test]
fn equal_rate_resampling_preserves_the_original_two_views() {
    let input = input(257);
    let input = chain(&input, 129);
    let mut resampler = ptr::null_mut();
    let mut workspace = ptr::null_mut();
    assert_eq!(
        unsafe { lfm_resampler_create(24_000, 24_000, &mut resampler) },
        0
    );
    assert_eq!(unsafe { lfm_resampler_workspace_create(&mut workspace) }, 0);
    let mut output = Chain::default();
    assert_eq!(
        unsafe {
            lfm_resampler_process_spans(
                resampler,
                workspace,
                &input,
                ptr::null_mut(),
                0,
                &mut output,
            )
        },
        0
    );
    assert_eq!(output.count, 2);
    assert_eq!(output.length, input.length);
    for index in 0..2 {
        assert_eq!(output.spans[index].data, input.spans[index].data);
        assert_eq!(output.spans[index].length, input.spans[index].length);
    }
    assert_eq!(unsafe { lfm_resampler_workspace_destroy(workspace) }, 0);
    assert_eq!(unsafe { lfm_resampler_destroy(resampler) }, 0);
}
