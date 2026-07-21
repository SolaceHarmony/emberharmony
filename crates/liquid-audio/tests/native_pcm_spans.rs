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

#[repr(C)]
struct ResamplerStream {
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
    fn lfm_resampler_stream_create(
        orig_freq: u32,
        new_freq: u32,
        max_sample_count: u64,
        out: *mut *mut ResamplerStream,
    ) -> c_int;
    fn lfm_resampler_stream_destroy(stream: *mut ResamplerStream) -> c_int;
    fn lfm_resampler_stream_reset(stream: *mut ResamplerStream);
    fn lfm_resampler_stream_out_length(
        stream: *mut ResamplerStream,
        sample_count: u64,
        out_length: *mut u64,
    ) -> c_int;
    fn lfm_resampler_stream_process(
        stream: *mut ResamplerStream,
        input: *const f32,
        sample_count: u64,
        destination: *mut f32,
        destination_capacity: u64,
        result: *mut Span,
    ) -> c_int;
    fn lfm_internal_playback_rate_contract_test(
        preprocessor_rate: u32,
        playback_rate: u32,
        out_preprocessor_rate: *mut u32,
        out_audio_output_rate: *mut u32,
        out_playback_frames: *mut u64,
        out_direct: *mut u32,
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

fn stream(input: &[f32], chunks: &[usize], rate: u32) -> Vec<f32> {
    assert_eq!(chunks.iter().sum::<usize>(), input.len());
    let capacity = chunks.iter().copied().max().unwrap();
    let mut stream = ptr::null_mut();
    assert_eq!(
        unsafe { lfm_resampler_stream_create(24_000, rate, capacity as u64, &mut stream) },
        0
    );
    let mut output = Vec::new();
    let mut offset = 0;
    for &chunk in chunks {
        let mut length = 0;
        assert_eq!(
            unsafe { lfm_resampler_stream_out_length(stream, chunk as u64, &mut length) },
            0
        );
        let base = output.len();
        output.resize(base + length as usize, 0.0);
        let mut result = Span::default();
        assert_eq!(
            unsafe {
                lfm_resampler_stream_process(
                    stream,
                    input.as_ptr().add(offset),
                    chunk as u64,
                    output.as_mut_ptr().add(base),
                    length,
                    &mut result,
                )
            },
            0
        );
        assert_eq!(result.data, unsafe { output.as_ptr().add(base) });
        assert_eq!(result.length, length);
        offset += chunk;
    }
    assert_eq!(unsafe { lfm_resampler_stream_destroy(stream) }, 0);
    output
}

fn irregular_chunks(length: usize) -> Vec<usize> {
    let pattern = [1, 17, 511, 1_920, 73, 997, 24, 1_279];
    let mut chunks = Vec::new();
    let mut offset = 0;
    for chunk in pattern.into_iter().cycle() {
        if offset == length {
            return chunks;
        }
        let chunk = chunk.min(length - offset);
        chunks.push(chunk);
        offset += chunk;
    }
    unreachable!()
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

#[test]
fn detokenizer_playback_rate_is_distinct_from_the_preprocessor_rate() {
    for (device, frames, direct) in [(16_000, 1_280, 0), (24_000, 1_920, 1), (48_000, 3_840, 0)] {
        let mut preprocessor = 0;
        let mut output_rate = 0;
        let mut actual = 0;
        let mut actual_direct = 0;
        assert_eq!(
            unsafe {
                lfm_internal_playback_rate_contract_test(
                    16_000,
                    device,
                    &mut preprocessor,
                    &mut output_rate,
                    &mut actual,
                    &mut actual_direct,
                )
            },
            0
        );
        assert_eq!(preprocessor, 16_000, "device={device}");
        assert_eq!(output_rate, 24_000, "device={device}");
        assert_eq!(actual, frames, "device={device}");
        assert_eq!(actual_direct, direct, "device={device}");
    }
}

#[test]
fn detokenizer_frames_keep_exact_duration_across_streaming_rate_boundaries() {
    const OUTPUT_RATE: u64 = 24_000;
    const FRAME: usize = 1_920;
    let samples = input(FRAME * 2);

    for (device, output_per_frame) in [(16_000u32, 1_280usize), (48_000, 3_840)] {
        let mut stream = ptr::null_mut();
        assert_eq!(
            unsafe {
                lfm_resampler_stream_create(OUTPUT_RATE as u32, device, FRAME as u64, &mut stream)
            },
            0
        );
        let mut chunked = vec![0.0f32; output_per_frame * 2];
        for frame in 0..2 {
            let mut length = 0;
            assert_eq!(
                unsafe { lfm_resampler_stream_out_length(stream, FRAME as u64, &mut length) },
                0
            );
            assert_eq!(length, output_per_frame as u64, "device={device}");
            let source = unsafe { samples.as_ptr().add(frame * FRAME) };
            let destination = unsafe { chunked.as_mut_ptr().add(frame * output_per_frame) };
            let mut result = Span::default();
            assert_eq!(
                unsafe {
                    lfm_resampler_stream_process(
                        stream,
                        source,
                        FRAME as u64,
                        destination,
                        output_per_frame as u64,
                        &mut result,
                    )
                },
                0
            );
            assert_eq!(result.data, destination, "device={device} frame={frame}");
            assert_eq!(
                result.length, output_per_frame as u64,
                "device={device} frame={frame}"
            );
        }
        assert_eq!(unsafe { lfm_resampler_stream_destroy(stream) }, 0);

        let mut whole_stream = ptr::null_mut();
        assert_eq!(
            unsafe {
                lfm_resampler_stream_create(
                    OUTPUT_RATE as u32,
                    device,
                    samples.len() as u64,
                    &mut whole_stream,
                )
            },
            0
        );
        let mut whole = vec![0.0f32; output_per_frame * 2];
        let mut result = Span::default();
        assert_eq!(
            unsafe {
                lfm_resampler_stream_process(
                    whole_stream,
                    samples.as_ptr(),
                    samples.len() as u64,
                    whole.as_mut_ptr(),
                    whole.len() as u64,
                    &mut result,
                )
            },
            0
        );
        assert_eq!(unsafe { lfm_resampler_stream_destroy(whole_stream) }, 0);
        assert_eq!(result.length, whole.len() as u64, "device={device}");
        assert_eq!(chunked, whole, "stream phase drift at device={device}");
        assert_eq!(
            whole.len() as u64 * OUTPUT_RATE,
            samples.len() as u64 * u64::from(device),
            "duration drift at device={device}"
        );
    }
}

#[test]
fn streaming_playback_matches_the_offline_sinc_after_its_causal_delay() {
    let input = input(4_097);
    let mut resampler = ptr::null_mut();
    let mut workspace = ptr::null_mut();
    assert_eq!(
        unsafe { lfm_resampler_create(24_000, 16_000, &mut resampler) },
        0
    );
    assert_eq!(unsafe { lfm_resampler_workspace_create(&mut workspace) }, 0);
    assert_eq!(
        unsafe { lfm_resampler_workspace_reserve(resampler, workspace, input.len() as u64) },
        0
    );
    let mut length = 0;
    assert_eq!(
        unsafe { lfm_resampler_out_length(resampler, input.len() as u64, &mut length) },
        0
    );
    let mut offline = vec![0.0f32; length as usize];
    let mut result = Span::default();
    assert_eq!(
        unsafe {
            lfm_resampler_process(
                resampler,
                workspace,
                input.as_ptr(),
                input.len() as u64,
                offline.as_mut_ptr(),
                offline.len() as u64,
                &mut result,
            )
        },
        0
    );
    assert_eq!(unsafe { lfm_resampler_workspace_destroy(workspace) }, 0);
    assert_eq!(unsafe { lfm_resampler_destroy(resampler) }, 0);

    let online = stream(&input, &irregular_chunks(input.len()), 16_000);
    assert_eq!(online.len(), offline.len());
    assert_eq!(
        &online[8..],
        &offline[..offline.len() - 8],
        "the causal leaf must reuse the offline sinc table and operation order"
    );
}

#[test]
fn streaming_playback_sinc_preserves_passband_phase_across_chunks() {
    const INPUT_RATE: f64 = 24_000.0;
    const OUTPUT_RATE: f64 = 16_000.0;
    const FREQUENCY: f64 = 1_000.0;
    const AMPLITUDE: f64 = 0.5;
    const DELAY: f64 = 12.0;
    const LENGTH: usize = 1_920 * 8;

    let input = (0..LENGTH)
        .map(|index| {
            (AMPLITUDE * (std::f64::consts::TAU * FREQUENCY * index as f64 / INPUT_RATE).sin())
                as f32
        })
        .collect::<Vec<_>>();
    let chunked = stream(&input, &irregular_chunks(input.len()), OUTPUT_RATE as u32);
    let whole = stream(&input, &[input.len()], OUTPUT_RATE as u32);
    assert_eq!(
        chunked, whole,
        "chunk boundaries changed a polyphase result bit"
    );

    let body = &chunked[256..];
    let (sin, cos) = body
        .iter()
        .enumerate()
        .fold((0.0, 0.0), |(sin, cos), (offset, &sample)| {
            let index = offset + 256;
            let angle = std::f64::consts::TAU
                * FREQUENCY
                * (index as f64 / OUTPUT_RATE - DELAY / INPUT_RATE);
            (
                sin + f64::from(sample) * angle.sin(),
                cos + f64::from(sample) * angle.cos(),
            )
        });
    let scale = 2.0 / body.len() as f64;
    let gain = (sin.hypot(cos) * scale) / AMPLITUDE;
    let phase = cos.atan2(sin);
    assert!((gain - 1.0).abs() < 0.002, "passband gain={gain}");
    assert!(phase.abs() < 0.002, "passband phase error={phase}");
}

#[test]
fn streaming_playback_sinc_rejects_above_device_nyquist() {
    const FREQUENCY: f64 = 10_000.0;
    const AMPLITUDE: f64 = 0.5;
    const LENGTH: usize = 1_920 * 8;

    let input = (0..LENGTH)
        .map(|index| {
            (AMPLITUDE * (std::f64::consts::TAU * FREQUENCY * index as f64 / 24_000.0).sin()) as f32
        })
        .collect::<Vec<_>>();
    let output = stream(&input, &irregular_chunks(input.len()), 16_000);
    let body = &output[256..];
    let rms = (body
        .iter()
        .map(|&sample| f64::from(sample).powi(2))
        .sum::<f64>()
        / body.len() as f64)
        .sqrt();
    let gain = rms / (AMPLITUDE / 2.0f64.sqrt());
    assert!(gain < 0.01, "10 kHz stopband gain={gain}");
}

#[test]
fn streaming_playback_causal_delay_and_terminal_tail_are_explicit() {
    const FRAME: usize = 1_920;
    let mut onset = vec![0.0f32; FRAME];
    onset[0] = 1.0;
    let output = stream(&onset, &[FRAME], 16_000);
    let peak = output
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
        .unwrap()
        .0;
    assert_eq!(peak, 8, "12 source samples must be 8 device samples");

    let mut stream = ptr::null_mut();
    assert_eq!(
        unsafe { lfm_resampler_stream_create(24_000, 16_000, FRAME as u64, &mut stream) },
        0
    );
    let mut terminal = vec![0.0f32; FRAME];
    terminal[FRAME - 1] = 1.0;
    let mut first = vec![0.0f32; 1_280];
    let mut result = Span::default();
    assert_eq!(
        unsafe {
            lfm_resampler_stream_process(
                stream,
                terminal.as_ptr(),
                FRAME as u64,
                first.as_mut_ptr(),
                first.len() as u64,
                &mut result,
            )
        },
        0
    );
    assert!(first.iter().all(|&sample| sample == 0.0));

    let zeros = [0.0f32; 48];
    let mut tail = [0.0f32; 32];
    assert_eq!(
        unsafe {
            lfm_resampler_stream_process(
                stream,
                zeros.as_ptr(),
                zeros.len() as u64,
                tail.as_mut_ptr(),
                tail.len() as u64,
                &mut result,
            )
        },
        0
    );
    let peak = tail
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
        .unwrap()
        .0;
    assert_eq!(peak, 7);
    assert!(tail.iter().any(|&sample| sample != 0.0));

    unsafe { lfm_resampler_stream_reset(stream) };
    tail.fill(f32::NAN);
    assert_eq!(
        unsafe {
            lfm_resampler_stream_process(
                stream,
                zeros.as_ptr(),
                zeros.len() as u64,
                tail.as_mut_ptr(),
                tail.len() as u64,
                &mut result,
            )
        },
        0
    );
    assert!(tail.iter().all(|&sample| sample == 0.0));
    assert_eq!(unsafe { lfm_resampler_stream_destroy(stream) }, 0);
}

#[test]
fn streaming_playback_polyphase_has_no_scalar_cpp_loop_or_per_output_division() {
    let source = include_str!("../native/src/frontend/lfm_frontend.cpp");
    let begin = source
        .find("extern \"C\" int lfm_resampler_stream_process(")
        .unwrap();
    let end = source[begin..]
        .find("extern \"C\" int lfm_resample_f32(")
        .map(|offset| begin + offset)
        .unwrap();
    let process = &source[begin..end];
    assert!(process.contains("lfm_resampler_stream_polyphase_f32(&kernel)"));
    assert!(!process.contains("for (") && !process.contains("while ("));
    let hot = &process[process.find("if (!stream->history").unwrap()..];
    assert!(!hot.contains("memcpy") && !hot.contains("memmove"));

    for leaf in [
        include_str!("../native/kernels/aarch64/flashkern_frontend.S"),
        include_str!("../native/kernels/x86_64/flashkern_frontend.S"),
    ] {
        let begin = leaf
            .find("LFM_SYM(lfm_resampler_stream_polyphase_f32):")
            .unwrap();
        let end = leaf[begin..]
            .find("// void lfm_resample_conv_spans_f32")
            .map(|offset| begin + offset)
            .unwrap();
        let stream = &leaf[begin..end];
        assert!(!stream.contains("udiv") && !stream.contains("divq"));
        assert!(stream.contains("#112") || stream.contains("112(%r15)"));
    }
}
