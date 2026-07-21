#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use liquid_audio as _;
use serde::Deserialize;
use std::ffi::c_int;
use std::ptr;

const ABI: u32 = 1;
const MIC: u32 = 1;
const PLAYBACK: u32 = 2;

#[repr(C)]
struct NativeDetector {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Default, PartialEq)]
struct Decision {
    size: u32,
    abi_version: u32,
    sample_rate: u32,
    stream: u32,
    first_bin: u32,
    end_bin: u32,
    selected_bins: u32,
    threshold: u32,
    voice: u32,
    reserved0: u32,
    score: f64,
    adaptive_min: u32,
    adaptive_max: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct Window {
    first: *const f32,
    first_count: usize,
    second: *const f32,
    second_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Span {
    samples: *const f32,
    count: usize,
}

#[repr(C)]
struct Scatter {
    spans: *const Span,
    span_count: usize,
}

unsafe extern "C" {
    fn lfm_sesame_detector_create(rate: u32, out: *mut *mut NativeDetector) -> c_int;
    fn lfm_sesame_detector_destroy(detector: *mut NativeDetector) -> c_int;
    fn lfm_sesame_detector_reset(detector: *mut NativeDetector, stream: u32) -> c_int;
    fn lfm_sesame_detector_discontinuity(detector: *mut NativeDetector, stream: u32) -> c_int;
    fn lfm_sesame_detector_first_bin(detector: *const NativeDetector) -> u32;
    fn lfm_sesame_detector_end_bin(detector: *const NativeDetector) -> u32;
    fn lfm_sesame_detector_derived_bytes(detector: *const NativeDetector) -> u64;
    fn lfm_sesame_detector_process(
        detector: *mut NativeDetector,
        stream: u32,
        latest_256: *const f32,
        selected_bytes: *mut u8,
        selected_capacity: usize,
        decision: *mut Decision,
    ) -> c_int;
    fn lfm_sesame_detector_process_window(
        detector: *mut NativeDetector,
        stream: u32,
        window: *const Window,
        selected_bytes: *mut u8,
        selected_capacity: usize,
        decision: *mut Decision,
    ) -> c_int;
    fn lfm_sesame_detector_process_scatter_window(
        detector: *mut NativeDetector,
        stream: u32,
        window: *const Scatter,
        selected_bytes: *mut u8,
        selected_capacity: usize,
        decision: *mut Decision,
    ) -> c_int;
    fn lfm_sesame_detector_classify_bytes(
        detector: *mut NativeDetector,
        stream: u32,
        bytes: *const u8,
        count: usize,
        decision: *mut Decision,
    ) -> c_int;
}

struct Detector(*mut NativeDetector);

impl Detector {
    fn new(rate: u32) -> Self {
        let mut raw = ptr::null_mut();
        assert_eq!(unsafe { lfm_sesame_detector_create(rate, &mut raw) }, 0);
        assert!(!raw.is_null());
        Self(raw)
    }

    fn classify(&mut self, stream: u32, bytes: &[u8]) -> Decision {
        let mut decision = Decision::default();
        assert_eq!(
            unsafe {
                lfm_sesame_detector_classify_bytes(
                    self.0,
                    stream,
                    bytes.as_ptr(),
                    bytes.len(),
                    &mut decision,
                )
            },
            0
        );
        decision
    }
}

impl Drop for Detector {
    fn drop(&mut self) {
        assert_eq!(unsafe { lfm_sesame_detector_destroy(self.0) }, 0);
    }
}

fn bins(detector: &Detector) -> usize {
    (unsafe { lfm_sesame_detector_end_bin(detector.0) }
        - unsafe { lfm_sesame_detector_first_bin(detector.0) }) as usize
}

fn reset(detector: &Detector, stream: u32) {
    assert_eq!(unsafe { lfm_sesame_detector_reset(detector.0, stream) }, 0);
}

fn process(detector: &Detector, stream: u32, input: &[f32; 256]) -> (Vec<u8>, Decision) {
    let mut bytes = vec![0; bins(detector)];
    let mut decision = Decision::default();
    assert_eq!(
        unsafe {
            lfm_sesame_detector_process(
                detector.0,
                stream,
                input.as_ptr(),
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut decision,
            )
        },
        0
    );
    (bytes, decision)
}

fn process_scatter(detector: &Detector, stream: u32, spans: &[Span]) -> (Vec<u8>, Decision) {
    let window = Scatter {
        spans: spans.as_ptr(),
        span_count: spans.len(),
    };
    let mut bytes = vec![0; bins(detector)];
    let mut decision = Decision::default();
    assert_eq!(
        unsafe {
            lfm_sesame_detector_process_scatter_window(
                detector.0,
                stream,
                &window,
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut decision,
            )
        },
        0
    );
    (bytes, decision)
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    sequence: Sequence,
}

#[derive(Deserialize)]
struct Case {
    rate: u32,
    kind: String,
    lo: u32,
    hi: u32,
    bytes: Vec<u8>,
}

#[derive(Deserialize)]
struct Sequence {
    rate: u32,
    lo: u32,
    hi: u32,
    segment_kinds: Vec<String>,
    snapshots: Vec<Snapshot>,
}

#[derive(Deserialize)]
struct Snapshot {
    frame: usize,
    bytes: Vec<u8>,
}

fn samples(kind: &str, rate: u32) -> [f32; 256] {
    std::array::from_fn(|sample| match kind {
        "zero" => 0.0,
        "constant" => 0.125,
        "impulse" => {
            if sample == 0 {
                0.5
            } else {
                0.0
            }
        }
        "integer" => (((sample * 17) % 31) as i32 - 15) as f32 / 64.0,
        "selected-square" | "high-square" => {
            let hz = if kind == "selected-square" {
                1_000
            } else {
                4_000
            };
            let half = (rate as usize / hz / 2).max(1);
            if (sample / half) % 2 == 0 {
                0.25
            } else {
                -0.25
            }
        }
        kind => panic!("unknown browser fixture generator kind: {kind}"),
    })
}

#[test]
fn selected_bins_match_real_chrome_web_audio_bytes_exactly_for_both_streams() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../native/tests/fixtures/sesame/selected_bins_v1.json"
    ))
    .unwrap();

    for case in &fixture.cases {
        for (stream, threshold) in [(MIC, 50), (PLAYBACK, 10)] {
            let detector = Detector::new(case.rate);
            let input = samples(&case.kind, case.rate);
            let mut bytes = vec![0u8; case.bytes.len()];
            let mut decision = Decision::default();
            assert_eq!(
                unsafe {
                    lfm_sesame_detector_process(
                        detector.0,
                        stream,
                        input.as_ptr(),
                        bytes.as_mut_ptr(),
                        bytes.len(),
                        &mut decision,
                    )
                },
                0,
                "{} Hz {} stream {}",
                case.rate,
                case.kind,
                stream
            );
            assert_eq!(
                bytes, case.bytes,
                "Chrome byte mismatch at {} Hz for {} stream {}",
                case.rate, case.kind, stream
            );
            assert_eq!(
                unsafe { lfm_sesame_detector_first_bin(detector.0) },
                case.lo
            );
            assert_eq!(unsafe { lfm_sesame_detector_end_bin(detector.0) }, case.hi);
            assert_eq!(decision.size as usize, std::mem::size_of::<Decision>());
            assert_eq!(decision.abi_version, ABI);
            assert_eq!(decision.sample_rate, case.rate);
            assert_eq!(decision.stream, stream);
            assert_eq!(decision.first_bin, case.lo);
            assert_eq!(decision.end_bin, case.hi);
            assert_eq!(decision.selected_bins as usize, case.bytes.len());
            assert_eq!(decision.threshold, threshold);
            assert_eq!(
                unsafe { lfm_sesame_detector_derived_bytes(detector.0) },
                case.bytes.len() as u64 * 256 * 2 * 4 + 256 * 8
            );
        }
    }
}

#[test]
fn interleaved_microphone_and_playback_updates_keep_independent_history() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../native/tests/fixtures/sesame/selected_bins_v1.json"
    ))
    .unwrap();
    let sequence = fixture.sequence;
    let shared = Detector::new(sequence.rate);
    let mic = Detector::new(sequence.rate);
    let playback = Detector::new(sequence.rate);

    for (index, mic_kind) in sequence.segment_kinds.iter().enumerate() {
        let playback_kind = &sequence.segment_kinds[sequence.segment_kinds.len() - index - 1];
        for (stream, kind, isolated) in [
            (MIC, mic_kind.as_str(), &mic),
            (PLAYBACK, playback_kind.as_str(), &playback),
        ] {
            let input = samples(kind, sequence.rate);
            let spans = [
                Span {
                    samples: input[..37].as_ptr(),
                    count: 37,
                },
                Span {
                    samples: input[37..194].as_ptr(),
                    count: 157,
                },
                Span {
                    samples: input[194..].as_ptr(),
                    count: 62,
                },
            ];
            let (shared_bytes, shared_decision) = process_scatter(&shared, stream, &spans);
            let (isolated_bytes, isolated_decision) = process(isolated, stream, &input);
            assert_eq!(
                shared_bytes, isolated_bytes,
                "stream {stream} update {index}"
            );
            assert_eq!(
                shared_decision, isolated_decision,
                "stream {stream} classifier state diverged at update {index}"
            );
        }
    }
}

#[test]
fn smoothing_history_matches_real_chrome_snapshots_exactly() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../native/tests/fixtures/sesame/selected_bins_v1.json"
    ))
    .unwrap();
    let sequence = fixture.sequence;
    let detector = Detector::new(sequence.rate);
    assert_eq!(
        unsafe { lfm_sesame_detector_first_bin(detector.0) },
        sequence.lo
    );
    assert_eq!(
        unsafe { lfm_sesame_detector_end_bin(detector.0) },
        sequence.hi
    );

    for snapshot in &sequence.snapshots {
        let segment = snapshot.frame / 256 - 1;
        let input = samples(&sequence.segment_kinds[segment], sequence.rate);
        let mut bytes = vec![0u8; snapshot.bytes.len()];
        let mut decision = Decision::default();
        assert_eq!(
            unsafe {
                lfm_sesame_detector_process(
                    detector.0,
                    MIC,
                    input.as_ptr(),
                    bytes.as_mut_ptr(),
                    bytes.len(),
                    &mut decision,
                )
            },
            0
        );
        assert_eq!(
            bytes, snapshot.bytes,
            "Chrome smoothing-history mismatch at frame {} ({})",
            snapshot.frame, sequence.segment_kinds[segment]
        );
    }
}

#[test]
fn circular_window_splits_are_bit_exact_at_vector_and_phase_boundaries() {
    const SPLITS: &[usize] = &[
        1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 191, 192, 193, 252, 253, 254,
        255,
    ];

    for rate in [16_000, 44_100, 48_000] {
        let input = samples("integer", rate);
        for &split in SPLITS {
            let contiguous = Detector::new(rate);
            let split_detector = Detector::new(rate);
            let bins = (unsafe { lfm_sesame_detector_end_bin(contiguous.0) }
                - unsafe { lfm_sesame_detector_first_bin(contiguous.0) })
                as usize;
            let mut contiguous_bytes = vec![0u8; bins];
            let mut split_bytes = vec![0u8; bins];
            let mut contiguous_decision = Decision::default();
            let mut split_decision = Decision::default();
            assert_eq!(
                unsafe {
                    lfm_sesame_detector_process(
                        contiguous.0,
                        MIC,
                        input.as_ptr(),
                        contiguous_bytes.as_mut_ptr(),
                        contiguous_bytes.len(),
                        &mut contiguous_decision,
                    )
                },
                0
            );

            /* Separate allocations prove that the two-view leaf cannot rely
             * on virtual adjacency across the circular wrap. */
            let first = input[..split].to_vec();
            let second = input[split..].to_vec();
            let window = Window {
                first: first.as_ptr(),
                first_count: first.len(),
                second: second.as_ptr(),
                second_count: second.len(),
            };
            assert_eq!(
                unsafe {
                    lfm_sesame_detector_process_window(
                        split_detector.0,
                        MIC,
                        &window,
                        split_bytes.as_mut_ptr(),
                        split_bytes.len(),
                        &mut split_decision,
                    )
                },
                0,
                "{rate} Hz split {split}"
            );
            assert_eq!(
                split_bytes, contiguous_bytes,
                "selected evidence differs at {rate} Hz split {split}"
            );
            assert_eq!(
                split_decision, contiguous_decision,
                "classifier state differs at {rate} Hz split {split}"
            );
        }
    }
}

#[test]
fn scatter_fixture_cases_match_chrome_for_every_rate_and_stream() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../native/tests/fixtures/sesame/selected_bins_v1.json"
    ))
    .unwrap();

    for case in &fixture.cases {
        let input = samples(&case.kind, case.rate);
        let spans: [Span; 256] = std::array::from_fn(|index| Span {
            samples: unsafe { input.as_ptr().add(index) },
            count: 1,
        });
        for stream in [MIC, PLAYBACK] {
            let detector = Detector::new(case.rate);
            let (bytes, decision) = process_scatter(&detector, stream, &spans);
            assert_eq!(
                bytes, case.bytes,
                "scatter Chrome byte mismatch at {} Hz for {} stream {}",
                case.rate, case.kind, stream
            );
            assert_eq!(decision.stream, stream);
            assert_eq!(decision.first_bin, case.lo);
            assert_eq!(decision.end_bin, case.hi);
        }
    }
}

#[test]
fn every_scatter_boundary_is_bit_exact_for_every_rate_and_stream() {
    for rate in [16_000, 24_000, 44_100, 48_000] {
        let input = samples("integer", rate);
        for stream in [MIC, PLAYBACK] {
            let contiguous = Detector::new(rate);
            let scatter = Detector::new(rate);
            for boundary in 1..256 {
                reset(&contiguous, stream);
                reset(&scatter, stream);
                let spans = if boundary < 255 {
                    [
                        Span {
                            samples: input[..boundary].as_ptr(),
                            count: boundary,
                        },
                        Span {
                            samples: input[boundary..boundary + 1].as_ptr(),
                            count: 1,
                        },
                        Span {
                            samples: input[boundary + 1..].as_ptr(),
                            count: 255 - boundary,
                        },
                    ]
                } else {
                    [
                        Span {
                            samples: input[..1].as_ptr(),
                            count: 1,
                        },
                        Span {
                            samples: input[1..255].as_ptr(),
                            count: 254,
                        },
                        Span {
                            samples: input[255..].as_ptr(),
                            count: 1,
                        },
                    ]
                };
                let expected = process(&contiguous, stream, &input);
                let actual = process_scatter(&scatter, stream, &spans);
                assert_eq!(
                    actual, expected,
                    "{rate} Hz stream {stream} boundary {boundary}"
                );
            }
        }
    }
}

#[test]
fn disjoint_real_zero_real_spans_are_bit_exact() {
    for rate in [16_000, 24_000, 44_100, 48_000] {
        let first = samples("selected-square", rate)[..73].to_vec();
        let zero = vec![0.0f32; 91];
        let last = samples("integer", rate)[..92].to_vec();
        let mut input = [0.0f32; 256];
        input[..73].copy_from_slice(&first);
        input[73..164].copy_from_slice(&zero);
        input[164..].copy_from_slice(&last);
        let spans = [
            Span {
                samples: first.as_ptr(),
                count: first.len(),
            },
            Span {
                samples: zero.as_ptr(),
                count: zero.len(),
            },
            Span {
                samples: last.as_ptr(),
                count: last.len(),
            },
        ];
        for stream in [MIC, PLAYBACK] {
            let contiguous = Detector::new(rate);
            let scatter = Detector::new(rate);
            assert_eq!(
                process_scatter(&scatter, stream, &spans),
                process(&contiguous, stream, &input),
                "{rate} Hz stream {stream}"
            );
        }
    }
}

#[test]
fn alternating_one_sample_spans_are_bit_exact() {
    for rate in [16_000, 24_000, 44_100, 48_000] {
        let real = samples("selected-square", rate);
        let zero = [0.0f32; 128];
        let input: [f32; 256] =
            std::array::from_fn(|index| if index % 2 == 0 { real[index] } else { 0.0 });
        let spans: [Span; 256] = std::array::from_fn(|index| Span {
            samples: if index % 2 == 0 {
                unsafe { real.as_ptr().add(index) }
            } else {
                unsafe { zero.as_ptr().add(index / 2) }
            },
            count: 1,
        });
        for stream in [MIC, PLAYBACK] {
            let contiguous = Detector::new(rate);
            let scatter = Detector::new(rate);
            assert_eq!(
                process_scatter(&scatter, stream, &spans),
                process(&contiguous, stream, &input),
                "{rate} Hz stream {stream}"
            );
        }
    }
}

#[test]
fn adaptive_extrema_are_sticky_separate_and_equality_is_voice() {
    let mut detector = Detector::new(48_000);

    let equality = detector.classify(MIC, &[10, 20]);
    assert_eq!(equality.adaptive_min, 10);
    assert_eq!(equality.adaptive_max, 20);
    assert_eq!(equality.score, 50.0);
    assert_eq!(equality.voice, 1, "score == threshold is voice");

    let sticky = detector.classify(MIC, &[15, 15]);
    assert_eq!((sticky.adaptive_min, sticky.adaptive_max), (10, 20));
    assert_eq!(sticky.score, 50.0);
    assert_eq!(sticky.voice, 1);

    assert_eq!(
        unsafe { lfm_sesame_detector_discontinuity(detector.0, MIC) },
        0
    );
    let discontinuous = detector.classify(MIC, &[15]);
    assert_eq!(
        (discontinuous.adaptive_min, discontinuous.adaptive_max),
        (10, 20),
        "a device gap resets smoothing but not the session adaptive range"
    );

    let playback = detector.classify(PLAYBACK, &[0, 10, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!((playback.adaptive_min, playback.adaptive_max), (0, 10));
    assert_eq!(playback.score, 10.0);
    assert_eq!(playback.threshold, 10);
    assert_eq!(playback.voice, 1, "playback equality is also voice");

    let silent = detector.classify(PLAYBACK, &[0; 10]);
    assert_eq!((silent.adaptive_min, silent.adaptive_max), (0, 10));
    assert_eq!(silent.score, 0.0);
    assert_eq!(silent.voice, 0);

    assert_eq!(
        unsafe { lfm_sesame_detector_reset(detector.0, PLAYBACK) },
        0
    );
    let reset = detector.classify(PLAYBACK, &[7, 7, 7]);
    assert_eq!((reset.adaptive_min, reset.adaptive_max), (7, 7));
    assert_eq!(reset.score, 0.0);
    assert_eq!(reset.voice, 0);

    let mic_unchanged = detector.classify(MIC, &[15]);
    assert_eq!(
        (mic_unchanged.adaptive_min, mic_unchanged.adaptive_max),
        (10, 20)
    );
}

#[test]
fn nonfinite_windows_publish_zero_evidence_without_poisoning_state() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let detector = Detector::new(48_000);
        let input = [value; 256];
        let mut bytes = [0xffu8; 9];
        let mut decision = Decision::default();
        assert_eq!(
            unsafe {
                lfm_sesame_detector_process(
                    detector.0,
                    MIC,
                    input.as_ptr(),
                    bytes.as_mut_ptr(),
                    bytes.len(),
                    &mut decision,
                )
            },
            0
        );
        assert_eq!(bytes, [0; 9]);
        assert_eq!(decision.score, 0.0);
        assert_eq!(decision.voice, 0);
    }
}

#[test]
fn malformed_geometry_and_calls_are_rejected() {
    let mut raw = ptr::null_mut();
    assert_ne!(unsafe { lfm_sesame_detector_create(0, &mut raw) }, 0);
    assert_ne!(unsafe { lfm_sesame_detector_create(4_000, &mut raw) }, 0);

    let detector = Detector::new(48_000);
    let input = [0.0f32; 256];
    let mut decision = Decision::default();
    assert_ne!(
        unsafe {
            lfm_sesame_detector_process(
                detector.0,
                MIC,
                input.as_ptr(),
                ptr::null_mut(),
                1,
                &mut decision,
            )
        },
        0
    );
    assert_ne!(unsafe { lfm_sesame_detector_reset(detector.0, 99) }, 0);

    let valid = [Span {
        samples: input.as_ptr(),
        count: input.len(),
    }];
    let valid_window = Scatter {
        spans: valid.as_ptr(),
        span_count: valid.len(),
    };
    let mut bytes = vec![0u8; bins(&detector)];
    assert_ne!(
        unsafe {
            lfm_sesame_detector_process_scatter_window(
                detector.0,
                MIC,
                ptr::null(),
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut decision,
            )
        },
        0
    );
    for window in [
        Scatter {
            spans: ptr::null(),
            span_count: 1,
        },
        Scatter {
            spans: valid.as_ptr(),
            span_count: 0,
        },
        Scatter {
            spans: valid.as_ptr(),
            span_count: 257,
        },
    ] {
        assert_ne!(
            unsafe {
                lfm_sesame_detector_process_scatter_window(
                    detector.0,
                    MIC,
                    &window,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                    &mut decision,
                )
            },
            0
        );
    }

    let null_sample = [Span {
        samples: ptr::null(),
        count: 256,
    }];
    let zero_count = [Span {
        samples: input.as_ptr(),
        count: 0,
    }];
    let oversized = [Span {
        samples: input.as_ptr(),
        count: 257,
    }];
    let overflowing = [Span {
        samples: input.as_ptr(),
        count: usize::MAX,
    }];
    let short = [Span {
        samples: input.as_ptr(),
        count: 255,
    }];
    let long = [
        Span {
            samples: input.as_ptr(),
            count: 128,
        },
        Span {
            samples: unsafe { input.as_ptr().add(128) },
            count: 129,
        },
    ];
    for spans in [
        null_sample.as_slice(),
        zero_count.as_slice(),
        oversized.as_slice(),
        overflowing.as_slice(),
        short.as_slice(),
        long.as_slice(),
    ] {
        let window = Scatter {
            spans: spans.as_ptr(),
            span_count: spans.len(),
        };
        assert_ne!(
            unsafe {
                lfm_sesame_detector_process_scatter_window(
                    detector.0,
                    MIC,
                    &window,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                    &mut decision,
                )
            },
            0
        );
    }

    assert_ne!(
        unsafe {
            lfm_sesame_detector_process_scatter_window(
                detector.0,
                99,
                &valid_window,
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut decision,
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lfm_sesame_detector_process_scatter_window(
                detector.0,
                MIC,
                &valid_window,
                ptr::null_mut(),
                1,
                &mut decision,
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lfm_sesame_detector_process_scatter_window(
                detector.0,
                MIC,
                &valid_window,
                bytes.as_mut_ptr(),
                bytes.len().saturating_sub(1),
                &mut decision,
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lfm_sesame_detector_process_scatter_window(
                detector.0,
                MIC,
                &valid_window,
                bytes.as_mut_ptr(),
                bytes.len(),
                ptr::null_mut(),
            )
        },
        0
    );
}
