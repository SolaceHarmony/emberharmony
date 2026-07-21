#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use liquid_audio as _;
use std::ffi::c_int;

type Leaf<T> = unsafe extern "C" fn(*const T, *mut f32, usize, u32, usize) -> c_int;
type Fanout<T> = unsafe extern "C" fn(*const f32, *mut T, usize, u32, usize) -> c_int;

unsafe extern "C" {
    fn lfm_capture_downmix_f32(
        source: *const f32,
        destination: *mut f32,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
    ) -> c_int;
    fn lfm_capture_downmix_i16(
        source: *const i16,
        destination: *mut f32,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
    ) -> c_int;
    fn lfm_capture_downmix_u16(
        source: *const u16,
        destination: *mut f32,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
    ) -> c_int;
    fn lfm_playback_fanout_f32(
        source: *const f32,
        destination: *mut f32,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
    ) -> c_int;
    fn lfm_playback_fanout_i16(
        source: *const f32,
        destination: *mut i16,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
    ) -> c_int;
    fn lfm_playback_fanout_u16(
        source: *const f32,
        destination: *mut u16,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
    ) -> c_int;
}

fn check<T: Copy + PartialEq + std::fmt::Debug>(
    source: Vec<T>,
    frames: usize,
    channels: u32,
    leaf: Leaf<T>,
    reference: impl Fn(&[T]) -> f32,
) {
    const GUARD: u32 = 0x7fc1_2345;
    let saved = source.clone();
    let mut output = vec![f32::from_bits(GUARD); frames + 4];
    assert_eq!(
        unsafe {
            leaf(
                source.as_ptr(),
                output.as_mut_ptr().add(2),
                frames,
                channels,
                frames,
            )
        },
        0
    );
    assert_eq!(source, saved, "capture leaf changed its borrowed source");
    assert_eq!(output[0].to_bits(), GUARD);
    assert_eq!(output[1].to_bits(), GUARD);
    assert_eq!(output[frames + 2].to_bits(), GUARD);
    assert_eq!(output[frames + 3].to_bits(), GUARD);

    for frame in 0..frames {
        let offset = frame * channels as usize;
        let expected = reference(&source[offset..offset + channels as usize]);
        assert_eq!(
            output[frame + 2].to_bits(),
            expected.to_bits(),
            "format result drift at frame {frame}, channels {channels}"
        );
    }
}

fn fanout_float(source: &[f32], channels: u32) {
    const GUARD: u32 = 0x55aa_1234;
    let samples = source.len() * channels as usize;
    let mut output = vec![f32::from_bits(GUARD); samples + 4];
    let saved = source.to_vec();
    let first = 5;
    assert_eq!(
        unsafe {
            lfm_playback_fanout_f32(
                source.as_ptr(),
                output.as_mut_ptr().add(2),
                first,
                channels,
                samples,
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            lfm_playback_fanout_f32(
                source.as_ptr().add(first),
                output.as_mut_ptr().add(2 + first * channels as usize),
                source.len() - first,
                channels,
                samples - first * channels as usize,
            )
        },
        0
    );
    assert_eq!(output[0].to_bits(), GUARD);
    assert_eq!(output[1].to_bits(), GUARD);
    assert_eq!(output[samples + 2].to_bits(), GUARD);
    assert_eq!(output[samples + 3].to_bits(), GUARD);
    assert!(
        source
            .iter()
            .zip(&saved)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "playback fan-out changed its borrowed source"
    );
    for (frame, expected) in source.iter().enumerate() {
        for channel in 0..channels as usize {
            assert_eq!(
                output[2 + frame * channels as usize + channel].to_bits(),
                expected.to_bits()
            );
        }
    }
    assert_eq!(
        unsafe { lfm_playback_fanout_f32(std::ptr::null(), std::ptr::null_mut(), 0, channels, 0,) },
        0
    );
}

fn fanout_integer<T: Copy + Eq + std::fmt::Debug>(
    source: &[f32],
    channels: u32,
    guard: T,
    fanout: Fanout<T>,
    convert: impl Fn(f32) -> T,
) {
    let samples = source.len() * channels as usize;
    let mut output = vec![guard; samples + 4];
    let saved = source.to_vec();
    let first = 5;
    assert_eq!(
        unsafe {
            fanout(
                source.as_ptr(),
                output.as_mut_ptr().add(2),
                first,
                channels,
                samples,
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            fanout(
                source.as_ptr().add(first),
                output.as_mut_ptr().add(2 + first * channels as usize),
                source.len() - first,
                channels,
                samples - first * channels as usize,
            )
        },
        0
    );
    assert_eq!(output[0], guard);
    assert_eq!(output[1], guard);
    assert_eq!(output[samples + 2], guard);
    assert_eq!(output[samples + 3], guard);
    assert!(
        source
            .iter()
            .zip(&saved)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "playback fan-out changed its borrowed source"
    );
    for (frame, sample) in source.iter().enumerate() {
        const CONTEXT: &str = "playback format/fan-out drift";
        let expected = convert(*sample);
        for channel in 0..channels as usize {
            assert_eq!(
                output[2 + frame * channels as usize + channel],
                expected,
                "{CONTEXT} at frame {frame}, channel {channel}"
            );
        }
    }
}

fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

fn to_u16(sample: f32) -> u16 {
    (to_i16(sample) as i32 + 32768) as u16
}

#[test]
fn capture_formats_match_the_device_contract_without_crossing_canaries() {
    let frames = [1usize, 3, 4, 7, 8, 9, 16, 17];
    let channels = [1u32, 2, 3, 4, 6];
    for frames in frames {
        for channels in channels {
            let count = frames * channels as usize;
            check(
                (0..count)
                    .map(|index| (index as i32 % 19 - 9) as f32 / 11.0)
                    .collect(),
                frames,
                channels,
                lfm_capture_downmix_f32,
                |frame| frame.iter().copied().sum::<f32>() / frame.len() as f32,
            );
            check(
                (0..count)
                    .map(|index| (index.wrapping_mul(7919).wrapping_add(123) & 0xffff) as i16)
                    .collect(),
                frames,
                channels,
                lfm_capture_downmix_i16,
                |frame| {
                    frame.iter().map(|sample| *sample as f32).sum::<f32>()
                        / (frame.len() as f32 * i16::MAX as f32)
                },
            );
            check(
                (0..count)
                    .map(|index| (index.wrapping_mul(6151).wrapping_add(327) & 0xffff) as u16)
                    .collect(),
                frames,
                channels,
                lfm_capture_downmix_u16,
                |frame| {
                    frame
                        .iter()
                        .map(|sample| (*sample as f32 - 32768.0) / 32768.0)
                        .sum::<f32>()
                        / frame.len() as f32
                },
            );
        }
    }
}

#[test]
fn capture_format_validation_never_dereferences_empty_views() {
    assert_eq!(
        unsafe { lfm_capture_downmix_f32(std::ptr::null(), std::ptr::null_mut(), 0, 1, 0) },
        0
    );
    assert_eq!(
        unsafe { lfm_capture_downmix_i16(std::ptr::null(), std::ptr::null_mut(), 0, 1, 0) },
        0
    );
    assert_eq!(
        unsafe { lfm_capture_downmix_u16(std::ptr::null(), std::ptr::null_mut(), 0, 1, 0) },
        0
    );
    assert_eq!(
        unsafe { lfm_capture_downmix_f32(std::ptr::null(), std::ptr::null_mut(), 1, 1, 1) },
        -libc::EINVAL
    );
    assert_eq!(
        unsafe { lfm_capture_downmix_f32(std::ptr::null(), std::ptr::null_mut(), 0, 0, 0) },
        -libc::EINVAL
    );

    let mut samples = [0.0f32; 2];
    assert_eq!(
        unsafe { lfm_capture_downmix_f32(samples.as_ptr(), samples.as_mut_ptr(), 2, 1, 1) },
        -libc::ENOSPC
    );
}

#[test]
fn playback_formats_fan_out_exact_values_across_lease_boundaries() {
    let source = [
        -2.0f32, -1.0, -0.75, -0.0, 0.0, 0.125, 0.5, 0.999, 1.0, 2.0, -0.3, 0.7, -1.5, 1.5, -0.9,
        0.25, -0.125,
    ];
    for channels in [1u32, 2, 3, 4, 6] {
        fanout_float(&source, channels);
        fanout_integer(
            &source,
            channels,
            0x5aa5u16 as i16,
            lfm_playback_fanout_i16,
            to_i16,
        );
        fanout_integer(
            &source,
            channels,
            0x5aa5u16,
            lfm_playback_fanout_u16,
            to_u16,
        );
    }
}

#[test]
fn playback_integer_conversion_handles_nan_infinity_and_clamps() {
    let source = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        1.0,
        0.0,
        -0.5,
        0.5,
        2.0,
        -2.0,
        0.25,
        -0.25,
        0.75,
        -0.75,
        1.25,
        -1.25,
        -0.0,
    ];
    for channels in [1u32, 2, 3, 4, 6] {
        fanout_integer(
            &source,
            channels,
            0x5aa5u16 as i16,
            lfm_playback_fanout_i16,
            to_i16,
        );
        fanout_integer(
            &source,
            channels,
            0x5aa5u16,
            lfm_playback_fanout_u16,
            to_u16,
        );
    }
}

#[test]
fn playback_format_validation_is_bounded_and_never_dereferences_empty_views() {
    assert_eq!(
        unsafe { lfm_playback_fanout_i16(std::ptr::null(), std::ptr::null_mut(), 0, 1, 0) },
        0
    );
    assert_eq!(
        unsafe { lfm_playback_fanout_u16(std::ptr::null(), std::ptr::null_mut(), 0, 1, 0) },
        0
    );
    assert_eq!(
        unsafe { lfm_playback_fanout_f32(std::ptr::null(), std::ptr::null_mut(), 1, 1, 1) },
        -libc::EINVAL
    );
    assert_eq!(
        unsafe { lfm_playback_fanout_f32(std::ptr::null(), std::ptr::null_mut(), 0, 0, 0) },
        -libc::EINVAL
    );
    let one = [0.0f32];
    let mut output = [f32::from_bits(0x55aa_1234)];
    assert_eq!(
        unsafe { lfm_playback_fanout_f32(one.as_ptr(), output.as_mut_ptr(), 1, 2, 1) },
        -libc::ENOSPC
    );
    assert_eq!(output[0].to_bits(), 0x55aa_1234);
    assert_eq!(
        unsafe {
            lfm_playback_fanout_f32(
                std::ptr::null(),
                std::ptr::null_mut(),
                usize::MAX,
                2,
                usize::MAX,
            )
        },
        -libc::EOVERFLOW
    );
}
