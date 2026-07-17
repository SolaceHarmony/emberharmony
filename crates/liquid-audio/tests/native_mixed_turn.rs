#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct MixedTurnPlan {
    text_offset: usize,
    audio_offset: usize,
    assistant_offset: usize,
    total: usize,
}

#[repr(C)]
struct NativeEmission {
    kind: u32,
    text_bytes: u32,
    code_count: u32,
    flags: u32,
    position: u64,
    text: [u8; 512],
    codes: [u32; 64],
}

impl NativeEmission {
    fn audio(flags: u32) -> Self {
        let codes = if flags == 1 { [2048; 64] } else { [0; 64] };
        Self {
            kind: 2,
            text_bytes: 0,
            code_count: 8,
            flags,
            position: 0,
            text: [0; 512],
            codes,
        }
    }
}

unsafe extern "C" {
    fn lfm_mixed_turn_plan(
        capacity: usize,
        prefix_tokens: usize,
        text_tokens: usize,
        audio_rows: usize,
        assistant_tokens: usize,
        out: *mut MixedTurnPlan,
    ) -> i32;
    fn lfm_native_emission_needs_pcm(emission: *const NativeEmission) -> i32;
}

fn plan(
    capacity: usize,
    prefix: usize,
    text: usize,
    audio: usize,
    assistant: usize,
) -> Result<MixedTurnPlan, i32> {
    let mut plan = MixedTurnPlan::default();
    let status =
        unsafe { lfm_mixed_turn_plan(capacity, prefix, text, audio, assistant, &mut plan) };
    if status != 0 {
        assert_eq!(plan, MixedTurnPlan::default());
        return Err(status);
    }
    Ok(plan)
}

#[test]
fn eo_audio_stays_in_recurrence_and_never_enters_mimi() {
    use liquid_audio as _;
    assert_eq!(
        unsafe { lfm_native_emission_needs_pcm(&NativeEmission::audio(0)) },
        1
    );
    assert_eq!(
        unsafe { lfm_native_emission_needs_pcm(&NativeEmission::audio(1)) },
        0
    );
    assert_eq!(
        unsafe { lfm_native_emission_needs_pcm(&NativeEmission::audio(2)) },
        -libc::EINVAL
    );
}

#[test]
fn mixed_turn_plan_is_exact_and_orders_text_before_audio() {
    use liquid_audio as _;
    assert_eq!(
        plan(21, 2, 7, 9, 3),
        Ok(MixedTurnPlan {
            text_offset: 2,
            audio_offset: 9,
            assistant_offset: 18,
            total: 21,
        })
    );
    assert_eq!(plan(20, 2, 7, 9, 3), Err(-libc::ENOSPC));

    assert_eq!(
        plan(usize::MAX, usize::MAX - 3, 1, 1, 1),
        Ok(MixedTurnPlan {
            text_offset: usize::MAX - 3,
            audio_offset: usize::MAX - 2,
            assistant_offset: usize::MAX - 1,
            total: usize::MAX,
        })
    );
    assert_eq!(
        plan(usize::MAX, usize::MAX - 3, 2, 1, 1),
        Err(-libc::ENOSPC)
    );
}

#[test]
fn mixed_turn_plan_rejects_missing_modalities_before_reservation() {
    use liquid_audio as _;
    assert_eq!(plan(16, 0, 1, 1, 1), Err(-libc::EINVAL));
    assert_eq!(plan(16, 1, 0, 1, 1), Err(-libc::EINVAL));
    assert_eq!(plan(16, 1, 1, 0, 1), Err(-libc::EINVAL));
    assert_eq!(plan(16, 1, 1, 1, 0), Err(-libc::EINVAL));
    assert_eq!(plan(0, 1, 1, 1, 1), Err(-libc::EINVAL));
}
