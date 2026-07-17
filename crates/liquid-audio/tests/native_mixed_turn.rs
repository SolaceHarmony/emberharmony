#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct MixedTurnPlan {
    text_offset: usize,
    audio_offset: usize,
    assistant_offset: usize,
    total: usize,
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
