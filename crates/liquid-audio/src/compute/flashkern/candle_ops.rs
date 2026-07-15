//! Candle ownership rim for typed Flashkern passes. No numerical loop lives here.

use candle_core::{CpuStorage, Result, Tensor};

/// Pointer-only Candle ownership rim for Flashkern's CPU streaming depthwise
/// convolution. The native pass reads `x`, optional prior state, and
/// weights in place, then writes independent output/state planes. There is no
/// `[cache | x]` Tensor construction and no Rust numerical loop.
pub fn depthwise_conv1d_stream(
    x: &Tensor,
    weights: &Tensor,
    cache: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    use candle_core::{DType, Storage};

    if !x.device().is_cpu() || x.dtype() != DType::BF16 {
        candle_core::bail!("Flashkern depthwise stream requires CPU bf16");
    }
    let (batch, channels, steps) = x.dims3()?;
    let (weight_channels, kernel) = weights.dims2()?;
    if channels != weight_channels || kernel == 0 {
        candle_core::bail!(
            "flashkern depthwise stream: input {:?}, weights {:?}",
            x.shape(),
            weights.shape()
        );
    }
    let prior = kernel - 1;
    if let Some(state) = cache {
        if state.dims3()? != (batch, channels, prior)
            || state.dtype() != DType::BF16
            || !state.device().is_cpu()
        {
            candle_core::bail!(
                "flashkern depthwise stream: cache {:?} does not match ({batch},{channels},{prior}) bf16 CPU",
                state.shape()
            );
        }
    }

    fn bits<'a>(
        storage: &'a std::sync::RwLockReadGuard<'_, Storage>,
        layout: &candle_core::Layout,
    ) -> Result<&'a [u16]> {
        let Storage::Cpu(CpuStorage::BF16(values)) = &**storage else {
            candle_core::bail!("flashkern depthwise stream requires CPU bf16 storage");
        };
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("flashkern depthwise stream requires contiguous inputs".into())
        })?;
        // SAFETY: half::bf16 is transparent over its u16 representation.
        Ok(unsafe {
            std::slice::from_raw_parts(values[start..end].as_ptr().cast::<u16>(), end - start)
        })
    }

    let x = x.contiguous()?;
    let weights = weights.contiguous()?;
    let cache = cache.map(Tensor::contiguous).transpose()?;
    let (x_storage, x_layout) = x.storage_and_layout();
    let (weight_storage, weight_layout) = weights.storage_and_layout();
    let cache_storage = cache.as_ref().map(Tensor::storage_and_layout);
    let x_bits = bits(&x_storage, x_layout)?;
    let weight_bits = bits(&weight_storage, weight_layout)?;
    let cache_bits = cache_storage
        .as_ref()
        .map(|(storage, layout)| bits(storage, layout))
        .transpose()?;

    let mut output = vec![half::bf16::from_bits(0); batch * channels * steps];
    let mut next = vec![half::bf16::from_bits(0); batch * channels * prior];
    // SAFETY: both output vectors are uniquely borrowed and bf16 is transparent over u16.
    let output_bits =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u16>(), output.len()) };
    let next_bits =
        unsafe { std::slice::from_raw_parts_mut(next.as_mut_ptr().cast::<u16>(), next.len()) };
    if !super::native_engine::process_engine().depthwise_stream_bf16(
        x_bits,
        cache_bits,
        weight_bits,
        output_bits,
        next_bits,
        batch,
        channels,
        steps,
        kernel,
    ) {
        candle_core::bail!("typed native depthwise stream pass was rejected");
    }
    Ok((
        Tensor::from_vec(output, (batch, channels, steps), x.device())?,
        Tensor::from_vec(next, (batch, channels, prior), x.device())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn typed_stream_matches_metal_reference_contract_across_chunks() {
        if !super::super::native_engine::depthwise_stream_available() {
            eprintln!("depthwise stream opcodes unavailable on this runner - skipping");
            return;
        }
        let dev = Device::Cpu;
        let (batch, channels, kernel, total) = (2usize, 5usize, 3usize, 11usize);
        let source = (0..batch * channels * total)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 9.0)
            .collect::<Vec<_>>();
        let weights = (0..channels * kernel)
            .map(|i| ((i * 5 % 17) as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let source = Tensor::from_vec(source, (batch, channels, total), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weights = Tensor::from_vec(weights, (channels, kernel), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let mut native_state = None;
        let mut reference_state = None;
        let mut offset = 0;
        for steps in [1usize, 4, 2, 4] {
            let chunk = source
                .narrow(2, offset, steps)
                .unwrap()
                .contiguous()
                .unwrap();
            let (native, next) =
                depthwise_conv1d_stream(&chunk, &weights, native_state.as_ref()).unwrap();
            let (reference, reference_next) = candle_flashfftconv::depthwise_conv1d_stream(
                &chunk,
                &weights,
                reference_state.as_ref(),
            )
            .unwrap();
            assert_eq!(
                native
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<half::bf16>()
                    .unwrap(),
                reference
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<half::bf16>()
                    .unwrap()
            );
            assert_eq!(
                next.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap(),
                reference_next
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<half::bf16>()
                    .unwrap()
            );
            native_state = Some(next);
            reference_state = Some(reference_next);
            offset += steps;
        }
    }
}
