#ifndef FLASHKERN_FFT_H
#define FLASHKERN_FFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Double-double FFT convolution and inverse DFT through the resident lane team.
// Every payload is borrowed until the exact blocking completion returns.
int lfm_engine_fft_conv_dd(void *engine,
                           const float *input, size_t input_count,
                           const float *kernel, size_t kernel_count,
                           const float *skip, size_t skip_count,
                           float *out, size_t out_count,
                           size_t batch, size_t channels,
                           size_t steps, size_t fft_size);

int lfm_engine_irfft_dd(void *engine,
                        const float *real, size_t real_count,
                        const float *imag, size_t imag_count,
                        float *out, size_t out_count,
                        size_t rows, size_t fft_size,
                        float scale_hi, float scale_lo);

#ifdef __cplusplus
}
#endif

#endif
