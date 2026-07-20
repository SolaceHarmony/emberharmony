#ifndef FLASHKERN_CONV_H
#define FLASHKERN_CONV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CPU architecture leaf for streaming depthwise causal convolution. `cache`
 * is either NULL (fresh stream) or [B,D,K-1]. `out` is
 * [B,D,T] and `next` is [B,D,K-1]. Input payloads are borrowed. */
int lfm_depthwise_stream_bf16_available(void);
void lfm_depthwise_stream_bf16(const uint16_t *x, const uint16_t *cache,
                               const uint16_t *weights, uint16_t *out,
                               uint16_t *next,
                               int batch, int channels, int steps, int kernel);

#ifdef __cplusplus
}
#endif

#endif /* FLASHKERN_CONV_H */
