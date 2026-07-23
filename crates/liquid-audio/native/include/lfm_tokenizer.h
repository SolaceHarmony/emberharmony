#ifndef LFM_TOKENIZER_H
#define LFM_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_types.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct LfmTokenizer LfmTokenizer;
typedef struct LfmTokenizerWorkspace LfmTokenizerWorkspace;

typedef struct LfmTokenizerSpecial {
    uint32_t im_start;
    uint32_t im_end;
    uint32_t text_end;
    uint32_t audio_start;
} LfmTokenizerSpecial;

typedef struct LfmTokenizerWorkspaceInfo {
    uint64_t max_input_bytes;
    uint64_t storage_bytes;
    uint64_t encode_calls;
} LfmTokenizerWorkspaceInfo;

/* Private native model-construction interface. */
int lfm_tokenizer_open(const char *path, LfmTokenizer **out,
                       char *error, size_t error_length);
void lfm_tokenizer_close(LfmTokenizer *tokenizer);
int lfm_tokenizer_special(const LfmTokenizer *tokenizer,
                          LfmTokenizerSpecial *out);

/* Fixed-capacity hot-path storage. Creation performs exactly one allocation;
 * encode_bounded never grows it or materializes strings/symbol vectors. */
int lfm_tokenizer_workspace_create(size_t max_input_bytes,
                                   LfmTokenizerWorkspace **out);
void lfm_tokenizer_workspace_destroy(LfmTokenizerWorkspace *workspace);
int lfm_tokenizer_workspace_info(const LfmTokenizerWorkspace *workspace,
                                 LfmTokenizerWorkspaceInfo *out);

/* Allocation-free ByteLevel+BPE encoding into caller-owned storage. Input or
 * output beyond the readiness bound returns -ENOBUFS; malformed UTF-8 returns
 * -EINVAL. `out_count` receives the exact token count whenever the input fits. */
int lfm_tokenizer_encode_bounded(const LfmTokenizer *tokenizer,
                                 LfmTokenizerWorkspace *workspace,
                                 const char *text, size_t text_bytes,
                                 uint32_t *out, size_t out_capacity,
                                 size_t *out_count);

/* Encode without post-processor special tokens. `out_count` always receives
 * the required count; -ENOSPC leaves `out` untouched. */
int lfm_tokenizer_encode(const LfmTokenizer *tokenizer, const char *text,
                         size_t text_bytes, uint32_t *out, size_t out_capacity,
                         size_t *out_count);

/* Decode one vocabulary ID through the checkpoint's ByteLevel decoder.
 * Special tokens produce zero bytes when skip_special is nonzero. `out_bytes`
 * always receives the required byte count; -ENOSPC leaves `out` untouched. */
int lfm_tokenizer_decode_piece(const LfmTokenizer *tokenizer, uint32_t token,
                               uint32_t skip_special, uint8_t *out,
                               size_t out_capacity, size_t *out_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_TOKENIZER_H */
