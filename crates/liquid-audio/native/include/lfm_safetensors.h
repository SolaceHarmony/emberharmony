#ifndef LFM_SAFETENSORS_H
#define LFM_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_WEIGHT_ABI_VERSION 1u

typedef struct LfmWeightImage LfmWeightImage;

typedef enum LfmWeightStatus {
    LFM_WEIGHT_OK = 0,
    LFM_WEIGHT_INVALID_ARGUMENT = -1,
    LFM_WEIGHT_IO_ERROR = -2,
    LFM_WEIGHT_FORMAT_ERROR = -3,
    LFM_WEIGHT_OUT_OF_MEMORY = -4,
    LFM_WEIGHT_NOT_FOUND = -5,
} LfmWeightStatus;

typedef enum LfmWeightDType {
    LFM_DTYPE_INVALID = 0,
    LFM_DTYPE_BOOL = 1,
    LFM_DTYPE_F4 = 2,
    LFM_DTYPE_F6_E2M3 = 3,
    LFM_DTYPE_F6_E3M2 = 4,
    LFM_DTYPE_U8 = 5,
    LFM_DTYPE_I8 = 6,
    LFM_DTYPE_F8_E5M2 = 7,
    LFM_DTYPE_F8_E4M3 = 8,
    LFM_DTYPE_F8_E8M0 = 9,
    LFM_DTYPE_I16 = 10,
    LFM_DTYPE_U16 = 11,
    LFM_DTYPE_F16 = 12,
    LFM_DTYPE_BF16 = 13,
    LFM_DTYPE_I32 = 14,
    LFM_DTYPE_U32 = 15,
    LFM_DTYPE_F32 = 16,
    LFM_DTYPE_C64 = 17,
    LFM_DTYPE_F64 = 18,
    LFM_DTYPE_I64 = 19,
    LFM_DTYPE_U64 = 20,
} LfmWeightDType;

/* Every pointer remains valid until lfm_weights_close(image). `offset` is
 * relative to lfm_weights_data(image), so native kernels may bind one base
 * pointer and keep compact offset descriptors instead of retaining C structs. */
typedef struct LfmTensorView {
    uint32_t size;
    uint32_t abi_version;
    const char *name;
    const void *data;
    const uint64_t *shape;
    uint64_t offset;
    uint64_t elements;
    uint64_t bytes;
    uint32_t rank;
    uint32_t dtype;
    uint32_t shard;
    uint32_t reserved;
} LfmTensorView;

/* Open one .safetensors file, a model.safetensors.index.json file, or a
 * checkpoint directory. A directory prefers the Hugging Face shard index,
 * then model.safetensors, then sorted model-*.safetensors shards. All selected
 * files are read directly into one 64-byte-aligned resident allocation. */
int lfm_weights_open(const char *path, LfmWeightImage **out,
                     char *err, size_t errlen);

/* Explicit multi-file entry point for hosts that already resolved a shard set. */
int lfm_weights_open_files(const char *const *paths, size_t count,
                           LfmWeightImage **out, char *err, size_t errlen);

void lfm_weights_close(LfmWeightImage *image);

const void *lfm_weights_data(const LfmWeightImage *image);
uint64_t lfm_weights_resident_bytes(const LfmWeightImage *image);
size_t lfm_weights_count(const LfmWeightImage *image);

int lfm_weights_at(const LfmWeightImage *image, size_t index,
                   LfmTensorView *out);
int lfm_weights_find(const LfmWeightImage *image, const char *name,
                     LfmTensorView *out);

const char *lfm_weights_dtype_name(uint32_t dtype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_SAFETENSORS_H */
