#ifndef LFM_SAFETENSORS_H
#define LFM_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_WEIGHT_ABI_VERSION 1u

typedef struct LfmWeightImage LfmWeightImage;

typedef enum LfmWeightComponent {
    LFM_WEIGHT_COMPONENT_INVALID = 0,
    LFM_WEIGHT_COMPONENT_MAIN = 1,
    LFM_WEIGHT_COMPONENT_CODEC = 2,
    LFM_WEIGHT_COMPONENT_COUNT = 3,
} LfmWeightComponent;

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

/* Metadata-only borrowed view: this descriptor owns no payload storage and
 * performs no conversion, packing, alignment repair, or relocation. Every
 * pointer remains valid until lfm_weights_close(image). `offset` is relative to
 * lfm_weights_data(image), so native kernels may bind one base pointer and keep
 * compact offset descriptors instead of retaining C structs. */
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

typedef struct LfmWeightLoadStatsV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t source_bytes;
    uint64_t resident_bytes;
    uint32_t task_count;
    uint32_t worker_count;
} LfmWeightLoadStatsV1;

/* Open one .safetensors file, a model.safetensors.index.json file, or a
 * checkpoint directory. A directory prefers the Hugging Face shard index,
 * then model.safetensors, then sorted model-*.safetensors shards. All selected
 * files are opened first and read by a bounded positioned-I/O team directly
 * into one page-aligned resident region. After complete validation the region
 * is sealed read-only; returned tensor views remain byte-exact and immutable. */
LFM_ORACLE_API int lfm_weights_open(const char *path, LfmWeightImage **out,
                                    char *err, size_t errlen);

/* Explicit multi-file entry point for hosts that already resolved a shard set. */
LFM_ORACLE_API int lfm_weights_open_files(const char *const *paths, size_t count,
                                          LfmWeightImage **out, char *err,
                                          size_t errlen);

/* Load the model checkpoint and Mimi codec with one allocation and one read
 * team. Tensor names are scoped by component, so identical keys in Main and
 * Codec are legal while duplicates inside either component remain errors. */
LFM_ORACLE_API int lfm_weights_open_bundle(const char *main_path,
                                           const char *codec_path,
                                           LfmWeightImage **out, char *err,
                                           size_t errlen);

LFM_ORACLE_API void lfm_weights_close(LfmWeightImage *image);

LFM_ORACLE_API const void *lfm_weights_data(const LfmWeightImage *image);
LFM_ORACLE_API uint64_t
lfm_weights_resident_bytes(const LfmWeightImage *image);
LFM_ORACLE_API size_t lfm_weights_count(const LfmWeightImage *image);
LFM_ORACLE_API size_t
lfm_weights_component_count(const LfmWeightImage *image, uint32_t component);
/* Transitional native-only accounting. Source bytes exclude alignment padding;
 * resident bytes include it. The function initializes the complete V1 output. */
LFM_ORACLE_API int lfm_weights_load_stats(const LfmWeightImage *image,
                                          LfmWeightLoadStatsV1 *out);

LFM_ORACLE_API int lfm_weights_at(const LfmWeightImage *image, size_t index,
                                  LfmTensorView *out);
LFM_ORACLE_API int lfm_weights_find(const LfmWeightImage *image,
                                    const char *name, LfmTensorView *out);
LFM_ORACLE_API int lfm_weights_at_component(const LfmWeightImage *image,
                                            uint32_t component, size_t index,
                                            LfmTensorView *out);
LFM_ORACLE_API int lfm_weights_find_component(const LfmWeightImage *image,
                                              uint32_t component,
                                              const char *name,
                                              LfmTensorView *out);

LFM_ORACLE_API const char *lfm_weights_dtype_name(uint32_t dtype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_SAFETENSORS_H */
