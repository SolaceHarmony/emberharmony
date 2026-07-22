#ifndef LFM_SAFETENSORS_H
#define LFM_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_WEIGHT_ABI_VERSION 2u

typedef struct LfmWeightImage LfmWeightImage;

typedef enum LfmWeightComponent {
    LFM_WEIGHT_COMPONENT_INVALID = 0,
    LFM_WEIGHT_COMPONENT_MAIN = 1,
    LFM_WEIGHT_COMPONENT_DETOKENIZER = 2,
    /* Reserved for the separate native Moshi model tranche. The LFM2.5
     * loader never populates this component. */
    LFM_WEIGHT_COMPONENT_MIMI = 3,
    LFM_WEIGHT_COMPONENT_COUNT = 4,
} LfmWeightComponent;

typedef enum LfmWeightStatus {
    LFM_WEIGHT_OK = 0,
    LFM_WEIGHT_INVALID_ARGUMENT = -1,
    LFM_WEIGHT_IO_ERROR = -2,
    LFM_WEIGHT_FORMAT_ERROR = -3,
    LFM_WEIGHT_OUT_OF_MEMORY = -4,
    LFM_WEIGHT_NOT_FOUND = -5,
    /* A matching segment is being built by a live owner. Synchronous callers
     * must return this edge to their owning continuation; they must not poll,
     * sleep, or park a physical worker beside the BUILDING generation. */
    LFM_WEIGHT_IN_PROGRESS = -6,
    /* A same-name shared object failed the identity/layout/security ladder. */
    LFM_WEIGHT_REJECTED = -7,
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

typedef enum LfmWeightLoadFlags {
    LFM_WEIGHT_LOAD_BUILT = 1u << 0,
    LFM_WEIGHT_LOAD_ATTACHED = 1u << 1,
    LFM_WEIGHT_LOAD_WIRED = 1u << 2,
    LFM_WEIGHT_LOAD_REGISTRY_REUSED = 1u << 3,
} LfmWeightLoadFlags;

/* Per-process provenance for one immutable shared weight segment. The two
 * ownership byte counts are mutually exclusive for the process's first lease:
 * a builder reports segment_constructed_bytes; a new mapping reports
 * attached_shared_bytes; a process-registry lease reports neither. Shared
 * pages are never presented as a fresh private allocation for every model.
 * process_resident_bytes describes the one process mapping and is deliberately
 * non-additive across handles or processes. */
typedef struct LfmWeightLoadStatsV2 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t source_bytes;
    uint64_t segment_bytes;
    uint64_t segment_constructed_bytes;
    uint64_t attached_shared_bytes;
    uint64_t wired_bytes;
    uint64_t process_resident_bytes;
    uint64_t build_ns;
    uint64_t attach_ns;
    uint64_t generation;
    uint32_t task_count;
    uint32_t worker_count;
    uint32_t flags;
    uint32_t source_count;
    /* Tensor-payload I/O performed by this open. These are the decisive
     * build-vs-attach counters: a builder reports one record per positioned
     * read task and all source bytes; every attach reports zero for both. */
    uint64_t payload_read_calls;
    uint64_t payload_read_bytes;
    uint8_t identity_digest[32];
    uint8_t content_digest[32];
} LfmWeightLoadStatsV2;

/* Open one .safetensors file, a model.safetensors.index.json file, or a
 * checkpoint directory. A directory prefers the Hugging Face shard index,
 * then model.safetensors, then sorted model-*.safetensors shards. All selected
 * files are opened first and read by a bounded positioned-I/O team directly
 * into one page-aligned resident region. After complete validation the region
 * is sealed read-only; returned tensor views remain byte-exact and immutable. */
LFM_INTERNAL_API int lfm_weights_open(const char *path, LfmWeightImage **out,
                                    char *err, size_t errlen);

/* Explicit multi-file entry point for hosts that already resolved a shard set. */
LFM_INTERNAL_API int lfm_weights_open_files(const char *const *paths, size_t count,
                                          LfmWeightImage **out, char *err,
                                          size_t errlen);

/* Load the model checkpoint and LFM2.5 audio detokenizer with one allocation
 * and one read
 * team. Tensor names are scoped by component, so identical keys in Main and
 * Detokenizer are legal while duplicates inside either component remain
 * errors. */
LFM_INTERNAL_API int lfm_weights_open_bundle(const char *main_path,
                                           const char *detokenizer_path,
                                           LfmWeightImage **out, char *err,
                                           size_t errlen);

LFM_INTERNAL_API void lfm_weights_close(LfmWeightImage *image);

/* Remove only the machine-wide name for an exact checkpoint identity. Existing
 * mappings and views remain alive through their leases; a later open elects a
 * new builder. Detach never calls this function implicitly. */
LFM_INTERNAL_API int lfm_weights_evict(const uint8_t identity_digest[32],
                                      char *err, size_t errlen);

LFM_INTERNAL_API const void *lfm_weights_data(const LfmWeightImage *image);
LFM_INTERNAL_API uint64_t
lfm_weights_resident_bytes(const LfmWeightImage *image);
LFM_INTERNAL_API size_t lfm_weights_count(const LfmWeightImage *image);
LFM_INTERNAL_API size_t
lfm_weights_component_count(const LfmWeightImage *image, uint32_t component);
/* Native-only shared-segment accounting. The function initializes the complete
 * V2 output, including identity and content digests. */
LFM_INTERNAL_API int lfm_weights_load_stats(const LfmWeightImage *image,
                                          LfmWeightLoadStatsV2 *out);

LFM_INTERNAL_API int lfm_weights_at(const LfmWeightImage *image, size_t index,
                                  LfmTensorView *out);
LFM_INTERNAL_API int lfm_weights_find(const LfmWeightImage *image,
                                    const char *name, LfmTensorView *out);
LFM_INTERNAL_API int lfm_weights_at_component(const LfmWeightImage *image,
                                            uint32_t component, size_t index,
                                            LfmTensorView *out);
LFM_INTERNAL_API int lfm_weights_find_component(const LfmWeightImage *image,
                                              uint32_t component,
                                              const char *name,
                                              LfmTensorView *out);

LFM_INTERNAL_API const char *lfm_weights_dtype_name(uint32_t dtype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_SAFETENSORS_H */
