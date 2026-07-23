// Private retained Conformer continuation. This is deliberately not a product
// ABI: one Flashkern audio ticket owns this storage from admission through the
// adapter's final GEMM. No C++ stack frame or worker-local state survives a
// team generation.
#ifndef LFM_CONFORMER_PROGRAM_H
#define LFM_CONFORMER_PROGRAM_H

#include "lfm_conformer.h"

#include <cstddef>
#include <cstdint>

enum : uint32_t {
    LFM_CONFORMER_STAGE_SERIAL = 1,
    LFM_CONFORMER_STAGE_GEMM = 2,
    LFM_CONFORMER_STAGE_DONE = 3,
};

struct LfmConformerGemmStage {
    const uint16_t *activation = nullptr;
    size_t activation_count = 0;
    const void *weight_bytes = nullptr;
    size_t weight_count = 0;
    const void *bias_bytes = nullptr;
    size_t bias_count = 0;
    uint16_t *out = nullptr;
    size_t out_count = 0;
    size_t rows = 0;
    size_t columns = 0;
    size_t inner = 0;
};

// The implementation asserts that its pointer/cursor-only state fits here.
// Storage is embedded in the pass slot, so suspension never depends on an
// allocator or on a blocked native stack.
struct alignas(16) LfmConformerProgram {
    unsigned char storage[1024] = {};
};

int lfm_conformer_program_begin(
    LfmConformerProgram *program, const LfmConformer *conformer,
    LfmConformerWorkspace *workspace, const uint16_t *mel,
    uint64_t mel_frames, uint16_t *out_rows, uint64_t out_capacity_values);
uint32_t lfm_conformer_program_stage(const LfmConformerProgram *program);
int lfm_conformer_program_run_serial(LfmConformerProgram *program);
int lfm_conformer_program_gemm(const LfmConformerProgram *program,
                               LfmConformerGemmStage *stage);
// Called only by the fixed-team generation callback. Returns one while another
// stage is runnable, zero after the adapter output is complete, or a negative
// errno-style status on a state/protocol fault.
int lfm_conformer_program_advance(LfmConformerProgram *program);

#endif // LFM_CONFORMER_PROGRAM_H
