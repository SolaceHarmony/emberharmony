#ifndef LFM_PAYLOAD_READER_H
#define LFM_PAYLOAD_READER_H

#include "lfm_runtime.h"
#include "kcoro_stackless.h"

#include <stddef.h>
#include <stdint.h>

/* Private construction seam shared by the native model owner, safetensors
 * loader, and tokenizer. A scope is acquired before any path inspection,
 * open, or read. Publication and a live scope are mutually exclusive; each
 * successful logical payload read is then reported exactly once. */
struct LfmPayloadReadOwner {
    void *context;
    int (*begin)(void *context, uint32_t declared_sources,
                 uint64_t attempted_bytes);
    int (*record)(void *context, uint32_t source, uint64_t bytes);
    void (*end)(void *context);
};

class LfmPayloadReadScope final {
  public:
    LfmPayloadReadScope(const LfmPayloadReadOwner *owner,
                        uint32_t declared_sources,
                        uint64_t attempted_bytes = 0)
        : owner_(owner), status_(owner ? owner->begin(
                                            owner->context, declared_sources,
                                            attempted_bytes)
                                      : 0),
          active_(owner && status_ == 0) {}

    ~LfmPayloadReadScope() {
        if (active_) owner_->end(owner_->context);
    }

    LfmPayloadReadScope(const LfmPayloadReadScope &) = delete;
    LfmPayloadReadScope &operator=(const LfmPayloadReadScope &) = delete;

    int status() const { return status_; }

    int record(uint32_t source, uint64_t bytes) const {
        return owner_ ? owner_->record(owner_->context, source, bytes) : 0;
    }

  private:
    const LfmPayloadReadOwner *owner_;
    int status_;
    bool active_;
};

struct LfmWeightImage;
struct LfmTokenizer;

/* These are C++-private owner entry points, not product or oracle ABI. */
int lfm_weights_open_owned(const char *path,
                           const LfmPayloadReadOwner *owner,
                           LfmWeightImage **out, char *error,
                           size_t error_length);
int lfm_weights_open_bundle_owned(const char *main_path,
                                  const char *detokenizer_path,
                                  const LfmPayloadReadOwner *owner,
                                  LfmWeightImage **out, char *error,
                                  size_t error_length);
/* Continuation-aware twins for model readiness. A live BUILDING generation
 * returns LFM_WEIGHT_IN_PROGRESS after retaining this exact GOSUB identity;
 * builder publication resumes it. The caller dehydrates immediately and
 * retries the same open on resume. No physical thread waits beside the image. */
int lfm_weights_open_owned_continuation(
    const char *path, const LfmPayloadReadOwner *owner,
    koro_cont_t *continuation, LfmWeightImage **out, char *error,
    size_t error_length);
int lfm_weights_open_bundle_owned_continuation(
    const char *main_path, const char *detokenizer_path,
    const LfmPayloadReadOwner *owner, koro_cont_t *continuation,
    LfmWeightImage **out, char *error, size_t error_length);
void lfm_weights_cancel_readiness(koro_cont_t *continuation);
int lfm_tokenizer_open_owned(const char *path,
                             const LfmPayloadReadOwner *owner,
                             LfmTokenizer **out, char *error,
                             size_t error_length);

#endif /* LFM_PAYLOAD_READER_H */
