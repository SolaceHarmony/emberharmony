// monarch_fused_coop.c — standalone cooperative two-factor FFT probe.
//
// This executable links the same kc_runtime, kc_service, and kc_team
// implementations used by the production flashkern engine, but it creates its
// own runtime, team, context, and storage. It does not enter the engine request
// bridge or replace its FFT.
//
// For row-major real input x[n][l], the columns-first factorization is:
//   A[k2][l] = sum_n x[n][l] W_N^(n*k2)
//   A[k2][l] *= W_(N*L)^(l*k2)
//   X[k1*N+k2] = sum_l A[k2][l] W_L^(l*k1)
// It is the length-N*L DFT of x.flatten() in natural order K=k1*N+k2.
//
// A separate rows-first formula is also checked. It is the length-N*L DFT of
// x.T.flatten(), in natural order K=p*L+q. That factor-order convention is the
// one used by the recovered MLX formula; it is valid and is not evidence of a
// broken DFT. This probe does not call the MLX kernel itself.
//
// The cooperative transform is one durable PASS ticket with two retained
// phases. Stage A and stage B use separate kc_team generations. The final
// member return publishes the exact ticket and wakes its dormant kc_service;
// that continuation advances A -> B or settles the ticket. There is no host
// redispatch, operation waiter, member barrier, or ticket per stage.
// The intermediate is explicitly materialized in ordinary static storage. The
// probe has no cache or memory-traffic counters, so it makes no residency claim.
// BFDOT is Arm BF16 dot product with FP32 accumulation, not an 8x8 tensor-matrix
// operation. Only a forward transform of real input is implemented here.
//
// Build from this directory (add -D_GNU_SOURCE on Linux):
//   KA=../../../kcoro-sys/vendor/kcoro_arena
//   clang -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror \
//     -ffp-contract=off -march=armv8.6-a+bf16 monarch_fused_coop.c \
//     "$KA/core/src/kc_runtime.c" "$KA/core/src/kc_service.c" \
//     "$KA/core/src/kcoro_stackless.c" "$KA/core/src/kc_team.c" \
//     "$KA/core/src/kc_doorbell.c" "$KA/port/posix.c" \
//     -I"$KA/include" -I"$KA/port" -pthread -lm -o /tmp/mfc && /tmp/mfc

#if !defined(__aarch64__)
#error "monarch_fused_coop requires AArch64"
#endif
#if !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error "monarch_fused_coop requires Arm BF16 vector arithmetic"
#endif

#include "kc_identity.h"
#include "kc_port.h"
#include "kc_runtime.h"
#include "kc_service.h"
#include "kc_team.h"

#include <arm_neon.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MFC_N
#define MFC_N 32
#endif
#ifndef MFC_L
#define MFC_L 32
#endif
#ifndef MFC_BF16_LIMIT
#define MFC_BF16_LIMIT 0.15
#endif

#define N MFC_N
#define L MFC_L
#define NL (N * L)

_Static_assert(N > 0 && L > 0, "Monarch factors must be positive");
_Static_assert(N % 8 == 0 && L % 8 == 0,
               "Monarch factors must be multiples of 8");

static const double PI = 3.14159265358979323846;

static uint16_t f2b(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

// Constant DFT tables, built once and then read by every member.
static uint16_t dnr[N][N];
static uint16_t dni[N][N];
static uint16_t dlr[L][L];
static uint16_t dli[L][L];
static float twr[N][L];
static float twi[N][L];

static void build_tables(void)
{
    for (int a = 0; a < N; ++a) {
        for (int b = 0; b < N; ++b) {
            const double phase = 2.0 * PI * a * b / N;
            dnr[a][b] = f2b((float)cos(phase));
            dni[a][b] = f2b((float)-sin(phase));
        }
    }
    for (int a = 0; a < L; ++a) {
        for (int b = 0; b < L; ++b) {
            const double phase = 2.0 * PI * a * b / L;
            dlr[a][b] = f2b((float)cos(phase));
            dli[a][b] = f2b((float)-sin(phase));
        }
    }
    for (int k2 = 0; k2 < N; ++k2) {
        for (int l = 0; l < L; ++l) {
            const double phase = 2.0 * PI * k2 * l / (double)NL;
            twr[k2][l] = (float)cos(phase);
            twi[k2][l] = (float)-sin(phase);
        }
    }
}

static inline float bfdot(const uint16_t *a, const uint16_t *b, int count)
{
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int k = 0; k + 8 <= count; k += 8) {
        sum = vbfdotq_f32(sum, vreinterpretq_bf16_u16(vld1q_u16(a + k)),
                          vreinterpretq_bf16_u16(vld1q_u16(b + k)));
    }
    return vaddvq_f32(sum);
}

enum {
    PHASE_A = 1,
    PHASE_B = 2,
};

typedef struct {
    kc_ticket_id ticket;
    uint32_t phase;
} Pass;

typedef struct {
    const uint16_t *input;
    uint16_t *scratch_r;
    uint16_t *scratch_i;
    float *output_r;
    float *output_i;
    Pass pass;
} Coop;

static void stage_a(Coop *coop, uint32_t member, uint32_t members)
{
    uint16_t column[N];
    for (int l = (int)member; l < L; l += (int)members) {
        for (int n = 0; n < N; ++n)
            column[n] = coop->input[(size_t)n * L + l];
        for (int k2 = 0; k2 < N; ++k2) {
            const float ar = bfdot(column, dnr[k2], N);
            const float ai = bfdot(column, dni[k2], N);
            const float tr = twr[k2][l];
            const float ti = twi[k2][l];
            coop->scratch_r[(size_t)k2 * L + l] =
                f2b(ar * tr - ai * ti);
            coop->scratch_i[(size_t)k2 * L + l] =
                f2b(ar * ti + ai * tr);
        }
    }
}

static void stage_b(Coop *coop, uint32_t member, uint32_t members,
                    float *output_r, float *output_i)
{
    for (int k2 = (int)member; k2 < N; k2 += (int)members) {
        const uint16_t *ar = coop->scratch_r + (size_t)k2 * L;
        const uint16_t *ai = coop->scratch_i + (size_t)k2 * L;
        for (int k1 = 0; k1 < L; ++k1) {
            const float rr = bfdot(ar, dlr[k1], L);
            const float ii = bfdot(ai, dli[k1], L);
            const float ri = bfdot(ar, dli[k1], L);
            const float ir = bfdot(ai, dlr[k1], L);
            output_r[(size_t)k1 * N + k2] = rr - ii;
            output_i[(size_t)k1 * N + k2] = ri + ir;
        }
    }
}

static void member_fn(void *context, uint32_t member, uint32_t members,
                      uint64_t generation)
{
    (void)generation;
    Coop *coop = context;
    if (coop->pass.ticket.runtime_epoch == 0 ||
        coop->pass.ticket.sequence == 0 ||
        coop->pass.ticket.generation == 0 ||
        coop->pass.ticket.kind != KC_TICKET_KIND_PASS)
        abort();
    if (coop->pass.phase == PHASE_A) {
        stage_a(coop, member, members);
        return;
    }
    if (coop->pass.phase == PHASE_B) {
        stage_b(coop, member, members, coop->output_r, coop->output_i);
        return;
    }
    abort();
}

typedef struct {
    kc_team_t *team;
    kc_service_t *service;
    kc_service_notifier_t *notifier;
    Coop *coop;
    Pass published;
    _Atomic uint64_t ready;
    uint64_t published_generation;
    uint64_t epoch;
    uint64_t sequence;
    uint64_t generation;
    uint64_t completed;
    uint64_t batch_start;
    double best;
    uint32_t plan;
    uint32_t index;
    uint32_t round;
    uint32_t started;
    int status;
} Runner;

enum {
    PLAN_CORRECTNESS = 1,
    PLAN_WARMUP = 2,
    PLAN_MEASURE = 3,
    ADVANCE_DONE = 1,
};

#ifndef MFC_WARMUP_FFTS
#define MFC_WARMUP_FFTS 50
#endif
#ifndef MFC_BATCH_FFTS
#define MFC_BATCH_FFTS 400
#endif
#ifndef MFC_MEASURE_ROUNDS
#define MFC_MEASURE_ROUNDS 6
#endif

static int ticket_equal(const kc_ticket_id *a, const kc_ticket_id *b)
{
    return a->runtime_epoch == b->runtime_epoch &&
           a->sequence == b->sequence &&
           a->generation == b->generation && a->kind == b->kind;
}

static void team_complete(void *context, uint64_t generation);

static int dispatch(Runner *runner)
{
    const uint64_t next = runner->generation + 1;
    if (next == 0) return -EOVERFLOW;
    runner->generation = next;
    const int status = kc_team_dispatch_notify(
        runner->team, next, team_complete, runner);
    if (status == 0) return 0;
    runner->generation = next - 1;
    return status;
}

static int begin_fft(Runner *runner)
{
    const uint64_t next = runner->sequence + 1;
    if (next == 0 || next > UINT32_MAX) return -EOVERFLOW;
    runner->sequence = next;
    runner->coop->pass = (Pass){
        .ticket = {
            .runtime_epoch = runner->epoch,
            .sequence = next,
            .generation = (uint32_t)next,
            .kind = KC_TICKET_KIND_PASS,
        },
        .phase = PHASE_A,
    };
    return dispatch(runner);
}

static void team_complete(void *context, uint64_t generation)
{
    Runner *runner = context;
    if (generation != runner->generation ||
        runner->coop->pass.ticket.sequence == 0 ||
        atomic_load_explicit(&runner->ready, memory_order_relaxed) != 0)
        abort();

    // Publish the exact ticket and durable phase before making its dormant
    // orchestration continuation runnable.
    runner->published = runner->coop->pass;
    runner->published_generation = generation;
    atomic_store_explicit(&runner->ready, generation, memory_order_release);
    if (kc_service_notifier_notify(runner->notifier) != 0) abort();
}

static void finish_runner(Runner *runner, int status)
{
    if (runner->status == 0) runner->status = status;
    kc_team_request_stop(runner->team);
    const int complete = kc_service_complete_current(runner->service);
    if (runner->status == 0 && complete != 0) runner->status = complete;
}

static int advance(Runner *runner)
{
    if (runner->plan == PLAN_CORRECTNESS) {
        runner->plan = PLAN_WARMUP;
        runner->index = 0;
        return begin_fft(runner);
    }
    if (runner->plan == PLAN_WARMUP) {
        ++runner->index;
        if (runner->index < MFC_WARMUP_FFTS) return begin_fft(runner);
        runner->plan = PLAN_MEASURE;
        runner->index = 0;
        runner->round = 0;
        runner->batch_start = kc_port_monotonic_ns();
        return begin_fft(runner);
    }

    ++runner->index;
    if (runner->index < MFC_BATCH_FFTS) return begin_fft(runner);
    const double elapsed =
        (kc_port_monotonic_ns() - runner->batch_start) /
        1.0e9 / MFC_BATCH_FFTS;
    if (!isfinite(elapsed)) return -ERANGE;
    if (elapsed < runner->best) runner->best = elapsed;
    ++runner->round;
    if (runner->round == MFC_MEASURE_ROUNDS) return ADVANCE_DONE;
    runner->index = 0;
    runner->batch_start = kc_port_monotonic_ns();
    return begin_fft(runner);
}

static void runner_service(void *context)
{
    Runner *runner = context;
    const uint64_t generation = atomic_exchange_explicit(
        &runner->ready, 0, memory_order_acq_rel);
    if (!runner->started) {
        runner->started = 1;
        if (generation != 0) {
            finish_runner(runner, -EIO);
            return;
        }
        const int status = begin_fft(runner);
        if (status != 0) finish_runner(runner, status);
        return;
    }

    const Pass published = runner->published;
    if (generation == 0 || generation != runner->published_generation ||
        generation != runner->generation ||
        !ticket_equal(&published.ticket, &runner->coop->pass.ticket) ||
        published.phase != runner->coop->pass.phase) {
        finish_runner(runner, -EIO);
        return;
    }
    if (published.phase == PHASE_A) {
        // Phase is retained on the same ticket; only the team generation
        // changes at the numerical boundary.
        runner->coop->pass.phase = PHASE_B;
        const int status = dispatch(runner);
        if (status != 0) finish_runner(runner, status);
        return;
    }
    if (published.phase != PHASE_B) {
        finish_runner(runner, -EIO);
        return;
    }

    ++runner->completed;
    const int status = advance(runner);
    if (status == ADVANCE_DONE) {
        finish_runner(runner, 0);
        return;
    }
    if (status != 0) finish_runner(runner, status);
}

static int start_team(kc_runtime_t *runtime, uint32_t members,
                      kc_team_member_fn member, Coop *coop, kc_team_t **out)
{
    const kc_team_config config = {
        .member_count = members,
        .member = member,
        .context = coop,
        .runtime = runtime,
        .retired = NULL,
        .retired_context = NULL,
    };
    *out = NULL;
    int status = kc_team_create(&config, out);
    if (status != 0) return status;
    status = kc_team_start(*out);
    if (status == 0) return 0;
    kc_team_request_stop(*out);
    const int joined = kc_team_join(*out);
    const int destroyed = kc_team_destroy(*out);
    *out = NULL;
    if (joined != 0) return joined;
    if (destroyed != 0) return destroyed;
    return status;
}

static int run_runner(kc_runtime_t *runtime, Runner *runner)
{
    kc_service_t *service = NULL;
    kc_service_notifier_t *notifier = NULL;
    int status = 0;

    const kc_service_config service_config = {
        .callback = runner_service,
        .context = runner,
        .owner_init = NULL,
        .owner_fini = NULL,
    };
    if (status == 0)
        status = kc_service_create(runtime, &service_config, &service);
    runner->service = service;
    if (status == 0)
        status = kc_service_notifier_create(service, &notifier);
    runner->notifier = notifier;
    if (status == 0) status = kc_service_start(service);
    if (status == 0) status = kc_service_notifier_notify(notifier);

    if (status != 0) {
        kc_team_request_stop(runner->team);
        if (service) kc_service_request_stop(service);
    }

    /* The service publishes the terminal stop edge. The host observes only
     * whole-run retirement; it never shepherds either FFT phase or a team
     * generation. Team members and orchestration share this one runtime. */
    const int drained = kc_runtime_join_all(runtime);
    if (status == 0) status = drained;
    const int team_joined = kc_team_join(runner->team);
    if (status == 0) status = team_joined;

    if (service) {
        const int service_joined = kc_service_join(service);
        if (status == 0) status = service_joined;
        if (status == 0) status = runner->status;
        if (status == 0) {
            kc_service_snapshot snapshot = {.size = sizeof(snapshot)};
            status = kc_service_snapshot_get(service, &snapshot);
            const uint64_t callbacks = runner->generation + 1;
            if (status == 0 &&
                (snapshot.notifications != callbacks ||
                 snapshot.handled_notifications != callbacks ||
                 snapshot.callbacks != callbacks || !snapshot.joined))
                status = EXIT_FAILURE;
        }
    }

    const int notifier_destroyed =
        kc_service_notifier_destroy(notifier);
    if (status == 0) status = notifier_destroyed;
    const int service_destroyed = kc_service_destroy(service);
    if (status == 0) status = service_destroyed;
    runner->service = NULL;
    runner->notifier = NULL;
    return status;
}

// Columns-first factorization in double, with row-major input/output convention.
static void columns_double(const float *input, double *output_r,
                           double *output_i)
{
    static double ar[N][L];
    static double ai[N][L];
    for (int k2 = 0; k2 < N; ++k2) {
        for (int l = 0; l < L; ++l) {
            double real = 0.0;
            double imag = 0.0;
            for (int n = 0; n < N; ++n) {
                const double phase = 2.0 * PI * n * k2 / N;
                const double value = input[n * L + l];
                real += value * cos(phase);
                imag -= value * sin(phase);
            }
            const double phase = 2.0 * PI * k2 * l / (double)NL;
            const double tr = cos(phase);
            const double ti = -sin(phase);
            ar[k2][l] = real * tr - imag * ti;
            ai[k2][l] = real * ti + imag * tr;
        }
    }
    for (int k2 = 0; k2 < N; ++k2) {
        for (int k1 = 0; k1 < L; ++k1) {
            double real = 0.0;
            double imag = 0.0;
            for (int l = 0; l < L; ++l) {
                const double phase = 2.0 * PI * l * k1 / L;
                const double tr = cos(phase);
                const double ti = -sin(phase);
                real += tr * ar[k2][l] - ti * ai[k2][l];
                imag += tr * ai[k2][l] + ti * ar[k2][l];
            }
            output_r[k1 * N + k2] = real;
            output_i[k1 * N + k2] = imag;
        }
    }
}

// Rows-first factorization in double, with transpose-flatten input convention.
static void rows_double(const float *input, double *output_r, double *output_i)
{
    static double br[N][L];
    static double bi[N][L];
    for (int n = 0; n < N; ++n) {
        for (int q = 0; q < L; ++q) {
            double real = 0.0;
            double imag = 0.0;
            for (int l = 0; l < L; ++l) {
                const double phase = 2.0 * PI * l * q / L;
                const double value = input[n * L + l];
                real += value * cos(phase);
                imag -= value * sin(phase);
            }
            const double phase = 2.0 * PI * n * q / (double)NL;
            const double tr = cos(phase);
            const double ti = -sin(phase);
            br[n][q] = real * tr - imag * ti;
            bi[n][q] = real * ti + imag * tr;
        }
    }
    for (int p = 0; p < N; ++p) {
        for (int q = 0; q < L; ++q) {
            double real = 0.0;
            double imag = 0.0;
            for (int n = 0; n < N; ++n) {
                const double phase = 2.0 * PI * n * p / N;
                const double tr = cos(phase);
                const double ti = -sin(phase);
                real += tr * br[n][q] - ti * bi[n][q];
                imag += tr * bi[n][q] + ti * br[n][q];
            }
            output_r[p * L + q] = real;
            output_i[p * L + q] = imag;
        }
    }
}

static void dft_double(const float *input, double *output_r, double *output_i)
{
    for (int frequency = 0; frequency < NL; ++frequency) {
        double real = 0.0;
        double imag = 0.0;
        for (int sample = 0; sample < NL; ++sample) {
            const double phase = 2.0 * PI * sample * frequency / NL;
            real += input[sample] * cos(phase);
            imag -= input[sample] * sin(phase);
        }
        output_r[frequency] = real;
        output_i[frequency] = imag;
    }
}

static double max_double_delta(const double *actual_r, const double *actual_i,
                               const double *expected_r,
                               const double *expected_i)
{
    double maximum = 0.0;
    for (int k = 0; k < NL; ++k) {
        const double dr = actual_r[k] - expected_r[k];
        const double di = actual_i[k] - expected_i[k];
        const double error = hypot(dr, di);
        if (!isfinite(error)) return INFINITY;
        if (error > maximum) maximum = error;
    }
    return maximum;
}

// Maximum complex-bin error normalized by the reference RMS magnitude.
static double normalized_max(const float *actual_r, const float *actual_i,
                             const double *expected_r,
                             const double *expected_i)
{
    double energy = 0.0;
    for (int k = 0; k < NL; ++k)
        energy += expected_r[k] * expected_r[k] + expected_i[k] * expected_i[k];
    const double scale = sqrt(energy / NL) + 1.0e-12;
    if (!isfinite(scale)) return INFINITY;
    double maximum = 0.0;
    for (int k = 0; k < NL; ++k) {
        const double error =
            hypot((double)actual_r[k] - expected_r[k],
                  (double)actual_i[k] - expected_i[k]) /
            scale;
        if (!isfinite(error)) return INFINITY;
        if (error > maximum) maximum = error;
    }
    return maximum;
}

int main(void)
{
    build_tables();
    static float input[NL];
    static uint16_t input_bf16[NL];
    for (int sample = 0; sample < NL; ++sample) {
        input[sample] =
            ((int)(((uint32_t)sample * 2654435761u >> 13) % 2000) - 1000) /
            1024.0f;
        input_bf16[sample] = f2b(input[sample]);
    }

    static double exact_r[NL];
    static double exact_i[NL];
    columns_double(input, exact_r, exact_i);

    double columns_residual = -1.0;
    double rows_residual = -1.0;
    if (NL <= 4096) {
        static double direct_r[NL];
        static double direct_i[NL];
        dft_double(input, direct_r, direct_i);
        columns_residual =
            max_double_delta(exact_r, exact_i, direct_r, direct_i);

        static float factor_input[NL];
        for (int l = 0; l < L; ++l) {
            for (int n = 0; n < N; ++n)
                factor_input[l * N + n] = input[n * L + l];
        }
        static double rows_r[NL];
        static double rows_i[NL];
        static double factor_r[NL];
        static double factor_i[NL];
        rows_double(input, rows_r, rows_i);
        dft_double(factor_input, factor_r, factor_i);
        rows_residual =
            max_double_delta(rows_r, rows_i, factor_r, factor_i);
    }

    static uint16_t scratch_r[NL];
    static uint16_t scratch_i[NL];
    static float leaf_r[NL];
    static float leaf_i[NL];
    Coop leaf = {
        .input = input_bf16,
        .scratch_r = scratch_r,
        .scratch_i = scratch_i,
        .output_r = leaf_r,
        .output_i = leaf_i,
        .pass = {0},
    };
    stage_a(&leaf, 0, 1);
    stage_b(&leaf, 0, 1, leaf_r, leaf_i);
    const double leaf_error =
        normalized_max(leaf_r, leaf_i, exact_r, exact_i);

    printf("# Standalone cooperative two-factor BF16 FFT probe\n");
    printf("# N=%d, L=%d, points=%d; row-major output K=k1*N+k2\n", N, L,
           NL);
    printf("# ticketed = 1 PASS ticket with retained A/B state across 2 team "
           "generations\n\n");
    if (columns_residual >= 0.0) {
        printf("  columns-first vs DFT(x.flatten())       : %.2e\n",
               columns_residual);
        printf("  rows-first vs DFT(x.T.flatten())       : %.2e\n",
               rows_residual);
    } else {
        printf("  O(points^2) convention oracles         : skipped at %d points\n",
               NL);
    }
    printf("  BF16 leaf normalized maximum error     : %.2e (limit %.2e)\n",
           leaf_error, (double)MFC_BF16_LIMIT);

    if ((columns_residual >= 0.0 &&
         (columns_residual >= 1.0e-6 || rows_residual >= 1.0e-6)) ||
        !isfinite(leaf_error) || leaf_error >= MFC_BF16_LIMIT) {
        fprintf(stderr, "correctness gate failed before cooperative runs\n");
        return EXIT_FAILURE;
    }

    printf("\n  lanes | ticketed==leaf | ticketed err\n");
    printf("  ------+----------------+-------------\n");

    const uint64_t stamp = kc_port_monotonic_ns();
    const uint64_t epoch = stamp != 0 ? stamp : 1;
    uint64_t sequence = 0;
    double one = 0.0;
    double ticket_ms[4];
    uint32_t lane_count[4];
    int slot = 0;

    for (uint32_t members = 1; members <= 8; members *= 2) {
        static float ticket_r[NL];
        static float ticket_i[NL];
        Coop coop = {
            .input = input_bf16,
            .scratch_r = scratch_r,
            .scratch_i = scratch_i,
            .output_r = ticket_r,
            .output_i = ticket_i,
            .pass = {0},
        };
        const kc_runtime_config runtime_config = {
            .worker_count = members,
        };
        kc_runtime_t *runtime = NULL;
        int status = kc_runtime_create(&runtime_config, &runtime);
        if (status == 0) status = kc_runtime_start(runtime);
        kc_team_t *team = NULL;
        if (status == 0)
            status = start_team(runtime, members, member_fn, &coop, &team);
        if (status != 0) {
            fprintf(stderr, "team start failed for %u lanes: %d\n",
                    members, status);
            if (runtime) {
                kc_runtime_request_stop(runtime);
                (void)kc_runtime_join(runtime);
                (void)kc_runtime_destroy(runtime);
            }
            return EXIT_FAILURE;
        }
        Runner runner = {
            .team = team,
            .coop = &coop,
            .epoch = epoch,
            .sequence = sequence,
            .generation = 0,
            .completed = 0,
            .best = INFINITY,
            .plan = PLAN_CORRECTNESS,
            .status = 0,
        };
        atomic_init(&runner.ready, 0);
        const uint64_t first = sequence;
        status = run_runner(runtime, &runner);
        kc_team_snapshot snapshot = {.size = sizeof(snapshot)};
        if (status == 0) status = kc_team_snapshot_get(team, &snapshot);
        if (status == 0 &&
            (runner.completed !=
                 1 + MFC_WARMUP_FFTS +
                     MFC_BATCH_FFTS * MFC_MEASURE_ROUNDS ||
             runner.sequence - first != runner.completed ||
             runner.generation != runner.completed * 2 ||
             snapshot.completed_generation != runner.generation ||
             snapshot.dispatched_generation != runner.generation ||
             snapshot.completed_members != members || !snapshot.joined ||
             !isfinite(runner.best) || runner.published.phase != PHASE_B ||
             !ticket_equal(&runner.published.ticket,
                           &coop.pass.ticket)))
            status = EXIT_FAILURE;
        const int destroyed = kc_team_destroy(team);
        if (status == 0) status = destroyed;
        kc_runtime_request_stop(runtime);
        const int runtime_joined = kc_runtime_join(runtime);
        if (status == 0) status = runtime_joined;
        const int runtime_destroyed = kc_runtime_destroy(runtime);
        if (status == 0) status = runtime_destroyed;
        if (status != 0) {
            fprintf(stderr, "ticketed run failed for %u lanes: %d\n",
                    members, status);
            return EXIT_FAILURE;
        }
        sequence = runner.sequence;

        const int equal =
            memcmp(ticket_r, leaf_r, sizeof(ticket_r)) == 0 &&
            memcmp(ticket_i, leaf_i, sizeof(ticket_i)) == 0;
        const double error =
            normalized_max(ticket_r, ticket_i, exact_r, exact_i);

        printf("  %5u | %14s |    %.2e\n", members,
               equal ? "yes" : "NO", error);
        if (!equal || !isfinite(error) || error >= MFC_BF16_LIMIT) {
            fprintf(stderr, "ticketed correctness gate failed for %u lanes\n",
                    members);
            return EXIT_FAILURE;
        }

        if (members == 1) one = runner.best;
        lane_count[slot] = members;
        ticket_ms[slot] = runner.best * 1.0e3;
        ++slot;
    }

    printf("\n  lanes | ticketed ms | speedup\n");
    printf("  ------+-------------+--------\n");
    for (int index = 0; index < slot; ++index) {
        printf("  %5u | %11.3f | %6.2fx\n", lane_count[index],
               ticket_ms[index], one / (ticket_ms[index] / 1.0e3));
    }

    printf("\n# Timings include retained-service dispatch/completion across both "
           "ticket phases.\n");
    printf("# Wall clock is measurement only; no timer makes work runnable.\n");
    printf("# Direct convention oracles gate builds up to 4096 points; larger "
           "builds use the factorized double reference.\n");
    printf("# Repeated partition equivalence is gated; there is no inverse or "
           "convolution.\n");
    return EXIT_SUCCESS;
}
