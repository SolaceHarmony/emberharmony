// monarch_fused_coop.c — standalone cooperative two-factor FFT probe.
//
// This executable links the same kc_team and kc_collective implementations used
// by the production flashkern engine, but it creates its own team, context, and
// storage. It does not enter the engine request bridge or replace its FFT.
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
// Two synchronization modes run the same arithmetic:
//   coarse:     stage A and stage B are separate kc_team generations, with the
//               main thread waiting and redispatching at the boundary;
//   collective: one kc_team generation runs stage A, kc_collective_arrive, then
//               stage B. Members rendezvous without a mid-FFT host redispatch.
// The intermediate is explicitly materialized in ordinary static storage. The
// probe has no cache or memory-traffic counters, so it makes no residency claim.
// BFDOT is Arm BF16 dot product with FP32 accumulation, not an 8x8 tensor-matrix
// operation. Only a forward transform of real input is implemented here.
//
// Build from this directory (add -D_GNU_SOURCE on Linux):
//   KA=../../../kcoro-sys/vendor/kcoro_arena
//   clang -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror \
//     -ffp-contract=off -march=armv8.6-a+bf16 monarch_fused_coop.c \
//     "$KA/core/src/kc_team.c" "$KA/core/src/kc_collective.c" \
//     "$KA/core/src/kc_doorbell.c" "$KA/port/posix.c" \
//     -I"$KA/include" -I"$KA/port" -pthread -lm -o /tmp/mfc && /tmp/mfc

#if !defined(__aarch64__)
#error "monarch_fused_coop requires AArch64"
#endif
#if !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error "monarch_fused_coop requires Arm BF16 vector arithmetic"
#endif

#include "kc_collective.h"
#include "kc_port.h"
#include "kc_team.h"

#include <arm_neon.h>
#include <math.h>
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

typedef struct {
    const uint16_t *input;
    uint16_t *scratch_r;
    uint16_t *scratch_i;
    float *output_r;
    float *output_i;
    kc_collective_t *collective;
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

static void stage_b(Coop *coop, uint32_t member, uint32_t members)
{
    for (int k2 = (int)member; k2 < N; k2 += (int)members) {
        const uint16_t *ar = coop->scratch_r + (size_t)k2 * L;
        const uint16_t *ai = coop->scratch_i + (size_t)k2 * L;
        for (int k1 = 0; k1 < L; ++k1) {
            const float rr = bfdot(ar, dlr[k1], L);
            const float ii = bfdot(ai, dli[k1], L);
            const float ri = bfdot(ar, dli[k1], L);
            const float ir = bfdot(ai, dlr[k1], L);
            coop->output_r[(size_t)k1 * N + k2] = rr - ii;
            coop->output_i[(size_t)k1 * N + k2] = ri + ir;
        }
    }
}

static void member_coarse(void *context, uint32_t member, uint32_t members,
                          uint64_t generation)
{
    Coop *coop = context;
    if (generation & 1u) {
        stage_a(coop, member, members);
        return;
    }
    stage_b(coop, member, members);
}

static void member_collective(void *context, uint32_t member,
                              uint32_t members, uint64_t generation)
{
    (void)generation;
    Coop *coop = context;
    stage_a(coop, member, members);
    if (kc_collective_arrive(coop->collective, member, NULL, NULL) < 0)
        abort();
    stage_b(coop, member, members);
}

static int run_coarse(kc_team_t *team, uint64_t *generation)
{
    const uint64_t first = *generation + 1;
    int status = kc_team_dispatch(team, first);
    if (status != 0) return status;
    status = kc_team_wait(team, first, 0);
    if (status != 0) return status;
    status = kc_team_dispatch(team, first + 1);
    if (status != 0) return status;
    status = kc_team_wait(team, first + 1, 0);
    if (status != 0) return status;
    *generation = first + 1;
    return 0;
}

static int run_collective(kc_team_t *team, uint64_t *generation)
{
    const uint64_t next = *generation + 1;
    int status = kc_team_dispatch(team, next);
    if (status != 0) return status;
    status = kc_team_wait(team, next, 0);
    if (status != 0) return status;
    *generation = next;
    return 0;
}

typedef int (*run_fn)(kc_team_t *team, uint64_t *generation);

static int measure(kc_team_t *team, uint64_t *generation, run_fn run,
                   double *best)
{
    for (int index = 0; index < 50; ++index) {
        const int status = run(team, generation);
        if (status != 0) return status;
    }
    *best = INFINITY;
    for (int round = 0; round < 5; ++round) {
        const uint64_t start = kc_port_monotonic_ns();
        for (int index = 0; index < 400; ++index) {
            const int status = run(team, generation);
            if (status != 0) return status;
        }
        const double elapsed =
            (kc_port_monotonic_ns() - start) / 1.0e9 / 400.0;
        if (elapsed < *best) *best = elapsed;
    }
    return isfinite(*best) ? 0 : EXIT_FAILURE;
}

static int start_team(uint32_t members, kc_team_member_fn member,
                      Coop *coop, kc_team_t **out)
{
    const kc_team_config config = {
        .size = sizeof(config),
        .abi_version = 1,
        .member_count = members,
        .reserved = 0,
        .member = member,
        .context = coop,
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

static int stop_team(kc_team_t *team)
{
    kc_team_request_stop(team);
    const int joined = kc_team_join(team);
    const int destroyed = kc_team_destroy(team);
    return joined != 0 ? joined : destroyed;
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
        .collective = NULL,
    };
    stage_a(&leaf, 0, 1);
    stage_b(&leaf, 0, 1);
    const double leaf_error =
        normalized_max(leaf_r, leaf_i, exact_r, exact_i);

    printf("# Standalone cooperative two-factor BF16 FFT probe\n");
    printf("# N=%d, L=%d, points=%d; row-major output K=k1*N+k2\n", N, L,
           NL);
    printf("# coarse = 2 team generations; collective = 1 team generation + "
           "1 member barrier\n\n");
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

    printf("\n  lanes | coarse==leaf | collective==leaf | modes exact | "
           "coarse err | collective err\n");
    printf("  ------+--------------+------------------+-------------+"
           "------------+---------------\n");

    double one_coarse = 0.0;
    double one_collective = 0.0;
    double coarse_ms[4];
    double collective_ms[4];
    uint32_t lane_count[4];
    int slot = 0;

    for (uint32_t members = 1; members <= 8; members *= 2) {
        static float coarse_r[NL];
        static float coarse_i[NL];
        Coop coarse = {
            .input = input_bf16,
            .scratch_r = scratch_r,
            .scratch_i = scratch_i,
            .output_r = coarse_r,
            .output_i = coarse_i,
            .collective = NULL,
        };
        kc_team_t *coarse_team = NULL;
        int status = start_team(members, member_coarse, &coarse, &coarse_team);
        if (status != 0) {
            fprintf(stderr, "coarse team start failed for %u lanes: %d\n",
                    members, status);
            return EXIT_FAILURE;
        }
        uint64_t coarse_generation = 0;
        status = run_coarse(coarse_team, &coarse_generation);
        const int coarse_equal =
            memcmp(coarse_r, leaf_r, sizeof(coarse_r)) == 0 &&
            memcmp(coarse_i, leaf_i, sizeof(coarse_i)) == 0;
        const double coarse_error =
            normalized_max(coarse_r, coarse_i, exact_r, exact_i);
        double coarse_best = INFINITY;
        if (status == 0)
            status = measure(coarse_team, &coarse_generation, run_coarse,
                             &coarse_best);
        const int coarse_stop = stop_team(coarse_team);
        if (status == 0) status = coarse_stop;
        if (status != 0) {
            fprintf(stderr, "coarse run failed for %u lanes: %d\n", members,
                    status);
            return EXIT_FAILURE;
        }

        static float collective_r[NL];
        static float collective_i[NL];
        kc_collective_t *barrier = NULL;
        status = kc_collective_create(members, &barrier);
        if (status != 0) {
            fprintf(stderr, "collective create failed for %u lanes: %d\n",
                    members, status);
            return EXIT_FAILURE;
        }
        Coop collective = {
            .input = input_bf16,
            .scratch_r = scratch_r,
            .scratch_i = scratch_i,
            .output_r = collective_r,
            .output_i = collective_i,
            .collective = barrier,
        };
        kc_team_t *collective_team = NULL;
        status = start_team(members, member_collective, &collective,
                            &collective_team);
        if (status != 0) {
            kc_collective_destroy(barrier);
            fprintf(stderr, "collective team start failed for %u lanes: %d\n",
                    members, status);
            return EXIT_FAILURE;
        }
        uint64_t collective_generation = 0;
        status = run_collective(collective_team, &collective_generation);
        const int collective_equal =
            memcmp(collective_r, leaf_r, sizeof(collective_r)) == 0 &&
            memcmp(collective_i, leaf_i, sizeof(collective_i)) == 0;
        const int modes_equal =
            memcmp(collective_r, coarse_r, sizeof(collective_r)) == 0 &&
            memcmp(collective_i, coarse_i, sizeof(collective_i)) == 0;
        const double collective_error =
            normalized_max(collective_r, collective_i, exact_r, exact_i);
        double collective_best = INFINITY;
        if (status == 0)
            status = measure(collective_team, &collective_generation,
                             run_collective, &collective_best);
        kc_collective_snapshot snapshot = {.size = sizeof(snapshot)};
        if (status == 0)
            status = kc_collective_snapshot_get(barrier, &snapshot);
        if (status == 0 && snapshot.generation != collective_generation)
            status = EXIT_FAILURE;
        const int collective_stop = stop_team(collective_team);
        if (status == 0) status = collective_stop;
        kc_collective_destroy(barrier);
        if (status != 0) {
            fprintf(stderr, "collective run failed for %u lanes: %d\n",
                    members, status);
            return EXIT_FAILURE;
        }

        printf("  %5u | %12s | %16s | %11s |   %.2e |      %.2e\n",
               members, coarse_equal ? "yes" : "NO",
               collective_equal ? "yes" : "NO",
               modes_equal ? "yes" : "NO", coarse_error,
               collective_error);
        if (!coarse_equal || !collective_equal || !modes_equal ||
            !isfinite(coarse_error) || !isfinite(collective_error) ||
            coarse_error >= MFC_BF16_LIMIT ||
            collective_error >= MFC_BF16_LIMIT) {
            fprintf(stderr, "cooperative correctness gate failed for %u lanes\n",
                    members);
            return EXIT_FAILURE;
        }

        if (members == 1) {
            one_coarse = coarse_best;
            one_collective = collective_best;
        }
        lane_count[slot] = members;
        coarse_ms[slot] = coarse_best * 1.0e3;
        collective_ms[slot] = collective_best * 1.0e3;
        ++slot;
    }

    printf("\n  lanes | coarse ms | collective ms | coarse speedup | "
           "collective speedup | coarse/collective\n");
    printf("  ------+-----------+---------------+----------------+"
           "--------------------+------------------\n");
    for (int index = 0; index < slot; ++index) {
        printf("  %5u | %9.3f | %13.3f | %13.2fx | %17.2fx | %16.2fx\n",
               lane_count[index], coarse_ms[index], collective_ms[index],
               one_coarse / (coarse_ms[index] / 1.0e3),
               one_collective / (collective_ms[index] / 1.0e3),
               coarse_ms[index] / collective_ms[index]);
    }

    printf("\n# Timings include host dispatch/wait; coarse/collective >1 favors "
           "the lane barrier.\n");
    printf("# This gates forward-factorization and partition equivalence only; "
           "no inverse or convolution.\n");
    return EXIT_SUCCESS;
}
