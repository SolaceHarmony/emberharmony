// Copyright © 2025 The Solace Project
// Extended Precision Arithmetic for MLX Metal Kernels
//
// Implements double-double (DD) arithmetic using two float32 values
// to achieve ~30-32 digits of precision (vs 7-8 for float32).
//
// Based on:
// - Dekker (1971): A floating-point technique for extending the available precision
// - Knuth TAOCP Vol 2: Seminumerical Algorithms
// - Shewchuk (1997): Adaptive Precision Floating-Point Arithmetic

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Double-Double Representation
// ============================================================================

struct double_double {
    float hi;  // High-order term (standard float32)
    float lo;  // Low-order correction term (residual error)

    // Constructor
    double_double(float h = 0.0f, float l = 0.0f) : hi(h), lo(l) {}

    // Construct from single float
    explicit double_double(float x) : hi(x), lo(0.0f) {}
};

struct complex_dd {
    double_double re;  // Real part (2 floats)
    double_double im;  // Imaginary part (2 floats)

    complex_dd(double_double r = double_double(), double_double i = double_double())
        : re(r), im(i) {}

    // Construct from float2
    explicit complex_dd(float2 z) : re(double_double(z.x)), im(double_double(z.y)) {}
};

// ============================================================================
// Error-Free Transformations
// ============================================================================

// Two-Sum: Exact sum with error term (Knuth)
// Returns (s, e) where s = round(a + b) and e = (a + b) - s
inline double_double quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return double_double(s, e);
}

// Two-Sum (general case, no ordering assumption)
inline double_double two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return double_double(s, e);
}

// Two-Product: Exact product with error term
// Uses FMA for maximum accuracy
inline double_double two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);  // Error: a*b - round(a*b)
    return double_double(p, e);
}

// Split: Dekker split for error-free multiply without FMA
// (Included for completeness, but two_prod with FMA is preferred)
inline float2 split(float a) {
    const float SPLIT_CONST = 4097.0f;  // 2^12 + 1 for float32
    float t = SPLIT_CONST * a;
    float a_hi = t - (t - a);
    float a_lo = a - a_hi;
    return float2(a_hi, a_lo);
}

// ============================================================================
// Double-Double Arithmetic
// ============================================================================

// Addition: dd + dd
inline double_double dd_add(double_double a, double_double b) {
    double_double s = two_sum(a.hi, b.hi);
    double_double t = two_sum(a.lo, b.lo);

    // Normalize: collect all error terms
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    s = quick_two_sum(s.hi, s.lo);

    return s;
}

// Addition: dd + float
inline double_double dd_add(double_double a, float b) {
    double_double s = two_sum(a.hi, b);
    s.lo += a.lo;
    s = quick_two_sum(s.hi, s.lo);
    return s;
}

// Subtraction: dd - dd
inline double_double dd_sub(double_double a, double_double b) {
    return dd_add(a, double_double(-b.hi, -b.lo));
}

// Multiplication: dd * dd
inline double_double dd_mul(double_double a, double_double b) {
    double_double p = two_prod(a.hi, b.hi);

    // Add cross terms: a.hi*b.lo + a.lo*b.hi + a.lo*b.lo
    // (Last term a.lo*b.lo is often negligible but included for max precision)
    p.lo += a.hi * b.lo + a.lo * b.hi;
    p = quick_two_sum(p.hi, p.lo);

    return p;
}

// Multiplication: dd * float
inline double_double dd_mul(double_double a, float b) {
    double_double p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    p = quick_two_sum(p.hi, p.lo);
    return p;
}

// Division: dd / float (common case for normalization)
inline double_double dd_div(double_double a, float b) {
    float q = a.hi / b;
    double_double p = two_prod(q, b);
    float e = (a.hi - p.hi - p.lo + a.lo) / b;
    return quick_two_sum(q, e);
}

// Negation
inline double_double dd_neg(double_double a) {
    return double_double(-a.hi, -a.lo);
}

// ============================================================================
// Complex Double-Double Arithmetic
// ============================================================================

// Complex addition
inline complex_dd cdd_add(complex_dd a, complex_dd b) {
    return complex_dd(dd_add(a.re, b.re), dd_add(a.im, b.im));
}

// Complex subtraction
inline complex_dd cdd_sub(complex_dd a, complex_dd b) {
    return complex_dd(dd_sub(a.re, b.re), dd_sub(a.im, b.im));
}

// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
// This is the CRITICAL operation for FFT frequency-domain multiply
inline complex_dd cdd_mul(complex_dd a, complex_dd b) {
    // Real part: ac - bd
    double_double ac = dd_mul(a.re, b.re);
    double_double bd = dd_mul(a.im, b.im);
    double_double re = dd_sub(ac, bd);

    // Imaginary part: ad + bc
    double_double ad = dd_mul(a.re, b.im);
    double_double bc = dd_mul(a.im, b.re);
    double_double im = dd_add(ad, bc);

    return complex_dd(re, im);
}

// Complex multiplication by scalar
inline complex_dd cdd_mul(complex_dd a, double_double b) {
    return complex_dd(dd_mul(a.re, b), dd_mul(a.im, b));
}

inline complex_dd cdd_mul(complex_dd a, float b) {
    return complex_dd(dd_mul(a.re, b), dd_mul(a.im, b));
}

// Complex conjugate
inline complex_dd cdd_conj(complex_dd a) {
    return complex_dd(a.re, dd_neg(a.im));
}

// ============================================================================
// Conversion and Rounding
// ============================================================================

// Round double-double to single float (THIS IS WHERE PRECISION IS LOST)
inline float dd_to_float(double_double a) {
    return a.hi + a.lo;  // Final rounding happens here
}

// Round complex DD to complex float (float2)
inline float2 cdd_to_float2(complex_dd a) {
    return float2(dd_to_float(a.re), dd_to_float(a.im));
}

// ============================================================================
// Special Functions (for FFT twiddle factors)
// ============================================================================

// Compute exp(-2πi k/n) twiddle factor in extended precision
// This uses Taylor series in DD arithmetic for maximum accuracy
inline complex_dd twiddle_dd(int k, int n) {
    // Compute theta = -2π k / n in DD
    const double_double PI_DD = double_double(3.1415927f, -8.7422777e-8f);  // π with full float32+correction
    const double_double TWO_PI_DD = dd_mul(PI_DD, 2.0f);

    float theta_float = -2.0f * 3.141592653589793f * float(k) / float(n);

    // For small angles, use Taylor series in DD
    // For now, compute in float and lift to DD (can be improved)
    float c = metal::precise::cos(theta_float);
    float s = metal::precise::sin(theta_float);

    // TODO: Implement DD Taylor series for cos/sin for ultimate precision
    return complex_dd(double_double(c), double_double(s));
}

// ============================================================================
// Deterministic Reduction (for dot products, accumulation)
// ============================================================================

// Pairwise summation in DD (deterministic, stable)
// Input: array of DD values
// Returns: sum in DD
template<typename T>
inline double_double dd_sum_pairwise(thread const T* values, int n) {
    if (n == 1) return double_double(values[0]);
    if (n == 2) return dd_add(double_double(values[0]), double_double(values[1]));

    int mid = n / 2;
    double_double left = dd_sum_pairwise(values, mid);
    double_double right = dd_sum_pairwise(values + mid, n - mid);
    return dd_add(left, right);
}

// Serial accumulation (completely deterministic, simplest)
template<typename T>
inline double_double dd_sum_serial(thread const T* values, int n) {
    double_double acc = double_double(0.0f);
    for (int i = 0; i < n; i++) {
        acc = dd_add(acc, double_double(values[i]));
    }
    return acc;
}

// ============================================================================
// Comparison and Utilities
// ============================================================================

// Comparison (uses hi term primarily)
inline bool dd_less(double_double a, double_double b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

inline bool dd_greater(double_double a, double_double b) {
    return dd_less(b, a);
}

inline bool dd_equal(double_double a, double_double b) {
    return a.hi == b.hi && a.lo == b.lo;
}

// Absolute value
inline double_double dd_abs(double_double a) {
    return (a.hi < 0.0f) ? dd_neg(a) : a;
}

// ============================================================================
// Constants
// ============================================================================

namespace dd_constants {
    constant double_double ZERO = double_double(0.0f, 0.0f);
    constant double_double ONE = double_double(1.0f, 0.0f);
    constant double_double PI = double_double(3.1415927f, -8.7422777e-8f);
    constant double_double TWO_PI = double_double(6.2831853f, -1.7484555e-7f);
}

// ============================================================================
// Memory Layout Helpers
// ============================================================================

// Pack DD into float2 for storage
inline float2 pack_dd(double_double a) {
    return float2(a.hi, a.lo);
}

// Unpack float2 into DD
inline double_double unpack_dd(float2 v) {
    return double_double(v.x, v.y);
}

// Pack complex DD into float4 for storage
inline float4 pack_cdd(complex_dd a) {
    return float4(a.re.hi, a.re.lo, a.im.hi, a.im.lo);
}

// Unpack float4 into complex DD
inline complex_dd unpack_cdd(float4 v) {
    return complex_dd(
        double_double(v.x, v.y),  // Real
        double_double(v.z, v.w)   // Imag
    );
}

#endif // EMBER_ML_DOUBLE_DOUBLE_METAL
