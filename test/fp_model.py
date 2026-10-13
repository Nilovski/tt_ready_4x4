# SPDX-License-Identifier: Apache-2.0
#
# Bit-exact reference for the BF16 datapath (bf16_mul.sv, fp_add.sv,
# mac_unit.sv), written independently of the RTL algorithm: every operation
# is computed exactly with Fractions and rounded once.
#
# Formats: 8-bit exponent (bias 127), M mantissa bits. M=7 -> BF16, M=23 -> FP32.
# Rules (must match the RTL header comments):
#   - subnormals are zero on input and flush to zero on output (FTZ/DAZ)
#   - round to nearest, ties to even
#   - NaN in, inf*0, inf + -inf -> canonical quiet NaN (sign 0, mantissa MSB)
#   - overflow -> signed inf; underflow -> signed zero
#   - add: exact cancellation -> +0; zero + zero -> -0 only if both -0;
#     x + 0 -> x (bit-identical)
#   - mul: any zero operand (finite other) -> signed zero

from fractions import Fraction
import struct

BF16_M = 7
FP32_M = 23


def fields(x, m):
    return (x >> (8 + m)) & 1, (x >> m) & 0xFF, x & ((1 << m) - 1)


def pack(s, e, f, m):
    return (s << (8 + m)) | (e << m) | f


def qnan(m):
    return pack(0, 0xFF, 1 << (m - 1), m)


def is_nan(x, m):
    _, e, f = fields(x, m)
    return e == 0xFF and f != 0


def is_inf(x, m):
    _, e, f = fields(x, m)
    return e == 0xFF and f == 0


def is_zero(x, m):  # includes subnormals (DAZ)
    return fields(x, m)[1] == 0


def value(x, m):
    """Exact value of a finite, non-zero (normal) encoding."""
    s, e, f = fields(x, m)
    v = Fraction((1 << m) | f, 1 << m) * (Fraction(2) ** (e - 127))
    return -v if s else v


def round_to(v, m, sign_if_zero=0):
    """Round exact non-zero Fraction v to the format, RNE, FTZ, overflow->inf."""
    if v == 0:
        return pack(sign_if_zero, 0, 0, m)
    s = 1 if v < 0 else 0
    a = -v if s else v
    # find E with 2^E <= a < 2^(E+1)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    scaled = a / (Fraction(2) ** (e - m))  # in [2^m, 2^(m+1))
    q = scaled.numerator // scaled.denominator
    rem = scaled - q
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and q & 1):
        q += 1
    if q == 1 << (m + 1):
        q >>= 1
        e += 1
    be = e + 127
    if be >= 0xFF:
        return pack(s, 0xFF, 0, m)
    if be <= 0:
        return pack(s, 0, 0, m)
    return pack(s, be, q - (1 << m), m)


def mul(a, b, out_m):
    """BF16 a * BF16 b -> format with out_m mantissa bits."""
    sa, sb = fields(a, BF16_M)[0], fields(b, BF16_M)[0]
    sy = sa ^ sb
    if is_nan(a, BF16_M) or is_nan(b, BF16_M):
        return qnan(out_m)
    if (is_inf(a, BF16_M) and is_zero(b, BF16_M)) or (is_inf(b, BF16_M) and is_zero(a, BF16_M)):
        return qnan(out_m)
    if is_inf(a, BF16_M) or is_inf(b, BF16_M):
        return pack(sy, 0xFF, 0, out_m)
    if is_zero(a, BF16_M) or is_zero(b, BF16_M):
        return pack(sy, 0, 0, out_m)
    return round_to(value(a, BF16_M) * value(b, BF16_M), out_m, sy)


def add(a, b, m):
    sa, sb = fields(a, m)[0], fields(b, m)[0]
    if is_nan(a, m) or is_nan(b, m):
        return qnan(m)
    if is_inf(a, m) and is_inf(b, m) and sa != sb:
        return qnan(m)
    if is_inf(a, m):
        return pack(sa, 0xFF, 0, m)
    if is_inf(b, m):
        return pack(sb, 0xFF, 0, m)
    if is_zero(a, m) and is_zero(b, m):
        return pack(sa & sb, 0, 0, m)
    if is_zero(a, m):
        return b
    if is_zero(b, m):
        return a
    v = value(a, m) + value(b, m)
    if v == 0:
        return 0
    # underflow sign = sign of the larger-magnitude operand = sign of v
    return round_to(v, m)


def mac(acc, a, b, acc_m, clr=False):
    """One mac_unit cycle with en=1."""
    return add(0 if clr else acc, mul(a, b, acc_m), acc_m)


# ---------------------------------------------------------------------------
# conversions for host code / tests
# ---------------------------------------------------------------------------
def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def to_bf16(x):
    """Python float -> BF16 bits, round to nearest even (via FP32)."""
    u = f32_bits(x)
    if (u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF):
        return 0x7FC0
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFFFFFF
    return u >> 16


def bf16_to_float(h):
    return bits_f32(h << 16)


def acc_to_float(x, acc_m):
    return bits_f32(x if acc_m == FP32_M else x << 16)


# ---------------------------------------------------------------------------
# cycle-accurate model of the 4x4 output-stationary array (gemm_top)
# ---------------------------------------------------------------------------
def array_run(a, b, acc_m, prev=None, n=4, k=4):
    """One gemm_top run on BF16 bit patterns a (n*k) and b (k*n), row-major.

    prev=None -> CLEAR_GEMM (clr on cycle 0); otherwise GEMM onto prev (list of
    n*n accumulator bit patterns). Mirrors the RTL exactly: PE(i,j) sees
    A[i][t-i-j] and B[t-i-j][j] on cycle t (+0 outside the window) for
    T = k + 2(n-1) cycles, accumulating in issue order.
    """
    T = k + 2 * (n - 1)
    out = []
    for i in range(n):
        for j in range(n):
            c = 0 if prev is None else prev[i * n + j]
            for t in range(T):
                kk = t - i - j
                av = a[i * k + kk] if 0 <= kk < k else 0
                bv = b[kk * n + j] if 0 <= kk < k else 0
                c = mac(c, av, bv, acc_m, clr=(prev is None and t == 0))
            out.append(c)
    return out


def acc_to_fp32_bits(x, acc_m):
    """What READ_C returns: the accumulator left-aligned to FP32."""
    return x if acc_m == FP32_M else (x << 16)
