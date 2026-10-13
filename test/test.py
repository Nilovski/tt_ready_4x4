# SPDX-License-Identifier: Apache-2.0
#
# cocotb tests for tt_um_sahpar_gemm4x4_bf16 -- 4x4 BF16 output-stationary
# systolic GEMM behind the 8-bit host byte protocol. Results are checked
# bit-exactly against a cycle-accurate model of the array (fp_model.array_run,
# itself built on an exact-arithmetic reference), and loosely against float64.

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

import fp_model as fm
from gemm_protocol import (
    BUSY, CHIP_ID, DONE, K, N, OP_CLEAR_GEMM, OP_GEMM, OP_ID, OP_LOAD_A, OP_NOP,
    RX_READY, TX_VALID, gemm4, load_matrix, matmul, read_c, recv_byte,
    send_byte, status, to_bf16, wait_idle,
)

# accumulator width of the build under test (top-level localparam ACC_W)
ACC_W = int(os.environ.get("ACC_W", "16"))
ACC_M = ACC_W - 9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def expect(a, b, prev=None):
    """Model result as READ_C would return it (FP32 bits), plus raw acc state."""
    acc = fm.array_run(a, b, ACC_M, prev)
    return [fm.acc_to_fp32_bits(x, ACC_M) for x in acc], acc


def model_matmul(a, b, n, k, m):
    """Same block order as gemm_protocol.matmul, through the array model."""
    c = [0] * (n * m)

    def block(mat, rows, cols, r0, c0):
        return [mat[r * cols + cc] if (r < rows and cc < cols) else 0
                for r in range(r0, r0 + N) for cc in range(c0, c0 + K)]

    for bi in range(0, n, N):
        for bj in range(0, m, N):
            acc = None
            for kb in range(0, k, K):
                acc = fm.array_run(block(a, n, k, bi, kb), block(b, k, m, kb, bj), ACC_M, acc)
            for r in range(N):
                for cc in range(N):
                    if bi + r < n and bj + cc < m:
                        c[(bi + r) * m + (bj + cc)] = fm.acc_to_fp32_bits(acc[r * N + cc], ACC_M)
    return c


def hexes(v):
    return " ".join(f"{x:08x}" for x in v)


async def run_and_check(dut, a, b, label=""):
    await gemm4(dut, a, b)
    got = await read_c(dut)
    exp, _ = expect(a, b)
    assert got == exp, f"{label}\n got={hexes(got)}\n exp={hexes(exp)}"
    return got


def rand_bf16(rng, lo=-4.0, hi=4.0):
    return to_bf16(rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_id(dut):
    """ID replies 0xB4; uio directions match the int8 chip."""
    await reset(dut)
    assert int(dut.uio_oe.value) == 0b0011_1010
    assert int(dut.uio_out.value) & (1 << RX_READY)
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID


@cocotb.test()
async def test_status_idle(dut):
    await reset(dut)
    await send_byte(dut, OP_NOP)
    assert await status(dut) == (False, False)
    assert int(dut.uio_out.value) & ((1 << BUSY) | (1 << DONE)) == 0


@cocotb.test()
async def test_exact_integers(dut):
    """Small integers are exact in BF16, so C must equal the integer product."""
    await reset(dut)
    ai = [((i * 3) % 7) - 3 for i in range(16)]
    bi = [((i * 5) % 9) - 4 for i in range(16)]
    got = await run_and_check(dut, [to_bf16(x) for x in ai], [to_bf16(x) for x in bi], "ints")
    exact = [sum(ai[i * 4 + k] * bi[k * 4 + j] for k in range(4)) for i in range(4) for j in range(4)]
    assert [fm.bits_f32(u) for u in got] == [float(x) for x in exact]


@cocotb.test()
async def test_identity(dut):
    await reset(dut)
    rng = random.Random(7)
    ident = [to_bf16(1.0 if i == j else 0.0) for i in range(4) for j in range(4)]
    b = [rand_bf16(rng, -100, 100) for _ in range(16)]
    got = await run_and_check(dut, ident, b, "identity")
    assert got == [x << 16 for x in b], "I x B must return B exactly"


@cocotb.test()
async def test_random_bit_exact(dut):
    """Random operands over a wide exponent range, bit-exact vs the model,
    and close to float64 (within the accumulator's rounding budget)."""
    await reset(dut)
    rng = random.Random(0xBF)
    tol = 2.0 ** -(ACC_M - 1)
    for run in range(8):
        scale = 2.0 ** rng.randint(-20, 20)
        a = [rand_bf16(rng, -scale, scale) for _ in range(16)]
        b = [rand_bf16(rng, -1.0, 1.0) for _ in range(16)]
        got = await run_and_check(dut, a, b, f"run {run}")
        af = [fm.bf16_to_float(x) for x in a]
        bf = [fm.bf16_to_float(x) for x in b]
        for i in range(4):
            for j in range(4):
                terms = [af[i * 4 + k] * bf[k * 4 + j] for k in range(4)]
                g = fm.bits_f32(got[i * 4 + j])
                assert abs(g - sum(terms)) <= 4 * tol * sum(abs(t) for t in terms) + 1e-30, \
                    f"C[{i}][{j}]={g} vs {sum(terms)}"


@cocotb.test()
async def test_specials(dut):
    """inf, NaN, -0, subnormal (flushed), overflow to inf: all bit-exact."""
    await reset(dut)
    INF, NINF, NAN, NZ = 0x7F80, 0xFF80, 0x7FC1, 0x8000
    SUB, BIG = 0x0001, 0x7F7F           # subnormal (-> 0), largest finite
    one, two = to_bf16(1.0), to_bf16(2.0)
    cases = [
        ("inf row", [INF, one, one, one] + [one] * 12, [one] * 16),
        ("inf - inf -> NaN", [INF, NINF, 0, 0] + [one] * 12, [one] * 16),
        ("NaN propagates", [NAN] + [one] * 15, [two] * 16),
        ("inf * 0 -> NaN", [INF] + [one] * 15, [0] + [one] * 15),
        ("-0 products", [NZ] * 16, [one] * 16),
        ("subnormals flush", [SUB] * 16, [BIG] * 16),
        ("overflow -> inf", [BIG] * 16, [two] * 16),
        ("negative overflow", [BIG] * 16, [to_bf16(-2.0)] * 16),
    ]
    for label, a, b in cases:
        await run_and_check(dut, a, b, label)


@cocotb.test()
async def test_accumulate_chain(dut):
    """CLEAR_GEMM then 3 GEMMs: C keeps accumulating, in issue order."""
    await reset(dut)
    rng = random.Random(0xACC)
    acc = None
    for step in range(4):
        a = [rand_bf16(rng) for _ in range(16)]
        b = [rand_bf16(rng) for _ in range(16)]
        await gemm4(dut, a, b, accumulate=(step > 0))
        exp, acc = expect(a, b, acc)
        got = await read_c(dut)
        assert got == exp, f"step {step}\n got={hexes(got)}\n exp={hexes(exp)}"


@cocotb.test()
async def test_tiled_8x8(dut):
    """8x8 BF16 matmul on the 4x4 chip via the shared host routine."""
    await reset(dut)
    rng = random.Random(0x88)
    A = [rand_bf16(rng) for _ in range(64)]
    B = [rand_bf16(rng) for _ in range(64)]
    assert await matmul(dut, A, B, 8, 8, 8) == model_matmul(A, B, 8, 8, 8)


@cocotb.test()
async def test_tiled_ragged(dut):
    """5x7 @ 7x3 (host pads with +0)."""
    await reset(dut)
    rng = random.Random(0x573)
    A = [rand_bf16(rng) for _ in range(35)]
    B = [rand_bf16(rng) for _ in range(21)]
    assert await matmul(dut, A, B, 5, 7, 3) == model_matmul(A, B, 5, 7, 3)


@cocotb.test()
async def test_busy_done_and_load_while_busy(dut):
    """Protocol behaviour carried over from the int8 chip with 2-byte operands:
    a LOAD_A mid-run (32 data bytes) is swallowed; DONE is sticky."""
    await reset(dut)
    rng = random.Random(3)
    a = [rand_bf16(rng) for _ in range(16)]
    b = [rand_bf16(rng) for _ in range(16)]
    exp, acc1 = expect(a, b)
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, 0x11, b)
    await send_byte(dut, OP_CLEAR_GEMM)
    await send_byte(dut, OP_LOAD_A)
    assert int(dut.uio_out.value) & (1 << BUSY), "BUSY should be high mid-run"
    for _ in range(32):
        await send_byte(dut, 0x7F)
    await wait_idle(dut)
    assert int(dut.uio_out.value) & (1 << DONE)
    assert await read_c(dut) == exp, "LOAD_A during busy corrupted the run"
    await send_byte(dut, OP_GEMM)
    await send_byte(dut, OP_NOP)
    v = int(dut.uio_out.value)
    assert (v & (1 << BUSY)) and not (v & (1 << DONE))
    await wait_idle(dut)
    exp2, _ = expect(a, b, acc1)
    assert await read_c(dut) == exp2, "stored A was overwritten by the swallowed LOAD"


@cocotb.test()
async def test_ena_low_ignores_input(dut):
    await reset(dut)
    dut.ena.value = 0
    await send_byte(dut, OP_ID)
    for _ in range(10):
        await RisingEdge(dut.clk)
        assert int(dut.uio_out.value) & (1 << TX_VALID) == 0
    dut.ena.value = 1
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID
