# SPDX-License-Identifier: Apache-2.0
#
# Host side of the tt_um_sahpar_gemm4x4_bf16 byte protocol.
#
# Shared by the cocotb RTL/gate-level tests (test/test.py) and the Tiny Tapeout
# demo board runner (host/tt_um_sahpar_gemm4x4_bf16.py, via microcotb), so the
# code that passes in simulation is the code that drives the chip. Uses only
# `struct` beyond the trigger import, so it runs under MicroPython too.
#
# All coroutines take a cocotb-style `dut` exposing ui_in, uo_out, uio_in,
# uio_out and clk.

import struct

try:
    from cocotb.triggers import RisingEdge
except ImportError:  # TT demo board: microcotb provides the same API
    from microcotb.triggers import RisingEdge

N = 4  # array size (C block is N x N)
K = 4  # inner dimension per run
ELEM_BYTES = 2  # BF16 operands, low byte first

# uio bit positions
RX_VALID = 0
RX_READY = 1
TX_READY = 2
TX_VALID = 3
BUSY = 4
DONE = 5

# the RP2/host drives these uio bits (everything else is an input to the host)
HOST_UIO_OE = (1 << RX_VALID) | (1 << TX_READY)

OP_NOP = 0x00
OP_LOAD_A = 0x10
OP_LOAD_B = 0x11
OP_GEMM = 0x20  # C += A x B
OP_CLEAR_GEMM = 0x21  # C  = A x B
OP_READ_C_ELEMENT = 0x31
OP_STATUS = 0x40
OP_ID = 0x41
CHIP_ID = 0xB4


class GemmTimeout(Exception):
    """Chip did not answer (MicroPython has no built-in TimeoutError)."""


# ---------------------------------------------------------------------------
# BF16 / FP32 conversions
# ---------------------------------------------------------------------------
def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def to_bf16(x):
    """Python float -> BF16 bit pattern, round to nearest even."""
    u = f32_bits(x)
    if (u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF):
        return 0x7FC0  # canonical quiet NaN
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return u >> 16


def bf16_to_float(h):
    return bits_f32(h << 16)


# ---------------------------------------------------------------------------
# byte-level protocol
# ---------------------------------------------------------------------------
def _set_uio_bit(dut, bit, val):
    cur = int(dut.uio_in.value)
    dut.uio_in.value = (cur | (1 << bit)) if val else (cur & ~(1 << bit))


async def send_byte(dut, value):
    """One byte on ui_in with a one-cycle RX_VALID pulse (RX_READY is tied high)."""
    dut.ui_in.value = value & 0xFF
    _set_uio_bit(dut, RX_VALID, 1)
    await RisingEdge(dut.clk)
    _set_uio_bit(dut, RX_VALID, 0)
    await RisingEdge(dut.clk)


async def recv_byte(dut, timeout_cycles=50):
    """Wait for TX_VALID, capture uo_out, pulse TX_READY to accept it."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.uio_out.value) & (1 << TX_VALID):
            byte = int(dut.uo_out.value)
            _set_uio_bit(dut, TX_READY, 1)
            await RisingEdge(dut.clk)
            _set_uio_bit(dut, TX_READY, 0)
            return byte
    raise GemmTimeout("timed out waiting for TX_VALID")


async def load_matrix(dut, opcode, elems):
    """LOAD_A / LOAD_B with 16 BF16 bit patterns (row-major), low byte first."""
    await send_byte(dut, opcode)
    for e in elems:
        for i in range(ELEM_BYTES):
            await send_byte(dut, (e >> (8 * i)) & 0xFF)


async def read_id(dut):
    await send_byte(dut, OP_ID)
    return await recv_byte(dut)


async def status(dut):
    """Returns (busy, done)."""
    await send_byte(dut, OP_STATUS)
    s = await recv_byte(dut)
    return bool(s & 1), bool(s & 2)


async def wait_idle(dut, max_polls=40):
    for _ in range(max_polls):
        busy, _ = await status(dut)
        if not busy:
            return
    raise GemmTimeout("GEMM never finished (busy stuck high)")


async def read_c(dut):
    """Read the N*N result block as raw FP32 bit patterns (4 LE bytes each)."""
    out = []
    for elem in range(N * N):
        value = 0
        for byte_i in range(4):
            await send_byte(dut, OP_READ_C_ELEMENT)
            await send_byte(dut, elem * 4 + byte_i)
            value |= (await recv_byte(dut)) << (8 * byte_i)
        out.append(value)
    return out


async def gemm4(dut, a, b, accumulate=False):
    """One 4x4x4 run on BF16 bit patterns (row-major): load, start, wait."""
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, OP_LOAD_B, b)
    await send_byte(dut, OP_GEMM if accumulate else OP_CLEAR_GEMM)
    await wait_idle(dut)


# ---------------------------------------------------------------------------
# arbitrary-size matmul by tiling onto the 4x4 array
# ---------------------------------------------------------------------------
def _block(mat, rows, cols, r0, c0):
    """4x4 block at (r0, c0) of a rows x cols row-major matrix, +0-padded."""
    out = []
    for r in range(r0, r0 + N):
        for c in range(c0, c0 + K):
            out.append(mat[r * cols + c] if (r < rows and c < cols) else 0)
    return out


async def matmul(dut, a, b, n, k, m):
    """C = A x B for n x k by k x m matrices of BF16 bit patterns (row-major).

    Each 4x4 output block is one CLEAR_GEMM plus a GEMM per further K-block,
    all accumulated on-chip (floats don't overflow the way the int8 chip's
    21-bit accumulators do). Returns FP32 bit patterns, n*m row-major.
    """
    c = [0] * (n * m)
    for bi in range(0, n, N):
        for bj in range(0, m, N):
            for idx, kb in enumerate(range(0, k, K)):
                await gemm4(dut, _block(a, n, k, bi, kb), _block(b, k, m, kb, bj),
                            accumulate=(idx > 0))
            blk = await read_c(dut)
            for r in range(N):
                for col in range(N):
                    if bi + r < n and bj + col < m:
                        c[(bi + r) * m + (bj + col)] = blk[r * N + col]
    return c


async def matmul_float(dut, a, b, n, k, m):
    """Convenience: Python floats in (rounded to BF16), Python floats out."""
    c = await matmul(dut, [to_bf16(x) for x in a], [to_bf16(x) for x in b], n, k, m)
    return [bits_f32(u) for u in c]
