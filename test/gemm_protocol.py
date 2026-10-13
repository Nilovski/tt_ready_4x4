# SPDX-License-Identifier: Apache-2.0
#
# Host side of the tt_um_sahpar_gemm4x4 byte protocol.
#
# Shared by the cocotb RTL/gate-level tests (test/test.py) and the Tiny Tapeout
# demo board runner (host/tt_um_sahpar_gemm4x4.py, via microcotb), so the
# tiling code that passes in simulation is the code that drives the chip.
#
# All coroutines take a cocotb-style `dut` exposing ui_in, uo_out, uio_in,
# uio_out and clk.

try:
    from cocotb.triggers import RisingEdge
except ImportError:  # TT demo board: microcotb provides the same API
    from microcotb.triggers import RisingEdge

N = 4  # array size (C block is N x N)
K = 4  # inner dimension per run

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
CHIP_ID = 0xA4

# 21-bit accumulators: at most 63 summed products, i.e. 15 runs of K=4
MAX_RUNS_PER_READ = 15


class GemmTimeout(Exception):
    """Chip did not answer (MicroPython has no built-in TimeoutError)."""


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


async def load_matrix(dut, opcode, values):
    await send_byte(dut, opcode)
    for v in values:
        await send_byte(dut, v)


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
    """Read the N*N result block (4 little-endian bytes each), signed 32-bit."""
    out = []
    for elem in range(N * N):
        value = 0
        for byte_i in range(4):
            await send_byte(dut, OP_READ_C_ELEMENT)
            await send_byte(dut, elem * 4 + byte_i)
            value |= (await recv_byte(dut)) << (8 * byte_i)
        if value & (1 << 31):
            value -= 1 << 32
        out.append(value)
    return out


async def gemm4(dut, a, b, accumulate=False):
    """One 4x4x4 run: load A and B (row-major, signed int8), start, wait."""
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, OP_LOAD_B, b)
    await send_byte(dut, OP_GEMM if accumulate else OP_CLEAR_GEMM)
    await wait_idle(dut)


# ---------------------------------------------------------------------------
# arbitrary-size matmul by tiling onto the 4x4 array
# ---------------------------------------------------------------------------
def matmul_ref(a, b, n, k, m):
    """Pure-Python reference: a is n*k row-major, b is k*m -> n*m row-major."""
    return [sum(a[i * k + t] * b[t * m + j] for t in range(k)) for i in range(n) for j in range(m)]


def _block(mat, rows, cols, r0, c0):
    """4x4 block at (r0, c0) of a rows x cols row-major matrix, zero-padded."""
    out = []
    for r in range(r0, r0 + N):
        for c in range(c0, c0 + K):
            out.append(mat[r * cols + c] if (r < rows and c < cols) else 0)
    return out


async def matmul(dut, a, b, n, k, m):
    """C = A x B for any n x k by k x m signed int8 matrices (row-major lists).

    Each 4x4 output block is accumulated on-chip over up to MAX_RUNS_PER_READ
    K-blocks (CLEAR_GEMM then GEMM ...), read out, and summed on the host if
    the inner dimension needs more runs than the accumulator budget allows.
    """
    for v in list(a) + list(b):
        if not -128 <= v <= 127:
            raise ValueError(f"operand {v} is not int8")
    c = [0] * (n * m)
    k_blocks = list(range(0, k, K))
    for bi in range(0, n, N):
        for bj in range(0, m, N):
            acc = [0] * (N * N)
            for g in range(0, len(k_blocks), MAX_RUNS_PER_READ):
                group = k_blocks[g:g + MAX_RUNS_PER_READ]
                for idx, kb in enumerate(group):
                    await gemm4(
                        dut,
                        _block(a, n, k, bi, kb),
                        _block(b, k, m, kb, bj),
                        accumulate=(idx > 0),
                    )
                part = await read_c(dut)
                acc = [x + y for x, y in zip(acc, part)]
            for r in range(N):
                for col in range(N):
                    if bi + r < n and bj + col < m:
                        c[(bi + r) * m + (bj + col)] = acc[r * N + col]
    return c
