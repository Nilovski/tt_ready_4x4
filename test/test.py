# SPDX-License-Identifier: Apache-2.0
#
# cocotb tests for tt_um_sahpar_gemm4x4 -- 4x4 integer output-stationary
# systolic GEMM behind the 8-bit host byte protocol. Every result is checked
# against a Python reference multiply.

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

N = 4
K = 4

# uio bit positions
RX_VALID = 0
RX_READY = 1
TX_READY = 2
TX_VALID = 3
BUSY = 4

OP_NOP = 0x00
OP_LOAD_A = 0x10
OP_LOAD_B = 0x11
OP_CLEAR_GEMM = 0x21
OP_READ_C_ELEMENT = 0x31
OP_STATUS = 0x40
OP_ID = 0x41
CHIP_ID = 0xA4


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
    raise TimeoutError("timed out waiting for TX_VALID")


async def load_matrix(dut, opcode, values):
    await send_byte(dut, opcode)
    for v in values:
        await send_byte(dut, v)


async def wait_idle(dut, max_polls=40):
    for _ in range(max_polls):
        await send_byte(dut, OP_STATUS)
        if (await recv_byte(dut)) & 0x1 == 0:
            return
    raise TimeoutError("GEMM never finished (busy stuck high)")


async def read_c(dut):
    """Read all N*N elements (4 little-endian bytes each), signed 32-bit."""
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


def matmul(a, b):
    """a: N*K row-major, b: K*N row-major -> N*N row-major."""
    return [
        sum(a[i * K + k] * b[k * N + j] for k in range(K))
        for i in range(N)
        for j in range(N)
    ]


def fmt(m):
    return "\n".join(str(m[r * N:(r + 1) * N]) for r in range(N))


async def run_and_check(dut, a, b, label=""):
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, OP_LOAD_B, b)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    got = await read_c(dut)
    exp = matmul(a, b)
    assert got == exp, f"{label} mismatch\nA=\n{fmt(a)}\nB=\n{fmt(b)}\ngot=\n{fmt(got)}\nexp=\n{fmt(exp)}"
    return got


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_id(dut):
    """ID replies 0xA4 and the static uio direction bits are right."""
    await reset(dut)
    assert int(dut.uio_oe.value) == 0b0001_1010, f"uio_oe={int(dut.uio_oe.value):#010b}"
    assert int(dut.uio_out.value) & (1 << RX_READY), "RX_READY should be high"
    await send_byte(dut, OP_ID)
    byte = await recv_byte(dut)
    assert byte == CHIP_ID, f"expected ID {CHIP_ID:#x}, got {byte:#x}"


@cocotb.test()
async def test_status_idle(dut):
    await reset(dut)
    await send_byte(dut, OP_NOP)
    await send_byte(dut, OP_STATUS)
    assert (await recv_byte(dut)) & 0x1 == 0
    assert int(dut.uio_out.value) & (1 << BUSY) == 0


@cocotb.test()
async def test_gemm_counting(dut):
    """Small positive values, easy to eyeball in a waveform."""
    await reset(dut)
    a = list(range(1, 17))
    b = list(range(16, 0, -1))
    await run_and_check(dut, a, b, "counting")


@cocotb.test()
async def test_gemm_identity(dut):
    """A = I  ->  C = B (checks the skew lines up every row/column)."""
    await reset(dut)
    ident = [1 if i == j else 0 for i in range(N) for j in range(K)]
    b = [(-1) ** k * (k * 7 + 3) for k in range(K * N)]
    got = await run_and_check(dut, ident, b, "identity")
    assert got == b


@cocotb.test()
async def test_gemm_extremes(dut):
    """Corner magnitudes: accumulator must hold +65536 and -65024 exactly."""
    await reset(dut)
    await run_and_check(dut, [-128] * 16, [-128] * 16, "max-positive")
    await run_and_check(dut, [-128] * 16, [127] * 16, "max-negative")
    await run_and_check(dut, [127] * 16, [127] * 16, "127^2")


@cocotb.test()
async def test_gemm_random(dut):
    """Random signed operands, several back-to-back runs without reset."""
    await reset(dut)
    rng = random.Random(0x5A4)
    for n in range(6):
        a = [rng.randint(-128, 127) for _ in range(N * K)]
        b = [rng.randint(-128, 127) for _ in range(K * N)]
        await run_and_check(dut, a, b, f"random run {n}")


@cocotb.test()
async def test_reuse_operands(dut):
    """Only reload B; A must persist from the previous LOAD_A."""
    await reset(dut)
    a = [3, -1, 4, 1, -5, 9, 2, -6, 5, 3, -5, 8, 9, -7, 9, 3]
    b1 = [2, 7, -1, 8, 2, -8, 1, 8, -2, 8, 4, 5, 9, 0, -4, 5]
    b2 = [v * -2 for v in b1]
    await run_and_check(dut, a, b1, "first")
    await load_matrix(dut, OP_LOAD_B, b2)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert await read_c(dut) == matmul(a, b2)


@cocotb.test()
async def test_busy_and_load_while_busy(dut):
    """BUSY pin rises during a run; a LOAD_A sent mid-run is swallowed whole
    (operands stay intact) and a CLEAR_GEMM mid-run is ignored."""
    await reset(dut)
    a = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]
    b = [2, 0, 1, -1, 0, 3, 1, 1, 1, 1, -2, 0, -1, 2, 0, 4]
    exp = matmul(a, b)
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, OP_LOAD_B, b)

    await send_byte(dut, OP_CLEAR_GEMM)
    # junk LOAD_A while busy: all 16 data bytes must be consumed and dropped
    await send_byte(dut, OP_LOAD_A)
    assert int(dut.uio_out.value) & (1 << BUSY), "BUSY pin should be high mid-run"
    for _ in range(16):
        await send_byte(dut, 0x7F)
    await wait_idle(dut)
    assert await read_c(dut) == exp, "LOAD_A during busy corrupted the result"

    # the stored A must still be the original: rerun and compare
    await send_byte(dut, OP_CLEAR_GEMM)
    await send_byte(dut, OP_CLEAR_GEMM)          # second one lands while busy -> ignored
    await wait_idle(dut)
    assert await read_c(dut) == exp, "LOAD_A during busy overwrote stored A"

    # protocol is still in sync: ID works
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID


@cocotb.test()
async def test_ena_low_ignores_input(dut):
    """While ena is low the decoder ignores the host byte stream."""
    await reset(dut)
    dut.ena.value = 0
    await send_byte(dut, OP_ID)
    for _ in range(10):
        await RisingEdge(dut.clk)
        assert int(dut.uio_out.value) & (1 << TX_VALID) == 0, "replied while ena=0"
    dut.ena.value = 1
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID
