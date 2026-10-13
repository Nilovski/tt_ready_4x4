# SPDX-License-Identifier: Apache-2.0
#
# Hardware-in-the-loop runner for tt_um_sahpar_gemm4x4_bf16 on the Tiny
# Tapeout demo board (RP2040/RP2350 + tt-micropython-firmware, which ships
# microcotb).
#
# It reuses ../test/gemm_protocol.py -- the exact host code the cocotb
# simulation tests exercise -- so what passes in sim is what drives the chip.
# (The bit-exact model, test/fp_model.py, needs `fractions`, which MicroPython
# lacks; on the board results are checked against float math with a tolerance.)
#
# Install (from the repo root, board on USB):
#     mpremote cp test/gemm_protocol.py :
#     mpremote cp host/tt_um_sahpar_gemm4x4_bf16.py :
# Run on the board's REPL:
#     >>> import tt_um_sahpar_gemm4x4_bf16 as g
#     >>> g.main()          # self-test on silicon
#     >>> g.demo_8x8()      # 8x8 BF16 matmul on the 4x4 chip, printed + checked
#
# Pins: the RP2 drives ui_in[7:0], uio[0] RX_VALID and uio[2] TX_READY; it
# reads uo_out[7:0], uio[1] RX_READY, uio[3] TX_VALID, uio[4] BUSY, uio[5] DONE.

import random

import microcotb as cocotb
from microcotb.clock import Clock
from microcotb.triggers import ClockCycles

cocotb.set_runner_scope(__name__)

from ttboard.demoboard import DemoBoard  # noqa: E402
from ttboard.mode import RPMode  # noqa: E402

import gemm_protocol as gp  # noqa: E402

PROJECT = "tt_um_sahpar_gemm4x4_bf16"
CLOCK_PERIOD_US = 10  # bit-banged project clock; the protocol is clock-agnostic
REL_TOL = 2.0 ** -6   # BF16 accumulator: ~8 significant bits per running sum


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_US, units="us").start())
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def _rand_floats(count, lo=-4.0, hi=4.0):
    return [lo + (hi - lo) * random.random() for _ in range(count)]


def _ref(a, b, n, k, m):
    """float reference on the BF16-rounded inputs."""
    ar = [gp.bf16_to_float(gp.to_bf16(x)) for x in a]
    br = [gp.bf16_to_float(gp.to_bf16(x)) for x in b]
    out, mag = [], []
    for i in range(n):
        for j in range(m):
            terms = [ar[i * k + t] * br[t * m + j] for t in range(k)]
            out.append(sum(terms))
            mag.append(sum(abs(x) for x in terms))
    return out, mag


def _close(got, a, b, n, k, m):
    exp, mag = _ref(a, b, n, k, m)
    for g, e, s in zip(got, exp, mag):
        if abs(g - e) > 4 * REL_TOL * s + 1e-30:
            return False
    return True


@cocotb.test()
async def test_id(dut):
    await reset(dut)
    got = await gp.read_id(dut)
    assert got == gp.CHIP_ID, "ID {:#x} != {:#x}".format(got, gp.CHIP_ID)


@cocotb.test()
async def test_exact_integers(dut):
    """Small integers are exact in BF16: C must equal the integer product."""
    await reset(dut)
    a = [float(((i * 3) % 7) - 3) for i in range(16)]
    b = [float(((i * 5) % 9) - 4) for i in range(16)]
    got = await gp.matmul_float(dut, a, b, 4, 4, 4)
    exp, _ = _ref(a, b, 4, 4, 4)
    assert got == exp, "integer matmul mismatch"


@cocotb.test()
async def test_random_4x4(dut):
    await reset(dut)
    random.seed(1)
    for _ in range(3):
        a, b = _rand_floats(16), _rand_floats(16)
        assert _close(await gp.matmul_float(dut, a, b, 4, 4, 4), a, b, 4, 4, 4)


@cocotb.test()
async def test_specials(dut):
    """inf propagates; inf - inf gives NaN."""
    await reset(dut)
    one = gp.to_bf16(1.0)
    await gp.gemm4(dut, [0x7F80, 0xFF80, 0, 0] + [one] * 12, [one] * 16)
    c = await gp.read_c(dut)
    assert c[0] == 0x7FC00000, "inf + -inf should be NaN, got {:08x}".format(c[0])
    await gp.gemm4(dut, [0x7F80] + [one] * 15, [one] * 16)
    c = await gp.read_c(dut)
    assert c[0] == 0x7F800000, "expected +inf, got {:08x}".format(c[0])


@cocotb.test()
async def test_tiled_8x8(dut):
    await reset(dut)
    random.seed(3)
    a, b = _rand_floats(64), _rand_floats(64)
    assert _close(await gp.matmul_float(dut, a, b, 8, 8, 8), a, b, 8, 8, 8)


# ---------------------------------------------------------------------------
# demo: registered once, in its own runner scope so main() doesn't run it
DEMO_SCOPE = PROJECT + "_demo"
_demo = {}

cocotb.set_runner_scope(DEMO_SCOPE)


@cocotb.test()
async def demo_matmul(dut):
    await reset(dut)
    _demo["c"] = await gp.matmul_float(dut, _demo["a"], _demo["b"], 8, 8, 8)


cocotb.set_runner_scope(__name__)


# ---------------------------------------------------------------------------
def _setup():
    tt = DemoBoard.get()
    getattr(tt.shuttle, PROJECT).enable()
    if tt.mode != RPMode.ASIC_RP_CONTROL:
        tt.mode = RPMode.ASIC_RP_CONTROL
    tt.uio_oe_pico.value = gp.HOST_UIO_OE  # RX_VALID, TX_READY driven by us
    return tt


def _make_dut():
    import ttboard.cocotb.dut

    class DUT(ttboard.cocotb.dut.DUT):
        def __init__(self):
            super().__init__("gemm4x4_bf16")

    return DUT()


def main():
    """Run the self-tests above against the chip."""
    _setup()
    runner = cocotb.get_runner(__name__)
    runner.test(_make_dut())
    return runner


def demo_8x8(seed=42):
    """One 8x8 BF16 matmul on silicon, printed and checked against floats."""
    _setup()
    random.seed(seed)
    _demo.clear()
    _demo["a"], _demo["b"] = _rand_floats(64), _rand_floats(64)
    cocotb.get_runner(DEMO_SCOPE).test(_make_dut())
    c = _demo.get("c")
    if c is None:
        print("demo did not complete")
        return None
    for r in range(8):
        print(" ".join("{:9.4f}".format(v) for v in c[r * 8:(r + 1) * 8]))
    print("within BF16 tolerance:", _close(c, _demo["a"], _demo["b"], 8, 8, 8))
    return c
