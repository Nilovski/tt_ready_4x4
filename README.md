![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg)

# 4x4 BF16 systolic GEMM (Tiny Tapeout, ttsky26d) — `bf16` branch

This branch is the floating-point version of the int8 chip on `main`. It uses
the same array, scheduler, decoder, host protocol, accumulate mode and
BUSY/DONE pins, with BF16 arithmetic in every PE.

- Protocol, pinout and test procedure: [docs/info.md](docs/info.md)
- Top module: `tt_um_sahpar_gemm4x4_bf16`. Tiles: **6x2**. ID byte: `0xB4`
- `main` (int8, 4x2, ID `0xA4`) is untouched

## What changed vs. `main`

| File | Change |
|---|---|
| `src/bf16_mul.sv` | **new.** BF16 x BF16 → BF16 or FP32 (8x8 mantissa product, exponent add, RNE) |
| `src/fp_add.sv` | **new.** Float adder (align + sticky, add/sub, LZC normalize, RNE), BF16 or FP32 |
| `src/mac_unit.sv` | Integer MAC replaced by `bf16_mul` → `fp_add`. **Same ports and contract**, so no other array file changed |
| `src/cmd_decoder.sv` | Multi-byte operands (low byte first), `C_FLOAT` readout (accumulator left-aligned to FP32) |
| `src/tt_um_sahpar_gemm4x4_bf16.sv` | `DATA_W=16`, `ACC_W=16`, ID `0xB4` |
| `src/config.json` | `CLOCK_PERIOD` 20 → 40 ns (the chip runs at 10 MHz; the FP datapath is ~15 ns of logic) |

`gemm_top.sv`, `systolic_array.sv`, `pe.sv`, `rx_byte_interface.sv` and
`tx_byte_interface.sv` are identical to `main`. The `mac_unit` swap point
worked as designed.

### Number handling

The Python model in `test/fp_model.py` mirrors these rules exactly.

- **Rounding:** round to nearest, ties to even, at the multiply and at the add.
- **Subnormals:** flushed to zero on input and output.
- **Specials:**
  - NaN in → canonical quiet NaN out
  - inf x 0 → NaN; inf − inf → NaN
  - overflow → ±inf
- **Zeros:** exact cancellation → +0, and x + 0 = x.

## Size and timing (yosys 0.33 + sky130_fd_sc_hd tt_025C_1v80, pre-placement)

| | One MAC | Full design | Flops | Logic delay | Fits |
|---|---|---|---|---|---|
| int8 (`main`) | 4.0k µm² | 83k µm² | 834 | ~5 ns | 4x2 @ 54% |
| **BF16 × BF16, BF16 accumulator (this build)** | 6.9k µm² | **142k µm²** | 1,204 | ~15 ns | **6x2 @ 61%** |
| BF16 × BF16, FP32 accumulator (`ACC_W = 32`) | 11.8k µm² | 221k µm² | 1,460 | ~18 ns | 8x2 @ 71% (tight) |

The project README specifies BF16 accumulators. The `mac_bf16_stub` comment
suggested FP32. FP32 gives much more accurate running sums (24 vs 8
significant bits) but costs 8x2 instead of 6x2. Switching is one line in the
top-level module plus the `tiles` value in `info.yaml`, and the whole test
suite runs on both (`ACC_W=32 make`).

## Verification

**Unit, bit-exact** (`python3 test/fp_unit/run.py`, also run in CI): 459,168
vectors with 0 mismatches.

- `bf16_mul` → BF16 / FP32: ~98k vectors each, including all 16k
  mantissa pairs.
- `fp_add` BF16 / FP32: ~132k vectors each, including near-total
  cancellation, close exponents and overflow/underflow edges.
- Every special pair (±0, subnormals, min/max normal, ±inf, NaNs) is checked
  against an exact rational-arithmetic reference.

Mutation checks — each injected bug is caught:

| Injected bug | Mismatches |
|---|---|
| Round-half-up instead of even | 349 |
| No sticky bit | 249 |
| inf x 0 ≠ NaN | 48 |

**System, cocotb** (`cd test && make`): 11 tests, all passing. They are
checked bit-exactly against a cycle-accurate model of the array, which
reproduces the skew, the +0 ramp/drain cycles and the accumulation order.

- exact-integer matmul
- identity
- random values across 2^±20
- specials through the array
- accumulate chain
- tiled 8x8
- ragged 5x7 @ 7x3
- BUSY/DONE + LOAD while busy
- `ena` low

These pass on:

- the RTL with a BF16 accumulator
- the RTL with an FP32 accumulator
- the yosys-synthesized netlist

Swapping operand byte order fails 8 of the 11 tests.

`verilator --lint-only -Wall` is clean, apart from the PE_I/PE_J trace-only
parameters inherited from `main`.

## Host

`test/gemm_protocol.py` handles BF16 conversion, the byte protocol and tiling.
`gemm_protocol.matmul_float(dut, A, B, n, k, m)` takes Python floats and
returns floats. `host/tt_um_sahpar_gemm4x4_bf16.py` runs the same code on the
Tiny Tapeout demo board through microcotb (not run on hardware yet).
