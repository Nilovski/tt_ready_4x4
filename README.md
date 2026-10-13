![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg)

# 4x4 integer systolic GEMM (Tiny Tapeout, ttsky26d)

A 4x4 output-stationary int8 systolic array with a clear mode
(`C = A x B`) and an accumulate mode (`C += A x B`), behind an 8-bit
byte-serial host protocol. The host tiles matmuls of any size onto it, for
example 8x8 in 4 output blocks x 2 K-blocks.

- Protocol, pinout and test procedure: [docs/info.md](docs/info.md)
- Top module: `tt_um_sahpar_gemm4x4`. Tiles: **4x2**. ID byte: `0xA4`

## Layout

```
src/    RTL: tt_um_sahpar_gemm4x4 -> cmd_decoder + rx/tx_byte_interface -> gemm_top -> systolic_array -> pe -> mac_unit
test/   cocotb tests (test.py) + gemm_protocol.py, the host protocol & tiling code
host/   tt_um_sahpar_gemm4x4.py, a demo-board runner (MicroPython + microcotb) that reuses gemm_protocol.py
```

## What came from where

This design merges the verified 2x2/4x4 bring-up with ideas from an FPGA 8x8
systolic array (PYNQ-Z2, AXI-GPIO driven).

| Feature | Source | Notes |
|---|---|---|
| Counter-scheduled skew, single operand copy, accumulators read in place | ours | FPGA used 32-deep shift registers per row/column (~8k flops at 8x8) and three operand copies |
| Signed int8, overflow-free accumulator | ours | FPGA was unsigned 16-bit with a 16-bit accumulator, so it overflowed silently |
| Real synchronous reset | ours | FPGA relied on `initial` blocks, which don't exist on silicon |
| **Accumulate across runs (`GEMM` 0x20)** | FPGA | Their PE never cleared between runs. Here it is a defined opcode, so the 4x4 chip can compute 8x8 / NxN products by tiling |
| **DONE / interrupt line (`uio[5]`, STATUS bit 1)** | FPGA | Their interrupt pin fired after a fixed cycle count. Ours is set on actual completion and cleared on the next start |
| **Host software layer** | FPGA | Their C driver on the PYNQ ARM core. Ours is `gemm_protocol.py`, used by both the sim tests and the demo-board runner |
| Row-major B (no pre-transpose) | ours | FPGA required software to transpose B |

## Changes from the 2x2 (`Nilovski/tiny_trial` `main`)

| | 2x2 | 4x4 |
|---|---|---|
| Template / CI | ttsky26c, LibreLane 2.4.2 | **ttsky26d**, LibreLane 3.0.14, cocotb 2.1.0 |
| LOAD_A / LOAD_B | 4 bytes | 16 bytes |
| READ_C addresses | 0-15 | 0-63 (still 4 LE bytes/element, signed 32-bit) |
| Opcodes | CLEAR_GEMM only | **GEMM (accumulate)** + CLEAR_GEMM |
| STATUS | `{7'b0, busy}` | `{6'b0, done, busy}` |
| ID | `0xA2` | `0xA4` |
| `uio[4]`, `uio[5]` | unused | **BUSY**, **DONE** outputs |
| Operand storage | decoder copy + `gemm_top` copy | decoder copy only (`gemm_top` `LATCH_OPERANDS=0`) |
| C readout | 64-byte shadow buffer | read straight from the accumulators |
| Accumulator | 32 bit | 21 bit (exact for 63 products), sign-extended to 32 on read |
| LOAD while busy | would corrupt the run | data bytes consumed and discarded |

`systolic_array.sv`, `pe.sv`, `mac_unit.sv`, `rx_byte_interface.sv` and
`tx_byte_interface.sv` are unchanged from the 2x2. `gemm_top.sv` gains the
`LATCH_OPERANDS` parameter (its default keeps the 2x2 behaviour) and an
`accumulate` input.

## Size and timing (yosys 0.33 + sky130_fd_sc_hd tt_025C_1v80, pre-placement)

| | Raw cell area | Flops |
|---|---|---|
| 2x2 on `tiny_trial` (same flow) | 29,846 µm² | 456 |
| 4x4, naive (N=K=4, no trimming) | 119,217 µm² | 1,774 |
| 4x4, clear-only, 19-bit accumulator | 79,962 µm² | 798 |
| **4x4, this design (accumulate + DONE, 21-bit)** | **83,212 µm²** | **834** |

The 4x2 die is 682.64 x 225.76 µm (≈154,000 µm²), so this design sits at
about 54% utilization. The 2x2 hardened cleanly at about 40%. ABC's buffered
critical-path estimate is about 5.4 ns, against the 20 ns `CLOCK_PERIOD`.

On accumulator width: the synthesized area moves non-monotonically, with
about ±8k µm² of noise. 19-21 bits land around 80-83k µm², while 22-24 bits
land around 92-96k µm². At 21 bits the chip can chain 15 runs; the host sums
anything longer.

For reference, the same RTL as a full 8x8 int8 array synthesizes to about
350k µm². That does not fit 8x2 (311k µm²), and would need 8x4 if that slot
is offered.

## Verification

The RTL tests (`cd test && make`) are 15 cocotb tests, all passing. The same
15 also pass on the yosys-synthesized netlist.

- **Base:** ID/uio directions, STATUS idle, counting, identity, extremes
  (+65536 / -65024 / 127²), 6 random back-to-back runs, operand reuse,
  BUSY + LOAD/CLEAR while busy, `ena` low
- **Merged features:**
  - DONE sticky/clear behaviour
  - GEMM accumulate vs CLEAR
  - the 21-bit limit at both extremes (15 runs: +983,040 / -975,360)
  - tiled 8x8
  - ragged 5x7 @ 7x3
  - inner dimension 64, which only passes if the host splits the sum, because
    2^20 overflows 21 bits

Mutation checks — each of these breaks the test that targets it:

| Injected bug | Caught by |
|---|---|
| Accumulate ignored | 3 tests |
| `ACC_W` = 20 | limit test |
| DONE never clears | DONE test |
| Host never splits long K | long-K test |
| (earlier) 17-bit accumulator | extremes test |
| (earlier) no busy-load guard | busy test |

Other checks:

- `verilator --lint-only -Wall` is clean, apart from the PE_I/PE_J
  trace-only parameters inherited from the 2x2.
- `tt_tool.py --create-user-config` and `--check-docs` pass.
- `host/` compiles and pyflakes is clean. It has not been run on a demo board,
  since there's no silicon yet.

## Build / submit

Each push runs GitHub Actions: `test`, then `gds` → `precheck` / `gl_test` /
`viewer`. For `viewer`, set Settings → Pages → Source to **GitHub Actions**.
When everything is green, submit this repo on app.tinytapeout.com for
ttsky26d.
