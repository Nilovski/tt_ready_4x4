![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg)

# 4x4 integer systolic GEMM (Tiny Tapeout, ttsky26d)

This is a 4x4 output-stationary int8 systolic array that computes C = A x B
behind an 8-bit byte-serial host protocol. It is the second bring-up step,
after the 2x2 on `main`, toward the 8x8 BF16 accelerator.

- Protocol, pinout and test procedure: [docs/info.md](docs/info.md)
- Top module: `tt_um_sahpar_gemm4x4`. Tiles: **4x2**. ID byte: `0xA4`

## Changes from the 2x2 (`main`)

| | 2x2 | 4x4 |
|---|---|---|
| Template / CI | ttsky26c, LibreLane 2.4.2 | **ttsky26d**, LibreLane 3.0.14, cocotb 2.1.0 |
| LOAD_A / LOAD_B | 4 bytes | 16 bytes |
| READ_C addresses | 0-15 | 0-63 (still 4 LE bytes/element, signed 32-bit) |
| ID | `0xA2` | `0xA4` |
| `uio[4]` | unused | **BUSY** output |
| Operand storage | decoder copy + `gemm_top` copy | decoder copy only (`gemm_top` `LATCH_OPERANDS=0`) |
| C readout | 64-byte shadow buffer | read straight from the accumulators |
| Accumulator | 32 bit | 19 bit (exact for int8, K=4), sign-extended to 32 on read |
| LOAD while busy | would corrupt the run | data bytes consumed and discarded |
| Decoder | `cmd_decoder_2x2.sv` | `cmd_decoder.sv` (generic in N/K) |

`systolic_array.sv`, `pe.sv`, `mac_unit.sv`, `rx_byte_interface.sv` and
`tx_byte_interface.sv` are unchanged from the 2x2. `gemm_top.sv` only gains
the `LATCH_OPERANDS` parameter. Its default of 1 keeps the 2x2 behaviour.

## Size and timing (yosys 0.33 + sky130_fd_sc_hd tt_025C_1v80, pre-placement)

| | Raw cell area | Flops |
|---|---|---|
| 2x2 on `main` (same flow) | 29,846 µm² | 456 |
| 4x4, naive (N=K=4, no trimming) | 119,217 µm² | 1,774 |
| **4x4, this design** | **79,962 µm²** | **798** |

The 4x2 die is 682.64 x 225.76 µm (≈154,000 µm²), so this design sits at
about 52% utilization before routing overhead. The 2x2 hardened cleanly at
about 40% in 2x2. ABC's buffered critical-path estimate is about 5.1 ns,
against the 20 ns `CLOCK_PERIOD` in `src/config.json`.

If the GDS run passes with headroom, you could try `tiles: "3x2"` (≈70%)
together with `PL_TARGET_DENSITY_PCT` of 75-80 to save a tile pair. 4x2 is
the safe choice.

## Verification

The RTL tests (`cd test && make`) are 9 cocotb tests, all passing:

- ID and uio direction
- STATUS when idle
- counting matrices
- identity
- extremes (+65536 / -65024 / 127²)
- 6 random signed back-to-back runs
- operand reuse (reload B only)
- BUSY pin, LOAD_A while busy, and CLEAR while busy
- `ena` low ignores input

The same 9 tests also pass on the yosys-synthesized netlist.

Mutation check: dropping the accumulator to 17 bits makes the extremes test
fail. Removing the busy-load guard makes the busy test fail.

`verilator --lint-only -Wall` is clean, apart from the PE_I/PE_J
trace-only parameters, which were already there on the 2x2.
`tt_tool.py --create-user-config` and `--check-docs` pass.

## Submit

```bash
git checkout gemm-4x4
git rm -q src/tt_um_sahpar_gemm2x2.sv src/cmd_decoder_2x2.sv test/test_gemm2x2.py
# unzip this folder's contents over the branch (keeps .git), then:
git add -A && git commit -m "4x4 GEMM on ttsky26d template" && git push
```

GitHub Actions then runs `test`, followed by `gds` → `precheck` / `gl_test` /
`viewer`. When everything is green, submit the repo + branch on
app.tinytapeout.com for the ttsky26d shuttle.
