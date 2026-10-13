## How it works

A 4x4 output-stationary integer systolic array computes **C = A x B** or
**C += A x B**, where A (4x4) and B (4x4) hold signed 8-bit values. A flows in
from the west edge and B from the north edge. Row *i* and column *j* are
delayed by *i* and *j* cycles, so each processing element PE(i,j) receives
A[i][k] and B[k][j] on the same cycle and accumulates C[i][j] in place. A run
takes K + 2(N-1) = 10 clock cycles.

Because results can accumulate across runs, a host can compute a matrix
product of **any size** on this 4x4 array. It splits the operands into 4x4
blocks and sums the K-blocks for each output block on-chip.

Two synchronous handshake channels carry bytes. `RX_VALID` + `ui[7:0]` bring
a byte in (`RX_READY` is always high), and `TX_VALID` + `uo[7:0]` send one
out, held until the host pulses `TX_READY`.

| Byte   | Op             | Followed by                         | Reply                   |
|--------|----------------|-------------------------------------|-------------------------|
| `0x00` | NOP            | --                                  | none                    |
| `0x10` | LOAD_A         | 16 bytes, row-major a00..a33        | none                    |
| `0x11` | LOAD_B         | 16 bytes, row-major b00..b33        | none                    |
| `0x20` | GEMM           | --                                  | none (C += A x B)       |
| `0x21` | CLEAR_GEMM     | --                                  | none (C = A x B)        |
| `0x31` | READ_C_ELEMENT | 1 address byte, 0-63                | 1 byte of C             |
| `0x40` | STATUS         | --                                  | `{6'b0, done, busy}`    |
| `0x41` | ID             | --                                  | `0xA4`                  |

- C is read as 16 elements x 4 bytes: signed 32-bit, little-endian. The
  address is `elem*4 + byte`, and `elem` goes in row-major order (c00, c01,
  ... c33).
- The accumulators are 21 bits, sign-extended to 32 on readout. They hold any
  sum of up to **63 products** exactly. That means one CLEAR_GEMM plus up to
  14 GEMMs (inner dimension up to 60) before C must be read out. For longer
  inner dimensions the host adds the partial results itself.
- A and B are held between runs, so you can reload just one of them.
- GEMM and CLEAR_GEMM are ignored while busy. A LOAD sent while busy has its
  data bytes consumed and discarded, so the stored operands stay intact and
  the byte stream stays in sync.
- **BUSY** (`uio[4]`) is high while a run is in progress.
- **DONE** (`uio[5]`) goes high when a run finishes and stays high until the
  next GEMM or CLEAR_GEMM. It works as an interrupt line. STATUS reports both
  flags.

## How to test

1. Reset, then send `0x41` (ID) and expect `0xA4`.
2. Send `0x10` and then 16 bytes of A. Send `0x11` and then 16 bytes of B.
   Negative values are two's complement.
3. Send `0x21` (CLEAR_GEMM) to start.
4. Wait for `uio[5]` DONE, or poll `0x40` until bit 0 (busy) reads 0.
5. For each address 0-63, send `0x31` and then the address, and collect the
   reply byte. Reassemble the 16 little-endian 32-bit values.

To accumulate, load new A/B blocks and send `0x20` instead of `0x21`.

**On the Tiny Tapeout demo board**, copy `test/gemm_protocol.py` and
`host/tt_um_sahpar_gemm4x4.py` to the RP2 with `mpremote`. Then, in the REPL:

```python
import tt_um_sahpar_gemm4x4 as g
g.main()        # self-tests on silicon (ID, identity, random, accumulate, tiled 8x8)
g.demo_8x8()    # prints an 8x8 int8 matmul computed on the 4x4 chip
```

These scripts use the same protocol and tiling code as the cocotb simulation
tests in `test/test.py`.

## External hardware

None. All you need is a host (the demo board's RP2040/RP2350, or any MCU or
FPGA) driving `ui`/`uo`/`uio[5:0]`.
