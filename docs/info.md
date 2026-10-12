## How it works

A 4x4 output-stationary integer systolic array computes **C = A x B**, where A
(4x4) and B (4x4) hold signed 8-bit values. A flows in from the west edge and B
from the north edge. Row *i* and column *j* are delayed by *i* and *j* cycles,
so each processing element PE(i,j) receives A[i][k] and B[k][j] on the same
cycle and accumulates C[i][j] in place. A run takes K + 2(N-1) = 10 clock
cycles.

The design is the second step of a bring-up toward an 8x8 BF16 accelerator.
It comes after a 2x2 version and uses the same host protocol.

Two synchronous handshake channels carry bytes. `RX_VALID` + `ui[7:0]` bring
a byte in (`RX_READY` is always high), and `TX_VALID` + `uo[7:0]` send one
out, held until the host pulses `TX_READY`.

| Byte   | Op             | Followed by                         | Reply             |
|--------|----------------|-------------------------------------|-------------------|
| `0x00` | NOP            | --                                  | none              |
| `0x10` | LOAD_A         | 16 bytes, row-major a00..a33        | none              |
| `0x11` | LOAD_B         | 16 bytes, row-major b00..b33        | none              |
| `0x21` | CLEAR_GEMM     | --                                  | none (C = A x B)  |
| `0x31` | READ_C_ELEMENT | 1 address byte, 0-63                | 1 byte of C       |
| `0x40` | STATUS         | --                                  | `{7'b0, busy}`    |
| `0x41` | ID             | --                                  | `0xA4`            |

- C is read as 16 elements x 4 bytes: signed 32-bit, little-endian. The
  address is `elem*4 + byte`, and `elem` goes in row-major order (c00, c01,
  ... c33). Internally the accumulators are 19 bits, which is exact for any
  int8 input, and they are sign-extended to 32 bits on readout.
- A and B are held between runs, so you can reload just one of them.
- CLEAR_GEMM is ignored while busy. A LOAD sent while busy has its data bytes
  consumed and discarded, so the stored operands stay intact and the byte
  stream stays in sync.
- C is read straight from the accumulators. Wait for busy = 0 first, either
  by polling STATUS or by watching the `BUSY` pin on `uio[4]`.

## How to test

1. Reset, then send `0x41` (ID) and expect `0xA4`.
2. Send `0x10` and then 16 bytes of A. Send `0x11` and then 16 bytes of B.
   Negative values are two's complement.
3. Send `0x21` to start.
4. Poll `0x40` until bit 0 reads 0, or wait for `uio[4]` (BUSY) to go low.
5. For each address 0-63, send `0x31` and then the address, and collect the
   reply byte. Reassemble the 16 little-endian 32-bit values.

Example: A = [[1..4],[5..8],[9..12],[13..16]], B = A reversed (16..1).
The full cocotb test in `test/test.py` checks against a Python reference. It
covers identity, extreme values (+65536 / -65024), random signed matrices,
operand reuse and LOAD-while-busy.

From a microcontroller (e.g. the TT demo board RP2040), drive `uio[0]` and
`uio[2]` as outputs and read `uio[1]`, `uio[3]` and `uio[4]`. Keep each
RX_VALID / TX_READY pulse at least one clock high.

## External hardware

None. All you need is a host (the demo board's RP2040, or any MCU or FPGA)
driving the byte protocol on `ui`/`uo`/`uio[4:0]`.
