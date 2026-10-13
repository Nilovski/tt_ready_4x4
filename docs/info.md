## How it works

A 4x4 output-stationary systolic array computes **C = A x B** or
**C += A x B** on **BF16** matrices: 1 sign bit, 8 exponent bits and 7
mantissa bits, the bfloat16 format used in ML accelerators. A flows in from
the west edge and B from the north edge. Row *i* and column *j* are delayed by
*i* and *j* cycles, so each processing element PE(i,j) receives A[i][k] and
B[k][j] on the same cycle. A run takes K + 2(N-1) = 10 clock cycles.

Each PE holds a BF16 multiplier (8x8-bit mantissa product, 9-bit exponent sum)
followed by a BF16 adder. The adder handles alignment shifting,
leading-zero normalization and rounding. Both round to nearest-even and flush
subnormals to zero. A dot product therefore accumulates as
`acc = round(acc + round(a*b))`. Infinities and NaNs propagate (inf - inf and
inf x 0 give NaN), and overflow saturates to infinity.

Because results can accumulate across runs, a host can compute a BF16 matrix
product of **any size** on the 4x4 array by sending it as 4x4 blocks.

Two synchronous handshake channels carry bytes. `RX_VALID` + `ui[7:0]` bring
a byte in (`RX_READY` is always high), and `TX_VALID` + `uo[7:0]` send one
out, held until the host pulses `TX_READY`.

| Byte   | Op             | Followed by                                    | Reply                |
|--------|----------------|------------------------------------------------|----------------------|
| `0x00` | NOP            | --                                             | none                 |
| `0x10` | LOAD_A         | 16 BF16 values, row-major, low byte first (32 bytes) | none           |
| `0x11` | LOAD_B         | 16 BF16 values, row-major, low byte first (32 bytes) | none           |
| `0x20` | GEMM           | --                                             | none (C += A x B)    |
| `0x21` | CLEAR_GEMM     | --                                             | none (C = A x B)     |
| `0x31` | READ_C_ELEMENT | 1 address byte, 0-63                           | 1 byte of C          |
| `0x40` | STATUS         | --                                             | `{6'b0, done, busy}` |
| `0x41` | ID             | --                                             | `0xB4`               |

- C is read as 16 elements x 4 bytes. Each element is an **FP32** value,
  little-endian. The address is `elem*4 + byte`, and `elem` goes in row-major
  order. The BF16 accumulator is returned exactly, widened to FP32
  (`{bf16, 16'h0000}`).
- A and B are held between runs. GEMM and CLEAR_GEMM are ignored while busy.
  A LOAD sent while busy is consumed and discarded.
- **BUSY** (`uio[4]`) is high while a run is in progress.
- **DONE** (`uio[5]`) goes high when a run finishes and stays high until the
  next GEMM or CLEAR_GEMM.

## How to test

1. Reset, then send `0x41` (ID) and expect `0xB4`.
2. Send `0x10` followed by A as 16 BF16 values, low byte then high byte each.
   For example, 1.0 = `0x3F80`, sent as `0x80 0x3F`. Do the same for B with
   `0x11`.
3. Send `0x21` (CLEAR_GEMM). Wait for `uio[5]` DONE, or poll `0x40`.
4. Read addresses 0-63 with `0x31` followed by the address. Every 4 bytes are
   one FP32 result, little-endian.

**On the Tiny Tapeout demo board**, copy `test/gemm_protocol.py` and
`host/tt_um_sahpar_gemm4x4_bf16.py` to the RP2 with `mpremote`. Then, in the
REPL:

```python
import tt_um_sahpar_gemm4x4_bf16 as g
g.main()        # self-tests on silicon
g.demo_8x8()    # an 8x8 BF16 matmul on the 4x4 chip, printed and checked
```

## External hardware

None. All you need is a host (the demo board's RP2040/RP2350, or any MCU or
FPGA) driving `ui`/`uo`/`uio[5:0]`.
