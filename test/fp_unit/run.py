#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Bit-exact check of bf16_mul / fp_add (both widths) against test/fp_model.py.
# Usage:  python3 test/fp_unit/run.py [vectors_per_class]
# Needs iverilog. Exits non-zero on any mismatch.

import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fp_model as fm  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")


def specials(m):
    P = lambda s, e, f: fm.pack(s, e, f, m)  # noqa: E731
    top = (1 << m) - 1
    out = []
    for s in (0, 1):
        out += [
            P(s, 0, 0), P(s, 0, 1), P(s, 0, top),            # zero, subnormals
            P(s, 1, 0), P(s, 1, top), P(s, 2, 0),            # smallest normals
            P(s, 127, 0), P(s, 127, 1), P(s, 127, top),      # around 1.0
            P(s, 126, top), P(s, 128, 0), P(s, 63, 5), P(s, 191, 9),
            P(s, 253, top), P(s, 254, 0), P(s, 254, top),    # largest normals
            P(s, 255, 0),                                    # inf
            P(s, 255, 1), P(s, 255, 1 << (m - 1)), P(s, 255, top),  # NaNs
        ]
    return out


def rand_normal(rng, m, elo=1, ehi=254):
    return fm.pack(rng.getrandbits(1), rng.randint(elo, ehi), rng.getrandbits(m), m)


def mul_vectors(rng, n):
    S = specials(fm.BF16_M)
    v = [(a, b) for a in S for b in S]
    v += [(rng.getrandbits(16), rng.getrandbits(16)) for _ in range(n)]
    # all mantissa pairs at a mid exponent (exhaustive rounding coverage)
    for fa in range(128):
        for fb in range(128):
            v.append((fm.pack(rng.getrandbits(1), 127, fa, 7), fm.pack(rng.getrandbits(1), 120, fb, 7)))
    # exponent sums near overflow / underflow boundaries
    for _ in range(n):
        ea = rng.randint(1, 254)
        target = rng.choice([1, 2, 3, 253, 254, 255, 256, 0, -1])
        eb = max(1, min(254, target + 127 - ea + rng.randint(-1, 1)))
        v.append((fm.pack(rng.getrandbits(1), ea, rng.getrandbits(7), 7),
                  fm.pack(rng.getrandbits(1), eb, rng.getrandbits(7), 7)))
    return v


def add_vectors(rng, n, m):
    S = specials(m)
    top = (1 << m) - 1
    v = [(a, b) for a in S for b in S]
    v += [(rng.getrandbits(9 + m), rng.getrandbits(9 + m)) for _ in range(n)]
    for _ in range(n):  # close exponents, random signs: cancellation + rounding
        a = rand_normal(rng, m, 2, 253)
        _, ea, _ = fm.fields(a, m)
        eb = max(1, min(254, ea + rng.randint(-(m + 4), m + 4)))
        v.append((a, fm.pack(rng.getrandbits(1), eb, rng.getrandbits(m), m)))
    for _ in range(n):  # near-total cancellation: b ~ -a
        a = rand_normal(rng, m, 2, 253)
        s, e, f = fm.fields(a, m)
        f2 = max(0, min(top, f + rng.randint(-3, 3)))
        e2 = e + rng.choice([0, 0, 0, -1, 1])
        v.append((a, fm.pack(1 - s, e2, f2, m)))
    for _ in range(n // 4):  # overflow / underflow edges
        e = rng.choice([1, 2, 3, 252, 253, 254])
        v.append((fm.pack(rng.getrandbits(1), e, rng.getrandbits(m), m),
                  fm.pack(rng.getrandbits(1), max(0, e + rng.randint(-2, 0)), rng.getrandbits(m), m)))
    return v


def run(op, vectors, ref, width_hex, tmp, vvp):
    fin = os.path.join(tmp, f"in{op}.txt")
    fout = os.path.join(tmp, f"out{op}.txt")
    with open(fin, "w") as f:
        for a, b in vectors:
            assert 0 <= a < (1 << 32) and 0 <= b < (1 << 32), (a, b)
            f.write(f"{a:08x} {b:08x}\n")
    subprocess.run(["vvp", "-n", vvp, f"+op={op}", f"+in={fin}", f"+out={fout}"],
                   check=True, stdout=subprocess.DEVNULL)
    got = [int(line, 16) for line in open(fout)]
    assert len(got) == len(vectors), (len(got), len(vectors))
    bad = []
    for (a, b), g in zip(vectors, got):
        e = ref(a, b)
        if g != e:
            bad.append((a, b, g, e))
    w = width_hex
    for a, b, g, e in bad[:10]:
        print(f"    MISMATCH a={a:0{w}x} b={b:0{w}x} rtl={g:0{w}x} model={e:0{w}x}")
    return len(vectors), len(bad)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    rng = random.Random(0xBF16)
    with tempfile.TemporaryDirectory() as tmp:
        vvp = os.path.join(tmp, "tb.vvp")
        subprocess.run(["iverilog", "-g2012", "-o", vvp,
                        os.path.join(HERE, "tb_fp.sv"),
                        os.path.join(SRC, "bf16_mul.sv"),
                        os.path.join(SRC, "fp_add.sv")], check=True)
        mv = mul_vectors(rng, n)
        a7 = add_vectors(rng, n, 7)
        a23 = add_vectors(rng, n, 23)
        cases = [
            ("bf16_mul -> BF16", 0, mv, lambda a, b: fm.mul(a, b, 7), 4),
            ("bf16_mul -> FP32", 1, mv, lambda a, b: fm.mul(a, b, 23), 8),
            ("fp_add BF16     ", 2, a7, lambda a, b: fm.add(a, b, 7), 4),
            ("fp_add FP32     ", 3, a23, lambda a, b: fm.add(a, b, 23), 8),
        ]
        total_bad = 0
        for name, op, vec, ref, w in cases:
            cnt, bad = run(op, vec, ref, w, tmp, vvp)
            total_bad += bad
            print(f"{name}: {cnt:7d} vectors, {bad} mismatches")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
