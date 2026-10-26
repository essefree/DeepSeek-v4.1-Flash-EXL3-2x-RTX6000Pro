#!/usr/bin/env python3
"""CPU parity for librow_store: stock fp8 rows (264 B) and int4 rows (144 B),
DEAD_ID=-1 zeros, out-of-range rows zeroed, pinned scales, pooled lookups."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    Path("/opt/dsv41/librow_store.so"),
    ROOT / "overlay" / "librow_store.so",
)


def _lib():
    for path in CANDIDATES:
        if path.is_file():
            return C.CDLL(str(path)), path
    raise SystemExit("librow_store.so missing — build the serving image first")


lib, lib_path = _lib()
U = C.c_uint64
P = C.c_void_p
lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
lib.row_store_open.restype = P
lib.row_store_open2.argtypes = [C.c_char_p, U, U, U, U, U, U]
lib.row_store_open2.restype = P
lib.row_store_close.argtypes = [P]
lib.row_store_stats.argtypes = [P, C.POINTER(U)]
lib.row_store_range.argtypes = [P, U, U]


class Work(C.Structure):
    _fields_ = [("store", P), ("ids", P), ("weights", P), ("scales", P), ("count", U)]


lib.row_store_lookup.argtypes = [C.POINTER(Work)]


def lookup(store, ids: list[int], wb: int, sb: int) -> tuple[bytes, bytes]:
    indices = (C.c_int64 * len(ids))(*ids)
    w = C.create_string_buffer(len(ids) * wb)
    s = C.create_string_buffer(len(ids) * sb)
    work = Work(store, C.addressof(indices), C.addressof(w), C.addressof(s), len(ids))
    lib.row_store_lookup(C.byref(work))
    return w.raw, s.raw


def check_format(label: str, wb: int, sb: int, legacy_open: bool) -> None:
    rows = 4099
    rng = random.Random(413 + wb)
    weights, scales = rng.randbytes(rows * wb), rng.randbytes(rows * sb)
    offset = 777
    with tempfile.NamedTemporaryFile() as f:
        f.write(bytes(offset) + weights + scales)
        f.flush()
        budget = (wb + sb + 8) * 64
        if legacy_open:
            store = lib.row_store_open(f.name.encode(), rows, offset, offset + len(weights), budget)
        else:
            store = lib.row_store_open2(
                f.name.encode(), rows, offset, offset + len(weights), budget, wb, sb
            )
        assert store, (label, lib_path)
        lo, hi = 16, rows - 7
        lib.row_store_range(store, lo, hi)

        ids = [-1, lo, hi - 1, 15, hi, 1000, 1000]
        w, s = lookup(store, ids, wb, sb)
        row = lambda i: (weights[i * wb : (i + 1) * wb], scales[i * sb : (i + 1) * sb])
        assert w[:wb] == bytes(wb) and s[:sb] == bytes(sb), f"{label}: DEAD_ID must zero the row"
        assert (w[wb : 2 * wb], s[sb : 2 * sb]) == row(lo), f"{label}: first owned row"
        assert (w[2 * wb : 3 * wb], s[2 * sb : 3 * sb]) == row(hi - 1), f"{label}: last owned row"
        assert w[3 * wb : 4 * wb] == bytes(wb) and s[3 * sb : 4 * sb] == bytes(sb), (
            f"{label}: row below the owned range must be zero"
        )
        assert w[4 * wb : 5 * wb] == bytes(wb), f"{label}: row above the owned range must be zero"
        assert (w[5 * wb : 6 * wb], s[5 * sb : 6 * sb]) == row(1000), f"{label}: miss"
        assert (w[6 * wb : 7 * wb], s[6 * sb : 7 * sb]) == row(1000), f"{label}: cache hit"

        # A wide pooled lookup: every owned row once, in random order.
        many = list(range(lo, hi))
        rng.shuffle(many)
        w, s = lookup(store, many, wb, sb)
        for n, i in enumerate(many[:: max(1, len(many) // 97)]):
            k = many.index(i)
            assert (w[k * wb : (k + 1) * wb], s[k * sb : (k + 1) * sb]) == row(i), (
                f"{label}: pooled row {i}"
            )
        out = (U * 9)()
        lib.row_store_stats(store, out)
        hits, misses, reads, cache_bytes, slots, scale_bytes = out[:6]
        assert hits >= 1 and misses >= len(many), (label, hits, misses)
        if os.environ.get("DSV41_RESIDENT_SCALES", "1") != "0":
            assert scale_bytes == (hi - lo) * sb, (label, scale_bytes)
        lib.row_store_close(store)
    print(f"test_row_store: {label} ok (rows {wb}+{sb} B)")


def main() -> None:
    check_format("fp8_e8m0 (row_store_open)", 256, 8, legacy_open=True)
    check_format("fp8_e8m0", 256, 8, legacy_open=False)
    check_format("int4_fp16s32", 128, 16, legacy_open=False)
    print(f"test_row_store: ok ({lib_path})")


if __name__ == "__main__":
    main()
