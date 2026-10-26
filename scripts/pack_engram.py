"""Repack this rank's owned Engram rows into a local contiguous shard.

In the checkpoint a row's weight bytes and its scale bytes sit in two tensors
tens of GB apart, so a cache miss costs two reads into unrelated 4 KiB pages.
Packing the owned rows as contiguous records (weight then scale: 264 B for the
stock fp8 tables, 144 B for diffbot's int4 tables) on local disk makes a miss
one local read. The format is read from the shard header (engram_format.py).

vLLM shards by hash-head buckets (not an even row split). This packer uses
that same [lo, hi) so row_store_attach_packed accepts the shard.

Per rank at TP=2 and per layer: ~48 GiB packed for fp8 rows, ~26 GiB for int4.

Run inside the serving image, where /models is the checkpoint and the output
directory is node-local storage:
    python3 scripts/pack_engram.py --rank 0 --tp 2 --out /engram
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "overlay", Path("/opt/dsv41")):
    if (_p / "engram_layout.py").is_file():
        sys.path.insert(0, str(_p))
        break
from engram_format import table_span  # noqa: E402
from engram_layout import layer_head_sizes_from_config, shard_range  # noqa: E402

MAGIC = 0x31344E4531565344  # "DSV41EN1"
HEADER_BYTES = 4096
ENGRAM_LAYERS = (1, 14)


def tensor_span(root: Path, layer_id: int):
    """(path, rows, weight_offset, scale_offset, weight_bytes, scale_bytes)."""
    s = table_span(root, layer_id)
    return (s.path, s.rows, s.weight_offset, s.scale_offset,
            s.fmt.weight_bytes, s.fmt.scale_bytes)


def owned_row_range(root: Path, layer_id: int, rank: int, tp: int) -> tuple[int, int]:
    """vLLM hash-head shard for this rank, not an even row split."""
    cfg = json.loads((root / "config.json").read_text())
    text = cfg.get("text_config") or cfg
    layer_ids = [int(x) for x in text["engram_layer_ids"]]
    sizes = layer_head_sizes_from_config(text, layer_ids.index(int(layer_id)))
    lo, hi, _, _ = shard_range(sizes, rank, tp)
    return lo, hi


def pack_layer(root: Path, out_dir: Path, layer_id: int, rank: int, tp: int,
               chunk_rows: int) -> Path:
    path, rows, weight_off, scale_off, WEIGHT_BYTES, SCALE_BYTES = tensor_span(root, layer_id)
    ROW_BYTES = WEIGHT_BYTES + SCALE_BYTES
    lo, hi = owned_row_range(root, layer_id, rank, tp)
    if not (0 <= lo < hi <= rows):
        raise SystemExit(
            f"layer {layer_id} shard [{lo},{hi}) outside table rows={rows}"
        )
    count = hi - lo
    target = out_dir / f'engram-l{layer_id}-r{rank}of{tp}.bin'
    partial = target.with_suffix('.partial')
    expected = HEADER_BYTES + count * ROW_BYTES

    if target.exists() and target.stat().st_size == expected:
        print(f'layer {layer_id}: {target} already complete ({expected/2**30:.1f} GiB)',
              flush=True)
        return target

    header = bytearray(HEADER_BYTES)
    struct.pack_into('<6Q', header, 0, MAGIC, layer_id, lo, hi, rows, ROW_BYTES)

    started = time.monotonic()
    written = 0
    with open(path, 'rb', buffering=0) as src, open(partial, 'wb', buffering=0) as dst:
        dst.write(header)
        for start in range(lo, hi, chunk_rows):
            stop = min(start + chunk_rows, hi)
            n = stop - start
            src.seek(weight_off + start * WEIGHT_BYTES)
            weights = src.read(n * WEIGHT_BYTES)
            src.seek(scale_off + start * SCALE_BYTES)
            scales = src.read(n * SCALE_BYTES)
            if len(weights) != n * WEIGHT_BYTES or len(scales) != n * SCALE_BYTES:
                raise SystemExit(f'short read from {path} at row {start}')
            block = numpy.empty((n, ROW_BYTES), dtype=numpy.uint8)
            block[:, :WEIGHT_BYTES] = numpy.frombuffer(
                weights, dtype=numpy.uint8).reshape(n, WEIGHT_BYTES)
            block[:, WEIGHT_BYTES:] = numpy.frombuffer(
                scales, dtype=numpy.uint8).reshape(n, SCALE_BYTES)
            dst.write(block.tobytes())
            written += n
            elapsed = time.monotonic() - started
            rate = written * ROW_BYTES / max(elapsed, 1e-6)
            print(f'layer {layer_id}: {written}/{count} rows '
                  f'({100.0*written/count:5.1f}%) {rate/2**20:.0f} MiB/s', flush=True)
        dst.flush()
        os.fsync(dst.fileno())
    partial.rename(target)
    print(f'layer {layer_id}: wrote {target} ({expected/2**30:.1f} GiB) in '
          f'{time.monotonic()-started:.0f}s', flush=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.environ.get('DSV41_SOURCE', '/models'))
    parser.add_argument('--out', default=os.environ.get('DSV41_PACKED_DIR', '/engram'))
    parser.add_argument('--rank', type=int, default=int(os.environ.get('NODE_RANK', '0')))
    parser.add_argument('--tp', type=int, default=int(os.environ.get('TP_SIZE', os.environ.get('TP', '2'))))
    parser.add_argument('--chunk-rows', type=int, default=1 << 18)
    parser.add_argument('--layers', default=','.join(str(x) for x in ENGRAM_LAYERS))
    args = parser.parse_args()

    root, out_dir = Path(args.model), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(',') if x.strip()]

    need = 0
    for layer_id in layers:
        lo, hi = owned_row_range(root, layer_id, args.rank, args.tp)
        _, _, _, _, wb, sb = tensor_span(root, layer_id)
        need += HEADER_BYTES + (hi - lo) * (wb + sb)
    free = os.statvfs(out_dir).f_bavail * os.statvfs(out_dir).f_frsize
    print(f'rank {args.rank}/{args.tp}: need {need/2**30:.1f} GiB, '
          f'{free/2**30:.1f} GiB free at {out_dir}', flush=True)
    if free < need:
        raise SystemExit(f'not enough space at {out_dir}')

    for layer_id in layers:
        pack_layer(root, out_dir, layer_id, args.rank, args.tp, args.chunk_rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
