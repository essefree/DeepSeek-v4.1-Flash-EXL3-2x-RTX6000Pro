#!/usr/bin/env python3
"""Expected resident weight bytes per TP rank for this EXL3 checkpoint.

Reads model.safetensors.index.json (dtype/shape via each shard header) and
applies the vLLM sharding of DeepseekV41ForCausalLM at TP=N:

  * routed experts (layers.N.ffn.experts.E.*), MTP experts (mtp.N.ffn.experts.*):
    gate/up column-parallel, down row-parallel -> 1/TP per rank
  * shared experts, attention wq_b / wkv_b-style column-parallel and wo row/col
    slices, embed (vocab-parallel), head (vocab-parallel): 1/TP
  * everything else (norms, router gates, MLA a-projections, indexer,
    Engram q/k/wkv, vision tower, aligner, mtp norms/heads): replicated

The split is approximate for attention (a-projections are replicated in
vLLM's MLA layout) but attention is ~3 GiB total, so the error is < 1 GiB.
Prints a table and the per-rank total; ./start.sh uses the total for its
pre-launch MemAvailable check.

usage: weight_budget.py [--model DIR] [--tp N] [--host-expert-layers SPEC] [--json]

--host-expert-layers (e.g. "33-39", same syntax as DSV41_EXL3_HOST_EXPERT_LAYERS)
moves those layers' routed experts out of the GPU total into a host total.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

GIB = 1024**3


def shard_headers(model: Path, files: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in sorted(files):
        with (model / name).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            out[k] = v
    return out


def nbytes(entry: dict) -> int:
    a, b = entry["data_offsets"]
    return int(b) - int(a)


REPLICATED_ATTN = ("wq_a", "wkv", "q_norm", "kv_norm", "indexer", "compress", "wo_b")


def classify(name: str) -> tuple[str, bool]:
    """(bucket, sharded?)"""
    if name.startswith("embed."):
        return "embed", True
    if name.startswith("head."):
        return "lm_head", True
    if name.startswith("vision.") or name.startswith("aligner.") or name.startswith("image_"):
        return "vision", False
    if name.startswith("mtp."):
        rest = name.split(".", 2)[2]
        if ".ffn.experts." in name:
            return "mtp_experts", True
        if "shared_experts" in name:
            return "mtp_shared", True
        if rest.startswith("attn."):
            return "mtp_attn", not any(t in rest for t in REPLICATED_ATTN)
        return "mtp_other", False
    m = re.match(r"layers\.(\d+)\.(.*)", name)
    if m:
        rest = m.group(2)
        if rest.startswith("ffn.experts."):
            return "routed_experts", True
        if rest.startswith("ffn.shared_experts"):
            return "shared_experts", True
        if rest.startswith("ffn.gate"):
            return "router", False
        if rest.startswith("attn."):
            return "attn", not any(t in rest for t in REPLICATED_ATTN)
        if rest.startswith("engram."):
            return "engram_proj", False
        return "layer_other", False
    return "other", False


def parse_layer_spec(raw: str) -> set[int]:
    out: set[int] = set()
    for part in (raw or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _expert_layer(name: str) -> int | None:
    m = re.match(r"layers\.(\d+)\.ffn\.experts\.", name)
    if m:
        return int(m.group(1))
    m = re.match(r"mtp\.(\d+)\.ffn\.experts\.", name)
    return 40 + int(m.group(1)) if m else None  # vLLM numbers the DSpark draft 40-42


def budget(model: Path, tp: int, host_layers: set[int] | None = None) -> dict:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    hdr = shard_headers(model, set(wm.values()))
    total = Counter()
    per_rank = Counter()
    host_per_rank = 0
    host_layers = host_layers or set()
    for name in wm:
        e = hdr.get(name)
        if e is None:
            continue
        b = nbytes(e)
        bucket, sharded = classify(name)
        total[bucket] += b
        if _expert_layer(name) in host_layers and name.endswith(".trellis"):
            host_per_rank += b // tp
            continue
        per_rank[bucket] += b // tp if sharded else b
    return {
        "tp": tp,
        "total_bytes": sum(total.values()),
        "per_rank_bytes": sum(per_rank.values()),
        "host_per_rank_bytes": host_per_rank,
        "buckets": {k: {"total": total[k], "per_rank": per_rank[k]} for k in sorted(total)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "model"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--host-expert-layers", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = budget(Path(a.model), a.tp, parse_layer_spec(a.host_expert_layers))
    if a.json:
        print(json.dumps(r))
        return 0
    print(f"{'bucket':16s} {'total GiB':>10s} {'per-rank GiB':>13s}")
    for k, v in r["buckets"].items():
        print(f"{k:16s} {v['total']/GIB:10.2f} {v['per_rank']/GIB:13.2f}")
    print(f"{'TOTAL':16s} {r['total_bytes']/GIB:10.2f} {r['per_rank_bytes']/GIB:13.2f}   (TP={r['tp']}, GPU-resident)")
    if r["host_per_rank_bytes"]:
        print(f"{'host experts':16s} {'':10s} {r['host_per_rank_bytes']/GIB:13.2f}   (pinned host RAM per rank)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
