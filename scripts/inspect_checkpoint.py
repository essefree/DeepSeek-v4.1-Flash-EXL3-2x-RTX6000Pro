#!/usr/bin/env python3
"""Check an EXL3 DeepSeek-V4.1-Flash checkpoint against what this recipe loads.

Reads only safetensors headers (a few MB), never the weights. Compares the
actual tensors in the shards with Mia-AiLab's 2.9bpw layout and the bundled
files/exl3_k_map.json, and reports anything the overlay will not load as-is.

usage: inspect_checkpoint.py /path/to/checkpoint [--kmap files/exl3_k_map.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Linears this overlay loads as EXL3 trellis (see Exl3Config.get_quant_method).
PACKED_SUFFIXES = ("trellis", "suh", "svh", "mul1")


def headers(model: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(model.glob("*.safetensors")):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            v["file"] = f.name
            out[k] = v
    return out


def norm(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--kmap", default=str(ROOT / "files" / "exl3_k_map.json"))
    a = ap.parse_args()
    model = Path(a.model).expanduser()
    problems: list[str] = []
    notes: list[str] = []

    cfg_path = model / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    q = cfg.get("quantization_config") or {}
    if not cfg:
        qp = model / "quantization_config.json"
        q = json.loads(qp.read_text()) if qp.is_file() else {}
        problems.append("no config.json in the checkpoint directory (vLLM needs it)")
    print(f"architectures : {cfg.get('architectures')}")
    print(f"quant         : {q.get('quant_method')} {q.get('codebook')} bits={q.get('bits', q.get('avg_bits'))} "
          f"head_bits={q.get('head_bits')} quantizer={q.get('version')}")
    if cfg and cfg.get("architectures") != ["DeepseekV41ForCausalLM"]:
        problems.append(f"architectures={cfg.get('architectures')} (image only has DeepseekV41ForCausalLM)")
    if q.get("quant_method") != "exl3" or q.get("codebook") not in ("mul1", "mcg"):
        problems.append("not an EXL3 mul1/mcg checkpoint")
    if q.get("version") and not str(q.get("version")).startswith("1.4"):
        notes.append(f"quantizer {q.get('version')} vs runtime exllamav3 v1.4.5 in the image: Mia's pack was "
                     "1.4.2; the trellis format has not been checked across versions")

    h = headers(model)
    if not h:
        print("no *.safetensors shards found here — run this where the weights are", file=sys.stderr)
        return 2
    idx_path = model / "model.safetensors.index.json"
    if idx_path.is_file():
        wm = json.loads(idx_path.read_text())["weight_map"]
        missing = sorted(k for k in wm if k not in h)
        stray = sorted(k for k in h if k not in wm)
        if missing:
            problems.append(f"index lists {len(missing)} tensors that no shard contains, e.g. {missing[:3]}")
        if stray:
            notes.append(f"{len(stray)} tensors are in shards but not in the index, e.g. {stray[:3]}")

    # 1. Native (unquantized) linears where the overlay expects trellis.
    packed_mods = {k.rsplit(".", 1)[0] for k in h if k.rsplit(".", 1)[-1] in PACKED_SUFFIXES}
    native = collections.defaultdict(list)
    for k, v in h.items():
        if "." not in k:
            continue
        mod, suf = k.rsplit(".", 1)
        if suf == "weight" and (".attn.wo_b" in k or ".shared_experts." in k or ".engram.wkv" in k):
            native[norm(mod)].append((k, v["dtype"], v["file"], mod in packed_mods))
    for mod, rows in sorted(native.items()):
        both = sum(r[3] for r in rows)
        dt = sorted({r[1] for r in rows})
        files = sorted({r[2] for r in rows})
        layers = sorted({int(m.group(1)) for r in rows if (m := re.search(r"layers\.(\d+)\.", r[0]))})
        if "engram.wkv" in mod and all(f.endswith("-of-00048.safetensors") for f in files):
            notes.append(f"{mod}: native {dt} copy from the original Engram shards {files} "
                         f"({'trellis copy also present' if both else 'NO trellis copy'})")
            if not both:
                problems.append(f"{mod} exists only as native {dt}; the overlay loads Engram wkv as EXL3")
        else:
            problems.append(f"{mod}: {len(rows)} native {dt} tensors in layers {layers} "
                            f"({'both native and trellis' if both else 'no trellis'}) — the overlay builds "
                            "these as EXL3 linears; needs a native-linear exception")

    # 2. Engram tables bundled in the model directory.
    eng = sorted({v["file"] for k, v in h.items() if ".engram.embed." in k})
    if eng:
        notes.append(f"Engram tables are inside this directory ({', '.join(eng)}): ./start.sh uses them as "
                     "ENGRAM_DIR and hides them from the model loader (a view without those shards); they "
                     "must be from the same DeepSeek release (0909 or 1017) as the quant")

    # 3. K per tensor against the K map.
    km = json.loads(Path(a.kmap).read_text())
    bad = []
    for k, v in h.items():
        m = re.match(r"layers\.(\d+)\.ffn\.experts\.\d+\.w\d\.trellis$", k)
        if m and km["routed"].get(m.group(1)) != v["shape"][-1] // 16:
            bad.append(k)
    if bad:
        problems.append(f"{len(bad)} routed-expert trellis tensors disagree with {a.kmap}, e.g. {bad[:2]} "
                        "— ship an exl3_k_map.json with the checkpoint")
    else:
        print("routed K map  : matches the bundled exl3_k_map.json")
    print(f"vision tower  : {'present' if any(k.startswith('vision.') for k in h) else 'absent'}"
          f"   bias_vl: {'present' if any(k.endswith('.bias_vl') for k in h) else 'absent'}")

    for n in notes:
        print(f"NOTE    {n}")
    for p in problems:
        print(f"PROBLEM {p}")
    print("verdict       : " + ("loads as-is" if not problems else f"{len(problems)} problem(s) before it can load"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
