"""Engram n-gram table row formats this recipe can serve.

Detected from the safetensors headers of the shard that holds a layer's
`layers.N.engram.embed.{weight,scale}` tensors, never from config.json:

  fp8_e8m0      stock DeepSeek: weight F8_E4M3 [rows, 256], scale F8_E8M0 [rows, 8]
                row = 256 B + 8 B = 264 B, ~95 GiB per layer shard
  int4_fp16s32  diffbot re-encode (convert_engram_int4.py): weight U8 [rows, 128]
                (two int4 per byte, element 2i low nibble, 2i+1 high, nibble =
                q + 8 with q in [-8, 7]), scale F16 [rows, 8] (one per 32 values)
                row = 128 B + 16 B = 144 B, ~52 GiB per layer shard

The row store only moves bytes; the GPU dequant kernel is picked per format.
Usable as a script: `engram_format.py <dir>` prints the format name (or
`none`) for start.sh.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import sys

DIM = 256
QUANT_BLOCK = 32
EMBED_LAYERS = (1, 14)


@dataclass(frozen=True)
class RowFormat:
    name: str
    weight_dtype: str
    weight_cols: int
    scale_dtype: str
    scale_cols: int

    @property
    def weight_bytes(self) -> int:
        return self.weight_cols * _DTYPE_BYTES[self.weight_dtype]

    @property
    def scale_bytes(self) -> int:
        return self.scale_cols * _DTYPE_BYTES[self.scale_dtype]

    @property
    def row_bytes(self) -> int:
        return self.weight_bytes + self.scale_bytes


_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "U8": 1, "I8": 1, "F16": 2, "BF16": 2}

FP8 = RowFormat("fp8_e8m0", "F8_E4M3", DIM, "F8_E8M0", DIM // QUANT_BLOCK)
INT4 = RowFormat("int4_fp16s32", "U8", DIM // 2, "F16", DIM // QUANT_BLOCK)
FORMATS = {f.name: f for f in (FP8, INT4)}
# Short names accepted from the environment / .env (ENGRAM_FORMAT).
ALIASES = {"fp8": FP8.name, "stock": FP8.name, "int4": INT4.name, "diffbot": INT4.name}


def resolve_name(name: str) -> str:
    n = (name or "").strip().lower()
    if n in ("", "auto"):
        return "auto"
    n = ALIASES.get(n, n)
    if n not in FORMATS:
        raise ValueError(f"unknown Engram format {name!r} (fp8 | int4 | auto)")
    return n


def shard_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length)), 8 + length


def format_of_tensors(weight: dict, scale: dict) -> RowFormat:
    for fmt in FORMATS.values():
        if (
            weight["dtype"] == fmt.weight_dtype
            and int(weight["shape"][1]) == fmt.weight_cols
            and scale["dtype"] == fmt.scale_dtype
            and int(scale["shape"][1]) == fmt.scale_cols
        ):
            return fmt
    raise ValueError(
        f"unsupported Engram row layout: weight {weight['dtype']}{weight['shape']} "
        f"scale {scale['dtype']}{scale['shape']} (known: fp8 e4m3 [N,256]+e8m0 [N,8], "
        "int4 U8 [N,128]+F16 [N,8])"
    )


@dataclass(frozen=True)
class TableSpan:
    path: Path
    rows: int
    weight_offset: int
    scale_offset: int
    fmt: RowFormat


def table_span(root: Path, layer_id: int) -> TableSpan:
    """Where layer `layer_id`'s rows live under `root` and in which format."""
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer_id}.engram.embed."
    name = index[prefix + "weight"]
    if index[prefix + "scale"] != name:
        raise RuntimeError("Engram weight and scale must share a shard")
    path = root / name
    header, base = shard_header(path)
    weight, scale = header[prefix + "weight"], header[prefix + "scale"]
    fmt = format_of_tensors(weight, scale)
    if int(weight["shape"][0]) != int(scale["shape"][0]):
        raise RuntimeError(f"{path}: weight rows != scale rows")
    return TableSpan(
        path,
        int(weight["shape"][0]),
        base + int(weight["data_offsets"][0]),
        base + int(scale["data_offsets"][0]),
        fmt,
    )


def detect(root: Path) -> RowFormat | None:
    """Format of the Engram tables under `root`, None when there are none.

    All layers must share one format; a mix is an error, not a guess.
    """
    root = Path(root)
    if not (root / "model.safetensors.index.json").is_file():
        return None
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    found: dict[str, RowFormat] = {}
    for layer_id in EMBED_LAYERS:
        key = f"layers.{layer_id}.engram.embed.weight"
        if key not in index or not (root / index[key]).is_file():
            continue
        found[str(layer_id)] = table_span(root, layer_id).fmt
    if not found:
        return None
    names = {f.name for f in found.values()}
    if len(names) != 1:
        raise RuntimeError(f"Engram layers use different row formats under {root}: {found}")
    return next(iter(found.values()))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: engram_format.py <dir>", file=sys.stderr)
        return 2
    try:
        fmt = detect(Path(argv[1]))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(fmt.name if fmt else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
