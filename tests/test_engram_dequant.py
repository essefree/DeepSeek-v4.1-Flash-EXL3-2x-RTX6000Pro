#!/usr/bin/env python3
"""The Engram row dequant kernels must decode their row formats exactly:
fp8 e4m3 bytes x 2^(e8m0-127) (not integer byte values), and diffbot int4
nibbles (q+8, low nibble first) x fp16 per-32 scales.
GPU test; prints a skip line without CUDA (image build)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _d in (Path("/opt/dsv41"), ROOT / "overlay"):
    if (_d / "engram_file_backend.py").is_file():
        sys.path.insert(0, str(_d))
        break


def int4_reference(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pure-torch decode of int4_fp16s32 rows, mirroring convert_engram_int4.py."""
    lo = (packed & 0x0F).to(torch.int32)
    hi = (packed >> 4).to(torch.int32)
    nib = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)  # element 2i, 2i+1
    q = (nib - 8).float()
    return q * scales.float().repeat_interleave(32, dim=1)


def check_fp8(fb) -> None:
    kernel = fb._dequant_kernel()
    rows, dim, qb = 64, 256, 32
    torch.manual_seed(0)
    w = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, device="cuda")
    w[w == 0x7F] = 0
    w[w == 0xFF] = 0  # NaN encodings
    s = torch.randint(120, 134, (rows, dim // qb), dtype=torch.uint8, device="cuda")
    ref = w.view(torch.float8_e4m3fn).float() * (2.0 ** (s.float() - 127)).repeat_interleave(qb, dim=1)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device="cuda")
    kernel[(4,)](w.view(torch.float8_e4m3fn), s, out, rows, DIM=dim, QUANT_BLOCK=qb, BLOCK_R=16, GRID=4)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel
    print(f"test_engram_dequant: fp8_e8m0 ok (rel_err={rel:.2e})")


def check_int4(fb) -> None:
    import engram_format as ef

    rows, dim, qb = 70, 256, 32
    torch.manual_seed(1)
    packed = torch.randint(0, 256, (rows, dim // 2), dtype=torch.uint8, device="cuda")
    scales = (torch.rand(rows, dim // qb, device="cuda") * 0.2 + 0.001).to(torch.float16)
    ref = int4_reference(packed, scales)
    # Same path the backend takes: uint8 staging buffers, fp16 view of the scale bytes.
    dev_s = scales.view(torch.uint8)
    assert dev_s.shape == (rows, ef.INT4.scale_bytes)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device="cuda")
    fb.dequant_rows(ef.INT4, packed, dev_s, out, rows, num_sms=4)
    torch.cuda.synchronize()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel
    # Nibble order: element 0 is the LOW nibble of byte 0.
    one = torch.zeros(1, dim // 2, dtype=torch.uint8, device="cuda")
    one[0, 0] = (8 + 3) | ((8 - 5) << 4)  # q0 = +3, q1 = -5
    sc = torch.full((1, dim // qb), 2.0, dtype=torch.float16, device="cuda")
    out1 = torch.empty(1, dim, dtype=torch.bfloat16, device="cuda")
    fb.dequant_rows(ef.INT4, one, sc.view(torch.uint8), out1, 1, num_sms=1)
    torch.cuda.synchronize()
    assert out1[0, 0].item() == 6.0 and out1[0, 1].item() == -10.0, out1[0, :2]
    assert out1[0, 2:].eq(-16.0).all().item(), "zero nibble must decode to q=-8 -> -8 * 2.0"
    print(f"test_engram_dequant: int4_fp16s32 ok (rel_err={rel:.2e})")


def main() -> None:
    if not torch.cuda.is_available():
        print("test_engram_dequant: no CUDA here; skipped (runs in the GPU self-check)")
        return
    import engram_file_backend as fb

    check_fp8(fb)
    check_int4(fb)


if __name__ == "__main__":
    main()
