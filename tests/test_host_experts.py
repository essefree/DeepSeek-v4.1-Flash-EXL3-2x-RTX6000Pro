#!/usr/bin/env python3
"""Host-resident routed experts (DSV41_EXL3_HOST_EXPERT_LAYERS): spec parsing,
prefix -> layer, and (GPU only) a UVA view that reads what the CPU backing holds."""
from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.quantization.exl3 import (
    host_resident_empty,
    parse_layer_spec,
    prefix_layer_index,
)


def test_spec() -> None:
    assert parse_layer_spec("") == set()
    assert parse_layer_spec("33-39") == set(range(33, 40))
    assert parse_layer_spec("30, 32,34-35") == {30, 32, 34, 35}
    try:
        parse_layer_spec("39-33")
    except ValueError:
        pass
    else:
        raise AssertionError("reversed range accepted")


def test_prefix() -> None:
    assert prefix_layer_index("model.layers.34.ffn.experts") == 34
    assert prefix_layer_index("language_model.model.layers.7.ffn.experts") == 7
    assert prefix_layer_index("model.layers.41.ffn.experts") == 41  # DSpark draft layer
    assert prefix_layer_index("lm_head") is None


def test_uva_view() -> None:
    if os.environ.get("EXL3_SELFCHECK_GPU", "1") == "0" or not torch.cuda.is_available():
        print("test_host_experts: UVA view skipped (no GPU)")
        return
    host, view = host_resident_empty((4, 3, 16), torch.int16)
    assert host.is_pinned() and view.is_cuda
    assert host.data_ptr() == view.data_ptr() or view.numel() == host.numel()
    host.copy_(torch.arange(host.numel(), dtype=torch.int16).view_as(host))
    torch.cuda.synchronize()
    assert torch.equal((view[2] + 0).cpu(), host[2])
    print("test_host_experts: UVA view ok")


if __name__ == "__main__":
    test_spec()
    test_prefix()
    test_uva_view()
    print("test_host_experts: ok")
