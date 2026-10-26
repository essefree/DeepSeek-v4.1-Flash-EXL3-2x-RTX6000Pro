# DeepSeek v4.1 Flash EXL3 2.9 bpw on 2× RTX PRO 6000 Blackwell

Fork of [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
for **one x86_64 host with two RTX PRO 6000 Blackwell cards (96 GB each, sm_120)** and 256 GB of host RAM.
Same checkpoint ([`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)),
same vLLM build, same overlay; OpenAI API on `:8888`, model id `DeepSeek-v4.1-Flash-EXL3`.

> **Status: not yet booted on real hardware.** The port was written and checked offline (syntax, the
> generated launch script, the weight budget computed from the real shard headers). The first boot on the
> target box is the real test: watch `./start.sh logs` and the `[dsv41-mem]` lines.

Checkpoint, speculation, Engram, vision, and prefix-cache details are unchanged from the Spark recipe:
[README.spark.md](README.spark.md).

## What changed versus the Spark recipe

| | 2× DGX Spark (upstream) | 2× RTX PRO 6000 (this fork) |
|---|---|---|
| Topology | 2 nodes, TP=2 over CX7 RoCE, `--nnodes 2` | 1 host, 1 container, TP=2 over PCIe |
| Arch | aarch64, `sm_121a` | x86_64, `sm_120a` (`TORCH_CUDA_ARCH_LIST=12.0a`) |
| Base image | `vllm/vllm-openai:deepseekv41-flash-0909` (arm64) | same build, `-amd64` tag |
| exllamav3 build | AVX targets stubbed for aarch64 | built as upstream (AVX2/AVX-512 compile on x86) |
| Image | public GHCR arm64 image | built locally by `./start.sh` (no x86 image published) |
| Memory | 121.7 GiB unified per node | 95.6 GiB VRAM per card + separate host RAM |
| Weights per rank | 99.5 GiB, all resident | **82.9 GiB on the GPU + 16.6 GiB in pinned host RAM** |
| Weight sharing | NFS/ZFS/rsync to the worker | none (one host) |
| Memory guard | MemAvailable watchdog, page-cache drop | off; prefill release keys on free VRAM (`DSV41_MEM_SOURCE=cuda`) |
| Engram scales | off (they would cost GPU memory on UMA) | on (`DSV41_RESIDENT_SCALES=1`, plain host RAM) |
| Context default | 600k | 262k (raise after a long-prompt test) |

## Why some experts live in host RAM

`scripts/weight_budget.py` on the real shard headers, TP=2:

| bucket | per rank (GiB) |
|---|---:|
| routed experts (40 layers) | 91.29 |
| DSpark draft experts | 3.17 |
| attention, shared experts, embed, head, vision, the rest | 5.02 |
| **total** | **99.48** |

An RTX PRO 6000 has 95.6 GiB, so this checkpoint cannot be fully GPU-resident at TP=2. The overlay
(`overlay/exl3.py`, *Host-resident routed experts*) allocates the routed-expert trellis of the layers in
`DSV41_EXL3_HOST_EXPERT_LAYERS` in pinned host memory and hands the kernels a UVA CUDA view of it (the same
zero-copy mechanism as vLLM's `--cpu-offload-gb`). The kernels and pointer tables are unchanged; the GPU reads those
experts over PCIe.

| `DSV41_EXL3_HOST_EXPERT_LAYERS` | GPU-resident / rank | pinned host RAM (both ranks) |
|---|---:|---:|
| *(none)* | 99.5 GiB — does not fit | 0 |
| `34-39` | 85.2 GiB | 28.5 GiB |
| `33-39` *(default)* | 82.9 GiB | 33.2 GiB |
| `32-39` | 80.5 GiB | 38.0 GiB |

```bash
python3 scripts/weight_budget.py --host-expert-layers 33-39
```

The cost is decode speed: each offloaded layer's active experts cross PCIe 5.0 (~50 GB/s) instead of GDDR7
(~1.8 TB/s). Offload as few layers as your context and KV settings allow. `LANGUAGE_MODEL_ONLY=1` frees another
0.9 GiB per rank.

## Quick start

Requirements: Linux x86_64, NVIDIA driver with CUDA 13 support, Docker with `nvidia-container-toolkit`, ~400 GB of
disk, and `pip install -U 'huggingface_hub[hf_transfer]'`.

```bash
cp .env.example .env      # first run also does this; review GPUS / DSV41_EXL3_HOST_EXPERT_LAYERS
./start.sh                # fetch weights, build the sm_120 image, GPU self-check, serve :8888
```

### Choosing the checkpoint

```bash
./start.sh --model /data/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw   # local directory, served as-is
./start.sh --model your-hf-user/your-exl3-quant                     # Hugging Face repo, fetched first
./start.sh --model your-hf-user/your-exl3-quant --revision v2       # a branch, tag or commit
```

Or set `MODEL=` in `.env` (same two forms). With nothing set, it uses Mia's 2.9bpw repo in `./model`.
- **Local directories:** mounted read-only and never downloaded into. An incomplete one stops the boot with the
  shard count it found.
- **Hugging Face repos:** each one gets its own `./models/<user>__<name>` directory. For a private repo,
  `export HF_TOKEN=...` first.
- **Shard count:** read from the checkpoint's `model.safetensors.index.json`.
- **K map:** a checkpoint that ships `exl3_k_map.json` next to its `config.json` uses that map instead of
  `files/exl3_k_map.json`. Set `KMAP_HOST` to force one.
- **Checks:** the preflight prints the checkpoint's `quant_method`/`codebook`/`avg_bits` and warns when they don't
  match what the overlay loads.

**DeepSeek 0909 vs 1017.** Both releases have the same architecture, config, tokenizer and prompt encoder,
so the image serves either. Every shard differs, though, so the Engram tables (shards 47+48) must come from
the same release as the quant. A checkpoint that bundles its own stock 47+48 is used as-is: the launcher
points `ENGRAM_DIR` at it and hides those shards from the model loader. Otherwise, for a 1017 quant, set
`HF_ENGRAM_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017`.

`python3 scripts/inspect_checkpoint.py <dir>` reads only the shard headers and reports whether a checkpoint
loads as-is: native linears where EXL3 is expected, index and shard mismatches, the K map, and bundled Engram.

`./download.sh --model ...` stages a download without booting. For a quant other than Mia's 2.9bpw, re-check the
VRAM split with `python3 scripts/weight_budget.py --model <dir> --host-expert-layers 33-39`.

The first run builds the image (it compiles exllamav3 and the E3 v2 kernels for `sm_120a`), then runs the GPU
self-check, including the new `test_host_experts.py` UVA check. Later runs rebuild only when
`Dockerfile`/`overlay`/`files`/`tests` change.

```bash
./start.sh status | logs | stop | restart
./start.sh pack           # optional: both ranks' Engram rows onto local NVMe
BUILD=1 ./start.sh        # force a rebuild
```

The smoke test is the same as upstream (thinking off):

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-v4.1-Flash-EXL3","messages":[{"role":"user","content":"What is 17*19? Reply with the integer only."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

## First-boot checklist

1. **NCCL / P2P.** If boot hangs at `ncclCommInitRank` or the first forward, set `NCCL_P2P_DISABLE=1` and
   `DISABLE_CUSTOM_ALL_REDUCE=1` in `.env`. Some boards and IOMMU settings break PCIe P2P.
2. **Host experts engaged.** The log should show `EXL3 routed experts of layer 33 on host ...` through layer 39, and
   each GPU should be around 83 GiB used after load.
3. **OOM at boot.** Offload one more layer (`32-39`), or lower `KV_CACHE_MEMORY_BYTES` / `MAX_MODEL_LEN`.
4. **Tuning up.** Once a long prompt runs cleanly, try `34-39` (faster decode). Re-measure `DSPARK_TOKENS`:
   these GPUs are much faster than GB10, so the best k may differ.

## Not carried over

- **Cooperative MoE** (`extensions/cooperative_moe/`): its pinned binary is `sm_121a`/GB10 only.
- **NFS/ZFS sharing, image shipping, memguard**: these were for the two-node UMA layout. The scripts remain but
  `start.sh` no longer calls them.

## Alternative for this exact hardware: diffbot's 3 bpw pack

[`diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000`](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
is a separate, already-measured stack for 2× RTX PRO 6000. It uses the 1017 weights at 3 bpw with plan-based
host experts and a VRAM hot cache, custom sm_120 MoE kernels, int4 Engram, an optional AVX-512 CPU expert runtime,
and a 1M context. It reports 224 tok/s single-stream (262 with CPU experts on a 96-core Threadripper). It asks for
~260 GiB of free host RAM and has its own image and vLLM pin. Its int4 Engram shards (`int4_fp16s32`, 144 B/row) are
**not** readable by this recipe's Engram backend, which expects fp8 rows of 264 B.
