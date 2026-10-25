#!/usr/bin/env bash
# ============================================================================
# start.sh — DeepSeek-V4.1-Flash EXL3 on one host with 2x RTX PRO 6000 (SM120)
# ============================================================================
#
# Serve the EXL3 2.9 bpw checkpoint (mul1, mixed K, native DSpark) on ONE
# x86_64 host with two 96 GB RTX PRO 6000 Blackwell cards: vLLM TP=2 over
# PCIe in a single container, OpenAI API on :8888.
# Base image: vllm/vllm-openai:deepseekv41-flash-0909-amd64 + the EXL3 overlay.
#
# Fork of the 2x DGX Spark recipe. Gone: worker node, CX7/RoCE GIDs, NFS/ZFS
# weight sharing, image shipping, UMA memory guards. New: part of the routed
# experts in pinned host RAM (DSV41_EXL3_HOST_EXPERT_LAYERS), because the
# checkpoint is 99.5 GiB per rank and a card has 95.6 GiB.
#
# What we do:
#   1. preflight  — docker + NVIDIA toolkit, 2 SM120 GPUs, VRAM/RAM budget
#   2. image      — docker build IMAGE (local tag); rebuild when the recipe
#                   stamp (Dockerfile/overlay/files/tests) changes
#   3. weights    — fetch EXL3 + Engram shards 47+48, build the slim Engram dir
#   4. launch     — one container, --gpus, vllm serve --tensor-parallel-size 2
#   5. wait       — poll /health, then a nonfatal DSpark/sampler warmup
#
# Usage:
#   ./start.sh                    start — default
#   ./start.sh pack               pack Engram rows (both ranks) onto local NVMe (optional)
#   ./start.sh stop               stop the container
#   ./start.sh restart            stop + start
#   ./start.sh status             container + API health + VRAM
#   ./start.sh logs               follow logs
#
# Settings live in .env (copied from .env.example on first run).
# One-shot overrides: BUILD=1 SKIP_BUILD=1 SKIP_OVERLAY_VERIFY=1 TAIL=1
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || { echo "ERROR: missing .env.example" >&2; exit 1; }
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    printf '\033[1;36m[dsv41-exl3]\033[0m wrote .env from .env.example\n'
fi
# Command-line values win over .env for these (same list as the Spark recipe).
_cli_keys=(SPEC_METHOD ENFORCE_EAGER EXL3_FUSED_MOE EXL3_MOE_ROW_TILE EXL3_TEMP_ROWS_FUSED
           EXL3_FAT_SORTED EXL3_FAT_BATCHED EXL3_FAT_KERNEL EXL3_FAT_GROUPED MAX_NUM_BATCHED_TOKENS
           IMAGE GPU_MEM_UTIL LANGUAGE_MODEL_ONLY MAX_NUM_SEQS MAX_MODEL_LEN DSPARK_TOKENS
           DSV41_EXL3_HOST_EXPERT_LAYERS GPUS)
declare -A _cli=()
for _k in "${_cli_keys[@]}"; do
    [ -n "${!_k-}" ] && _cli[$_k]="${!_k}"
done
_cli_spinwait_ms_set="${GLM53_SPINWAIT_MS+1}"
_cli_spinwait_ms="${GLM53_SPINWAIT_MS-}"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
for _k in "${!_cli[@]}"; do printf -v "$_k" '%s' "${_cli[$_k]}"; done
[ -n "${_cli_spinwait_ms_set}" ] && GLM53_SPINWAIT_MS="$_cli_spinwait_ms"

# ----------------------------- configuration -------------------------------
IMAGE="${IMAGE:-dsv41-flash-exl3-rtx6000:2.9bpw}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-v4.1-Flash-EXL3}"
GPUS="${GPUS:-all}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
CG_ESTIMATE="${CG_ESTIMATE:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TP="${TP:-2}"
PORT="${PORT:-8888}"

SPEC_METHOD="${SPEC_METHOD:-dspark}"
DSPARK_TOKENS="${DSPARK_TOKENS:-3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-64}"
PREFIX_CACHE_RETENTION_INTERVAL="${PREFIX_CACHE_RETENTION_INTERVAL-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1536}"
CHAT_TEMPLATE_HOST="${CHAT_TEMPLATE_HOST:-$SCRIPT_DIR/files/chat_template.jinja}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/opt/dsv41/chat_template.jinja}"
EXL3_OVERLAY_HOST="${EXL3_OVERLAY_HOST:-$SCRIPT_DIR/overlay/exl3.py}"
KMAP_HOST="${KMAP_HOST:-$SCRIPT_DIR/files/exl3_k_map.json}"
QUANTIZATION="${QUANTIZATION:-exl3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
if [ -z "${LIMIT_MM:-}" ]; then
    LIMIT_MM='{"image":100}'
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0a}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
if [ "${ENFORCE_EAGER}" != "1" ]; then
    case " ${EXTRA_ARGS:-} " in
        *" --cudagraph-capture-sizes "*|*" cudagraph-capture-sizes "*) ;;
        *) EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes 1 2 3 4 6 8 12 18 24" ;;
    esac
fi
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
EXL3_FAT_GROUPED="${EXL3_FAT_GROUPED:-1}"
EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-0}"
EXL3_FAT_SORTED="${EXL3_FAT_SORTED:-0}"
EXL3_FAT_BATCHED="${EXL3_FAT_BATCHED:-0}"
EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-16}"
DSV41_EXL3_HOST_EXPERT_LAYERS="${DSV41_EXL3_HOST_EXPERT_LAYERS-33-39}"

READY_TIMEOUT="${READY_TIMEOUT:-1500}"
DSV41_HANG_SECONDS="${DSV41_HANG_SECONDS:-420}"
GLM53_SUPPRESS_STOPS_IN_REASONING="${GLM53_SUPPRESS_STOPS_IN_REASONING:-1}"
GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-0}"
GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-stock}"
GLM53_SPINWAIT_MS="${GLM53_SPINWAIT_MS-stock}"
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
DSV41_BOOT_SHAPE_WARMUP="${DSV41_BOOT_SHAPE_WARMUP:-1}"
DSV41_WARMUP_REQ_TIMEOUT="${DSV41_WARMUP_REQ_TIMEOUT:-240}"
VLLM_API_KEY="${VLLM_API_KEY:-}"

CONTAINER="${CONTAINER:-dsv41-exl3}"

MODEL_HOST="${MODEL_HOST:-$SCRIPT_DIR/model}"
ENGRAM_DIR="${ENGRAM_DIR:-$SCRIPT_DIR/engram-src}"
HF_MODEL_REPO="${HF_MODEL_REPO:-Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
HF_ENGRAM_REPO="${HF_ENGRAM_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
# docker -v needs absolute paths; .env may give them relative to this checkout.
for _v in MODEL_HOST ENGRAM_DIR; do
    case "${!_v}" in /*) ;; *) printf -v "$_v" '%s' "$SCRIPT_DIR/${!_v#./}" ;; esac
done
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-dsv41-flash-exl3}"
ENGRAM_SRC="${ENGRAM_SRC:-$CACHE_ROOT/engram-src}"
TRITON_HOST_CACHE="${TRITON_HOST_CACHE:-$CACHE_ROOT/triton}"
TILELANG_HOST_CACHE="${TILELANG_HOST_CACHE:-$CACHE_ROOT/tilelang}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton/cache}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/root/.tilelang/cache}"
MODEL_DIR="${MODEL_DIR:-/model}"
ENGRAM_MOUNT="${ENGRAM_MOUNT:-/engram-src}"
# Engram is plain host RAM here (not VRAM): resident scales on by default.
DSV41_CACHE_GIB="${DSV41_CACHE_GIB:-0}"
DSV41_RESIDENT_SCALES="${DSV41_RESIDENT_SCALES:-1}"
DSV41_IO_THREADS="${DSV41_IO_THREADS:-96}"
DSV41_CACHE_WAYS="${DSV41_CACHE_WAYS:-4}"
DSV41_STATS_SECONDS="${DSV41_STATS_SECONDS:-60}"
DSV41_BOOT_MARGIN_GIB="${DSV41_BOOT_MARGIN_GIB:-8}"
WEIGHT_BUDGET_HOST="${WEIGHT_BUDGET_HOST:-$SCRIPT_DIR/scripts/weight_budget.py}"
PACKED_DIR="${PACKED_DIR:-$HOME/dsv41-engram}"
PACKED_MOUNT="${PACKED_MOUNT:-/engram-packed}"

LOGDIR="$SCRIPT_DIR/logs"
INNER_SCRIPT="$SCRIPT_DIR/.dsv41-exl3.inner.sh"
HEAD_SCRIPT="$INNER_SCRIPT"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-39}"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[dsv41-exl3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dsv41-exl3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[dsv41-exl3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# GLM53 numeric config guard (begin)
_glm53_canonical_positive_int() {
    local name="$1" value="$2" maximum="$3" canonical
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a positive base-10 integer (got: $value)" >&2
        return 2
    fi
    canonical="$value"
    while [ "${canonical#0}" != "$canonical" ]; do canonical="${canonical#0}"; done
    [ -n "$canonical" ] || canonical=0
    if [ "$canonical" = 0 ] \
       || [ "${#canonical}" -gt "${#maximum}" ] \
       || [ "$canonical" -gt "$maximum" ]; then
        echo "$name must be between 1 and $maximum (got: $value)" >&2
        return 2
    fi
    printf -v "$name" '%s' "$canonical"
    # shellcheck disable=SC2163
    export "$name"
}

_glm53_validate_enum() {
    local name="$1" value="$2" allowed
    shift 2
    for allowed in "$@"; do
        [ "$value" = "$allowed" ] && return 0
    done
    echo "$name must be one of: $* (got: $value)" >&2
    return 2
}

_glm53_validate_spinwait_ms() {
    if [ "$GLM53_SPINWAIT_MS" = "stock" ]; then
        export GLM53_SPINWAIT_MS
        return 0
    fi
    _glm53_canonical_positive_int \
        GLM53_SPINWAIT_MS "$GLM53_SPINWAIT_MS" 1000
}

_dsv41_validate_prefix_retention() {
    local value="${PREFIX_CACHE_RETENTION_INTERVAL-4096}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "PREFIX_CACHE_RETENTION_INTERVAL must be 0 or a positive multiple of 128 (got: $value)" >&2
        return 2
    fi
    while [ "${value#0}" != "$value" ]; do value="${value#0}"; done
    value="${value:-0}"
    if [ "${#value}" -gt 7 ] || [ "$value" -gt 1048576 ] || (( value % 128 != 0 )); then
        echo "PREFIX_CACHE_RETENTION_INTERVAL must be 0 or a multiple of 128 up to 1048576 (got: $value)" >&2
        return 2
    fi
    if [[ " ${EXTRA_ARGS:-} " =~ [[:space:]]--prefix-cache-retention-interval([[:space:]=]|$) ]]; then
        echo "Set PREFIX_CACHE_RETENTION_INTERVAL instead of duplicating --prefix-cache-retention-interval in EXTRA_ARGS" >&2
        return 2
    fi
    PREFIX_CACHE_RETENTION_INTERVAL="$value"
    export PREFIX_CACHE_RETENTION_INTERVAL
}

validate_numeric_config() {
    if ! [[ "$GPU_MEM_UTIL" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        echo "GPU_MEM_UTIL must be greater than 0 and at most 1 (got: $GPU_MEM_UTIL)" >&2
        return 2
    fi
    _glm53_canonical_positive_int MAX_MODEL_LEN "$MAX_MODEL_LEN" 1048576 || return
    _glm53_canonical_positive_int MAX_NUM_SEQS "$MAX_NUM_SEQS" 4096 || return
    _glm53_canonical_positive_int MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS" 8388608 || return
    _glm53_validate_enum GLM53_INDEXER_WORKSPACE "${GLM53_INDEXER_WORKSPACE-stock}" \
        stock rightsize || return
    _glm53_validate_spinwait_ms || return
    _dsv41_validate_prefix_retention || return
}
# GLM53 numeric config guard (end)

banner() {
    local label="${1:-start.sh}"
    printf '\n'
    printf '  \033[1;36m┌────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mDeepSeek-V4.1 Flash EXL3\033[0m  \033[2m%-10s\033[0m \033[1;36m│\033[0m\n' "$label"
    printf '  \033[1;36m└────────────────────────────────────────────┘\033[0m\n'
    printf '\n'
}

prepare_engram_src_dir() {
    python3 "$SCRIPT_DIR/scripts/prepare_engram_src.py" \
        --src "$ENGRAM_DIR" --dst "$ENGRAM_SRC" \
        || die "failed to build slim Engram src at $ENGRAM_SRC (need shards 47+48 under $ENGRAM_DIR)"
}

usage() { sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

count_model_shards() {
    find "$1" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

# ---------------------------- weight fetch ---------------------------------
# hf (huggingface_hub >= 0.34) or the older huggingface-cli. Resumable: both
# skip files already complete, so a killed download is restarted by re-running.
hf_cli() {
    if command -v hf >/dev/null 2>&1; then hf "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli "$@"
    else return 127
    fi
}

hf_fetch() {
    local repo="$1" dest="$2" label="$3"; shift 3
    hf_cli download "$repo" "$@" --local-dir "$dest" --max-workers "${HF_MAX_WORKERS:-8}" \
        || die "download of ${label} from ${repo} failed (re-run ./start.sh to resume, or fetch by hand into ${dest})"
}

# Engram needs only shards 47+48 of the 48-shard original (~95 GiB each), the
# index, and config.json; the other 46 shards are never read. Do not pull the
# whole 476 GiB repo.
ENGRAM_FILES=(
    "model-00047-of-00048.safetensors"
    "model-00048-of-00048.safetensors"
    "model.safetensors.index.json"
    "config.json"
)

engram_complete() {
    local f
    for f in "${ENGRAM_FILES[@]}"; do
        [ -f "$ENGRAM_DIR/$f" ] || return 1
    done
    return 0
}

fetch_weights() {
    local need_model=0 need_engram=0 have
    have="$(count_model_shards "$MODEL_HOST")"
    [ "${have:-0}" -ge "$EXPECTED_SHARDS" ] && [ -f "$MODEL_HOST/config.json" ] || need_model=1
    engram_complete || need_engram=1
    [ "$need_model" = "0" ] && [ "$need_engram" = "0" ] && return 0

    if [ "${AUTO_DOWNLOAD:-1}" != "1" ]; then
        warn "AUTO_DOWNLOAD=0 — not fetching; preflight will report what is missing"
        return 0
    fi
    command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1 \
        || die "need the Hugging Face CLI to fetch weights: pip install -U 'huggingface_hub[hf_transfer]' (or set AUTO_DOWNLOAD=0 and place them by hand)"
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

    if [ "$need_model" = "1" ]; then
        log "fetching EXL3 weights from ${HF_MODEL_REPO} into ${MODEL_HOST} (~197 GiB, resumable)"
        mkdir -p "$MODEL_HOST"
        hf_fetch "$HF_MODEL_REPO" "$MODEL_HOST" "EXL3 weights"
    fi
    if [ "$need_engram" = "1" ]; then
        log "fetching Engram shards 47+48 from ${HF_ENGRAM_REPO} into ${ENGRAM_DIR} (~190 GiB, resumable)"
        mkdir -p "$ENGRAM_DIR"
        local inc=() f
        for f in "${ENGRAM_FILES[@]}"; do inc+=(--include "$f"); done
        hf_fetch "$HF_ENGRAM_REPO" "$ENGRAM_DIR" "Engram tables" "${inc[@]}"
    fi
}

check_port_free() {
    local port="$1" name="$2"
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
        die "${name}=${port} is already bound on the head — stop the other server or pick a free port"
    fi
}

trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------

# --------------------------- image stamp -----------------------------------
overlay_recipe_hash() {
    (
        cd "$SCRIPT_DIR" || exit 1
        {
            printf '%s\n' "Dockerfile"
            find overlay files tests \
                -type f \
                ! -path '*/__pycache__/*' \
                ! -name '*.pyc' \
                ! -name '*.so' \
                2>/dev/null
        } | LC_ALL=C sort | xargs -d '\n' -r sha256sum | sha256sum | awk '{print $1}'
    )
}

image_recipe_stamp() {
    local stamp
    stamp="$(docker image inspect -f '{{ index .Config.Labels "dsv41.recipe.stamp" }}' "$IMAGE" 2>/dev/null || true)"
    case "$stamp" in
        ""|"<no value>"|"<nil>") printf '' ;;
        *) printf '%s' "$stamp" ;;
    esac
}

build_image() {
    local stamp
    stamp="$(overlay_recipe_hash)"
    log "building ${IMAGE} (DeepSeek-V4.1 EXL3 + SM120 ext) stamp=${stamp:0:12} (log: $LOGDIR/build-sm120.log) ..."
    docker build --build-arg "DSV41_RECIPE_STAMP=$stamp" -t "$IMAGE" "$SCRIPT_DIR" \
        >"$LOGDIR/build-sm120.log" 2>&1 \
        || { tail -n 40 "$LOGDIR/build-sm120.log" >&2; die "docker build of $IMAGE failed"; }
}

# Non-fatal: ensure_image falls back to building from this checkout when the

# ------------------------ inner container script ---------------------------
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[dsv41-exl3] $*"; }

SITE="$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).resolve().parent)')"
export GLM53_DETOKENIZER_PY="$SITE/v1/engine/detokenizer.py"
export GLM53_SCHEDULER_PY="$SITE/v1/core/sched/scheduler.py"
export GLM53_SPINWAIT_TARGET="$SITE/distributed/device_communicators/shm_broadcast.py"
export GLM53_XGRAMMAR_BACKEND_PY="$SITE/v1/structured_output/backend_xgrammar.py"
export GLM53_XGRAMMAR_MANAGER_PY="$SITE/v1/structured_output/__init__.py"
if [ -f /opt/dsv41/exl3.py ]; then
    cp /opt/dsv41/exl3.py "$SITE/model_executor/layers/quantization/exl3.py"
    say "installed runtime EXL3 overlay"
fi
say "MemAvailable=$(awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo) GiB host_expert_layers=${DSV41_EXL3_HOST_EXPERT_LAYERS:-none} engram: cache=${DSV41_CACHE_GIB:-?}GiB resident_scales=${DSV41_RESIDENT_SCALES:-default} io_threads=${DSV41_IO_THREADS:-default}"

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --prefix-cache-retention-interval "${PREFIX_CACHE_RETENTION_INTERVAL}"
    --distributed-executor-backend mp
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41
    --reasoning-parser deepseek_v41
    --enable-auto-tool-choice
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-enable-flashinfer-autotune
    --hf-overrides "{\"engram_table_dir\":\"${ENGRAM_MOUNT}\"}"
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
# Pin the KV pool; memory profiling is skipped entirely.
[ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
# 64-token KV blocks: the SM12x DeepGEMM paged indexer kernel accepts 32/64
# states per block only (patch_sm120_block64.py lets the backends take 64).
[ -n "${KV_BLOCK_SIZE:-}" ] && ARGS+=(--block-size "${KV_BLOCK_SIZE}")
[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ] && ARGS+=(--disable-custom-all-reduce)
if [ "${SPEC_METHOD:-dspark}" = "dspark" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
print(json.dumps({"method":"dspark","num_speculative_tokens":int(os.environ.get("DSPARK_TOKENS","5"))},separators=(",",":")))')")
elif [ "${SPEC_METHOD:-dspark}" = "none" ]; then
    :
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    ARGS+=(--mm-encoder-tp-mode data)
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} skip-mm-profiling=${SKIP_MM_PROFILING:-1}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
[ -d "${ENGRAM_MOUNT}" ] || { say "FATAL: ${ENGRAM_MOUNT} missing"; exit 1; }
for p in /opt/dsv41/patch_suppress_stops_in_reasoning.py \
         /opt/dsv41/patch_scheduler_decode_floor.py \
         /opt/dsv41/patch_xgrammar_termination.py \
         /opt/dsv41/patch_spinwait.py \
         /opt/dsv41/patch_responses_content_types.py \
         /opt/dsv41/patch_exl3_packed_names.py \
         /opt/dsv41/patch_exl3_lm_head.py \
         /opt/dsv41/patch_engram_secondary.py \
         /opt/dsv41/patch_engram_file.py \
         /opt/dsv41/patch_h2d_stage.py \
         /opt/dsv41/patch_sm120_block64.py \
         /opt/dsv41/patch_memory_log.py; do
    if [ -f "$p" ]; then
        python3 "$p" || say "WARN: $p returned $?"
    fi
done
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF
}


trap 'warn "interrupted — the container keeps running ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
gpu_query() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits; }

check_memory_headroom() {
    local json gpu_gib host_gib need_vram need_host
    json=$(python3 "$WEIGHT_BUDGET_HOST" --model "$MODEL_HOST" --tp "$TP" \
        --host-expert-layers "$DSV41_EXL3_HOST_EXPERT_LAYERS" --json 2>/dev/null || true)
    if [ -z "$json" ]; then
        warn "weight budget unavailable ($WEIGHT_BUDGET_HOST) — skipping headroom check"
        return 0
    fi
    gpu_gib=$(printf '%s' "$json" | python3 -c 'import json,sys; print("%.1f" % (json.load(sys.stdin)["per_rank_bytes"]/2**30))')
    host_gib=$(printf '%s' "$json" | python3 -c 'import json,sys; print("%.1f" % (json.load(sys.stdin)["host_per_rank_bytes"]/2**30))')
    need_vram=$(awk -v w="$gpu_gib" -v m="$DSV41_BOOT_MARGIN_GIB" 'BEGIN { printf "%.1f", w + m }')
    log "budget per GPU: ${gpu_gib} GiB weights + ${DSV41_BOOT_MARGIN_GIB} GiB margin = ${need_vram} GiB; host experts ${host_gib} GiB/rank pinned (layers ${DSV41_EXL3_HOST_EXPERT_LAYERS:-none})"
    local i=0 free bad=0
    while read -r free; do
        free=$(awk -v m="$free" 'BEGIN { printf "%.1f", m / 1024 }')
        log "  GPU${i}: ${free} GiB free"
        awk -v f="$free" -v n="$need_vram" 'BEGIN { exit !(f + 0 >= n + 0) }' || bad=1
        i=$((i + 1))
    done < <(gpu_query memory.free)
    [ "$bad" = "0" ] || die "not enough free VRAM: offload more layers (DSV41_EXL3_HOST_EXPERT_LAYERS, e.g. 32-39), set LANGUAGE_MODEL_ONLY=1, or stop other GPU processes"
    # Pinned memory cannot be swapped: two ranks of host experts + resident
    # Engram scales (~6 GiB) + ~20 GiB for vLLM itself must fit in MemAvailable.
    local need_host avail
    need_host=$(awk -v h="$host_gib" -v t="$TP" 'BEGIN { printf "%.1f", h * t + 26 }')
    avail=$(awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo)
    log "host RAM: need ~${need_host} GiB, MemAvailable ${avail} GiB"
    awk -v a="$avail" -v n="$need_host" 'BEGIN { exit !(a + 0 >= n + 0) }' \
        || die "not enough host RAM for the pinned host experts (${need_host} GiB)"
}

preflight() {
    command -v docker >/dev/null 2>&1 || die "docker not found"
    command -v curl >/dev/null 2>&1 || die "curl not found"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (NVIDIA driver)"
    docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon"
    docker info 2>/dev/null | grep -qi nvidia \
        || warn "docker reports no nvidia runtime — install nvidia-container-toolkit if --gpus fails"
    [ "$(uname -m)" = "x86_64" ] || warn "host is $(uname -m); this fork targets x86_64"

    local n cc
    n=$(gpu_query name | wc -l | tr -d ' ')
    [ "$n" -ge "$TP" ] || die "need ${TP} GPUs, nvidia-smi sees ${n}"
    gpu_query name,memory.total,compute_cap | sed 's/^/  /'
    cc=$(gpu_query compute_cap | sort -u | tr '\n' ' ')
    case "$cc" in
        "12.0 ") ;;
        *) warn "compute capability '${cc}' — this image is built for sm_120a (RTX PRO 6000 Blackwell)" ;;
    esac
    [ "$TP" = "2" ] || warn "TP=${TP} — this recipe is sized for TP=2"
    if [ "$GPUS" = "all" ] && [ "$n" -gt "$TP" ]; then
        warn "GPUS=all exposes ${n} GPUs; set GPUS='\"device=0,1\"' to pick two"
    fi

    check_port_free "$PORT" PORT
    [ -f "$EXL3_OVERLAY_HOST" ] || die "$EXL3_OVERLAY_HOST missing"
    [ -f "$KMAP_HOST" ] || die "$KMAP_HOST missing"
    [ -f "$CHAT_TEMPLATE_HOST" ] || die "$CHAT_TEMPLATE_HOST missing"
    fetch_weights

    [ -f "$MODEL_HOST/config.json" ] || die "EXL3 checkpoint missing config.json at $MODEL_HOST"
    local have
    have="$(count_model_shards "$MODEL_HOST")"
    [ "${have:-0}" -ge "$EXPECTED_SHARDS" ] \
        || die "EXL3 checkpoint at $MODEL_HOST has ${have:-0}/$EXPECTED_SHARDS shards"
    engram_complete || die "Engram shards 47+48 / index / config missing under ENGRAM_DIR=$ENGRAM_DIR"

    check_memory_headroom
    log "preflight OK ($(hostname))"
}

# ------------------------------ image --------------------------------------
ensure_image() {
    mkdir -p "$LOGDIR"
    local wanted have=""
    wanted="$(overlay_recipe_hash)"
    docker image inspect "$IMAGE" >/dev/null 2>&1 && have="$(image_recipe_stamp)"
    if [ "${BUILD:-0}" = "1" ]; then
        build_image
    elif [ -z "$have" ] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        log "no ${IMAGE} yet — building from this checkout (first build compiles exllamav3 for sm_120a)"
        build_image
    elif [ "$have" != "$wanted" ]; then
        if [ "${SKIP_BUILD:-0}" = "1" ]; then
            warn "SKIP_BUILD=1 — keeping ${IMAGE} (stamp ${have:0:12} != repo ${wanted:0:12})"
        else
            log "image recipe ${have:0:12} != repo ${wanted:0:12} — rebuilding"
            build_image
        fi
    fi
    if [ "${SKIP_OVERLAY_VERIFY:-0}" != "1" ]; then
        log "GPU EXL3 self-check on ${IMAGE} (log: $LOGDIR/overlay-verify.log) ..."
        docker run --rm --gpus "$GPUS" -e EXL3_SELFCHECK_GPU=1 \
            --entrypoint bash "$IMAGE" -c "python3 /opt/dsv41/test_exl3_overlay.py && python3 /opt/dsv41/test_engram_dequant.py && python3 /opt/dsv41/test_host_experts.py" \
            >"$LOGDIR/overlay-verify.log" 2>&1 \
            || { tail -n 80 "$LOGDIR/overlay-verify.log" >&2; die "EXL3 overlay GPU self-check failed"; }
        log "overlay verify OK"
    fi
}

# ------------------------------- pack --------------------------------------
pack_engram() {
    preflight
    ensure_image
    prepare_engram_src_dir
    mkdir -p "$PACKED_DIR"
    local r
    for r in $(seq 0 $((TP - 1))); do
        log "packing Engram rank ${r} of TP=${TP} into ${PACKED_DIR} ..."
        docker run --rm -v "$ENGRAM_SRC:/models:ro" -v "$ENGRAM_DIR:$ENGRAM_DIR:ro" -v "$PACKED_DIR:/engram" \
            --entrypoint python3 "$IMAGE" \
            /opt/dsv41/pack_engram.py --model /models --rank "$r" --tp "$TP" --out /engram
    done
    log "packed Engram shards written; ./start.sh mounts them at ${PACKED_MOUNT} when present"
}

# ------------------------------- launch ------------------------------------
launch() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    mkdir -p "$CACHE_ROOT" "$TRITON_HOST_CACHE" "$TILELANG_HOST_CACHE" "$LOGDIR"

    local -a packed=()
    if [ -f "$PACKED_DIR/engram-l1-r0of${TP}.bin" ]; then
        packed=(-v "$PACKED_DIR:$PACKED_MOUNT:ro" -e "DSV41_PACKED_DIR=$PACKED_MOUNT")
        log "packed Engram: $PACKED_DIR -> $PACKED_MOUNT"
    else
        log "no TP=${TP} packed Engram at $PACKED_DIR (optional ./start.sh pack)"
    fi

    local -a ov=()
    local f
    for f in patch_suppress_stops_in_reasoning.py patch_scheduler_decode_floor.py \
             patch_xgrammar_termination.py patch_spinwait.py patch_responses_content_types.py \
             patch_exl3_packed_names.py patch_exl3_lm_head.py patch_engram_secondary.py \
             patch_engram_file.py engram_file_backend.py engram_layout.py patch_memory_log.py \
             patch_h2d_stage.py patch_sm120_block64.py; do
        ov+=(-v "$SCRIPT_DIR/overlay/$f:/opt/dsv41/$f:ro")
    done

    local -a env=()
    local v
    for v in SERVED_MODEL_NAME PORT TP QUANTIZATION MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS \
             MAX_NUM_BATCHED_TOKENS LONG_PREFILL_TOKEN_THRESHOLD KV_CACHE_DTYPE SPEC_METHOD DSPARK_TOKENS \
             PREFIX_CACHE_RETENTION_INTERVAL LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING LIMIT_MM CHAT_TEMPLATE \
             ENFORCE_EAGER EXL3_FUSED_MOE EXL3_MOE_ROW_TILE EXL3_TEMP_ROWS_FUSED EXL3_FAT_SORTED \
             EXL3_FAT_BATCHED EXL3_FAT_KERNEL EXL3_FAT_GROUPED MODEL_DIR ENGRAM_MOUNT EXTRA_ARGS \
             DSV41_CACHE_GIB DSV41_RESIDENT_SCALES DSV41_IO_THREADS DSV41_CACHE_WAYS DSV41_STATS_SECONDS \
             KV_CACHE_MEMORY_BYTES KV_BLOCK_SIZE DSV41_SKIP_MIXED_WARMUP DSV41_DROP_PAGE_CACHE \
             DSV41_MEM_SOURCE DSV41_PREFILL_EMPTY_CACHE_TOKENS DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB \
             DSV41_PREFILL_END_EMPTY_CACHE VLLM_SPARSE_INDEXER_MAX_LOGITS_MB DSV41_INDEXER_PREFILL_FACTOR \
             DSV41_EXL3_HOST_EXPERT_LAYERS DSV41_EXL3_SERIAL_STREAMS DSV41_ENGRAM_DISABLE \
             VLLM_DISABLE_SHARED_EXPERTS_STREAM DISABLE_CUSTOM_ALL_REDUCE CUDA_LAUNCH_BLOCKING \
             NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_P2P_DISABLE \
             GLM53_SUPPRESS_STOPS_IN_REASONING GLM53_MIXED_PREFILL_CHUNK GLM53_INDEXER_WORKSPACE \
             GLM53_SPINWAIT_MS VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS TORCH_CUDA_ARCH_LIST \
             FLASHINFER_CUDA_ARCH_LIST PYTORCH_CUDA_ALLOC_CONF TRITON_CACHE_DIR TILELANG_CACHE_DIR \
             VLLM_API_KEY; do
        env+=(-e "$v=${!v:-}")
    done

    log "starting ${CONTAINER} (GPUs ${GPUS}, TP=${TP}) ..."
    # --ulimit memlock: the pinned host experts are cudaHostAlloc'd (~17 GiB/rank).
    docker run -d --name "$CONTAINER" \
        --gpus "$GPUS" --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --cap-add SYS_PTRACE --oom-score-adj 1000 \
        -v "$MODEL_HOST:/model:ro" \
        -v "$ENGRAM_SRC:/engram-src:ro" \
        -v "$ENGRAM_DIR:$ENGRAM_DIR:ro" \
        -v "$CACHE_ROOT:/root/.cache/vllm" \
        -v "$TRITON_HOST_CACHE:/root/.triton/cache" \
        -v "$TILELANG_HOST_CACHE:/root/.tilelang/cache" \
        -v "$INNER_SCRIPT:/start.sh:ro" \
        -v "$CHAT_TEMPLATE_HOST:$CHAT_TEMPLATE:ro" \
        -v "$EXL3_OVERLAY_HOST:/opt/dsv41/exl3.py:ro" \
        -v "$KMAP_HOST:/opt/dsv41/exl3_k_map.json:ro" \
        "${ov[@]}" "${packed[@]}" "${env[@]}" \
        -e NCCL_CUMEM_ENABLE=0 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e VLLM_CACHE_ROOT=/root/.cache/vllm \
        -e DSV41_SOURCE="$ENGRAM_MOUNT" \
        -e DSV41_EXL3_K_MAP=/opt/dsv41/exl3_k_map.json \
        -e GLM53_DENSE_FP8=off \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e "VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT" \
        -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$CG_ESTIMATE" \
        -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
        --entrypoint bash "$IMAGE" /start.sh >/dev/null
    log "container up — ${CONTAINER}"
}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (timeout ${READY_TIMEOUT}s) — Ctrl-C detaches, the server keeps running"
    local logpid=""
    docker logs -f --tail 0 "$CONTAINER" 2>&1 &
    logpid=$!
    trap '[ -n "$logpid" ] && kill "$logpid" 2>/dev/null; warn "interrupted — container keeps running"; exit 130' INT

    local elapsed=0 healthy=0 last=0 quiet=0 cur
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        cur=$(docker logs "$CONTAINER" 2>&1 | grep -avc "shm_broadcast.py" || true)
        if [ "$cur" -eq "$last" ]; then quiet=$((quiet + 10)); else quiet=0; last=$cur; fi
        if [ "$quiet" -ge "$DSV41_HANG_SECONDS" ]; then
            warn "no log line for ${quiet}s — dumping Python stacks (py-spy) to $LOGDIR/hang-pyspy.txt"
            docker exec "$CONTAINER" bash /opt/dsv41/pyspy_dump.sh >"$LOGDIR/hang-pyspy.txt" 2>&1 || true
            break
        fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
            warn "container exited during startup"; break
        fi
        if [ $((elapsed % 60)) -eq 0 ]; then
            log "t+${elapsed}s VRAM free (MiB): $(gpu_query memory.free | tr '\n' ' ') host MemAvailable=$(awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo) GiB"
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done
    kill "$logpid" 2>/dev/null || true
    wait "$logpid" 2>/dev/null || true
    trap 'warn "interrupted"; exit 130' INT
    [ "$healthy" = "1" ] && log "health check passed after ${elapsed}s"
    [ "$healthy" = "1" ]
}

post_ready_warmup() {
    if [ "${DSV41_BOOT_SHAPE_WARMUP:-1}" = "0" ]; then
        log "boot shape warmup skipped (DSV41_BOOT_SHAPE_WARMUP=0)"
        return 0
    fi
    log "post-ready DSpark/sampler warmup (nonfatal) ..."
    GLM53_WARMUP_MAX_CONCURRENCY="$MAX_NUM_SEQS" \
    GLM53_WARMUP_REQ_TIMEOUT="$DSV41_WARMUP_REQ_TIMEOUT" \
    GLM53_WARMUP_DSPARK_K="${DSPARK_TOKENS:-3}" \
    GLM53_WARMUP_TRITON_CACHE_DIR="$TRITON_HOST_CACHE" \
    GLM53_WARMUP_BEARER="${VLLM_API_KEY:-}" \
        bash "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" "http://127.0.0.1:${PORT}" "$SERVED_MODEL_NAME" \
        || warn "boot shape warmup incomplete — uncovered shapes may JIT mid-serve"
}

on_ready() {
    log "======================================================================"
    log "DeepSeek-V4.1-Flash EXL3 is UP (TP=${TP}, 2x RTX PRO 6000)"
    log "  endpoint   : http://127.0.0.1:${PORT}/v1   model: ${SERVED_MODEL_NAME}"
    log "  host experts: layers ${DSV41_EXL3_HOST_EXPERT_LAYERS:-none} (pinned RAM, PCIe zero-copy)"
    log "  VRAM free  : $(gpu_query memory.free | tr '\n' ' ')MiB"
    log "  quick test :"
    log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \\"
    log "      -d '{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"17*19=\"}],\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
    log "  manage     : ./start.sh status | logs | stop"
    log "======================================================================"
    if [ "${TAIL:-0}" = "1" ]; then
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER" || true
    fi
}

stop() {
    log "stopping ${CONTAINER} ..."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || log "  (not running)"
}

status() {
    docker ps -a --filter "name=${CONTAINER}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "API: not responding"
    fi
    gpu_query index,memory.used,memory.total,utilization.gpu | sed 's/^/  GPU /'
}

logs() { trap '' INT; docker logs -f --tail 100 "$CONTAINER" || true; }

start() {
    preflight
    ensure_image
    prepare_engram_src_dir
    write_inner_scripts
    log "config: image=${IMAGE} tp=${TP} spec=${SPEC_METHOD} k=${DSPARK_TOKENS} max-len=${MAX_MODEL_LEN} mnbt=${MAX_NUM_BATCHED_TOKENS} kv-pool=${KV_CACHE_MEMORY_BYTES:-profiled} lm-only=${LANGUAGE_MODEL_ONLY} host-experts=${DSV41_EXL3_HOST_EXPERT_LAYERS:-none}"
    launch
    if wait_for_health; then
        post_ready_warmup
        on_ready
        return
    fi
    docker logs "$CONTAINER" >"$LOGDIR/serve.log" 2>&1 || true
    echo "---- last 60 lines ($LOGDIR/serve.log) ----"
    tail -n 60 "$LOGDIR/serve.log" || true
    die "server did not become healthy — full log in $LOGDIR/serve.log"
}

main() {
    local cmd="${1:-start}"
    case "$cmd" in start|restart) validate_numeric_config ;; esac
    banner "${cmd}"
    case "$cmd" in
        start)   start ;;
        pack)    pack_engram ;;
        stop)    stop ;;
        restart) stop; start ;;
        status)  status ;;
        logs)    logs ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
