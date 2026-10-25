#!/usr/bin/env bash
# download.sh — fetch the weights this recipe serves, without booting it.
#
# EXL3 weights come from MODEL / HF_MODEL_REPO (default Mia 2.9bpw, 39 shards,
# ~197 GiB, into ./model; another repo goes to ./models/<org>__<name>).
# A local MODEL directory is only verified, never downloaded into.
#   ./download.sh [--model <dir | hf-org/name>] [--revision <rev>]
# Engram tables are never quantized and never copied into the EXL3 tree: only
# shards 47+48 of the original 48-shard checkpoint (~95 GiB each), the index,
# and config.json are pulled from HF_ENGRAM_REPO into ENGRAM_DIR.
#
# Resumable — re-run after an interruption. ./start.sh runs the same fetch
# automatically, so this script is only for staging the download separately.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
_cli_model="" _cli_rev=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) _cli_model="$2"; shift 2 ;;
        --model=*) _cli_model="${1#--model=}"; shift ;;
        --revision) _cli_rev="$2"; shift 2 ;;
        --revision=*) _cli_rev="${1#--revision=}"; shift ;;
        *) echo "usage: $0 [--model <dir | hf-org/name>] [--revision <rev>]" >&2; exit 2 ;;
    esac
done
_cli_model="${_cli_model:-${MODEL:-}}"
[ -f "$SCRIPT_DIR/.env" ] || cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
if [ -n "$_cli_model" ]; then MODEL="$_cli_model"; MODEL_HOST=""; HF_MODEL_REPO=""; fi
[ -n "$_cli_rev" ] && HF_MODEL_REVISION="$_cli_rev"
[ -n "${KMAP_HOST:-}" ] && _DSV41_KMAP_PINNED=1
KMAP_HOST="${KMAP_HOST:-$SCRIPT_DIR/files/exl3_k_map.json}"
# shellcheck source=scripts/resolve_model.sh
source "$SCRIPT_DIR/scripts/resolve_model.sh"
resolve_model || exit 2

ENGRAM_DIR="${ENGRAM_DIR:-$SCRIPT_DIR/engram-src}"
HF_ENGRAM_REPO="${HF_ENGRAM_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# With `set -euo pipefail`, probing a missing destination with find(1) exits
# before the first download. A clean checkout has neither directory yet.
mkdir -p "$ENGRAM_DIR"
if [ "$MODEL_SOURCE" = "local" ]; then
    [ -d "$MODEL_HOST" ] || { echo "MODEL directory $MODEL_HOST does not exist" >&2; exit 1; }
else
    mkdir -p "$MODEL_HOST"
fi

hf_cli() {
    if command -v hf >/dev/null 2>&1; then hf "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli "$@"
    else
        echo "need the Hugging Face CLI: pip install -U 'huggingface_hub[hf_transfer]'" >&2
        exit 1
    fi
}

have=$(find "$MODEL_HOST" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')
echo "EXL3    $MODEL_HOST  ${have:-0}/$EXPECTED_SHARDS shards"
if [ "${have:-0}" -lt "$EXPECTED_SHARDS" ] || [ ! -f "$MODEL_HOST/config.json" ] \
   || [ ! -f "$MODEL_HOST/model.safetensors.index.json" ]; then
    if [ "$MODEL_SOURCE" = "local" ]; then
        echo "local checkpoint $MODEL_HOST is incomplete (${have:-0}/$EXPECTED_SHARDS shards); not downloading into a local path" >&2
        exit 1
    fi
    rev=()
    [ -n "${HF_MODEL_REVISION:-}" ] && rev=(--revision "$HF_MODEL_REVISION")
    echo "fetching $HF_MODEL_REPO${HF_MODEL_REVISION:+@$HF_MODEL_REVISION} -> $MODEL_HOST"
    mkdir -p "$MODEL_HOST"
    hf_cli download "$HF_MODEL_REPO" ${rev[@]+"${rev[@]}"} --local-dir "$MODEL_HOST" --max-workers "${HF_MAX_WORKERS:-8}"
    resolve_model_files
fi

ENGRAM_FILES=(
    "model-00047-of-00048.safetensors"
    "model-00048-of-00048.safetensors"
    "model.safetensors.index.json"
    "config.json"
)
missing=()
for f in "${ENGRAM_FILES[@]}"; do
    [ -f "$ENGRAM_DIR/$f" ] || missing+=("$f")
done
echo "Engram  $ENGRAM_DIR  $(( ${#ENGRAM_FILES[@]} - ${#missing[@]} ))/${#ENGRAM_FILES[@]} files"
if [ "${#missing[@]}" -gt 0 ]; then
    echo "fetching $HF_ENGRAM_REPO (shards 47+48 + index/config only) -> $ENGRAM_DIR"
    mkdir -p "$ENGRAM_DIR"
    inc=()
    for f in "${ENGRAM_FILES[@]}"; do inc+=(--include "$f"); done
    hf_cli download "$HF_ENGRAM_REPO" "${inc[@]}" --local-dir "$ENGRAM_DIR" --max-workers "${HF_MAX_WORKERS:-8}"
fi

have=$(find "$MODEL_HOST" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')
[ "${have:-0}" -ge "$EXPECTED_SHARDS" ] || { echo "still ${have:-0}/$EXPECTED_SHARDS EXL3 shards" >&2; exit 1; }
for f in "${ENGRAM_FILES[@]}"; do
    [ -f "$ENGRAM_DIR/$f" ] || { echo "missing $ENGRAM_DIR/$f" >&2; exit 1; }
done
check_model_config
echo "complete. ./start.sh serves both trees from this host."
if [ -n "${MODEL:-}" ]; then
    echo "serve it with: ./start.sh --model '${MODEL}'   (or set MODEL in .env)"
fi
