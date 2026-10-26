# shellcheck shell=bash
# resolve_model.sh — sourced by start.sh and download.sh after .env.
#
# Picks the EXL3 checkpoint from one setting:
#   MODEL=/data/models/my-dsv41-exl3     local directory: served as-is, never downloaded
#   MODEL=my-hf-user/my-dsv41-exl3       Hugging Face repo: fetched into MODEL_HOST
#   (unset)                              legacy MODEL_HOST / HF_MODEL_REPO, default Mia 2.9bpw
# Optional: HF_MODEL_REVISION (branch/tag/commit), HF_TOKEN (private repos; read by the hf CLI).
#
# Sets: MODEL_HOST (absolute), HF_MODEL_REPO, MODEL_SOURCE (local|hf), and
#       EXPECTED_SHARDS / KMAP_HOST derived from the checkpoint unless set explicitly.

_DSV41_DEFAULT_REPO="Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw"

_dsv41_abs() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        "~"/*) printf '%s' "$HOME/${1#\~/}" ;;
        *) printf '%s' "$SCRIPT_DIR/${1#./}" ;;
    esac
}

resolve_model() {
    local m="${MODEL:-}"
    if [ -n "$m" ]; then
        case "$m" in
            /*|./*|../*|"~"/*) MODEL_SOURCE=local ;;
            *)
                if [ -d "$(_dsv41_abs "$m")" ]; then
                    MODEL_SOURCE=local
                elif [[ "$m" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
                    MODEL_SOURCE=hf
                else
                    echo "MODEL='$m' is neither an existing directory nor a Hugging Face repo id (org/name)" >&2
                    return 2
                fi
                ;;
        esac
        if [ "$MODEL_SOURCE" = "local" ]; then
            MODEL_HOST="$(_dsv41_abs "$m")"
        else
            HF_MODEL_REPO="$m"
            # A different repo must not land in (and mix with) the default ./model tree.
            if [ -z "${MODEL_HOST:-}" ]; then
                if [ "$m" = "$_DSV41_DEFAULT_REPO" ]; then
                    MODEL_HOST="$SCRIPT_DIR/model"
                else
                    MODEL_HOST="$SCRIPT_DIR/models/${m//\//__}"
                fi
            fi
        fi
    else
        MODEL_SOURCE=hf
        HF_MODEL_REPO="${HF_MODEL_REPO:-$_DSV41_DEFAULT_REPO}"
        MODEL_HOST="${MODEL_HOST:-$SCRIPT_DIR/model}"
    fi
    HF_MODEL_REPO="${HF_MODEL_REPO:-$_DSV41_DEFAULT_REPO}"
    MODEL_HOST="$(_dsv41_abs "$MODEL_HOST")"
    resolve_model_files
}

# Re-run after a download: shard count and K-map come from the checkpoint itself.
resolve_model_files() {
    local idx="$MODEL_HOST/model.safetensors.index.json" n
    if [ -z "${_DSV41_SHARDS_PINNED:-}" ]; then
        if [ -f "$idx" ]; then
            n=$(python3 -c 'import json,sys; print(len(set(json.load(open(sys.argv[1]))["weight_map"].values())))' "$idx" 2>/dev/null || echo "")
            [ -n "$n" ] && EXPECTED_SHARDS="$n"
        fi
        EXPECTED_SHARDS="${EXPECTED_SHARDS:-39}"
    fi
    # A checkpoint can ship its own per-tensor K map (exl3_k_map.json next to config.json).
    if [ -z "${_DSV41_KMAP_PINNED:-}" ] && [ -f "$MODEL_HOST/exl3_k_map.json" ]; then
        KMAP_HOST="$MODEL_HOST/exl3_k_map.json"
    fi
}

# --- Engram tables ---------------------------------------------------------
# Only shards 47+48 of a 48-shard tree hold the n-gram tables; with the index
# and config.json that is all the backend reads. The row format is detected
# from the shard headers (overlay/engram_format.py):
#   int4_fp16s32  diffbot's re-encode, 144 B/row, ~103 GiB for both layers (1017)
#   fp8_e8m0      stock DeepSeek, 264 B/row, ~190 GiB (0909 or 1017 release)
ENGRAM_FILES=(
    "model-00047-of-00048.safetensors"
    "model-00048-of-00048.safetensors"
    "model.safetensors.index.json"
    "config.json"
)
_DSV41_STOCK_ENGRAM_REPO="deepseek-ai/DeepSeek-V4.1-Flash"
_DSV41_INT4_ENGRAM_REPO="diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000"

# Format of the tables under $1: fp8_e8m0 | int4_fp16s32 | none.
engram_dir_format() {
    python3 "$SCRIPT_DIR/overlay/engram_format.py" "$1" 2>/dev/null || printf 'none'
}

# ENGRAM_FORMAT (int4 default | fp8 | auto) -> canonical ENGRAM_FORMAT plus the
# HF_ENGRAM_REPO / ENGRAM_DIR defaults for it. Explicit values are kept.
resolve_engram() {
    ENGRAM_FORMAT="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import engram_format
try:
    print(engram_format.resolve_name(sys.argv[2]))
except ValueError as exc:
    print(f"ENGRAM_FORMAT: {exc}", file=sys.stderr); sys.exit(2)' "$SCRIPT_DIR/overlay" "${ENGRAM_FORMAT:-int4}")" || return 2
    case "$ENGRAM_FORMAT" in
        int4_fp16s32)
            HF_ENGRAM_REPO="${HF_ENGRAM_REPO:-$_DSV41_INT4_ENGRAM_REPO}"
            ENGRAM_DIR="${ENGRAM_DIR:-$SCRIPT_DIR/engram-int4}"
            ;;
        *)
            HF_ENGRAM_REPO="${HF_ENGRAM_REPO:-$_DSV41_STOCK_ENGRAM_REPO}"
            ENGRAM_DIR="${ENGRAM_DIR:-$SCRIPT_DIR/engram-src}"
            ;;
    esac
    ENGRAM_DIR="$(_dsv41_abs "$ENGRAM_DIR")"
}

engram_complete() {
    local f
    for f in "${ENGRAM_FILES[@]}"; do
        [ -f "$ENGRAM_DIR/$f" ] || return 1
    done
    return 0
}

# The tables under $1 must exist and match ENGRAM_FORMAT (unless auto). Prints
# the format found.
check_engram_format() {
    local have
    have="$(engram_dir_format "$1")"
    if [ "$have" = "none" ]; then
        echo "no Engram tables (layers.{1,14}.engram.embed.*) under $1" >&2
        return 1
    fi
    if [ "$ENGRAM_FORMAT" != "auto" ] && [ "$have" != "$ENGRAM_FORMAT" ]; then
        echo "Engram tables under $1 are $have, but ENGRAM_FORMAT=$ENGRAM_FORMAT; set ENGRAM_FORMAT=auto to serve them as they are, or point ENGRAM_DIR at $ENGRAM_FORMAT tables" >&2
        return 1
    fi
    printf '%s' "$have"
}

# Warn (never fail) when config.json is not the EXL3 layout this overlay loads.
check_model_config() {
    [ -f "$MODEL_HOST/config.json" ] || return 0
    python3 - "$MODEL_HOST/config.json" "$KMAP_HOST" <<'PY' || true
import json, sys
cfg = json.load(open(sys.argv[1]))
q = cfg.get("quantization_config") or {}
km = json.load(open(sys.argv[2])) if sys.argv[2] else {}
warn = []
if cfg.get("architectures") != ["DeepseekV41ForCausalLM"]:
    warn.append(f"architectures={cfg.get('architectures')} (expected DeepseekV41ForCausalLM)")
if q.get("quant_method") != "exl3":
    warn.append(f"quant_method={q.get('quant_method')} (expected exl3)")
if q.get("codebook") and km.get("codebook") and q["codebook"] != km["codebook"]:
    warn.append(f"codebook {q['codebook']} != K-map codebook {km['codebook']}")
print(f"checkpoint: exl3 avg_bits={q.get('avg_bits')} codebook={q.get('codebook')} head_bits={q.get('head_bits')}")
for w in warn:
    print(f"WARNING: {w}", file=sys.stderr)
if q.get("avg_bits") not in (None, 2.9):
    print("WARNING: not the 2.9 bpw layout files/exl3_k_map.json describes; ship an exl3_k_map.json "
          "with the checkpoint (or set KMAP_HOST) if its per-tensor K differs", file=sys.stderr)
PY
}
