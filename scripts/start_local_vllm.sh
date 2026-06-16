#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.venv-vllm}"
VLLM_PYTHON="${VLLM_PYTHON:-3.12}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${VLLM_MODEL##*/}}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-vllm}"

if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  echo "Creating vLLM environment at $VLLM_ENV with Python $VLLM_PYTHON"
  uv venv --python "$VLLM_PYTHON" --seed "$VLLM_ENV"
  uv pip install --python "$VLLM_ENV/bin/python" --torch-backend "$UV_TORCH_BACKEND" vllm
fi

args=(
  serve "$VLLM_MODEL"
  --served-model-name "$VLLM_SERVED_MODEL_NAME"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --api-key "$VLLM_API_KEY"
  --dtype "$VLLM_DTYPE"
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --generation-config "$VLLM_GENERATION_CONFIG"
)

if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  args+=(--quantization "$VLLM_QUANTIZATION")
fi

if [[ -n "${VLLM_DOWNLOAD_DIR:-}" ]]; then
  args+=(--download-dir "$VLLM_DOWNLOAD_DIR")
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($VLLM_EXTRA_ARGS)
  args+=("${extra_args[@]}")
fi

echo "Starting vLLM:"
echo "  model: $VLLM_MODEL"
echo "  served name: $VLLM_SERVED_MODEL_NAME"
echo "  endpoint: http://$VLLM_HOST:$VLLM_PORT/v1"
echo "  tensor parallel size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "  max model length: $VLLM_MAX_MODEL_LEN"

exec "$VLLM_ENV/bin/vllm" "${args[@]}"
