#!/bin/bash
#SBATCH --job-name=dala_gec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DISABLED=true
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" outputs

CONTAINER="/ceph/container/vllm-openai_latest.sif"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-main}"
ATTENTION="${ATTENTION:-eager}"
WARMUP="${WARMUP:-2}"
ALLOW_FLOATING_REVISION="${ALLOW_FLOATING_REVISION:-0}"
EXTRA_ARGS=()
if [[ "$ALLOW_FLOATING_REVISION" == "1" ]]; then EXTRA_ARGS+=(--allow-floating-revision); fi
# Output name must be unique per model + configuration. Override for other models.
OUTPUT="${OUTPUT:-outputs/qwen25_3b_bf16.jsonl}"
LIMIT="${LIMIT:-0}"
QUANTIZATION="${QUANTIZATION:-none}"

# Inference only: do not install training dependencies (trl, peft etc.).
# Run this once per environment if dependencies are not already available:
# singularity exec --nv "$CONTAINER" python3 -m pip install --user \
#   'transformers==4.49.0' 'accelerate==1.4.0' 'sentencepiece' 'protobuf'

singularity exec --nv "$CONTAINER" python3 run_gec.py \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --input data/pooled/evaluation_inputs.csv \
  --prompt-file gec_prompt.txt \
  --output "$OUTPUT" \
  --quantization "$QUANTIZATION" \
  --max-new-tokens 384 \
  --max-input-tokens 2048 \
  --limit "$LIMIT" \
  --attention "$ATTENTION" \
  --warmup "$WARMUP" \
  "${EXTRA_ARGS[@]}"
