#!/usr/bin/env bash
#SBATCH --job-name=munin_gec
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=munin_%j.out
#SBATCH --error=munin_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Munin-only dependencies; does not overwrite ~/.local or the container packages.
CONTAINER="/ceph/container/vllm-openai_latest.sif"
VENV="$PWD/.munin_venv"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DISABLED=true
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" outputs

# Only create/install on first Munin run. Don't run two Munin setup jobs simultaneously.
if [[ ! -f "$VENV/.munin_ready_4576" ]]; then
    echo "Creating isolated Munin Python environment: $VENV"
    if [[ ! -x "$VENV/bin/python" ]]; then
        singularity exec --nv "$CONTAINER" python3 -m venv --system-site-packages "$VENV"
    fi
    singularity exec --nv "$CONTAINER" "$VENV/bin/python" -m pip install \
        'transformers==4.57.6' 'accelerate>=1.4,<2' 'sentencepiece' 'protobuf'
    singularity exec --nv "$CONTAINER" "$VENV/bin/python" -c \
        'import torch, transformers, accelerate; from transformers import ApertusConfig; print("Torch:", torch.__version__, "Transformers:", transformers.__version__, "Accelerate:", accelerate.__version__)'
    touch "$VENV/.munin_ready_4576"
fi

# The pre-existing run_gec.py stays unchanged. Use the exact same evaluation input/prompt.
MODEL="danish-foundation-models/munin-apertus-8b"
MODEL_REVISION="${MODEL_REVISION:-main}"
LIMIT="${LIMIT:-5}"                   # Smoke-test by default; set LIMIT=0 for all 3328
OUTPUT="${OUTPUT:-outputs/munin_apertus_8b_smoke.jsonl}"

singularity exec --nv "$CONTAINER" "$VENV/bin/python" -c \
    'import transformers; from transformers import ApertusConfig; print("Munin env Transformers:", transformers.__version__)'

singularity exec --nv "$CONTAINER" "$VENV/bin/python" run_gec.py \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --allow-floating-revision \
    --input data/pooled/evaluation_inputs.csv \
    --prompt-file gec_prompt.txt \
    --output "$OUTPUT" \
    --quantization none \
    --max-new-tokens 384 \
    --max-input-tokens 2048 \
    --attention eager \
    --warmup 2 \
    --limit "$LIMIT"
