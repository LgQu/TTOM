#!/bin/bash
# Batch video generation with CogVideoX + TTOM guidance
#
# Usage:
#   bash scripts/run_benchmarks_batch.sh
#
# Configure the variables below before running.

CONDA_PYTHON="python"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# === Configuration ===
CACHE_TYPE="4_motion_binding_gpt-4o"
MODEL_PATH="THUDM/CogVideoX-5b"
GUIDANCE_TYPE="main"
GUIDANCE_ATTN_STRATEGY="middle_2"
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=10
MAX_ITER=5
LR=5e-3
SEED=42
NUM_FRAMES=49
PREFIX="benchmark"

START_PID=0
END_PID=199

# === Run ===
for PID in $(seq $START_PID $END_PID); do
    echo "============================================"
    echo "Generating video for pid=$PID"
    echo "============================================"

    $CONDA_PYTHON generation/gen_benchmarks.py \
        --pid "$PID" \
        --cache_type "$CACHE_TYPE" \
        --model_path "$MODEL_PATH" \
        --guidance_type "$GUIDANCE_TYPE" \
        --guidance_attn_strategy "$GUIDANCE_ATTN_STRATEGY" \
        --guidance_start_step $GUIDANCE_START_STEP \
        --guidance_end_step $GUIDANCE_END_STEP \
        --max_iter $MAX_ITER \
        --lr $LR \
        --set_amf_loss \
        --seed $SEED \
        --num_frames $NUM_FRAMES \
        --skip_existed_prompt \
        --prefix "$PREFIX"

    if [ $? -ne 0 ]; then
        echo "[Error] Failed for pid=$PID, continuing..."
    fi
done

echo "Batch generation complete."
