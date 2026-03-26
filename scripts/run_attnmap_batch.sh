#!/bin/bash
# Batch attention map generation for CogVideoX
#
# Usage:
#   bash scripts/run_attnmap_batch.sh
#
# Configure the variables below before running.

CONDA_PYTHON="python"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# === Configuration ===
CACHE_TYPE="4_motion_binding_gpt-4o"
MODEL_PATH="THUDM/CogVideoX-5b"
SEED=42
GUIDANCE_TYPE="none"
NUM_FRAMES=49
OUTPUT_DIR="data/attn_maps/cogvideox"

START_PID=0
END_PID=199

# === Run ===
for PID in $(seq $START_PID $END_PID); do
    echo "============================================"
    echo "Generating attention maps for pid=$PID"
    echo "============================================"

    $CONDA_PYTHON generation/get_attnmap.py \
        --pid "$PID" \
        --cache_type "$CACHE_TYPE" \
        --model_path "$MODEL_PATH" \
        --seed $SEED \
        --guidance_type "$GUIDANCE_TYPE" \
        --num_frames $NUM_FRAMES \
        --output_dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        echo "[Error] Failed for pid=$PID, continuing..."
    fi
done

echo "Batch attention map generation complete."
