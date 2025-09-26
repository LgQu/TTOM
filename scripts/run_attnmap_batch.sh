#!/bin/bash

# Script to generate attention maps for PIDs 0-199 with GPU monitoring
# Usage: bash run_attnmap_batch.sh
PID_START=0
PID_END=199
# Use conda environment Python directly instead of conda activate
CONDA_PYTHON="/path/to/your/conda/envs/ttom/bin/python"
echo "Starting batch attention map generation for PIDs ${PID_START}-${PID_END}"
echo "=================================================="

# Set the base directory
BASE_DIR="/path/to/TTOM"
SCRIPT_PATH="$BASE_DIR/generation/get_attnmap.py"

# Create logs directory
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

# Set log file names with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/attnmap_batch_${TIMESTAMP}.log"
ERR_FILE="$LOG_DIR/attnmap_batch_${TIMESTAMP}.err"
SUMMARY_FILE="$LOG_DIR/attnmap_batch_${TIMESTAMP}_summary.txt"

# GPU monitoring settings (adjust these as needed)
GPU_THRESHOLD=15  # GPU utilization threshold (percentage) - GPUs below this will be considered available
CHECK_INTERVAL=30  # Check interval in seconds - how often to check GPU status
MAX_WAIT_TIME=7200
GPU_RANGE="0-3"  # GPU IDs to check (0-3 means check GPUs 0, 1, 2, 3)

echo "Log file: $LOG_FILE"
echo "Error file: $ERR_FILE"
echo "Summary file: $SUMMARY_FILE"
echo "GPU threshold: ${GPU_THRESHOLD}%"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Change to the project directory
cd "$BASE_DIR"

# Counter for tracking progress
success_count=0
error_count=0

# Function to check GPU utilization
check_gpu_utilization() {
    local gpu_id=$1
    # Get GPU utilization percentage using nvidia-smi
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id=$gpu_id 2>/dev/null | head -1
}

# Function to find available GPU
find_available_gpu() {
    local start_time=$(date +%s)
    local current_time
    local elapsed_time
    
    while true; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        # Check if we've exceeded maximum wait time
        if [ $elapsed_time -gt $MAX_WAIT_TIME ]; then
            echo "ERROR: Maximum wait time (${MAX_WAIT_TIME}s) exceeded. No GPU available."
            return -1
        fi
        
        # Check each GPU (0-3)
        for gpu_id in {0..3}; do
            local utilization=$(check_gpu_utilization $gpu_id)
            
            # Check if nvidia-smi command was successful and utilization is below threshold
            if [ $? -eq 0 ] && [ -n "$utilization" ] && [ "$utilization" -lt $GPU_THRESHOLD ]; then
                echo "GPU $gpu_id is available (utilization: ${utilization}%)"
                return $gpu_id
            fi
        done
        
        # If no GPU is available, wait and check again
        echo "All GPUs are busy. Waiting ${CHECK_INTERVAL}s before checking again..."
        echo "Elapsed wait time: ${elapsed_time}s / ${MAX_WAIT_TIME}s"
        sleep $CHECK_INTERVAL
    done
}

# Function to display GPU status
display_gpu_status() {
    echo "Current GPU status:"
    echo "==================="
    for gpu_id in {0..3}; do
        local utilization=$(check_gpu_utilization $gpu_id)
        if [ $? -eq 0 ] && [ -n "$utilization" ]; then
            echo "GPU $gpu_id: ${utilization}% utilization"
        else
            echo "GPU $gpu_id: Unable to check status"
        fi
    done
    echo ""
}

# Display initial GPU status
display_gpu_status

# Loop through PIDs from PID_START to PID_END
for pid in $(seq $PID_START $PID_END); do
    echo ""
    echo "Processing PID: $pid"
    echo "-------------------"
    
    # Find available GPU
    echo "Looking for available GPU..."
    find_available_gpu
    gpu_id=$?
    
    # Check if GPU allocation was successful
    if [ $gpu_id -eq -1 ]; then
        echo "✗ Failed to allocate GPU for PID: $pid"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: PID $pid - No GPU available" >> "$LOG_FILE"
        echo "PID $pid failed - No GPU available after ${MAX_WAIT_TIME}s wait" >> "$ERR_FILE"
        ((error_count++))
        continue
    fi
    
    echo "Using GPU: $gpu_id"
    
    # Log the start time for this PID
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting PID: $pid on GPU: $gpu_id" >> "$LOG_FILE"
    
    # Run the attention map generation script and capture both stdout and stderr
    CUDA_VISIBLE_DEVICES=$gpu_id "$CONDA_PYTHON" "$SCRIPT_PATH" --pid "$pid" >> "$LOG_FILE" 2>> "$ERR_FILE"

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed PID: $pid on GPU: $gpu_id"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: PID $pid on GPU $gpu_id" >> "$LOG_FILE"
        ((success_count++))
    else
        echo "✗ Error processing PID: $pid on GPU: $gpu_id"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: PID $pid on GPU $gpu_id" >> "$LOG_FILE"
        echo "PID $pid failed on GPU $gpu_id - check $ERR_FILE for details" >> "$ERR_FILE"
        ((error_count++))
    fi
    
    # Display GPU status after processing
    echo "GPU status after processing PID $pid:"
    display_gpu_status
    
    # Optional: Add a small delay between runs to prevent system overload
    # sleep 1
done

echo ""
echo "=================================================="
echo "Batch processing completed!"
echo "Successful: $success_count"
echo "Errors: $error_count"
echo "Total processed: $((success_count + error_count))"

# Write summary to file
echo "Batch Processing Summary" > "$SUMMARY_FILE"
echo "========================" >> "$SUMMARY_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "Successful: $success_count" >> "$SUMMARY_FILE"
echo "Errors: $error_count" >> "$SUMMARY_FILE"
echo "Total processed: $((success_count + error_count))" >> "$SUMMARY_FILE"
echo "Success rate: $(( success_count * 100 / (success_count + error_count) ))%" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Log files:" >> "$SUMMARY_FILE"
echo "- Main log: $LOG_FILE" >> "$SUMMARY_FILE"
echo "- Error log: $ERR_FILE" >> "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo "Main log: $LOG_FILE"
echo "Error log: $ERR_FILE"
