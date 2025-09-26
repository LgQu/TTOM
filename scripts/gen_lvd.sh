#!/bin/bash
cd /path/to/TTOM
PY_SCRIPT="generation/gen_benchmarks.py"


if [[ -n "${GROUP_ID:-}" ]]; then
    CSV_PATH=${CSV_PATH:?Must set CSV_PATH}
    mapfile -t PIDS < <(
        awk -F, -v gid="$GROUP_ID" 'NR>1 && $1==gid {print $2}' "$CSV_PATH"
    )
elif [[ -n "${PID_START:-}" && -n "${PID_END:-}" ]]; then
    PIDS=($(seq "${PID_START:?}" "${PID_END:?}"))
elif [[ -n "${PIDLIST:-}" ]]; then
    PIDLIST_CLEAN=$(echo "$PIDLIST" | tr ',' ' ')
    PIDS=($PIDLIST_CLEAN)
else
    echo "[ERROR] must set either PIDLIST or (PID_START & PID_END)" >&2
    exit 1
fi
CACHE_TYPE=${CACHE_TYPE:?Must set CACHE_TYPE}

JSD=0
COM=1
MAX_G=5
MAX_I=8
MIN_L=0.05

Time=$(date "+%m%d_%H%M")
OUTPUT_LOG="logs/wan_lvd/${CACHE_TYPE}/${PBS_JOBID}_G${MAX_G}_LS${MAX_LS}_lora[${TARGET_M}]_${Time}.log"
ERROR_LOG="logs/wan_lvd/${CACHE_TYPE}/${PBS_JOBID}_G${MAX_G}_LS${MAX_LS}_lora[${TARGET_M}]_${Time}.err"
mkdir -p "$(dirname "$OUTPUT_LOG")"

printf '===== [%s] | Running PIDs: %s =====\n' "$CACHE_TYPE" "${PIDS[*]}" | tee -a "$OUTPUT_LOG"

for pid in "${PIDS[@]}"; do
    echo "********** Generating video for pid=$pid **********" | tee -a "$OUTPUT_LOG"
    /path/to/your/conda/envs/ttom/bin/python -u "$PY_SCRIPT" \
        --cache_type $CACHE_TYPE --pid $pid --jsd_loss_weight $JSD --com_loss_weight $COM --save_lora_weight True \
        --min_loss_value $MIN_L --max_guidance_step $MAX_G --max_lora_step $MAX_LS --max_iter $MAX_I --target_modules "$TARGET_M" --prefix "lvd" --strat_id $STRAT \
        >> "$OUTPUT_LOG" 2>> "$ERROR_LOG"
    echo "********** Finished pid=$pid **********" | tee -a "$OUTPUT_LOG"
    sleep 10
done

echo "All jobs submitted." | tee -a "$OUTPUT_LOG"