#!/usr/bin/env bash
set -e

# Single sim ID per invocation (default 108 for manual/debug runs).
# For batch runs, submit_agama_exp_array.sh passes the sim ID per array task.
SIM_ID="${1:-108}"

dir_name="suite_${SIM_ID}"
mkdir -p logs
/usr/bin/time -v -o "logs/time_${SIM_ID}.log" python build_agama_expansion.py \
    "sheng" \
    "sheng_iso" \
    "$SIM_ID" \
    --coefs_freq=10 \
    --output_dir="agama_expansions/${dir_name}"
echo "Done computing agama expansion: $dir_name"

