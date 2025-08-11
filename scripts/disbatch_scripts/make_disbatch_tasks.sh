#!/bin/bash
# make_com_tasks.sh
# Generate disBatch task files for each simulation folder in MWLMC/

# === Configuration ===

OUT_DIR="/mnt/home/nico/ceph/projects/XMC-Atlas/scripts/outputs"
SCRIPT="/mnt/home/nico/ceph/projects/XMC-Atlas/scripts/process_com_snapshot.py"
BASE_DIR="/mnt/home/nico/ceph/gadget_runs/MWLMC"
TASKS_DIR="/mnt/home/nico/ceph/projects/XMC-Atlas/scripts/disbatch_scripts"


for SIM_DIR in "$BASE_DIR"/*/; do
    SIM_NAME=$(basename "$SIM_DIR")
    SNAP_DIR="${SIM_DIR}out"

    # Only proceed if out/ exists
    if [[ ! -d "$SNAP_DIR" ]]; then
        echo "Skipping $SIM_NAME — no 'out/' directory found."
        continue
    fi

    TASKFILE="$TASKS_DIR/com_tasks_${SIM_NAME}.dbtask"
    rm -f "$TASKFILE"

    echo "Generating task file: $TASKFILE"

    # Loop through snapshot files in out/
    for SNAPFILE in "$SNAP_DIR"/*.hdf5; do
        if [[ -f "$SNAPFILE" ]]; then
            echo "python $SCRIPT_PATH $SNAPFILE ./outputs/$SIM_NAME" >> "$TASKFILE"
        fi
    done

    if [[ ! -s "$TASKFILE" ]]; then
        echo "Warning: No snapshots found in $SNAP_DIR"
        rm -f "$TASKFILE"
    fi
done

echo "All task files created in: $TASKS_DIR"

