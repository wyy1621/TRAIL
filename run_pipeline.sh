#!/usr/bin/env bash
# Run the complete TRAIL research pipeline in dependency order.
#
# Key parameters:
#   PYTHON_BIN: Python executable used for every stage (default: python).
#
# Outputs:
#   output/pipeline_state/*.done records successfully completed stages.
#   output/performance/*.json contains per-stage and aggregate performance data.
#
# Recommended for remote sessions:
#   tmux new-session -s trail 'bash run_pipeline.sh 2>&1 | tee -a pipeline.log'

set -Eeuo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly STATE_DIR="${PROJECT_ROOT}/output/pipeline_state"

mkdir -p "${STATE_DIR}"
cd "${PROJECT_ROOT}"

run_stage() {
    local stage_name="$1"
    local entrypoint="$2"
    local done_marker="${STATE_DIR}/${stage_name}.done"

    if [[ -f "${done_marker}" ]]; then
        printf '[pipeline] skip completed stage: %s\n' "${stage_name}"
        return
    fi

    printf '[pipeline] start stage: %s\n' "${stage_name}"
    "${PYTHON_BIN}" "${entrypoint}"
    date --iso-8601=seconds > "${done_marker}"
    printf '[pipeline] completed stage: %s\n' "${stage_name}"
}

run_stage "01_download_data" "download_data.py"
run_stage "02_build_dataset" "build_dataset.py"
run_stage "03_train_reconstructor" "train_reconstructor.py"
run_stage "04_train_detector" "train_detector.py"
run_stage "05_evaluate" "evaluate.py"

printf '[pipeline] all stages completed\n'
