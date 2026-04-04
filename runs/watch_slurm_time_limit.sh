#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash runs/watch_slurm_time_limit.sh <log_file> [sbatch_script]

Wait for a SLURM log to emit the time-limit cancellation line:
  slurmstepd: error: *** JOB <id> ON <node> CANCELLED AT <timestamp> DUE TO TIME LIMIT ***

When the line appears, submit a replacement batch job with sbatch and exit.

If sbatch_script is omitted, the script will infer one for known log prefixes:
  runs/d24-h100-<jobid>.log -> runs/d24_h100_resume.slurm
  runs/d24-h200-<jobid>.log -> runs/d24_h200_resume.slurm

Environment:
  POLL_SECONDS   Poll interval while waiting for new log content. Default: 15
  STATE_DIR      Directory for "already resubmitted" markers. Default: runs/.resubmitted
EOF
}

infer_sbatch_script() {
    local log_file="$1"
    local base
    base="$(basename "$log_file")"
    case "$base" in
        d24-h100-*.log) echo "runs/d24_h100_resume.slurm" ;;
        d24-h200-*.log) echo "runs/d24_h200_resume.slurm" ;;
        *)
            echo "ERROR: Could not infer sbatch script from log file '$log_file'." >&2
            echo "Pass the sbatch script explicitly." >&2
            return 1
            ;;
    esac
}

submit_once() {
    local job_id="$1"
    local sbatch_script="$2"
    local state_dir="$3"
    local stamp_file="$state_dir/job_${job_id}.submitted"

    mkdir -p "$state_dir"

    if [ -f "$stamp_file" ]; then
        echo "Job $job_id was already resubmitted; skipping duplicate sbatch."
        return 0
    fi

    echo "Detected time-limit cancellation for job $job_id. Submitting $sbatch_script"
    sbatch "$sbatch_script"
    date '+%Y-%m-%dT%H:%M:%S%z' > "$stamp_file"
}

main() {
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        usage
        exit 0
    fi

    if [ $# -lt 1 ] || [ $# -gt 2 ]; then
        usage >&2
        exit 1
    fi

    local log_file="$1"
    local sbatch_script="${2:-}"
    local poll_seconds="${POLL_SECONDS:-15}"
    local state_dir="${STATE_DIR:-runs/.resubmitted}"
    local pattern='slurmstepd: error: \*\*\* JOB ([0-9]+) ON .* DUE TO TIME LIMIT \*\*\*'

    if [ ! -f "$log_file" ]; then
        echo "ERROR: Log file '$log_file' does not exist." >&2
        exit 1
    fi

    if [ -z "$sbatch_script" ]; then
        sbatch_script="$(infer_sbatch_script "$log_file")"
    fi

    if [ ! -f "$sbatch_script" ]; then
        echo "ERROR: sbatch script '$sbatch_script' does not exist." >&2
        exit 1
    fi

    echo "Watching $log_file"
    echo "Will submit $sbatch_script after a SLURM time-limit cancellation."

    while true; do
        local match
        match="$(grep -Eo "$pattern" "$log_file" | tail -1 || true)"
        if [ -n "$match" ]; then
            if [[ "$match" =~ JOB[[:space:]]+([0-9]+)[[:space:]]+ON ]]; then
                submit_once "${BASH_REMATCH[1]}" "$sbatch_script" "$state_dir"
                exit 0
            fi
            echo "ERROR: Matched cancellation line but could not parse job id." >&2
            exit 1
        fi
        sleep "$poll_seconds"
    done
}

main "$@"
