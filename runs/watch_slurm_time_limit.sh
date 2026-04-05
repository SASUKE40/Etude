#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash runs/watch_slurm_time_limit.sh <log_file_or_prefix> [sbatch_script]

Wait for a SLURM log to emit the time-limit cancellation line:
  slurmstepd: error: *** JOB <id> ON <node> CANCELLED AT <timestamp> DUE TO TIME LIMIT ***

When the line appears, submit a replacement batch job with sbatch and keep
watching the resubmitted job chain until interrupted.

The first argument may be either:
  - a specific log file, such as runs/d24-h100-5677184.log
  - a known log prefix, such as d24-h100 or runs/d24-h100

When given a known prefix, the watcher picks the newest matching log file by
modification time automatically. If no matching log exists yet, it waits for
the first one to appear. If an explicit log path is given and the file does not
exist yet, it waits for that file.

If sbatch_script is omitted, the script will infer one for known log prefixes:
  runs/d24-h100-<jobid>.log -> runs/d24_h100_resume.slurm
  runs/d24-h200-<jobid>.log -> runs/d24_h200_resume.slurm
  runs/d24-rust-<jobid>.log -> runs/d24_rust_resume.slurm

Environment:
  POLL_SECONDS   Poll interval while waiting for new log content. Default: 15
  STATE_DIR      Directory for "already resubmitted" markers. Default: runs/.resubmitted
EOF
}

normalize_log_prefix() {
    local input="$1"
    local prefix

    prefix="${input#runs/}"
    if [[ "$prefix" == *.log ]]; then
        prefix="${prefix%.log}"
        prefix="${prefix%-*}"
    fi

    case "$prefix" in
        d24-h100|d24-h200|d24-rust)
            ;;
        *)
            echo "ERROR: Could not infer a log file from '$input'." >&2
            echo "Pass an explicit log file or a known prefix such as d24-h100." >&2
            return 1
            ;;
    esac

    printf '%s\n' "$prefix"
}

resolve_latest_log() {
    local input="$1"
    local prefix
    local matches=()

    prefix="$(normalize_log_prefix "$input")" || return 1

    shopt -s nullglob
    matches=(runs/"${prefix}"-*.log)
    shopt -u nullglob

    if [ ${#matches[@]} -eq 0 ]; then
        echo "ERROR: No log files found for prefix '$prefix'." >&2
        return 1
    fi

    ls -t "${matches[@]}" | head -1
}

infer_sbatch_script() {
    local input="$1"
    local prefix

    prefix="$(normalize_log_prefix "$input")" || return 1

    case "$prefix" in
        d24-h100) echo "runs/d24_h100_resume.slurm" ;;
        d24-h200) echo "runs/d24_h200_resume.slurm" ;;
        d24-rust) echo "runs/d24_rust_resume.slurm" ;;
        *)
            echo "ERROR: Could not infer sbatch script from '$input'." >&2
            echo "Pass the sbatch script explicitly." >&2
            return 1
            ;;
    esac
}

infer_output_template() {
    local sbatch_script="$1"
    local output_template

    output_template="$(
        awk '
            /^#SBATCH[[:space:]]+--output=/ {
                sub(/^#SBATCH[[:space:]]+--output=/, "", $0)
                print
                exit
            }
            /^#SBATCH[[:space:]]+-o[[:space:]]+/ {
                sub(/^#SBATCH[[:space:]]+-o[[:space:]]+/, "", $0)
                print
                exit
            }
        ' "$sbatch_script"
    )"

    if [ -z "$output_template" ]; then
        echo "ERROR: Could not infer the log output path from '$sbatch_script'." >&2
        return 1
    fi

    output_template="${output_template%\"}"
    output_template="${output_template#\"}"
    output_template="${output_template%\'}"
    output_template="${output_template#\'}"
    printf '%s\n' "$output_template"
}

render_log_path() {
    local output_template="$1"
    local job_id="$2"
    local log_path="$output_template"

    log_path="${log_path//%j/$job_id}"
    log_path="${log_path//%A/$job_id}"
    printf '%s\n' "$log_path"
}

read_submitted_job_id() {
    local stamp_file="$1"
    local stamp_value

    stamp_value="$(head -1 "$stamp_file" 2>/dev/null || true)"
    if [[ "$stamp_value" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$stamp_value"
    fi
}

submit_or_reuse() {
    local job_id="$1"
    local sbatch_script="$2"
    local state_dir="$3"
    local stamp_file="$state_dir/job_${job_id}.submitted"
    local submitted_job_id

    mkdir -p "$state_dir"

    if [ -f "$stamp_file" ]; then
        submitted_job_id="$(read_submitted_job_id "$stamp_file")"
        if [ -n "$submitted_job_id" ]; then
            echo "Job $job_id was already resubmitted as job $submitted_job_id; skipping duplicate sbatch." >&2
            printf '%s\n' "$submitted_job_id"
            return 0
        fi
        echo "Job $job_id was already resubmitted; skipping duplicate sbatch." >&2
        return 0
    fi

    echo "Detected time-limit cancellation for job $job_id. Submitting $sbatch_script" >&2
    local sbatch_output
    sbatch_output="$(sbatch "$sbatch_script")"
    echo "$sbatch_output" >&2

    if [[ "$sbatch_output" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
        submitted_job_id="${BASH_REMATCH[1]}"
        printf '%s\n' "$submitted_job_id" > "$stamp_file"
        printf '%s\n' "$submitted_job_id"
        return 0
    fi

    echo "ERROR: Could not parse the new job id from sbatch output: $sbatch_output" >&2
    return 1
}

wait_for_next_log() {
    local current_log_file="$1"
    local log_selector="$2"
    local poll_seconds="$3"
    local output_template="$4"
    local next_job_id="${5:-}"
    local next_log_file=""

    if [ -n "$next_job_id" ] && [ -n "$output_template" ]; then
        next_log_file="$(render_log_path "$output_template" "$next_job_id")"
        if [[ "$next_log_file" != *%* ]]; then
            echo "Waiting for next log file $next_log_file" >&2
            while [ ! -f "$next_log_file" ]; do
                sleep "$poll_seconds"
            done
            printf '%s\n' "$next_log_file"
            return 0
        fi
    fi

    while true; do
        next_log_file="$(resolve_latest_log "$log_selector" 2>/dev/null || true)"
        if [ -n "$next_log_file" ] && [ "$next_log_file" != "$current_log_file" ]; then
            printf '%s\n' "$next_log_file"
            return 0
        fi
        sleep "$poll_seconds"
    done
}

wait_for_existing_log() {
    local log_file="$1"
    local poll_seconds="$2"

    echo "Waiting for log file $log_file" >&2
    while [ ! -f "$log_file" ]; do
        sleep "$poll_seconds"
    done
    printf '%s\n' "$log_file"
}

wait_for_first_log_for_prefix() {
    local prefix="$1"
    local poll_seconds="$2"
    local log_file=""

    echo "No log files found yet for prefix '$prefix'. Waiting for the first matching log." >&2
    while true; do
        log_file="$(resolve_latest_log "$prefix" 2>/dev/null || true)"
        if [ -n "$log_file" ]; then
            printf '%s\n' "$log_file"
            return 0
        fi
        sleep "$poll_seconds"
    done
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

    local log_input="$1"
    local log_file="$log_input"
    local sbatch_script="${2:-}"
    local poll_seconds="${POLL_SECONDS:-15}"
    local state_dir="${STATE_DIR:-runs/.resubmitted}"
    local pattern='slurmstepd: error: \*\*\* JOB ([0-9]+) ON .* DUE TO TIME LIMIT \*\*\*'
    local output_template
    local prefix=""

    if [ -z "$sbatch_script" ]; then
        sbatch_script="$(infer_sbatch_script "$log_input")"
    fi

    if [ ! -f "$sbatch_script" ]; then
        echo "ERROR: sbatch script '$sbatch_script' does not exist." >&2
        exit 1
    fi

    if [ -f "$log_input" ]; then
        log_file="$log_input"
    elif [[ "$log_input" == *.log ]]; then
        log_file="$(wait_for_existing_log "$log_input" "$poll_seconds")"
    else
        prefix="$(normalize_log_prefix "$log_input")"
        log_file="$(resolve_latest_log "$prefix" 2>/dev/null || true)"
        if [ -z "$log_file" ]; then
            log_file="$(wait_for_first_log_for_prefix "$prefix" "$poll_seconds")"
        fi
    fi

    output_template="$(infer_output_template "$sbatch_script")"

    echo "Watching $log_file"
    echo "Will submit $sbatch_script after a SLURM time-limit cancellation and continue with each replacement job."

    while true; do
        local match
        match="$(grep -Eo "$pattern" "$log_file" | tail -1 || true)"
        if [ -n "$match" ]; then
            if [[ "$match" =~ JOB[[:space:]]+([0-9]+)[[:space:]]+ON ]]; then
                local next_job_id=""
                next_job_id="$(submit_or_reuse "${BASH_REMATCH[1]}" "$sbatch_script" "$state_dir")"
                log_file="$(wait_for_next_log "$log_file" "$log_input" "$poll_seconds" "$output_template" "$next_job_id")"
                echo "Watching $log_file"
                continue
            fi
            echo "ERROR: Matched cancellation line but could not parse job id." >&2
            exit 1
        fi
        sleep "$poll_seconds"
    done
}

main "$@"
