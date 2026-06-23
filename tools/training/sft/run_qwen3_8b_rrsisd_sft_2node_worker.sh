#!/usr/bin/env bash
# Qwen3-VL-8B RRSIS-D SFT worker wrapper.
#
# The two-node SFT implementation is shared with the 4B baseline.  Keep this
# thin wrapper so 8B experiment launchers have an explicit worker entry point
# and default model path while still using the tested rendezvous/training code.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-8B-Instruct}"
export MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH}"
exec bash "${script_dir}/run_qwen3_4b_rrsisd_sft_2node_worker.sh" "$@"
