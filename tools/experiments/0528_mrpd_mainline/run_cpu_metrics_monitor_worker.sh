#!/usr/bin/env bash
# CPU-side periodic monitor/aggregator for the 0528 MRPD mainline experiments.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
env_root="${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
cd "$repo_root"

export PATH="${env_root}/bin:${PATH}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-${env_root}/bin/python}"
state_root="${STATE_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
interval="${INTERVAL_SEC:-120}"
max_ticks="${MAX_TICKS:-240}"
worker_log="${WORKER_LOG:-${state_root}/cpu_metrics_monitor_$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$state_root" "$(dirname "$worker_log")"

log() {
  echo "[$(date '+%F %T')][0528-cpu-monitor] $*" | tee -a "$worker_log"
}

acceptance_complete() {
  local status_json="${state_root}/0528_acceptance_status.json"
  [ -f "$status_json" ] || return 1
  "$python_bin" - "$status_json" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    sys.exit(1)
tasks = data.get("tasks", {})
required = (
    "pass8_gap_reduction",
    "route_mechanism_analysis",
    "selective_distillation_ablation",
)
sys.exit(0 if all(tasks.get(name, {}).get("complete") for name in required) else 1)
PY
}

parse_route_log() {
  local log_path="$1"
  local out_dir="$2"
  local name="$3"
  if [ -f "$log_path" ]; then
    "$python_bin" tools/experiments/0528_mrpd_mainline/parse_route_stats.py \
      --log "$log_path" \
      --out_dir "$out_dir" \
      --name "$name" >> "$worker_log" 2>&1 || true
  fi
}

aggregate_pass8_manifest() {
  local manifest="$1"
  local dir
  dir="$(dirname "$manifest")"
  if [ -f "$manifest" ]; then
    "$python_bin" tools/experiments/0528_mrpd_mainline/aggregate_pass8_gap.py \
      --manifest "$manifest" \
      --out_dir "$dir" >> "$worker_log" 2>&1 || true
  fi
}

compute_route_teacher_gap() {
  local nohint="$1"
  local teacher="$2"
  local out_dir="$3"
  local name="$4"
  if [ -f "$nohint" ] && [ -f "$teacher" ]; then
    "$python_bin" tools/experiments/0528_mrpd_mainline/route_teacher_gap.py \
      --student_nohint_per_sample "$nohint" \
      --teacher_hint_per_sample "$teacher" \
      --out_dir "$out_dir" \
      --name "$name" >> "$worker_log" 2>&1 || true
  fi
}

write_acceptance_status() {
  "$python_bin" - "$state_root" <<'PY'
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

state_root = Path(sys.argv[1])

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

pass8 = []
for manifest_path in sorted(state_root.glob("pass8_*/manifest.json")):
    manifest = load_json(manifest_path) or {}
    eval_root = Path(manifest.get("eval_root", ""))
    expected = []
    done = 0
    for model in manifest.get("models", []):
        for item in model.get("pass8", []):
            summary = Path(item.get("summary", ""))
            status = "done" if summary.exists() else "missing"
            if status == "done":
                done += 1
            expected.append({
                "model_id": model.get("id"),
                "hint_mode": item.get("hint_mode"),
                "summary": str(summary),
                "status": status,
            })
    pass8.append({
        "manifest": str(manifest_path),
        "eval_root": str(eval_root),
        "done": done,
        "expected": len(expected),
        "jobs": expected,
        "gap_summary": str(manifest_path.parent / "pass8_gap_summary.json"),
    })

routes = []
for route_status in sorted(state_root.glob("**/route_*/status.json")):
    status = load_json(route_status) or {}
    summary = load_json(status.get("summary_path", "")) if status.get("summary_path") else None
    routes.append({
        "status_path": str(route_status),
        "num_records": status.get("num_records"),
        "summary_path": status.get("summary_path"),
        "global": (summary or {}).get("global", {}),
    })

teacher_gaps = []
for gap_status in sorted(state_root.glob("**/route_teacher_gap*/status.json")):
    status = load_json(gap_status) or {}
    summary = load_json(status.get("summary_path", "")) if status.get("summary_path") else None
    teacher_gaps.append({
        "status_path": str(gap_status),
        "total_candidates": status.get("total_candidates"),
        "summary_path": status.get("summary_path"),
        "routes": (summary or {}).get("routes", {}),
    })

scope_runs = []
scope_eval_root = Path(os.environ.get(
    "SCOPE_EVAL_ROOT",
    "/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/scope_tests",
))
for tsv in sorted(state_root.glob("distill_scope_*/rjobs.tsv")):
    with open(tsv, errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            rjob, scope, group, log_path = parts[:4]
            tmux_name = parts[4] if len(parts) > 4 else ""
            run_name = parts[5] if len(parts) > 5 else ""
            log_text = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, errors="ignore") as lf:
                        tail = lf.readlines()[-200:]
                    log_text = "".join(tail)
                except Exception:
                    pass
            eval_summary = scope_eval_root / f"{scope}_{rjob}_test" / "summary.json"
            if (
                "run completed" in log_text
                or (
                    eval_summary.exists()
                    and any(token in log_text for token in (" latest_ckpt=", "last_model_checkpoint:", " completed"))
                    and "global_step/max_steps': '870/870'" in log_text
                )
            ):
                inferred = "completed"
            elif any(token in log_text for token in ("Traceback", "RuntimeError", "OutOfMemory", "ERROR")):
                inferred = "needs_check"
            elif os.path.exists(log_path):
                inferred = "running_or_starting"
            else:
                inferred = "submitted"
            eval_status_path = scope_eval_root / f"{scope}_{rjob}_test" / "status.json"
            eval_status = load_json(eval_status_path) if eval_status_path.exists() else None
            scope_runs.append({
                "rjob": rjob,
                "scope": scope,
                "group": group,
                "tmux": tmux_name,
                "run_name": run_name,
                "log_path": log_path,
                "inferred_status": inferred,
                "state_dir": str(tsv.parent),
                "eval_summary": str(eval_summary),
                "eval_status": (eval_status or {}).get("status") if eval_status else ("done" if eval_summary.exists() else "missing"),
            })

required_scopes = {"failed", "failed_ambiguous", "all"}
completed_scope_tests = {
    r["scope"]
    for r in scope_runs
    if r["scope"] in required_scopes
    and r["inferred_status"] == "completed"
    and Path(r["eval_summary"]).exists()
}

acceptance = {
    "updated_at": time.strftime("%F %T"),
    "tasks": {
        "pass8_gap_reduction": {
            "complete": bool(pass8) and all(item["done"] >= item["expected"] and item["expected"] >= 8 for item in pass8),
            "runs": pass8,
        },
        "route_mechanism_analysis": {
            "complete": (
                len(routes) >= 1
                and any((r.get("num_records") or 0) > 0 for r in routes)
                and len(teacher_gaps) >= 1
                and any((g.get("total_candidates") or 0) > 0 for g in teacher_gaps)
            ),
            "runs": routes,
            "teacher_gap_runs": teacher_gaps,
        },
        "selective_distillation_ablation": {
            "complete": required_scopes.issubset(completed_scope_tests),
            "required_scopes": sorted(required_scopes),
            "completed_scope_tests": sorted(completed_scope_tests),
            "runs": scope_runs,
        },
    },
}
out = state_root / "0528_acceptance_status.json"
out.write_text(json.dumps(acceptance, indent=2, ensure_ascii=False))

csv_path = state_root / "0528_scope_runs_status.csv"
with csv_path.open("w", newline="") as f:
    fieldnames = [
        "rjob", "scope", "group", "tmux", "run_name", "inferred_status",
        "eval_status", "log_path", "eval_summary", "state_dir"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in scope_runs:
        writer.writerow({key: row.get(key) for key in fieldnames})
print(json.dumps({"status": "done", "acceptance": str(out), "scope_csv": str(csv_path)}, ensure_ascii=False))
PY
}

log "start state_root=${state_root} interval=${interval} max_ticks=${max_ticks}"
tick=0
while [ "$tick" -lt "$max_ticks" ]; do
  tick=$((tick + 1))
  log "tick=${tick}"

  parse_route_log \
    "/data/codes/gui_grounding/data/logs/training/qwen3_4b_sft_to_opsd_gauss_e1.console.log" \
    "${state_root}/route_mrpd_metric_reference" \
    "mrpd_metric_reference"

  for tsv in "$state_root"/distill_scope_*/rjobs.tsv; do
    [ -f "$tsv" ] || continue
    while IFS='|' read -r rjob scope group log_path tmux_name run_name; do
      [ -n "${rjob:-}" ] || continue
      parse_route_log "$log_path" "$(dirname "$tsv")/route_${scope}" "$scope"
    done < "$tsv"
  done

  for manifest in "$state_root"/pass8_*/manifest.json; do
    [ -f "$manifest" ] || continue
    aggregate_pass8_manifest "$manifest"
  done

  compute_route_teacher_gap \
    "/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/pass8_0528p8/sft_none/per_sample.jsonl" \
    "/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/pass8_0528p8/sft_gaussian/per_sample.jsonl" \
    "${state_root}/route_teacher_gap_sft" \
    "sft_nohint_vs_gaussian_teacher"
  compute_route_teacher_gap \
    "/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/pass8_0528p8/mrpd_metric_none/per_sample.jsonl" \
    "/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/pass8_0528p8/mrpd_metric_gaussian/per_sample.jsonl" \
    "${state_root}/route_teacher_gap_mrpd_metric" \
    "mrpd_metric_nohint_vs_gaussian_teacher"

  STATE_ROOT="$state_root" write_acceptance_status >> "$worker_log" 2>&1 || true
  if acceptance_complete; then
    log "acceptance complete; CPU monitor exiting to release rjob resources"
    break
  fi
  sleep "$interval"
done

log "completed max_ticks=${max_ticks}"
