#!/usr/bin/env bash
# Mandatory kill -9 continuity test on the tiny fixture.
#
# It accelerates the checkpoint cadence (every 10 steps) so the test finishes
# in minutes; production uses 6-minute/60-step cadence.  The invariants proved
# here are the ones that matter: only complete commits become visible, LATEST
# never points at a partial directory, resume restores the exact cursor, and
# the number of steps lost is bounded by one checkpoint interval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/t4llm-chaos.XXXXXX")"
export T4LLM_HUB_ROOT="$WORK/hub"
export T4LLM_RUNS_ROOT="$WORK/runs"
export T4LLM_ALLOW_SMALL_BATCH=1
export T4LLM_ALLOW_SYNTHETIC=1
export T4LLM_ALLOW_NON_T4="${T4LLM_ALLOW_NON_T4:-0}"
KEEP="${T4LLM_CHAOS_KEEP:-0}"

log() { printf '[chaos] %s\n' "$*"; }
cleanup() {
  if [[ "$KEEP" == "1" ]]; then
    log "artifacts kept at $WORK"
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT

cd "$ROOT"
log "fixture at $WORK"
python -m t4llm fixture \
  --root "$WORK/fixture" \
  --seq-len 64 \
  --samples 4000 \
  --max-steps 1000 \
  --ckpt-steps 10 \
  --micro-batch 1 \
  --grad-accum 1 >"$WORK/fixture.json"
CFG="$WORK/fixture/config.yaml"
DATA="$WORK/fixture/data"

read_status() {
  python -m t4llm status --config "$CFG" --json >"$WORK/status.json" 2>"$WORK/status.err" || true
  python - "$WORK" <<'PY'
import json, pathlib, sys
w = pathlib.Path(sys.argv[1])
st = json.loads((w / "status.json").read_text()) if (w / "status.json").stat().st_size else {}
run_id = st.get("run_id") or "unknown"
step = 0
state = w / "runs" / run_id / "state.json"
if state.is_file():
    try:
        step = int(json.loads(state.read_text()).get("global_step", 0))
    except Exception:
        pass
latest = st.get("latest_checkpoint")
print(json.dumps({"run_id": run_id, "state_step": step, "latest": latest,
                  "checkpoints": st.get("checkpoint_count", 0)}))
PY
}

assert_consistent() {
  local tag="$1"
  local js
  js="$(read_status)"
  python - "$WORK" "$tag" "$js" <<'PY'
import json, pathlib, sys
from t4llm.state import read_manifest, validate_checkpoint_dir

w = pathlib.Path(sys.argv[1])
tag = sys.argv[2]
st = json.loads(sys.argv[3])
hub = w / "hub"
visible = sorted(p.name for p in hub.iterdir() if p.is_dir() and p.name.startswith("ckpt-"))
partial = []
for name in visible:
    try:
        validate_checkpoint_dir(hub / name)
    except Exception as exc:
        partial.append(f"{name}: {exc}")
if partial:
    raise SystemExit(f"[{tag}] incomplete visible checkpoint(s): {partial}")
if st["latest"] is not None:
    latest = hub / f"ckpt-{int(st['latest']):08d}"
    if not (latest / "manifest.json").is_file():
        raise SystemExit(f"[{tag}] LATEST points at missing manifest: {latest}")
if st["state_step"] < (st["latest"] or 0):
    raise SystemExit(f"[{tag}] state step {st['state_step']} < LATEST {st['latest']}")
print(f"[{tag}] run={st['run_id']} state_step={st['state_step']} latest={st['latest']} visible={len(visible)}")
PY
}

latest_step() {
  python - "$WORK/hub" <<'PY'
import json, pathlib, sys
hub = pathlib.Path(sys.argv[1])
try:
    text = (hub / "LATEST").read_text().strip()
    step = int(text.split("-")[-1])
except Exception:
    step = 0
print(step)
PY
}

wait_for_checkpoint() {
  local pid="$1" min="$2" i step
  for i in $(seq 1 80); do
    step="$(latest_step)"
    if [[ "$step" -ge "$min" ]]; then
      printf '%s' "$step"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      printf '%s' "$step"
      return 1
    fi
    sleep 0.5
  done
  printf '%s' "$step"
  return 1
}

# Three crash cycles.  Each kill happens after at least one brand-new checkpoint
# is durable, then the next launch must resume from that commit.
prior=0
for i in 1 2 3; do
  log "cycle $i: launching"
  python -m t4llm train --config "$CFG" --data-dir "$DATA" --allow-synthetic \
    --time-budget 120s --max-steps 1000 >"$WORK/train-$i.log" 2>&1 &
  pid=$!
  if ! reached="$(wait_for_checkpoint "$pid" $((prior + 1)))"; then
    log "cycle $i: process exited or timed out before a new checkpoint (latest=$reached)"
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    tail -40 "$WORK/train-$i.log" || true
    exit 1
  fi
  prior="$reached"
  log "cycle $i: checkpoint $reached durable; killing -9"
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  assert_consistent "after-kill-$i"
done

# Resume normally and finish a short horizon so a clean exit is also exercised.
BASE_STEP="$(read_status | python -c 'import json,sys; print(json.load(sys.stdin)["state_step"])')"
TARGET=$((BASE_STEP + 20))
log "final continuous run to step $TARGET"
python -m t4llm train --config "$CFG" --data-dir "$DATA" --allow-synthetic \
  --time-budget 120s --max-steps "$TARGET" >"$WORK/train-final.log" 2>&1
assert_consistent "final"

python - "$WORK" "$CFG" "$DATA" "$TARGET" <<'PY'
import json, pathlib, subprocess, sys
w = pathlib.Path(sys.argv[1])
cfg, data, target = sys.argv[2], sys.argv[3], int(sys.argv[4])
status = json.loads((w / "status.json").read_text())
run_id = status["run_id"]
run = w / "runs" / run_id
events = [json.loads(x) for x in (run / "resume_events.jsonl").read_text().splitlines() if x.strip()]
if len(events) < 3:
    raise SystemExit(f"expected >=3 resume events, got {len(events)}")
lost = [int(e.get("lost_steps", 0)) for e in events]
if any(x > 10 for x in lost):
    raise SystemExit(f"lost more than one checkpoint interval: {lost}")
state = json.loads((run / "state.json").read_text())
if int(state["global_step"]) < target:
    raise SystemExit(f"final step {state['global_step']} < target {target}")
latest = subprocess.check_output(
    ["python", "-m", "t4llm", "resume", "--config", cfg, "--data-dir", data,
     "--allow-synthetic"], text=True
)
resolved = json.loads(latest)
if resolved["step"] != int(state["global_step"]):
    raise SystemExit(f"resume resolved {resolved['step']} != state {state['global_step']}")
print(f"[chaos] OK: {len(events)} resumes, lost_steps={lost}, final_step={resolved['step']}")
PY

log "chaos continuity test passed"
