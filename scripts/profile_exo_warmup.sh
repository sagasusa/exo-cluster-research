#!/usr/bin/env bash
# profile_exo_warmup.sh
# Diagnose whether an EXO node is stuck in single-threaded MLX-CPU warmup.
#
# Usage:  ./profile_exo_warmup.sh [duration_seconds]
# Run this in a SECOND terminal WHILE an EXO model is loading / "WARMING UP".
#
# It captures: the runner's thread-level CPU use (top -H), three py-spy
# stack samples, and a per-core mpstat log — the evidence needed to tell
# compute-bound single-threading apart from a synchronization stall.
#
# From an independent study comparing EXO vs llama.cpp on a CPU/GPU cluster.

set -euo pipefail
DUR="${1:-120}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="exo_warmup_profile_${STAMP}"
mkdir -p "$OUT"
echo "Writing evidence to ./$OUT/"

# --- 1. Find the EXO runner PID (the process doing compute) ---
PID="$(pgrep -f 'exo.main' | head -1 || true)"
if [ -z "${PID:-}" ]; then
  PID="$(pgrep -f 'multiprocessing.spawn' | head -1 || true)"
fi
if [ -z "${PID:-}" ]; then
  echo "ERROR: no EXO runner process found. Is EXO running and warming up?"
  echo "Try: pgrep -af exo"
  exit 1
fi
echo "Runner PID: $PID"
ps -p "$PID" -o pid,comm,cmd | tee "$OUT/pid.txt"

# --- 2. Thread-level snapshot: is one thread pinned while others idle? ---
echo "Capturing top -H ..."
top -H -b -n 1 -p "$PID" > "$OUT/top_H.txt" 2>&1 || true
RUNNING=$(grep -cE ' R ' "$OUT/top_H.txt" || true)
echo "  threads in R (running) state: ${RUNNING:-unknown}"

# --- 3. Per-core utilisation over time (needs sysstat) ---
if command -v mpstat >/dev/null 2>&1; then
  echo "Logging mpstat per-core for ${DUR}s ..."
  mpstat -P ALL 5 "$((DUR/5))" > "$OUT/mpstat.txt" 2>&1 &
  MPPID=$!
else
  echo "  (mpstat not installed — skipping. apt install sysstat to enable.)"
  MPPID=""
fi

# --- 4. Stack sampling: MLX compute frames vs lock/poll frames? ---
command -v py-spy >/dev/null 2>&1 || pip install py-spy --break-system-packages -q 2>/dev/null || true
for i in 1 2 3; do
  echo "py-spy dump #$i ..."
  sudo env "PATH=$PATH" py-spy dump --pid "$PID" > "$OUT/pyspy_dump_${i}.txt" 2>&1 || \
    py-spy dump --pid "$PID" > "$OUT/pyspy_dump_${i}.txt" 2>&1 || true
  sleep $(( DUR / 3 ))
done

[ -n "$MPPID" ] && wait "$MPPID" 2>/dev/null || true

# --- 5. Verdict heuristic ---
echo ""
echo "===== QUICK READ ====="
if grep -qiE 'mlx|generate_step|stream_generate' "$OUT"/pyspy_dump_*.txt 2>/dev/null; then
  echo "COMPUTE-BOUND: stacks show MLX generation frames."
  echo "  -> consistent with single-threaded MLX-CPU warmup (not a deadlock)."
elif grep -qiE 'poll|recv|acquire|lock|wait' "$OUT"/pyspy_dump_*.txt 2>/dev/null; then
  echo "SYNC-BOUND: stacks show polling/lock frames."
  echo "  -> consistent with a synchronization/communication stall."
else
  echo "Inconclusive — inspect $OUT/pyspy_dump_*.txt by hand."
fi
echo "All evidence saved in ./$OUT/"
