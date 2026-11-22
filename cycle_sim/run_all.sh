#!/usr/bin/env bash
# Run every generated GEMM program through atalla-sim, strictly one at a time.
# usage: cycle_sim/run_all.sh <programs dir> [extra run_kernel args...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PROGS="$1"; shift
PY="${ATALLA_SIM_PY:-$HERE/../../atalla-sim-itap/.venv/bin/python}"
for d in "$PROGS"/*/; do
    d="${d%/}"
    [ -f "$d/gemm.c" ] || continue
    if [ -f "$d/summary.json" ] && grep -q '"rc": 0' "$d/summary.json"; then
        echo "skip (done): $d"; continue
    fi
    echo "== $d"
    ( cd "$d" && rm -f run.log perf.json C_cycle.npy prog.in prog.s
      start=$(date +%s)
      # 30 min time box via the simulator's own cycle limit is not wall time; use a watchdog.
      "$PY" "$HERE/run_atalla_sim.py" gemm.c $(cat args.txt) --save prog.in --save-asm prog.s \
          --report --perf-json perf.json --check "$@" > run.log 2>&1 &
      pid=$!
      ( sleep 1800; kill $pid 2>/dev/null ) >/dev/null 2>&1 & wd=$!
      wait $pid; rc=$?
      kill $wd 2>/dev/null; pkill -P $wd 2>/dev/null
      end=$(date +%s)
      printf '{"rc": %d, "wall_s": %d}\n' "$rc" "$((end - start))" > run_status.json
      echo "rc=$rc wall=$((end - start))s"; grep -m1 " cycles (" run.log )
done
