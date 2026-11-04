#!/usr/bin/env bash
# Launch the Atalla Benchmark Workbench (FastAPI + static UI) on this machine's LAN IP.
#
#   bench/workbench.sh              # http://<your-ip>:8765  (and http://127.0.0.1:8765)
#   bench/workbench.sh --port 9000
#   bench/workbench.sh --local      # bind 127.0.0.1 only
#   bench/workbench.sh --reload     # dev mode, restarts on code changes
#
# Python: $ATALLA_PYTHON, else $VIRTUAL_ENV, else ~/.venvs/atalla, else python3.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8765; HOST=0.0.0.0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --local) HOST=127.0.0.1; shift ;;
    --reload) EXTRA+=(--reload); shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
PY="${ATALLA_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PY="${PY:-$HOME/.venvs/atalla/bin/python}"
[ -x "$PY" ] || PY=python3
"$PY" -c "import fastapi, uvicorn" 2>/dev/null || { echo "installing fastapi/uvicorn into $PY"; "$PY" -m pip install -q fastapi 'uvicorn[standard]'; }

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo "Atalla Benchmark Workbench"
echo "  local:   http://127.0.0.1:${PORT}"
[ "$HOST" = "0.0.0.0" ] && [ -n "$LAN_IP" ] && echo "  network: http://${LAN_IP}:${PORT}"
echo "  repo:    $REPO   (runs land in docs/baselines/, exports in output/)"
cd "$REPO"
exec "$PY" -m uvicorn bench.ui.server:app --host "$HOST" --port "$PORT" "${EXTRA[@]}"
