#!/bin/bash
# Export precomputed analyses so web/ works with no backend.
#
# The interface normally posts a capture to the FastAPI server, which loads the
# flow table and runs the model. Static hosting can do neither, so this runs
# each capture day through the real endpoint once and saves the response. The
# page falls back to these files when /api/health does not answer, which makes
# the static build the same interface rather than a reduced copy of it.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8090}

if ! curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null; then
  echo "no backend on :$PORT -- start it first:" >&2
  echo "  PYTHONPATH=. python -m uvicorn foresight.server:app --port $PORT" >&2
  exit 1
fi

mkdir -p web/data
DAYS=$(curl -s "http://127.0.0.1:$PORT/api/sample" \
       | python3 -c 'import json,sys;print(" ".join(json.load(sys.stdin)["days"]))')
for day in $DAYS; do
  curl -sf -o "web/data/$day.json" -X POST "http://127.0.0.1:$PORT/api/analyse?day=$day"
  echo "  $day  $(du -h "web/data/$day.json" | cut -f1)"
done
python3 - "$DAYS" <<'PY'
import json, sys, pathlib
pathlib.Path("web/data/manifest.json").write_text(
    json.dumps({"days": sorted(sys.argv[1].split())}, indent=2))
PY
echo "exported $(ls web/data/2017-*.json | wc -l | tr -d ' ') days to web/data"
