#!/bin/bash
# Brain19 Full Stack Launcher
# Usage: ./start.sh [--api-only|--frontend-only|--all]

DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:---all}"

cleanup() {
    echo "[Brain19] Shutting down..."
    kill $(jobs -p) 2>/dev/null
    wait
    echo "[Brain19] Done."
}
trap cleanup EXIT

# Start API server
if [[ "$MODE" == "--all" || "$MODE" == "--api-only" ]]; then
    echo "[Brain19] Starting API server on :8019..."
    cd "$DIR"
    .venv/bin/python api/server.py &
    sleep 3
fi

# Start Frontend dev server
if [[ "$MODE" == "--all" || "$MODE" == "--frontend-only" ]]; then
    echo "[Brain19] Starting Frontend on :3019..."
    cd "$DIR/frontend"
    npx vite --port 3019 --host 0.0.0.0 &
fi

echo "[Brain19] Stack running!"
echo "  API:      http://localhost:8019"
echo "  Frontend: http://localhost:3019"
echo ""
echo "Press Ctrl+C to stop."
wait
