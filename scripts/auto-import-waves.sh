#!/bin/bash
# Auto-import all wave files that haven't been imported yet
cd /home/hirschpekf/brain19
IMPORTED_LOG="data/.imported_waves"
touch "$IMPORTED_LOG"

CHANGED=0
for f in data/wave*_*.json data/knowledge_expansion_wave*.json; do
    [ -f "$f" ] || continue
    if grep -qF "$f" "$IMPORTED_LOG" 2>/dev/null; then continue; fi
    
    # Validate JSON first
    python3 -c "import json; json.load(open('$f'))" 2>/dev/null || { echo "INVALID: $f"; continue; }
    
    echo "Importing: $f"
    bash scripts/import-wave.sh "$f"
    echo "$f" >> "$IMPORTED_LOG"
    CHANGED=1
done

if [ "$CHANGED" = "1" ]; then
    echo "Restarting brain19-api..."
    sudo systemctl restart brain19-api
    sleep 15
    echo "Status after import:"
    curl -s http://localhost:8019/api/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('raw',''))" 2>/dev/null
fi
