#!/bin/bash
# Import a wave JSON into foundation_full.json and restart brain19
FILE="$1"
if [ ! -f "$FILE" ]; then echo "File not found: $FILE"; exit 1; fi

cd /home/hirschpekf/brain19

python3 << PYEOF
import json

with open('data/foundation_full.json') as f:
    foundation = json.load(f)
with open('$FILE') as f:
    expansion = json.load(f)

def get_rel_key(r):
    s = r.get('source', r.get('from', ''))
    t = r.get('target', r.get('to', ''))
    tp = r.get('type', r.get('relation', ''))
    return (s, t, tp)

def normalize_rel(r):
    return {
        'source': r.get('source', r.get('from', '')),
        'target': r.get('target', r.get('to', '')),
        'type': r.get('type', r.get('relation', '')),
        'weight': r.get('weight', r.get('strength', 0.9))
    }

existing_labels = set(c['label'] for c in foundation.get('concepts', []))
new_concepts = [c for c in expansion['concepts'] if c['label'] not in existing_labels]
foundation['concepts'].extend(new_concepts)

existing_rels = set(get_rel_key(r) for r in foundation.get('relations', []))
new_rels = [normalize_rel(r) for r in expansion['relations'] if get_rel_key(r) not in existing_rels]
foundation['relations'].extend(new_rels)

with open('data/foundation_full.json', 'w') as f:
    json.dump(foundation, f)

print(f'Added {len(new_concepts)} concepts, {len(new_rels)} relations')
print(f'Total: {len(foundation["concepts"])} concepts, {len(foundation["relations"])} relations')
PYEOF
