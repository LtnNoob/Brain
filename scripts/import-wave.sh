#!/bin/bash
# Import a wave JSON into foundation_full.json — inheritance-aware merge
# Respects the OO IS_A hierarchy: cycle detection, definition specificity,
# trust propagation direction, CONTRADICTS blocking.
FILE="$1"
if [ ! -f "$FILE" ]; then echo "File not found: $FILE"; exit 1; fi

cd /home/hirschpekf/brain19

python3 << PYEOF
import json, sys
from collections import defaultdict

with open('data/foundation_full.json') as f:
    foundation = json.load(f)
with open('$FILE') as f:
    expansion = json.load(f)

def get_rel_key(r):
    s = r.get('source', r.get('from', ''))
    t = r.get('target', r.get('to', ''))
    tp = r.get('type', r.get('relation', '')).upper()
    return (s, t, tp)

def normalize_rel(r):
    return {
        'source': r.get('source', r.get('from', '')),
        'target': r.get('target', r.get('to', '')),
        'type': r.get('type', r.get('relation', '')).upper(),
        'weight': r.get('weight', r.get('strength', 0.9))
    }

# Inheritable relation types (must match C++ PropertyInheritance::inheritable_types)
INHERITABLE_TYPES = {'HAS_PROPERTY', 'REQUIRES', 'USES', 'PRODUCES'}

# =============================================================================
# Build IS_A hierarchy from existing KB
# =============================================================================
isa_parents = defaultdict(set)   # concept → set of IS_A parents
isa_children = defaultdict(set)  # concept → set of IS_A children
contradicts = defaultdict(set)   # concept → set of concepts it contradicts

for r in foundation.get('relations', []):
    rtype = r.get('type', r.get('relation', '')).upper()
    src = r.get('source', r.get('from', ''))
    tgt = r.get('target', r.get('to', ''))
    if rtype == 'IS_A':
        isa_parents[src].add(tgt)
        isa_children[tgt].add(src)
    elif rtype == 'CONTRADICTS':
        contradicts[src].add(tgt)
        contradicts[tgt].add(src)  # symmetric

def get_ancestors(label, max_depth=20):
    """Walk IS_A chain upward — returns set of all ancestors."""
    visited = set()
    frontier = {label}
    for _ in range(max_depth):
        next_f = set()
        for c in frontier:
            for p in isa_parents.get(c, set()):
                if p not in visited:
                    visited.add(p)
                    next_f.add(p)
        if not next_f:
            break
        frontier = next_f
    return visited

def get_descendants(label, max_depth=20):
    """Walk IS_A chain downward — returns set of all children/grandchildren."""
    visited = set()
    frontier = {label}
    for _ in range(max_depth):
        next_f = set()
        for c in frontier:
            for ch in isa_children.get(c, set()):
                if ch not in visited:
                    visited.add(ch)
                    next_f.add(ch)
        if not next_f:
            break
        frontier = next_f
    return visited

# =============================================================================
# Concept merge — inheritance-aware
# =============================================================================
concept_idx = {c['label']: i for i, c in enumerate(foundation.get('concepts', []))}
added_concepts = 0
merged_concepts = 0

for c in expansion.get('concepts', []):
    if c['label'] in concept_idx:
        existing = foundation['concepts'][concept_idx[c['label']]]
        changed = False

        # Definition merge: prefer longer/more specific definition.
        # In OO hierarchy, children should have MORE specific definitions than parents.
        new_def = c.get('definition', '')
        old_def = existing.get('definition', '')
        if new_def and (not old_def or old_def == ''):
            existing['definition'] = new_def
            changed = True
        elif new_def and old_def and len(new_def) > len(old_def) * 1.3:
            # Wave has significantly more detail → use it
            existing['definition'] = new_def
            changed = True

        # Epistemic type: fill if missing, don't overwrite
        for key in ('epistemic_type', 'type'):
            new_val = c.get(key)
            old_val = existing.get(key)
            if new_val and (not old_val or old_val == ''):
                existing[key] = new_val
                changed = True

        # Trust: take higher value. For parent concepts (with IS_A children),
        # trust is especially important because it propagates via inheritance.
        if 'trust' in c and c['trust'] > existing.get('trust', 0):
            existing['trust'] = c['trust']
            changed = True

        if changed:
            merged_concepts += 1
    else:
        foundation['concepts'].append(c)
        concept_idx[c['label']] = len(foundation['concepts']) - 1
        added_concepts += 1

# =============================================================================
# Relation merge — inheritance-aware
# =============================================================================
rel_idx = {}
for i, r in enumerate(foundation.get('relations', [])):
    rel_idx[get_rel_key(r)] = i

all_labels = set(c['label'] for c in foundation['concepts'])

added_rels = 0
merged_rels = 0
skipped_rels = 0
cycle_blocked = 0
contradicts_blocked = 0
affected_subtrees = set()  # parent labels whose children may need re-inheritance

for r in expansion.get('relations', []):
    nr = normalize_rel(r)

    # Skip if source or target not in KB
    if nr['source'] not in all_labels or nr['target'] not in all_labels:
        skipped_rels += 1
        continue

    rtype = nr['type']
    key = (nr['source'], nr['target'], rtype)

    # --- IS_A cycle detection ---
    # Adding "A IS_A B" would create a cycle if B is already a descendant of A
    if rtype == 'IS_A':
        descendants_of_source = get_descendants(nr['source'])
        if nr['target'] in descendants_of_source or nr['target'] == nr['source']:
            print(f'  CYCLE BLOCKED: {nr["source"]} IS_A {nr["target"]}', file=sys.stderr)
            cycle_blocked += 1
            continue

    # --- CONTRADICTS check for inheritable types ---
    # If we're adding HAS_PROPERTY/REQUIRES/USES/PRODUCES for concept C,
    # check that C doesn't CONTRADICTS the target (would be blocked at runtime anyway,
    # but keeping the data clean is better).
    if rtype in INHERITABLE_TYPES:
        if nr['target'] in contradicts.get(nr['source'], set()):
            print(f'  CONTRADICTS BLOCKED: {nr["source"]} {rtype} {nr["target"]}', file=sys.stderr)
            contradicts_blocked += 1
            continue

    if key in rel_idx:
        # Existing relation — merge weight
        existing = foundation['relations'][rel_idx[key]]
        old_w = existing.get('weight', existing.get('strength', 0))
        new_w = nr['weight']
        if new_w > old_w:
            foundation['relations'][rel_idx[key]] = nr
            merged_rels += 1

            # If updating an inheritable relation on a parent concept,
            # mark its subtree as affected (trust change propagates)
            if rtype in INHERITABLE_TYPES and nr['source'] in isa_children:
                affected_subtrees.add(nr['source'])
    else:
        foundation['relations'].append(nr)
        rel_idx[key] = len(foundation['relations']) - 1
        added_rels += 1

        # Track hierarchy impact
        if rtype == 'IS_A':
            # New IS_A → update hierarchy maps
            isa_parents[nr['source']].add(nr['target'])
            isa_children[nr['target']].add(nr['source'])
        elif rtype in INHERITABLE_TYPES and nr['source'] in isa_children:
            affected_subtrees.add(nr['source'])

# =============================================================================
# Property inheritance propagation (multiple inheritance)
# For each concept that gained a new IS_A parent (or is newly added with IS_A),
# inherit HAS_PROPERTY/REQUIRES/USES/PRODUCES from ALL IS_A parents.
# Trust decays 0.9x per hop. CONTRADICTS blocks inheritance.
# =============================================================================
DECAY = 0.9
TRUST_FLOOR = 0.3

# Build outgoing-relations index by (source, type)
outgoing = defaultdict(list)  # source → [(target, type, weight), ...]
for r in foundation['relations']:
    src = r.get('source', r.get('from', ''))
    tgt = r.get('target', r.get('to', ''))
    rtype = r.get('type', r.get('relation', '')).upper()
    w = r.get('weight', r.get('strength', 0.9))
    outgoing[src].append((tgt, rtype, w))

def collect_inherited_properties(concept_label, max_depth=20):
    """Walk IS_A chain upward through ALL parents (multiple inheritance).
    Collect inheritable properties with decayed trust. Best trust wins per target."""
    inherited = {}  # (target, type) → best_weight
    visited = set()

    # BFS with depth tracking: [(label, depth)]
    queue = [(concept_label, 0)]
    visited.add(concept_label)

    while queue:
        current, depth = queue.pop(0)
        if depth > max_depth:
            continue

        for parent in isa_parents.get(current, set()):
            hop_decay = DECAY ** (depth + 1)

            # Collect inheritable relations from this parent
            for tgt, rtype, w in outgoing.get(parent, []):
                if rtype not in INHERITABLE_TYPES:
                    continue
                decayed_w = w * hop_decay
                if decayed_w < TRUST_FLOOR:
                    continue
                # CONTRADICTS check
                if tgt in contradicts.get(concept_label, set()):
                    continue
                # Self-loop check
                if tgt == concept_label:
                    continue
                key = (tgt, rtype)
                if key not in inherited or decayed_w > inherited[key]:
                    inherited[key] = decayed_w

            # Continue up the chain (all parents)
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, depth + 1))

    return inherited

# Find concepts that need inheritance propagation:
# - newly added concepts with IS_A parents
# - existing concepts that got new IS_A parents
needs_propagation = set()
for r in expansion.get('relations', []):
    nr = normalize_rel(r)
    if nr['type'] == 'IS_A' and nr['source'] in all_labels:
        needs_propagation.add(nr['source'])
for c in expansion.get('concepts', []):
    if c['label'] in isa_parents:
        needs_propagation.add(c['label'])

inherited_added = 0
for concept_label in needs_propagation:
    props = collect_inherited_properties(concept_label)
    for (tgt, rtype), weight in props.items():
        key = (concept_label, tgt, rtype)
        if key in rel_idx:
            # Already exists — update weight if inherited is stronger
            existing = foundation['relations'][rel_idx[key]]
            old_w = existing.get('weight', existing.get('strength', 0))
            if weight > old_w:
                foundation['relations'][rel_idx[key]] = {
                    'source': concept_label, 'target': tgt,
                    'type': rtype, 'weight': round(weight, 4)
                }
                inherited_added += 1
        else:
            new_rel = {
                'source': concept_label, 'target': tgt,
                'type': rtype, 'weight': round(weight, 4)
            }
            foundation['relations'].append(new_rel)
            rel_idx[key] = len(foundation['relations']) - 1
            inherited_added += 1

with open('data/foundation_full.json', 'w') as f:
    json.dump(foundation, f)

# =============================================================================
# Report
# =============================================================================
print(f'Concepts: +{added_concepts} new, {merged_concepts} merged')
print(f'Relations: +{added_rels} new, {merged_rels} merged, {skipped_rels} skipped (missing labels)')
if cycle_blocked:
    print(f'IS_A cycles blocked: {cycle_blocked}')
if contradicts_blocked:
    print(f'CONTRADICTS conflicts blocked: {contradicts_blocked}')
if affected_subtrees:
    total_affected = sum(len(get_descendants(p)) for p in affected_subtrees)
    print(f'Inheritance impact: {len(affected_subtrees)} parent(s) updated, ~{total_affected} descendants may need re-propagation')
if inherited_added:
    print(f'Property inheritance: {inherited_added} relations propagated from parents ({len(needs_propagation)} concepts)')
print(f'Total: {len(foundation["concepts"])} concepts, {len(foundation["relations"])} relations')
PYEOF
