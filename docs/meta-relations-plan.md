# Brain19 Meta-Relations Integration Plan

## Status: GEPLANT
**Erstellt:** 2026-02-14
**Geschätzter Aufwand:** 10-14 Tage (3 Phasen)

---

## 1. Bestehende Architektur

| Komponente | Datei | Rolle |
|---|---|---|
| **RelationType enum** | `memory/active_relation.hpp` | Alle Typen als `uint16_t`. IMPLIES(18), CONTRADICTS(6), REQUIRES(11) existieren bereits! |
| **RelationTypeRegistry** | `memory/relation_type_registry.hpp/.cpp` | Singleton mit Embeddings, Namen, Kategorien. Runtime-Erweiterung via `register_type()` ab ID 1000 |
| **RelationCategory** | `memory/relation_type_registry.hpp` | Gruppierung: CAUSAL, OPPOSITION, FUNCTIONAL, etc. |
| **RelationInfo** | `ltm/relation.hpp` | Persistente Relation mit weight, dynamic_weight, inhibition_factor |
| **FocusCursor** | `cursor/focus_cursor.cpp` | Multi-Hop Traversal via `evaluate_edge()` → ConceptModel.predict_refined(). **Kennt keine Relation-Semantik!** |
| **PatternDiscovery** | `evolution/pattern_discovery.cpp` | Findet Cluster, Hierarchien, Bridges, Cycles, Gaps. **Keine logische Inference.** |

### Kritischer Befund
- IMPLIES, CONTRADICTS, REQUIRES existieren als Relation-Types — werden aber **nirgends semantisch interpretiert**
- EXCLUDES fehlt komplett
- FocusCursor bewertet Kanten nur via ConceptModel-Prediction, ohne logische Regeln
- PatternDiscovery kennt nur IS_A für Hierarchien

---

## 2. EXCLUDES hinzufügen

**Datei:** `memory/active_relation.hpp`
```cpp
EXCLUDES = 20,
```

**Datei:** `memory/relation_type_registry.cpp`
```cpp
register_one(RelationType::EXCLUDES, "EXCLUDES",
    "schließt aus", "excludes",
    RelationCategory::OPPOSITION,
    {0.0, 0.1, 0.0, -0.6, 0.0, -0.8, 0.7, 0.5, 0.5, 0.8, ...}, true);
```

**Datei:** `memory/relation_type_registry.hpp` — neue Kategorie:
```cpp
LOGICAL,  // IMPLIES, CONTRADICTS, REQUIRES, EXCLUDES
```

---

## 3. Inference Rules im FocusCursor

**Kernproblem:** `evaluate_edge()` delegiert blind an ConceptModel.

**Datei:** `cursor/focus_cursor.cpp` — neue Methode:

```cpp
double FocusCursor::apply_meta_relation_logic(
    ConceptId from, ConceptId to, RelationType type, double base_score) const 
{
    switch (type) {
        case RelationType::IMPLIES:
            return base_score;  // Transitiv-Chaining in expand_implies()
            
        case RelationType::CONTRADICTS:
            return -base_score * 0.8;  // Negativ, mit Dämpfung
            
        case RelationType::REQUIRES:
            return check_requires_satisfied(to) ? base_score : base_score * 0.1;
            
        case RelationType::EXCLUDES:
            return is_concept_active(to) ? 0.0 : base_score;
            
        default:
            return base_score;
    }
}
```

**Transitive IMPLIES-Expansion:**
```cpp
std::vector<ConceptId> FocusCursor::expand_implies(ConceptId start, int max_depth = 3) const {
    std::vector<ConceptId> implied;
    std::queue<std::pair<ConceptId, double>> queue;
    std::unordered_set<ConceptId> seen;
    queue.push({start, 1.0});
    seen.insert(start);
    
    while (!queue.empty()) {
        auto [current, score] = queue.front(); queue.pop();
        if (score < 0.1) continue;
        
        for (auto& rel : ltm_.get_outgoing_relations(current)) {
            if (rel.type == RelationType::IMPLIES && !seen.count(rel.target)) {
                double new_score = score * rel.weight;
                implied.push_back(rel.target);
                seen.insert(rel.target);
                queue.push({rel.target, new_score});
            }
        }
    }
    return implied;
}
```

**Integration in `get_candidates()`:**
```cpp
// Transitive IMPLIES-Targets als virtuelle Candidates
auto implied = expand_implies(current_);
for (auto target : implied) {
    if (!visited_.count(target)) {
        candidates.push_back({target, RelationType::IMPLIES, 0.6, true});
    }
}

// CONTRADICTS-Filter: Candidates die aktiven Concepts widersprechen abwerten
for (auto& c : candidates) {
    for (auto visited_id : visited_) {
        auto rels = ltm_.get_relations_between(c.concept_id, visited_id);
        for (auto& r : rels) {
            if (r.type == RelationType::CONTRADICTS) {
                c.score *= 0.2;
            }
        }
    }
}
```

---

## 4. Pattern Discovery Erweiterung

**Datei:** `evolution/pattern_discovery.hpp/.cpp`

```cpp
std::vector<DiscoveredPattern> find_meta_relations();
std::vector<DiscoveredPattern> find_potential_implies();   // Co-Occurrence + Temporal
std::vector<DiscoveredPattern> find_potential_excludes();   // Co-Absence
std::vector<DiscoveredPattern> find_contradictions();       // Widersprüchliche Pfade
```

**Logik:**
- **IMPLIES-Kandidaten:** A TEMPORAL_BEFORE B + A CAUSES/ENABLES B → IMPLIES
- **EXCLUDES-Kandidaten:** A und B nie im selben Cluster/Pfad (Co-Absence bei >N Samples)
- **CONTRADICTS-Kandidaten:** Pfad A→...→B und Pfad A→...→¬B (widersprüchliche Schlussfolgerungen)

---

## 5. ConceptModel Impact

**Empfehlung: Kein eigenes ConceptModel pro Meta-Relation.** Stattdessen:

1. RelationType als Feature reicht — Registry-Embedding (16D) encodiert Semantik
2. Bestehende ConceptModels lernen Relation-Type-spezifisches Verhalten über `predict_refined()`
3. `is_meta` Flag in RelationTypeInfo → FocusCursor wendet logische Regeln VOR ConceptModel an

---

## 6. Lücken & Risiken

| Risiko | Schwere | Mitigation |
|---|---|---|
| Zyklen in IMPLIES (A→B→C→A) | 🔴 Hoch | Max-depth + visited-Set in expand_implies() |
| Widersprüchliche CONTRADICTS | 🟡 Mittel | Trust-Level entscheidet, niedrigerer Trust abgewertet |
| Performance bei IMPLIES-Expansion | 🟡 Mittel | Lazy Evaluation + Cache |
| REQUIRES als AND-Gate blockiert | 🟡 Mittel | Soft-REQUIRES (Score-Reduktion statt Hard-Block) |
| Binärkompatibilität | 🟢 Niedrig | EXCLUDES=20 passt in uint16_t |
| Pattern Discovery False Positives | 🟡 Mittel | Confidence-Threshold + Review |

---

## 7. Phasen-Plan

### Phase 1: Minimal Viable Meta-Relations (2-3 Tage)

| Task | Datei | Aufwand |
|---|---|---|
| EXCLUDES enum + Registry | `active_relation.hpp`, `relation_type_registry.cpp` | 30 min |
| `RelationCategory::LOGICAL` + `is_meta` Flag | `relation_type_registry.hpp` | 30 min |
| `apply_meta_relation_logic()` | `cursor/focus_cursor.cpp/.hpp` | 3h |
| `expand_implies()` mit Cycle-Safety | `cursor/focus_cursor.cpp/.hpp` | 2h |
| CONTRADICTS-Filter in `get_candidates()` | `cursor/focus_cursor.cpp` | 1h |
| REQUIRES AND-Gate Check | `cursor/focus_cursor.cpp` | 1h |
| EXCLUDES Mutual-Exclusion Check | `cursor/focus_cursor.cpp` | 1h |
| Tests | Neues `test_meta_relations.cpp` | 3h |
| Persistenz-Kompatibilität | `persistent/persistent_records.hpp` | 1h |

### Phase 2: Auto-Discovery (3-4 Tage)

| Task | Datei | Aufwand |
|---|---|---|
| `find_potential_implies()` | `evolution/pattern_discovery.cpp` | 4h |
| `find_potential_excludes()` | `evolution/pattern_discovery.cpp` | 4h |
| `find_contradictions()` | `evolution/pattern_discovery.cpp` | 6h |
| Meta-Relation Proposals | Neues `evolution/meta_relation_proposer.hpp/.cpp` | 4h |
| Integration in `discover_all()` | `evolution/pattern_discovery.cpp` | 1h |
| CLI-Command | `tools/brain19_cli.cpp` | 2h |

### Phase 3: Full Inference Engine (5-7 Tage)

| Task | Datei | Aufwand |
|---|---|---|
| `LogicalInferenceEngine` | Neues `inference/logical_engine.hpp/.cpp` | 8h |
| Forward-Chaining | `inference/logical_engine.cpp` | 6h |
| Contradiction Detection & Resolution | `inference/contradiction_resolver.hpp/.cpp` | 6h |
| REQUIRES-Dependency-Graph | `inference/dependency_graph.hpp/.cpp` | 4h |
| Integration ThinkingPipeline | `core/thinking_pipeline.cpp` | 4h |
| Integration ChatInterface | `llm/chat_interface.cpp` | 2h |
| Performance-Cache | `inference/inference_cache.hpp` | 3h |
| Tests | `inference/test_*.cpp` | 4h |
