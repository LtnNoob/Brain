# Graph Densification Plan — Pattern→Relation Generator

**Datum:** 2026-02-14  
**Ziel:** Graph-Dichte von 2.95 → 8-12 Relations/Concept  
**Geschätzter Aufwand:** 3-4 Tage  
**Aktuell:** 29.675 Concepts, 71.929 Relations, 5.9M Patterns, 5.5M Gaps

---

## Übersicht

Drei Phasen, jede einzeln deploybar und testbar:

```
Phase 1: Pattern Co-Activation → Relations    (Tag 1-2)  → +200-400K Relations
Phase 2: Sibling Gap-Filling                   (Tag 2-3)  → +30-80K Relations  
Phase 3: Transitive Inference                  (Tag 3-4)  → +20-50K Relations
```

**Erwartetes Ergebnis:** ~8-12 Relations/Concept (von 2.95)

---

## Phase 1: Pattern Co-Activation → Typed Relations (HAUPTHEBEL)

### Konzept

Die 5.9M Patterns enthalten implizite semantische Relations. Wenn zwei Concepts regelmäßig in denselben Patterns co-aktiviert werden, besteht eine echte Beziehung. Der Trick ist: den **Typ** der Relation aus dem Kontext ableiten.

### Algorithmus

```
Für jedes Pattern P mit Concepts {C1, C2, ..., Cn}:
  Für jedes Paar (Ci, Cj) wo i < j:
    co_activation_count[Ci][Cj] += 1

Für jedes Paar mit co_activation_count > threshold:
  jaccard = co_count / (patterns_with_Ci + patterns_with_Cj - co_count)
  
  IF jaccard > 0.3:
    relation_type = infer_type(Ci, Cj)   // Siehe unten
    weight = jaccard
    → Neue Relation (Ci, relation_type, Cj, weight)
```

### Relation-Type Inference (KEIN Pattern Matching!)

Typ wird aus **bestehender Graph-Struktur** abgeleitet, nicht aus Strings:

```cpp
RelationType infer_relation_type(ConceptId a, ConceptId b, const LTM& ltm) {
    // 1. Hierarchie-Check: Teilen sie einen IS_A Parent?
    auto parents_a = get_parents(a);  // IS_A targets
    auto parents_b = get_parents(b);
    auto shared_parents = intersection(parents_a, parents_b);
    
    if (!shared_parents.empty())
        return SIMILAR_TO;  // Geschwister = ähnlich
    
    // 2. Kompositions-Check: Ist einer PART_OF vom anderen?
    if (has_relation(a, PART_OF, any) && has_relation(b, PART_OF, any))
        if (share_parent_composite(a, b))
            return ASSOCIATED_WITH;  // Co-Teile eines Ganzen
    
    // 3. Kausaler Check: Existiert kausale Kette A→...→B?
    if (causal_path_exists(a, b, max_hops=3))
        return ENABLES;  // Kausale Nähe
    
    // 4. Hierarchie-Abstand
    int dist = hierarchy_distance(a, b);
    if (dist == 1)
        return IS_A;  // Direkte Hierarchie-Nähe
    if (dist <= 3)
        return ASSOCIATED_WITH;
    
    // 5. Default: Co-Aktivierung ohne klaren Typ
    return ASSOCIATED_WITH;  // Sicherer Default
}
```

**Wichtig:** Typ kommt aus Graph-Topologie, NICHT aus Concept-Labels oder String-Parsing.

### Thresholds & Quality Gates

| Parameter | Wert | Begründung |
|-----------|------|------------|
| Min co-activation count | 5 | Eliminiert Zufall (5 von 5.9M = sehr konservativ) |
| Jaccard threshold | 0.15 | Niedriger als ursprünglich gedacht — Graph ist sparse |
| Max neue Relations/Concept | 30 | Verhindert Hub-Explosion |
| Trust Level | 0.6 (THEORY) | Unter manuell eingefügten FACTS (0.98) |
| Duplikat-Check | Ja | Keine Relation wenn schon existiert |
| Self-Loop Check | Ja | Kein A→A |

### Quality Gate: Stichproben-Validierung

Nach Generierung: 100 zufällige neue Relations samplen und prüfen:
- Macht die Relation semantisch Sinn?
- Ist der Typ korrekt?
- Akzeptanzrate sollte >80% sein

### Implementation

**Neues File:** `backend/evolution/pattern_relation_generator.hpp/.cpp`

```cpp
class PatternRelationGenerator {
public:
    PatternRelationGenerator(LongTermMemory& ltm, const PatternDiscovery& pd);
    
    struct GenerationConfig {
        size_t min_co_activation = 5;
        double jaccard_threshold = 0.15;
        size_t max_relations_per_concept = 30;
        double trust_level = 0.6;  // THEORY level
    };
    
    struct GenerationResult {
        size_t relations_generated;
        size_t duplicates_skipped;
        size_t below_threshold;
        std::map<RelationType, size_t> type_distribution;
    };
    
    // Phase 1: Co-Activation basiert
    GenerationResult generate_from_coactivation(const GenerationConfig& config);
    
    // Phase 2: Gap-Filling (aus bestehender find_gaps)
    GenerationResult fill_sibling_gaps(double min_confidence = 0.5);
    
    // Phase 3: Transitive Inference
    GenerationResult infer_transitive(size_t max_chain_length = 3);
    
    // Stichproben für Quality Gate
    std::vector<std::tuple<ConceptId, RelationType, ConceptId, double>> 
        sample_generated(size_t n = 100);

private:
    LongTermMemory& ltm_;
    const PatternDiscovery& pd_;
    
    RelationType infer_relation_type(ConceptId a, ConceptId b) const;
    double compute_jaccard(ConceptId a, ConceptId b, 
                           const CoActivationMap& coact) const;
};
```

### Memory-Profil

Co-Activation Map für 29K Concepts:
- Worst case: 29K² = 870M Paare → zu viel
- **Lösung:** Nur Paare zählen die tatsächlich co-aktiviert sind
- Bei 5.9M Patterns mit avg 3-5 Concepts pro Pattern: ~30-50M Paare
- Pro Paar: 2×ConceptId (8B) + count (4B) = 12B → ~360-600MB
- **Alternative:** Batch-Processing per Component (aus find_components) → <100MB

→ Batch per Connected Component empfohlen (i5-6600K hat 58GB RAM, kein Problem)

---

## Phase 2: Sibling Gap-Filling (bestehende Infrastruktur)

### Konzept

`PatternDiscovery::find_gaps()` findet bereits 5.5M Gaps. Diese sind: "Concept A hat Sibling B, B hat Relation R zu X, A hat R zu X nicht."

### Verbesserung gegenüber aktuellem Code

Aktuell: Gaps werden nur **entdeckt** und geloggt. Neu: Selektiv **füllen**.

```
Filter:
1. Gap confidence > 0.5  
2. Mindestens 2 Siblings haben dieselbe Relation (Voting)
3. Nicht IS_A (zu riskant für automatische Inference)
4. Target-Concept existiert und ist aktiv
```

**Beispiel:**
- Dog IS_A Animal, Cat IS_A Animal
- Cat HAS_PROPERTY "Independent" — kein Gap für Dog (Dogs sind nicht independent)
- Cat REQUIRES Food, Dog REQUIRES Food — ✓ Beide brauchen Food
- **Voting:** Wenn 3+ Siblings eine Relation haben → hohe Confidence

### Expected Yield

Von 5.5M Gaps nach Filtering: ~30-80K hochwertige neue Relations

---

## Phase 3: Transitive Inference

### Regeln

```
1. IS_A Transitivität:
   A IS_A B, B IS_A C → A IS_A C  (trust *= 0.9 per hop)
   Max 3 Hops, Trust-Floor 0.5

2. PART_OF Transitivität:
   A PART_OF B, B PART_OF C → A PART_OF C  (trust *= 0.85)
   Max 2 Hops

3. CAUSES Ketten:
   A CAUSES B, B CAUSES C → A ENABLES C  (trust *= 0.7)
   Max 2 Hops, höherer Decay weil unsicherer

4. Property Inheritance:
   A IS_A B, B HAS_PROPERTY P → A HAS_PROPERTY P  (trust *= 0.8)
   Nur 1 Hop (direkte Parent-Properties)
```

### Vorsichtsmaßnahmen

- **Trust-Decay** pro Hop verhindert Garbage-Ketten
- **Cycle-Detection** (A→B→C→A darf nicht zu Trust-Aufbau führen)  
- **Override-Check:** Wenn A explizit CONTRADICTS was B hat → nicht erben
- Alle inferrierten Relations markiert als `source: INFERRED` (unterscheidbar von `IMPORTED` und `DISCOVERED`)

### Expected Yield

~20-50K neue Relations, aber höchste Qualität (logisch ableitbar)

---

## Integration & API

### Neuer API-Endpoint

```
POST /api/densify
{
    "phase": 1|2|3|"all",
    "dry_run": true|false,
    "config": { ... optional overrides ... }
}

Response:
{
    "relations_generated": 245891,
    "duplicates_skipped": 12044,
    "type_distribution": { "SIMILAR_TO": 89234, "ASSOCIATED_WITH": 65123, ... },
    "sample": [ ... 10 examples ... ],
    "density_before": 2.95,
    "density_after": 11.2
}
```

### Persistierung

Neue Relations → gleiche LTM wie manuelle Relations, aber mit:
- `trust: 0.6` (THEORY, nicht FACT)
- `source: "pattern_coactivation"` / `"sibling_gap"` / `"transitive_inference"`
- Damit später filterbar und revertierbar

---

## Risiken & Mitigations

| Risiko | Wahrscheinlichkeit | Mitigation |
|--------|-------------------|------------|
| Garbage Relations (false positives) | Mittel | Quality Gate (Stichprobe), Jaccard threshold, Trust-Level niedrig |
| Memory-Explosion bei Co-Activation Map | Niedrig | Batch per Component, 58GB RAM reicht |
| Decoder-Training verschlechtert sich | Niedrig | A/B Test: Decoder vor und nach Densification trainieren |
| Hub-Concepts bekommen zu viele Relations | Mittel | Max 30 neue/Concept, Relation-Cap |
| Performance-Regression bei Queries | Niedrig | Clustering (geplant) kompensiert |

---

## Reihenfolge & Dependencies

```
                    ┌─────────────────┐
                    │ Phase 1:        │
                    │ Co-Activation   │ ← Braucht: PatternDiscovery patterns
                    │ → +200-400K     │    Hat: 5.9M patterns ✓
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Phase 2:        │
                    │ Gap-Filling     │ ← Braucht: find_gaps() + neue Relations aus Ph.1
                    │ → +30-80K       │    Hat: 5.5M gaps ✓
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Phase 3:        │
                    │ Transitive      │ ← Braucht: Dichteren Graph aus Ph.1+2
                    │ → +20-50K       │    Profitiert von mehr Input-Relations
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Re-Train CMs    │ ← Alle ConceptModels neu trainieren
                    │ + Decoder       │    mit dichterem Graph
                    └─────────────────┘
```

---

## Erfolgskriterien

- [ ] Relations/Concept ≥ 8 (von 2.95)
- [ ] Quality Gate: >80% der Stichproben semantisch korrekt
- [ ] Kein Concept mit >100 Relations (Anti-Hub)
- [ ] Decoder Loss sinkt nach Re-Training (aktuell 2.45-3.13)
- [ ] API Response-Time bleibt <2s für Queries
- [ ] Alle neuen Relations revertierbar (source-Tag)
