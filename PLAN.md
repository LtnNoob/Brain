# Brain19 Convergence v2 — Implementation Plan

> Basiert auf: `/home/claude/.openclaw/workspace/brain19-convergence-design-v2.md`
> Abgleich mit aktuellem Code-Stand vom 16.02.2026

---

## Diskrepanzen: Design-Dokument vs. Realität

Vor der Step-by-Step-Planung muss klar sein, wo das Design-Dokument von der Realität abweicht:

| Design-Annahme | Realität | Konsequenz |
|---|---|---|
| `RelationCategory` hat 8 Werte (ASSOCIATIVE, DEPENDENCY, CREATIVE) | Actual hat 9 Werte (SIMILARITY, FUNCTIONAL, EPISTEMIC, CUSTOM_CATEGORY) | Mapping anpassen: SIMILARITY≈ASSOCIATIVE, FUNCTIONAL≈DEPENDENCY, CREATIVE fehlt → als CUSTOM oder neue Kategorie |
| ConceptModel input_dim = 90 | Actual INPUT_DIM = 20 (16 core + 4 cyclic-compressed) | KAN-Input-Port projection muss 256→4 statt 256→32 sein, oder INPUT_DIM wird erweitert |
| Dateien als Python (`deep_integration.py`) | Codebase ist reines C++ | Alles in C++ implementieren |
| KAN Layer 1: 90→256 | DeepKAN existiert mit {90,256,128}, aber das ist der Sprachmodell-KAN (Encoder→Decoder) | Convergence-KAN ist ein NEUES Modul, NICHT der language-KAN |
| Gated Residual, Router, Pipeline existieren nicht | Korrekt — alles muss neu | Neue Dateien unter `backend/convergence/` |
| `inhibit_concept()` in STM | Existiert nicht | Muss hinzugefügt werden |
| `has_contradictions()` → instant demotion | Bestätigt: jede einzelne CONTRADICTS-Relation → sofortige Demotion | Muss zu Ratio-basiert geändert werden |
| Spreading activation ist relation-unaware | Bestätigt: nutzt `rel.weight` uniform | Muss RelationBehavior-aware werden |
| `learn_from_graph()` hat keine Relation-Awareness | Bestätigt: nudge richtung gewichteter Durchschnitt aller Nachbarn | Muss per-category alpha nutzen |

---

## Implementation-Reihenfolge (18 Steps)

### PHASE A: Foundation — Relation-Aware Config (keine Abhängigkeiten)

#### Step 1: RelationBehavior Config (`core/relation_config.hpp`)
**Neue Datei** — Zentrale Konfiguration die ALLE Subsysteme nutzen.

```
Erstelle: backend/core/relation_config.hpp
```

- `enum class InheritDirection { NONE, FORWARD, REVERSE, BOTH }`
- `struct RelationBehavior { spreading_weight, spreading_direction, embedding_alpha, trust_decay_per_hop, inherit_properties, inherit_dir }`
- `const std::unordered_map<RelationCategory, RelationBehavior> RELATION_BEHAVIORS` — konstante Lookup-Tabelle
- Map die existierenden 9 RelationCategory-Werte (inkl. SIMILARITY→ASSOCIATIVE-Semantik, FUNCTIONAL→DEPENDENCY-Semantik, EPISTEMIC, CUSTOM_CATEGORY mit Defaults)
- Convenience-Funktion: `const RelationBehavior& get_behavior(RelationCategory cat)`

**Effort:** ~100 LOC, 30min

---

### PHASE B: Subsystem-Fixes (parallel, jeweils abhängig von Step 1)

#### Step 2: STM Inhibition (`memory/stm.hpp`, `memory/stm.cpp`)
**Existierende Dateien modifizieren.**

- Neue Methode: `void inhibit_concept(ContextId, ConceptId, double amount)`
  - Reduziert Activation um `amount`, clamp auf 0.0
  - Nutzt existierende `Context.concepts` map
- Neue Methode: `void inhibit_relation(ContextId, ConceptId source, ConceptId target, double amount)` (optional, für Vollständigkeit)

**Effort:** ~30 LOC, 15min

#### Step 3: Spreading Activation Fix (`cognitive/cognitive_dynamics.cpp`)
**Existierende Datei modifizieren.** Abhängig von Step 1 + Step 2.

- `spread_recursive()` bekommt zusätzlichen Parameter: `const RelationTypeRegistry& registry`
- In der for-Schleife (Zeile 170-206): RelationCategory via `registry.get_category(rel.type)` holen
- `propagated *= behavior.spreading_weight * behavior.spreading_direction`
- Wenn `spreading_direction < 0`: `stm.inhibit_concept()` statt `activate/boost`
- Target-INVALIDATED-Check vor Propagation (Audit #10 — derzeit nur Source gecheckt)
- Header (`cognitive_dynamics.hpp`) anpassen: Signatur erweitern
- **Alle Aufrufe** von `spread_recursive` und `spread_activation` in der Codebase finden und Registry-Parameter durchreichen

**Effort:** ~60 LOC Änderungen, 45min

#### Step 4: Embedding Training Fix (`micromodel/concept_embedding_store.cpp`)
**Existierende Datei modifizieren.** Abhängig von Step 1.

- `learn_from_graph()` bekommt zusätzlichen Parameter: `const RelationTypeRegistry& registry`
- Pro Nachbar-Relation: `alpha *= behavior.embedding_alpha`
- Negativer alpha für OPPOSITION → Embedding wird weggedrückt statt angezogen
- Header anpassen + Aufrufer aktualisieren

**Effort:** ~40 LOC Änderungen, 30min

#### Step 5: Property Inheritance Fix (`evolution/property_inheritance.cpp`, `evolution/graph_densifier.cpp`)
**Existierende Dateien modifizieren.** Abhängig von Step 1.

- Trust-Decay per-hop aus `RELATION_BEHAVIORS[cat].trust_decay_per_hop` statt hardcoded 0.9
- PART_OF/COMPOSITIONAL: Properties fließen von Teil→Ganzes (REVERSE direction)
- `is_aggregatable()` Funktion: nur physische/quantitative Properties propagieren über PART_OF
- `phase_partof_property()` in GraphDensifier prüfen und ggf. Richtung korrigieren

**Effort:** ~60 LOC Änderungen, 45min

#### Step 6: Epistemic Demotion Fix (`evolution/epistemic_promotion.cpp`)
**Existierende Datei modifizieren.** Unabhängig.

- `has_contradictions()` (Zeile 170-185) ersetzen durch `contradiction_ratio()`:
  - Zählt SUPPORTS und CONTRADICTS Relations
  - Return float ratio = contradicts / (supports + contradicts)
- `check_demotion()` (Zeile 340+) nutzt Ratio statt Bool:
  - `ratio > 0.3` → Demotion (statt: jede einzelne CONTRADICTS → instant demotion)
- Bestehende Tests anpassen

**Effort:** ~40 LOC Änderungen, 30min

#### Step 7: Salience Normalization Fix (`cognitive/cognitive_dynamics.cpp`)
**Existierende Datei modifizieren.** Unabhängig.

- `compute_salience()` und `compute_salience_batch()` vereinheitlichen
- Formel: `connectivity = log(count + 1) / log(max_global_count + 1)`

**Effort:** ~20 LOC, 15min

#### Step 8: Dynamic Block Boundaries (`language/language_training.cpp`)
**Existierende Datei modifizieren.** Unabhängig.

- Hardcoded 64/80 ersetzen durch:
  ```cpp
  const int flex_start = FUSED_DIM;
  const int dimctx_start = FUSED_DIM + flex_dim;
  ```

**Effort:** ~10 LOC, 10min

---

### PHASE C: Convergence Pipeline (neue Dateien)

> **Architektur-Entscheidung:** Alle Convergence-Komponenten kommen unter `backend/convergence/`.
> Dies ist C++, nicht Python wie im Design-Dokument. Die DeepKAN-Infrastruktur (`EfficientKANLayer`) existiert bereits in `language/deep_kan.hpp` und wird wiederverwendet.

#### Step 9: Convergence Directory + Grundstruktur
**Neue Dateien erstellen.**

```
backend/convergence/
├── convergence_config.hpp       — Dimensionskonstanten, Hyperparameter
├── convergence_kan.hpp/cpp      — 3-Layer KAN mit CM-Feedback-Port
├── concept_router.hpp/cpp       — Centroid-basiertes Routing (Top-K)
├── gated_residual.hpp/cpp       — Gated Residual PoE + Ignition
├── convergence_pipeline.hpp/cpp — End-to-End DeepIntegratedPipeline
└── Makefile.convergence         — Build-Konfiguration
```

**convergence_config.hpp:**
- `QUERY_DIM = 90` (aus DeepKAN Input)
- `KAN_L1_OUT = 256`
- `KAN_PROJ_OUT = 32`
- `CM_OUTPUT_DIM = 32`
- `KAN_L2_IN = KAN_L1_OUT + CM_OUTPUT_DIM = 288`
- `KAN_L2_OUT = 128`
- `KAN_L3_OUT = 32`  (= Output dim)
- `ROUTER_TOP_K = 4`
- `IGNITION_FAST = 0.85`, `IGNITION_DELIBERATE = 0.40`

**Effort:** ~50 LOC, 15min

#### Step 10: Convergence KAN (`convergence/convergence_kan.hpp/cpp`)
**Neue Dateien.** Nutzt `EfficientKANLayer` aus `language/deep_kan.hpp`.

- `ConvergenceKAN` Klasse:
  - `layer1_`: EfficientKANLayer(90, 256)
  - `projection_`: Linear(256, 32) — shared Projektion für CM-Input
  - `layer2_`: EfficientKANLayer(288, 128) — 256 (k1) + 32 (CM feedback)
  - `layer3_`: EfficientKANLayer(128, 32)
- `forward_layer1(h) → k1`
- `project_for_cm(k1) → k1_proj`
- `forward_layer2_3(k1, cm_output) → G(h)`
- Backward-Pass für Training (nutzt existierende `backward()` von EfficientKANLayer)

**Effort:** ~250 LOC, 2h

#### Step 11: Concept Router (`convergence/concept_router.hpp/cpp`)
**Neue Datei.**

- `CentroidRouter` Klasse:
  - `centroids_`: vector<vector<double>> [N × 90] — pro Concept ein Centroid
  - `route(h, K=4) → vector<pair<ConceptId, double>>` — Top-K via Dot-Product
  - `update_centroids(assignments)` — K-Means-artiges Update
  - `init_from_embeddings(ConceptEmbeddingStore&)` — Initialisierung aus existierenden Embeddings
- Optimierung: Batch-Dot-Product, optional CUDA

**Effort:** ~200 LOC, 1.5h

#### Step 12: Gated Residual PoE + Ignition (`convergence/gated_residual.hpp/cpp`)
**Neue Datei.**

- `GatedResidualPoE` Klasse:
  - `W_gate_`: [32 × 90], `b_gate_`: [32] — 2912 Parameter
  - `converge(h, G_out, L_out) → fused`
    - `γ = σ(W_gate · h + b_gate)`
    - `ε = L_out - G_out`
    - `fused = G_out + γ ⊙ ε`
  - `compute_agreement(G_out, L_out) → float`
    - `1 - ||ε|| / (||G|| + ||L||)`
  - `check_ignition(agreement) → enum { FAST, DELIBERATE, CONFLICT }`
- Bias-Initialisierung: `b_gate = log(local_precision / global_precision)`

**Effort:** ~150 LOC, 1h

#### Step 13: Convergence Pipeline (`convergence/convergence_pipeline.hpp/cpp`)
**Neue Datei.** Abhängig von Step 10, 11, 12.

- `ConvergencePipeline` Klasse — verdrahtet alles:
  1. `kan_.forward_layer1(h)` → k1
  2. `router_.route(h, K=4)` → indices, weights (parallel mit L1)
  3. `kan_.project_for_cm(k1)` → k1_proj
  4. ConceptModels forwarden mit h ⊕ k1_proj → L_out (Summe gewichtet nach Router)
  5. `kan_.forward_layer2_3(k1, L_out)` → G_out
  6. `gate_.converge(h, G_out, L_out)` → fused
  7. Ignition-Check: bei CONFLICT → Expansion + Re-Iteration
- `forward(h) → {fused, G_out, L_out, agreement}`
- `train_step(h, target, lr)` → Backward durch gesamte Pipeline

**ACHTUNG:** ConceptModel-Integration:
- Aktuelle ConceptModels haben INPUT_DIM=20
- Design will h(90) ⊕ k1_proj(32) = 122
- **Pragmatischer Ansatz:** Statt alle 25769 CMs zu resizen, nutze die existierenden CMs mit ihrem 20-dim Input und addiere einen NEUEN lightweight "KAN-Adapter" pro CM: kleiner Linear(32→4) der k1_proj auf 4 Dims projiziert und zu den 20-dim Input hinzufügt → INPUT_DIM wird 24. Das ist der minimale Eingriff.
- ALTERNATIVE (ehrgeiziger): INPUT_DIM auf 24 resizen, neuen 4-dim Slot befüllen. Erfordert Persistence-Migration.

**Effort:** ~350 LOC, 3h

#### Step 14: Makefile + Build-Integration
**Neue Datei:** `backend/Makefile.convergence`

- Kompiliert alle convergence/*.cpp
- Linkt gegen existierende Objekte (deep_kan.o, concept_model.o, etc.)
- Test-Binary: `test_convergence`

**Effort:** ~50 LOC, 30min

---

### PHASE D: ConceptModel Anpassungen

#### Step 15: CM Training mit Epistemic-Weighted Targets (`cmodel/concept_trainer.cpp`)
**Existierende Datei modifizieren.**

- `compute_target()` Funktion: `target = rel.weight * epistemic_trust(source) * epistemic_trust(target)`
- `epistemic_trust()` Mapping: FACT→1.0, THEORY→0.7, HYPOTHESIS→0.4, SPECULATION→0.2, INVALIDATED→0.0
- Relation-Category als One-Hot-Features (8-dim) zum Training-Input hinzufügen
- Hard Negative Sampling: Nearest-Neighbor-basiert via ConceptEmbeddingStore::most_similar()

**Effort:** ~120 LOC Änderungen, 1.5h

#### Step 16: CM KAN-Input-Port (optional, Phase 2 Training)
**Abhängig von Step 13 — kann auch zunächst übersprungen werden.**

- ConceptModel INPUT_DIM erweitern (20 → 24 mit 4-dim KAN-Projection)
- Persistence: neue Version der Flat-Array-Serialisierung
- Backward-Compat: alte Models laden mit Null-Padding

**Effort:** ~200 LOC, 2h (inkl. Persistence-Migration)

---

### PHASE E: Language + Decoder Anpassungen

#### Step 17: Relation-Aware Decoder (`language/kan_decoder.cpp`)
**Existierende Datei modifizieren.**

- Output-Dimensionen nach RelationCategory aufteilen (Section 9 des Designs)
- Category-Slices für die 32-dim Output-Logits
- `decode(logits, relation_type)` Methode

**Effort:** ~80 LOC, 45min

#### Step 18: Language Templates
**Existierende Dateien modifizieren** — Template-Engine erweitern.

- Per-Relation Syntax-Templates (Section 13)
- Epistemic-Modality Framing basierend auf Trust + EpistemicType
- CONTRADICTS-Template hinzufügen

**Effort:** ~60 LOC, 30min

---

### PHASE F: Dead Code Cleanup (Audit #16, #17, #20)

#### Step 19: Cleanup
- `e_init_` und `c_init_` aus ConceptModel entfernen (spart 32 doubles/model)
- `ConceptPatternWeights` bewerten: in `predict_refined()` integriert oder entfernen
- `execute()` und `execute_with_goal()` in ThinkingPipeline mergen

**Effort:** ~100 LOC Entfernung, 1h

---

### PHASE G: Tests

#### Step 20: Tests für alle neuen/modifizierten Komponenten

Tests für:
- RelationBehavior Config — Lookup, alle Categories abgedeckt
- STM inhibit_concept — Korrekte Reduktion, Clamp auf 0
- Spreading Activation — Inhibitorische Propagation bei CONTRADICTS, Per-Category Gewichtung
- Embedding Training — Repulsive Nudge bei OPPOSITION
- Property Inheritance — PART_OF → REVERSE, per-type Decay
- Epistemic Demotion — Ratio-basiert statt instant
- ConvergenceKAN — Forward-Pass Dimensionen
- CentroidRouter — Top-K korrekt, Ties
- GatedResidualPoE — Convergence Formel, Ignition Thresholds
- ConvergencePipeline — End-to-End Forward

**Effort:** ~500 LOC, 3h

---

## Abhängigkeitsgraph

```
Step 1: RelationBehavior Config
├── Step 2: STM Inhibition (parallel)
├── Step 3: Spreading Fix (braucht 1+2)
├── Step 4: Embedding Fix (braucht 1)
├── Step 5: Inheritance Fix (braucht 1)
├── Step 17: Relation Decoder (braucht 1)
└── Step 18: Language Templates (braucht 1)

Step 6: Demotion Fix (unabhängig)
Step 7: Salience Fix (unabhängig)
Step 8: Block Boundaries Fix (unabhängig)

Step 9: Convergence Directory (unabhängig)
├── Step 10: Convergence KAN (braucht 9)
├── Step 11: Router (braucht 9)
├── Step 12: Gated Residual (braucht 9)
└── Step 13: Pipeline (braucht 10+11+12)
    ├── Step 14: Makefile (braucht 13)
    ├── Step 15: CM Epistemic Training (parallel)
    └── Step 16: CM KAN-Input-Port (braucht 13)

Step 19: Cleanup (unabhängig, niedrige Priorität)
Step 20: Tests (nach allen anderen)
```

## Kritischer Pfad

**1 → 2 → 3** (Spreading-Fix blockiert Convergence-Korrektheit)
**9 → 10+11+12 → 13 → 14** (Convergence-Pipeline ist das Hauptziel)

## Empfohlene Implementierungsreihenfolge

| Batch | Steps | Beschreibung | Parallelisierbar |
|-------|-------|-------------|------------------|
| **Batch 1** | 1, 6, 7, 8, 9 | Foundation + Quick Fixes | Ja, alle 5 parallel |
| **Batch 2** | 2, 4, 5 | STM Inhibition, Embedding Fix, Inheritance Fix | Ja, parallel |
| **Batch 3** | 3 | Spreading Activation Fix (braucht 1+2) | Nein |
| **Batch 4** | 10, 11, 12 | Convergence-Komponenten | Ja, parallel |
| **Batch 5** | 13, 14, 15 | Pipeline Wiring + CM Training + Build | Teilweise parallel |
| **Batch 6** | 16, 17, 18 | CM KAN-Port, Decoder, Templates | Ja, parallel |
| **Batch 7** | 19, 20 | Cleanup + Tests | Sequentiell |

## Geschätzte Gesamtgröße

- **Neue Dateien:** ~8 Dateien, ~1200 LOC
- **Modifizierte Dateien:** ~12 Dateien, ~500 LOC Änderungen
- **Tests:** ~500 LOC
- **Total:** ~2200 LOC

## Offene Fragen (vor Implementierung zu klären)

1. **ConceptModel INPUT_DIM:** Design will 122-dim Input (90+32). Aktuelle CMs haben 20-dim. Sollen wir:
   - (a) CMs auf 24-dim erweitern (20 + 4 projected KAN dims) — minimal-invasiv
   - (b) CMs komplett auf 122-dim redesignen — breaking change, alle CMs müssen neu trainiert werden
   - (c) Erstmal ohne KAN-Input-Port starten, CMs nutzen nur ihre 20-dim — v1-Kompatibilität

2. **RelationCategory-Mapping:** Design hat 8 Kategorien, Code hat 9. Vorschlag:
   - SIMILARITY → Übernimmt ASSOCIATIVE-Rolle (spread_weight 0.6, alpha 0.2)
   - FUNCTIONAL → Übernimmt DEPENDENCY-Rolle (spread_weight 0.8, alpha 0.2, inherit true)
   - EPISTEMIC → Eigene Behavior (spread_weight 0.7, alpha 0.15, für SUPPORTS)
   - CUSTOM_CATEGORY → Fallback-Defaults
   - CREATIVE fehlt → zu CUSTOM_CATEGORY oder als neue Kategorie hinzufügen

3. **Convergence-KAN vs. Language-KAN:** Das Design-Dokument beschreibt den Convergence-KAN als separates Modul. Der existierende DeepKAN in `language/deep_kan.hpp` ist der Sprach-KAN. Bestätigung: wir erstellen einen NEUEN KAN für die Convergence-Pipeline, wiederverwendend die `EfficientKANLayer`-Klasse.
