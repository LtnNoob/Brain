# REFACTOR_REVIEW — Code Review of GRAPH_ARCHITECTURE_REFACTOR.md

**Datum:** 2026-02-12
**Reviewer:** Claude (Code-Reviewer, systematischer Abgleich gegen bestehenden Code)
**Status:** REVIEW COMPLETE
**Ergebnis:** 14 Probleme identifiziert, davon 5 kritisch

---

## Zusammenfassung

Das GRAPH_ARCHITECTURE_REFACTOR.md ist ein ambitionierter und architektonisch sinnvoller Plan. Die Vision eines Dual-Mode-Systems (Global Dynamics + Focus Traversal) ist schlüssig. **Aber:** Der Plan enthält mehrere falsche Annahmen über den bestehenden Code, phantom-Typen die nicht existieren, und unterschätzt die Integrationsarbeit erheblich. Die Behauptung "nur Erweiterungen, keine Breaking Changes" (§8) ist **falsch**.

---

## Kritische Probleme (Showstopper)

### K1. `VecN` existiert nicht — Phantom-Typ

**Refactor-Plan §2.1:** `VecN embedding;` als "existing" Feld in Concept
**FOCUS_CURSOR_DESIGN §2:** `VecN context_embedding_`, `VecN context_at_entry`, etc.

**Tatsächlicher Code:**
- `backend/micromodel/micro_model.hpp:29` definiert `Vec10 = std::array<double, 10>`
- `EMBED_DIM = 10` (Konstante)
- Kein `VecN` Typ existiert irgendwo im Codebase
- Es gibt auch kein `Vec16` — das ist ein geplantes Upgrade (Phase 1)

**Impact:** Jede Stelle im Refactor-Plan und FOCUS_CURSOR_DESIGN, die `VecN` verwendet, referenziert einen nicht-existenten Typ. Das betrifft:
- `FocusCursorState::accumulated_energy` — ok (double)
- `CursorView::context_embedding` — muss `Vec10` sein
- `TraversalStep::context_at_entry` — muss `Vec10` sein
- `Concept::embedding` — existiert überhaupt nicht (siehe K2)

**Fix:** Entweder `Vec10` verwenden (aktueller Stand) oder zuerst Phase 1 (Vec16-Upgrade) implementieren und dann einen `VecN`-Alias definieren. FOCUS_CURSOR_DESIGN muss durchgehend korrigiert werden.

---

### K2. `Concept` Struct existiert nicht — Falscher Struct-Name und falsche Felder

**Refactor-Plan §2.1 zeigt:**
```cpp
struct Concept {
    ConceptId id;
    std::string label;
    VecN embedding;                    // "existing"
    double activation_score;           // NEW
    double salience_score;             // NEW
    double structural_confidence;      // NEW
    double semantic_confidence;        // NEW
    EpistemicMetadata epistemic;       // "existing"
};
```

**Tatsächlicher Code** (`backend/ltm/long_term_memory.hpp:22-53`):
```cpp
struct ConceptInfo {
    ConceptId id;
    std::string label;
    std::string definition;           // ← EXISTIERT, im Plan FEHLEND
    EpistemicMetadata epistemic;

    ConceptInfo() = delete;           // ← Kein Default-Konstruktor!
    ConceptInfo(ConceptId, const std::string&, const std::string&, EpistemicMetadata);
};
```

**Probleme:**
1. Struct heißt `ConceptInfo`, nicht `Concept`
2. **`VecN embedding` existiert NICHT** in ConceptInfo. Embeddings werden nicht per-Concept gespeichert, sondern über `EmbeddingManager` verwaltet (nur Relation-Embeddings und Named Context-Embeddings)
3. `definition` Feld existiert im Code, fehlt im Plan
4. `ConceptInfo() = delete` — Kein Default-Konstruktor. Neue Felder erfordern Konstruktor-Änderung
5. Neue Felder = ABI-Break + Serialisierungs-Break (PersistentLTM, WAL, Checkpoints)

**Impact:** §2.1 ist in seiner aktuellen Form nicht umsetzbar ohne grundlegende Anpassungen.

---

### K3. `Relation` Struct — Falscher Name, falsche Feldnamen

**Refactor-Plan §2.2 zeigt:**
```cpp
struct Relation {
    ConceptId source_id;
    ConceptId target_id;
    RelationType type;
    double weight;
    double dynamic_weight;             // NEW
    double inhibition_factor;          // NEW
    double structural_strength;        // NEW
};
```

**Tatsächlicher Code** (`backend/ltm/relation.hpp:28-65`):
```cpp
struct RelationInfo {
    RelationId id;          // ← EXISTIERT, im Plan FEHLEND
    ConceptId source;       // ← Heißt "source", nicht "source_id"
    ConceptId target;       // ← Heißt "target", nicht "target_id"
    RelationType type;
    double weight;

    RelationInfo() = delete;  // ← Kein Default-Konstruktor!
};
```

**Probleme:**
1. Struct heißt `RelationInfo`, nicht `Relation`
2. `RelationId id` fehlt im Plan — wird aber überall im Code verwendet
3. Feldnamen `source_id`/`target_id` vs tatsächlich `source`/`target`
4. Kein Default-Konstruktor — neue Felder erfordern Konstruktor-Anpassung
5. `RelationInfo` hat `clamp_weight()` Validierung — neue Felder brauchen eigene Validierung
6. ABI-Break + Serialisierungs-Break

---

### K4. `get_concept_embedding()` existiert nicht

**FOCUS_CURSOR_DESIGN §3.3 und §3.4:**
```
target_emb ← embeddings.get_concept_embedding(to)
new_emb ← embeddings.get_concept_embedding(new_concept)
```

**Tatsächlicher Code** (`backend/micromodel/embedding_manager.hpp`):
```cpp
class EmbeddingManager {
    const Vec10& get_relation_embedding(RelationType type) const;
    const Vec10& get_context_embedding(const std::string& name);
    Vec10 make_context_embedding(const std::string& name) const;
    Vec10 make_target_embedding(size_t context_hash, uint64_t source_id, uint64_t target_id) const;
};
```

**Es gibt kein `get_concept_embedding()`** — weder in EmbeddingManager noch irgendwo sonst im Codebase (grep bestätigt: 0 Treffer).

**Impact:** Die Kernalgorithmen `compute_weight()` und `accumulate_context()` des FocusCursors basieren auf einer Methode, die nicht existiert. Das ist ein fundamentaler Design-Fehler im FOCUS_CURSOR_DESIGN, der sich ins Refactor-Dokument durchzieht.

**Mögliche Lösung:** `EmbeddingManager` um per-Concept-Embeddings erweitern oder `make_target_embedding()` als Workaround nutzen. Aber das erfordert ein Embedding-Storage-Konzept, das aktuell nicht existiert.

---

### K5. "Nur Erweiterungen, keine Breaking Changes" — FALSCH

**Refactor-Plan §8:**
> "Only extensions required — no breaking changes to existing data structures."

**Tatsächliche Situation:**
1. `ConceptInfo` und `RelationInfo` haben `= delete` Default-Konstruktoren — neue Felder erzwingen Konstruktor-Änderungen
2. Jede Änderung an diesen Structs bricht:
   - `PersistentLTM` Serialisierung (`backend/persistent/persistent_ltm.hpp`)
   - WAL-Format (`backend/persistent/wal.hpp`)
   - Checkpoint-Format (`backend/persistent/checkpoint_manager.hpp`)
   - STM-Snapshot-Format
3. Alle bestehenden Tests und Demo-Programme, die ConceptInfo/RelationInfo konstruieren, müssen angepasst werden
4. Binary-Kompatibilität mit gespeicherten Brain-Daten geht verloren (Migration nötig)

**§8 muss komplett umgeschrieben werden.** Die Persistence-Layer ist eben NICHT unberührt.

---

## Erhebliche Probleme (Müssen vor Implementierung gelöst werden)

### E1. CuriosityEngine API-Mismatch

**Refactor-Plan §6 definiert:**
```cpp
struct CuriosityTrigger {
    enum class TriggerType {
        LOW_CONFIDENCE, UNEXPLORED_REGION, CONTRADICTION, HIGH_NOVELTY
    };
    TriggerType trigger_type;
    ConceptId target_node_id;
    double priority;
};
```

**Tatsächlicher Code** (`backend/curiosity/curiosity_trigger.hpp`):
```cpp
enum class TriggerType {
    SHALLOW_RELATIONS,
    MISSING_DEPTH,
    LOW_EXPLORATION,
    RECURRENT_WITHOUT_FUNCTION,
    UNKNOWN
};

struct CuriosityTrigger {
    TriggerType type;                              // ← "type", nicht "trigger_type"
    ContextId context_id;                          // ← existiert, im Plan fehlend
    std::vector<ConceptId> related_concept_ids;    // ← Liste, nicht einzelne ConceptId
    std::string description;                       // ← existiert, im Plan fehlend
    // KEIN priority Feld!
};
```

**Impact:** Die TriggerTypes sind komplett verschieden, die Struct-Felder stimmen nicht überein. Das Refactor-Dokument definiert eine Fantasie-API. Entweder muss CuriosityEngine komplett umgebaut werden (was §8 widerspricht) oder die Refactor-API muss an den bestehenden Code angepasst werden.

Zusätzlich: `curiosity.get_pending_triggers()` (§6) existiert nicht. Die tatsächliche API ist `observe_and_generate_triggers(observations)` — sie hat keine Queue, sondern generiert Trigger on-demand aus Observationen.

---

### E2. Fehlende RelationTypes in Beispielen

**FOCUS_CURSOR_DESIGN §4, §9 verwendet:**
- `PRODUCES` — existiert nicht
- `REQUIRES` — existiert nicht
- `USES` — existiert nicht
- `SOURCE` — existiert nicht
- `HAS_PART` — existiert nicht (es gibt `PART_OF`)

**Tatsächliche RelationType-Enum** (`backend/memory/active_relation.hpp:11-22`):
```cpp
enum class RelationType {
    IS_A, HAS_PROPERTY, CAUSES, ENABLES, PART_OF,
    SIMILAR_TO, CONTRADICTS, SUPPORTS, TEMPORAL_BEFORE, CUSTOM
};
```

**Impact:** Die Beispiele ("Was passiert wenn Eis schmilzt?", "Warum brauchen Pflanzen Licht?") verwenden Relations, die im System nicht darstellbar sind. Die Beispiele müssen entweder angepasst werden (alles auf `CUSTOM` mappen?) oder RelationType muss erweitert werden (was EmbeddingManager betrifft, da `NUM_RELATION_TYPES = 10` hardcoded ist und `relation_embeddings_` ein Fixed-Size-Array ist).

---

### E3. ThinkingPipeline wird ignoriert

**Refactor-Plan §7** führt `Brain19ControlLoop` ein als unified control loop.

**Bestehender Code:** `ThinkingPipeline` (`backend/core/thinking_pipeline.hpp`) orchestriert bereits einen 10-Schritte Thinking-Cycle:
1. Activate seeds → 2. Spreading Activation → 3. Salience + Focus → 4. RelevanceMaps → 5. Combine Maps → 6. ThoughtPaths → 7. Curiosity → 8. Understanding → 9. KAN Validation → 10. Return result

Der Refactor-Plan erwähnt `ThinkingPipeline` **nicht einmal**. Es ist unklar:
- Wird ThinkingPipeline ersetzt?
- Wird sie erweitert?
- Wie koexistieren Brain19ControlLoop und ThinkingPipeline?

Dasselbe gilt für `SystemOrchestrator`, der alle Subsysteme besitzt und orchestriert.

---

### E4. CognitiveDynamics Überlappung ungeklärt

Der "Global Dynamics Operator" (§5.1) überlappt stark mit dem existierenden `CognitiveDynamics` System:

| Feature | CognitiveDynamics (existiert) | Global Dynamics (Plan) |
|---------|------------------------------|----------------------|
| Spreading Activation | ✅ Ja | ✅ Ja (anders) |
| Salience | ✅ Ja | Teilweise |
| Focus Management | ✅ Ja (Miller's 7±2) | ❌ Nein |
| Thought Path Ranking | ✅ Ja (Beam Search) | ❌ Nein |
| Inhibition | ❌ Nein | ✅ Ja (neu) |
| Damping/Decay | ✅ Ja (damping_factor) | ✅ Ja (anders) |
| Trust-weighted | ✅ Ja | ❌ Nicht erwähnt |

Der Plan sagt nicht, ob CognitiveDynamics bleiben, ersetzt, oder mit Global Dynamics zusammengeführt werden soll.

---

### E5. KAN-Policy ist eine Phantomkomponente

**Refactor-Plan §7:** `kan_policy_.evaluate(cursor)` und `policy.should_shift_focus`, `policy.suggested_relation`.

**Tatsächlicher KANAdapter** (`backend/adapter/kan_adapter.hpp`):
- Kann KAN-Module erstellen, trainieren, evaluieren
- Evaluierung: `evaluate_kan_module(uint64_t module_id, const std::vector<double>& inputs)` → raw `vector<double>`
- Hat KEINE Cursor-Awareness
- Hat KEIN TraversalPolicy-Konzept
- Kennt keine FocusCursor-Zustände

Die FOCUS_CURSOR_DESIGN (§5.2) definiert ein detailliertes KAN-Policy-Interface mit `PolicyInput` struct und spezifischen Feature-Dimensionen ([context_embedding, depth, view_summary, query_embedding] → ℝ⁵²). Dieses Interface existiert nicht und müsste komplett neu gebaut werden, einschließlich Training-Data-Pipeline.

---

## Moderate Probleme (Sollten gelöst werden)

### M1. LanguageEngine existiert nicht

**Refactor-Plan §7:** `language_engine_.generate(result)` — kein solches Klasse/Interface existiert.
Es gibt `ChatInterface` und `OllamaClient`, aber keine dedizierte "LanguageEngine" für Template-basierte Satzgenerierung aus Cursor-Ketten. Muss neu erstellt werden.

---

### M2. Embedding-Dimension 10 vs Plan-unspezifisch

Der Plan nutzt `VecN` ohne Dimension zu spezifizieren. Der bestehende Code ist hart auf `EMBED_DIM = 10` verdrahtet:
- `Vec10 = std::array<double, 10>`
- `Mat10x10 = std::array<double, 100>`
- `FLAT_SIZE = 430` für MicroModel-Serialisierung
- `NUM_RELATION_TYPES = 10` (Zufall: gleiche Zahl, aber unrelated)

Wenn Phase 1 (Vec16) zuerst kommen soll, muss `EMBED_DIM` parametrisiert werden — was weitreichende Änderungen an MicroModel, EmbeddingManager, Persistence, und Training bedeutet.

---

### M3. FocusCursor Constructor-Dependencies

Der FocusCursor nimmt `const LongTermMemory&` als Referenz. Aber im Multi-Threaded System gibt es `SharedLTM` mit Lock-Guards. Der FocusCursor-Konstruktor müsste entweder:
- Nur im subsystem_mtx_-geschützten Kontext laufen, oder
- SharedLTM statt LongTermMemory nutzen

Dasselbe gilt für `MicroModelRegistry&` (→ `SharedRegistry`) und `EmbeddingManager&` (→ `SharedEmbeddings`).

---

## Konsistenz FOCUS_CURSOR_DESIGN ↔ GRAPH_ARCHITECTURE_REFACTOR

### Konsistent:
- FocusCursorState-Grundstruktur (§2.4 ↔ FOCUS_CURSOR §2)
- Navigations-Algorithmen (step, deepen, branch, backtrack, shift_focus)
- TraversalResult / TraversalStep Structs
- Terminierung via max_depth / min_weight_threshold
- MicroModel-Integration (predict(e, c))
- STM-Persistenz-Konzept (persist_to_stm)

### Inkonsistent:
1. **GoalState:** GRAPH_ARCHITECTURE_REFACTOR fügt GoalState hinzu (§2.3), FOCUS_CURSOR_DESIGN kennt keine Goals. Der Refactor sagt FocusCursor wird "goal-aware" (§5.2: "Now goal-directed, not just greedy traversal"), aber die FocusCursor-Algorithmen im FOCUS_CURSOR_DESIGN haben keine Goal-Integration.
2. **ExplorationMode:** Der Refactor fügt `GOAL_DIRECTED` als ExplorationMode hinzu (§2.4), FOCUS_CURSOR_DESIGN hat nur GREEDY/EXPLORATORY (implizit).
3. **Energy Budget:** `accumulated_energy` (§2.4) ist neu im Refactor, fehlt in FOCUS_CURSOR_DESIGN.
4. **Termination Logic:** Der Refactor hat eine erweiterte Terminierungslogik (§3) mit 6 Kriterien inkl. Goal-Completion und Energy-Exhaustion. FOCUS_CURSOR_DESIGN hat nur 2 (max_depth, min_weight).
5. **Conflict Resolution:** §4 (α·structural + β·semantic + γ·activation) ist komplett neu, FOCUS_CURSOR_DESIGN hat kein Äquivalent.

**Fazit:** FOCUS_CURSOR_DESIGN ist ein Subset des Refactor-Plans, aber der Refactor-Plan klärt nicht, welche Teile von FOCUS_CURSOR_DESIGN aktualisiert werden müssen. Die Aussage "See FOCUS_CURSOR_DESIGN.md for the full 860-line specification — it remains valid" (Appendix A) stimmt nur teilweise.

---

## Betroffene Dateien (vollständige Liste)

| Datei | Änderungstyp | Begründung |
|-------|-------------|-----------|
| `backend/ltm/long_term_memory.hpp` | **BREAKING** | ConceptInfo um 4 Felder erweitern, Konstruktor ändern |
| `backend/ltm/relation.hpp` | **BREAKING** | RelationInfo um 3 Felder erweitern, Konstruktor ändern |
| `backend/common/types.hpp` | Erweiterung | Ggf. VecN-Alias, GoalState-IDs |
| `backend/micromodel/micro_model.hpp` | Ggf. Erweiterung | Vec10→VecN wenn Phase 1 zuerst |
| `backend/micromodel/embedding_manager.hpp` | **ERWEITERUNG** | `get_concept_embedding()` fehlt komplett |
| `backend/micromodel/micro_model_registry.hpp` | Unverändert | API passt |
| `backend/memory/stm.hpp` | Unverändert | API passt für persist_to_stm |
| `backend/memory/active_relation.hpp` | Ggf. Erweiterung | RelationType um PRODUCES, REQUIRES etc. |
| `backend/cognitive/cognitive_dynamics.hpp` | **UNKLAR** | Relationship zu Global Dynamics ungeklärt |
| `backend/curiosity/curiosity_engine.hpp` | **BREAKING** | API-Mismatch: TriggerType, Felder, Methoden |
| `backend/curiosity/curiosity_trigger.hpp` | **BREAKING** | Komplett andere Enum + Struct |
| `backend/adapter/kan_adapter.hpp` | **ERWEITERUNG** | KAN-Policy Interface fehlt komplett |
| `backend/core/thinking_pipeline.hpp` | **UNKLAR** | Ersetzt? Erweitert? Koexistenz? |
| `backend/core/system_orchestrator.hpp` | **ERWEITERUNG** | Brain19ControlLoop integrieren |
| `backend/persistent/persistent_ltm.hpp` | **BREAKING** | ConceptInfo/RelationInfo Layout-Änderung |
| `backend/persistent/wal.hpp` | **BREAKING** | Serialisierungsformat-Änderung |
| `backend/persistent/checkpoint_manager.hpp` | **BREAKING** | Checkpoint-Format-Änderung |
| `backend/persistent/stm_snapshot.hpp` | Prüfen | Ggf. betroffen durch neue Felder |
| **NEU: `backend/cursor/`** | **NEU** | focus_cursor.hpp/cpp, config, traversal_result |
| **NEU: GoalState** | **NEU** | Neuer Struct (Ort unklar — common? cursor?) |
| **NEU: GlobalDynamicsOperator** | **NEU** | Neue Klasse |
| **NEU: LanguageEngine** | **NEU** | Neue Klasse |
| **NEU: Brain19ControlLoop** | **NEU** | Neue Klasse |

---

## Aufwandsschätzung: Korrektur

Der Plan schätzt **11-14 Tage**. Unter Berücksichtigung der identifizierten Probleme:

| Zusätzlicher Aufwand | Begründung |
|--------------------|-----------|
| +1-2 Tage | ConceptInfo/RelationInfo-Umbau + alle Stellen die sie konstruieren |
| +1-2 Tage | Persistence-Layer Migration (WAL, Checkpoints, PersistentLTM) |
| +1 Tag | EmbeddingManager: get_concept_embedding() implementieren |
| +0.5 Tag | RelationType erweitern + EmbeddingManager-Anpassung |
| +1 Tag | CuriosityEngine API-Umbau |
| +1-2 Tage | KAN-Policy Interface (TraversalPolicy, Training Pipeline) |
| +1 Tag | ThinkingPipeline/SystemOrchestrator Integration |
| +0.5 Tag | Thread-Safety (SharedLTM/SharedRegistry statt direkte Refs) |

**Korrigierte Schätzung: 18-24 Tage** (statt 11-14)

---

## Empfehlungen

### 1. Vor Implementierung
- [ ] Alle Struct-Namen und Felder im Plan an den tatsächlichen Code anpassen (ConceptInfo, RelationInfo, source/target)
- [ ] `VecN` → `Vec10` (oder Phase 1 zuerst machen und dann refactoren)
- [ ] Klären: Was passiert mit ThinkingPipeline und CognitiveDynamics?
- [ ] `get_concept_embedding()` Design klären — woher kommen per-Concept-Embeddings?
- [ ] Persistence-Migration planen (alter Checkpoint → neues Format)

### 2. Implementierungsreihenfolge (korrigiert)
1. ~~Extended fields~~ → Zuerst EmbeddingManager erweitern (Concept-Embeddings)
2. ConceptInfo/RelationInfo erweitern (mit Persistence-Migration)
3. FocusCursor implementieren (mit korrekten Typen)
4. GoalState
5. CuriosityEngine API-Umbau
6. KAN-Policy Interface
7. Global Dynamics / CognitiveDynamics Konsolidierung
8. Unified Control Loop
9. Tests

### 3. Designentscheidungen die vorher getroffen werden MÜSSEN
- Werden Concept-Embeddings im ConceptInfo gespeichert oder extern verwaltet?
- CognitiveDynamics → ersetzen, erweitern, oder parallel?
- ThinkingPipeline → ersetzen oder erweitern?
- RelationType → erweitern oder alles auf CUSTOM?
- VecN Dimension: 10 bleiben oder sofort auf 16?

---

## Fazit

Das Refactor-Dokument beschreibt eine **gute Zielarchitektur**, aber es wurde offensichtlich **ohne detaillierte Code-Analyse** geschrieben. Die Annahmen über bestehende Interfaces, Struct-Namen, verfügbare Methoden und Seiteneffekte auf die Persistence-Layer sind in mehreren Fällen **faktisch falsch**.

Vor Beginn der Implementierung müssen die 5 kritischen Probleme (K1-K5) und die 5 erheblichen Probleme (E1-E5) gelöst werden, sonst läuft die Implementierung in Sackgassen.

Der Aufwand wird auf **18-24 Abende** statt der geschätzten 11-14 geschätzt. Das ist keine dramatische Abweichung, aber die Planung sollte die realen Dependencies und Breaking Changes berücksichtigen.
