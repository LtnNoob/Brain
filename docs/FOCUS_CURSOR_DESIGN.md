# Focus Cursor Design — Brain19's Denkprozess

**Datum:** 2026-02-11  
**Autor:** Feature-Designer (Subagent)  
**Status:** Design-Dokument  
**Ersetzt:** InteractionLayer (Phase 2 aus IMPLEMENTATION_PLAN_V2)

---

## 1. Executive Summary

Der FocusCursor ist ein **navigierender Aufmerksamkeits-Cursor** der den Wissensgraph sequentiell traversiert. Statt alle relevanten Nodes gleichzeitig zu aktivieren (InteractionLayer/Hopfield), folgt der Cursor gewichteten Ketten — wie ein Mensch, der in einer Mindmap von Konzept zu Konzept springt.

**Kernprinzip:** Nur der aktuelle Fokus + direkte Nachbarn sind aktiv. Kein Explosionsproblem. Kein Attractor-Finding. Stattdessen: kontrolliertes, sequentielles Denken.

---

## 2. Klassen-Spezifikation (C++ Header)

```cpp
#pragma once

#include "../common/types.hpp"
#include "../micromodel/micro_model.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/stm.hpp"

#include <vector>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <string>

namespace brain19 {

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// A single weighted neighbor visible from the current focus position.
struct NeighborView {
    ConceptId id;
    RelationType relation;
    double weight;            // from predict(e, c) — dynamic, not stored
    std::string label;        // concept label for language generation
    bool is_outgoing;         // direction of relation
};

/// Snapshot of what the cursor "sees" at its current position.
struct CursorView {
    ConceptId focus;                          // current concept
    std::string focus_label;
    size_t depth;                             // steps from seed
    std::vector<NeighborView> neighbors;      // sorted by weight descending
    double focus_activation;                  // predict score of current concept in context
    VecN context_embedding;                   // accumulated context
};

/// One step in the traversal history.
struct TraversalStep {
    ConceptId concept;
    RelationType relation_used;       // relation that led here
    double weight_at_entry;           // weight when this step was taken
    VecN context_at_entry;            // context embedding at this point
};

/// Result of a complete cursor traversal (the "thought").
struct TraversalResult {
    std::vector<TraversalStep> chain;           // the full path
    std::vector<ConceptId> concept_sequence;    // just the concept IDs in order
    std::vector<RelationType> relation_sequence;// relations between consecutive concepts
    size_t total_steps = 0;
    bool terminated_by_threshold = false;       // stopped because weight < min
    bool terminated_by_depth = false;           // stopped because max depth
    bool terminated_by_user = false;            // stopped by external signal
    double chain_score = 0.0;                   // product of weights along chain
};

/// Configuration for the FocusCursor.
struct FocusCursorConfig {
    // Navigation
    double min_weight_threshold = 0.15;     // stop if best neighbor < this
    size_t max_depth = 12;                  // max traversal depth
    size_t max_neighbors_evaluated = 30;    // limit neighbor evaluation per step
    
    // Context accumulation
    double context_decay = 0.85;            // how much old context is retained per step
    double new_context_weight = 0.15;       // weight of new concept's embedding
    
    // Widening
    size_t default_widen_k = 5;             // how many neighbors get_view returns
    
    // Branching
    size_t max_branches = 4;                // max parallel cursors from branch()
    
    // Stability (from STABILITY_ANALYSIS)
    double damping = 0.1;                   // damping factor for weight computation
    bool normalize_weights = true;          // normalize neighbor weights to sum=1
};

// ============================================================================
// FOCUS CURSOR
// ============================================================================

/// The FocusCursor navigates Brain19's knowledge graph sequentially.
///
/// INVARIANTS:
/// - Only current focus + direct neighbors are evaluated (no explosion)
/// - Weights are DYNAMIC: computed via MicroModel::predict(e, c) at each step
/// - Context embedding accumulates along the path
/// - History is maintained as "working memory" / "thought protocol"
///
/// REPLACES: InteractionLayer (Phase 2 of IMPLEMENTATION_PLAN_V2)
///
class FocusCursor {
public:
    FocusCursor(
        const LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        FocusCursorConfig config = {}
    );
    
    // ─── Initialization ──────────────────────────────────────────────────
    
    /// Position cursor on a seed concept. Resets history and depth.
    void seed(ConceptId concept_id);
    
    /// Position cursor with an initial context embedding (e.g. from query).
    void seed(ConceptId concept_id, const VecN& initial_context);
    
    // ─── Core Navigation ─────────────────────────────────────────────────
    
    /// Follow the strongest-weighted neighbor. Returns the new focus concept.
    /// Returns nullopt if no neighbor exceeds min_weight_threshold.
    std::optional<ConceptId> step();
    
    /// Step to a specific neighbor. Returns false if not a direct neighbor.
    bool step_to(ConceptId target);
    
    /// Go back one step. Returns false if already at seed.
    bool backtrack();
    
    /// Automatically follow the strongest chain until threshold or max_depth.
    /// Returns the full chain traversed.
    TraversalResult deepen();
    
    // ─── Focus Control ───────────────────────────────────────────────────
    
    /// Change which relation type is preferred for weight computation.
    /// When set, neighbors connected by this relation type get a bonus.
    void shift_focus(RelationType preferred_relation);
    
    /// Clear relation preference (evaluate all relations equally).
    void clear_focus_preference();
    
    /// Widen: return more neighbors at current position (increases view).
    CursorView widen(size_t k = 0);  // 0 = use config default
    
    // ─── View ────────────────────────────────────────────────────────────
    
    /// Get current view: focus concept + weighted neighbors.
    CursorView get_view() const;
    
    /// Get current position.
    ConceptId position() const { return current_; }
    
    /// Get current depth.
    size_t depth() const { return depth_; }
    
    /// Get accumulated context embedding.
    const VecN& context() const { return context_embedding_; }
    
    /// Get full traversal history.
    const std::vector<TraversalStep>& history() const { return history_; }
    
    /// Get the concept chain as IDs.
    std::vector<ConceptId> chain() const;
    
    /// Is the cursor positioned (has seed been called)?
    bool is_active() const { return active_; }
    
    // ─── Branching ───────────────────────────────────────────────────────
    
    /// Create parallel cursors for the top-K neighbors.
    /// Each branch starts at a different neighbor of the current focus.
    /// The original cursor is NOT modified.
    std::vector<FocusCursor> branch(size_t k = 0) const;  // 0 = config default
    
    // ─── Scoring ─────────────────────────────────────────────────────────
    
    /// Score the current chain (product of weights along the path).
    double chain_score() const;
    
    /// Build a TraversalResult from current state.
    TraversalResult result() const;
    
    // ─── Configuration ───────────────────────────────────────────────────
    
    const FocusCursorConfig& config() const { return config_; }
    void set_config(const FocusCursorConfig& config) { config_ = config; }
    
private:
    // References (not owned)
    const LongTermMemory& ltm_;
    MicroModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    
    // State
    FocusCursorConfig config_;
    ConceptId current_ = 0;
    size_t depth_ = 0;
    VecN context_embedding_{};
    std::vector<TraversalStep> history_;
    bool active_ = false;
    
    // Optional relation focus
    std::optional<RelationType> preferred_relation_;
    
    // ─── Internal ────────────────────────────────────────────────────────
    
    /// Evaluate all neighbors of current concept, return sorted by weight.
    std::vector<NeighborView> evaluate_neighbors() const;
    
    /// Compute weight for a specific neighbor using MicroModel::predict.
    double compute_weight(ConceptId from, ConceptId to, RelationType rel) const;
    
    /// Update context embedding when moving to a new concept.
    void accumulate_context(ConceptId new_concept);
    
    /// Record a step in history.
    void record_step(ConceptId concept, RelationType rel, double weight);
};

// ============================================================================
// FOCUS CURSOR MANAGER
// ============================================================================

/// Manages multiple cursors for a single query. Handles multi-seed scenarios
/// and cursor lifecycle within STM.
class FocusCursorManager {
public:
    FocusCursorManager(
        const LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        ShortTermMemory& stm,
        FocusCursorConfig config = {}
    );
    
    /// Process a query: find seeds, create cursors, traverse, return results.
    struct QueryResult {
        std::vector<TraversalResult> chains;      // one per seed cursor
        std::vector<ConceptId> all_activated;      // union of all concepts touched
        TraversalResult best_chain;                // highest-scoring chain
    };
    
    QueryResult process_seeds(
        const std::vector<ConceptId>& seeds,
        const VecN& query_context
    );
    
    /// Store cursor state into STM as "thought protocol".
    void persist_to_stm(
        ContextId ctx,
        const TraversalResult& result
    );
    
private:
    const LongTermMemory& ltm_;
    MicroModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    ShortTermMemory& stm_;
    FocusCursorConfig config_;
};

} // namespace brain19
```

---

## 3. Algorithmen

### 3.1 Core: `step()` — Folge stärkster Gewichtung

```
ALGORITHM step():
    IF NOT active THEN RETURN nullopt
    
    neighbors ← evaluate_neighbors()
    IF neighbors.empty() OR neighbors[0].weight < config.min_weight_threshold THEN
        RETURN nullopt   // dead end or below threshold
    
    best ← neighbors[0]  // highest weight
    
    // Record current position in history
    record_step(current_, best.relation, best.weight)
    
    // Move cursor
    current_ ← best.id
    depth_ ← depth_ + 1
    
    // Update context embedding
    accumulate_context(best.id)
    
    RETURN best.id
```

### 3.2 `evaluate_neighbors()` — Dynamische Gewichtung via MicroModels

```
ALGORITHM evaluate_neighbors():
    outgoing ← ltm.get_outgoing_relations(current_)
    incoming ← ltm.get_incoming_relations(current_)
    
    neighbors ← []
    
    FOR EACH rel IN outgoing ∪ incoming:
        target ← (rel is outgoing) ? rel.target : rel.source
        
        // DYNAMIC weight from MicroModel — NOT static!
        weight ← compute_weight(current_, target, rel.type)
        
        // Apply relation focus bonus
        IF preferred_relation_ AND rel.type == preferred_relation_:
            weight ← weight * 1.5  // 50% bonus for preferred relation
        
        neighbors.append(NeighborView{target, rel.type, weight, ...})
    
    // Normalize if configured
    IF config.normalize_weights:
        total ← sum(n.weight for n in neighbors)
        IF total > 0:
            FOR EACH n IN neighbors: n.weight /= total
    
    // Sort descending by weight
    SORT neighbors BY weight DESC
    
    // Limit to max_neighbors_evaluated
    TRUNCATE neighbors TO config.max_neighbors_evaluated
    
    RETURN neighbors
```

### 3.3 `compute_weight()` — MicroModel predict mit akkumuliertem Kontext

```
ALGORITHM compute_weight(from, to, rel_type):
    model ← registry.get_model(from)
    IF model == NULL: RETURN 0.0
    
    e ← embeddings.get_relation_embedding(rel_type)
    
    // Context = accumulated context embedding (NOT just target embedding)
    // This is what makes weights change as the cursor moves!
    c ← context_embedding_
    
    // Mix in target's own embedding for specificity
    target_emb ← embeddings.get_concept_embedding(to)
    FOR i IN 0..EMBED_DIM-1:
        c_mixed[i] ← 0.7 * context_embedding_[i] + 0.3 * target_emb[i]
    
    score ← model->predict(e, c_mixed)  // σ(eᵀ·(W·c+b)) ∈ (0,1)
    
    // Apply damping
    score ← score * (1.0 - config.damping)
    
    RETURN score
```

### 3.4 `accumulate_context()` — Kontext-Embedding Akkumulation

```
ALGORITHM accumulate_context(new_concept):
    new_emb ← embeddings.get_concept_embedding(new_concept)
    
    FOR i IN 0..EMBED_DIM-1:
        context_embedding_[i] ← config.context_decay * context_embedding_[i]
                                + config.new_context_weight * new_emb[i]
    
    // L2-normalize to prevent drift
    norm ← sqrt(sum(context_embedding_[i]² for i))
    IF norm > 1e-10:
        FOR i: context_embedding_[i] /= norm
```

### 3.5 `deepen()` — Automatische Tiefentraversierung

```
ALGORITHM deepen():
    result ← TraversalResult{}
    
    WHILE depth_ < config.max_depth:
        next ← step()
        IF next == nullopt:
            result.terminated_by_threshold ← true
            BREAK
        result.chain.append(history_.back())
        result.concept_sequence.append(next)
        result.total_steps++
    
    IF depth_ >= config.max_depth:
        result.terminated_by_depth ← true
    
    result.chain_score ← chain_score()
    RETURN result
```

### 3.6 `branch()` — Parallele Cursor

```
ALGORITHM branch(k):
    neighbors ← evaluate_neighbors()
    k ← min(k OR config.max_branches, neighbors.size())
    
    branches ← []
    FOR i IN 0..k-1:
        cursor ← FocusCursor(ltm_, registry_, embeddings_, config_)
        cursor.context_embedding_ ← context_embedding_  // copy context
        cursor.history_ ← history_                        // copy history
        cursor.depth_ ← depth_
        cursor.active_ ← true
        cursor.current_ ← neighbors[i].id
        cursor.depth_++
        cursor.accumulate_context(neighbors[i].id)
        cursor.record_step(neighbors[i].id, neighbors[i].relation, neighbors[i].weight)
        branches.append(cursor)
    
    RETURN branches
```

### 3.7 `shift_focus()` — Gewichtungsfokus ändern

```
ALGORITHM shift_focus(rel_type):
    preferred_relation_ ← rel_type
    // Next call to step() or get_view() will apply the 1.5x bonus
    // to neighbors connected by this relation type.
    // This changes WHICH path the cursor follows without backtracking.
```

---

## 4. Konkreter Algorithmus: "Was passiert wenn Eis schmilzt?"

### Schritt-für-Schritt mit Zahlen

```
INPUT: "Was passiert wenn Eis schmilzt?"

═══════════════════════════════════════════════════════════
PHASE 1: Seed-Konzepte finden
═══════════════════════════════════════════════════════════
Tokenize → Label-Search in LTM:
  "Eis"       → ConceptId 247
  "Schmelzen" → ConceptId 891
  "passiert"  → kein Treffer (zu generisch)
  "wenn"      → kein Treffer (Funktionswort)

Seeds: [247, 891]
Query-Context q via KAN-Encoder: [0.69, 0.38, -0.11, 0.85, ...]₁₆

═══════════════════════════════════════════════════════════
PHASE 2: Cursor 1 auf "Eis" (247)
═══════════════════════════════════════════════════════════

cursor1.seed(247, q)
  current_ = 247 ("Eis")
  depth_ = 0
  context_embedding_ = q (Query-Context)

── Step 0: get_view() ──────────────────────────────────
  evaluate_neighbors():
    MicroModel(247).predict(e_IS_A, context)         → 503("Aggregatzustand_fest")  = 0.72
    MicroModel(247).predict(e_CAUSES, context)        → 891("Schmelzen")            = 0.89
    MicroModel(247).predict(e_HAS_PROPERTY, context)  → 156("Temp < 0°C")           = 0.61
    MicroModel(247).predict(e_PART_OF, context)        → 712("Wasser")               = 0.55
    MicroModel(247).predict(e_HAS_PROPERTY, context)  → 921("Kristallstruktur")     = 0.43

  Query enthält "Schmelzen" → shift_focus(CAUSES)
    → CAUSES-Relationen bekommen 1.5x Bonus:
    
  Nach Bonus + Normalisierung:
    891("Schmelzen")            = 0.89 × 1.5 / Σ = 0.38  ← STÄRKSTE
    503("Aggregatzustand_fest") = 0.72 / Σ        = 0.20
    156("Temp < 0°C")          = 0.61 / Σ        = 0.17
    712("Wasser")              = 0.55 / Σ        = 0.15
    921("Kristallstruktur")    = 0.43 / Σ        = 0.10

── Step 1: step() → Schmelzen (891) ────────────────────
  current_ = 891
  depth_ = 1
  history_ = [{247, CAUSES, 0.38}]
  context_embedding_ = 0.85 × q + 0.15 × embed(891) → normalisiert

  evaluate_neighbors() vom neuen Standpunkt "Schmelzen":
    MicroModel(891).predict(e_PRODUCES, context')     → 712("Wasser")           = 0.85
    MicroModel(891).predict(e_REQUIRES, context')     → 334("Wärmeenergie")     = 0.78
    MicroModel(891).predict(e_IS_A, context')         → 445("Phasenübergang")   = 0.71
    MicroModel(891).predict(e_CAUSES, context')       → 556("Volumenänderung")  = 0.42

  CAUSES-Bonus immer noch aktiv:
    712("Wasser")          = 0.85 × 1.5 / Σ = 0.37  ← (PRODUCES ≈ CAUSES hier)
    334("Wärmeenergie")    = 0.78 / Σ        = 0.23
    445("Phasenübergang")  = 0.71 / Σ        = 0.21
    556("Volumenänderung") = 0.42 / Σ        = 0.12

  Anmerkung: PRODUCES bekommt CAUSES-Bonus weil es kausal ist.

── Step 2: step() → Wasser (712) ──────────────────────
  current_ = 712
  depth_ = 2
  history_ = [{247, CAUSES, 0.38}, {891, PRODUCES, 0.37}]
  context_embedding_ = 0.85 × context' + 0.15 × embed(712)

  evaluate_neighbors() vom Standpunkt "Wasser":
    MicroModel(712).predict(e_HAS_PROPERTY, context'') → 801("flüssig")       = 0.82
    MicroModel(712).predict(e_IS_A, context'')         → 503("Aggregatzustand")= 0.68
    MicroModel(712).predict(e_USED_FOR, context'')     → 999("Trinken")       = 0.31
    MicroModel(712).predict(e_CAUSES, context'')       → 444("Verdunstung")   = 0.28

  CAUSES-Bonus:
    801("flüssig")        = 0.82 / Σ = 0.35
    503("Aggregatzustand")= 0.68 / Σ = 0.29
    444("Verdunstung")    = 0.28 × 1.5 / Σ = 0.18
    999("Trinken")        = 0.31 / Σ = 0.13

  Kein Nachbar über min_weight=0.15 in Kontext "Eis schmilzt"?
  → Doch, "flüssig" hat 0.35. Aber die Kette ist semantisch "fertig":
  → System (KAN-Policy) entscheidet: STOP (siehe §5.4)

═══════════════════════════════════════════════════════════
PHASE 3: Ergebnis
═══════════════════════════════════════════════════════════

Cursor 1 Kette:
  Eis(247) →[CAUSES]→ Schmelzen(891) →[PRODUCES]→ Wasser(712)
  
  chain_score = 0.38 × 0.37 = 0.14  (Produkt der Gewichte)
  depth = 2
  terminated_by = KAN-Policy (genug Information)

Cursor 2 (parallel auf Seed "Schmelzen"):
  Schmelzen(891) →[PRODUCES]→ Wasser(712)
  Redundant mit Cursor 1 → merge

═══════════════════════════════════════════════════════════
PHASE 4: Language Generation
═══════════════════════════════════════════════════════════

Chain: [Eis, CAUSES, Schmelzen, PRODUCES, Wasser]
Relations: [CAUSES, PRODUCES]

Template-Mapping:
  X →[CAUSES]→ Y →[PRODUCES]→ Z
  = "Wenn X Y, entsteht Z"
  = "Wenn Eis schmilzt, entsteht Wasser."

OUTPUT: "Wenn Eis schmilzt, entsteht Wasser."
```

---

## 5. Integration mit bestehenden Systemen

### 5.1 Integration mit MicroModels

**Bei jedem Fokuswechsel:**
1. MicroModel des aktuellen Konzepts wird geladen: `registry.get_model(current_)`
2. Alle Nachbarn werden via `predict(e_rel, context)` bewertet
3. **Context ist NICHT statisch** — er akkumuliert entlang des Pfads
4. Dadurch ändern sich die Gewichtungen bei jedem Schritt dynamisch

**Warum das funktioniert:**
- `predict(e, c) = σ(eᵀ·(W·c+b))` — das `c` (Context) ändert sich
- Am Anfang: context = Query-Embedding → "Eis schmelzen"  
- Nach Step 1: context enthält auch "Schmelzen" → PRODUCES-Relationen werden relevanter
- Die MicroModels **lernen** welche Nachbarn in welchem Kontext relevant sind

### 5.2 Integration mit KAN

KAN-Nodes steuern die **Traversierungs-Policy** — sie entscheiden:

```cpp
// In SystemOrchestrator, nach Cursor-Erstellung:
struct TraversalPolicy {
    bool should_continue;        // weitermachen oder stoppen?
    RelationType suggested_focus; // welcher Relationstyp als nächstes?
    bool should_branch;          // hier verzweigen?
    size_t branch_count;         // wie viele Branches?
};

// KAN evaluiert den aktuellen Cursor-Zustand:
TraversalPolicy policy = kan_adapter_->evaluate_traversal(
    cursor.context(),           // aktuelles Context-Embedding
    cursor.depth(),             // wie tief sind wir?
    cursor.get_view(),          // was sehen wir?
    query_embedding             // was war die Frage?
);

// B-Splines lernen optimale Strategien:
// - "Bei kausalen Fragen: folge CAUSES-Ketten tiefer"
// - "Bei Definitionsfragen: stopp nach IS_A + HAS_PROPERTY"
// - "Bei Vergleichsfragen: branch an jedem Konzept"
```

**KAN-Training:**
- Input: (context_embedding, depth, view_summary, query_embedding) → ℝ⁵²
- Output: (continue_prob, relation_type_probs, branch_prob) → ℝ¹²
- Training: Reinforcement aus erfolgreichen Antworten

### 5.3 Integration mit STM (Short-Term Memory)

Der FocusCursor **lebt im STM**:

```cpp
void FocusCursorManager::persist_to_stm(ContextId ctx, const TraversalResult& result) {
    // Jedes traversierte Konzept wird im STM aktiviert
    for (size_t i = 0; i < result.chain.size(); ++i) {
        const auto& step = result.chain[i];
        
        // Activation = weight × depth_decay
        double activation = step.weight_at_entry * std::pow(0.9, i);
        
        stm_.activate_concept(ctx, step.concept,
            activation, ActivationClass::CONTEXTUAL);
        
        // Relationen zwischen aufeinanderfolgenden Konzepten
        if (i > 0) {
            stm_.activate_relation(ctx,
                result.chain[i-1].concept, step.concept,
                step.relation_used, step.weight_at_entry);
        }
    }
}
```

**Denkprotokoll-Persistenz:**
- Die gesamte `TraversalResult` Kette bleibt als "Denkprotokoll" im STM
- Bei Rückkehr zur gleichen Frage: STM-Aktivierungen boosten die Seed-Auswahl
- Decay lässt alte Denkprotokolle natürlich verblassen
- Konzepte die oft traversiert werden bekommen Core-Decay-Rate (langsamer)

### 5.4 Integration mit KAN-Policy (wann stoppen?)

```cpp
// KAN-Policy Node entscheidet ob der Cursor weitermachen soll.
// Input-Features:
struct PolicyInput {
    double depth_ratio;          // depth / max_depth
    double best_neighbor_weight; // stärkster Nachbar
    double chain_score;          // bisherige Ketten-Score
    double context_similarity;   // cos(context, query) — driften wir ab?
    double query_coverage;       // wie viele Query-Seeds wurden berührt?
};

// Trainiert auf: "War die Antwort gut wenn wir hier gestoppt haben?"
// Output: continue_probability ∈ (0, 1)

bool should_continue = kan_policy.evaluate(policy_input) > 0.5;
```

### 5.5 Integration mit Language Engine

**Jeder Cursor-Schritt = potentieller Satz-Baustein:**

```
Kette: [Eis, CAUSES, Schmelzen, PRODUCES, Wasser]

Relation-Templates:
  CAUSES    → "verursacht", "führt zu", "wenn X dann Y"
  PRODUCES  → "entsteht", "wird zu", "erzeugt"
  IS_A      → "ist ein", "gehört zu"
  HAS_PROP  → "hat die Eigenschaft", "ist"
  PART_OF   → "ist Teil von", "besteht aus"

Chain → Satz:
  Eis →[CAUSES]→ Schmelzen →[PRODUCES]→ Wasser
  = "Wenn Eis [CAUSES] Schmelzen, [PRODUCES] Wasser"
  = "Wenn Eis schmilzt, entsteht Wasser."

Komplexere Kette:
  Auto →[HAS_PART]→ Motor →[USES]→ Benzin →[PRODUCES]→ CO2
  = "Ein Auto hat einen Motor, der Benzin verwendet und CO2 produziert."
```

Der **Pfad IST die Antwort**. Kein separater Decoder nötig für einfache Fragen.
Für komplexere Formulierungen: KAN-Decoder verfeinert die Template-Ausgabe.

---

## 6. Vergleich: FocusCursor vs. InteractionLayer

| Aspekt | InteractionLayer (alt) | FocusCursor (neu) |
|--------|----------------------|-------------------|
| **Aktivierung** | Alle N Nodes gleichzeitig | Nur aktueller Node + Nachbarn |
| **Komplexität/Step** | O(N² × d) pro Iteration | O(d̄ × d) pro Step (d̄ = Grad) |
| **Memory** | O(N × d) Aktivierungsvektoren | O(depth × d) History |
| **Konvergenz** | Braucht 14-30 Zyklen, Stabilitätsbeweis nötig | Kein Konvergenz-Problem |
| **Determinismus** | Attractor abhängig von Init | Deterministisch (stärkste Kette) |
| **Erklärbarkeit** | Fixpunkt schwer interpretierbar | Kette direkt lesbar |
| **Steuerbarkeit** | Context-Node (indirekt) | shift_focus, step_to (direkt) |
| **Stabilität** | Erfordert Normalisierung, Symmetrie, λ-Tuning | Inhärent stabil (bounded depth) |
| **Sprachgeneration** | Fixpunkt → Decoder (komplex) | Kette → Template (direkt) |
| **Skalierung** | O(N²) bei großem Subgraph | O(depth × d̄), N-unabhängig |
| **Hub-Problem** | Hubs destabilisieren Dynamik | Hubs = mehr Nachbarn, aber begrenzt |

### Warum besser?

1. **Kein Explosionsproblem:** InteractionLayer muss N=50-1000 Nodes gleichzeitig propagieren. FocusCursor evaluiert nur ~10-30 Nachbarn pro Schritt.

2. **Keine Stabilitäts-Mathematik nötig:** InteractionLayer braucht Lyapunov-Beweis, Spektralradius-Monitoring, Normalisierung, Symmetrie-Erzwingung. FocusCursor hat max_depth als einzige Begrenzung.

3. **Deterministisch und erklärbar:** "Warum hast du gesagt X?" → "Weil die Kette A→B→C→X die stärkste war." Bei InteractionLayer: "Der Attractor-Zustand hat sich ergeben" (Blackbox).

4. **Natürliche Sprachgeneration:** Die Kette IST die Argumentation. Jede Relation = ein Satz-Baustein. Kein aufwändiger Decoder nötig.

5. **Kontrollierbar:** User kann sagen "geh tiefer", "geh zurück", "fokussier dich auf Ursachen". Bei InteractionLayer gibt es keine solche Steuerung.

---

## 7. Änderungen am IMPLEMENTATION_PLAN_V2

### Ersetzt:
- **Phase 2 (InteractionLayer):** Komplett ersetzt durch FocusCursor
- **Phase 3 (Context-Node):** Unnötig — Context wird im Cursor akkumuliert
- **Phase 4 (Excitation/Inhibition):** Teilweise ersetzt — Inhibition durch `min_weight_threshold`, Excitation durch `shift_focus`

### Bleibt:
- **Phase 0 (Kritische Fixes):** Unverändert
- **Phase 1 (Vec16):** Unverändert — FocusCursor profitiert von d=16
- **Phase 5 (KAN-MiniLLM):** Angepasst — KAN steuert Traversierungs-Policy statt InteractionLayer
- **Phase 6 (Inkrementelles Training):** Angepasst — Re-Training wenn Cursor unerwartete Ergebnisse liefert
- **Phase 7 (Stabilitäts-Tests):** Stark vereinfacht — keine Konvergenz/Spektralradius-Tests nötig

### Neuer Plan:

| Phase | Beschreibung | Aufwand | Status |
|-------|-------------|---------|--------|
| 0 | Kritische Fixes | 0.5 Tage | unverändert |
| 1 | Vec16 Upgrade | 1-2 Tage | unverändert |
| **2** | **FocusCursor** | **2-3 Tage** | **NEU (ersetzt InteractionLayer)** |
| 3 | ~~Context-Node~~ | ~~1-2 Tage~~ | **ENTFÄLLT** |
| 4 | ~~Excitation/Inhibition~~ | ~~2-3 Tage~~ | **ENTFÄLLT** |
| **2b** | **KAN Traversal Policy** | **1-2 Tage** | **NEU** |
| 5 | KAN-MiniLLM + Language | 5-7 Tage | angepasst |
| 6 | Inkrementelles Training | 2-3 Tage | angepasst |
| 7 | Tests | 1-2 Tage | vereinfacht |

**Zeitersparnis: ~4-6 Tage** (Phase 3+4 entfallen, Phase 7 einfacher)

---

## 8. Neue Dateien

```
backend/cursor/
├── focus_cursor.hpp          # FocusCursor + FocusCursorManager Klassen
├── focus_cursor.cpp          # Implementation
├── focus_cursor_config.hpp   # FocusCursorConfig struct
└── traversal_result.hpp      # TraversalResult, TraversalStep, NeighborView, CursorView

tests/
├── test_focus_cursor.cpp     # Unit-Tests für Navigation
└── test_cursor_integration.cpp  # Integration mit STM, MicroModels, KAN
```

### LoC-Schätzung

| Datei | LoC |
|-------|:----|
| focus_cursor.hpp | ~180 |
| focus_cursor.cpp | ~350 |
| focus_cursor_config.hpp | ~40 |
| traversal_result.hpp | ~60 |
| test_focus_cursor.cpp | ~300 |
| test_cursor_integration.cpp | ~200 |
| **Gesamt** | **~1130** |

vs. InteractionLayer: ~510 LoC + ~130 (Phase 4) + ~65 (Phase 3) = ~705 LoC

FocusCursor ist etwas mehr Code, dafür entfallen 3 komplexe Phasen.

---

## 9. Zweites Beispiel: "Warum brauchen Pflanzen Licht?"

```
Seeds: [Pflanze(101), Licht(205)]
Query-Context: q = encode("Warum brauchen Pflanzen Licht")

Cursor auf Pflanze(101), shift_focus(REQUIRES):

Step 0: Pflanze(101)
  Nachbarn:
    Photosynthese(42)  [REQUIRES→]  w=0.91 × 1.5(REQUIRES-Bonus) = norm 0.41
    Wasser(712)        [REQUIRES→]  w=0.84 × 1.5 = norm 0.38
    Blatt(333)         [HAS_PART→]  w=0.72 = norm 0.10
    Wurzel(334)        [HAS_PART→]  w=0.65 = norm 0.09

Step 1: → Photosynthese(42)
  context' = 0.85·q + 0.15·embed(Photosynthese)
  
  Nachbarn:
    Licht(205)         [REQUIRES→]  w=0.93 × 1.5 = norm 0.39  ← Query-Seed!
    CO2(440)           [REQUIRES→]  w=0.81 × 1.5 = norm 0.34
    Glucose(441)       [PRODUCES→]  w=0.88 = norm 0.15
    Chlorophyll(442)   [USES→]      w=0.76 = norm 0.12

Step 2: → Licht(205)  [Query-Seed erreicht!]
  context'' = 0.85·context' + 0.15·embed(Licht)
  
  Nachbarn:
    Energie(500)       [IS_A→]      w=0.79 = norm 0.32
    Sonne(501)         [SOURCE→]    w=0.74 = norm 0.30
    Wellenlänge(502)   [HAS_PROP→]  w=0.52 = norm 0.21

  KAN-Policy: query_coverage = 2/2 (beide Seeds berührt) → STOP

Kette: Pflanze →[REQUIRES]→ Photosynthese →[REQUIRES]→ Licht
Relations: [REQUIRES, REQUIRES]

Template: X →[REQUIRES]→ Y →[REQUIRES]→ Z
= "X braucht Y, und Y braucht Z"
= "Pflanzen brauchen Photosynthese, und Photosynthese braucht Licht."

Oder besser (Chain-Reversal für "Warum"-Fragen):
= "Pflanzen brauchen Licht für die Photosynthese."
```

---

## 10. Stabilitätsgarantien

Der FocusCursor hat inhärente Stabilitätsgarantien **ohne** die komplexe Mathematik der InteractionLayer:

| Garantie | Mechanismus |
|----------|-------------|
| **Terminierung** | `max_depth` begrenzt Traversierung auf 12 Schritte |
| **Bounded Activation** | `predict()` gibt σ(x) ∈ (0,1) zurück — immer bounded |
| **Keine Zyklen** | History-Tracking: bereits besuchte Konzepte werden übersprungen |
| **Keine Explosion** | Nur Nachbarn des aktuellen Knotens werden evaluiert |
| **Kontrollierte Tiefe** | KAN-Policy oder User entscheidet wann stoppen |
| **Damping** | `config.damping` reduziert Gewichte (Stabilitäts-Analyse §6.2 gilt analog) |
| **Normalisierung** | `normalize_weights` sorgt für Σw=1 pro Schritt |
| **Schwellen** | `min_weight_threshold` verhindert Traversierung schwacher Kanten |

Die STABILITY_ANALYSIS-Empfehlungen werden wie folgt übernommen:
- **λ=0.1** → `config.damping = 0.1` (analog)
- **Normalisierung: zwingend** → `config.normalize_weights = true`
- **max_cycles=20-30** → `config.max_depth = 12` (weniger nötig, da kein Konvergenz-Problem)
- **Symmetrie** → nicht nötig (unidirektionale Traversierung)

---

*Dieses Design-Dokument beschreibt den FocusCursor als Ersatz für die InteractionLayer. Es folgt exakt Felix' Vision eines sequentiell navigierenden Aufmerksamkeits-Cursors. Die Implementierung kann sofort nach Phase 1 (Vec16) beginnen.*
