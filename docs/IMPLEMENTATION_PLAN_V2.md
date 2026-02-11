# Brain19 Implementation Plan V2

**Datum:** 2026-02-11
**Autor:** Lead Architect (konsolidiert aus 5 Analysen)
**Status:** Approved for implementation

---

## Übersicht

| Phase | Beschreibung | Aufwand | Abhängigkeit |
|-------|-------------|---------|-------------|
| 0 | Kritische Fixes | 0.5 Tage | — |
| 1 | Dimensionalitäts-Upgrade 10→16 | 1-2 Tage | Phase 0 |
| 2 | InteractionLayer | 3-5 Tage | Phase 1 |
| 3 | Context-Node | 1-2 Tage | Phase 2 |
| 4 | Excitation/Inhibition | 2-3 Tage | Phase 2 |
| 5 | KAN-MiniLLM Hybrid Engine | 5-7 Tage | Phase 1 |
| 6 | Inkrementelles Training | 2-3 Tage | Phase 2 |
| 7 | Stabilitäts-Tests | 2-3 Tage | Phase 2 |

**Kritischer Pfad:** 0 → 1 → 2 → 3/4/6/7 (parallel)
**Gesamtdauer:** ~4-5 Wochen (1 Entwickler), ~2-3 Wochen (2 parallel)

---

## Phase 0: Kritische Fixes (P0, sofort)

### 0.1 Validation Loop verdrahten

**Problem:** `ThinkingResult.validated_hypotheses` wird in `thinking_pipeline.cpp:84` berechnet, aber in `system_orchestrator.cpp:383-407` **nie ausgelesen**. Validierte Hypothesen gehen verloren.

**Datei:** `backend/core/system_orchestrator.cpp`
**Stelle:** Nach Zeile 407 (nach `return chat_->ask_with_context(...)`)
**Genauer:** In der Funktion `ask()`, zwischen `run_thinking_cycle()` und `chat_->ask_with_context()`, nach Zeile ~385.

**Einzufügender Code (~50 Zeilen):**

```cpp
// In SystemOrchestrator::ask(), nach run_thinking_cycle():
// Anchor validated hypotheses back to LTM
for (const auto& vr : thinking_result.validated_hypotheses) {
    if (!vr.validated) continue;

    // Store validated hypothesis as new concept
    EpistemicMetadata meta(
        vr.assessment.metadata.type,    // THEORY or HYPOTHESIS from KAN
        EpistemicStatus::ACTIVE,
        vr.assessment.metadata.trust
    );

    auto hyp_id = ltm_->store_concept(
        vr.original_hypothesis,         // hypothesis_statement as label
        vr.assessment.reasoning,        // KAN reasoning as definition
        meta
    );

    // Add SUPPORTS relations from evidence
    for (const auto& evidence_cid : salient_ids) {
        ltm_->add_relation(evidence_cid, hyp_id,
            RelationType::SUPPORTS, meta.trust);
    }

    // Create + train MicroModel for new concept
    if (registry_) {
        registry_->create_model(hyp_id);
        if (trainer_ && embeddings_) {
            auto* model = registry_->get_model(hyp_id);
            if (model) {
                auto samples = trainer_->generate_samples(
                    hyp_id, *embeddings_, *ltm_);
                MicroTrainingConfig cfg;
                model->train(samples, cfg);
            }
        }
    }
}

// Feed evolution AFTER anchoring
run_evolution_after_thinking(thinking_result);
```

**Prüfe:** Ist `run_evolution_after_thinking()` bereits aufgerufen? → Ja, Zeile 592. Muss sichergestellt werden, dass der Validation-Anchor **davor** passiert.

**LoC:** ~45 neue Zeilen
**Risiko:** Gering — alle verwendeten APIs existieren und sind getestet.

### 0.2 .gitignore erweitern

**Datei:** `.gitignore` (existiert bereits)

**Hinzufügen:**
```
build/
*.d
*.dSYM/
.cache/
compile_commands.json
brain19_data/
checkpoints/
*.bin
```

**LoC:** ~8 Zeilen

### 0.3 ensure_models_for() nur für neue Konzepte

**Datei:** `backend/core/system_orchestrator.cpp`
**Stelle:** Nach `ingest_text()` Aufruf (suche `ensure_models_for`)

**Änderung:** Statt `registry_->ensure_models_for(*ltm_)` nur neue IDs:
```cpp
for (auto new_id : result.stored_concept_ids) {
    if (!registry_->has_model(new_id)) {
        registry_->create_model(new_id);
    }
}
```

**LoC:** ~5 geändert

### Akzeptanzkriterien Phase 0
- [ ] `validated_hypotheses` werden in LTM geschrieben (Unit-Test: mock KanValidator → check LTM)
- [ ] `make brain19` kompiliert ohne Warnings
- [ ] `.gitignore` enthält alle Build-Artefakte
- [ ] `ensure_models_for` scannt nicht mehr alle Konzepte

---

## Phase 1: Dimensionalitäts-Upgrade Vec10 → Vec16

### Begründung (aus STABILITY_ANALYSIS.md §3.4)

d=16 ist der optimale Kompromiss:
- VCdim: 100 → 256 (+156%)
- RAM: 3.4 MB → 4.2 MB (+24%)
- FLOPs/predict: 110 → 272 (+147%, immer noch <1μs)
- Max komfortable Nachbarn: 50 → 100

### 1.1 Zentrale Konstanten ändern

**Datei:** `backend/micromodel/micro_model.hpp`

```cpp
// ALT:
static constexpr size_t EMBED_DIM = 10;
static constexpr size_t FLAT_SIZE = 430;
using Vec10 = std::array<double, EMBED_DIM>;
using Mat10x10 = std::array<double, EMBED_DIM * EMBED_DIM>;

// NEU:
static constexpr size_t EMBED_DIM = 16;
// FLAT_SIZE = 256(W) + 16(b) + 16(e_init) + 16(c_init) + TrainingState
// TrainingState: 256(dW_m) + 16(db_m) + 256(dW_v) + 16(db_v) + 16(e_grad) + 16(c_grad) + 5 scalars + reserved
// TrainingState reserved anpassen: 55 → anpassen damit FLAT_SIZE stimmt
static constexpr size_t FLAT_SIZE = 944;
// 256 + 16 + 16 + 16 + 256 + 16 + 256 + 16 + 16 + 16 + 5 + 55 = 944
using VecN = std::array<double, EMBED_DIM>;        // war Vec10
using MatNxN = std::array<double, EMBED_DIM * EMBED_DIM>;  // war Mat10x10
// Aliase für Backward-Kompatibilität:
using Vec10 = VecN;
using Mat10x10 = MatNxN;
```

**FLAT_SIZE Berechnung:**
- W: 16×16 = 256
- b: 16
- e_init: 16
- c_init: 16
- TrainingState: 256(dW_m) + 16(db_m) + 256(dW_v) + 16(db_v) + 16(e_grad) + 16(c_grad) + 5(scalars) + 55(reserved) = 636
- **Total: 256 + 16 + 16 + 16 + 636 = 940**

→ Reserved von 55 auf 59 anpassen → FLAT_SIZE = 944 (8-Byte aligned)

### 1.2 TrainingState anpassen

**Datei:** `backend/micromodel/micro_model.hpp`

```cpp
struct TrainingState {
    MatNxN dW_momentum{};       // 256
    VecN   db_momentum{};       //  16
    MatNxN dW_variance{};       // 256
    VecN   db_variance{};       //  16
    VecN   e_grad_accum{};      //  16
    VecN   c_grad_accum{};      //  16
    double timestep = 0.0;
    double last_loss = 0.0;
    double best_loss = 1e9;
    double total_samples = 0.0;
    double reserved_scalar = 0.0;
    std::array<double, 59> reserved{};  // war 55, jetzt 59 für Alignment
};
```

### 1.3 Betroffene Dateien (vollständige Liste)

| Datei | Änderung | LoC |
|-------|----------|-----|
| `backend/micromodel/micro_model.hpp` | EMBED_DIM, FLAT_SIZE, Typen, TrainingState | ~30 |
| `backend/micromodel/micro_model.cpp` | Automatisch via Konstanten (Loops nutzen EMBED_DIM) | 0 |
| `backend/micromodel/persistence.cpp` | MODEL_SIZE Konstante automatisch, aber Binary-Format bricht! | ~10 |
| `backend/micromodel/embedding_manager.hpp` | Return-Typen VecN (alias bleibt) | ~5 |
| `backend/micromodel/embedding_manager.cpp` | Embedding-Initialisierung (Schleifen nutzen EMBED_DIM) | ~5 |
| `backend/micromodel/relevance_map.cpp` | Keine Änderung (nutzt Vec10 alias) | 0 |
| `backend/micromodel/micro_trainer.cpp` | Keine Änderung (nutzt Vec10 alias) | 0 |

### 1.4 Backward-Kompatibilität für Checkpoints

**Problem:** Bestehende `.bin` Checkpoints haben 430 doubles pro Model. Neue haben 944.

**Lösung:** Versions-Header in Persistence:

```cpp
// backend/micromodel/persistence.cpp
static constexpr uint32_t PERSISTENCE_VERSION = 2;  // war implizit 1
static constexpr size_t MODEL_SIZE_V1 = 8 + 430 * 8;  // 3448 bytes
static constexpr size_t MODEL_SIZE_V2 = 8 + 944 * 8;  // 7560 bytes

bool load_models(...) {
    uint32_t version = read_u32();
    if (version == 1) {
        // Legacy load: read 430 doubles, zero-pad to 944
        std::array<double, 430> flat_v1;
        // ... read ...
        std::array<double, FLAT_SIZE> flat_v2{};
        std::copy(flat_v1.begin(), flat_v1.end(), flat_v2.begin());
        model.from_flat(flat_v2);
    } else {
        // Normal V2 load
    }
}
```

**LoC:** ~40 Zeilen in persistence.cpp

### 1.5 EmbeddingManager anpassen

**Datei:** `backend/micromodel/embedding_manager.hpp` und `.cpp`

Die Relation-Embeddings sind hardcoded als 10D-Vektoren. Müssen auf 16D erweitert werden:

```cpp
// embedding_manager.cpp - init_relation_embeddings()
// Bestehende 10 Werte bleiben, 6 neue Dimensionen mit sin/cos-Erweiterung:
for (size_t i = 10; i < EMBED_DIM; ++i) {
    emb[i] = 0.1 * std::sin(static_cast<double>(type) * (i + 1));
}
```

**LoC:** ~15

### Akzeptanzkriterien Phase 1
- [ ] `EMBED_DIM == 16` kompiliert und alle Tests bestehen
- [ ] Bestehende V1-Checkpoints werden korrekt geladen (zero-padded)
- [ ] Neue Checkpoints werden als V2 gespeichert
- [ ] `sizeof(MicroModel)` Sanity-Check im Unit-Test
- [ ] predict()-Output bleibt in (0,1) für Random-Input

### Risiken
- **Mittel:** Alle Checkpoints müssen migriert oder neu trainiert werden
- **Gering:** Performance-Impact vernachlässigbar (<1μs mehr pro predict)

---

## Phase 2: InteractionLayer (3-5 Tage)

### Begründung
Das **fehlende Kernstück** (ARCHITECTURE_ANALYSIS Lücke 4). MicroModels kommunizieren nur über passive RelevanceMap-Aggregation. Keine echte Inter-Modell-Kopplung.

### 2.1 Neue Dateien

**`backend/interaction/interaction_layer.hpp`** (~200 LoC)
**`backend/interaction/interaction_layer.cpp`** (~350 LoC)

### 2.2 Klassendesign

```cpp
#pragma once
#include "../micromodel/micro_model.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"
#include <unordered_map>
#include <vector>

namespace brain19 {

struct InteractionConfig {
    double lambda = 0.1;           // Decay/Self-Inhibition (§6.2: optimal mit Normalisierung)
    double epsilon = 1e-4;         // Konvergenz-Schwelle
    size_t max_cycles = 25;        // Max Iterationen (§2.7: 20-30 empfohlen)
    size_t top_k_select = 10;      // Subgraph-Selektion: Top-K
    size_t neighbor_expand = 1;    // Nachbar-Expansion (1-hop)
    double beta_context = 0.3;     // Context-Node Kopplungsstärke (Phase 3)
    bool normalize_weights = true; // KRITISCH: muss true sein (§2.5)
    bool enforce_symmetry = true;  // KRITISCH: für Lyapunov-Konvergenz (§2.6)
};

struct InteractionResult {
    std::unordered_map<ConceptId, VecN> final_activations;
    size_t cycles_used = 0;
    bool converged = false;
    double final_delta = 0.0;
    double spectral_radius_estimate = 0.0;  // empirisch aus Konvergenzrate
    std::vector<double> energy_trace;        // E(t) pro Iteration
};

class InteractionLayer {
public:
    explicit InteractionLayer(InteractionConfig config = {});

    // =========================================================================
    // HAUPTMETHODE: Activation Propagation über Subgraph
    // =========================================================================
    InteractionResult propagate(
        const std::vector<ConceptId>& seed_concepts,
        const LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings
    );

    // Subgraph-Selektion: Top-K seeds + 1-hop Nachbarn
    std::vector<ConceptId> select_subgraph(
        const std::vector<ConceptId>& seeds,
        const LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings
    ) const;

    // activate() Modus: Zwischenvektor statt Skalar (§4.2 Review)
    // v = σ_elementwise(W·c + b), gibt VecN zurück
    static VecN activate(const MicroModel& model, const VecN& context);

    // Gewichtsnormalisierung (§2.5: ZWINGEND)
    // w_hat_ij = w_ij / sum_k(w_ik)
    void normalize_weights(
        std::unordered_map<ConceptId,
            std::unordered_map<ConceptId, double>>& W
    ) const;

    // Symmetrie erzwingen (§2.6: für Lyapunov)
    // w_ij = (w_ij + w_ji) / 2
    void enforce_symmetry(
        std::unordered_map<ConceptId,
            std::unordered_map<ConceptId, double>>& W
    ) const;

    // Kopplungsgewicht: w_ij = predict_i(e_rel, a_j)
    double coupling_weight(
        const MicroModel& model_i,
        const VecN& activation_j,
        const VecN& relation_embedding
    ) const;

    // Energy Function (§1 Review, korrigiert mit Decay-Term):
    // E = -0.5 * Σ w_ij * a_i · a_j + λ * Σ ∫σ⁻¹(a_i)da_i
    double compute_energy(
        const std::unordered_map<ConceptId, VecN>& activations,
        const std::unordered_map<ConceptId,
            std::unordered_map<ConceptId, double>>& W
    ) const;

    const InteractionConfig& config() const { return config_; }

private:
    InteractionConfig config_;

    // Ein Propagationsschritt
    void propagate_one_step(
        const std::vector<ConceptId>& subgraph,
        const LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        const std::unordered_map<ConceptId,
            std::unordered_map<ConceptId, double>>& W_norm,
        std::unordered_map<ConceptId, VecN>& activations
    );

    // Konvergenz-Check
    double compute_delta(
        const std::unordered_map<ConceptId, VecN>& prev,
        const std::unordered_map<ConceptId, VecN>& curr
    ) const;

    // Compute coupling matrix from current activations
    void compute_coupling_matrix(
        const std::vector<ConceptId>& subgraph,
        const LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        const std::unordered_map<ConceptId, VecN>& activations,
        std::unordered_map<ConceptId,
            std::unordered_map<ConceptId, double>>& W
    );
};

} // namespace brain19
```

### 2.3 Kernalgorithmus (Pseudocode → C++)

```cpp
InteractionResult InteractionLayer::propagate(...) {
    // 1. Subgraph selektieren
    auto subgraph = select_subgraph(seeds, ltm, registry, embeddings);

    // 2. Aktivierungen initialisieren (activate() Modus)
    std::unordered_map<ConceptId, VecN> activations;
    for (auto cid : subgraph) {
        auto* model = registry.get_model(cid);
        if (model) {
            activations[cid] = activate(*model, embeddings.recall_context());
        }
    }

    // 3. Kopplungsmatrix berechnen
    std::unordered_map<ConceptId, std::unordered_map<ConceptId, double>> W;
    compute_coupling_matrix(subgraph, ltm, registry, embeddings, activations, W);

    // 4. Normalisierung + Symmetrie (KRITISCH)
    if (config_.normalize_weights) normalize_weights(W);
    if (config_.enforce_symmetry) enforce_symmetry(W);

    // 5. Iterative Propagation
    InteractionResult result;
    for (size_t t = 0; t < config_.max_cycles; ++t) {
        auto prev = activations;
        propagate_one_step(subgraph, ltm, registry, embeddings, W, activations);

        double delta = compute_delta(prev, activations);
        result.energy_trace.push_back(compute_energy(activations, W));

        if (delta < config_.epsilon) {
            result.converged = true;
            result.cycles_used = t + 1;
            result.final_delta = delta;
            break;
        }

        // Empirischer Spektralradius
        if (t > 0 && result.energy_trace.size() >= 2) {
            double prev_delta = (t > 1) ?
                compute_delta(/* t-2 */, /* t-1 */) : delta;
            if (prev_delta > 1e-10)
                result.spectral_radius_estimate = delta / prev_delta;
        }
    }

    result.final_activations = activations;
    return result;
}
```

### 2.4 activate() Implementation

```cpp
// Neuer Modus: Gibt VecN zurück statt Skalar
VecN InteractionLayer::activate(const MicroModel& model, const VecN& c) {
    const auto& W = model.weights();
    const auto& b = model.bias();
    VecN v;
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        double sum = b[i];
        for (size_t j = 0; j < EMBED_DIM; ++j) {
            sum += W[i * EMBED_DIM + j] * c[j];
        }
        v[i] = sigmoid(sum);  // element-wise sigmoid
    }
    return v;
}
```

### 2.5 Integration in SystemOrchestrator

**Datei:** `backend/core/system_orchestrator.hpp`
```cpp
#include "../interaction/interaction_layer.hpp"
// Member:
std::unique_ptr<InteractionLayer> interaction_layer_;
```

**Datei:** `backend/core/system_orchestrator.cpp`
- Init-Stage 12.5 (nach Evolution, vor Shared Wrappers):
  ```cpp
  interaction_layer_ = std::make_unique<InteractionLayer>();
  ```
- In `run_thinking_cycle()`, nach Step 5 (RelevanceMap), vor Step 6:
  ```cpp
  if (interaction_layer_) {
      auto interaction_result = interaction_layer_->propagate(
          salient_ids, *ltm_, *registry_, *embeddings_);
      // Merge interaction activations into thinking result
      result.interaction_activations = interaction_result.final_activations;
      result.interaction_converged = interaction_result.converged;
  }
  ```

### 2.6 Dateien Übersicht

| Datei | Aktion | LoC |
|-------|--------|-----|
| `backend/interaction/interaction_layer.hpp` | NEU | ~120 |
| `backend/interaction/interaction_layer.cpp` | NEU | ~350 |
| `backend/core/system_orchestrator.hpp` | ÄNDERN (+include, +member) | +5 |
| `backend/core/system_orchestrator.cpp` | ÄNDERN (+init, +integration) | +30 |
| `backend/core/thinking_pipeline.hpp` | ÄNDERN (+interaction fields in ThinkingResult) | +5 |
| `Makefile` | ÄNDERN (+interaction_layer.cpp) | +1 |

**Gesamt: ~510 neue LoC**

### Akzeptanzkriterien Phase 2
- [ ] InteractionLayer konvergiert für Testgraph (N=50, d̄=10) in <25 Zyklen
- [ ] Energie ist monoton fallend (bei symmetrischen Gewichten)
- [ ] Normalisierte Gewichte: Zeilensumme = 1.0 ± 1e-10
- [ ] Symmetrie: |W_ij - W_ji| < 1e-10 nach enforce_symmetry()
- [ ] Spektralradius-Schätzung < 1.0
- [ ] Integration in ThinkingPipeline liefert InteractionResult

### Risiken
- **Hoch:** Konvergenz bei Scale-Free-Graphen (Hub-Knoten). Mitigation: Grad-abhängige Normalisierung.
- **Mittel:** Performance bei N>500 Knoten im Subgraph. Mitigation: Top-K Selektion begrenzt Subgraph.

---

## Phase 3: Context-Node (1-2 Tage)

### Abhängigkeit: Phase 2

### 3.1 Design (aus STABILITY_ANALYSIS §7)

Der Context-Node ist ein **virtueller Knoten** der die User-Query repräsentiert und selektiv in die Aktivierungsdynamik einkoppelt.

**Formel:**
```
ã_i(t) = F_i(a(t-1)) + β · α_i · a_ctx
```
wobei:
- `a_ctx` = Embedding der User-Query (VecN)
- `α_i = softmax_i(a_ctx^T · a_i)` = Attention-Gewicht
- `β = 0.3` (konfigurierbar, empfohlen 0.2-0.5)

### 3.2 Implementation

**Datei:** `backend/interaction/interaction_layer.cpp` (erweitern)

```cpp
// In propagate_one_step(), nach normalem Update:
if (context_activation_.has_value()) {
    // Compute attention weights
    std::vector<double> attention_scores;
    double attention_sum = 0.0;
    for (auto cid : subgraph) {
        double dot = 0.0;
        for (size_t d = 0; d < EMBED_DIM; ++d)
            dot += context_activation_.value()[d] * activations[cid][d];
        double score = std::exp(dot);
        attention_scores.push_back(score);
        attention_sum += score;
    }

    // Apply context bias
    for (size_t idx = 0; idx < subgraph.size(); ++idx) {
        double alpha = attention_scores[idx] / attention_sum;
        for (size_t d = 0; d < EMBED_DIM; ++d) {
            activations[subgraph[idx]][d] +=
                config_.beta_context * alpha * context_activation_.value()[d];
            activations[subgraph[idx]][d] =
                std::clamp(activations[subgraph[idx]][d], 0.0, 1.0);
        }
    }
}
```

### 3.3 API-Erweiterung

```cpp
// In InteractionLayer:
void set_context(const VecN& query_embedding);
void clear_context();

// In propagate():
InteractionResult propagate(
    const std::vector<ConceptId>& seeds,
    const LongTermMemory& ltm,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    const VecN* context_embedding = nullptr  // NEU: optional
);
```

### 3.4 Dateien

| Datei | Aktion | LoC |
|-------|--------|-----|
| `backend/interaction/interaction_layer.hpp` | ÄNDERN (+context member, +API) | +15 |
| `backend/interaction/interaction_layer.cpp` | ÄNDERN (+context logic in propagate) | +40 |
| `backend/core/system_orchestrator.cpp` | ÄNDERN (Query→Embedding→Context) | +10 |

**Gesamt: ~65 neue LoC**

### Akzeptanzkriterien Phase 3
- [ ] Context-Node verschiebt Fixpunkt Richtung Query (messbar via Cosine-Similarity)
- [ ] Ohne Context: gleiches Ergebnis wie Phase 2
- [ ] β=0 deaktiviert Context-Node effektiv
- [ ] Stabilitätsbedingungen bleiben erfüllt (Context ändert Jacobi-Matrix nicht)

### Risiken
- **Gering:** Query→Embedding Qualität hängt vom EmbeddingManager ab. Fallback: Context-Node als Broadcast (Option B).

---

## Phase 4: Excitation/Inhibition (2-3 Tage)

### Abhängigkeit: Phase 2

### 4.1 Negative Gewichte für antagonistische Relationen

**Aus MICROMODEL_INTERACTION_ARCHITECTURE.pdf §5:** "Inhibitory weights allowed (negative edges)"

**Mapping:**

| RelationType | Gewichts-Vorzeichen | Stärke |
|-------------|:-------------------:|--------|
| IS_A | + | 1.0 |
| CAUSES | + | 0.9 |
| PART_OF | + | 0.8 |
| HAS_PROPERTY | + | 0.7 |
| SUPPORTS | + | 0.8 |
| SIMILAR_TO | + | 0.6 |
| USED_FOR | + | 0.5 |
| PRECEDES | + | 0.4 |
| **CONTRADICTS** | **−** | **0.9** |
| **CONTRASTS_WITH** | **−** | **0.5** |

### 4.2 Winner-Takes-All Dynamik

```cpp
// Nach Konvergenz der Interaction Phase:
void apply_winner_takes_all(
    std::unordered_map<ConceptId, VecN>& activations,
    double wta_threshold = 0.3  // Aktivierungen unter threshold → suppressed
) {
    // Finde max Aktivierung (L2 Norm)
    double max_norm = 0.0;
    for (auto& [cid, act] : activations) {
        double norm = 0.0;
        for (size_t d = 0; d < EMBED_DIM; ++d) norm += act[d] * act[d];
        norm = std::sqrt(norm);
        max_norm = std::max(max_norm, norm);
    }

    // Suppress schwache Aktivierungen
    for (auto& [cid, act] : activations) {
        double norm = 0.0;
        for (size_t d = 0; d < EMBED_DIM; ++d) norm += act[d] * act[d];
        norm = std::sqrt(norm);
        if (norm < wta_threshold * max_norm) {
            act.fill(0.0);  // suppressed
        }
    }
}
```

### 4.3 Energy Function (korrigiert, aus STABILITY_ANALYSIS §2.6)

```cpp
double InteractionLayer::compute_energy(
    const std::unordered_map<ConceptId, VecN>& activations,
    const CouplingMatrix& W
) const {
    double E = 0.0;

    // Quadratischer Kopplungsterm
    for (auto& [i, neighbors] : W) {
        for (auto& [j, w_ij] : neighbors) {
            double dot = 0.0;
            for (size_t d = 0; d < EMBED_DIM; ++d)
                dot += activations.at(i)[d] * activations resistance.at(j)[d];
            E -= 0.5 * w_ij * dot;
        }
    }

    // Entropie-Term (Decay/Regularisierung)
    for (auto& [i, act] : activations) {
        for (size_t d = 0; d < EMBED_DIM; ++d) {
            double a = std::clamp(act[d], 1e-10, 1.0 - 1e-10);
            E += (1.0 / config_.lambda) *
                 (a * std::log(a) + (1.0 - a) * std::log(1.0 - a));
        }
    }

    return E;
}
```

### 4.4 Dateien

| Datei | Aktion | LoC |
|-------|--------|-----|
| `backend/interaction/interaction_layer.hpp` | ÄNDERN (+WTA, +sign mapping) | +20 |
| `backend/interaction/interaction_layer.cpp` | ÄNDERN (+inhibition in coupling, +WTA, +energy) | +100 |
| `backend/ltm/relation.hpp` | ÄNDERN (+is_inhibitory() helper) | +10 |

**Gesamt: ~130 neue LoC**

### Akzeptanzkriterien Phase 4
- [ ] CONTRADICTS-Relationen erzeugen negative Kopplungsgewichte
- [ ] WTA unterdrückt schwach aktivierte Konzepte
- [ ] Energy Function ist monoton fallend (Test 5 aus STABILITY_ANALYSIS)
- [ ] Widersprüchliche Konzepte können nicht gleichzeitig hoch aktiviert sein

### Risiken
- **Mittel:** Negative Gewichte können Symmetrie-Erzwingung komplizieren. Mitigation: Symmetrie auf |W| anwenden, Vorzeichen separat.

---

## Phase 5: KAN-MiniLLM Hybrid Engine (5-7 Tage)

### Abhängigkeit: Phase 1 (Vec16)

### 5.1 KANEncoder: Text → KAN-Embedding

**Neue Datei:** `backend/hybrid/kan_encoder.hpp` + `.cpp`

```cpp
class KANEncoder {
public:
    // Tokenizer: Einfaches BPE mit 8K Vocabulary
    // Trainiert auf Brain19-Daten (Konzeptlabels + Definitionen)
    explicit KANEncoder(const std::string& vocab_path);

    // Text → Token-IDs → KAN-Layers → VecN Embedding
    VecN encode(const std::string& text) const;

    // Tokenizer
    std::vector<uint16_t> tokenize(const std::string& text) const;

    // Train tokenizer from corpus
    void train_tokenizer(const std::vector<std::string>& corpus, size_t vocab_size = 8192);

private:
    // KAN-basierter Encoder: 3 Layers
    // Layer 1: token_dim(64) → 32  (64 KANNodes)
    // Layer 2: 32 → 16             (32 KANNodes)
    // Layer 3: 16 → EMBED_DIM      (16 KANNodes)
    std::unique_ptr<KANModule> encoder_kan_;

    // BPE Vocabulary
    std::unordered_map<std::string, uint16_t> vocab_;
    std::vector<std::string> id_to_token_;

    // Positional encoding for token sequence → fixed-size input
    VecN bag_of_embeddings(const std::vector<uint16_t>& tokens) const;

    // Token embeddings (8K × 64D lookup table)
    std::vector<std::array<double, 64>> token_embeddings_;
};
```

### 5.2 KANDecoder: KAN-Embedding → Text

**Neue Datei:** `backend/hybrid/kan_decoder.hpp` + `.cpp`

```cpp
class KANDecoder {
public:
    explicit KANDecoder(const KANEncoder& encoder);

    // VecN → Nächster Token (greedy) → Text
    std::string decode(const VecN& embedding, size_t max_tokens = 50) const;

    // VecN → Top-K nächste Konzeptlabels (aus LTM)
    std::vector<std::pair<ConceptId, double>> nearest_concepts(
        const VecN& embedding,
        const LongTermMemory& ltm,
        const EmbeddingManager& embeddings,
        size_t k = 5
    ) const;

private:
    std::unique_ptr<KANModule> decoder_kan_;  // Inverse des Encoders
    const KANEncoder& encoder_;
};
```

### 5.3 KAN-basiertes MiniLLM (ersetzt Ollama für bestimmte Tasks)

**Neue Datei:** `backend/understanding/kan_mini_llm.hpp` + `.cpp`

```cpp
class KANMiniLLM : public MiniLLM {
public:
    KANMiniLLM(
        std::shared_ptr<KANEncoder> encoder,
        std::shared_ptr<KANDecoder> decoder,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings
    );

    std::string get_model_id() const override { return "kan-mini-llm-v1.0"; }

    // Hypothesis Generation: Graph-Muster → KAN-Validation → Textformulierung
    std::vector<HypothesisProposal> generate_hypotheses(...) const override;

    // Analogy Detection: RelevanceMap-Strukturvergleich (KEIN LLM nötig)
    std::vector<AnalogyProposal> detect_analogies(...) const override;

    // Contradiction Detection: Graph + Epistemic Check (KEIN LLM nötig)
    std::vector<ContradictionProposal> detect_contradictions(...) const override;

    // Meaning: Weiterhin LLM-basiert, aber mit KAN-Context
    std::vector<MeaningProposal> extract_meaning(...) const override;

    // NEU: activate() → VecN statt predict() → Skalar
    VecN compute_activation(
        const std::vector<ConceptId>& concepts,
        const LongTermMemory& ltm
    ) const;
};
```

### 5.4 Tokenizer (8K BPE)

**Neue Datei:** `backend/hybrid/tokenizer.hpp` + `.cpp` (~200 LoC)

Einfache Byte-Pair-Encoding Implementation:
- Training: Iteratives Merge der häufigsten Paare
- Vocab: 8192 Tokens (ausreichend für Brain19-Domäne)
- Trainingscorpus: Alle Konzeptlabels + Definitionen aus LTM
- Persistenz: `vocab.bpe` Datei (Text-Format)

### 5.5 Dateien Übersicht

| Datei | Aktion | LoC |
|-------|--------|-----|
| `backend/hybrid/kan_encoder.hpp` | NEU | ~60 |
| `backend/hybrid/kan_encoder.cpp` | NEU | ~250 |
| `backend/hybrid/kan_decoder.hpp` | NEU | ~40 |
| `backend/hybrid/kan_decoder.cpp` | NEU | ~150 |
| `backend/hybrid/tokenizer.hpp` | NEU | ~40 |
| `backend/hybrid/tokenizer.cpp` | NEU | ~200 |
| `backend/understanding/kan_mini_llm.hpp` | NEU | ~80 |
| `backend/understanding/kan_mini_llm.cpp` | NEU | ~300 |
| `backend/core/system_orchestrator.cpp` | ÄNDERN (+KANMiniLLM Registration) | +15 |
| `Makefile` | ÄNDERN (+neue Sources) | +4 |

**Gesamt: ~1140 neue LoC**

### Akzeptanzkriterien Phase 5
- [ ] Tokenizer trainiert auf LTM-Daten, Roundtrip: tokenize(detokenize(text)) == text
- [ ] KANEncoder: encode("photosynthesis") gibt konsistentes VecN
- [ ] KANDecoder: nearest_concepts(encode("photosynthesis")) enthält "Photosynthesis"
- [ ] KANMiniLLM: detect_analogies() findet bekannte Analogien aus KG ohne Ollama
- [ ] KANMiniLLM: detect_contradictions() findet CONTRADICTS-Relationen ohne Ollama
- [ ] Ollama weiterhin für extract_meaning() und Chat-Verbalisierung verwendet

### Risiken
- **Hoch:** KAN-Encoder Qualität für Text ist ungetestet. Mitigation: Fallback auf OllamaMiniLLM.
- **Mittel:** 8K Vocab reicht evtl. nicht für Open-Domain. Mitigation: 16K oder 32K als Option.
- **Hoch:** Training des KAN-Encoders braucht ausreichend LTM-Daten (>500 Konzepte).

---

## Phase 6: Inkrementelles Training (2-3 Tage)

### Abhängigkeit: Phase 2

### 6.1 Re-Training bei neuen Relationen

**Datei:** `backend/micromodel/micro_trainer.hpp` + `.cpp`

```cpp
// NEUE Methode:
void MicroTrainer::retrain_affected(
    ConceptId changed_concept,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    const LongTermMemory& ltm
) {
    // 1. Re-train the changed concept
    retrain_single(changed_concept, registry, embeddings, ltm);

    // 2. Re-train 1-hop neighbors
    for (const auto& rel : ltm.get_outgoing_relations(changed_concept)) {
        retrain_single(rel.target, registry, embeddings, ltm);
    }
    for (const auto& rel : ltm.get_incoming_relations(changed_concept)) {
        retrain_single(rel.source, registry, embeddings, ltm);
    }
}

void MicroTrainer::retrain_single(
    ConceptId cid,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    const LongTermMemory& ltm
) {
    auto* model = registry.get_model(cid);
    if (!model) return;
    auto samples = generate_samples(cid, embeddings, ltm);
    MicroTrainingConfig cfg;
    cfg.max_epochs = 50;  // Weniger als initial, da warm-start
    model->train(samples, cfg);
}
```

### 6.2 Online Learning Trigger

**Datei:** `backend/core/system_orchestrator.cpp`

Nach jeder LTM-Modifikation (store_concept, add_relation):

```cpp
void SystemOrchestrator::on_ltm_changed(ConceptId affected) {
    if (trainer_ && registry_ && embeddings_) {
        trainer_->retrain_affected(affected, *registry_, *embeddings_, *ltm_);
    }
}
```

### 6.3 Feedback InteractionLayer → MicroModel Weights

**Idee:** Wenn die InteractionLayer konvergiert, können die finalen Aktivierungen als "gewünschte" Outputs interpretiert werden. MicroModels, deren predict() stark vom Attractor-Zustand abweicht, werden nachtrainiert.

```cpp
void retrain_from_interaction(
    const InteractionResult& result,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    const LongTermMemory& ltm
) {
    for (auto& [cid, final_act] : result.final_activations) {
        auto* model = registry.get_model(cid);
        if (!model) continue;

        // Für jeden Nachbarn: predict() soll final_activation approximieren
        for (auto& rel : ltm.get_outgoing_relations(cid)) {
            if (result.final_activations.count(rel.target)) {
                auto& target_act = result.final_activations.at(rel.target);
                auto& e = embeddings.get_relation_embedding(rel.type);
                double target_score = l2_norm(target_act) / std::sqrt(EMBED_DIM);
                model->train_step(e, target_act, target_score, MicroTrainingConfig{});
            }
        }
    }
}
```

### 6.4 Dateien

| Datei | Aktion | LoC |
|-------|--------|-----|
| `backend/micromodel/micro_trainer.hpp` | ÄNDERN (+retrain_affected, +retrain_single) | +10 |
| `backend/micromodel/micro_trainer.cpp` | ÄNDERN (+Implementation) | +50 |
| `backend/core/system_orchestrator.cpp` | ÄNDERN (+on_ltm_changed, +interaction feedback) | +40 |

**Gesamt: ~100 neue LoC**

### Akzeptanzkriterien Phase 6
- [ ] Neues Konzept + Relation → betroffene MicroModels werden re-trainiert
- [ ] Re-Training dauert <100ms für 1-hop Nachbarschaft
- [ ] predict()-Scores verbessern sich nach Re-Training (messbar)
- [ ] InteractionLayer-Feedback verbessert Konvergenzzeit bei wiederholten Queries

### Risiken
- **Mittel:** Catastrophic Forgetting bei aggressivem Re-Training. Mitigation: Warm-start mit weniger Epochs.

---

## Phase 7: Stabilitäts-Tests (2-3 Tage)

### Abhängigkeit: Phase 2

### 7.1 Testdatei

**Neue Datei:** `tests/test_stability.cpp` (~500 LoC)

### 7.2 Konkreter Testplan (aus STABILITY_ANALYSIS §8)

| Test | Funktion | Schwelle | Build-Gate? |
|------|----------|----------|:-----------:|
| Konvergenz-Grundtest | `test_convergence(N, λ, max_iter)` | T_c < 50 | ✅ |
| Attractor-Konsistenz | `test_attractor_consistency(N, λ, trials)` | C_A > 0.95 | ❌ Nightly |
| Perturbation-Robustheit | `test_basin_size(N, λ)` | R_p > 0.05 | ❌ Nightly |
| Bifurkationsdiagramm | `test_bifurcation_diagram(N)` | Monoton fallend | ❌ Weekly |
| Energie-Monotonie | `test_energy_monotonicity(N, λ)` | ΔE ≤ 0 | ✅ |
| Skalierungstest | `test_scaling()` | <10ms/20 iter (N=1000) | ❌ Weekly |

### 7.3 Test-Parameter Matrix

```
N ∈ {20, 50, 100, 500}
λ ∈ {0.05, 0.1, 0.2, 0.3}
d̄ ∈ {5, 10, 15}
100 Trials pro Konfiguration
```

### 7.4 Metriken

```cpp
struct StabilityMetrics {
    size_t convergence_time;          // Iterationen bis ε-Konvergenz
    double attractor_consistency;     // % gleicher Attraktor bei Perturbation
    double basin_radius;              // Max L2-Perturbation die Attraktor erhält
    double spectral_radius;           // Empirisch aus Konvergenzrate
    double attractor_entropy;         // Shannon-Entropie über Attraktor-Verteilung
    bool energy_monotone;             // ΔE ≤ 0 für alle t?
};
```

### 7.5 Dateien

| Datei | Aktion | LoC |
|-------|--------|-----|
| `tests/test_stability.cpp` | NEU | ~500 |
| `tests/test_interaction_layer.cpp` | NEU (Unit-Tests) | ~200 |
| `Makefile` | ÄNDERN (+test targets) | +10 |

**Gesamt: ~710 neue LoC**

### Akzeptanzkriterien Phase 7
- [ ] Alle Build-Gate-Tests bestehen bei `make test`
- [ ] Nightly-Tests haben Baseline-Werte dokumentiert
- [ ] Regressions werden automatisch erkannt

---

## Gesamtübersicht

### Timeline (1 Entwickler)

```
Woche 1:  Phase 0 (0.5d) + Phase 1 (1.5d) + Phase 2 Start (3d)
Woche 2:  Phase 2 Ende + Phase 7 Start (parallel)
Woche 3:  Phase 3 (1.5d) + Phase 4 (2.5d) + Phase 6 (1d)
Woche 4:  Phase 5 Start (KAN-MiniLLM)
Woche 5:  Phase 5 Ende + Phase 7 Ende + Integration Tests
```

### Kritischer Pfad

```
Phase 0 → Phase 1 → Phase 2 → Phase 3
                              → Phase 4  (parallel mit 3)
                              → Phase 6  (parallel mit 3,4)
                              → Phase 7  (parallel mit 3,4,6)
         Phase 1 → Phase 5              (parallel mit 2,3,4)
```

### Lines of Code Summary

| Phase | Neue LoC | Geänderte LoC | Neue Dateien |
|-------|:--------:|:-------------:|:------------:|
| 0 | 45 | 15 | 0 |
| 1 | 40 | 50 | 0 |
| 2 | 470 | 40 | 2 |
| 3 | 55 | 10 | 0 |
| 4 | 100 | 30 | 0 |
| 5 | 1120 | 20 | 8 |
| 6 | 90 | 10 | 0 |
| 7 | 700 | 10 | 2 |
| **Total** | **~2620** | **~185** | **12** |

### Hardware-Anforderungen

| Szenario | CPU | RAM | Disk |
|----------|-----|-----|------|
| Dev/Test (100 Konzepte) | Beliebig | <100 MB | <50 MB |
| Produktiv (1K Konzepte, d=16) | i5-6600K+ | ~500 MB | ~200 MB |
| Produktiv (10K Konzepte, d=16) | Empfohlen: 8+ Kerne | ~2 GB | ~1 GB |
| Max (100K Konzepte, d=16) | EPYC 80C | ~20 GB | ~10 GB |

**GPU:** Nicht erforderlich. Alle Operationen sind CPU-optimal bei <10K Konzepten. GPU lohnt erst ab >100K Konzepten (Kernel-Launch-Overhead dominiert).

### Offene Fragen

1. **Template vs. Constexpr für EMBED_DIM?** Empfehlung: constexpr reicht, Template ist Overengineering für 1 Dimension.
2. **Ollama komplett ersetzen?** Nein. Ollama bleibt für Chat und Meaning Extraction. KAN ersetzt nur Analogy/Contradiction/Hypothesis.
3. **WAL/Persistence wann?** Orthogonal zu diesem Plan. Sollte als separates Workstream parallel laufen.
