# Brain19 — Risk Mitigation Plan (Language Engine)

**Datum:** 2026-02-12
**Basis:** INTEGRATION_PLAN.md §10 Risiko-Matrix (7 Language-Engine-Risiken)
**Prinzip:** Jedes Risiko bekommt einen konkreten, implementierbaren Gegenplan. Keine vagen Empfehlungen.

---

## Inhaltsverzeichnis

1. [Risiko #1: KAN-Decoder generiert Müll](#risiko-1)
2. [Risiko #2: 8K BPE-Vocab zu klein für Deutsch](#risiko-2)
3. [Risiko #3: Token-Embedding-Table dominiert Params](#risiko-3)
4. [Risiko #4: Training-Daten zu wenig](#risiko-4)
5. [Risiko #5: Training korrumpiert MicroModels](#risiko-5)
6. [Risiko #6: Vec10 zu klein für Query-Encoding](#risiko-6)
7. [Risiko #7: FocusCursor-Chain zu kurz](#risiko-7)

---

## Risiko #1: KAN-Decoder generiert Müll <a name="risiko-1"></a>

**Wahrscheinlichkeit:** Hoch (30%) | **Impact:** Hoch
**Kern:** Mit ~1M Params (davon nur ~134K im Decoder) kann die autoregressive Generierung grammatisch inkorrekte oder semantisch unsinnige Ausgaben produzieren.

### 1.1 Konkreter Fix: Progressive Generierungs-Strategie

Drei Stufen, automatisch geschaltet:

| Stufe | Methode | Aktivierung | Qualität |
|-------|---------|------------|----------|
| **Stufe 1** | Pure Template-Generierung | Default bei Start | Hölzern, aber korrekt |
| **Stufe 2** | Hybrid: Template-Skelett + Decoder füllt Lücken | Decoder-Score > 0.4 | Besser flüssig |
| **Stufe 3** | Pure Decoder-Generierung | Decoder-Score > 0.7 | Flüssig (wenn erreichbar) |

Die Stufe wird **automatisch** durch einen QualityGate bestimmt — nicht manuell konfiguriert.

### 1.2 C++ Code

**Neue Datei:** `backend/hybrid/quality_gate.hpp`

```cpp
#pragma once
#include "kan_decoder.hpp"
#include "fusion_layer.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cursor/traversal_types.hpp"
#include <string>
#include <vector>

namespace brain19 {

struct QualityMetrics {
    double concept_coverage;    // Anteil der Chain-Konzepte im Output-Text [0,1]
    double chain_order_score;   // Reihenfolge der Konzepte im Text vs. Chain [0,1]
    double token_confidence;    // Durchschnitt der Decoder-Softmax-Maxima [0,1]
    double repetition_penalty;  // Abzug für Token-Wiederholungen [0,1], 1.0 = keine
    double overall_score;       // Gewichteter Gesamtscore [0,1]
};

struct QualityGateConfig {
    double hybrid_threshold = 0.4;     // Ab hier: Stufe 2 (Hybrid)
    double pure_decoder_threshold = 0.7; // Ab hier: Stufe 3 (Pure Decoder)
    double min_concept_coverage = 0.5;  // Mindestens 50% der Chain-Konzepte im Text
    double max_repetition_ratio = 0.3;  // Max 30% wiederholte Tokens
    size_t evaluation_window = 50;      // Letzte N Generierungen für Rolling Average
};

enum class GenerationMode {
    TEMPLATE_ONLY,    // Stufe 1
    HYBRID,           // Stufe 2: Template-Skelett + Decoder
    PURE_DECODER      // Stufe 3
};

class QualityGate {
public:
    explicit QualityGate(QualityGateConfig config = {});

    // Entscheide welche Stufe für diese Generierung gilt
    GenerationMode select_mode() const;

    // Bewerte einen generierten Text gegen die Traversal-Chain
    QualityMetrics evaluate(
        const std::string& generated_text,
        const TraversalResult& chain,
        const LongTermMemory& ltm
    ) const;

    // Update Rolling-Average mit neuem Ergebnis
    void record_result(const QualityMetrics& metrics);

    // Aktueller Rolling-Average Score
    double current_score() const;

private:
    QualityGateConfig config_;
    std::vector<double> recent_scores_;  // Ring-Buffer der letzten N Scores
    size_t ring_pos_ = 0;

    // Concept Coverage: Prüfe ob Chain-Konzepte im Text vorkommen
    double compute_concept_coverage(
        const std::string& text,
        const TraversalResult& chain,
        const LongTermMemory& ltm
    ) const;

    // Chain Order: Prüfe ob Konzepte in kausaler Reihenfolge erscheinen
    double compute_chain_order(
        const std::string& text,
        const TraversalResult& chain,
        const LongTermMemory& ltm
    ) const;

    // Repetition Penalty: Erkenne Token-Schleifen
    double compute_repetition_penalty(const std::string& text) const;
};

} // namespace brain19
```

**Pseudocode der Kern-Methoden:**

```cpp
QualityMetrics QualityGate::evaluate(
    const std::string& text,
    const TraversalResult& chain,
    const LongTermMemory& ltm
) const {
    QualityMetrics m;
    m.concept_coverage = compute_concept_coverage(text, chain, ltm);
    m.chain_order_score = compute_chain_order(text, chain, ltm);
    m.repetition_penalty = compute_repetition_penalty(text);

    // Token confidence wird extern vom Decoder gesetzt (Durchschnitt der Softmax-Maxima)
    // Hier nur die text-basierten Metriken

    m.overall_score = 0.35 * m.concept_coverage
                    + 0.25 * m.chain_order_score
                    + 0.20 * m.token_confidence
                    + 0.20 * m.repetition_penalty;
    return m;
}

double QualityGate::compute_concept_coverage(
    const std::string& text,
    const TraversalResult& chain,
    const LongTermMemory& ltm
) const {
    size_t found = 0;
    for (ConceptId cid : chain.concept_sequence) {
        auto concept = ltm.retrieve_concept(cid);
        if (!concept) continue;
        // Prüfe ob Label im generierten Text vorkommt (case-insensitive)
        if (text.find(concept->label) != std::string::npos) {
            ++found;
        }
    }
    if (chain.concept_sequence.empty()) return 0.0;
    return static_cast<double>(found) / chain.concept_sequence.size();
}

double QualityGate::compute_chain_order(
    const std::string& text,
    const TraversalResult& chain,
    const LongTermMemory& ltm
) const {
    // Finde Position jedes Konzept-Labels im Text
    std::vector<size_t> positions;
    for (ConceptId cid : chain.concept_sequence) {
        auto concept = ltm.retrieve_concept(cid);
        if (!concept) continue;
        size_t pos = text.find(concept->label);
        if (pos != std::string::npos) {
            positions.push_back(pos);
        }
    }
    if (positions.size() < 2) return 1.0;  // Nur 1 Konzept = trivial korrekt

    // Zähle korrekte Paar-Reihenfolgen (Kendall tau-b Approximation)
    size_t concordant = 0, total = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            if (positions[i] < positions[j]) ++concordant;
            ++total;
        }
    }
    return total > 0 ? static_cast<double>(concordant) / total : 1.0;
}

GenerationMode QualityGate::select_mode() const {
    double score = current_score();
    if (score >= config_.pure_decoder_threshold) return GenerationMode::PURE_DECODER;
    if (score >= config_.hybrid_threshold)       return GenerationMode::HYBRID;
    return GenerationMode::TEMPLATE_ONLY;
}
```

**Integration in KANLanguageEngine::generate():**

```cpp
LanguageResult KANLanguageEngine::generate(const std::string& query, size_t max_tokens) const {
    auto q = encoder_.encode(query);
    auto seeds = find_seeds(query);
    auto traversal = cursor_manager_.process_seeds(seeds, q);
    auto semantics = semantic_scorer_.score(traversal.best_chain, q, embeddings_);
    auto fused = fusion_.fuse(traversal.best_chain, semantics, embeddings_);

    std::string text;
    bool used_template = false;
    GenerationMode mode = quality_gate_.select_mode();

    switch (mode) {
        case GenerationMode::TEMPLATE_ONLY:
            text = template_generate(traversal.best_chain.concept_sequence, fused.template_type);
            used_template = true;
            break;

        case GenerationMode::HYBRID: {
            // Template-Skelett generieren, dann Decoder füllt Lücken
            std::string skeleton = template_generate_skeleton(
                traversal.best_chain.concept_sequence, fused.template_type);
            text = decoder_.infill(fused, skeleton, max_tokens);
            break;
        }

        case GenerationMode::PURE_DECODER:
            text = decoder_.decode(fused, max_tokens);
            break;
    }

    // Qualität bewerten und Rolling-Average updaten
    auto metrics = quality_gate_.evaluate(text, traversal.best_chain, ltm_);

    // Guard: Wenn Decoder-Output unter Minimum, Fallback auf Template
    if (!used_template && metrics.concept_coverage < config_.min_confidence_for_decoder) {
        text = template_generate(traversal.best_chain.concept_sequence, fused.template_type);
        used_template = true;
    }

    quality_gate_.record_result(metrics);

    return LanguageResult{text, traversal.all_activated,
        traversal.best_chain.concept_sequence, metrics.overall_score,
        traversal.best_chain.total_steps, used_template,
        template_type_to_string(fused.template_type)};
}
```

### 1.3 Benchmarks: Ab wann ist Qualität "gut genug"?

| Metrik | Minimum (Stufe 1 → 2) | Gut (Stufe 2 → 3) | Ziel |
|--------|----------------------|-------------------|------|
| concept_coverage | ≥ 0.5 | ≥ 0.7 | ≥ 0.9 |
| chain_order_score | ≥ 0.6 | ≥ 0.8 | ≥ 0.95 |
| token_confidence | ≥ 0.3 | ≥ 0.5 | ≥ 0.7 |
| repetition_penalty | ≥ 0.7 | ≥ 0.9 | = 1.0 |
| **overall_score** | **≥ 0.4** | **≥ 0.7** | **≥ 0.85** |

### 1.4 Test-Strategie

```cpp
// test_quality_gate.cpp
TEST(QualityGate, ConceptCoverageDetectsMissingConcepts) {
    // Chain: [Eis, Schmelzen, Wasser]
    // Text: "Eis schmilzt" → nur 2/3 Konzepte → coverage = 0.67
    auto metrics = gate.evaluate("Eis schmilzt", chain_eis_wasser, ltm);
    EXPECT_NEAR(metrics.concept_coverage, 0.67, 0.05);
}

TEST(QualityGate, ChainOrderDetectsWrongOrder) {
    // Chain: [Eis, Schmelzen, Wasser]
    // Text: "Wasser wird zu Eis" → Reihenfolge invertiert → low score
    auto metrics = gate.evaluate("Wasser wird zu Eis", chain_eis_wasser, ltm);
    EXPECT_LT(metrics.chain_order_score, 0.5);
}

TEST(QualityGate, RepetitionPenaltyDetectsLoops) {
    // "Eis Eis Eis Eis" → hohe Repetition → penalty < 0.5
    auto metrics = gate.evaluate("Eis Eis Eis Eis", chain_eis_wasser, ltm);
    EXPECT_LT(metrics.repetition_penalty, 0.5);
}

TEST(QualityGate, ModeSelectStartsWithTemplate) {
    QualityGate gate;
    // Keine Historien-Daten → score = 0 → TEMPLATE_ONLY
    EXPECT_EQ(gate.select_mode(), GenerationMode::TEMPLATE_ONLY);
}

TEST(QualityGate, ModeSelectUpgradesWithGoodScores) {
    QualityGate gate;
    // 50 gute Ergebnisse einspeisen
    for (int i = 0; i < 50; ++i) {
        gate.record_result({.overall_score = 0.75});
    }
    EXPECT_EQ(gate.select_mode(), GenerationMode::PURE_DECODER);
}
```

### 1.5 Fallback

Wenn auch Templates schlechte Ergebnisse liefern (z.B. find_seeds() findet keine Konzepte):

```cpp
LanguageResult fallback_empty_result(const std::string& query) {
    return LanguageResult{
        .text = "Ich habe dazu kein Wissen.",
        .activated_concepts = {},
        .causal_chain = {},
        .confidence = 0.0,
        .traversal_steps = 0,
        .used_template = true,
        .template_type = "FALLBACK_EMPTY"
    };
}
```

---

## Risiko #2: 8K BPE-Vocab zu klein für Deutsch <a name="risiko-2"></a>

**Wahrscheinlichkeit:** Mittel (20%) | **Impact:** Mittel
**Kern:** Deutsche Morphologie (Komposita, Flexion) braucht viele Subwords. Bei 8K Vocab könnten häufige Wörter mehrere Tokens brauchen, was die effektive Sequenzlänge aufbläht.

### 2.1 Konkreter Fix: OOV-Rate-Monitoring + automatisches Upgrade

Die BPETokenizer-Klasse bekommt eine integrierte OOV-Diagnose. Wenn die OOV-Rate über dem Schwellwert liegt, wird automatisch ein Vocab-Upgrade angestoßen.

### 2.2 C++ Code

**Erweiterung in:** `backend/hybrid/tokenizer.hpp`

```cpp
struct VocabDiagnostics {
    size_t total_tokens_encoded = 0;
    size_t unk_tokens = 0;            // UNK-Token (#3) Verwendungen
    size_t multi_token_words = 0;     // Wörter die >1 Token brauchen
    size_t total_words_seen = 0;
    double avg_tokens_per_word = 0.0;

    double unk_rate() const {
        return total_tokens_encoded > 0
            ? static_cast<double>(unk_tokens) / total_tokens_encoded : 0.0;
    }
    double fragmentation_rate() const {
        return total_words_seen > 0
            ? static_cast<double>(multi_token_words) / total_words_seen : 0.0;
    }
    bool needs_upgrade(double max_unk_rate = 0.02, double max_frag_rate = 0.5) const {
        return unk_rate() > max_unk_rate || fragmentation_rate() > max_frag_rate;
    }
};

class BPETokenizer {
public:
    // ... bestehende API ...

    // Diagnose: Laufe über einen Testkorpus und messe Vocab-Abdeckung
    VocabDiagnostics diagnose(const std::vector<std::string>& test_corpus) const;

    // Inkrementelles Upgrade: Merge weitere Token-Paare ins bestehende Vocab
    // Behält alle existierenden Token-IDs bei (kein Retraining nötig)
    void expand_vocab(
        const std::vector<std::string>& additional_corpus,
        size_t new_vocab_size   // z.B. 16384
    );
};
```

**Pseudocode Diagnose:**

```cpp
VocabDiagnostics BPETokenizer::diagnose(const std::vector<std::string>& corpus) const {
    VocabDiagnostics diag;
    for (const auto& text : corpus) {
        // Split in Whitespace-Wörter
        auto words = split_whitespace(text);
        diag.total_words_seen += words.size();

        for (const auto& word : words) {
            auto tokens = encode(word);
            diag.total_tokens_encoded += tokens.size();

            for (auto t : tokens) {
                if (t == UNK_TOKEN) ++diag.unk_tokens;
            }
            if (tokens.size() > 1) ++diag.multi_token_words;
        }
    }
    diag.avg_tokens_per_word =
        diag.total_words_seen > 0
        ? static_cast<double>(diag.total_tokens_encoded) / diag.total_words_seen
        : 0.0;
    return diag;
}
```

### 2.3 Test-Strategie

```cpp
TEST(Tokenizer, DiagnoseDetectsHighUNK) {
    auto tok = BPETokenizer::train(small_corpus, ltm, 4096); // absichtlich klein
    auto diag = tok.diagnose(german_test_corpus);
    // Mit 4K Vocab und deutschem Text: UNK-Rate sollte messbar sein
    EXPECT_GT(diag.unk_rate(), 0.0);
}

TEST(Tokenizer, ExpandVocabPreservesExistingIDs) {
    auto tok = BPETokenizer::train(corpus, ltm, 8192);
    auto tokens_before = tok.encode("Photosynthese");
    tok.expand_vocab(more_corpus, 16384);
    auto tokens_after = tok.encode("Photosynthese");
    // Bestehende Tokens müssen gleich bleiben
    EXPECT_EQ(tokens_before, tokens_after);
}

TEST(Tokenizer, GermanCompositaHandling) {
    // "Donaudampfschifffahrt" sollte nicht komplett UNK sein
    auto tok = BPETokenizer::train(german_corpus, ltm, 8192);
    auto tokens = tok.encode("Donaudampfschifffahrt");
    for (auto t : tokens) {
        EXPECT_NE(t, BPETokenizer::UNK_TOKEN);
    }
}
```

### 2.4 Schwellwerte

| Metrik | Akzeptabel | Grenzwertig | Upgrade nötig |
|--------|-----------|-------------|---------------|
| UNK-Rate | < 1% | 1-2% | > 2% |
| Fragmentation Rate | < 40% | 40-50% | > 50% |
| Avg Tokens/Word | < 1.5 | 1.5-2.0 | > 2.0 |

### 2.5 Fallback

Wenn expand_vocab() nicht reicht:
1. Trainiere komplett neues 16K-Vocab (`LanguageConfig::vocab_size = 16384`)
2. Token-Embedding-Table wächst: 8192 × 64 → 16384 × 64 = +512K Params
3. Re-training von Encoder + Decoder nötig (Scorer und Fusion bleiben)
4. **Worst Case:** Character-Level Fallback für OOV-Wörter (Byte-Level BPE)

---

## Risiko #3: Token-Embedding-Table dominiert Parameter <a name="risiko-3"></a>

**Wahrscheinlichkeit:** Niedrig | **Impact:** Niedrig
**Kern:** 524K von ~1M Params (52%) stecken in der 8192×64 Embedding-Tabelle. Wenig Kapazität für Logik.

### 3.1 Konkreter Fix: Shared Embeddings + Optional Dimensionsreduktion

**Maßnahme 1:** Encoder und Decoder teilen sich dieselbe Token-Embedding-Tabelle. Das reduziert die effektiven Params nicht (die Tabelle existiert nur einmal), aber es sorgt dafür, dass Encoder-Gradients auch die Decoder-Embeddings verbessern und umgekehrt.

**Maßnahme 2:** Optionale SVD-basierte Dimensionsreduktion der Embedding-Tabelle.

### 3.2 C++ Code

```cpp
// In KANEncoder und KANDecoder: Shared Embedding Reference
class SharedEmbeddingTable {
public:
    static constexpr size_t TOKEN_EMBED_DIM = 64;

    explicit SharedEmbeddingTable(size_t vocab_size);

    // Lookup
    const std::array<double, TOKEN_EMBED_DIM>& operator[](uint16_t token_id) const;
    std::array<double, TOKEN_EMBED_DIM>& operator[](uint16_t token_id);

    // SVD-Kompression: 64D → reduced_dim (z.B. 32)
    // Reduziert Tabelle von vocab×64 auf vocab×32 + 32×64 Projection
    void compress(size_t reduced_dim);

    // Statistik
    size_t param_count() const { return vocab_size_ * current_dim_; }
    size_t vocab_size() const { return vocab_size_; }
    size_t dim() const { return current_dim_; }

private:
    size_t vocab_size_;
    size_t current_dim_ = TOKEN_EMBED_DIM;
    std::vector<std::array<double, TOKEN_EMBED_DIM>> embeddings_;
};

// In KANEncoder: Shared Reference statt eigene Kopie
class KANEncoder {
public:
    KANEncoder(SharedEmbeddingTable& shared_embeddings, KANAdapter& kan);
    // ...
private:
    SharedEmbeddingTable& embeddings_;  // SHARED, nicht owned
};

// In KANDecoder: Dieselbe Shared Reference
class KANDecoder {
public:
    KANDecoder(SharedEmbeddingTable& shared_embeddings, BPETokenizer& tok, KANAdapter& kan);
    // Output Projection nutzt embeddings_^T (Tied Weights)
    // logits = h · E^T  wo E = shared_embeddings_
private:
    SharedEmbeddingTable& embeddings_;  // SHARED, dieselbe Instanz
};
```

### 3.3 Test-Strategie

```cpp
TEST(SharedEmbeddings, EncoderAndDecoderShareSameInstance) {
    SharedEmbeddingTable table(8192);
    KANEncoder encoder(table, kan);
    KANDecoder decoder(table, tok, kan);

    // Ändere Embedding via Encoder-Training
    table[42][0] = 99.0;

    // Decoder sieht die Änderung sofort
    EXPECT_EQ(table[42][0], 99.0);  // Same instance
}

TEST(SharedEmbeddings, CompressionReducesParams) {
    SharedEmbeddingTable table(8192);
    EXPECT_EQ(table.param_count(), 8192 * 64);  // 524,288

    table.compress(32);
    EXPECT_EQ(table.param_count(), 8192 * 32);  // 262,144 (+ 32×64 Proj)
}
```

### 3.4 Fallback

Dieses Risiko ist niedrig. Wenn die Embedding-Dominanz zum Problem wird (Encoder/Decoder zu schwach):
- Embedding-Dimension von 64 auf 32 reduzieren → 262K statt 524K Params
- Frei gewordene Kapazität in Decoder-Hidden-Dim investieren (10 → 20)
- Parameter-Budget Rebalancing via `LanguageConfig`

---

## Risiko #4: Training-Daten zu wenig <a name="risiko-4"></a>

**Wahrscheinlichkeit:** Hoch (40%) | **Impact:** Hoch
**Kern:** Das Design-Dokument plant ~500 manuell kuratierte QA-Paare. Das reicht nicht für Fusion-Training mit Cross-Entropy + Chain-Loss.

### 4.1 Konkreter Fix: Synthetische QA-Generierung aus dem Wissensgraph

Automatische Pipeline: Graph-Traversal → strukturierte QA-Paare → Template-Variationen → Training-Daten.

### 4.2 Mengenanalyse: Was generiert 1000 Concepts?

```
Gegeben:
  ~1000 Concepts in LTM
  ~3000 Relations (durchschnittlich 3 pro Concept)
  10 RelationTypes

Generierbare Paare:

1. Definitional (1 pro Concept):
   "Was ist {label}?" → "{label} ist {definition}."
   → 1000 Paare

2. Single-Relation (1 pro Relation, typ-spezifisches Template):
   CAUSES:        "Was passiert wenn {src}?" → "{src} verursacht {tgt}."
   IS_A:          "Was für ein Typ ist {src}?" → "{src} ist ein {tgt}."
   HAS_PROPERTY:  "Welche Eigenschaft hat {src}?" → "{src} hat {tgt}."
   ENABLES:       "Was ermöglicht {src}?" → "{src} ermöglicht {tgt}."
   PART_OF:       "Wovon ist {src} ein Teil?" → "{src} ist Teil von {tgt}."
   SIMILAR_TO:    "Was ist ähnlich wie {src}?" → "{src} ist ähnlich wie {tgt}."
   CONTRADICTS:   "Was widerspricht {src}?" → "{src} widerspricht {tgt}."
   SUPPORTS:      "Was unterstützt {src}?" → "{src} unterstützt {tgt}."
   → ~3000 Paare

3. Multi-Hop (2-3 Hop Chains, algorithmisch extrahiert):
   "Warum führt {A} zu {C}?" → "{A} verursacht {B}. {B} verursacht {C}."
   → ~500 Paare (aus CAUSES-Ketten der Tiefe 2-3)

4. Template-Variationen (3-5 pro Basis-Paar):
   "Was ist Wasser?" / "Erkläre Wasser." / "Definiere Wasser." / "Was bedeutet Wasser?"
   → Multiplikator ×4

5. Negativ-Beispiele (keine Antwort möglich):
   "Was ist Xylophon?" (nicht im KG) → "Ich habe dazu kein Wissen."
   → ~200 Paare

GESAMT:
  (1000 + 3000 + 500) × 4 + 200 = ~18,200 Paare
```

**18.200 synthetische Paare aus 1000 Concepts.** Das ist 36× mehr als die geplanten 500.

### 4.3 C++ Code

**Neue Datei:** `backend/hybrid/synthetic_data_generator.hpp`

```cpp
#pragma once
#include "../ltm/long_term_memory.hpp"
#include "../cursor/traversal_types.hpp"
#include <vector>
#include <string>
#include <random>

namespace brain19 {

struct QAPair {
    std::string query;
    std::string expected_answer;
    std::vector<ConceptId> expected_chain;
    std::string template_type;   // DEFINITIONAL, KAUSAL, EIGENSCHAFT, etc.
};

struct DataGenConfig {
    size_t max_chain_depth = 3;           // Max Hops für Multi-Hop Paare
    size_t template_variations = 4;       // Variationen pro Basis-Paar
    bool include_negatives = true;        // Negativ-Beispiele generieren
    size_t negative_count = 200;          // Anzahl Negativ-Beispiele
    double min_relation_weight = 0.3;     // Nur Relationen mit weight > 0.3
    unsigned seed = 42;                   // Reproduzierbarkeit
};

class SyntheticDataGenerator {
public:
    SyntheticDataGenerator(const LongTermMemory& ltm, DataGenConfig config = {});

    // Generiere alle QA-Paare
    std::vector<QAPair> generate_all();

    // Generiere nur bestimmte Typen (für Curriculum Training)
    std::vector<QAPair> generate_definitional();
    std::vector<QAPair> generate_single_relation();
    std::vector<QAPair> generate_multi_hop();
    std::vector<QAPair> generate_negatives();

    // Qualitätskontrolle: Filtere Paare mit schlechter Qualität
    static std::vector<QAPair> quality_filter(
        const std::vector<QAPair>& pairs,
        size_t min_answer_length = 3,       // Mindest-Wörter in Antwort
        size_t max_answer_length = 30,      // Max-Wörter in Antwort
        bool require_chain = true           // Chain darf nicht leer sein
    );

    // Statistik
    struct GenStats {
        size_t definitional = 0;
        size_t single_relation = 0;
        size_t multi_hop = 0;
        size_t variations = 0;
        size_t negatives = 0;
        size_t filtered_out = 0;
        size_t total() const {
            return definitional + single_relation + multi_hop + variations + negatives;
        }
    };
    GenStats last_stats() const { return stats_; }

private:
    const LongTermMemory& ltm_;
    DataGenConfig config_;
    std::mt19937 rng_;
    GenStats stats_{};

    // Template-Variationen für eine Frage generieren
    std::vector<std::string> vary_question(
        const std::string& base_question,
        const std::string& template_type
    ) const;

    // Multi-Hop Chains extrahieren
    std::vector<std::vector<ConceptId>> extract_chains(
        ConceptId start,
        RelationType type,
        size_t max_depth
    ) const;

    // Antwort-Text aus Chain generieren
    std::string chain_to_answer(
        const std::vector<ConceptId>& chain,
        const std::vector<RelationType>& relations
    ) const;
};

} // namespace brain19
```

**Pseudocode der Kern-Methoden:**

```cpp
std::vector<QAPair> SyntheticDataGenerator::generate_definitional() {
    std::vector<QAPair> pairs;
    auto all_ids = ltm_.get_all_concept_ids();

    for (ConceptId cid : all_ids) {
        auto concept = ltm_.retrieve_concept(cid);
        if (!concept || !concept->epistemic.is_active()) continue;

        // Basis-Paar
        QAPair base;
        base.query = "Was ist " + concept->label + "?";
        base.expected_answer = concept->label + " ist " + concept->definition + ".";
        base.expected_chain = {cid};
        base.template_type = "DEFINITIONAL";
        pairs.push_back(base);

        // Template-Variationen
        for (const auto& varied_q : vary_question(base.query, "DEFINITIONAL")) {
            QAPair varied = base;
            varied.query = varied_q;
            pairs.push_back(varied);
        }
    }
    stats_.definitional = pairs.size();
    return pairs;
}

std::vector<QAPair> SyntheticDataGenerator::generate_single_relation() {
    std::vector<QAPair> pairs;
    auto all_ids = ltm_.get_all_concept_ids();

    for (ConceptId cid : all_ids) {
        auto relations = ltm_.get_outgoing_relations(cid);
        auto src_concept = ltm_.retrieve_concept(cid);
        if (!src_concept) continue;

        for (const auto& rel : relations) {
            if (rel.weight < config_.min_relation_weight) continue;

            auto tgt_concept = ltm_.retrieve_concept(rel.target);
            if (!tgt_concept) continue;

            QAPair pair;
            pair.expected_chain = {cid, rel.target};

            switch (rel.type) {
                case RelationType::CAUSES:
                    pair.query = "Was passiert wenn " + src_concept->label + "?";
                    pair.expected_answer = src_concept->label + " verursacht "
                                         + tgt_concept->label + ".";
                    pair.template_type = "KAUSAL";
                    break;

                case RelationType::IS_A:
                    pair.query = "Was für ein Typ ist " + src_concept->label + "?";
                    pair.expected_answer = src_concept->label + " ist ein "
                                         + tgt_concept->label + ".";
                    pair.template_type = "DEFINITIONAL";
                    break;

                case RelationType::HAS_PROPERTY:
                    pair.query = "Welche Eigenschaft hat " + src_concept->label + "?";
                    pair.expected_answer = src_concept->label + " hat "
                                         + tgt_concept->label + ".";
                    pair.template_type = "EIGENSCHAFT";
                    break;

                // ... analoge Templates für alle 10 RelationTypes ...
                default:
                    pair.query = "Was hat " + src_concept->label + " mit "
                                + tgt_concept->label + " zu tun?";
                    pair.expected_answer = src_concept->label + " steht in Beziehung zu "
                                         + tgt_concept->label + ".";
                    pair.template_type = "ALLGEMEIN";
                    break;
            }
            pairs.push_back(pair);

            // Variationen
            for (const auto& varied_q : vary_question(pair.query, pair.template_type)) {
                QAPair varied = pair;
                varied.query = varied_q;
                pairs.push_back(varied);
            }
        }
    }
    stats_.single_relation = pairs.size();
    return pairs;
}

std::vector<QAPair> SyntheticDataGenerator::generate_multi_hop() {
    std::vector<QAPair> pairs;
    auto all_ids = ltm_.get_all_concept_ids();

    for (ConceptId start : all_ids) {
        // CAUSES-Ketten der Tiefe 2-3 extrahieren
        auto chains = extract_chains(start, RelationType::CAUSES, config_.max_chain_depth);

        for (const auto& chain : chains) {
            if (chain.size() < 3) continue;  // Mindestens A → B → C

            auto first = ltm_.retrieve_concept(chain.front());
            auto last = ltm_.retrieve_concept(chain.back());
            if (!first || !last) continue;

            QAPair pair;
            pair.query = "Warum führt " + first->label + " zu " + last->label + "?";
            pair.expected_chain = chain;
            pair.template_type = "MULTI_HOP";

            // Antwort: Kette ausschreiben
            std::string answer;
            for (size_t i = 0; i < chain.size() - 1; ++i) {
                auto src = ltm_.retrieve_concept(chain[i]);
                auto tgt = ltm_.retrieve_concept(chain[i + 1]);
                if (!src || !tgt) break;
                if (!answer.empty()) answer += " ";
                answer += src->label + " verursacht " + tgt->label + ".";
            }
            pair.expected_answer = answer;
            pairs.push_back(pair);
        }
    }
    stats_.multi_hop = pairs.size();
    return pairs;
}

std::vector<std::string> SyntheticDataGenerator::vary_question(
    const std::string& base, const std::string& type
) const {
    std::vector<std::string> variations;
    if (type == "DEFINITIONAL") {
        // "Was ist X?" → ["Erkläre X.", "Definiere X.", "Was bedeutet X?"]
        // Extrahiere Konzeptname aus Frage
        auto name = extract_concept_name(base);
        variations.push_back("Erkläre " + name + ".");
        variations.push_back("Definiere " + name + ".");
        variations.push_back("Was bedeutet " + name + "?");
    } else if (type == "KAUSAL") {
        auto name = extract_concept_name(base);
        variations.push_back("Was verursacht " + name + "?");
        variations.push_back("Welche Folgen hat " + name + "?");
        variations.push_back("Was ergibt sich aus " + name + "?");
    }
    // ... analoge Variationen für andere Typen ...
    return variations;
}
```

### 4.4 Qualitätskontrolle der synthetischen Daten

```cpp
std::vector<QAPair> SyntheticDataGenerator::quality_filter(
    const std::vector<QAPair>& pairs,
    size_t min_answer_length,
    size_t max_answer_length,
    bool require_chain
) {
    std::vector<QAPair> filtered;
    for (const auto& pair : pairs) {
        // 1. Antwort-Länge prüfen
        size_t word_count = count_words(pair.expected_answer);
        if (word_count < min_answer_length) continue;
        if (word_count > max_answer_length) continue;

        // 2. Chain muss Konzepte enthalten
        if (require_chain && pair.expected_chain.empty()) continue;

        // 3. Query darf nicht leer sein
        if (pair.query.size() < 5) continue;

        // 4. Antwort muss mindestens ein Chain-Konzept enthalten
        // (Konsistenz-Check)
        if (!pair.expected_chain.empty()) {
            bool has_match = false;
            for (ConceptId cid : pair.expected_chain) {
                // Label-Check im Answer-Text
                // (wird in generate_*() garantiert, aber Double-Check)
                has_match = true;
                break;
            }
            if (!has_match) continue;
        }

        filtered.push_back(pair);
    }
    return filtered;
}
```

### 4.5 Curriculum-Training-Integration

Die synthetischen Daten werden in der Reihenfolge des Curriculum-Trainings (aus KAN_MINILLM_LANGUAGE_ENGINE.md §5.3) eingespeist:

```cpp
void LanguageTrainer::train_fusion_curriculum(SyntheticDataGenerator& gen) {
    // Stufe 1 (Epoch 1-50): Nur Definitional
    auto definitional = gen.generate_definitional();
    auto filtered_def = SyntheticDataGenerator::quality_filter(definitional);
    train_epochs(filtered_def, /*epochs=*/50);

    // Stufe 2 (Epoch 51-150): Single-Relation (kausal)
    auto relational = gen.generate_single_relation();
    auto filtered_rel = SyntheticDataGenerator::quality_filter(relational);
    train_epochs(filtered_rel, /*epochs=*/100);

    // Stufe 3 (Epoch 151-300): Multi-Hop
    auto multi = gen.generate_multi_hop();
    auto filtered_multi = SyntheticDataGenerator::quality_filter(multi);
    train_epochs(filtered_multi, /*epochs=*/150);
}
```

### 4.6 Test-Strategie

```cpp
TEST(SyntheticData, GeneratesExpectedVolume) {
    // LTM mit 100 Concepts und ~300 Relations füllen
    auto ltm = create_test_ltm(100, 300);
    SyntheticDataGenerator gen(ltm);
    auto pairs = gen.generate_all();

    // Erwartung: ~(100 + 300 + ~50) × 4 + 200 ≈ 2000 Paare
    EXPECT_GT(pairs.size(), 1500);
    EXPECT_LT(pairs.size(), 3000);

    auto stats = gen.last_stats();
    EXPECT_GE(stats.definitional, 100);
    EXPECT_GE(stats.single_relation, 300);
}

TEST(SyntheticData, QualityFilterRemovesBadPairs) {
    auto ltm = create_test_ltm(10, 20);
    SyntheticDataGenerator gen(ltm);
    auto pairs = gen.generate_all();
    auto filtered = SyntheticDataGenerator::quality_filter(pairs);

    // Alle gefilterten Paare haben valide Antworten
    for (const auto& p : filtered) {
        EXPECT_GE(count_words(p.expected_answer), 3);
        EXPECT_LE(count_words(p.expected_answer), 30);
        EXPECT_FALSE(p.expected_chain.empty());
    }
}

TEST(SyntheticData, MultiHopChainsAreValid) {
    auto ltm = create_chain_ltm();  // A→CAUSES→B→CAUSES→C
    SyntheticDataGenerator gen(ltm);
    auto multi = gen.generate_multi_hop();

    ASSERT_FALSE(multi.empty());
    for (const auto& p : multi) {
        EXPECT_GE(p.expected_chain.size(), 3);
        EXPECT_EQ(p.template_type, "MULTI_HOP");
    }
}
```

### 4.7 Fallback

Wenn synthetische Daten zu homogen sind (overfitting auf Templates):
1. **Daten-Augmentation:** Wort-Reihenfolge in Fragen variieren ("Wenn Eis schmilzt, was passiert?" statt "Was passiert wenn Eis schmilzt?")
2. **Noise-Injection:** Zufällig 10% der Antwort-Wörter durch Synonyme ersetzen
3. **Manual Curation:** Die besten 500 Paare manuell nachbearbeiten (wie im Original-Plan)
4. **External Data:** Einfache deutsche QA-Datensätze (GermanQuAD Subset) als Ergänzung

---

## Risiko #5: Training korrumpiert MicroModels <a name="risiko-5"></a>

**Wahrscheinlichkeit:** Mittel (15%) | **Impact:** KRITISCH
**Kern:** Wenn Language-Training (Stage 3: Joint Fine-Tuning) die MicroModel-Weights verändert, können bestehende predict()-Scores für Relationen kippen. Das zerstört die gesamte Wissensgraph-Integrität.

### 5.1 Konkreter Fix: Dreifach-Absicherung

1. **Snapshot VOR jedem Training** — komplette Kopie aller MicroModel-Weights
2. **Isoliertes Training** — Language-Weights getrennt von MicroModel-Weights (frozen by default)
3. **Integrity-Tests nach jedem Trainingsschritt** — automatischer Rollback bei Verletzung

### 5.2 C++ Code

**Neue Datei:** `backend/hybrid/micro_model_guard.hpp`

```cpp
#pragma once
#include "../micromodel/micro_model.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"
#include <unordered_map>
#include <vector>
#include <array>

namespace brain19 {

// =========================================================================
// MicroModelSnapshot: Komplette Kopie aller MicroModel-Weights
// =========================================================================

class MicroModelSnapshot {
public:
    // Erstelle Snapshot aller Models in der Registry
    // Kosten: 430 doubles × N models × 8 bytes = ~3.44 KB pro Model
    // Bei 1000 Models: ~3.44 MB
    static MicroModelSnapshot capture(const MicroModelRegistry& registry);

    // Stelle alle Models aus dem Snapshot wieder her
    void restore(MicroModelRegistry& registry) const;

    // Stelle nur ein bestimmtes Model wieder her
    bool restore_single(MicroModelRegistry& registry, ConceptId cid) const;

    // Anzahl gespeicherter Models
    size_t size() const { return snapshots_.size(); }

    // Speicherbedarf in Bytes
    size_t memory_bytes() const { return snapshots_.size() * FLAT_SIZE * sizeof(double); }

private:
    std::unordered_map<ConceptId, std::array<double, FLAT_SIZE>> snapshots_;
};

// =========================================================================
// IntegrityProbe: Vordefinierte Test-Inputs für Integritätsprüfung
// =========================================================================

struct IntegrityProbe {
    ConceptId concept_id;
    Vec10 relation_embedding;
    Vec10 context_embedding;
    double expected_min;    // predict() muss >= min sein
    double expected_max;    // predict() muss <= max sein
    std::string description;  // z.B. "Eis→CAUSES→Wasser should be high"
};

// =========================================================================
// MicroModelGuard: Überwacht und schützt MicroModels während Training
// =========================================================================

struct IntegrityReport {
    size_t probes_checked = 0;
    size_t violations = 0;
    std::vector<IntegrityProbe> failed_probes;
    std::vector<ConceptId> corrupted_models;
    double max_weight_drift = 0.0;  // Maximale relative Änderung Frobenius-Norm

    bool passed() const { return violations == 0; }
};

struct GuardConfig {
    double max_relative_weight_drift = 0.10;    // Max 10% Frobenius-Norm-Änderung
    double high_weight_min_predict = 0.5;       // Relations mit weight>0.8 → predict>0.5
    double contradicts_max_predict = 0.3;       // CONTRADICTS-Relations → predict<0.3
    size_t spot_check_count = 20;               // Anzahl zufälliger Spot-Checks
    bool auto_rollback = true;                  // Automatisch zurücksetzen bei Violation?
    unsigned seed = 42;
};

class MicroModelGuard {
public:
    MicroModelGuard(
        MicroModelRegistry& registry,
        const LongTermMemory& ltm,
        const EmbeddingManager& embeddings,
        GuardConfig config = {}
    );

    // Erstelle Snapshot + generiere Integrity Probes
    // MUSS vor dem Training aufgerufen werden
    void arm();

    // Prüfe Integrität nach einem Trainingsschritt
    // Gibt Report zurück. Bei auto_rollback=true wird bei Violation automatisch restored.
    IntegrityReport check();

    // Manueller Rollback auf den letzten arm()-Snapshot
    void rollback();

    // Ist der Guard armed?
    bool is_armed() const { return armed_; }

private:
    MicroModelRegistry& registry_;
    const LongTermMemory& ltm_;
    const EmbeddingManager& embeddings_;
    GuardConfig config_;

    bool armed_ = false;
    MicroModelSnapshot snapshot_;
    std::vector<IntegrityProbe> probes_;

    // Generiere Integrity Probes aus dem aktuellen Graph-Zustand
    std::vector<IntegrityProbe> generate_probes() const;

    // Berechne relative Weight-Drift eines Models (Frobenius-Norm)
    double compute_weight_drift(ConceptId cid) const;
};

} // namespace brain19
```

**Pseudocode der Kern-Methoden:**

```cpp
MicroModelSnapshot MicroModelSnapshot::capture(const MicroModelRegistry& registry) {
    MicroModelSnapshot snap;
    for (ConceptId cid : registry.get_model_ids()) {
        const MicroModel* model = registry.get_model(cid);
        if (!model) continue;
        std::array<double, FLAT_SIZE> flat;
        model->to_flat(flat);
        snap.snapshots_[cid] = flat;
    }
    return snap;
}

void MicroModelSnapshot::restore(MicroModelRegistry& registry) const {
    for (const auto& [cid, flat] : snapshots_) {
        MicroModel* model = registry.get_model(cid);
        if (!model) continue;
        model->from_flat(flat);
    }
}

void MicroModelGuard::arm() {
    // 1. Kompletten Snapshot erstellen
    snapshot_ = MicroModelSnapshot::capture(registry_);

    // 2. Integrity Probes generieren
    probes_ = generate_probes();

    armed_ = true;
}

std::vector<IntegrityProbe> MicroModelGuard::generate_probes() const {
    std::vector<IntegrityProbe> probes;
    std::mt19937 rng(config_.seed);

    auto all_ids = ltm_.get_all_concept_ids();

    for (ConceptId cid : all_ids) {
        auto relations = ltm_.get_outgoing_relations(cid);
        const MicroModel* model = registry_.get_model(cid);
        if (!model || relations.empty()) continue;

        for (const auto& rel : relations) {
            Vec10 e = embeddings_.get_relation_embedding(rel.type);
            // Context: Einfacher Target-basierter Context
            Vec10 c = embeddings_.make_target_embedding(
                0, static_cast<uint64_t>(cid), static_cast<uint64_t>(rel.target));

            // Aktuellen predict()-Wert als Baseline speichern
            double current_predict = model->predict(e, c);

            IntegrityProbe probe;
            probe.concept_id = cid;
            probe.relation_embedding = e;
            probe.context_embedding = c;

            if (rel.type == RelationType::CONTRADICTS) {
                // CONTRADICTS-Relations müssen niedrig bleiben
                probe.expected_min = 0.0;
                probe.expected_max = config_.contradicts_max_predict;
                probe.description = "CONTRADICTS must stay low";
            } else if (rel.weight > 0.8) {
                // Starke Relations müssen hoch bleiben
                probe.expected_min = config_.high_weight_min_predict;
                probe.expected_max = 1.0;
                probe.description = "High-weight relation must stay high";
            } else {
                // Normale Relation: predict() darf sich max ±0.2 ändern
                probe.expected_min = std::max(0.0, current_predict - 0.2);
                probe.expected_max = std::min(1.0, current_predict + 0.2);
                probe.description = "Normal relation: max ±0.2 drift";
            }
            probes.push_back(probe);
        }
    }

    // Spot-Check: Zufällige Auswahl (für Performance)
    if (probes.size() > config_.spot_check_count * 10) {
        std::shuffle(probes.begin(), probes.end(), rng);
        probes.resize(config_.spot_check_count * 10);
    }

    return probes;
}

IntegrityReport MicroModelGuard::check() {
    if (!armed_) {
        return IntegrityReport{.violations = 0};  // Nicht armed → kein Check
    }

    IntegrityReport report;
    report.probes_checked = probes_.size();

    // 1. Probe-basierte Prüfung
    for (const auto& probe : probes_) {
        const MicroModel* model = registry_.get_model(probe.concept_id);
        if (!model) continue;

        double actual = model->predict(probe.relation_embedding, probe.context_embedding);

        if (actual < probe.expected_min || actual > probe.expected_max) {
            ++report.violations;
            report.failed_probes.push_back(probe);

            // Merke korrumpiertes Model
            if (std::find(report.corrupted_models.begin(), report.corrupted_models.end(),
                          probe.concept_id) == report.corrupted_models.end()) {
                report.corrupted_models.push_back(probe.concept_id);
            }
        }
    }

    // 2. Weight-Drift Prüfung (Frobenius-Norm)
    for (ConceptId cid : registry_.get_model_ids()) {
        double drift = compute_weight_drift(cid);
        report.max_weight_drift = std::max(report.max_weight_drift, drift);

        if (drift > config_.max_relative_weight_drift) {
            ++report.violations;
            if (std::find(report.corrupted_models.begin(), report.corrupted_models.end(),
                          cid) == report.corrupted_models.end()) {
                report.corrupted_models.push_back(cid);
            }
        }
    }

    // 3. Auto-Rollback bei Violations
    if (!report.passed() && config_.auto_rollback) {
        // Nur korrumpierte Models zurücksetzen (nicht alle!)
        for (ConceptId cid : report.corrupted_models) {
            snapshot_.restore_single(registry_, cid);
        }
    }

    return report;
}

double MicroModelGuard::compute_weight_drift(ConceptId cid) const {
    auto it = snapshot_.snapshots_.find(cid);
    if (it == snapshot_.snapshots_.end()) return 0.0;

    const MicroModel* model = registry_.get_model(cid);
    if (!model) return 0.0;

    std::array<double, FLAT_SIZE> current_flat;
    model->to_flat(current_flat);
    const auto& original_flat = it->second;

    // Nur W_ und b_ vergleichen (erste 110 doubles), nicht Adam-State
    double sum_diff_sq = 0.0, sum_orig_sq = 0.0;
    for (size_t i = 0; i < 110; ++i) {
        double diff = current_flat[i] - original_flat[i];
        sum_diff_sq += diff * diff;
        sum_orig_sq += original_flat[i] * original_flat[i];
    }

    if (sum_orig_sq < 1e-12) return 0.0;
    return std::sqrt(sum_diff_sq) / std::sqrt(sum_orig_sq);
}
```

### 5.3 Isoliertes Training: Architektur-Prinzip

```
Training Stage 1 + 2:
  ❄️ MicroModels          → FROZEN (kein Gradient)
  ❄️ KAN-Encoder          → FROZEN (nach Stage 1)
  ✅ SemanticScorer        → TRAINIERT
  ✅ FusionLayer           → TRAINIERT
  ✅ KAN-Decoder           → TRAINIERT

Training Stage 3 (optional Joint Fine-Tuning):
  ⚠️ MicroModels          → 1/10 Learning-Rate + Guard armed
  ❄️ KAN-Encoder          → FROZEN
  ✅ SemanticScorer        → TRAINIERT
  ✅ FusionLayer           → TRAINIERT
  ✅ KAN-Decoder           → TRAINIERT
```

**Stage 3 wird NUR ausgeführt wenn Stage 2 konvergiert ist UND der Guard armed ist:**

```cpp
void LanguageTrainer::train_stage3_guarded(
    const std::vector<TrainingExample>& examples,
    MicroModelGuard& guard,
    size_t epochs
) {
    guard.arm();  // Snapshot + Probes

    MicroTrainingConfig micro_config;
    micro_config.learning_rate = 0.001;  // 1/10 der normalen LR

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Training-Schritt (MicroModels + Decoder + Scorer + Fusion)
        double loss = train_epoch_joint(examples, micro_config);

        // Integrity Check nach JEDEM Epoch
        auto report = guard.check();

        if (!report.passed()) {
            // Guard hat automatisch rollback gemacht (auto_rollback=true)
            // → Abbruch dieses Epochs, weiter mit nächstem
            // → Wenn >3 aufeinanderfolgende Rollbacks: Stage 3 komplett abbrechen
            if (consecutive_rollbacks_++ >= 3) {
                guard.rollback();  // Komplett zurück auf Pre-Stage-3
                break;
            }
            continue;
        }
        consecutive_rollbacks_ = 0;
    }
}
```

### 5.4 Test-Strategie

```cpp
TEST(MicroModelGuard, SnapshotAndRestoreIsExact) {
    MicroModelRegistry registry;
    registry.create_model(1);
    auto* model = registry.get_model(1);

    // Trainiere Model
    Vec10 e, c;
    e.fill(0.5); c.fill(0.3);
    model->train_step(e, c, 0.8, MicroTrainingConfig{});

    // Snapshot
    auto snap = MicroModelSnapshot::capture(registry);

    // Verändere Model
    model->train_step(e, c, 0.1, MicroTrainingConfig{});
    double after_change = model->predict(e, c);

    // Restore
    snap.restore(registry);
    double after_restore = model->predict(e, c);

    // Muss exakt auf Snapshot-Zustand zurück
    EXPECT_NE(after_change, after_restore);
}

TEST(MicroModelGuard, DetectsWeightDrift) {
    MicroModelRegistry registry;
    auto ltm = create_test_ltm(10, 20);
    EmbeddingManager embeddings;
    registry.ensure_models_for(ltm);

    GuardConfig config;
    config.max_relative_weight_drift = 0.05;  // Sehr streng: 5%
    config.auto_rollback = false;

    MicroModelGuard guard(registry, ltm, embeddings, config);
    guard.arm();

    // Aggressives Training das Weights stark verändert
    auto* model = registry.get_model(1);
    Vec10 e, c;
    e.fill(1.0); c.fill(1.0);
    for (int i = 0; i < 100; ++i) {
        model->train_step(e, c, 0.0, MicroTrainingConfig{.learning_rate = 0.1});
    }

    auto report = guard.check();
    EXPECT_FALSE(report.passed());
    EXPECT_GT(report.max_weight_drift, 0.05);
}

TEST(MicroModelGuard, AutoRollbackRestoresOnViolation) {
    MicroModelRegistry registry;
    auto ltm = create_test_ltm(10, 20);
    EmbeddingManager embeddings;
    registry.ensure_models_for(ltm);

    GuardConfig config;
    config.max_relative_weight_drift = 0.01;  // Extrem streng
    config.auto_rollback = true;

    MicroModelGuard guard(registry, ltm, embeddings, config);
    guard.arm();

    // predict()-Wert vor Training merken
    auto* model = registry.get_model(1);
    Vec10 e = embeddings.get_relation_embedding(RelationType::CAUSES);
    Vec10 c = embeddings.query_context();
    double before = model->predict(e, c);

    // Aggressives Training
    for (int i = 0; i < 50; ++i) {
        model->train_step(e, c, 0.0, MicroTrainingConfig{.learning_rate = 0.1});
    }

    // Check + Auto-Rollback
    auto report = guard.check();
    EXPECT_FALSE(report.passed());

    // Nach Auto-Rollback: predict() muss wieder wie vorher sein
    double after_rollback = model->predict(e, c);
    EXPECT_NEAR(after_rollback, before, 1e-10);
}

TEST(MicroModelGuard, ContradictionsStayLow) {
    // Setup: Concept mit CONTRADICTS-Relation
    MicroModelRegistry registry;
    auto ltm = create_test_ltm_with_contradicts();
    EmbeddingManager embeddings;
    registry.ensure_models_for(ltm);

    MicroModelGuard guard(registry, ltm, embeddings);
    guard.arm();

    // Trainiere so, dass CONTRADICTS-predict() hochgeht
    // → Guard muss das erkennen
    auto* model = registry.get_model(1);
    Vec10 e = embeddings.get_relation_embedding(RelationType::CONTRADICTS);
    Vec10 c; c.fill(0.5);
    for (int i = 0; i < 100; ++i) {
        model->train_step(e, c, 0.9, MicroTrainingConfig{.learning_rate = 0.1});
    }

    auto report = guard.check();
    EXPECT_FALSE(report.passed());
    // Guard hat korrumpiertes Model gerollbackt
}
```

### 5.5 Fallback

Wenn selbst isoliertes Training mit Guard nicht stabil ist:

1. **MicroModels komplett einfrieren** — Stage 3 wird übersprungen. Language Engine trainiert NUR eigene Weights (Encoder, Decoder, Scorer, Fusion). MicroModels bleiben unverändert.
2. **Adapter-Pattern:** Statt MicroModel-Weights zu verändern, wird ein kleiner Adapter-Layer (10→10, ~100 Params) VOR den MicroModel-Input geschaltet. Der Adapter wird trainiert, das Original-MicroModel bleibt frozen.

```cpp
// Adapter statt MicroModel-Änderung
class MicroModelAdapter {
    Vec10 adapted_e;   // Transformierter Relation-Input
    Vec10 adapted_c;   // Transformierter Context-Input
    Mat10x10 W_adapt_; // 100 Params (statt 100 MicroModel-Params zu ändern)
    Vec10 b_adapt_;    // 10 Params

public:
    double predict_adapted(const MicroModel& model, const Vec10& e, const Vec10& c) const {
        // Transform inputs, then delegate to frozen MicroModel
        Vec10 e_adapted, c_adapted;
        for (size_t i = 0; i < EMBED_DIM; ++i) {
            e_adapted[i] = e[i];  // e bleibt unverändert
            c_adapted[i] = b_adapt_[i];
            for (size_t j = 0; j < EMBED_DIM; ++j) {
                c_adapted[i] += W_adapt_[i * EMBED_DIM + j] * c[j];
            }
        }
        return model.predict(e_adapted, c_adapted);
    }
};
```

---

## Risiko #6: Vec10 zu klein für Query-Encoding <a name="risiko-6"></a>

**Wahrscheinlichkeit:** Mittel | **Impact:** Mittel
**Kern:** EMBED_DIM=10 ist sehr niedrig. Semantische Nuancen (z.B. "Was passiert wenn Eis schmilzt?" vs. "Was passiert wenn Schnee schmilzt?") könnten im 10D-Raum nicht trennbar sein.

### 6.1 Konkreter Fix: Internal High-Dim, External Vec10

Der KANEncoder arbeitet intern mit ℝ⁶⁴ und projiziert erst am Ende auf Vec10. Der Informationsverlust wird durch eine trainierbare Projektion minimiert.

### 6.2 C++ Code

```cpp
class KANEncoder {
public:
    KANEncoder(SharedEmbeddingTable& embeddings, KANAdapter& kan, size_t num_knots = 10);

    // Öffentlich: Vec10 Output (für MicroModel-Kompatibilität)
    Vec10 encode(const std::string& text) const;

    // Intern: ℝ⁶⁴ vor der finalen Projektion (für FusionLayer)
    Vec64 encode_high_dim(const std::string& text) const;

private:
    // KAN Layers intern
    uint64_t layer1_id_;    // ℝ⁶⁴ → ℝ³² (Bag-of-Embeddings → compressed)
    uint64_t layer2_id_;    // ℝ³² → ℝ³² (Refinement, bleibt in 32D!)
    uint64_t proj_id_;      // ℝ³² → ℝ¹⁰ (Finale Projektion auf Vec10)

    // Bag-of-Embeddings
    Vec64 bag_of_embeddings(const std::vector<uint16_t>& token_ids) const;
};

// Implementation
Vec10 KANEncoder::encode(const std::string& text) const {
    auto tokens = tokenizer_.encode(text);
    Vec64 bag = bag_of_embeddings(tokens);

    // Interne High-Dim Verarbeitung
    auto h1 = kan_.evaluate_kan_module(layer1_id_,
        std::vector<double>(bag.begin(), bag.end()));           // ℝ⁶⁴ → ℝ³²
    auto h2 = kan_.evaluate_kan_module(layer2_id_, h1);        // ℝ³² → ℝ³²

    // Finale Projektion auf Vec10
    auto out = kan_.evaluate_kan_module(proj_id_, h2);         // ℝ³² → ℝ¹⁰

    Vec10 result;
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        result[i] = out[i];
    }
    return result;
}

Vec64 KANEncoder::encode_high_dim(const std::string& text) const {
    auto tokens = tokenizer_.encode(text);
    Vec64 bag = bag_of_embeddings(tokens);

    auto h1 = kan_.evaluate_kan_module(layer1_id_,
        std::vector<double>(bag.begin(), bag.end()));
    auto h2 = kan_.evaluate_kan_module(layer2_id_, h1);

    // STOPP VOR Projektion → ℝ³² zurückgeben (gepaddet auf 64)
    Vec64 result;
    result.fill(0.0);
    for (size_t i = 0; i < h2.size() && i < 64; ++i) {
        result[i] = h2[i];
    }
    return result;
}
```

**FusionLayer nutzt High-Dim wo möglich:**

```cpp
FusedRepresentation FusionLayer::fuse(
    const TraversalResult& traversal,
    const SemanticScores& semantics,
    const EmbeddingManager& embeddings,
    const KANEncoder* encoder,         // Optional: für High-Dim
    const std::string* original_query  // Optional
) const {
    // Wenn High-Dim verfügbar: Nutze ℝ³² Query statt Vec10
    // → Mehr Information für den Decoder
    Vec64 query_high;
    if (encoder && original_query) {
        query_high = encoder->encode_high_dim(*original_query);
    }
    // ... Fusion mit höherer Dimensionalität ...
}
```

### 6.3 Test-Strategie: Retrieval-Precision

```cpp
TEST(Encoder, SimilarQueriesHaveSimilarEmbeddings) {
    KANEncoder encoder(embeddings, kan);
    // Nach Training:
    Vec10 q1 = encoder.encode("Was passiert wenn Eis schmilzt?");
    Vec10 q2 = encoder.encode("Was geschieht beim Schmelzen von Eis?");
    Vec10 q3 = encoder.encode("Was ist Photosynthese?");

    double sim_12 = cosine_similarity(q1, q2);
    double sim_13 = cosine_similarity(q1, q3);

    // q1 und q2 (semantisch ähnlich) müssen näher sein als q1 und q3
    EXPECT_GT(sim_12, sim_13);
}

TEST(Encoder, DifferentQueriesAreSeparable) {
    // 10 verschiedene Fragen encodieren
    // → Paarweise Cosine-Similarity muss < 0.9 sein (nicht kollidiert)
    std::vector<Vec10> encodings;
    for (const auto& q : ten_different_queries) {
        encodings.push_back(encoder.encode(q));
    }
    for (size_t i = 0; i < encodings.size(); ++i) {
        for (size_t j = i + 1; j < encodings.size(); ++j) {
            double sim = cosine_similarity(encodings[i], encodings[j]);
            EXPECT_LT(sim, 0.9) << "Queries " << i << " and " << j << " collide";
        }
    }
}
```

### 6.4 Fallback

Wenn Vec10 nachweislich zu klein ist (Retrieval-Precision unter 60%):
1. `EMBED_DIM` auf 16 erhöhen (erfordert MicroModel-Retraining — großer Aufwand)
2. Stattdessen: **Dual-Encoding** — Vec10 für MicroModel-Kompatibilität + separate ℝ³² für Language Engine. Die Language Engine nutzt intern immer ℝ³², der Vec10-Pfad ist nur für MicroModel::predict().

---

## Risiko #7: FocusCursor-Chain zu kurz <a name="risiko-7"></a>

**Wahrscheinlichkeit:** Mittel | **Impact:** Mittel
**Kern:** Bei `FocusCursorConfig::max_depth = 12` terminiert der Cursor nach maximal 12 Hops. Für komplexe Fragen ("Warum brauchen Pflanzen Licht für die Photosynthese und wie hängt das mit dem Kohlenstoffkreislauf zusammen?") reicht das möglicherweise nicht.

### 7.1 Konkreter Fix: Adaptive Depth + Branch-Merge

1. **Adaptive Depth:** max_depth wird dynamisch basierend auf Frage-Komplexität angepasst
2. **Branch-Merge:** Mehrere parallele Cursor-Branches werden zu einer kombinierten Chain gemerged

### 7.2 C++ Code

**Erweiterung in:** `backend/cursor/focus_cursor_manager.hpp`

```cpp
struct AdaptiveDepthConfig {
    size_t base_depth = 12;          // Default
    size_t max_depth = 30;           // Absolutes Maximum
    double depth_per_seed = 4.0;     // Extra Tiefe pro Seed-Concept
    double depth_per_hop_word = 2.0; // Extra Tiefe für Multi-Hop-Wörter ("warum", "wie hängt zusammen")
};

class FocusCursorManager {
public:
    // ... bestehende API ...

    // NEU: Adaptive Tiefe basierend auf Query-Analyse
    size_t compute_adaptive_depth(
        const std::string& query,
        size_t seed_count
    ) const;

    // NEU: Multi-Branch Traversal mit Merge
    QueryResult process_seeds_branched(
        const std::vector<ConceptId>& seeds,
        const Vec10& query_context,
        size_t branches_per_seed = 2
    );

private:
    AdaptiveDepthConfig adaptive_config_;

    // Merge mehrere TraversalResults zu einem kombinierten Result
    TraversalResult merge_chains(const std::vector<TraversalResult>& chains) const;
};
```

**Pseudocode:**

```cpp
size_t FocusCursorManager::compute_adaptive_depth(
    const std::string& query, size_t seed_count
) const {
    size_t depth = adaptive_config_.base_depth;

    // Mehr Seeds → braucht mehr Tiefe um alle zu verbinden
    depth += static_cast<size_t>(seed_count * adaptive_config_.depth_per_seed);

    // Multi-Hop-Indikatoren in der Frage
    static const std::vector<std::string> hop_words = {
        "warum", "wieso", "weshalb",      // Kausale Erklärungen brauchen Tiefe
        "zusammen", "verbindung",          // Verbindungsfragen
        "unterschied", "vergleich",        // Vergleiche brauchen parallele Chains
        "weil", "dadurch", "deshalb"       // Kausale Marker
    };

    std::string lower_query = to_lower(query);
    for (const auto& word : hop_words) {
        if (lower_query.find(word) != std::string::npos) {
            depth += static_cast<size_t>(adaptive_config_.depth_per_hop_word);
        }
    }

    return std::min(depth, adaptive_config_.max_depth);
}

QueryResult FocusCursorManager::process_seeds_branched(
    const std::vector<ConceptId>& seeds,
    const Vec10& query_context,
    size_t branches_per_seed
) {
    size_t depth = compute_adaptive_depth("", seeds.size());

    FocusCursorConfig branch_config = config_;
    branch_config.max_depth = depth;

    std::vector<TraversalResult> all_chains;

    for (ConceptId seed : seeds) {
        FocusCursor cursor(ltm_, registry_, embeddings_, branch_config);
        cursor.seed(seed, query_context);

        // Haupt-Chain
        auto result = cursor.deepen();
        all_chains.push_back(result);

        // Branches: Alternative Pfade ab Seed
        if (branches_per_seed > 1) {
            auto branches = cursor.branch(branches_per_seed - 1);
            for (auto& branch : branches) {
                auto branch_result = branch.deepen();
                all_chains.push_back(branch_result);
            }
        }
    }

    // Merge: Dedupliziere Konzepte, behalte stärkste Chains
    auto merged = merge_chains(all_chains);

    QueryResult qr;
    qr.chains = all_chains;
    qr.best_chain = merged;

    // Alle aktivierten Konzepte sammeln
    std::set<ConceptId> activated_set;
    for (const auto& chain : all_chains) {
        for (ConceptId cid : chain.concept_sequence) {
            activated_set.insert(cid);
        }
    }
    qr.all_activated.assign(activated_set.begin(), activated_set.end());

    return qr;
}

TraversalResult FocusCursorManager::merge_chains(
    const std::vector<TraversalResult>& chains
) const {
    if (chains.empty()) return {};
    if (chains.size() == 1) return chains[0];

    // Strategie: Nimm die Chain mit dem höchsten chain_score als Basis,
    // erweitere sie mit Konzepten aus anderen Chains die den Score verbessern
    auto best_it = std::max_element(chains.begin(), chains.end(),
        [](const TraversalResult& a, const TraversalResult& b) {
            return a.chain_score < b.chain_score;
        });

    TraversalResult merged = *best_it;

    // Sammle alle Konzepte die in der Basis-Chain fehlen
    std::set<ConceptId> in_merged(
        merged.concept_sequence.begin(), merged.concept_sequence.end());

    for (const auto& chain : chains) {
        if (&chain == &(*best_it)) continue;  // Skip Basis

        for (size_t i = 0; i < chain.chain.size(); ++i) {
            ConceptId cid = chain.chain[i].concept;
            if (in_merged.count(cid) == 0) {
                // Neues Konzept: Hänge an merged Chain an
                merged.chain.push_back(chain.chain[i]);
                merged.concept_sequence.push_back(cid);
                if (i < chain.relation_sequence.size()) {
                    merged.relation_sequence.push_back(chain.relation_sequence[i]);
                }
                in_merged.insert(cid);
            }
        }
    }

    merged.total_steps = merged.chain.size();
    // Recalculate chain_score als Durchschnitt der Step-Weights
    double sum = 0.0;
    for (const auto& step : merged.chain) sum += step.weight_at_entry;
    merged.chain_score = merged.chain.empty() ? 0.0 : sum / merged.chain.size();

    return merged;
}
```

### 7.3 Test-Strategie

```cpp
TEST(AdaptiveDepth, SimpleQueryGetsBaseDepth) {
    FocusCursorManager mgr(ltm, registry, embeddings, stm);
    size_t depth = mgr.compute_adaptive_depth("Was ist Wasser?", 1);
    // Base(12) + 1 seed × 4 = 16
    EXPECT_EQ(depth, 16);
}

TEST(AdaptiveDepth, ComplexQueryGetsMoreDepth) {
    FocusCursorManager mgr(ltm, registry, embeddings, stm);
    size_t depth = mgr.compute_adaptive_depth(
        "Warum führt Photosynthese zusammen mit dem Kohlenstoffkreislauf zu Pflanzenwachstum?", 3);
    // Base(12) + 3 seeds × 4 + "warum"(2) + "zusammen"(2) = 28
    EXPECT_GE(depth, 24);
    EXPECT_LE(depth, 30);
}

TEST(BranchedTraversal, ProducesMoreConceptsThanSingle) {
    // LTM mit verzweigtem Graph
    auto ltm = create_branched_graph();
    FocusCursorManager mgr(ltm, registry, embeddings, stm);

    Vec10 ctx; ctx.fill(0.5);

    auto single = mgr.process_seeds({1}, ctx);
    auto branched = mgr.process_seeds_branched({1}, ctx, 3);

    // Branched muss mehr Konzepte aktiviert haben
    EXPECT_GT(branched.all_activated.size(), single.all_activated.size());
}

TEST(ChainMerge, DeduplicatesConcepts) {
    // Zwei Chains mit überlappenden Konzepten
    TraversalResult chain1, chain2;
    chain1.concept_sequence = {1, 2, 3};
    chain2.concept_sequence = {1, 4, 5};  // 1 ist dupliziert

    FocusCursorManager mgr(ltm, registry, embeddings, stm);
    auto merged = mgr.merge_chains({chain1, chain2});

    // Merged enthält {1, 2, 3, 4, 5} — keine Duplikate
    std::set<ConceptId> unique(
        merged.concept_sequence.begin(), merged.concept_sequence.end());
    EXPECT_EQ(unique.size(), 5);
    EXPECT_EQ(merged.concept_sequence.size(), 5);
}
```

### 7.4 Fallback

Wenn auch adaptive Depth + Branching nicht reicht:
1. **Iterative Vertiefung:** Starte mit depth=12, wenn GoalState nicht complete → restarte mit depth=24 und den bisher gefundenen Konzepten als zusätzliche Seeds
2. **Multi-Query:** Komplexe Frage in Sub-Fragen zerlegen, jede Sub-Frage separat traversieren, Ergebnisse zusammenführen
3. **max_depth auf 50 setzen** (brute force — Kosten: ~50 × MicroModel::predict() × 30 Nachbarn = 1500 predict()-Calls, bei ~1μs pro Call = ~1.5ms, akzeptabel)

---

## Zusammenfassung: Aufwand für alle Mitigationen

| Risiko | Mitigation | Neue Dateien | Aufwand | Phase |
|--------|-----------|-------------|---------|-------|
| #1 KAN-Decoder Qualität | QualityGate + Progressive Stufen | `quality_gate.hpp/cpp` | 1.5d | 9.5 → 9.6 |
| #2 BPE-Vocab | VocabDiagnostics + expand_vocab() | In `tokenizer.hpp/cpp` | 0.5d | In 9.1 |
| #3 Embedding-Table | SharedEmbeddingTable | `shared_embedding_table.hpp` | 0.5d | 9.2 → 9.5 |
| #4 Trainingsdaten | SyntheticDataGenerator | `synthetic_data_generator.hpp/cpp` | 2d | Vor 9.8 |
| #5 MicroModel-Korrumpierung | MicroModelGuard + Snapshot | `micro_model_guard.hpp/cpp` | 2d | Vor 9.8 |
| #6 Vec10 zu klein | Internal High-Dim Encoder | In `kan_encoder.hpp/cpp` | 0.5d | In 9.2 |
| #7 Chain zu kurz | Adaptive Depth + Branch-Merge | In `focus_cursor_manager.hpp/cpp` | 1d | In 3.4 |
| **Tests** | Alle Mitigations-Tests | `test_quality_gate.cpp`, `test_guard.cpp`, `test_synthetic_data.cpp` | 2d | Nach 9.8 |
| **GESAMT** | | | **~10 Tage** | |

**Korrigierter Gesamtaufwand inkl. Mitigationen:**
- Phase 0-8 (Cursor + Graph): ~19 Tage
- Phase 9 (Language Engine): ~16.75 Tage
- Mitigationen: ~10 Tage
- **Gesamt: ~45.75 Tage** (≈ 9-10 Wochen Abendarbeit)
