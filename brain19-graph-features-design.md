# Brain19 Graph Features Design — Systems Architecture

> **Autor:** Brain19 Architekt  
> **Datum:** 2026-02-16  
> **Status:** Design-Dokument  
> **Scope:** 3 zusammenhängende Features für den Wissensgraph

---

## Architektur-Prinzipien

1. **Maximale Wiederverwendung** — Keine Parallel-Infrastruktur. Neue Features sind Erweiterungen, keine Neubauten.
2. **Graph-Native** — Alles ist Concept + Relation. Kein Sonderdatenmodell.
3. **Lazy Evaluation** — Komplexe Berechnungen (Similarity, Complexity) werden bei Bedarf berechnet, nicht bei Insertion.
4. **Feature-Synergie** — Die 3 Features bilden einen Kreislauf: Sprach-Graph liefert Struktur → Trust-Propagation bewertet → Retention entscheidet über Lebensdauer.

---

## Feature 1: Sprach-Graph (Ebene 2)

### 1.1 Konzept

Grammatische Strukturen werden als First-Class Concepts im bestehenden Graph modelliert. Ein Satz wie *"Katzen jagen Mäuse"* erzeugt:

```
[Sentence_0x4A2]  ──SUBJECT_OF──→  [Word:Katzen]
                   ──VERB_OF────→  [Word:jagen]
                   ──OBJECT_OF──→  [Word:Mäuse]
```

Jedes Wort-Concept hat ein `surface_form` Property und verlinkt via `DENOTES` auf sein semantisches Concept im Wissensgraph.

```
[Word:Katzen] ──DENOTES──→ [Concept:Katze]   (bestehend, ~30K)
[Word:jagen]  ──DENOTES──→ [Concept:Jagd]    (bestehend)
```

### 1.2 Bestehende Klassen/Methoden — Wiederverwendung

| Klasse | Methode | Nutzung |
|--------|---------|---------|
| `LongTermMemory` | `store_concept()` | Satz-Concepts, Wort-Concepts speichern |
| `LongTermMemory` | `add_relation()` | Grammatische Relations (SUBJECT_OF, VERB_OF etc.) |
| `LongTermMemory` | `get_outgoing_relations()` | Satz→Teile traversieren |
| `GraphDensifier` | `pair_exists()` | Duplikat-Check: existiert Word:X ──DENOTES──→ Concept:Y schon? |
| `ConceptProposal` | `structural_similarity()` | Satz-Ähnlichkeit für Paraphrasen-Erkennung |
| `EpistemicPromotion` | Promotion-Logik | Satz-Level Trust (wiederholte Sätze → höherer EpistemicType) |

### 1.3 Erweiterungen bestehender Klassen

#### `RelationType` enum — Erweitern

```cpp
// Bestehend: IS_A, HAS_PROPERTY, CAUSES, etc.
// NEU:
enum class RelationType {
    // ... bestehende ...
    
    // Grammatik-Relations (Ebene 2)
    SUBJECT_OF,      // Wort → Satz: "Katzen" ist Subjekt von Satz_X
    OBJECT_OF,       // Wort → Satz: "Mäuse" ist Objekt von Satz_X  
    VERB_OF,         // Wort → Satz: "jagen" ist Verb von Satz_X
    MODIFIER_OF,     // Wort → Wort/Satz: "schnell" modifiziert "jagen"
    DENOTES,         // Wort-Concept → Semantisches Concept
    PART_OF_SENTENCE,// Generisch: Wort gehört zu Satz (Fallback)
    TEMPORAL_OF,     // Zeitangabe → Satz
    LOCATIVE_OF,     // Ortsangabe → Satz
    PRECEDES,        // Satz → Satz (Diskurs-Reihenfolge)
};
```

#### `Concept` struct — Erweitern

```cpp
struct Concept {
    // ... bestehende Felder ...
    
    // NEU: Concept-Layer Marker (kein neues Feld, nutze bestehende Properties)
    // Convention: property "layer" = "semantic" | "linguistic" | "phonetic"
    // Convention: property "surface_form" = "Katzen" (nur für Wort-Concepts)
    // Convention: property "pos_tag" = "NOUN" | "VERB" | "ADJ" | ... 
};
```

**Kein neues struct-Feld nötig.** Layer-Info wird als Property im bestehenden Property-System gespeichert. Das hält Concept schlank und vermeidet Schema-Migration.

#### `GraphDensifier` — Erweitern

```cpp
class GraphDensifier {
    // ... bestehende Methoden ...
    
    // NEU: Duplikat-Check für Wort-Concepts
    // Nutzt pair_exists() intern, erweitert um surface_form-Match
    bool word_concept_exists(const std::string& surface_form, 
                             const std::string& pos_tag) const;
    
    // NEU: Satz-Duplikat-Erkennung via strukturelle Ähnlichkeit
    // Prüft ob ein Satz mit gleichen SUBJECT/VERB/OBJECT Relations existiert
    std::optional<ConceptId> find_duplicate_sentence(
        ConceptId subject, ConceptId verb, ConceptId object) const;
    
    // NEU: Generiert DENOTES-Relations wenn Wort↔Semantik-Mapping fehlt
    // Nutzt bestehende Co-Activation Logik
    std::vector<Relation> infer_denotes_relations(ConceptId word_concept) const;
};
```

#### `ConceptProposal` — Erweitern

```cpp
class ConceptProposal {
    // ... bestehende Methoden ...
    
    // NEU: Satz-Ähnlichkeit (nutzt structural_similarity intern)
    // Vergleicht grammatische Struktur: gleiche Rollen, ähnliche Füller
    float sentence_similarity(ConceptId sentence_a, ConceptId sentence_b) const;
    
    // NEU: Paraphrasen-Erkennung
    // sentence_similarity > threshold UND DENOTES-Ziele überlappen
    bool is_paraphrase(ConceptId sentence_a, ConceptId sentence_b, 
                       float threshold = 0.75f) const;
};
```

### 1.4 Neue Klassen

#### `SentenceParser` — Fassade für Satz→Graph Konversion

```cpp
// sentence_parser.h
#pragma once
#include "long_term_memory.h"
#include "graph_densifier.h"
#include "concept_proposal.h"

struct ParsedSentence {
    ConceptId sentence_id;
    ConceptId subject_word;    // Wort-Concept
    ConceptId verb_word;       // Wort-Concept  
    ConceptId object_word;     // Wort-Concept (optional)
    std::vector<ConceptId> modifiers;
    
    // DENOTES-Targets (semantische Concepts, nullable)
    std::optional<ConceptId> subject_semantic;
    std::optional<ConceptId> verb_semantic;
    std::optional<ConceptId> object_semantic;
};

class SentenceParser {
public:
    explicit SentenceParser(LongTermMemory& ltm, 
                            GraphDensifier& densifier,
                            ConceptProposal& proposal);
    
    // Hauptmethode: Satz → Graph
    // 1. Tokenize + POS-Tag (intern, regelbasiert)
    // 2. Für jedes Token: word_concept_exists() → sonst store_concept()
    // 3. Grammatische Relations via add_relation()
    // 4. DENOTES-Links via infer_denotes_relations() oder explizites Mapping
    // 5. Duplikat-Check via find_duplicate_sentence()
    ParsedSentence parse_and_store(const std::string& sentence);
    
    // Batch: Mehrere Sätze, erzeugt PRECEDES-Kette
    std::vector<ParsedSentence> parse_discourse(
        const std::vector<std::string>& sentences);
    
    // Lookup: Surface Form → alle Wort-Concepts mit diesem Label
    std::vector<ConceptId> lookup_word(const std::string& surface_form) const;
    
    // Semantik-Link nachträglich setzen
    void link_word_to_concept(ConceptId word, ConceptId semantic_concept);
    
private:
    LongTermMemory& ltm_;
    GraphDensifier& densifier_;
    ConceptProposal& proposal_;
    
    // Internes POS-Tagging (regelbasiert, kein ML)
    struct Token {
        std::string surface;
        std::string pos;      // NOUN, VERB, ADJ, ADV, DET, PREP, CONJ, PUNCT
        int position;
    };
    std::vector<Token> tokenize(const std::string& sentence) const;
    
    // Wort→Concept mit Duplikat-Check
    ConceptId get_or_create_word_concept(const Token& token);
};
```

**Design-Entscheidung:** `SentenceParser` besitzt KEIN eigenes Storage. Alles geht durch `LongTermMemory`. Der Parser ist reine Logik + Orchestrierung.

### 1.5 Datenfluß

```
Input: "Katzen jagen Mäuse"
  │
  ├─ tokenize() → [("Katzen",NOUN,0), ("jagen",VERB,1), ("Mäuse",NOUN,2)]
  │
  ├─ get_or_create_word_concept("Katzen", NOUN)
  │     └─ densifier_.word_concept_exists("Katzen", "NOUN")
  │           ├─ JA → return existing ConceptId
  │           └─ NEIN → ltm_.store_concept({layer:"linguistic", surface_form:"Katzen", pos:"NOUN"})
  │                     densifier_.infer_denotes_relations() → link zu Concept:Katze
  │
  ├─ [analog für "jagen", "Mäuse"]
  │
  ├─ densifier_.find_duplicate_sentence(Katzen, jagen, Mäuse)
  │     ├─ DUPLIKAT → return existing, EpistemicPromotion bumpt Trust
  │     └─ KEIN DUPLIKAT → ltm_.store_concept({layer:"linguistic", type:SENTENCE})
  │
  └─ ltm_.add_relation(Katzen_word, SUBJECT_OF, sentence)
     ltm_.add_relation(jagen_word, VERB_OF, sentence)
     ltm_.add_relation(Mäuse_word, OBJECT_OF, sentence)
```

---

## Feature 2: Trust-Propagation bei Invalidation

### 2.1 Konzept

Wenn ein Concept als falsch markiert wird (`invalidate_concept()`), soll das System automatisch **ähnliche Concepts im Trust herunterstufen**. "Ähnlich" wird durch drei Signale definiert:

1. **Strukturelle Ähnlichkeit** — gleiche Relation-Patterns (via `ConceptProposal::structural_similarity`)
2. **Co-Activation** — häufig gemeinsam aktivierte Concepts (via `GraphDensifier`)
3. **Gemeinsame Herkunft** — gleiche Quell-Relations (z.B. gleicher Inference-Pfad)

### 2.2 Bestehende Klassen — Wiederverwendung

| Klasse | Methode | Nutzung |
|--------|---------|---------|
| `LongTermMemory` | `invalidate_concept()` | Auslöser der Propagation |
| `LongTermMemory` | `get_outgoing_relations()` | Nachbar-Traversierung |
| `LongTermMemory` | `get_incoming_relations()` | Rückwärts-Traversierung |
| `LongTermMemory` | `get_relations_between()` | Direkte Verbindung prüfen |
| `ConceptProposal` | `structural_similarity()` | Ähnlichkeits-Score |
| `EpistemicPromotion` | `contradiction_ratio()` | Vorhandene Widersprüche gewichten |
| `EpistemicPromotion` | `count_supporting_relations()` | Stützende Evidenz zählen |
| `GraphDensifier` | Co-Activation Logik | Cluster-Ähnlichkeit |

### 2.3 Erweiterungen bestehender Klassen

#### `LongTermMemory` — Hook erweitern

```cpp
class LongTermMemory {
    // ... bestehend ...
    
    // ERWEITERT: invalidate_concept() bekommt optionalen Propagation-Parameter
    void invalidate_concept(ConceptId id, 
                            bool propagate_trust = true,    // NEU
                            float propagation_radius = 0.5f); // NEU: min similarity für Propagation
    
    // NEU: Callback-System für Invalidation-Events
    using InvalidationCallback = std::function<void(ConceptId, float old_trust)>;
    void register_invalidation_hook(InvalidationCallback cb);
    
private:
    std::vector<InvalidationCallback> invalidation_hooks_;  // NEU
};
```

#### `EpistemicPromotion` — Trust-Adjustment erweitern

```cpp
class EpistemicPromotion {
    // ... bestehend ...
    
    // NEU: Trust-Reduktion basierend auf Ähnlichkeit zu invalidiertem Concept
    // Returns: Map von betroffenen Concepts → neuer Trust-Wert
    std::unordered_map<ConceptId, float> compute_trust_propagation(
        ConceptId invalidated,
        float propagation_radius) const;
    
    // NEU: Anwenden der berechneten Trust-Änderungen
    void apply_trust_propagation(
        const std::unordered_map<ConceptId, float>& adjustments);
    
    // NEU: Prüfen ob Concept durch mehrfache Propagation Trust 0 erreicht hat
    bool should_force_invalidate(ConceptId id) const;
    
    // ERWEITERT: Demotion-Logik nutzt jetzt auch Propagation-History
    // Bestehende contradiction_ratio() bleibt, wird aber intern um 
    // propagation_penalty ergänzt
};
```

### 2.4 Neue Klassen

#### `TrustPropagator` — Orchestrierung der Propagation

```cpp
// trust_propagator.h
#pragma once
#include "long_term_memory.h"
#include "epistemic_promotion.h"
#include "concept_proposal.h"
#include "graph_densifier.h"

struct PropagationResult {
    ConceptId source;                          // Das invalidierte Concept
    std::vector<std::pair<ConceptId, float>> affected;  // (Concept, neuer Trust)
    std::vector<ConceptId> force_invalidated;  // Trust→0 durch Kumulierung
    size_t concepts_checked;
    size_t concepts_adjusted;
};

struct PropagationConfig {
    float similarity_threshold = 0.5f;  // Min. Ähnlichkeit für Propagation
    float max_trust_reduction = 0.3f;   // Max Trust-Reduktion pro Propagation
    float cumulative_invalidation_threshold = 0.1f;  // Trust unter dem → force invalidate
    size_t max_hop_distance = 3;        // Max Graph-Distanz für Kandidaten
    bool propagate_to_linguistic = false; // Sprach-Graph mitpropagieren?
};

class TrustPropagator {
public:
    TrustPropagator(LongTermMemory& ltm,
                    EpistemicPromotion& promotion,
                    ConceptProposal& proposal,
                    GraphDensifier& densifier,
                    PropagationConfig config = {});
    
    // Hauptmethode: Wird von LongTermMemory::invalidate_concept() getriggert
    PropagationResult propagate(ConceptId invalidated);
    
    // Ähnlichkeits-Score zwischen zwei Concepts (kombiniert alle Signale)
    // Gewichtung: structural 0.4, co-activation 0.35, shared_source 0.25
    float combined_similarity(ConceptId a, ConceptId b) const;
    
    // Trust-Reduktion berechnen (nicht anwenden)
    // reduction = base_trust * similarity * (1 - support_ratio)
    float compute_reduction(ConceptId target, 
                            float similarity_to_invalidated) const;
    
    // History: Welche Propagations haben ein Concept betroffen?
    std::vector<ConceptId> get_propagation_sources(ConceptId target) const;
    
private:
    LongTermMemory& ltm_;
    EpistemicPromotion& promotion_;
    ConceptProposal& proposal_;
    GraphDensifier& densifier_;
    PropagationConfig config_;
    
    // Tracking: ConceptId → Liste von (source_invalidation, reduction_applied)
    std::unordered_map<ConceptId, 
        std::vector<std::pair<ConceptId, float>>> propagation_history_;
    
    // Kandidaten-Suche: Finde ähnliche Concepts innerhalb max_hop_distance
    std::vector<ConceptId> find_candidates(ConceptId source) const;
    
    // Co-Activation Score (nutzt GraphDensifier intern)
    float co_activation_score(ConceptId a, ConceptId b) const;
    
    // Shared-Source Score: Wieviele gemeinsame Eingangs-Relations?
    float shared_source_score(ConceptId a, ConceptId b) const;
};
```

### 2.5 Trust-Reduktions-Formel

```
Für jedes Kandidaten-Concept C, gegeben invalidiertes Concept I:

similarity = 0.40 * structural_similarity(I, C)      // ConceptProposal
           + 0.35 * co_activation_score(I, C)         // GraphDensifier  
           + 0.25 * shared_source_score(I, C)          // Relations-Overlap

IF similarity >= config.similarity_threshold:
    support = count_supporting_relations(C) / max(total_relations(C), 1)
    contradiction = contradiction_ratio(C)
    
    reduction = C.trust 
              * similarity 
              * (1.0 - support)
              * (1.0 + contradiction)   // Widersprüche verstärken Reduktion
              
    reduction = min(reduction, config.max_trust_reduction)
    
    C.trust = max(C.trust - reduction, 0.0)
    
    IF C.trust < config.cumulative_invalidation_threshold:
        force_invalidate(C)  // → Löst rekursive Propagation aus (mit depth limit!)
```

### 2.6 Datenfluß

```
invalidate_concept(Concept_A)
  │
  ├─ EpistemicStatus → INVALIDATED, Trust → 0.0
  │
  ├─ [Hook triggert TrustPropagator::propagate(A)]
  │     │
  │     ├─ find_candidates(A)  // BFS bis max_hop_distance=3
  │     │     └─ Filterung: nur ACTIVE/CONTEXTUAL Concepts
  │     │
  │     ├─ Für jeden Kandidaten C:
  │     │     ├─ combined_similarity(A, C)
  │     │     ├─ Falls > threshold: compute_reduction(C, similarity)
  │     │     └─ apply_trust_propagation({C → new_trust})
  │     │
  │     ├─ Falls C.trust < 0.1 → force_invalidate(C)
  │     │     └─ REKURSIV: propagate(C) mit depth_limit--
  │     │
  │     └─ Return PropagationResult
  │
  └─ [Feature 3: Retention-Check für A und alle force-invalidated]
```

---

## Feature 3: Komplexitäts-basierte Retention

### 3.1 Konzept

Nicht alles Falsche ist gleich wertlos. Ein einfacher Irrtum (*"Berlin ist die Hauptstadt von Frankreich"*) kann vergessen werden. Aber ein komplexer Fehlschluss (*"Aus A folgt B, B impliziert C, also muss D gelten — aber D ist falsch"*) ist wertvolles Anti-Wissen: Er dokumentiert einen Denkfehler, den das System nicht wiederholen sollte.

**Entscheidungsmatrix:**

```
                    Einfach (< threshold)    Komplex (≥ threshold)
                   ┌────────────────────────┬────────────────────────┐
  Trust > 0        │  Normal (aktiv)         │  Normal (aktiv)        │
                   ├────────────────────────┼────────────────────────┤
  Trust = 0        │  GC-Kandidat            │  ANTI-KNOWLEDGE        │
  (INVALIDATED)    │  (kann vergessen        │  (behalten als         │
                   │   werden)               │   bekannter Fehlpfad)  │
                   └────────────────────────┴────────────────────────┘
```

### 3.2 Bestehende Klassen — Wiederverwendung

| Klasse | Methode | Nutzung |
|--------|---------|---------|
| `LongTermMemory` | `get_outgoing_relations()` | Kausalketten-Traversierung |
| `LongTermMemory` | `get_incoming_relations()` | Abhängigkeiten zählen |
| `LongTermMemory` | `invalidate_concept()` | Auslöser der Retention-Entscheidung |
| `GraphDensifier` | Causal Transitivity | Kausalketten-Erkennung |
| `EpistemicPromotion` | EpistemicType | INFERENCE-Ketten = höhere Komplexität |
| `ConceptProposal` | `structural_similarity()` | Anti-Knowledge Clustering |

### 3.3 Erweiterungen bestehender Klassen

#### `Concept` struct — Flag erweitern

```cpp
struct Concept {
    // ... bestehende Felder ...
    
    // NEU: Anti-Knowledge Marker
    bool is_anti_knowledge = false;     // Falsches Wissen, bewusst behalten
    float complexity_score = 0.0f;      // Cached Komplexitäts-Score
    
    // NEU: Optional — Warum wurde es behalten?
    // Gespeichert als Property: "retention_reason" = "causal_chain_depth:5,involved_concepts:12"
};
```

**Design-Entscheidung:** `is_anti_knowledge` als echtes Bool-Feld (nicht Property), weil es im GC-Hot-Path geprüft wird. `complexity_score` als Cache, weil die Berechnung O(n) ist und nicht bei jedem GC-Lauf neu berechnet werden soll.

#### `LongTermMemory` — GC erweitern

```cpp
class LongTermMemory {
    // ... bestehend ...
    
    // ERWEITERT: Garbage Collection respektiert Anti-Knowledge
    // Bestehende GC-Logik (falls vorhanden) wird um Retention-Check erweitert
    size_t garbage_collect(size_t max_removals = 1000);
    
    // NEU: Anti-Knowledge Queries
    std::vector<ConceptId> get_anti_knowledge() const;
    std::vector<ConceptId> get_gc_candidates() const;  // INVALIDATED && !is_anti_knowledge
    
    // NEU: Concept als Anti-Knowledge markieren
    void mark_as_anti_knowledge(ConceptId id, const std::string& reason);
    
    // NEU: Anti-Knowledge aufheben (z.B. wenn Kontext sich ändert)
    void unmark_anti_knowledge(ConceptId id);
};
```

#### `EpistemicPromotion` — Retention-Awareness

```cpp
class EpistemicPromotion {
    // ... bestehend ...
    
    // NEU: Bei Invalidation → Retention-Check einbauen
    // Wird NACH invalidate_concept() und NACH trust_propagation aufgerufen
    void evaluate_retention(ConceptId invalidated);
};
```

### 3.4 Neue Klassen

#### `ComplexityAnalyzer` — Komplexitäts-Bewertung

```cpp
// complexity_analyzer.h
#pragma once
#include "long_term_memory.h"
#include "graph_densifier.h"

struct ComplexityMetrics {
    size_t causal_chain_length;    // Längste CAUSES-Kette durch dieses Concept
    size_t involved_concepts;      // Unique Concepts in der Kausalkette
    size_t relation_depth;         // Max Tiefe des Subgraphs (BFS-Distanz)
    size_t inference_steps;        // Anzahl INFERENCE-typed Concepts in der Kette
    float normalized_score;        // Gewichtete Kombination, [0.0 - 1.0]
};

struct RetentionConfig {
    float complexity_threshold = 0.4f;       // Ab diesem Score → Anti-Knowledge
    size_t min_causal_chain = 3;             // Mindest-Kettenlänge
    size_t min_involved_concepts = 5;        // Mindest-beteiligte Concepts
    size_t max_traversal_depth = 10;         // BFS-Limit für Komplexitäts-Analyse
    
    // Gewichte für normalized_score Berechnung
    float weight_causal_chain = 0.35f;
    float weight_involved_concepts = 0.25f;
    float weight_relation_depth = 0.20f;
    float weight_inference_steps = 0.20f;
};

class ComplexityAnalyzer {
public:
    ComplexityAnalyzer(LongTermMemory& ltm,
                       GraphDensifier& densifier,
                       RetentionConfig config = {});
    
    // Hauptmethode: Komplexität eines Concepts berechnen
    ComplexityMetrics analyze(ConceptId id) const;
    
    // Entscheidung: Soll dieses invalidierte Concept behalten werden?
    bool should_retain(ConceptId invalidated) const;
    
    // Batch: Alle INVALIDATED Concepts bewerten und markieren
    // Returns: Anzahl neu markierter Anti-Knowledge Concepts
    size_t evaluate_all_invalidated();
    
    // Kausalkette extrahieren (nützlich für Erklärungen)
    // Gibt die längste CAUSES-Kette zurück, die durch dieses Concept geht
    std::vector<ConceptId> extract_causal_chain(ConceptId id) const;
    
    // Subgraph extrahieren: Alle Concepts die transitiv abhängen
    std::unordered_set<ConceptId> extract_dependency_subgraph(
        ConceptId id, size_t max_depth) const;
    
private:
    LongTermMemory& ltm_;
    GraphDensifier& densifier_;
    RetentionConfig config_;
    
    // BFS für Kausalketten
    size_t longest_causal_chain(ConceptId start) const;
    
    // Zähle INFERENCE-typed Concepts in Kette
    size_t count_inference_steps(const std::vector<ConceptId>& chain) const;
};
```

#### `RetentionManager` — Orchestrierung von GC + Anti-Knowledge

```cpp
// retention_manager.h
#pragma once
#include "long_term_memory.h"
#include "complexity_analyzer.h"
#include "trust_propagator.h"

struct RetentionStats {
    size_t total_invalidated;
    size_t marked_anti_knowledge;
    size_t gc_candidates;
    size_t actually_removed;
    std::vector<ConceptId> new_anti_knowledge;  // In diesem Lauf neu markiert
};

class RetentionManager {
public:
    RetentionManager(LongTermMemory& ltm,
                     ComplexityAnalyzer& analyzer,
                     TrustPropagator& propagator);
    
    // Vollständiger Retention-Zyklus nach Invalidation
    // 1. ComplexityAnalyzer::should_retain() für das invalidierte Concept
    // 2. Falls retain → mark_as_anti_knowledge()
    // 3. Falls nicht retain → gc_candidate
    void on_invalidation(ConceptId invalidated);
    
    // Periodischer GC-Lauf
    // 1. Sammle alle GC-Kandidaten
    // 2. Prüfe ob sich Komplexität geändert hat (neue Relations seit letzter Analyse)
    // 3. Entferne einfache, behalt komplexe
    RetentionStats run_gc_cycle(size_t max_removals = 500);
    
    // Anti-Knowledge Lookup: "Warum war X falsch?"
    // Traversiert die Kausalkette und gibt eine Erklärung zurück
    std::string explain_anti_knowledge(ConceptId id) const;
    
    // Anti-Knowledge als Negativfilter: "Wurde dieser Pfad schon als falsch erkannt?"
    // Nutzt ConceptProposal::structural_similarity() gegen alle Anti-Knowledge Concepts
    bool resembles_known_error(ConceptId candidate, float threshold = 0.7f) const;
    
private:
    LongTermMemory& ltm_;
    ComplexityAnalyzer& analyzer_;
    TrustPropagator& propagator_;
};
```

### 3.5 Komplexitäts-Score Formel

```
Für ein invalidiertes Concept C:

causal_chain   = longest_causal_chain(C)           // z.B. 0-15
involved       = extract_dependency_subgraph(C).size()  // z.B. 0-50
depth          = max BFS distance from C           // z.B. 0-10
inferences     = count_inference_steps(chain)      // z.B. 0-10

// Normalisierung auf [0.0, 1.0] mit Sättigungsfunktion
norm(x, cap) = min(x / cap, 1.0)

normalized_score = weight_causal_chain     * norm(causal_chain, 10)
                 + weight_involved_concepts * norm(involved, 20)
                 + weight_relation_depth    * norm(depth, 8)
                 + weight_inference_steps   * norm(inferences, 5)

should_retain = normalized_score >= config.complexity_threshold
             && causal_chain >= config.min_causal_chain
             && involved >= config.min_involved_concepts
```

---

## Feature-Interaktionen

### Kreislauf der 3 Features

```
                    ┌──────────────────────────────┐
                    │     Feature 1: Sprach-Graph   │
                    │   (Grammatik als Knowledge)   │
                    └──────────┬───────────────────┘
                               │
                    Sätze erzeugen Concepts mit 
                    DENOTES→ semantische Concepts
                               │
                               ▼
              ┌────────────────────────────────────┐
              │  Bestehender Graph (~30K Concepts)  │
              │     EpistemicType, Trust, Status     │
              └───────────┬────────────────────────┘
                          │
            invalidate_concept() wird aufgerufen
            (extern oder durch Widerspruch)
                          │
                          ▼
              ┌────────────────────────────────────┐
              │  Feature 2: Trust-Propagation       │
              │  Ähnliche Concepts Trust ↓          │
              │  Ggf. kaskadierende Invalidation    │
              └───────────┬────────────────────────┘
                          │
            Für jedes invalidierte Concept:
                          │
                          ▼
              ┌────────────────────────────────────┐
              │  Feature 3: Retention               │
              │  Komplexität → behalten oder GC?    │
              │  Anti-Knowledge für Fehlpfade        │
              └────────────────────────────────────┘
```

### Konkrete Interaktionspunkte

#### 1→2: Sprach-Graph informiert Trust-Propagation

```cpp
// Wenn ein semantisches Concept invalidiert wird,
// werden AUCH die Sätze geprüft, die via DENOTES darauf verweisen.

// In TrustPropagator::find_candidates():
// ZUSÄTZLICH zu Graph-Nachbarn auch Satz-Concepts finden,
// deren DENOTES-Ziele dem invalidierten Concept ähneln.

// Beispiel:
// Concept:Pluto_ist_Planet invalidiert
// → Satz "Pluto ist der neunte Planet" (DENOTES→Pluto_ist_Planet)
// → Satz.trust wird reduziert
// → Weitere Sätze mit ähnlicher Struktur (sentence_similarity) 
//   werden ebenfalls geprüft
```

#### 2→3: Trust-Propagation triggert Retention-Entscheidung

```cpp
// In LongTermMemory::invalidate_concept():
void LongTermMemory::invalidate_concept(ConceptId id, bool propagate, float radius) {
    // 1. Bestehende Logik: Status → INVALIDATED, Trust → 0
    set_status(id, EpistemicStatus::INVALIDATED);
    set_trust(id, 0.0f);
    
    // 2. NEU: Trust-Propagation (Feature 2)
    if (propagate) {
        for (auto& hook : invalidation_hooks_) {
            hook(id, old_trust);  // TrustPropagator::propagate()
        }
    }
    
    // 3. NEU: Retention-Check (Feature 3)
    // Wird NACH Propagation aufgerufen, damit der volle Impact bekannt ist
    retention_manager_->on_invalidation(id);
}
```

#### 3→1: Anti-Knowledge beeinflusst Sprach-Graph

```cpp
// Wenn ein Satz als Anti-Knowledge markiert wird,
// bekommen seine Wort-Concepts ein Signal.

// RetentionManager::on_invalidation() prüft:
// Ist das invalidierte Concept ein Satz (layer=="linguistic")?
// → Die DENOTES-Targets werden NICHT invalidiert (semantische Concepts 
//   können in anderen, wahren Sätzen vorkommen)
// → Aber der Satz selbst wird als Anti-Knowledge behalten
// → Nützlich für: "Dieses Satz-Muster wurde schon als falsch erkannt"

// In SentenceParser::parse_and_store():
// VOR dem Speichern eines neuen Satzes prüfen:
ParsedSentence SentenceParser::parse_and_store(const std::string& sentence) {
    auto parsed = /* ... tokenize, create concepts ... */;
    
    // NEU: Anti-Knowledge Check (Feature 3→1)
    if (retention_manager_->resembles_known_error(parsed.sentence_id)) {
        // Satz ähnelt bekanntem Fehlpfad!
        // → Niedrigerer initialer Trust
        // → EpistemicType::SPECULATION statt HYPOTHESIS
        set_trust(parsed.sentence_id, 0.3f);  // Statt default 0.5
        set_type(parsed.sentence_id, EpistemicType::SPECULATION);
    }
    
    return parsed;
}
```

#### 1→3: Sprachliche Komplexität erhöht Retention-Score

```cpp
// ComplexityAnalyzer berücksichtigt Sprach-Graph Daten:
ComplexityMetrics ComplexityAnalyzer::analyze(ConceptId id) const {
    auto metrics = /* ... bestehende Analyse ... */;
    
    // NEU: Wenn das Concept linguistische Referenzen hat,
    // erhöht das die "involved_concepts" Metrik
    auto incoming = ltm_.get_incoming_relations(id);
    size_t linguistic_refs = std::count_if(incoming.begin(), incoming.end(),
        [](const Relation& r) { return r.type == RelationType::DENOTES; });
    
    // Mehr Sätze referenzieren dieses Concept → komplexer → eher behalten
    metrics.involved_concepts += linguistic_refs;
    
    // Re-normalize
    metrics.normalized_score = /* ... neu berechnen ... */;
    return metrics;
}
```

---

## Zusammenfassung: Neue Artefakte

### Neue Dateien

| Datei | Klasse | LOC (geschätzt) |
|-------|--------|-----------------|
| `sentence_parser.h/.cpp` | `SentenceParser` | ~250 |
| `trust_propagator.h/.cpp` | `TrustPropagator` | ~300 |
| `complexity_analyzer.h/.cpp` | `ComplexityAnalyzer` | ~200 |
| `retention_manager.h/.cpp` | `RetentionManager` | ~150 |

### Erweiterte Dateien

| Datei | Änderung |
|-------|----------|
| `relation_type.h` | +8 neue RelationTypes |
| `concept.h` | +2 Felder (`is_anti_knowledge`, `complexity_score`) |
| `long_term_memory.h/.cpp` | +6 neue Methoden, 1 erweiterte Signatur |
| `epistemic_promotion.h/.cpp` | +3 neue Methoden |
| `graph_densifier.h/.cpp` | +3 neue Methoden |
| `concept_proposal.h/.cpp` | +2 neue Methoden |

### Neue Structs

| Struct | In Feature |
|--------|-----------|
| `ParsedSentence` | 1 |
| `Token` (privat) | 1 |
| `PropagationResult` | 2 |
| `PropagationConfig` | 2 |
| `ComplexityMetrics` | 3 |
| `RetentionConfig` | 3 |
| `RetentionStats` | 3 |

### Implementierungs-Reihenfolge

```
Phase 1: Grundlagen
  ├─ RelationType enum erweitern
  ├─ Concept struct erweitern (is_anti_knowledge, complexity_score)
  └─ LongTermMemory: Invalidation-Hooks

Phase 2: Feature 3 (Retention) — Bottom-Up
  ├─ ComplexityAnalyzer
  ├─ RetentionManager (ohne Trust-Propagation Integration)
  └─ LongTermMemory GC-Erweiterung

Phase 3: Feature 2 (Trust-Propagation) — Kernstück
  ├─ TrustPropagator
  ├─ EpistemicPromotion Erweiterungen
  ├─ Integration: Propagation → Retention

Phase 4: Feature 1 (Sprach-Graph) — Aufbauend
  ├─ GraphDensifier Erweiterungen (word_concept_exists, find_duplicate_sentence)
  ├─ ConceptProposal Erweiterungen (sentence_similarity, is_paraphrase)
  ├─ SentenceParser
  └─ Integration: Anti-Knowledge Check bei Satz-Parsing

Phase 5: Cross-Feature Integration
  ├─ Sprach-Graph → Trust-Propagation (DENOTES-Traversierung)
  ├─ Linguistische Refs → Complexity Score
  └─ End-to-End Tests
```

---

## Offene Design-Fragen

1. **Rekursions-Limit bei Trust-Propagation:** Aktuell `max_hop_distance=3`. Zu niedrig für tiefe Kausalketten? Zu hoch für Performance?

2. **Anti-Knowledge Expiry:** Soll Anti-Knowledge irgendwann verfallen? Oder ist es permanent? Vorschlag: `complexity_score` decay über Zeit, re-evaluate bei GC.

3. **Sprach-Graph Granularität:** Wie tief parsen? Nur SVO-Tripel oder auch Nebensätze, Relativsätze, Passiv-Konstruktionen? Vorschlag: Start mit SVO, erweitern bei Bedarf.

4. **Trust-Propagation bei Sprach-Concepts:** Soll ein invalidierter Satz die Trust seiner Wörter beeinflussen? Vorschlag: Nein — Wörter sind Layer-2, semantische Concepts sind Layer-1. Nur DENOTES-Targets werden geprüft.

5. **Performance bei 88K Relations:** `find_candidates()` BFS mit depth=3 kann teuer werden. Vorschlag: Candidate-Set caching, oder similarity-basiertes Pruning nach Hop 1.
