# KAN-MiniLLM Hybrid Language Engine

**Datum:** 2026-02-11  
**Autor:** KI-Architekt (Subagent)  
**Status:** Design-Dokument / Machbarkeitsstudie  
**Abhängigkeit:** Phase 1 (Vec16), Phase 2 (InteractionLayer)

---

## Inhaltsverzeichnis

1. [Executive Summary](#1-executive-summary)
2. [Simulation: "Was passiert wenn Eis schmilzt?"](#2-simulation)
3. [Architektur-Design](#3-architektur-design)
4. [Tokenizer-Design](#4-tokenizer-design)
5. [Training-Strategie](#5-training-strategie)
6. [Kapazitätsanalyse](#6-kapazitätsanalyse)
7. [Limitationen](#7-limitationen)
8. [Implementierungsplan](#8-implementierungsplan)

---

## 1. Executive Summary

### Die Idee

Brain19's KAN-Nodes und MicroModel-Ensemble werden zu einer synergetischen Sprachgenerierungs-Pipeline kombiniert:

- **KAN** liefert: Logische Ketten, kausale Relationen, Schlussfolgerungs-Skelette
- **MiniLLM-Ensemble** liefert: Semantische Gewichtung, Kontextrelevanz, Disambiguierung
- **KAN-Decoder** generiert: Token-Sequenzen aus der fusionierten Repräsentation

### Kernthese

Das System generiert Sprache **aus Verständnis** statt aus statistischer Wahrscheinlichkeit. Ein Satz wie "Eis wird zu Wasser" entsteht, weil das System die kausale Kette `Eis → CAUSES(Schmelzen) → RESULTS_IN(Wasser)` traversiert hat — nicht weil "Wasser" statistisch nach "Eis schmilzt" wahrscheinlich ist.

### Realitätscheck

Mit ~500K-2M Parametern (vs. 124M bei GPT-2-Small) kann dieses System **keine flüssige Open-Domain-Sprache** generieren. Es kann aber **domänenspezifische, faktisch korrekte, kurze Antworten** erzeugen, die auf dem Wissensgraph verankert sind. Das ist der Unique Value: Jeder generierte Satz ist epistemisch nachvollziehbar.

---

## 2. Simulation: "Was passiert wenn Eis schmilzt?" <a name="2-simulation"></a>

### Schritt 1: Tokenisierung

```
Input: "Was passiert wenn Eis schmilzt?"
Tokens (BPE-8K): ["Was", " pass", "iert", " wenn", " Eis", " schm", "ilzt", "?"]
Token-IDs: [142, 3891, 567, 2204, 1847, 6023, 4112, 31]
```

### Schritt 2: KAN-Encoding (Token-IDs → VecN)

Bag-of-Embeddings mit Positional Weighting:

```
Token-Embeddings (64D Lookup → gemittelt mit Position-Decay):

  e("Eis")      = [0.82, -0.14, 0.67, 0.23, ...] × pos_weight(4/8) = 0.75
  e("schmilzt") = [0.71, 0.45, -0.32, 0.88, ...] × pos_weight(5/8) = 0.68
  e("passiert") = [0.11, 0.33, 0.52, -0.08, ...] × pos_weight(1/8) = 0.93
  e("wenn")     = [0.05, 0.02, 0.91, 0.04, ...]  × pos_weight(3/8) = 0.81
  ... (restliche Tokens mit niedrigerem Gewicht)

Bag-of-Embeddings (gewichtete Summe → L2-normiert): 
  x_bag ∈ ℝ⁶⁴ = [0.41, 0.18, 0.52, 0.33, ...]

KAN-Encoder (3 Layers, B-Spline):
  Layer 1: ℝ⁶⁴ → ℝ³² (64×32 = 2048 KANNodes)
    φ₁(x_bag) = [0.67, -0.23, 0.81, 0.12, ...] ∈ ℝ³²
    
  Layer 2: ℝ³² → ℝ¹⁶ (32×16 = 512 KANNodes)
    φ₂(φ₁) = [0.73, 0.41, -0.15, 0.88, 0.22, 0.56, -0.31, 0.64,
               0.19, -0.47, 0.83, 0.35, 0.08, -0.62, 0.44, 0.71]
    
  Layer 3: ℝ¹⁶ → ℝ¹⁶ (16×16 = 256 KANNodes, Refinement)
    q = φ₃(φ₂) = [0.69, 0.38, -0.11, 0.85, 0.25, 0.52, -0.28, 0.61,
                   0.21, -0.44, 0.79, 0.32, 0.11, -0.58, 0.41, 0.68]
```

**Query-Embedding q ∈ ℝ¹⁶** repräsentiert die Frage im MicroModel-Raum.

### Schritt 3: Wissensgraph-Traversierung (Seed Selection)

```
Label-Search in LTM:
  "Eis"       → ConceptId 247 (Label: "Eis", Def: "Gefrorenes Wasser, H₂O im festen Aggregatzustand")
  "schmilzt"  → ConceptId 891 (Label: "Schmelzen", Def: "Phasenübergang fest → flüssig")
  "passiert"  → kein Treffer (zu generisch)
  "wenn"      → kein Treffer (Funktionswort)
  
Seeds: {247, 891}
```

### Schritt 4: InteractionLayer-Propagation

```
Subgraph-Expansion (1-hop von Seeds):
  247 (Eis) → Relationen:
    IS_A(247, 503)         → 503: "Aggregatzustand_fest"    w=0.9
    CAUSES(891, 247→712)   → 891: "Schmelzen", 712: "Wasser" 
    HAS_PROPERTY(247, 156) → 156: "Temperatur < 0°C"       w=0.8
    PART_OF(247, 712)      → 712: "Wasser"                  w=0.7
    
  891 (Schmelzen) → Relationen:
    CAUSES(891, 712)       → 712: "Wasser"                  w=0.85
    CAUSES(891, 334)       → 334: "Energieaufnahme"         w=0.7
    IS_A(891, 445)         → 445: "Phasenübergang"          w=0.9
    
Subgraph: {247, 891, 503, 712, 156, 334, 445} (N=7)

Initiale Aktivierungen (MicroModel.activate(context=q)):
  a₂₄₇(0) = σ(W₂₄₇·q + b₂₄₇) = [0.72, 0.58, 0.41, 0.83, 0.29, 0.65, 0.47, 0.61,
                                     0.33, 0.52, 0.78, 0.44, 0.18, 0.56, 0.39, 0.71]
  a₈₉₁(0) = [0.68, 0.62, 0.37, 0.79, 0.34, 0.71, 0.42, 0.55, ...]
  a₇₁₂(0) = [0.55, 0.43, 0.29, 0.61, 0.22, 0.48, 0.35, 0.52, ...]
  a₅₀₃(0) = [0.31, 0.28, 0.44, 0.52, 0.19, 0.37, 0.51, 0.29, ...]
  a₃₃₄(0) = [0.42, 0.35, 0.31, 0.48, 0.27, 0.41, 0.38, 0.44, ...]
  a₁₅₆(0) = [0.38, 0.31, 0.52, 0.45, 0.21, 0.34, 0.48, 0.36, ...]
  a₄₄₅(0) = [0.44, 0.39, 0.33, 0.57, 0.25, 0.46, 0.41, 0.49, ...]

Kopplungsgewichte (normalisiert, symmetrisiert):
  ŵ(247↔891) = predict₂₄₇(e_CAUSES, a₈₉₁) = 0.78 → norm: 0.31
  ŵ(247↔712) = predict₂₄₇(e_PART_OF, a₇₁₂) = 0.65 → norm: 0.26
  ŵ(891↔712) = predict₈₉₁(e_CAUSES, a₇₁₂) = 0.82 → norm: 0.33
  ŵ(891↔334) = predict₈₉₁(e_CAUSES, a₃₃₄) = 0.58 → norm: 0.23
  ... (restliche Paare)

Context-Node (Query q als externer Bias, β=0.3):
  α_i = softmax(qᵀ · a_i) über Subgraph
  α₂₄₇ = 0.28 (Eis: hohe Relevanz)
  α₈₉₁ = 0.31 (Schmelzen: höchste Relevanz)
  α₇₁₂ = 0.18 (Wasser: mittel)
  α₃₃₄ = 0.09 (Energieaufnahme: niedrig)
  α₅₀₃ = 0.05
  α₁₅₆ = 0.04
  α₄₄₅ = 0.05

Propagation (20 Zyklen, λ=0.1):
  t=1:  δ = 0.182   E = -1.34
  t=2:  δ = 0.091   E = -1.52
  t=5:  δ = 0.023   E = -1.71
  t=10: δ = 0.003   E = -1.78
  t=14: δ = 9.2e-5  E = -1.79  ← Konvergenz (ε = 1e-4)
```

### Schritt 5: Attractor-Zustand (Fixpunkt)

```
Finale Aktivierungen a*(14):
  a*₂₄₇ (Eis)            ‖a‖ = 2.41  ← STARK (Seed + Context)
  a*₈₉₁ (Schmelzen)      ‖a‖ = 2.63  ← STÄRKSTE (Seed + Context + kausale Verstärkung)
  a*₇₁₂ (Wasser)         ‖a‖ = 2.18  ← STARK (kausales Ziel)
  a*₃₃₄ (Energieaufnahme) ‖a‖ = 1.34  ← MITTEL
  a*₄₄₅ (Phasenübergang)  ‖a‖ = 1.21  ← MITTEL
  a*₅₀₃ (Aggregat_fest)   ‖a‖ = 0.72  ← SCHWACH
  a*₁₅₆ (Temp < 0°C)     ‖a‖ = 0.58  ← SCHWACH (WTA-unterdrückt)

Attractor-Interpretation:
  Der Fixpunkt repräsentiert die KAUSALE KETTE:
    Eis(2.41) →[CAUSES]→ Schmelzen(2.63) →[CAUSES]→ Wasser(2.18)
  mit Nebeninformation:
    Energieaufnahme(1.34), Phasenübergang(1.21)
```

### Schritt 6: MiniLLM-Ensemble (Semantische Bewertung)

```
Parallel zu Schritt 4-5 laufen MiniLLM-Ensemble-Bewertungen:

MiniLLM₁ (Relevanz-Scorer):
  Für jedes aktive Konzept → Relevanz zur Query:
  rel(Eis, query)            = 0.92
  rel(Schmelzen, query)      = 0.95
  rel(Wasser, query)         = 0.88
  rel(Energieaufnahme, query)= 0.62
  rel(Phasenübergang, query) = 0.71
  
MiniLLM₂ (Kausalitäts-Scorer):
  Kausalketten-Stärke:
  causal(Eis → Schmelzen → Wasser) = 0.91
  causal(Schmelzen → Energieaufnahme) = 0.67
  
MiniLLM₃ (Antwort-Typ-Klassifikator):
  P(kausal_erklärend) = 0.78
  P(definitional)     = 0.12
  P(aufzählend)       = 0.10
  → Template: KAUSAL_ERKLÄREND
```

### Schritt 7: FusionLayer (Logik + Semantik → Fused Representation)

```
Gating-Mechanismus (lernbare Gewichte α_gate):

Für jedes Konzept i im Subgraph:
  gate_i = σ(w_gate · [a*_i; rel_i; causal_i] + b_gate)
  
  gate(Eis)            = σ(w·[2.41; 0.92; 0.91]) = 0.89
  gate(Schmelzen)      = σ(w·[2.63; 0.95; 0.91]) = 0.93
  gate(Wasser)         = σ(w·[2.18; 0.88; 0.91]) = 0.87
  gate(Energieaufnahme)= σ(w·[1.34; 0.62; 0.67]) = 0.61
  gate(Phasenübergang) = σ(w·[1.21; 0.71; 0.67]) = 0.58

Fused Concept Scores (sortiert):
  1. Schmelzen       0.93 × 2.63 = 2.45
  2. Eis             0.89 × 2.41 = 2.14
  3. Wasser          0.87 × 2.18 = 1.90
  4. Energieaufnahme 0.61 × 1.34 = 0.82
  5. Phasenübergang  0.58 × 1.21 = 0.70

Geordnete Konzeptkette (kausale Reihenfolge via Relations):
  [Eis] →CAUSES→ [Schmelzen] →CAUSES→ [Wasser]
  (+ optional: [Energieaufnahme], [Phasenübergang])
  
Antwort-Template: KAUSAL_ERKLÄREND
  Skeleton: "<SUBJEKT> <VERB> <ERGEBNIS>"
  Mapped:   "Eis" "schmilzt zu" "Wasser"
```

### Schritt 8: KAN-Decoder (Fused Representation → Token-Sequenz)

```
Decoder-Input: 
  Fused-Vector f = concat(a*₂₄₇, a*₈₉₁, a*₇₁₂, gate_scores, template_id)
  f ∈ ℝ⁶⁴ (3×16 Aktivierungen + 5 Gate-Scores + 1 Template + 10 Padding)

KAN-Decoder (autoregressive Token-Generierung):

Step 0: Decoder-State s₀ = KANDecoder.init(f)
  KAN Layer 1: ℝ⁶⁴ → ℝ³²
  KAN Layer 2: ℝ³² → ℝ¹⁶ (hidden state h₀)

Step 1: h₀ → Token-Logits über Vocab (8K)
  Projection: h₀ · E_vocab^T → logits ∈ ℝ⁸¹⁹²
  Top-5 nach Softmax:
    "Wenn"  : 0.34
    "Eis"   : 0.28
    "Das"   : 0.15
    "Bei"   : 0.08
    "Es"    : 0.06
  → Greedy: "Wenn" (aber Template-Constraint bevorzugt Subjekt-Start)
  → Constrained: "Eis" (P=0.28, Template: SUBJEKT zuerst)

Step 2: h₁ = KAN_update(h₀, embed("Eis"))
  Top-5:
    "schmilzt" : 0.41  ← STARK (kausale Kette + Verb-Slot im Template)
    "wird"     : 0.22
    "ist"      : 0.11
    "verwandelt": 0.08
    "..."      : ...
  → Output: "schmilzt"

Step 3: h₂ = KAN_update(h₁, embed("schmilzt"))
  Top-5:
    ","     : 0.29
    "zu"    : 0.25
    "und"   : 0.14
    "."     : 0.12
    "es"    : 0.07
  → Output: ","

Step 4-8: Weitere Token...
  "," → "wird" → "es" → "zu" → "Wasser" → "."
  
  "Wasser" erhält hohen Score weil:
    - a*₇₁₂(Wasser) = 2.18 (starke Aktivierung im Attractor)
    - Kausalkette: Schmelzen →CAUSES→ Wasser
    - Embedding-Nähe: embed("Wasser") · h_t = 0.87 (hohe Cosine-Similarity)
```

### Generierte Antwort

```
"Eis schmilzt, wird es zu Wasser."

Alternative (mit Elaboration, wenn max_tokens > 10):
"Wenn Eis schmilzt, wird es zu Wasser. Dabei nimmt es Energie auf."
```

### Zusammenfassung des Datenflusses

```
"Was passiert wenn Eis schmilzt?"
         │
    ┌────┴────┐
    │ Tokenize │ → 8 Tokens
    └────┬────┘
         │
    ┌────┴────────┐
    │ KAN-Encoder  │ → q ∈ ℝ¹⁶ (Query-Embedding)
    └────┬────────┘
         │
    ┌────┴──────────────┐        ┌─────────────────────┐
    │ LTM Label-Search   │        │ MiniLLM-Ensemble     │
    │ Seeds: {247, 891}  │        │ (parallel)           │
    └────┬──────────────┘        │ Relevanz: [0.92,0.95]│
         │                        │ Kausalität: 0.91     │
    ┌────┴───────────────┐       │ Template: KAUSAL     │
    │ InteractionLayer    │       └─────────┬───────────┘
    │ 7 Nodes, 14 Zyklen │                  │
    │ Attractor:          │                  │
    │  Schmelzen(2.63)    │                  │
    │  Eis(2.41)          │                  │
    │  Wasser(2.18)       │                  │
    └────┬───────────────┘                  │
         │                                   │
    ┌────┴───────────────────────────────────┴──┐
    │ FusionLayer (Gated Merge)                  │
    │ Kausalkette: Eis → Schmelzen → Wasser      │
    │ Gate-Scores: [0.93, 0.89, 0.87, 0.61, 0.58]│
    └────┬──────────────────────────────────────┘
         │
    ┌────┴────────┐
    │ KAN-Decoder  │ → "Eis schmilzt, wird es zu Wasser."
    │ 8 Steps      │
    │ Autoregress. │
    └─────────────┘
```

---

## 3. Architektur-Design <a name="3-architektur-design"></a>

### 3.1 Übersicht

```
┌──────────────────────────────────────────────────────────────────┐
│                  KAN-MiniLLM HYBRID LANGUAGE ENGINE              │
│                                                                  │
│  ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐  │
│  │ KAN     │   │Reasoning │   │ Semantic  │   │ KAN       │  │
│  │ Encoder │──►│ Layer    │──►│ Fusion    │──►│ Decoder   │  │
│  │         │   │          │   │ Layer     │   │           │  │
│  └─────────┘   └──────────┘   └───────────┘   └────────────┘  │
│       ▲              ▲              ▲                           │
│       │              │              │                           │
│  ┌─────────┐   ┌──────────┐   ┌───────────┐                   │
│  │Tokenizer│   │   LTM    │   │ MiniLLM   │                   │
│  │ (BPE-8K)│   │(~1K nodes)│  │ Ensemble  │                   │
│  └─────────┘   └──────────┘   └───────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 KANEncoder

**Funktion:** Text → Query-Embedding q ∈ ℝ¹⁶

```
Input:  std::string text
Output: VecN (= std::array<double, 16>)

Pipeline:
  1. Tokenize(text) → token_ids ∈ ℕ^L  (L ≤ 128)
  2. Lookup: token_ids → embeddings ∈ ℝ^(L×64) 
  3. Bag-of-Embeddings mit Positional Decay: x_bag ∈ ℝ⁶⁴
  4. KAN Layer 1: ℝ⁶⁴ → ℝ³²  (64×32 = 2048 KANNodes à 10 Knoten)
  5. KAN Layer 2: ℝ³² → ℝ¹⁶  (32×16 = 512 KANNodes à 10 Knoten)
  6. KAN Layer 3: ℝ¹⁶ → ℝ¹⁶  (16×16 = 256 KANNodes à 10 Knoten)
```

| Komponente | Input-Shape | Output-Shape | Parameter |
|:-----------|:------------|:-------------|:----------|
| Token-Embedding-Table | token_id (scalar) | ℝ⁶⁴ | 8192 × 64 = 524,288 |
| Positional Decay | ℝ^(L×64) | ℝ⁶⁴ | 0 (hardcoded) |
| KAN Layer 1 | ℝ⁶⁴ | ℝ³² | 2048 × 10 = 20,480 |
| KAN Layer 2 | ℝ³² | ℝ¹⁶ | 512 × 10 = 5,120 |
| KAN Layer 3 | ℝ¹⁶ | ℝ¹⁶ | 256 × 10 = 2,560 |
| **Encoder Total** | | | **552,448** |

### 3.3 ReasoningLayer

**Funktion:** Query-Embedding + Seeds → Attractor-Aktivierungen

```
Input:  VecN query, std::vector<ConceptId> seeds
Output: std::unordered_map<ConceptId, VecN> activations, 
        std::vector<ConceptId> causal_chain

Pipeline:
  1. Subgraph-Expansion: seeds + 1-hop → N Knoten (≤ 100)
  2. InteractionLayer::propagate(seeds, ltm, registry, embeddings, &query)
  3. Causal Chain Extraction: 
     Sortiere Konzepte nach ‖a*‖ → Extrahiere CAUSES-Ketten
  4. Return: {activations, causal_chain, converged, cycles}
```

| Komponente | Input-Shape | Output-Shape | Parameter |
|:-----------|:------------|:-------------|:----------|
| MicroModels (N aktive) | VecN × VecN | scalar ∈ (0,1) | N × (16² + 16) = N × 272 |
| InteractionLayer | N × VecN | N × VecN | 0 (nutzt MicroModel-Params) |
| Causal Chain Extract | N × VecN | ordered list | 0 (algorithmisch) |
| **Reasoning (1000 Models)** | | | **272,000 (MicroModels)** |

### 3.4 SemanticLayer (MiniLLM-Ensemble)

**Funktion:** Parallele semantische Bewertung durch spezialisierte Scorer

```
Input:  std::vector<ConceptId> active_concepts, VecN query, LTM& ltm
Output: SemanticScores {relevance[], causality[], template_id}

Drei spezialisierte Scorer (jeweils ein kleines KAN-Modul):

Scorer 1: RelevanzScorer
  Input: concat(a_i, query) ∈ ℝ³² → KAN(32→16→1) → relevance_i ∈ (0,1)
  
Scorer 2: KausalitätsScorer  
  Input: concat(a_i, a_j, e_rel) ∈ ℝ⁴⁸ → KAN(48→16→1) → causal_strength ∈ (0,1)
  
Scorer 3: TemplateKlassifikator
  Input: bag(a_active) ∈ ℝ¹⁶ → KAN(16→8→4) → softmax → template_probs
  Templates: {KAUSAL_ERKLÄREND, DEFINITIONAL, AUFZÄHLEND, VERGLEICHEND}
```

| Komponente | Input-Shape | Output-Shape | Parameter |
|:-----------|:------------|:-------------|:----------|
| RelevanzScorer KAN | ℝ³² | ℝ¹ | (32×16 + 16×1) × 10 = 5,280 |
| KausalitätsScorer KAN | ℝ⁴⁸ | ℝ¹ | (48×16 + 16×1) × 10 = 7,840 |
| TemplateKlassifikator KAN | ℝ¹⁶ | ℝ⁴ | (16×8 + 8×4) × 10 = 1,600 |
| **Semantic Total** | | | **14,720** |

### 3.5 FusionLayer

**Funktion:** Logik-Aktivierungen + Semantik-Scores → Geordnete Fused Representation

```
Input:  activations: Map<ConceptId, VecN>, 
        semantic: SemanticScores
Output: FusedRepresentation {ordered_concepts[], fused_vector ∈ ℝ⁶⁴}

Algorithmus: Gated Fusion
  Für jedes Konzept i:
    gate_i = σ(W_gate · [‖a*_i‖; rel_i; causal_i] + b_gate)
    score_i = gate_i × ‖a*_i‖
    
  Sortierung: Nach kausaler Reihenfolge (CAUSES-Kette), bei Gleichrang nach score_i
  
  Fused Vector: 
    f = concat(top-3 a*_i gewichtet, gate_scores, template_one_hot)
    f ∈ ℝ⁶⁴
```

| Komponente | Input-Shape | Output-Shape | Parameter |
|:-----------|:------------|:-------------|:----------|
| Gate-Weights | ℝ³ | ℝ¹ | 3 + 1 = 4 |
| Fusion-Projection | 3×ℝ¹⁶ + ℝ⁵ + ℝ⁴ | ℝ⁶⁴ | 57 × 64 = 3,648 |
| **Fusion Total** | | | **3,652** |

### 3.6 KANDecoder

**Funktion:** Fused Representation → Token-Sequenz (autoregressive Generierung)

```
Input:  FusedRepresentation f ∈ ℝ⁶⁴
Output: std::string (max 50 Tokens)

Pipeline (pro Token-Step):
  1. Init: h₀ = KAN_init(f):  ℝ⁶⁴ →[KAN Layer]→ ℝ¹⁶
  2. Loop (t = 0, 1, ...):
     a. Logits = h_t · E_vocab^T ∈ ℝ⁸¹⁹²
     b. Constrained Sampling (Template + aktive Konzepte boosten)
     c. token_t = argmax(logits_constrained)
     d. h_{t+1} = KAN_update(h_t, embed(token_t)):
        concat(h_t, embed(token_t)) ∈ ℝ⁸⁰ → KAN(80→32→16) → h_{t+1}
  3. Stop wenn: token_t == "." oder t >= max_tokens
```

| Komponente | Input-Shape | Output-Shape | Parameter |
|:-----------|:------------|:-------------|:----------|
| KAN Init Layer | ℝ⁶⁴ | ℝ¹⁶ | 64×16 × 10 = 10,240 |
| KAN Update Layer 1 | ℝ⁸⁰ | ℝ³² | 80×32 × 10 = 25,600 |
| KAN Update Layer 2 | ℝ³² | ℝ¹⁶ | 32×16 × 10 = 5,120 |
| Output Projection | ℝ¹⁶ | ℝ⁸¹⁹² | 16 × 8192 = 131,072 |
| **Decoder Total** | | | **172,032** |

### 3.7 Gesamt-Parameterübersicht

| Komponente | Parameter | Anteil |
|:-----------|:----------|:-------|
| Token-Embedding-Table | 524,288 | 51.7% |
| KAN-Encoder (3 Layers) | 28,160 | 2.8% |
| MicroModels (1000 × 272) | 272,000 | 26.8% |
| Semantic Scorer (3 KANs) | 14,720 | 1.5% |
| FusionLayer | 3,652 | 0.4% |
| KAN-Decoder + Projection | 172,032 | 17.0% |
| **GESAMT** | **1,014,852** | **100%** |

**~1M Parameter.** Davon sind 52% die Token-Embedding-Tabelle (die geteilt wird zwischen Encoder und Decoder).

---

## 4. Tokenizer-Design <a name="4-tokenizer-design"></a>

### 4.1 Design-Entscheidung: BPE mit 8K Vocabulary

| Option | Vocab | Pro | Contra |
|:-------|:------|:----|:-------|
| Character-Level | ~256 | Kein OOV | Sehr lange Sequenzen, schwer zu lernen |
| BPE-8K | 8,192 | Guter Kompromiss | Muss trainiert werden |
| BPE-16K | 16,384 | Bessere Abdeckung | Größere Embedding-Table (+512K Params) |
| WordPiece | ~8K | Ähnlich BPE | Etwas komplexer, kein Vorteil hier |

**Entscheidung: BPE-8K** (Byte-Pair-Encoding mit 8192 Tokens)

### 4.2 Trainingskorpus

```
Primär (Brain19-intern):
  - ~1000 Konzeptlabels
  - ~1000 Konzeptdefinitionen (je 10-50 Wörter)
  - ~1500 Relationstyp-Labels
  → ~25,000 Tokens Rohtext

Sekundär (extern, deutsch):
  - Wikipedia DE Dumps (Subset: Naturwissenschaften, ~10MB)
  - Deutsche Wortliste (50K häufigste Wörter)
  → ~5M Tokens Rohtext

BPE-Training: 
  - 256 Basis-Bytes + iteratives Merge der häufigsten Paare
  - 50 Merge-Iterationen → ~8K Vocabulary
  - Spezialtoken: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3, [SEP]=4
```

### 4.3 Wissensgraph-Verknüpfung

Jedes Konzept im KG erhält einen **Konzept-Token**:

```
Token 5-1004: Konzeptlabels (1000 Konzepte)
  Token 5    → ConceptId 0 ("Photosynthese")
  Token 6    → ConceptId 1 ("Wasser")
  ...
  Token 1004 → ConceptId 999 ("Evolution")

Token 1005-8191: Normale BPE-Tokens
  Token 1005 → "der"
  Token 1006 → "die"
  ...
```

**Vorteil:** Wenn der Decoder "Wasser" generieren will, hat er zwei Pfade:
1. Normal: BPE-Token-Sequenz ["Was", "ser"] 
2. Konzept: Einzel-Token [6] direkt aus dem Wissensgraph

Konzept-Tokens werden im Training bevorzugt (höheres Gewicht im Loss), weil sie epistemisch verankert sind.

### 4.4 Tokenizer-Implementierung

```cpp
class BPETokenizer {
public:
    explicit BPETokenizer(const std::string& vocab_path);
    
    std::vector<uint16_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint16_t>& tokens) const;
    
    // Train from corpus
    static BPETokenizer train(
        const std::vector<std::string>& corpus,
        const LongTermMemory& ltm,  // für Konzept-Tokens
        size_t vocab_size = 8192
    );
    
    // Konzept-Token Mapping
    std::optional<ConceptId> token_to_concept(uint16_t token) const;
    std::optional<uint16_t> concept_to_token(ConceptId cid) const;
    
    size_t vocab_size() const { return vocab_size_; }
    
    void save(const std::string& path) const;
    
private:
    size_t vocab_size_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, uint16_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<uint16_t, ConceptId> token_concept_map_;
    std::unordered_map<ConceptId, uint16_t> concept_token_map_;
};
```

**LoC: ~250**

---

## 5. Training-Strategie <a name="5-training-strategie"></a>

### 5.1 Multi-Stage-Training (NICHT End-to-End)

End-to-End Backprop durch das gesamte System ist **nicht machbar**, weil:
1. Die InteractionLayer ist ein iterativer Prozess (14+ Zyklen) — kein differenzierbarer Forward-Pass
2. MicroModels haben unabhängige Adam-Optimizer
3. Die Wissensgraph-Traversierung ist diskret (nicht differenzierbar)

**Stattdessen: Mehrstufiges Training mit gefrorenen Komponenten.**

### 5.2 Stage 1: Basis-Training (isoliert)

**MicroModels** (bereits implementiert):
```
Daten: KG-Relationen (positive/negative Samples)
Loss: MSE(predict(e,c), target)
Optimizer: Adam (lr=0.01, 100 Epochs)
→ Bereits funktional in micro_trainer.cpp
```

**KAN-Encoder** (neu):
```
Daten: (Text, Konzept-Embedding) Paare aus LTM
  - "Wasser" → embed(ConceptId=712)
  - "Photosynthese umwandeln Licht" → embed(ConceptId=42)
  
Loss: MSE(KANEncoder(text), target_embedding)
  target_embedding = MicroModel.e_init für das Konzept
  
Trainingssamples: ~1000 (je Konzept: Label + Definition)
Optimizer: Adam auf KAN-Koeffizienten (lr=0.001, 200 Epochs)
```

**KAN-Decoder** (neu):
```
Daten: (Konzept-Embedding, Token-Sequenz) Paare
  - embed(712) → "Wasser"  
  - embed(247) → "Eis, gefrorenes Wasser"
  
Loss: Cross-Entropy(predicted_token, target_token)
  Summiert über alle Token in der Sequenz
  
Trainingssamples: ~1000 × avg_len(5) = ~5000 Token-Steps
Optimizer: Adam auf KAN-Koeffizienten + Output-Projection (lr=0.001)
```

### 5.3 Stage 2: Fusion-Training (MicroModels eingefroren)

```
Daten: (Query, erwartete Kausalkette, erwartete Antwort)
  - "Was ist Photosynthese?" → [Photosynthese] → "Photosynthese ist ein Prozess..."
  - "Was passiert wenn Eis schmilzt?" → [Eis, Schmelzen, Wasser] → "Eis wird zu Wasser"

Trainierte Komponenten: 
  ✅ Semantic Scorer (3 KANs)
  ✅ FusionLayer (Gate + Projection)
  ✅ KAN-Decoder (Fine-Tuning)
  ❄️ MicroModels (eingefroren)
  ❄️ KAN-Encoder (eingefroren)
  ❄️ InteractionLayer (nicht trainierbar, Hyperparameter fest)

Loss:
  L_total = α·L_token + β·L_chain + γ·L_concept

  L_token = CrossEntropy(decoder_output, target_tokens)  [α=1.0]
  L_chain = MSE(predicted_chain_order, target_chain_order) [β=0.3]
  L_concept = -log(P(richtige_Konzepte_aktiviert))        [γ=0.5]
  
Trainingssamples: ~500 QA-Paare (manuell kuratiert)
Curriculum:
  Stufe 1 (Epoch 1-50):   Definitional ("Was ist X?") 
  Stufe 2 (Epoch 51-150):  Kausal ("Was passiert wenn X?")
  Stufe 3 (Epoch 151-300): Komplex ("Warum ist X wichtig für Y?")
```

### 5.4 Stage 3: Joint Fine-Tuning (optional)

```
Nachdem Stage 2 konvergiert:
  - MicroModels werden mit 1/10 der Learning-Rate mittrainiert
  - Aber: Epistemic Integrity Guard verhindert Trust-Korrumption
  
Guard-Bedingung:
  Für jedes MicroModel m_i:
    δ_i = ‖W_i(after) - W_i(before)‖_F / ‖W_i(before)‖_F
    ASSERT(δ_i < 0.1)  // Max 10% relative Änderung
    
  Falls verletzt → Rollback dieses MicroModels auf Pre-Training-Zustand
```

### 5.5 Epistemische Integritäts-Sicherung

```
Invarianten die NIEMALS verletzt werden dürfen:

1. MicroModel predict(e,c) ∈ (0,1) ← sigmoid garantiert
2. Relationen mit weight > 0.8 müssen predict > 0.5 behalten
3. CONTRADICTS-Relationen müssen predict < 0.3 behalten
4. Trust-Scores in LTM werden durch Training NICHT verändert
5. FACT-Status-Konzepte sind unveränderlich

Monitoring:
  Nach jedem Training-Batch:
    - Spot-Check: 10 zufällige Relationen → predict-Scores prüfen
    - Wenn >2 Verletzungen → Training-Abbruch + Alarm
```

---

## 6. Kapazitätsanalyse <a name="6-kapazitätsanalyse"></a>

### 6.1 Parameter-Count

| Modell | Parameter | RAM (FP64) | RAM (FP32) |
|:-------|:----------|:-----------|:-----------|
| **Brain19 Hybrid** | **~1.01M** | **~8.1 MB** | **~4.1 MB** |
| GPT-2 Small | 124M | — | ~496 MB |
| TinyLlama 1.1B | 1,100M | — | ~4.4 GB |
| GPT-2 Medium | 355M | — | ~1.4 GB |
| DistilGPT-2 | 82M | — | ~328 MB |

**Brain19 ist ~120× kleiner als GPT-2-Small.**

### 6.2 Kapazitätsvergleich: Was können die Modelle?

| Fähigkeit | Brain19 Hybrid | GPT-2-Small | TinyLlama |
|:----------|:-:|:-:|:-:|
| Faktisch korrekte Domänen-Antworten | ✅ | ⚠️ | ⚠️ |
| Epistemisch nachvollziehbar | ✅ | ❌ | ❌ |
| Flüssiges Deutsch | ❌ | ⚠️ | ✅ |
| Open-Domain Fragen | ❌ | ⚠️ | ✅ |
| Kausale Schlüsse | ✅ | ❌ | ⚠️ |
| Kreatives Schreiben | ❌ | ⚠️ | ⚠️ |
| Zusammenhängende Absätze | ❌ | ✅ | ✅ |
| Latenz (Token/s) | ~1000 | ~50 | ~20 |

### 6.3 Minimale Modellgröße für verständliche deutsche Sätze

Aus der Literatur (Character-Level Language Models):

| Parameter | Qualitätsstufe |
|:----------|:---------------|
| <100K | Unverständlich, Buchstabensalat |
| 100K-500K | Einzelne erkennbare Wörter, keine Grammatik |
| 500K-2M | Kurze grammatische Fragmente, eingeschränktes Vocab |
| 2M-10M | Einfache Sätze, limitiertes Themenspektrum |
| 10M-50M | Zusammenhängende kurze Texte |
| >100M | Flüssige Sprache |

**Brain19 mit ~1M liegt im Bereich "kurze grammatische Fragmente".** 

**ABER:** Brain19 betrügt auf clevere Weise:
- Das Wissensgraph-Template erzwingt grammatische Struktur
- Konzept-Tokens garantieren korrekte Fachbegriffe
- Kausalketten liefern die logische Struktur

→ Effektive Qualität liegt höher als reine Parameterzahl suggeriert: **~2-5M äquivalent** für den eingeschränkten Domänenbereich.

### 6.4 Skalierungsszenarien

| Szenario | Vocab | Konzepte | Decoder-Hidden | Gesamt-Params | Qualität |
|:---------|:------|:---------|:---------------|:--------------|:---------|
| Minimal | 4K | 500 | 8 | ~320K | Telegramm-Stil |
| **Standard** | **8K** | **1000** | **16** | **~1M** | **Kurze korrekte Sätze** |
| Erweitert | 16K | 2000 | 32 | ~3.5M | Einfache Erklärungen |
| Maximal | 32K | 5000 | 64 | ~15M | Zusammenhängende Absätze |

---

## 7. Limitationen <a name="7-limitationen"></a>

### 7.1 Was das System NICHT kann

1. **Keine Open-Domain-Sprache**: Kann nur über Konzepte im Wissensgraph reden. Frage nach etwas nicht im KG → "Ich weiß nicht" oder Halluzination.

2. **Keine langen Texte**: Maximale Ausgabelänge ~20-30 Token sinnvoll. Darüber degeneriert die Qualität rapide, weil der KAN-Decoder keinen echten Attention-Mechanismus hat (kein Key-Value Cache, kein Multi-Head Attention).

3. **Keine stilistische Vielfalt**: Immer der gleiche Template-basierte Stil. Kann nicht zwischen formell/informell, wissenschaftlich/umgangssprachlich wechseln.

4. **Keine echte Pragmatik**: Versteht nicht "warum fragst du?" oder implizite Bedeutung. Rein propositional.

5. **Kein Deutsch-Grammatik-Verständnis**: Kasus, Genus, Konjugation werden vom Decoder "geraten" basierend auf Token-Statistik, nicht auf grammatischem Wissen. Fehler sind wahrscheinlich.

6. **Keine Negation/Konditionale zuverlässig**: "Was passiert wenn Eis NICHT schmilzt?" wird wahrscheinlich wie die positive Version beantwortet.

### 7.2 Die Grenze zwischen "versteht" und "klingt natürlich"

```
Brain19 VERSTEHT:                   Brain19 klingt NICHT natürlich:
┌─────────────────────────┐        ┌──────────────────────────────┐
│ Kausale Relationen       │        │ Grammatische Korrektheit     │
│ Konzept-Hierarchien      │        │ Stilistischer Fluss          │
│ Widersprüche im KG       │        │ Pronomen-Referenzen          │
│ Epistemische Unsicherheit │        │ Nebensätze, Relativsätze     │
│ Was es NICHT weiß         │        │ Idiome, Metaphern            │
│ Analogie-Strukturen       │        │ Kontextabhängige Bedeutung   │
└─────────────────────────┘        └──────────────────────────────┘
```

**Brain19 generiert Sätze die inhaltlich korrekt aber stilistisch hölzern sind.**

Beispiel:
- GPT-2: "Wenn Eis schmilzt, verwandelt es sich in flüssiges Wasser. Dabei wird Energie aus der Umgebung aufgenommen, die die Wassermoleküle in Bewegung versetzt."
- Brain19: "Eis schmilzt, wird zu Wasser. Braucht Energie."

### 7.3 Komplexitätsgrenzen

| Frage-Typ | Erwartete Qualität | Beispiel |
|:-----------|:-------------------|:---------|
| Definitional ("Was ist X?") | ✅ Gut | "Photosynthese ist Umwandlung von Licht zu Energie in Pflanzen." |
| Einfach kausal ("Was passiert wenn X?") | ✅ Gut | "Eis schmilzt, wird zu Wasser." |
| Eigenschaft ("Welche Farbe hat X?") | ✅ Gut | "Blut ist rot." |
| Multi-hop kausal ("Warum führt X zu Z?") | ⚠️ Akzeptabel | "X verursacht Y. Y verursacht Z." (Kette, kein zusammenhängender Satz) |
| Vergleich ("Unterschied X und Y?") | ⚠️ Schwach | Listet Eigenschaften, vergleicht nicht elegant |
| Hypothetisch ("Was wäre wenn X nicht existierte?") | ❌ Versagt | Kontrafaktisches Reasoning nicht modelliert |
| Meta ("Warum denkst du das?") | ❌ Versagt | Keine Selbstreflexion |
| Open-Domain ("Erzähl mir einen Witz") | ❌ Versagt | Nicht im KG |

### 7.4 Risiken

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|:-------|:-------------------|:-------|:-----------|
| KAN-Decoder generiert Müll | Hoch (30%) | Hoch | Fallback auf Template-basierte Generierung |
| Training korumpiert MicroModels | Mittel (15%) | Kritisch | Epistemischer Integrity Guard (§5.5) |
| 8K Vocab zu klein für Deutsch | Mittel (20%) | Mittel | Upgrade auf 16K möglich |
| InteractionLayer konvergiert nicht | Niedrig (5%) | Hoch | Gewichtsnormalisierung (bereits im Plan) |
| Training-Daten zu wenig | Hoch (40%) | Hoch | Synthetische Daten-Augmentation aus KG |

---

## 8. Implementierungsplan <a name="8-implementierungsplan"></a>

### 8.1 Neue Dateien

```
backend/hybrid/
├── kan_language_engine.hpp    # Hauptklasse: Orchestriert alles
├── kan_language_engine.cpp    # Implementation
├── kan_encoder.hpp            # Text → VecN
├── kan_encoder.cpp            # Implementation
├── kan_decoder.hpp            # VecN → Text
├── kan_decoder.cpp            # Implementation  
├── tokenizer.hpp              # BPE-8K Tokenizer
├── tokenizer.cpp              # Implementation
├── semantic_scorer.hpp        # MiniLLM-Ensemble (3 KAN-Scorer)
├── semantic_scorer.cpp        # Implementation
├── fusion_layer.hpp           # Gated Fusion
├── fusion_layer.cpp           # Implementation
├── language_training.hpp      # Multi-Stage Training Pipeline
├── language_training.cpp      # Implementation
└── language_config.hpp        # Konfiguration & Konstanten

tests/
├── test_tokenizer.cpp         # BPE Roundtrip, Konzept-Token-Mapping
├── test_kan_encoder.cpp       # Encoding-Konsistenz
├── test_kan_decoder.cpp       # Decoding-Qualität
├── test_language_engine.cpp   # End-to-End Test
└── test_semantic_scorer.cpp   # Scorer-Kalibrierung
```

### 8.2 Hauptklasse

```cpp
#pragma once

#include "kan_encoder.hpp"
#include "kan_decoder.hpp"
#include "tokenizer.hpp"
#include "semantic_scorer.hpp"
#include "fusion_layer.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../interaction/interaction_layer.hpp"
#include "../ltm/long_term_memory.hpp"
#include "language_config.hpp"

namespace brain19 {

struct LanguageResult {
    std::string text;                              // Generierter Text
    std::vector<ConceptId> activated_concepts;     // Welche Konzepte aktiv waren
    std::vector<ConceptId> causal_chain;           // Kausalkette
    double confidence;                             // Gesamt-Konfidenz
    size_t interaction_cycles;                     // InteractionLayer-Zyklen
    bool used_template;                            // Template-Fallback?
    std::string template_type;                     // KAUSAL / DEFINITIONAL / etc.
};

class KANLanguageEngine {
public:
    KANLanguageEngine(
        const LanguageConfig& config,
        LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        InteractionLayer& interaction
    );

    // Hauptfunktion: Frage → Antwort
    LanguageResult generate(const std::string& query, size_t max_tokens = 30) const;

    // Einzelne Phasen exponiert (für Testing/Debugging)
    VecN encode(const std::string& text) const;
    std::vector<ConceptId> find_seeds(const std::string& text) const;
    InteractionResult reason(const std::vector<ConceptId>& seeds, const VecN& query) const;
    SemanticScores score_semantics(
        const InteractionResult& reasoning, const VecN& query) const;
    FusedRepresentation fuse(
        const InteractionResult& reasoning, const SemanticScores& semantics) const;
    std::string decode(const FusedRepresentation& fused, size_t max_tokens) const;

    // Training
    void train_encoder(const std::vector<std::pair<std::string, VecN>>& pairs);
    void train_decoder(const std::vector<std::pair<VecN, std::string>>& pairs);
    void train_fusion(const std::vector<TrainingExample>& examples);
    
    // Tokenizer
    const BPETokenizer& tokenizer() const { return tokenizer_; }
    
    // Persistence
    void save(const std::string& dir) const;
    void load(const std::string& dir);

private:
    LanguageConfig config_;
    BPETokenizer tokenizer_;
    KANEncoder encoder_;
    KANDecoder decoder_;
    SemanticScorer semantic_scorer_;
    FusionLayer fusion_;
    
    // References to Brain19 core (nicht owned)
    LongTermMemory& ltm_;
    MicroModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    InteractionLayer& interaction_;
    
    // Template-basierter Fallback
    std::string template_generate(
        const std::vector<ConceptId>& chain,
        const std::string& template_type) const;
};

} // namespace brain19
```

### 8.3 Lines of Code Schätzung

| Datei | LoC | Komplexität |
|:------|:----|:------------|
| kan_language_engine.hpp | 80 | Niedrig |
| kan_language_engine.cpp | 350 | Hoch |
| kan_encoder.hpp | 50 | Niedrig |
| kan_encoder.cpp | 200 | Mittel |
| kan_decoder.hpp | 50 | Niedrig |
| kan_decoder.cpp | 250 | Hoch |
| tokenizer.hpp | 60 | Niedrig |
| tokenizer.cpp | 300 | Hoch |
| semantic_scorer.hpp | 40 | Niedrig |
| semantic_scorer.cpp | 150 | Mittel |
| fusion_layer.hpp | 30 | Niedrig |
| fusion_layer.cpp | 100 | Mittel |
| language_training.hpp | 40 | Niedrig |
| language_training.cpp | 200 | Hoch |
| language_config.hpp | 40 | Niedrig |
| test_tokenizer.cpp | 150 | Mittel |
| test_kan_encoder.cpp | 100 | Mittel |
| test_kan_decoder.cpp | 150 | Mittel |
| test_language_engine.cpp | 200 | Hoch |
| test_semantic_scorer.cpp | 100 | Mittel |
| **GESAMT** | **~2,640** | |

### 8.4 Abhängigkeiten

```
Phase 1 (Vec16) ──────────────────┐
                                    ▼
Phase 2 (InteractionLayer) ──► Phase 5 (Language Engine)
                                    ▲
Phase 0 (Validation Fix) ─────────┘

Intern:
  tokenizer ← (keine Abhängigkeit)
  kan_encoder ← KANModule, tokenizer
  semantic_scorer ← KANModule
  fusion_layer ← (keine externe Abhängigkeit)
  kan_decoder ← KANModule, tokenizer
  kan_language_engine ← ALLES
  language_training ← kan_language_engine, LTM
```

### 8.5 Zeitschätzung

| Aufgabe | Tage | Abhängigkeit |
|:--------|:-----|:-------------|
| Tokenizer (BPE) | 2 | — |
| KAN-Encoder | 2 | Tokenizer |
| Semantic Scorer | 1.5 | KANModule |
| FusionLayer | 1 | — |
| KAN-Decoder | 3 | Tokenizer, KANModule |
| Language Engine (Integration) | 2 | Alles oben |
| Training Pipeline | 2 | Language Engine |
| Tests | 2 | Language Engine |
| Template-Fallback | 1 | Language Engine |
| **GESAMT** | **~16.5 Tage** | |

**Mit Phase-1+2 Abhängigkeit: ~4.5 Wochen gesamt** (davon ~2.5 Wochen für Phase 5 allein).

### 8.6 Meilensteine

| Woche | Meilenstein | Akzeptanzkriterium |
|:------|:------------|:-------------------|
| W1 | Tokenizer + Encoder fertig | tokenize(detokenize(text)) == text; encode gibt konsistentes VecN |
| W2 | Decoder + Scorer fertig | Decoder erzeugt Token-Sequenzen aus VecN; Scorer produziert Scores |
| W3 | Integration + Training | End-to-End generate() funktioniert mit Template-Fallback |
| W4 | Tests + Fine-Tuning | 10 Testfragen korrekt beantwortet; Kein MicroModel korrumpiert |

### 8.7 Empfohlene Implementierungsreihenfolge

```
1. tokenizer.hpp/cpp              ← Basis, keine Abhängigkeit
2. language_config.hpp            ← Konstanten
3. kan_encoder.hpp/cpp            ← Braucht Tokenizer + KANModule
4. semantic_scorer.hpp/cpp        ← Braucht KANModule
5. fusion_layer.hpp/cpp           ← Einfach, keine externe Dep
6. kan_decoder.hpp/cpp            ← Braucht Tokenizer + KANModule
7. kan_language_engine.hpp/cpp    ← Integration aller Teile
8. language_training.hpp/cpp      ← Braucht Engine + LTM-Daten
9. Tests                          ← Parallel zu 7-8
```

---

## Anhang A: Mathematische Details

### A.1 B-Spline KAN im Decoder

Der KAN-Decoder nutzt B-Spline Funktionen (Grad 3) als lernbare Aktivierungsfunktionen. Jeder KANNode berechnet:

$$\phi_{ij}(x) = \sum_{k=0}^{K-1} c_k \cdot B_{k,3}(x)$$

wobei $B_{k,3}$ die kubische B-Spline Basisfunktion und $c_k$ die lernbaren Koeffizienten sind (K=10 pro Node).

**Gradient:** $\frac{\partial \phi}{\partial c_k} = B_{k,3}(x)$ — analytisch, kein numerischer Gradient nötig.

### A.2 Token-Generierung als Attractor-Projektion

Die Hypothese: Im Attractor-Zustand der InteractionLayer kodieren die Aktivierungsmuster **implizit** die Token-Sequenz. Der Decoder projiziert dieses Pattern in den Token-Raum.

Formal: Sei $\mathbf{a}^* = \{a^*_1, ..., a^*_N\}$ der Fixpunkt der InteractionLayer. Die Projektion auf Token $t$ ist:

$$P(t | \mathbf{a}^*) = \text{softmax}\left(\mathbf{h}^T \cdot \mathbf{E}_t\right)$$

wobei $\mathbf{h} = \text{KAN}_\text{decode}(\text{fuse}(\mathbf{a}^*))$ und $\mathbf{E}_t$ das Embedding von Token $t$.

### A.3 Gating-Fusion Formalismus

$$g_i = \sigma\left(\mathbf{w}_g^T \begin{bmatrix} \|a^*_i\| \\ r_i \\ c_i \end{bmatrix} + b_g\right)$$

$$\mathbf{f} = \text{Project}\left(\bigoplus_{i \in \text{top-3}} g_i \cdot a^*_i \ \Vert\ \mathbf{g} \ \Vert\ \mathbf{t}\right)$$

wobei $\bigoplus$ die Konkatenation, $r_i$ die Relevanz, $c_i$ die Kausalitätsstärke, und $\mathbf{t}$ der Template-One-Hot-Vektor ist.

---

## Anhang B: Vergleich mit verwandten Ansätzen

| Ansatz | Mechanismus | Stärke | Schwäche |
|:-------|:-----------|:-------|:---------|
| RAG (Retrieval-Augmented Generation) | LLM + Dokumenten-Retrieval | Flüssige Sprache + Fakten | Kein echtes Verständnis, Halluzination |
| KGQA (Knowledge Graph QA) | Graph-Traversierung + Template | Exakt, nachvollziehbar | Hölzerne Sprache, limitierte Komplexität |
| Neuro-Symbolic | Neuronales Netz + Logik-Engine | Beweisbar korrekte Schlüsse | Schwer trainierbar, spröde |
| **Brain19 Hybrid** | **KAN + MicroModels + KG** | **Verständnis + Nachvollziehbarkeit** | **Limitierte Sprachqualität** |

Brain19's Unique Selling Point: **Epistemische Transparenz.** Jeder generierte Token ist auf einen Attractor-Zustand zurückführbar, der auf konkreten Wissensgraph-Relationen basiert. Das System weiß, was es weiß — und was nicht.

---

*Dieses Dokument ist eine Machbarkeitsstudie. Die tatsächliche Sprachqualität kann erst nach Implementation und Training bewertet werden. Der Template-Fallback stellt sicher, dass auch bei Decoder-Versagen verständliche Ausgaben möglich sind.*
