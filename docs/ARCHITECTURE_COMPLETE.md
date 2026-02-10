# Brain19 — Vollständige Architektur-Dokumentation

> Generiert aus Source-Code-Analyse, 10. Februar 2026.
> Kein Raten — nur was im Code steht.

---

## Inhaltsverzeichnis

1. [Gesamtarchitektur (ASCII-Diagramm)](#1-gesamtarchitektur)
2. [Subsystem-Übersicht](#2-subsystem-übersicht)
3. [Klassen-Katalog](#3-klassen-katalog)
4. [Subsystem-Interaktionen](#4-subsystem-interaktionen)
5. [Datenfluss-Diagramme](#5-datenfluss)
6. [ASCII-Diagramme](#6-ascii-diagramme)
7. [Klassen-Dependency-Graph](#7-dependency-graph)

---

## 1. Gesamtarchitektur

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BRAIN19 BACKEND                              │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────────┐   │
│  │  tools/   │   │  importers/  │   │       ingestor/           │   │
│  │brain19_cli│──▶│ Wikipedia    │──▶│ IngestionPipeline         │   │
│  │           │   │ Scholar      │   │  TextChunker              │   │
│  └──────────┘   └──────────────┘   │  EntityExtractor          │   │
│                                     │  RelationExtractor        │   │
│                                     │  TrustTagger              │   │
│                                     │  ProposalQueue            │   │
│                                     │  KnowledgeIngestor        │   │
│                                     └─────────┬─────────────────┘   │
│                                               │ commit_approved()   │
│                                               ▼                     │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                    ltm/ LongTermMemory                     │      │
│  │  ConceptInfo (id, label, definition, EpistemicMetadata)    │      │
│  │  RelationInfo (id, source, target, type, weight)           │      │
│  │  *** READ-ONLY für alle außer Ingestor ***                 │      │
│  └──────┬────────────────────┬───────────────────────────────┘      │
│         │ READ               │ READ                                  │
│         ▼                    ▼                                       │
│  ┌──────────────┐   ┌────────────────────────┐                      │
│  │  memory/     │   │  cognitive/             │                      │
│  │  STM         │◀──│  CognitiveDynamics      │                      │
│  │  BrainCtrl   │   │  (Spreading, Salience,  │                      │
│  └──────┬───────┘   │   Focus, ThoughtPaths)  │                      │
│         │           └────────────┬─────────────┘                     │
│         │ STM activation         │                                   │
│         ▼                        ▼                                   │
│  ┌────────────────────────────────────────────┐                      │
│  │          understanding/                     │                      │
│  │  UnderstandingLayer                         │                      │
│  │  MiniLLM (Interface) ← StubMiniLLM         │                      │
│  │                       ← OllamaMiniLLM       │                      │
│  │  Proposals: Meaning, Hypothesis, Analogy,   │                      │
│  │             Contradiction (ALL HYPOTHESIS)   │                      │
│  └─────────────────────────────────────────────┘                     │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐       │
│  │  micromodel/  │   │  kan/         │   │  adapter/          │       │
│  │  MicroModel   │   │  KANNode      │   │  KANAdapter        │       │
│  │  Registry     │   │  KANLayer     │   │  (Brain↔KAN       │       │
│  │  Trainer      │   │  KANModule    │   │   Interface)       │       │
│  │  RelevanceMap │   │  FuncHypo     │   └────────────────────┘       │
│  │  EmbedMgr     │   └──────────────┘                                │
│  │  Persistence  │                                                    │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐     │
│  │  curiosity/   │   │  llm/         │   │  snapshot_generator  │     │
│  │  CuriosityEng │   │  OllamaClient│   │  (JSON für Frontend) │     │
│  │  (Trigger-    │   │  ChatInterfce│   └──────────────────────┘     │
│  │   Generator)  │   └──────────────┘                                │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐                                 │
│  │  common/      │   │  epistemic/   │                                │
│  │  types.hpp    │   │  EpistemicMeta│                                │
│  │  (ConceptId,  │   │  (Type,Status │                                │
│  │   ContextId,  │   │   Trust)      │                                │
│  │   RelationId) │   └──────────────┘                                │
│  └──────────────┘                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Subsystem-Übersicht

| Subsystem | Verzeichnis | Verantwortung |
|-----------|-------------|---------------|
| **Common** | `common/` | Basis-Typen (ConceptId, ContextId, RelationId) |
| **Epistemic** | `epistemic/` | EpistemicMetadata, EpistemicType, EpistemicStatus — Kern-Invarianten |
| **LTM** | `ltm/` | Persistenter Wissensspeicher (Concepts + Relations) |
| **Memory** | `memory/` | STM (Aktivierungen), BrainController (Orchestrierung) |
| **Cognitive** | `cognitive/` | Spreading Activation, Salience, Focus, ThoughtPaths |
| **MicroModel** | `micromodel/` | Per-Concept Bilineare Modelle, Training, Relevanz |
| **KAN** | `kan/` | Kolmogorov-Arnold Networks (B-Spline Funktionsapproximation) |
| **Adapter** | `adapter/` | Brücke zwischen BrainController und KAN |
| **Understanding** | `understanding/` | Semantische Analyse via Mini-LLMs, Proposal-Generierung |
| **LLM** | `llm/` | Ollama-Client, Chat-Interface für Verbalisierung |
| **Curiosity** | `curiosity/` | Signal-Generator (erkennt Muster, emittiert Trigger) |
| **Ingestor** | `ingestor/` | Wissens-Eingabe-Pipeline (JSON/CSV/Text → LTM) |
| **Importers** | `importers/` | Wikipedia/Scholar-Import → KnowledgeProposal |
| **Tools** | `tools/` | CLI-Tool (brain19_cli) |
| **Snapshot** | Root | JSON-Snapshot-Generator für Frontend-Visualisierung |

---

## 3. Klassen-Katalog

### 3.1 common/types.hpp

**`ConceptId`** — `uint64_t`
**`ContextId`** — `uint64_t`
**`RelationId`** — `uint64_t`
Basis-Typen für IDs im gesamten System.

---

### 3.2 epistemic/epistemic_metadata.hpp

#### `EpistemicType` (enum class)
- **File:** `epistemic/epistemic_metadata.hpp`
- **Werte:** FACT, DEFINITION, THEORY, HYPOTHESIS, INFERENCE, SPECULATION
- **Invariante:** Kein UNKNOWN — Abwesenheit ist Compile-Error

#### `EpistemicStatus` (enum class)
- **Werte:** ACTIVE, CONTEXTUAL, SUPERSEDED, INVALIDATED
- **Invariante:** Kein UNKNOWN, INVALIDATED = nie gelöscht

#### `EpistemicMetadata` (struct)
- **Verantwortung:** Pflicht-Metadaten für JEDES Wissenselement
- **Members:** `type`, `status`, `trust` (double [0.0, 1.0])
- **Konstruktor:** Explizit mit allen 3 Feldern, Default-Konstruktor = deleted
- **Invarianten:**
  - Trust validated bei Konstruktion (throw bei out_of_range)
  - INVALIDATED + trust ≥ 0.2 → Debug-Assertion
- **Genutzt von:** ConceptInfo, LTM, Ingestor, SnapshotGenerator, ChatInterface, alle Demos/Tests

---

### 3.3 ltm/

#### `ConceptInfo` (struct) — `ltm/long_term_memory.hpp`
- **Verantwortung:** Wissenselement mit Pflicht-Epistemic-Metadata
- **Members:** `id` (ConceptId), `label`, `definition`, `epistemic` (EpistemicMetadata)
- **Default-Konstruktor:** deleted
- **Genutzt von:** LTM, ChatInterface, SnapshotGenerator, UnderstandingLayer

#### `RelationInfo` (struct) — `ltm/relation.hpp`
- **Verantwortung:** Persistente gerichtete Relation zwischen Concepts
- **Members:** `id`, `source`, `target`, `type` (RelationType), `weight` [0,1]
- **Genutzt von:** LTM, CognitiveDynamics (Spreading), MicroTrainer

#### `LongTermMemory` (class) — `ltm/long_term_memory.hpp/.cpp`
- **Verantwortung:** Persistenter Wissensspeicher — Concepts + Relations
- **Wichtige Members:**
  - `concepts_` (unordered_map<ConceptId, ConceptInfo>)
  - `relations_` (unordered_map<RelationId, RelationInfo>)
  - `outgoing_relations_`, `incoming_relations_` (Indizes)
- **Wichtige Methoden:**
  - `store_concept(label, definition, EpistemicMetadata)` → ConceptId
  - `retrieve_concept(id)` → optional<ConceptInfo>
  - `invalidate_concept(id)` — setzt Status auf INVALIDATED, löscht NICHT
  - `add_relation(src, tgt, type, weight)` → RelationId
  - `get_outgoing_relations(src)`, `get_incoming_relations(tgt)`
  - `get_concepts_by_type(type)`, `get_concepts_by_status(status)`
  - `get_all_concept_ids()`, `get_active_concepts()`
- **Invarianten:**
  - Jedes Concept hat EpistemicMetadata (erzwungen durch Constructor)
  - Wissen wird NIE gelöscht, nur INVALIDATED
- **Genutzt von:** Praktisch ALLES — CognitiveDynamics, UnderstandingLayer, ChatInterface, IngestionPipeline, MicroTrainer, SnapshotGenerator

---

### 3.4 memory/

#### `ActivationLevel` (enum class) — `memory/activation_level.hpp`
- LOW (<0.3), MEDIUM (0.3-0.7), HIGH (≥0.7)

#### `ActivationClass` (enum class)
- CORE_KNOWLEDGE (langsamer Decay), CONTEXTUAL (schneller Decay)

#### `STMEntry` (struct) — `memory/stm_entry.hpp`
- **Members:** `concept_id`, `activation` [0,1], `classification`, `last_used`

#### `RelationType` (enum class) — `memory/active_relation.hpp`
- IS_A, HAS_PROPERTY, CAUSES, ENABLES, PART_OF, SIMILAR_TO, CONTRADICTS, SUPPORTS, TEMPORAL_BEFORE, CUSTOM

#### `ActiveRelation` (struct) — `memory/active_relation.hpp`
- **Members:** `source`, `target`, `type`, `activation`, `last_used`

#### `ShortTermMemory` (class) — `memory/stm.hpp/.cpp`
- **Verantwortung:** Rein mechanische Aktivierungsschicht — speichert NUR Aktivierungen, NIE Wissen
- **Wichtige Members:**
  - `contexts_` (map<ContextId, Context>), wobei Context = {concepts, relations}
  - Decay-Raten: `core_decay_rate_` (0.05), `contextual_decay_rate_` (0.15), `relation_decay_rate_` (0.25)
- **Wichtige Methoden:**
  - `create_context()`, `destroy_context()`, `clear_context()`
  - `activate_concept()`, `activate_relation()`, `boost_concept()`, `boost_relation()`
  - `get_concept_activation()`, `get_active_concepts(threshold)`, `get_active_relations(threshold)`
  - `decay_all(context, time_delta)` — exponentieller Decay, Two-Phase für Relations
- **Genutzt von:** BrainController, CognitiveDynamics, UnderstandingLayer

#### `BrainController` (class) — `memory/brain_controller.hpp/.cpp`
- **Verantwortung:** Minimaler Orchestrierungs-Layer — Context-Management, Flow-Koordination
- **Darf NICHT:** Lernen, Schlussfolgern, Wichtigkeit bewerten
- **Besitzt:** `unique_ptr<ShortTermMemory> stm_`
- **Wichtige Methoden:**
  - `initialize()`, `shutdown()`
  - `create_context()`, `destroy_context()`
  - `begin_thinking()`, `end_thinking()`
  - `activate_concept_in_context()`, `activate_relation_in_context()`
  - `decay_context()`
  - `get_stm()` (const), `get_stm_mutable()` (für CognitiveDynamics)
- **Genutzt von:** Alle Demos, CLI, UnderstandingLayer, CognitiveDynamics

---

### 3.5 cognitive/

#### `CognitiveDynamicsConfig` (struct) — `cognitive/cognitive_config.hpp`
- Sub-Configs: `ActivationSpreaderConfig`, `FocusManagerConfig`, `SalienceComputerConfig`, `ThoughtPathConfig`
- Salience-Formel-Gewichte: activation (0.4), trust (0.3), connectivity (0.2), recency (0.1)

#### State-Types (alle in `cognitive_config.hpp`):
- `ActivationEntry`, `FocusEntry`, `SalienceScore`, `ThoughtPathNode`, `ThoughtPath`, `SpreadingStats`

#### `CognitiveDynamics` (class) — `cognitive/cognitive_dynamics.hpp/.cpp`
- **Verantwortung:** Additive Schicht für Spreading Activation, Salience, Focus, ThoughtPaths
- **Architektur-Vertrag:**
  - ✅ READ-ONLY auf LTM und Trust
  - ✅ Schreibt NUR in STM (Aktivierungen) und eigenen Zustand (Focus)
  - ❌ Darf NICHT: Wissen erzeugen, Trust ändern, epistemische Entscheidungen treffen
- **Wichtige Methoden:**
  - **Spreading:** `spread_activation(source, activation, ctx, ltm, stm)` — Formel: `act(B) += act(A) × weight × trust × damping^depth`
  - **Salience:** `compute_salience(cid, ctx, ltm, stm)` → SalienceScore, `get_top_k_salient()`, `compute_query_salience()`
  - **Focus:** `init_focus()`, `focus_on()`, `decay_focus()`, `get_focus_set()` — Miller's 7±2
  - **Paths:** `find_best_paths()`, `find_paths_to()` — Beam-Search
- **Wichtige Members:**
  - `config_` (CognitiveDynamicsConfig)
  - `focus_sets_` (map<ContextId, vector<FocusEntry>>)
  - `stats_` (atomic Counters)
- **Genutzt von:** UnderstandingLayer (perform_understanding_cycle), demo_cognitive_dynamics

---

### 3.6 micromodel/

#### `MicroModel` (class) — `micromodel/micro_model.hpp/.cpp`
- **Verantwortung:** Bilineares per-Concept Modell: `w = σ(eᵀ · (W·c + b))`
- **Members:** `W_` (10×10), `b_` (10), `e_init_` (10), `c_init_` (10), `state_` (Adam-Optimizer, 300 params)
- **Methoden:** `predict(e, c)`, `train_step(e, c, target, config)`, `train(samples, config)`, `to_flat()/from_flat()`
- **Flat-Size:** 430 doubles

#### `MicroModelRegistry` (class) — `micromodel/micro_model_registry.hpp/.cpp`
- **Verantwortung:** Eine MicroModel pro ConceptId
- **Members:** `models_` (unordered_map<ConceptId, MicroModel>)
- **Methoden:** `create_model()`, `get_model()`, `ensure_models_for(ltm)`, `size()`

#### `EmbeddingManager` (class) — `micromodel/embedding_manager.hpp/.cpp`
- **Verantwortung:** 10D Embeddings für RelationTypes (10 feste) und benannte Contexts (auto-created)
- **Members:** `relation_embeddings_` (10×Vec10), `context_embeddings_` (map<string, Vec10>)
- **Heuristische Dimensionen:** 0=hierarchical, 1=causal, 2=compositional, 3=similarity, 4=temporal, 5=support, 6=specificity, 7=directionality, 8=abstractness, 9=strength

#### `MicroTrainer` (class) — `micromodel/micro_trainer.hpp/.cpp`
- **Verantwortung:** Generiert Trainingsdaten aus KG-Struktur, trainiert MicroModels
- **Positives:** Outgoing relations (target=weight), Incoming (target=weight×0.8)
- **Negatives:** 3× pro Positive, target≈0.05 (non-connected Concepts)
- **Methoden:** `train_all(registry, embeddings, ltm)`, `train_single()`, `generate_samples()`

#### `RelevanceMap` (class) — `micromodel/relevance_map.hpp/.cpp`
- **Verantwortung:** Evaluiert MicroModel über alle KG-Nodes → scored Relevanz-Map
- **Methoden:** `compute(source, registry, embeddings, ltm, rel_type, context)`, `top_k()`, `above_threshold()`
- **Overlay:** `overlay(other, mode, weight)` — ADDITION, MAX, WEIGHTED_AVERAGE
- **Genutzt für:** Phase 3 Creativity (Kombination multipler Perspektiven)

#### `persistence` (namespace) — `micromodel/persistence.hpp/.cpp`
- **Verantwortung:** Binäre Serialisierung von MicroModels + Embeddings
- **Format:** Magic "BM19", Header (32 bytes), Models (3448 bytes each), Relation Embeddings, Context Embeddings, XOR-Checksum
- **Methoden:** `save()`, `load()`, `validate()`

---

### 3.7 kan/

#### `KANNode` (class) — `kan/kan_node.hpp/.cpp`
- **Verantwortung:** Univariate lernbare Funktion via kubischen B-Splines (Cox-de Boor)
- **Members:** `num_knots_`, `knots_`, `coefficients_`
- **Methoden:** `evaluate(x)`, `gradient(x)`, `set_coefficients()`

#### `KANLayer` (class) — `kan/kan_layer.hpp/.cpp`
- **Verantwortung:** Kollektion von KANNodes, additive Kombination
- **Members:** `nodes_` (vector<unique_ptr<KANNode>>)

#### `KANModule` (class) — `kan/kan_module.hpp/.cpp`
- **Verantwortung:** Kompletter Funktionsapproximator f: R^n → R^m
- **Members:** `input_dim_`, `output_dim_`, `layers_` (ein Layer pro Output-Dimension)
- **Methoden:** `evaluate(inputs)`, `train(dataset, config)`, `compute_mse()`

#### `FunctionHypothesis` (struct) — `kan/function_hypothesis.hpp`
- **Verantwortung:** Daten-Wrapper für gelernte Funktion (keine Logik)
- **Members:** `input_dim`, `output_dim`, `module` (shared_ptr), `training_iterations`, `training_error`

---

### 3.8 adapter/

#### `KANAdapter` (class) — `adapter/kan_adapter.hpp/.cpp`
- **Verantwortung:** Saubere Schnittstelle zwischen BrainController und KAN
- **Members:** `modules_` (map<uint64_t, KANModuleEntry>), `next_module_id_`
- **Methoden:** `create_kan_module()`, `train_kan_module()`, `evaluate_kan_module()`, `destroy_kan_module()`

---

### 3.9 understanding/

#### `MiniLLM` (abstract class) — `understanding/mini_llm.hpp`
- **Verantwortung:** Interface für semantische Modelle — READ-ONLY, alle Outputs HYPOTHESIS
- **Methoden (pure virtual):** `extract_meaning()`, `generate_hypotheses()`, `detect_analogies()`, `detect_contradictions()`
- **Invariante:** DARF NICHT in LTM schreiben, Trust setzen, FACT-Promotion durchführen

#### `StubMiniLLM` (class) — `understanding/mini_llm.hpp/.cpp`
- **Verantwortung:** Test-Placeholder, gibt Dummy-Proposals zurück
- **Verifiziert:** Epistemische Invarianten bei jedem Output

#### `OllamaMiniLLM` (class) — `understanding/ollama_mini_llm.hpp/.cpp`
- **Verantwortung:** Echte semantische Analyse via Ollama API
- **Members:** `ollama_` (OllamaClient), `config_`, `proposal_counter_`
- **Baut Prompts aus Concept-Descriptions (READ-ONLY LTM)**

#### `MiniLLMFactory` (class) — `understanding/mini_llm_factory.hpp`
- **Status:** TODO — geplant für KAN-LLM Hybrid Layer
- **Konzept:** Erzeugt spezialisierte Mini-LLMs für gelernte Konzeptbereiche

#### `SpecializedMiniLLM` (class) — `understanding/mini_llm_factory.hpp`
- **Status:** TODO — geplant für KAN-LLM Hybrid Layer

#### Proposal-Types — `understanding/understanding_proposals.hpp`:
- **`MeaningProposal`:** Semantischer Vorschlag, `epistemic_type = HYPOTHESIS` (immer, hardcoded)
- **`HypothesisProposal`:** Vorgeschlagene Hypothese, `SuggestedEpistemic.suggested_type = HYPOTHESIS` (erzwungen)
- **`AnalogyProposal`:** Strukturelle Analogie zwischen Concept-Sets
- **`ContradictionProposal`:** Erkannte potenzielle Inkonsistenz

#### `UnderstandingLayer` (class) — `understanding/understanding_layer.hpp/.cpp`
- **Verantwortung:** Semantische Analyse-Schicht über Cognitive Dynamics
- **Architektur-Vertrag:**
  - ✅ READ-ONLY auf LTM
  - ✅ Alle Outputs = HYPOTHESIS
  - ❌ DARF NICHT: KG modifizieren, Trust setzen, Regeln generieren
- **Methoden:**
  - `analyze_meaning()`, `propose_hypotheses()`, `find_analogies()`, `check_contradictions()`
  - `perform_understanding_cycle(seed, cognitive_dynamics, ltm, stm, ctx)` — Vollzyklus:
    1. Spreading Activation via CognitiveDynamics
    2. Salience-Berechnung für wichtige Concepts
    3. Mini-LLMs auf salient Concepts anwenden
    4. Proposals generieren + filtern

---

### 3.10 llm/

#### `OllamaConfig` (struct) — `llm/ollama_client.hpp`
- **Members:** `host`, `model` ("llama3.2:3b"), `temperature`, `num_predict`, `stream`

#### `OllamaClient` (class) — `llm/ollama_client.hpp/.cpp`
- **Verantwortung:** HTTP-Client für Ollama REST API
- **Members:** `config_`, `initialized_`
- **Methoden:** `initialize()`, `is_available()`, `list_models()`, `generate()`, `chat()`
- **Dependencies:** libcurl, nlohmann/json

#### `ChatInterface` (class) — `llm/chat_interface.hpp/.cpp`
- **Verantwortung:** LLM-gestützte Verbalisierung von Brain19-Wissen — LLM ist TOOL, kein Agent
- **Invariante:** LLM verbalisiert NUR vorhandenes LTM-Wissen (READ-ONLY)
- **Methoden:** `ask()`, `explain_concept()`, `compare()`, `list_knowledge()`, `get_summary()`
- **System-Prompt:** Erzwingt epistemische Rigorosität in Antworten (Type, Trust, Warnungen)

---

### 3.11 curiosity/

#### `TriggerType` (enum class) — `curiosity/curiosity_trigger.hpp`
- SHALLOW_RELATIONS, MISSING_DEPTH, LOW_EXPLORATION, RECURRENT_WITHOUT_FUNCTION

#### `CuriosityTrigger` (struct)
- **Members:** `type`, `context_id`, `related_concept_ids`, `description`

#### `CuriosityEngine` (class) — `curiosity/curiosity_engine.hpp/.cpp`
- **Verantwortung:** Reiner Signal-Generator — beobachtet, emittiert Trigger, KEINE Aktionen
- **Methoden:** `observe_and_generate_triggers(observations)` → vector<CuriosityTrigger>
- **Detektoren:** `detect_shallow_relations()` (relations < 30% der concepts), `detect_low_exploration()`

---

### 3.12 ingestor/

#### `TextChunker` (class) — `ingestor/text_chunker.hpp/.cpp`
- **Verantwortung:** Satzbasis-Chunking von Plaintext
- **Config:** `sentences_per_chunk` (3), `overlap_sentences` (1), `max_chunk_chars` (2000)

#### `EntityExtractor` (class) — `ingestor/entity_extractor.hpp/.cpp`
- **Verantwortung:** Pattern-basierte Entity-Extraktion (keine externen NLP-Dependencies)
- **Strategien:** Capitalized phrases, Quoted terms, Definition patterns ("X is a ..."), Frequent terms
- **Output:** `ExtractedEntity` (label, context, frequency, flags)

#### `RelationExtractor` (class) — `ingestor/relation_extractor.hpp/.cpp`
- **Verantwortung:** Pattern-basierte Relation-Extraktion
- **Patterns:** "X is a Y" → IS_A, "X causes Y" → CAUSES, "X is part of Y" → PART_OF, etc.
- **Output:** `ExtractedRelation` (source, target, type, evidence, confidence)

#### `TrustTagger` (class) — `ingestor/trust_tagger.hpp/.cpp`
- **Verantwortung:** Mappt Trust-Kategorien auf existierendes Epistemic-System
- **Trust-Ranges:** FACTS (0.95-0.99), DEFINITIONS (0.90-0.99), THEORIES (0.85-0.95), HYPOTHESES (0.50-0.80), INFERENCES (0.40-0.70), SPECULATION (0.10-0.40), INVALIDATED (0.01-0.10)
- **Text-Signale:** Hedging-Sprache (↓), Certainty-Sprache (↑), Definition-Muster, Citation-Marker

#### `ProposalQueue` (class) — `ingestor/proposal_queue.hpp/.cpp`
- **Verantwortung:** Queue für unvalidierte Knowledge-Proposals vor LTM-Einspeisung
- **Status-Lifecycle:** PENDING → APPROVED/REJECTED/MODIFIED/EXPIRED
- **Methoden:** `enqueue()`, `review()`, `auto_approve_all()`, `pop_approved()`, `expire_old()`
- **Epistemische Regel:** Nichts geht in LTM ohne durch diese Queue

#### `KnowledgeIngestor` (class) — `ingestor/knowledge_ingestor.hpp/.cpp`
- **Verantwortung:** Parst JSON/CSV → StructuredInput → IngestProposals
- **JSON-Parser:** Minimaler Hand-geschriebener Parser (keine externe JSON-Library im Ingestor)
- **Methoden:** `parse_json()`, `parse_csv_concepts()`, `parse_csv_relations()`, `to_proposals()`

#### `IngestionPipeline` (class) — `ingestor/ingestion_pipeline.hpp/.cpp`
- **Verantwortung:** Komplette Pipeline: Input → Chunker → Extractor → TrustTagger → ProposalQueue → LTM
- **Architektur:** Pipeline schreibt NIE direkt in LTM — immer über ProposalQueue
- **Modi:** Interactive (Queue-Review) oder Auto-Approve (trusted sources)
- **Methoden:** `ingest_json()`, `ingest_csv()`, `ingest_text()`, `commit_approved()`
- **Invarianten:** Existing LTM-Data wird NICHT modifiziert, Duplicates werden übersprungen

---

### 3.13 importers/

#### `KnowledgeProposal` (struct) — `importers/knowledge_proposal.hpp`
- **Verantwortung:** Reiner Datencontainer für Vorschläge — KEINE epistemischen Entscheidungen
- **Epistemische Regel:** Importers DÜRFEN NICHT EpistemicType/Trust/Status zuweisen, nur `SuggestedEpistemicType`
- **Members:** `source_type`, `title`, `extracted_text`, `suggested_concepts`, `suggested_relations`, `suggested_epistemic_type`

#### `WikipediaImporter` (class) — `importers/wikipedia_importer.hpp/.cpp`
- **Verantwortung:** Extrahiert strukturierte Übersicht aus Wikipedia-Artikeln → KnowledgeProposal
- **Methoden:** `import_article()`, `parse_wikipedia_text()`
- **Output:** DEFINITION_CANDIDATE (Suggestion only)

#### `ScholarImporter` (class) — `importers/scholar_importer.hpp/.cpp`
- **Verantwortung:** Extrahiert Forschungswissen aus Papers → KnowledgeProposal
- **Methoden:** `import_paper_by_doi()`, `parse_paper_text()`
- **Output:** HYPOTHESIS_CANDIDATE oder THEORY_CANDIDATE (basierend auf Uncertainty-Language)

---

### 3.14 Root-Level

#### `SnapshotGenerator` (class) — `snapshot_generator.hpp/.cpp`
- **Verantwortung:** Erzeugt JSON-Snapshots für Frontend-Visualisierung
- **Invariante:** MUSS epistemische Metadaten in Snapshot exponieren
- **Methoden:** `generate_json_snapshot(brain, ltm, curiosity, context_id)`

---

## 4. Subsystem-Interaktionen

### 4.1 Wer ruft wen auf?

```
IngestionPipeline ──▶ TextChunker, EntityExtractor, RelationExtractor
                 ──▶ TrustTagger, KnowledgeIngestor
                 ──▶ ProposalQueue
                 ──▶ LongTermMemory (NUR commit_approved, WRITE)

UnderstandingLayer ──▶ CognitiveDynamics (Spreading, Salience)
                   ──▶ MiniLLM-Implementierungen (StubMiniLLM, OllamaMiniLLM)
                   ──▶ LTM (READ-ONLY)
                   ──▶ STM (via CognitiveDynamics für Aktivierungen)

CognitiveDynamics ──▶ LTM (READ-ONLY: Concepts, Relations, Trust)
                  ──▶ STM (WRITE: Aktivierungen setzen/boosten)

BrainController ──▶ STM (owns, creates, delegates)

MicroTrainer ──▶ LTM (READ-ONLY: Relations für Trainingsdaten)
             ──▶ EmbeddingManager (Embeddings für Training)
             ──▶ MicroModelRegistry (Models trainieren)

ChatInterface ──▶ LTM (READ-ONLY: find_relevant_concepts)
              ──▶ OllamaClient (LLM-Calls)

SnapshotGenerator ──▶ BrainController (STM-Daten)
                  ──▶ LTM (READ-ONLY: Epistemic-Metadata)
                  ──▶ CuriosityEngine (Trigger)
```

### 4.2 Read-Only vs. Read-Write Zugriffe

| Subsystem | LTM Zugriff | STM Zugriff | Eigener State |
|-----------|-------------|-------------|---------------|
| CognitiveDynamics | **READ-ONLY** | **READ-WRITE** | WRITE (Focus) |
| UnderstandingLayer | **READ-ONLY** | READ (via CogDyn) | WRITE (Stats) |
| IngestionPipeline | **READ-WRITE** (nur commit) | — | WRITE (Queue) |
| ChatInterface | **READ-ONLY** | — | — |
| MicroTrainer | **READ-ONLY** | — | — |
| SnapshotGenerator | **READ-ONLY** | READ | — |
| CuriosityEngine | — | — | WRITE (Thresholds) |
| BrainController | — | **OWNS** | WRITE (ThinkingState) |

### 4.3 Ownership

| Objekt | Owner | Lifecycle |
|--------|-------|-----------|
| STM | BrainController (unique_ptr) | init→shutdown |
| Context | STM (intern) | create→destroy |
| MicroModel | MicroModelRegistry | create→remove/clear |
| KANModule | KANAdapter (shared_ptr) | create→destroy |
| MiniLLM | UnderstandingLayer (unique_ptr) | register→~destructor |
| IngestProposal | ProposalQueue | enqueue→pop_approved |
| FocusEntry | CognitiveDynamics | init_focus→clear_focus |

---

## 5. Datenfluss

### 5.1 Von User-Input bis Concept-Storage

```
User gibt JSON/CSV/Text ein
        │
        ▼
IngestionPipeline.ingest_json() / ingest_text()
        │
        ├──▶ KnowledgeIngestor.parse_json()
        │    oder TextChunker + EntityExtractor + RelationExtractor
        │
        ▼
TrustTagger.suggest_from_text() → TrustAssignment
        │
        ▼
IngestProposal (mit TrustAssignment) → ProposalQueue.enqueue()
        │
        ▼
[Human Review oder Auto-Approve]
        │
        ▼
ProposalQueue.pop_approved()
        │
        ▼
IngestionPipeline.commit_approved()
        │
        ├──▶ TrustAssignment.to_epistemic_metadata() → EpistemicMetadata
        │
        ▼
LTM.store_concept(label, definition, EpistemicMetadata)
LTM.add_relation(source, target, type, weight)
```

### 5.2 Spreading Activation Flow

```
CognitiveDynamics.spread_activation(source, initial_act, ctx, ltm, stm)
        │
        ▼
1. Validiere: source existiert in LTM? activation > threshold?
2. stm.activate_concept(ctx, source, activation, CONTEXTUAL)
3. Starte rekursives Spreading:
        │
        ▼
spread_recursive(current, activation, depth, ctx, ltm, stm, visited, stats)
        │
        ├── BASE CASE: depth >= max_depth (3) → return
        ├── BASE CASE: activation < threshold (0.01) → return
        ├── BASE CASE: already visited → return
        ├── Skip: INVALIDATED concepts propagieren NICHT
        │
        ▼
4. Für jede ausgehende Relation (READ von LTM):
   propagated = activation × rel.weight × source_trust × damping^(depth+1)
        │
        ▼
5. stm.activate_concept() oder stm.boost_concept() auf Target
6. Rekursiv: spread_recursive(target, propagated, depth+1, ...)
```

### 5.3 Salience Computation Flow

```
CognitiveDynamics.compute_salience(cid, ctx, ltm, stm, tick)
        │
        ▼
salience = activation_weight × stm.get_concept_activation(ctx, cid)    [0.4]
         + trust_weight     × ltm.retrieve_concept(cid).trust          [0.3]
         + connectivity_wt  × (relation_count / max_connectivity)       [0.2]
         + recency_weight   × exp(-0.07 × ticks_since_last_access)     [0.1]
         + query_boost      (wenn cid in Query-Concepts oder direkt verbunden)
        │
        ▼
clamp_salience(salience) → [0.0, max_salience=1.0]
```

### 5.4 Understanding/LLM Cycle Flow

```
UnderstandingLayer.perform_understanding_cycle(seed, cogdyn, ltm, stm, ctx)
        │
        ▼
PHASE 1: cogdyn.spread_activation(seed, 1.0, ctx, ltm, stm)
        │
        ▼
PHASE 2: Hole aktive Concepts aus STM
         cogdyn.compute_salience_batch() → Top salient Concepts
        │
        ▼
PHASE 3: Für jedes registrierte MiniLLM:
         mini_llm.extract_meaning(salient_concepts, ltm, stm, ctx)
         mini_llm.generate_hypotheses(...)
         mini_llm.detect_contradictions(...)
        │
        ├──▶ OllamaMiniLLM: Baut Prompt aus Concept-Descriptions
        │    → Ollama API Call → Parse Response → Proposal
        │
        ├──▶ StubMiniLLM: Gibt Dummy-Proposals zurück
        │
        ▼
PHASE 4: find_analogies (wenn genug Concepts)
        │
        ▼
Filter Proposals by confidence thresholds
        │
        ▼
Return UnderstandingResult (ALL PROPOSALS = HYPOTHESIS)
```

### 5.5 Curiosity Trigger Flow

```
CuriosityEngine.observe_and_generate_triggers(observations)
        │
        ▼
Für jede SystemObservation:
        │
        ├── detect_shallow_relations():
        │   ratio = active_relations / active_concepts
        │   if ratio < 0.3 → CuriosityTrigger(SHALLOW_RELATIONS)
        │
        ├── detect_low_exploration():
        │   if concepts > 0 && concepts < 5 → CuriosityTrigger(LOW_EXPLORATION)
        │
        ▼
Return vector<CuriosityTrigger>
```

### 5.6 Ingestion Pipeline Flow (Vollständig)

```
Structured Input (JSON/CSV):
  KnowledgeIngestor.parse_json()/parse_csv_concepts()
  → StructuredInput {concepts, relations}
  → to_proposals(input, tagger) → vector<IngestProposal>

Plain Text Input:
  TextChunker.chunk_text(text) → vector<TextChunk>
  EntityExtractor.extract_from_chunks(chunks) → vector<ExtractedEntity>
  RelationExtractor.extract_relations(text, entities) → vector<ExtractedRelation>
  TrustTagger.suggest_from_text(context) → TrustAssignment
  → vector<IngestProposal>

Alle Proposals → ProposalQueue.enqueue()
                  │
                  ▼
          [Review: approve/reject/modify]
                  │
                  ▼
          ProposalQueue.pop_approved()
                  │
                  ▼
commit_approved():
  1. Baue label→ID Map aus existierendem LTM
  2. Für jeden Proposal: LTM.store_concept() (skip Duplicates)
  3. Für jede Relation: LTM.add_relation() (skip existierende)
```

---

## 6. ASCII-Diagramme

### 6.1 Gesamtarchitektur

(Siehe Diagramm in Abschnitt 1)

### 6.2 Datenfluss

```
        ┌──────────┐     ┌──────────────┐     ┌──────────────────┐
        │   User   │     │  Importers   │     │  External Sources │
        │  Input   │     │ Wiki/Scholar │     │ (Ollama LLM)     │
        └────┬─────┘     └──────┬───────┘     └───────┬──────────┘
             │                  │                      │
             ▼                  ▼                      │
    ┌────────────────────────────────────┐             │
    │     INGESTION PIPELINE             │             │
    │  Chunker → Entity → Relation →    │             │
    │  TrustTagger → ProposalQueue      │             │
    └──────────────┬─────────────────────┘             │
                   │ commit                            │
                   ▼                                   │
    ┌──────────────────────────────────┐               │
    │        LONG-TERM MEMORY          │               │
    │  ConceptInfo + EpistemicMetadata  │               │
    │  RelationInfo (directed graph)    │               │
    └───┬──────────────┬───────────────┘               │
        │ READ         │ READ                          │
        ▼              ▼                               │
    ┌────────┐  ┌──────────────────┐                   │
    │  STM   │  │  COGNITIVE       │                   │
    │ (activ │◀─│  DYNAMICS        │                   │
    │  ation │  │  Spread/Salience │                   │
    │  only) │  │  Focus/Paths     │                   │
    └────┬───┘  └────────┬─────────┘                   │
         │               │                             │
         ▼               ▼                             │
    ┌──────────────────────────────────┐               │
    │     UNDERSTANDING LAYER          │◀──────────────┘
    │  MiniLLMs → Proposals            │  (Ollama API)
    │  (ALL HYPOTHESIS)                │
    └──────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────┐
    │  MICRO-MODEL SUBSYSTEM           │
    │  Per-Concept Bilinear Models     │
    │  Training → RelevanceMap         │
    └──────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────┐
    │  OUTPUT                          │
    │  ChatInterface (LLM verbalize)   │
    │  SnapshotGenerator (JSON/viz)    │
    │  CuriosityEngine (triggers)      │
    └──────────────────────────────────┘
```

### 6.3 Ownership/Lifecycle

```
    BrainController
        │ owns (unique_ptr)
        ▼
    ShortTermMemory
        │ contains
        ├── Context {concepts, relations}  ←── create/destroy Lifecycle
        └── Decay-Konfiguration

    IngestionPipeline
        │ contains
        ├── ProposalQueue (IngestProposals: enqueue → pop)
        ├── TrustTagger
        ├── TextChunker, EntityExtractor, RelationExtractor
        ├── KnowledgeIngestor
        │ references (&)
        └── LongTermMemory

    UnderstandingLayer
        │ owns (unique_ptr)
        └── vector<MiniLLM>  ←── register → destructor

    KANAdapter
        │ owns (shared_ptr)
        └── map<id, KANModule>  ←── create → destroy

    MicroModelRegistry
        │ owns (by-value)
        └── map<ConceptId, MicroModel>  ←── create → remove
```

### 6.4 Epistemischer Fluss

```
    ┌──────────────────────────────────────────────────┐
    │           EPISTEMIC ENFORCEMENT                   │
    └──────────────────────────────────────────────────┘

    1. IMPORT → SuggestedEpistemicType (NICHT EpistemicType!)
       Wiki/Scholar Importer
            │
            ▼ (SUGGESTION ONLY)
       KnowledgeProposal.suggested_epistemic_type

    2. INGESTION → TrustAssignment (SUGGESTION)
       TrustTagger.suggest_from_text()
            │
            ▼
       IngestProposal.trust_assignment

    3. REVIEW → Mensch entscheidet!
       ProposalQueue.review()
            │
            ▼ (override möglich)
       ReviewDecision {trust_override, new_status}

    4. COMMIT → EpistemicMetadata (FINAL)
       TrustAssignment.to_epistemic_metadata()
            │
            ▼
       LTM.store_concept(label, def, EpistemicMetadata)
       *** Compile-Error ohne EpistemicMetadata ***

    5. PROPAGATION → Trust fließt NUR lesend
       CognitiveDynamics:
         activation(B) = act(A) × weight × trust(A) × damping
         Trust wird GELESEN, nie GEÄNDERT

    6. UNDERSTANDING → Immer HYPOTHESIS
       MiniLLM Proposals:
         epistemic_type = HYPOTHESIS (hardcoded, kann nicht überschrieben werden)
         Epistemic Core entscheidet über Akzeptanz

    7. INVALIDATION → NIEMALS Deletion
       LTM.invalidate_concept():
         status = INVALIDATED
         trust = 0.05
         Wissen BLEIBT in LTM (epistemische Historie)

    8. SNAPSHOT → Epistemische Transparenz
       SnapshotGenerator:
         JEDES Concept hat type, status, trust im JSON
         STM-only Concepts → HYPOTHESIS/CONTEXTUAL/0.5
```

---

## 7. Dependency-Graph

```
                            common/types.hpp
                                  │
                    ┌─────────────┼──────────────┐
                    │             │              │
                    ▼             ▼              ▼
          memory/active_relation  memory/stm_entry  memory/activation_level
          (RelationType)          (STMEntry)         (ActivationLevel/Class)
                    │             │
                    ▼             ▼
              memory/stm.hpp ◀───┘
                    │
                    ▼
          memory/brain_controller.hpp
                    │
                    ▼
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼

  epistemic/epistemic_metadata.hpp
          │
          ▼
  ltm/relation.hpp ◀── memory/active_relation.hpp
          │
          ▼
  ltm/long_term_memory.hpp
          │
          ├──────────────────────────────────────────┐
          │                   │           │          │
          ▼                   ▼           ▼          ▼
  cognitive/             micromodel/  ingestor/  understanding/
  cognitive_config.hpp   micro_model  knowledge_ understanding_
  cognitive_dynamics.hpp  registry    ingestor   proposals.hpp
          │              embedding_   trust_     mini_llm.hpp
          │              micro_trainer tagger    understanding_
          │              relevance_map proposal_q  layer.hpp
          │              persistence   ingestion_  ollama_mini_
          │                            pipeline    llm.hpp
          │
          ▼
  curiosity/
  curiosity_trigger.hpp
  curiosity_engine.hpp

  kan/                         llm/
  kan_node.hpp                 ollama_client.hpp
  kan_layer.hpp ◀── kan_node   chat_interface.hpp ◀── ollama_client
  kan_module.hpp ◀── kan_layer                     ◀── ltm
  function_hypothesis ◀── kan_module

  adapter/
  kan_adapter.hpp ◀── kan_module, function_hypothesis

  importers/
  knowledge_proposal.hpp
  wikipedia_importer.hpp ◀── knowledge_proposal
  scholar_importer.hpp ◀── knowledge_proposal

  ingestor/
  text_chunker.hpp
  entity_extractor.hpp ◀── text_chunker, knowledge_proposal
  relation_extractor.hpp ◀── entity_extractor, active_relation
  trust_tagger.hpp ◀── epistemic_metadata, knowledge_proposal
  proposal_queue.hpp ◀── trust_tagger, entity_extractor, relation_extractor
  knowledge_ingestor.hpp ◀── knowledge_proposal, trust_tagger, proposal_queue
  ingestion_pipeline.hpp ◀── ltm, knowledge_ingestor, all ingestor components

  snapshot_generator.hpp ◀── brain_controller, ltm, curiosity_engine

  tools/brain19_cli.cpp ◀── ingestion_pipeline, micromodel/*, ltm
```

### Dependency-Richtungen (vereinfacht)

```
  common ──▶ epistemic ──▶ ltm
    │                       │
    ▼                       ├──▶ cognitive ──▶ understanding
  memory ◀──────────────────┤
    │                       ├──▶ micromodel
    │                       ├──▶ ingestor ◀── importers
    │                       ├──▶ llm/chat_interface
    │                       └──▶ snapshot_generator
    │
    └──▶ curiosity

  kan ──▶ adapter (eigenständig, über KANAdapter an Brain angebunden)
```

---

## Zusammenfassung der Kern-Invarianten

1. **Epistemic Explicitness:** JEDES Wissenselement HAT EpistemicMetadata — kein Default-Konstruktor, compile-time enforced
2. **No Implicit Trust:** Trust wird IMMER explizit gesetzt — kein stiller Fallback
3. **Knowledge Never Deleted:** INVALIDATED, nie gelöscht — epistemische Historie bleibt erhalten
4. **Importers ≠ Authority:** Importers liefern SUGGESTIONS, Mensch entscheidet
5. **Understanding = HYPOTHESIS:** Alle Mini-LLM-Outputs sind HYPOTHESIS — nie FACT
6. **LTM = Read-Only (für fast alle):** Nur IngestionPipeline.commit_approved() schreibt
7. **STM = Pure Activation:** Speichert NIE Wissen, nur Aktivierungslevel
8. **CognitiveDynamics = Read-Only LTM:** Liest Trust für Salience/Spreading, ändert NICHTS
9. **Bounded & Deterministic:** Alle Aktivierungen ∈ [0.0, 1.0], gleiche Inputs → gleiche Outputs
10. **Cycle-Safe:** Spreading Activation mit Visited-Set und Depth-Limit (max 3)
