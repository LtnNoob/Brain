# Brain19 — Vollständige Architektur-Dokumentation & Professor-Review

> **Stand:** 2026-02-10  
> **Codebase:** ~16.755 LOC C++20 (Backend) + React Frontend  
> **Grundlage:** Code-Audit aller 4 Teil-Agenten, vollständige Source-Analyse  
> **Autor:** Architektur-Dokumentarist (automatisch generiert)

---

# TEIL 1: ARCHITEKTUR-DOKUMENTATION

---

## 1. Gesamtübersicht

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BRAIN19 COGNITIVE ARCHITECTURE                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        BrainController                              │    │
│  │              (Orchestrierung, Lifecycle, Delegation)                 │    │
│  │              memory/brain_controller.hpp (~207 LOC)                  │    │
│  └──────┬──────────────┬──────────────┬──────────────┬─────────────────┘    │
│         │              │              │              │                       │
│         ▼              ▼              ▼              ▼                       │
│  ┌──────────┐   ┌───────────┐  ┌───────────┐  ┌──────────────────┐         │
│  │   STM    │   │    LTM    │  │ Cognitive  │  │   Epistemic      │         │
│  │ Short-   │   │  Long-    │  │ Dynamics   │  │   System         │         │
│  │ Term     │   │  Term     │  │            │  │                  │         │
│  │ Memory   │   │  Memory   │  │ Spreading  │  │ 6 Types          │         │
│  │          │   │  (KG)     │  │ Activation │  │ 4 States         │         │
│  │ Contexts │   │ Concepts  │  │ Salience   │  │ Compile-Time     │         │
│  │ Entries  │   │ Relations │  │ Focus Mgmt │  │ Enforcement      │         │
│  │ Decay    │   │ Epistemic │  │ ThoughtPath│  │                  │         │
│  └────┬─────┘   └─────┬─────┘  └─────┬──────┘  └────────┬─────────┘        │
│       │               │              │                   │                  │
│       │         ┌─────┴──────────────┴───────────────────┘                  │
│       │         │                                                           │
│       ▼         ▼                                                           │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐           │
│  │  MicroModel Layer    │    │  Curiosity Engine                │           │
│  │                      │    │                                  │           │
│  │  MicroModel (130P)   │    │  Pattern Detection               │           │
│  │  MicroModelRegistry  │◄───│  Trigger Generation              │           │
│  │  MicroTrainer        │    │  SHALLOW_RELATIONS               │           │
│  │  EmbeddingManager    │    │  LOW_CONNECTIVITY                │           │
│  │  RelevanceMap        │    │  HIGH_UNCERTAINTY                │           │
│  │  (Overlay/Combine)   │    │  (MISSING_DEPTH,RECURRENT n/a)  │           │
│  └──────────┬───────────┘    └──────────────────────────────────┘           │
│             │                                                               │
│             ▼                                                               │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐           │
│  │  KAN Subsystem       │    │  Understanding Layer (OPTIONAL)  │           │
│  │                      │    │                                  │           │
│  │  KANModule           │    │  MiniLLM (Stub/Ollama)           │           │
│  │  KANLayer            │    │  UnderstandingLayer              │           │
│  │  KANNode (B-Spline)  │    │  Proposals (Hypothesis,         │           │
│  │  KANAdapter          │    │   Analogy, Contradiction,       │           │
│  │  FunctionHypothesis  │    │   Meaning)                      │           │
│  └──────────────────────┘    │  Trust-Ceiling: 0.3-0.5         │           │
│                              └──────────────────────────────────┘           │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐           │
│  │  Ingestor Pipeline   │    │  Importers                      │           │
│  │                      │    │                                  │           │
│  │  TextChunker         │    │  WikipediaImporter               │           │
│  │  EntityExtractor     │    │  ScholarImporter                 │           │
│  │  RelationExtractor   │    │  KnowledgeProposal              │           │
│  │  TrustTagger         │    │                                  │           │
│  │  ProposalQueue       │    │                                  │           │
│  │  IngestionPipeline   │    └──────────────────────────────────┘           │
│  │  KnowledgeIngestor   │                                                   │
│  └──────────────────────┘    ┌──────────────────────────────────┐           │
│                              │  Snapshot Generator              │           │
│  ┌──────────────────────┐    │  (Read-Only State Export)        │           │
│  │  LLM / Chat          │    │  → snapshot.json                │           │
│  │  OllamaClient        │    └──────────────────────────────────┘           │
│  │  ChatInterface       │                                                   │
│  └──────────────────────┘    ┌──────────────────────────────────┐           │
│                              │  CLI Tool                        │           │
│                              │  brain19_cli                     │           │
│                              │  (ingest/query/explore/snapshot) │           │
│                              └──────────────────────────────────┘           │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     React Frontend (Read-Only)                       │   │
│  │              STM Graph (SVG) │ Epistemic Panel │ Curiosity Panel     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Klassen-Katalog

### 2.1 Kern-Subsysteme

| Klasse/Struct | Datei | Verantwortung | Zugriff |
|---|---|---|---|
| `BrainController` | `memory/brain_controller.hpp` | Orchestrierung, STM-Ownership, Thinking-Lifecycle | Schreibt: STM (via Delegation) |
| `ShortTermMemory` | `memory/stm.hpp` | Aktivierungsspeicher, Context-Isolation, Decay | Schreibt: eigenen State |
| `STMEntry` | `memory/stm_entry.hpp` | Einzelner Aktivierungseintrag (ID, activation, class, timestamp) | Datenstruktur |
| `ActiveRelation` | `memory/active_relation.hpp` | Aktive Relation im STM (source, target, type, activation) | Datenstruktur |
| `LongTermMemory` | `ltm/long_term_memory.hpp` | Persistenter Knowledge Graph (Concepts + Relations) | Schreibt: eigenen State |
| `ConceptInfo` | `ltm/long_term_memory.hpp` | Konzept mit EpistemicMetadata (kein Default-Ctor!) | Datenstruktur |
| `RelationInfo` | `ltm/relation.hpp` | Relation mit source, target, type, trust, label | Datenstruktur |
| `EpistemicMetadata` | `epistemic/epistemic_metadata.hpp` | Type + Status + Trust + Provenance (kein Default-Ctor!) | Datenstruktur |

### 2.2 Cognitive Dynamics

| Klasse/Struct | Datei | Verantwortung | Zugriff |
|---|---|---|---|
| `CognitiveDynamics` | `cognitive/cognitive_dynamics.hpp` | Spreading Activation, Salience, Focus Mgmt, ThoughtPaths | Liest: LTM, STM; Schreibt: STM |
| `SalienceScore` | `cognitive/cognitive_config.hpp` | Gewichteter Score (frequency, recency, connectivity, epistemics) | Datenstruktur |
| `FocusEntry` | `cognitive/cognitive_config.hpp` | Fokus-Eintrag pro Kontext (concept_id, strength, tick) | Datenstruktur |
| `ActivationEntry` | `cognitive/cognitive_config.hpp` | Spreading-Ergebnis (concept_id, activation, depth) | Datenstruktur |
| `ThoughtPath` | `cognitive/cognitive_config.hpp` | Kette von Konzepten mit Score | Datenstruktur |
| `SpreadingStats` | `cognitive/cognitive_config.hpp` | Statistiken über Spreading-Durchläufe | Datenstruktur |
| `CognitiveDynamicsConfig` | `cognitive/cognitive_config.hpp` | Konfiguration (Weights, Thresholds, Limits) | Config |

### 2.3 MicroModel Layer

| Klasse/Struct | Datei | Verantwortung | Zugriff |
|---|---|---|---|
| `MicroModel` | `micromodel/micro_model.hpp` | 130-Parameter bilineares Modell (W·c+b → eᵀ·v → σ) | Schreibt: eigene Weights |
| `MicroModelRegistry` | `micromodel/micro_model_registry.hpp` | Verwaltung aller MicroModels per ConceptId | Schreibt: Registry |
| `MicroTrainer` | `micromodel/micro_trainer.hpp` | Adam-Optimizer, Batch-Training | Schreibt: MicroModel-Weights |
| `EmbeddingManager` | `micromodel/embedding_manager.hpp` | Relation/Context-Embeddings (10D), Caching | Schreibt: Embedding-Cache |
| `RelevanceMap` | `micromodel/relevance_map.hpp` | Relevanz-Scores eines Konzepts für alle anderen | Read-Only (nach compute) |
| `TrainingSample` | `micromodel/micro_model.hpp` | (embedding, context, target) Trainingsdatum | Datenstruktur |
| `TrainingState` | `micromodel/micro_model.hpp` | Adam-State (momentum, variance, timestep) | Datenstruktur |

### 2.4 KAN Subsystem

| Klasse/Struct | Datei | Verantwortung | Zugriff |
|---|---|---|---|
| `KANNode` | `kan/kan_node.hpp` | Einzelne B-Spline-Funktion (Cox-de Boor) | Schreibt: Koeffizienten |
| `KANLayer` | `kan/kan_layer.hpp` | Schicht von KANNodes | Schreibt: via Nodes |
| `KANModule` | `kan/kan_module.hpp` | Gesamt-KAN-Netzwerk (Training + Inference) | Schreibt: via Layers |
| `KANAdapter` | `adapter/kan_adapter.hpp` | KAN↔Brain19 Interface, Modul-Lifecycle | Schreibt: Module-Map |
| `FunctionHypothesis` | `kan/function_hypothesis.hpp` | KAN-Ergebnis mit Provenance + shared_ptr<KANModule> | Datenstruktur |
| `DataPoint` | `kan/kan_module.hpp` | (input, output) Trainingsdatum | Datenstruktur |
| `KanTrainingConfig` | `kan/kan_module.hpp` | Lernrate, Epochen, Regularisierung | Config |

### 2.5 Understanding Layer

| Klasse/Struct | Datei | Verantwortung | Zugriff |
|---|---|---|---|
| `UnderstandingLayer` | `understanding/understanding_layer.hpp` | Semantische Analyse via MiniLLMs | Liest: STM, LTM; Schreibt: Proposals |
| `MiniLLM` | `understanding/mini_llm.hpp` | Abstrakte LLM-Schnittstelle | Interface |
| `StubMiniLLM` | `understanding/mini_llm.hpp` | Deterministischer Stub für Tests | Implementation |
| `OllamaMiniLLM` | `understanding/ollama_mini_llm.hpp` | Ollama-basierte Implementation | Implementation |
| `MiniLLMFactory` | `understanding/mini_llm_factory.hpp` | Factory für spezialisierte MiniLLMs | Factory |
| `HypothesisProposal` | `understanding/understanding_proposals.hpp` | LLM-generierte Hypothese | Datenstruktur |
| `AnalogyProposal` | `understanding/understanding_proposals.hpp` | Analogie-Vorschlag | Datenstruktur |
| `ContradictionProposal` | `understanding/understanding_proposals.hpp` | Widerspruchs-Erkennung | Datenstruktur |
| `MeaningProposal` | `understanding/understanding_proposals.hpp` | Bedeutungs-Extraktion | Datenstruktur |

### 2.6 Ingestor Pipeline

| Klasse/Struct | Datei | Verantwortung | Zugriff |
|---|---|---|---|
| `IngestionPipeline` | `ingestor/ingestion_pipeline.hpp` | Orchestrierung: Text → LTM | Schreibt: LTM (via commit) |
| `KnowledgeIngestor` | `ingestor/knowledge_ingestor.hpp` | JSON-Parsing → StructuredInput | Transformer |
| `TextChunker` | `ingestor/text_chunker.hpp` | Text → Chunks (konfigurierbar) | Transformer |
| `EntityExtractor` | `ingestor/entity_extractor.hpp` | Text → Entities (Regex+Heuristik) | Transformer |
| `RelationExtractor` | `ingestor/relation_extractor.hpp` | Text → Relationen (Pattern-basiert) | Transformer |
| `TrustTagger` | `ingestor/trust_tagger.hpp` | Source → Trust-Kategorie → Trust-Score | Transformer |
| `ProposalQueue` | `ingestor/proposal_queue.hpp` | Puffer zwischen Ingestion und LTM-Commit | Queue |
| `IngestProposal` | `ingestor/proposal_queue.hpp` | Einzelner Ingestion-Vorschlag mit Status | Datenstruktur |

### 2.7 Importers & LLM

| Klasse/Struct | Datei | Verantwortung | Zugriff |
|---|---|---|---|
| `WikipediaImporter` | `importers/wikipedia_importer.hpp` | URL/Text → KnowledgeProposal | Transformer |
| `ScholarImporter` | `importers/scholar_importer.hpp` | DOI/URL/Text → KnowledgeProposal | Transformer |
| `KnowledgeProposal` | `importers/knowledge_proposal.hpp` | Strukturierter Import-Vorschlag | Datenstruktur |
| `OllamaClient` | `llm/ollama_client.hpp` | HTTP-Client für Ollama API | Netzwerk |
| `ChatInterface` | `llm/chat_interface.hpp` | Chat-basierte Interaktion mit Brain19 | Schreibt: STM, LTM |
| `SnapshotGenerator` | `snapshot_generator.hpp` | JSON-Export des Systemzustands | Read-Only |

### 2.8 Enums

| Enum | Datei | Werte |
|---|---|---|
| `EpistemicType` | `epistemic/epistemic_metadata.hpp` | FACT, THEORY, HYPOTHESIS, SPECULATION, DEFINITION, META |
| `EpistemicStatus` | `epistemic/epistemic_metadata.hpp` | ACTIVE, DEPRECATED, INVALIDATED, UNDER_REVIEW |
| `ActivationLevel` | `memory/activation_level.hpp` | INACTIVE, LOW, MODERATE, HIGH, PEAK |
| `ActivationClass` | `memory/activation_level.hpp` | CORE_KNOWLEDGE, CONTEXTUAL |
| `RelationType` | `memory/active_relation.hpp` | IS_A, HAS_PROPERTY, CAUSES, RELATED_TO, ... |
| `TriggerType` | `curiosity/curiosity_trigger.hpp` | SHALLOW_RELATIONS, LOW_CONNECTIVITY, HIGH_UNCERTAINTY, MISSING_DEPTH, RECURRENT_WITHOUT_FUNCTION |
| `OverlayMode` | `micromodel/relevance_map.hpp` | MULTIPLY, HARMONIC_MEAN, SURPRISE, WEIGHTED_AVERAGE |
| `ProposalStatus` | `ingestor/proposal_queue.hpp` | PENDING, APPROVED, REJECTED, COMMITTED |
| `SourceType` | `importers/knowledge_proposal.hpp` | WIKIPEDIA, SCHOLAR, MANUAL, ... |

---

## 3. Datenfluss-Diagramme

### 3.1 Input → Ingestion → LTM

```
User Input (Text/URL/JSON)
      │
      ▼
┌─────────────────┐     ┌──────────────────┐
│ WikipediaImporter│     │ KnowledgeIngestor│
│ ScholarImporter  │     │ (JSON-Parser)    │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
   KnowledgeProposal      StructuredInput
         │                       │
         └───────┬───────────────┘
                 ▼
      ┌──────────────────┐
      │ IngestionPipeline │
      │                  │
      │  1. TextChunker  │  Text → Chunks (overlap-aware)
      │  2. EntityExtract│  Chunks → Entities (Regex)
      │  3. RelationExtr │  Chunks → Relations (Pattern)
      │  4. TrustTagger  │  Source → Trust-Score
      │  5. ProposalQueue│  Puffer (PENDING)
      └────────┬─────────┘
               │ commit()
               ▼
      ┌──────────────────┐
      │   LongTermMemory │
      │                  │
      │  add_concept()   │  ← EpistemicMetadata PFLICHT
      │  add_relation()  │  ← Trust von TrustTagger
      └──────────────────┘
```

### 3.2 Spreading Activation → Salience → Focus

```
Trigger: Konzept wird aktiviert (STM)
      │
      ▼
┌─────────────────────────────────────┐
│ CognitiveDynamics::spread_activation│
│                                     │
│  1. Source-Konzept holen            │
│  2. Outgoing Relations aus LTM     │
│  3. Für jede Relation:             │
│     activation *= relation.trust   │
│     activation *= decay_per_depth  │
│  4. Rekursiv (max_depth=3)         │
│  5. min_relevance Threshold        │
│  6. STM: activate_concept()        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ CognitiveDynamics::compute_salience │
│                                     │
│  score = w_freq  × frequency_norm   │  (0.4)
│        + w_rec   × recency_factor   │  (0.0 — BUG!)
│        + w_conn  × connectivity     │  (0.2)
│        + w_epist × epistemic_trust  │  (0.3)
│                                     │
│  Batch: relativ normalisiert        │
│  Single: absolut (inkonsistent!)    │
└────────────────┬────────────────────┘
                 │ Top-K
                 ▼
┌─────────────────────────────────────┐
│ CognitiveDynamics::update_focus     │
│                                     │
│  focus_sets_[context] ← Top-K nach  │
│  Salience-Score                     │
│  Focus-Decay pro Tick               │
│  Promotion/Demotion                 │
└─────────────────────────────────────┘
```

### 3.3 Understanding Cycle

```
┌────────────────────────────────────┐
│ UnderstandingLayer::               │
│   perform_understanding_cycle()    │
│                                    │
│  1. Aktive Konzepte aus STM holen  │
│     (BUG: IDs werden erfunden!)    │
│                                    │
│  2. Für jede registrierte MiniLLM: │
│     ┌─────────────────────────┐    │
│     │ MiniLLM::generate()     │    │
│     │ (Stub oder Ollama)      │    │
│     └──────────┬──────────────┘    │
│                │                   │
│  3. Response parsen → Proposals    │
│     ┌──────────────────────────┐   │
│     │ HypothesisProposal       │   │
│     │ AnalogyProposal          │   │
│     │ ContradictionProposal    │   │
│     │ MeaningProposal          │   │
│     └──────────┬───────────────┘   │
│                │                   │
│  4. Trust-Ceiling erzwingen        │
│     max_trust = 0.3-0.5           │
│                                    │
│  5. → ProposalQueue oder direkt    │
│       in LTM (nach Review)         │
└────────────────────────────────────┘
```

### 3.4 MicroModel Training + Inference

```
TRAINING:
┌────────────────────────────────────┐
│ MicroTrainer::train()              │
│                                    │
│  Für jedes Konzept im LTM:        │
│  1. TrainingSamples generieren     │
│     - Positive: echte Relationen   │
│     - Negative: zufällige Paare    │
│                                    │
│  2. EmbeddingManager:              │
│     e = get_relation_embedding()   │  10D Vektor
│     c = get_context_embedding()    │  10D Vektor
│                                    │
│  3. MicroModel::train_step()       │
│     v = W·c + b                    │  ℝ¹⁰
│     z = eᵀ·v                      │  ℝ¹ (Skalar)
│     pred = σ(z)                    │  (0,1)
│     loss = BCE(pred, target)       │
│     Adam-Update(W, b, e, c)       │
│                                    │
│  4. → MicroModelRegistry           │
└────────────────────────────────────┘

INFERENCE (RelevanceMap):
┌────────────────────────────────────┐
│ RelevanceMap::compute(source)      │
│                                    │
│  Für jedes target ≠ source:        │
│  1. model = registry.get(source)   │
│  2. score = model->predict(e, c)   │
│     (BUG: target wird ignoriert!)  │
│  3. scores_[target] = score        │
│                                    │
│ RelevanceMap::combine(maps, mode)  │
│  MULTIPLY: ∏ scores               │
│  HARMONIC: 2·a·b/(a+b)            │
│  SURPRISE: |a-b|·max(a,b)         │
│  WEIGHTED_AVG: Σ wᵢ·scoreᵢ       │
└────────────────────────────────────┘
```

### 3.5 Curiosity Trigger Flow

```
┌────────────────────────────────────┐
│ CuriosityEngine::analyze()         │
│                                    │
│  Input: SystemObservation          │
│    - active_concepts (aus STM)     │
│    - ltm_stats                     │
│    - spreading_stats               │
│                                    │
│  Checks:                           │
│  ┌──────────────────────────────┐  │
│  │ 1. SHALLOW_RELATIONS         │  │
│  │    Konzept hat < 3 Relations │──│──→ CuriosityTrigger
│  │                              │  │    (priority, concept_ids)
│  │ 2. LOW_CONNECTIVITY          │  │
│  │    Isolierte Subgraphen      │──│──→ CuriosityTrigger
│  │                              │  │
│  │ 3. HIGH_UNCERTAINTY          │  │
│  │    Trust < threshold         │──│──→ CuriosityTrigger
│  │                              │  │
│  │ 4. MISSING_DEPTH (TODO)      │  │
│  │ 5. RECURRENT_W/O_FUNC (TODO)│  │
│  └──────────────────────────────┘  │
│                                    │
│  Output: vector<CuriosityTrigger>  │
│    → Kann MicroModel-Overlay       │
│      triggern (Kreativität)        │
│    → Kann Ingestion triggern       │
│    → Kann Understanding triggern   │
└────────────────────────────────────┘
```

---

## 4. Dependency-Graph

```
                    common/types.hpp
                    (ConceptId, ContextId, RelationId)
                          │
          ┌───────────────┼───────────────────────┐
          │               │                       │
          ▼               ▼                       ▼
    ┌───────────┐   ┌──────────┐          ┌──────────────┐
    │ epistemic/│   │ memory/  │          │ ltm/         │
    │ epistemic_│   │ stm.hpp  │          │ long_term_   │
    │ metadata  │   │ stm_entry│          │ memory.hpp   │
    │           │   │ active_  │          │ relation.hpp │
    │           │   │ relation │          │              │
    └─────┬─────┘   └────┬─────┘          └──────┬───────┘
          │               │                       │
          └───────┬───────┴───────────────────────┘
                  │
                  ▼
          ┌──────────────────┐
          │ memory/           │
          │ brain_controller  │─────────────┐
          └────────┬─────────┘              │
                   │                        │
       ┌───────────┼────────────┐           │
       │           │            │           │
       ▼           ▼            ▼           ▼
┌───────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐
│ cognitive/│ │curiosity/│ │snapshot_│ │micromodel/  │
│ cognitive │ │curiosity │ │generator│ │micro_model  │
│ _dynamics │ │_engine   │ │         │ │embedding_mgr│
│           │ │          │ │         │ │relevance_map│
│           │ │          │ │         │ │micro_trainer│
│           │ │          │ │         │ │registry     │
└───────────┘ └─────────┘ └─────────┘ └──────┬──────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │ kan/          │
                                       │ kan_module    │
                                       │ kan_layer     │
                                       │ kan_node      │
                                       └──────┬───────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │ adapter/      │
                                       │ kan_adapter   │
                                       └──────────────┘

┌──────────────┐     ┌───────────────┐
│understanding/│────→│ llm/          │
│understanding │     │ ollama_client │
│_layer        │     │ chat_interface│
│mini_llm*     │     └───────────────┘
└──────────────┘

┌──────────────┐
│ ingestor/    │────→ LTM, EpistemicMetadata
│ ingestion_   │
│ pipeline     │
│ text_chunker │
│ entity_extr  │
│ relation_extr│
│ trust_tagger │
│ proposal_q   │
└──────────────┘

┌──────────────┐
│ importers/   │────→ Ingestor Pipeline
│ wikipedia_   │
│ scholar_     │
└──────────────┘
```

### Abhängigkeitsrichtung (vereinfacht):

```
types.hpp ← epistemic ← ltm ← brain_controller ← cognitive_dynamics
                                      ↑                    ↑
                                      │                    │
                                    stm ←──────────────────┘
                                      ↑
                               curiosity_engine
                               snapshot_generator

micromodel ← kan ← kan_adapter
understanding ← llm/ollama
ingestor ← ltm + epistemic
importers ← ingestor
```

---

## 5. Ownership-Map

```
BrainController
 └── unique_ptr<ShortTermMemory> stm_        ← EXKLUSIV

KANModule
 └── vector<unique_ptr<KANLayer>> layers_    ← EXKLUSIV
      └── vector<unique_ptr<KANNode>> nodes_ ← EXKLUSIV

KANAdapter
 └── map<id, KANModuleEntry>
      └── shared_ptr<KANModule> module       ← ⚠️ PROBLEM: No-Op Deleter!

FunctionHypothesis
 └── shared_ptr<KANModule> module            ← Shared mit KANAdapter

UnderstandingLayer
 └── vector<unique_ptr<MiniLLM>> mini_llms_  ← EXKLUSIV

ChatInterface
 └── unique_ptr<OllamaClient> ollama_        ← EXKLUSIV

WikipediaImporter / ScholarImporter
 └── return unique_ptr<KnowledgeProposal>    ← Transfer zum Aufrufer

LongTermMemory
 └── unordered_map<ConceptId, ConceptInfo>   ← Wert-Semantik (kein Pointer)
 └── unordered_map<RelationId, RelationInfo> ← Wert-Semantik

CognitiveDynamics
 └── Referenzen auf STM, LTM                ← NICHT-BESITZEND
 └── unordered_map<ContextId, vector<FocusEntry>> ← Wert-Semantik

MicroModelRegistry
 └── unordered_map<ConceptId, MicroModel>    ← Wert-Semantik

EmbeddingManager
 └── unordered_map<string, vector<double>>   ← Wert-Semantik (Cache)
```

### Ownership-Probleme:

1. **KANAdapter → FunctionHypothesis:** `shared_ptr` mit No-Op Deleter auf `unique_ptr`-verwaltetem Speicher → Dangling Pointer wenn Adapter zerstört wird (BUG-H8)
2. **CognitiveDynamics:** Hält Referenzen auf STM/LTM ohne Lifetime-Garantie → Aufrufer muss sicherstellen
3. **EmbeddingManager:** Cache wächst unbegrenzt (BUG-4: Embedding-Leak bei RelevanceMap)

---

## 6. Epistemischer Fluss

```
Wissensquelle                    Trust-Zuweisung              LTM-Eintrag
─────────────                    ───────────────              ──────────
Wikipedia-Import ──→ TrustTagger ──→ THEORY (0.85-0.95) ──→ ConceptInfo
Scholar-Import   ──→ TrustTagger ──→ THEORY (0.90-0.95) ──→   + EpistemicMetadata
Manual Input     ──→ TrustTagger ──→ FACT   (0.95-0.99) ──→     (Type, Status,
LLM-Hypothesis   ──→ Trust-Ceil  ──→ HYPOTHESIS (≤0.50) ──→      Trust, Source)
LLM-Speculation  ──→ Trust-Ceil  ──→ SPECULATION (≤0.30) ──→

Trust-Propagation durch Spreading Activation:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Konzept A (FACT, trust=0.98)                           │
│       │                                                  │
│       │ Relation (trust=0.90)                           │
│       ▼                                                  │
│  Konzept B (THEORY, trust=0.92)                         │
│       │                                                  │
│       │ Relation (trust=0.70)                           │
│       ▼                                                  │
│  Konzept C (HYPOTHESIS, trust=0.45)                     │
│                                                          │
│  Activation(C) = source_act × relation_trust × decay    │
│                = 0.98 × 0.90 × 0.70 × decay_factor     │
│                                                          │
│  → Trust degradiert entlang der Kette                   │
│  → HYPOTHESIS kann nie zu FACT promoviert werden        │
│    durch bloße Spreading Activation                     │
│                                                          │
│  Salience-Einfluss:                                      │
│  epistemic_weight (0.3) × trust_score                   │
│  → Hochvertrauenswürdige Konzepte haben höhere Salience │
│  → Fokus bevorzugt FACT/THEORY über HYPOTHESIS          │
└──────────────────────────────────────────────────────────┘

Status-Transitionen:
  ACTIVE ──→ DEPRECATED (veraltet)
  ACTIVE ──→ INVALIDATED (widerlegt)
  ACTIVE ──→ UNDER_REVIEW (wird geprüft)
  UNDER_REVIEW ──→ ACTIVE (bestätigt)
  UNDER_REVIEW ──→ INVALIDATED (widerlegt)
  
  ⚠️ INVALIDATED → ACTIVE ist möglich aber nicht validiert (BUG-M2)
  ⚠️ INVALIDATED + trust ≥ 0.2 → assert(false) Crash (BUG-M1)
```

---

# TEIL 2: PROFESSOR-REVIEW

---

## A) Stärken

### A1: Epistemische Integrität als Compile-Time-Garantie
Die Entscheidung, `ConceptInfo() = delete` zu machen, ist **hervorragend**. Es ist strukturell unmöglich, Wissen ohne epistemische Klassifikation ins System zu bringen. Das ist besser als jede Runtime-Validierung — der Compiler wird zum epistemischen Wächter. In der gesamten kognitiven Architektur-Literatur (SOAR, ACT-R, OpenCog) gibt es nichts Vergleichbares.

### A2: Klare Separation of Concerns
Die 9 Subsysteme haben saubere Verantwortlichkeiten. Besonders gut:
- **CuriosityEngine ist Read-Only** — sie beobachtet und signalisiert, modifiziert aber nichts
- **ProposalQueue** als Puffer zwischen Ingestion und LTM verhindert unkontrollierte Writes
- **Understanding Layer ist optional** — das System funktioniert ohne LLM

### A3: MicroModel-Philosophie "Einfache Teile, komplexe Komposition"
Die mathematische Analyse zeigt: Ein einzelnes MicroModel ist ein linearer Klassifikator (11 effektive Parameter). Aber K MicroModels + Kombinationsschicht = universeller Approximator (Cybenko, 1989). Das ist ein elegantes Designprinzip mit solider theoretischer Fundierung.

### A4: Spreading Activation löst das Skalierungsproblem
O(K·D) statt O(N²) durch gezielte Aktivierungsausbreitung. Bei 100K Konzepten: 40.000× Speedup gegenüber Brute-Force. Das ist der richtige algorithmische Ansatz.

### A5: "Mechanik statt Magie"
Die konsequente Philosophie, dass das LLM nur Verbalizer ist und nie im Denkpfad sitzt, ist architektonisch mutig und korrekt. Brain19 kann nicht halluzinieren — das ist ein echter Differentiator.

---

## B) Schwächen & Risiken

### B1: KRITISCH — RelevanceMap ist funktional kaputt
`RelevanceMap::compute()` ignoriert das Target-Konzept (BUG-C1). Das bedeutet: **Die gesamte MicroModel-Inferenz liefert identische Scores für alle Targets.** Kreativität durch Map-Überlagerung ist damit wertlos. Dies ist der schwerwiegendste Bug — er unterminiert das Kernversprechen der Architektur.

**Code-Referenz:** `micromodel/relevance_map.cpp:24-28`

### B2: KRITISCH — Salience-Berechnung systematisch falsch
- `recency_weight = 0.0` → Weights summieren auf 0.9 statt 1.0
- `compute_recency_factor()` returniert hardcoded 0.5
- Single vs. Batch Salience inkonsistent
- Root Cause: `update_access_time()` nie implementiert

**Code-Referenz:** `cognitive/cognitive_dynamics.cpp:232-236`, `cognitive/cognitive_config.hpp`

### B3: HOCH — Null Thread-Safety
Kein einziger Mutex, kein atomic, keine Synchronisation. Jeder zukünftige Multi-Threading-Versuch (Phase 3-5 der Roadmap) baut auf einem System auf, das bei concurrent Access sofort UB produziert. Die Roadmap plant Thread-Safety als Phase 3 — aber die mentale Last wird unterschätzt.

### B4: HOCH — W→e Parameterredundanz (91%)
Die mathematische Analyse beweist: Von 110 Parametern in W und e sind nur 11 funktional unabhängig. Bei 100K Konzepten: 396 MB verschwendeter RAM. Das ist kein Bug, aber eine architektonische Schuld, die bei Skalierung teuer wird.

**Empfehlung:** Prüfen ob der Zwischenvektor v = W·c + b extern genutzt wird. Falls nein → Reduktion auf (a, β).

### B5: MITTEL — KAN ist architektonisch limitiert
- Nur 1-Schicht möglich (KANLayer hat n_in Nodes mit je 1 Output)
- const_cast UB in gradient() (BUG-H6)
- Numerische statt analytische Gradienten (O(n²) statt O(n))
- Rand-Bug bei x=1.0

KAN soll in Phase 7 das Validierungs-Backbone werden. In seinem jetzigen Zustand ist es dafür nicht bereit.

### B6: MITTEL — Snapshot zeigt nur 50% des Systems
Relations immer leer, KAN/MicroModels/Ingestor/CognitiveDynamics fehlen. Debugging und Monitoring sind damit stark eingeschränkt.

### B7: NIEDRIG — Understanding Layer arbeitet mit erfundenen IDs
`perform_understanding_cycle()` erzeugt Concept-IDs als 1..N statt echte LTM-IDs. Alle Proposals referenzieren nicht-existente Konzepte.

**Code-Referenz:** `understanding/understanding_layer.cpp:161-170`

### B8: Architektonische Schulden-Zusammenfassung

| Schuld | Schwere | Aufwand | Blockiert |
|--------|---------|---------|-----------|
| RelevanceMap kaputt | 🔴 | 1 Abend | Kreativität, MicroModel-Nutzen |
| Salience falsch | 🔴 | 3h | Focus Management, Korrekte Priorisierung |
| Keine Thread-Safety | 🟠 | 2 Wochen | Phase 3-5 der Roadmap |
| W→e Redundanz | 🟡 | 1 Woche | Skalierung auf 100K+ |
| KAN 1-Schicht | 🟡 | 2 Abende | Phase 7 (KAN-LLM Hybrid) |
| Snapshot unvollständig | 🟡 | 1 Abend | Debugging |
| Understanding IDs | 🟡 | 1 Abend | Understanding Cycle |

---

## C) Roadmap-Validierung

### Gesamtbewertung: Gut strukturiert, realistisch priorisiert (7/10)

### C1: Reihenfolge ist sinnvoll ✅
Phase 0 (Stabilisierung) → Phase 1 (Persistence) → Phase 2 (Snapshot+KAN) ist korrekt. "Persistence before Performance" ist die richtige Entscheidung für einen Solo-Entwickler. Der Dependency-Graph ist schlüssig.

### C2: Phase 0 wird unterschätzt ⚠️
"3-5 Abende" für 5 Bugfix-Gruppen klingt optimistisch. Der Salience-Cluster (Bugs 1+2+3+9) allein ist 3h, aber nur wenn alles glatt geht. Realistischer: **1-2 Wochen** mit Tests und Debugging.

### C3: Phase 1 (Persistence) hat versteckte Komplexität ⚠️
mmap-basierte Persistence mit WAL ist kein Anfängerprojekt. Die Persistent Memory Architecture Docs sind gut, aber:
- **StringPool** für Labels ist ein eigenes Projekt (Fragmentation, Compaction)
- **WAL Recovery** braucht extensive Crash-Tests
- **Dual-Mode LTM** (Heap/Persistent) verdoppelt die Testmatrix

**Empfehlung:** Zuerst einfache File-basierte Serialisierung (JSON/MessagePack), dann mmap als Optimierung. Das halbiert Phase 1 auf 1-2 Wochen und gibt sofort Persistence.

### C4: Phase 3 (Thread-Safety) fehlt ein Schritt ⚠️
Die Roadmap plant "Shared-State Wrappers" als Adapter-Pattern. Das ist konzeptionell richtig, aber es fehlt:
- **Immutable State Pattern** für LTM-Reads (Copy-on-Write)
- **Message-Passing** als Alternative zu shared_mutex
- **Benchmarking** — shared_mutex ist auf vielen Workloads langsamer als Kopien

### C5: Phase 5 (Multi-Stream) vor Phase 6 (Dynamic Concepts) ✅
Richtig. Multi-Stream ist infrastrukturell, Dynamic Concepts ist Feature. Infrastruktur vor Features.

### C6: Phase 7 (KAN-LLM Hybrid) Timeline ist optimistisch ⚠️
"6-8 Wochen" für bidirektionalen KAN-LLM-Dialog ist Research, nicht Engineering. In der Forschung dauern solche Iterationen typischerweise 3-6 Monate. **Empfehlung:** Phase 7 als offenes Forschungsprojekt ohne feste Timeline betrachten.

### C7: Fehlende Schritte

1. **Benchmarking-Framework** — Nirgends eingeplant. Ohne Benchmarks keine validierbaren Performance-Claims.
2. **Regressions-Tests** — Phase 0 hat Tests pro Bug, aber kein Gesamt-Regressions-Setup (CI/CD).
3. **Monitoring/Logging** — SnapshotGenerator reicht nicht für Produktivbetrieb. Strukturiertes Logging fehlt.
4. **Error Handling** — Kein `try/catch` in den Demos, keine Error-Recovery-Strategie.

---

## D) Fehlende Erweiterungen

### D1: Persistence (mmap)

**Andockpunkt:** `LongTermMemory` hat bereits klar definierte Datenstrukturen:
```
concepts_: unordered_map<ConceptId, ConceptInfo>
relations_: unordered_map<RelationId, RelationInfo>
outgoing_relations_: unordered_map<ConceptId, vector<RelationId>>
```

**Problem:** `unordered_map` ist nicht mmap-kompatibel (Heap-Pointer in Buckets). Alternativen:
- **Flat HashMap** (robin_map, absl::flat_hash_map) → mmap-fähig mit Einschränkungen
- **Custom Arena Allocator** → `PersistentStore<T>` aus der Persistent Memory Architecture
- **SQLite als Zwischenschritt** → Sofortige Persistence, später mmap

**Empfehlung:** SQLite/LMDB für Phase 1 (1 Woche), mmap als Phase 1.5 (2-3 Wochen). Das gibt sofort Persistence ohne die StringPool-Komplexität.

### D2: Multi-Threading — Engpässe

| Subsystem | Art | Engpass |
|---|---|---|
| LTM | Read-heavy | shared_mutex OK, aber Lock-Contention bei vielen Readern |
| STM | Read+Write | Per-Context-Mutex möglich (Contexts sind unabhängig) |
| CognitiveDynamics | Write-heavy (STM) | Spreading Activation modifiziert STM → Bottleneck |
| MicroModelRegistry | Read-heavy | Inference ist embarrassingly parallel |
| EmbeddingManager | Read+Write (Cache) | concurrent_unordered_map oder Lock-Free Cache |
| KAN Training | CPU-bound | Embarrassingly parallel pro Modul |

**Hauptengpass:** `CognitiveDynamics::spread_activation()` liest LTM und schreibt STM. Bei Multi-Stream teilen sich alle Streams denselben STM → Serialisierung. **Lösung:** STM per Stream kopieren, am Ende mergen (wie in Multi-Stream Architecture Docs beschrieben).

### D3: KAN-LLM Hybrid — Andockpunkte

```
Aktuell:                          Ziel:
┌──────────┐                     ┌──────────┐
│MicroModel│──→ Relevanz         │MicroModel│──→ Relevanz
└──────────┘                     └──────────┘
                                       │
                                       ▼
                                 ┌──────────┐
                                 │ KAN      │──→ Funktions-Validierung
                                 │ (Multi-  │     (B-Spline → inspizierbar)
                                 │  Layer)  │
                                 └──────────┘
                                       │
                                       ▼
                                 ┌──────────┐
                                 │ LLM      │──→ Hypothesen-Generierung
                                 │(Ollama)  │     (kreativ aber unvalidiert)
                                 └──────────┘
                                       │
                                       ▼
                                 ┌──────────────────┐
                                 │ EpistemicBridge   │
                                 │ KAN-MSE → Trust   │
                                 │ Convergence → Type│
                                 └──────────────────┘
```

**Andockpunkte:**
1. `KANAdapter` → erweitern um `EpistemicBridge`
2. `UnderstandingLayer` → Hypothesen an KAN zur Validierung weiterleiten
3. `FunctionHypothesis` → um epistemische Metriken erweitern (MSE → Trust)
4. Neues Interface: `HypothesisTranslator` (LLM-Text → KAN-Trainingsproblem)

### D4: Skalierung auf 100K+ Konzepte — Was bricht

| Komponente | Bei 100K | Problem | Lösung |
|---|---|---|---|
| LTM (Heap) | ~50 MB | OK | - |
| MicroModels (130P×4B) | 52 MB (voll) / 8.4 MB (reduziert) | RAM bei Redundanz | W→e Reduktion |
| Embeddings (Cache) | Unbegrenzt wachsend | OOM | LRU-Cache + Eviction |
| Spreading Activation | O(50³) = 125K Evals | OK | - |
| RelevanceMap (alle) | 100K × 100K Paare | Unmöglich | Spreading statt Brute-Force |
| STM (kein GC) | Unbegrenzt wachsend | OOM | GC implementieren (BUG-11) |
| Salience (Batch) | 100K Scores sortieren | O(N log N) | Top-K statt Sort-All |
| KAN Training | 100K Module × Epochen | CPU-bound | Parallelisierung (Phase 5) |

**Was zuerst bricht:** EmbeddingManager-Cache (kein Eviction) und STM (kein GC). Beide sind einfache Fixes.

---

## E) Architektonische Empfehlungen (priorisiert, 3-6 Monate)

### Priorität 1: Fundament reparieren (Monat 1) 🔴

1. **RelevanceMap::compute() fixen** — Target-Embedding einbauen. Ohne das ist das Kernfeature nutzlos.
   - Aufwand: 1 Abend
   - Impact: Schaltet MicroModel-Inferenz und Kreativität frei

2. **Salience-Cluster fixen** (Bugs 1+2+3+9) — update_access_time() implementieren, Weights auf 1.0
   - Aufwand: 3h
   - Impact: Korrekte Priorisierung, valides Focus Management

3. **Build-System reparieren** — Alle Demos kompilierbar
   - Aufwand: 30 Min
   - Impact: Testbarkeit

### Priorität 2: Persistence (Monat 1-2) 🟠

4. **Einfache Persistence zuerst** — SQLite oder JSON-Serialisierung statt mmap
   - Aufwand: 1 Woche
   - Impact: Brain19 überlebt Restarts → Game-Changer für Entwicklung

5. **mmap-Optimierung** — Wenn SQLite-Persistence funktioniert
   - Aufwand: 2-3 Wochen
   - Impact: Performance für 100K+ Konzepte

### Priorität 3: Korrektheit (Monat 2-3) 🟡

6. **KAN: Multi-Layer + analytische Gradienten** — Voraussetzung für Phase 7
   - Aufwand: 2 Abende
   - Impact: KAN wird nutzbar für echte Funktionsapproximation

7. **W→e Redundanz klären** — Prüfen ob Zwischenvektor v extern genutzt wird, ggf. reduzieren
   - Aufwand: 1 Abend (Analyse) + 1 Woche (Refactoring)
   - Impact: 84% RAM-Ersparnis bei MicroModels

8. **Snapshot vervollständigen** + Structured Logging einführen
   - Aufwand: 1 Abend + 2 Abende
   - Impact: Debugging wird möglich

### Priorität 4: Skalierung vorbereiten (Monat 3-4) 🟢

9. **STM Garbage Collection** — Konzepte unter Threshold entfernen
10. **EmbeddingManager LRU-Cache** — Begrenzung + Eviction
11. **Benchmarking-Framework** — Reproduzierbare Performance-Tests
12. **Regressions-Test-Suite** — make test, CI-fähig

### Priorität 5: Multi-Threading Fundament (Monat 4-6) 🟢

13. **Thread-Safety per Adapter-Pattern** (wie in Roadmap Phase 3 geplant)
14. **Aber zuerst:** Immutable-Read-Path für LTM (Copy-on-Write oder Snapshot-Isolation)
15. **Benchmarking vor Optimierung** — Messen wo die echten Bottlenecks sind

### Anti-Empfehlungen ❌

- **NICHT** Phase 7 (KAN-LLM Hybrid) anfangen bevor Phase 0-2 fertig sind
- **NICHT** mmap vor einfacher Persistence implementieren
- **NICHT** Multi-Threading einführen bevor Single-Threaded korrekt funktioniert
- **NICHT** auf 100K Konzepte skalieren bevor die Grundalgorithmen korrekt sind

---

## Schlusswort

Brain19 ist ein ambitioniertes und architektonisch durchdachtes Projekt. Die Kernidee — epistemisch integre, selbständig denkende kognitive Architektur mit MicroModels statt LLM — ist originell und theoretisch fundiert. Die größte Gefahr ist nicht die Architektur selbst, sondern die Diskrepanz zwischen dem, was die Dokumentation verspricht, und dem, was der Code aktuell liefert. RelevanceMap, Salience und Understanding Layer sind funktional kaputt — das Fundament muss repariert werden, bevor die beeindruckende Roadmap Sinn ergibt.

Die ADHS-optimierte Roadmap mit kleinen, pushbaren Paketen ist klug. Der wichtigste Rat: **Phase 0 fertigmachen, bevor irgendetwas anderes passiert.** Jeder Fix dort hat sofortige, sichtbare Auswirkung — das ist der Dopamin-Hit, der die Motivation für die nächsten 6 Monate liefert.

---

*Generiert 2026-02-10 | Brain19 Architecture Complete v1.0*  
*Basierend auf: Code-Audit, ROADMAP.md, MATHEMATICAL_ANALYSIS.md, FIX_PLAN.md, TECHNICAL_ANALYSIS.md, ARCHITECTURE_OVERVIEW.md*
