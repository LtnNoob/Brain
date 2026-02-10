# Brain19 – Vollständige Architektur-Dokumentation

> Automatisch generiert aus dem Quellcode in `backend/`. Nur dokumentiert was im Code steht.

---

## Inhaltsverzeichnis

1. [Klassen-Katalog](#1-klassen-katalog)
2. [Subsystem-Map](#2-subsystem-map)
3. [Datenfluss-Diagramme](#3-datenfluss-diagramme)
4. [Ownership & Lifecycle](#4-ownership--lifecycle)
5. [Epistemischer Fluss](#5-epistemischer-fluss)
6. [Dependency Graph](#6-dependency-graph)

---

## 1. Klassen-Katalog

### 1.1 common/

| Klasse/Typedef | File | Verantwortung |
|---|---|---|
| `ConceptId` (uint64_t) | `common/types.hpp` | Eindeutige Konzept-ID |
| `ContextId` (uint64_t) | `common/types.hpp` | Eindeutige Kontext-ID |
| `RelationId` (uint64_t) | `common/types.hpp` | Eindeutige Relations-ID |

---

### 1.2 epistemic/

#### `EpistemicMetadata` — `epistemic/epistemic_metadata.hpp`
- **Verantwortung:** Pflicht-Metadaten für jedes Wissenselement. Erzwingt epistemische Explizitheit.
- **Key Members:** `EpistemicType type`, `EpistemicStatus status`, `double trust` [0.0, 1.0]
- **Key Methods:** `is_valid()`, `is_active()`, `is_invalidated()`, `is_superseded()`, `is_contextual()`
- **Constraints:** Default-Konstruktor gelöscht. Trust wird im Konstruktor validiert. INVALIDATED + trust≥0.2 löst Debug-Assert aus.
- **Dependencies:** Keine.

#### `EpistemicType` (enum class)
- Werte: `FACT`, `DEFINITION`, `THEORY`, `HYPOTHESIS`, `INFERENCE`, `SPECULATION`
- Kein `UNKNOWN` — Abwesenheit ist ein Compile-Error.

#### `EpistemicStatus` (enum class)
- Werte: `ACTIVE`, `CONTEXTUAL`, `SUPERSEDED`, `INVALIDATED`
- Kein `UNKNOWN`.

---

### 1.3 ltm/ (Long-Term Memory)

#### `ConceptInfo` — `ltm/long_term_memory.hpp`
- **Verantwortung:** Wissenselement mit obligatorischer epistemischer Metadaten.
- **Key Members:** `ConceptId id`, `string label`, `string definition`, `EpistemicMetadata epistemic`
- **Constraints:** Default-Konstruktor gelöscht. Epistemic Metadata pflicht bei Konstruktion.
- **Dependencies:** `EpistemicMetadata`

#### `LongTermMemory` — `ltm/long_term_memory.hpp/.cpp`
- **Verantwortung:** Persistenter Wissensgraph. Speichert Konzepte und Relationen.
- **Key Members:**
  - `unordered_map<ConceptId, ConceptInfo> concepts_`
  - `unordered_map<RelationId, RelationInfo> relations_`
  - `unordered_map<ConceptId, vector<RelationId>> outgoing_relations_`, `incoming_relations_`
- **Key Methods:**
  - `store_concept(label, definition, EpistemicMetadata)` → `ConceptId` — **erfordert explizite Epistemic Metadata**
  - `retrieve_concept(id)` → `optional<ConceptInfo>`
  - `update_epistemic_metadata(id, new_metadata)` — einziger Weg Status zu ändern
  - `invalidate_concept(id, trust=0.05)` — setzt Status auf INVALIDATED, löscht NICHT
  - `add_relation(source, target, type, weight)` → `RelationId`
  - `get_outgoing_relations(source)`, `get_incoming_relations(target)`
  - `get_concepts_by_type(type)`, `get_concepts_by_status(status)`, `get_active_concepts()`
  - `get_all_concept_ids()`, `exists(id)`, `get_relation_count(id)`
- **Dependencies:** `EpistemicMetadata`, `RelationInfo`
- **Invariante:** Wissen wird NIEMALS gelöscht, nur invalidiert.

#### `RelationInfo` — `ltm/relation.hpp`
- **Verantwortung:** Persistente gerichtete Relation zwischen Konzepten.
- **Key Members:** `RelationId id`, `ConceptId source`, `ConceptId target`, `RelationType type`, `double weight` [0.0, 1.0]
- **Constraints:** Default-Konstruktor gelöscht. Weight wird geclampt.
- **Dependencies:** `RelationType` (aus `active_relation.hpp`)

#### `RelationType` (enum class) — `memory/active_relation.hpp`
- Werte: `IS_A`, `HAS_PROPERTY`, `CAUSES`, `ENABLES`, `PART_OF`, `SIMILAR_TO`, `CONTRADICTS`, `SUPPORTS`, `TEMPORAL_BEFORE`, `CUSTOM`

---

### 1.4 memory/ (Short-Term Memory & Controller)

#### `STMEntry` — `memory/stm_entry.hpp`
- **Verantwortung:** Aktivierungszustand eines Konzepts in STM.
- **Key Members:** `ConceptId concept_id`, `double activation` [0.0, 1.0], `ActivationClass classification`, `time_point last_used`
- **Dependencies:** `ActivationClass`

#### `ActiveRelation` — `memory/active_relation.hpp`
- **Verantwortung:** Aktive Relation in STM (kurzfristig).
- **Key Members:** `ConceptId source/target`, `RelationType type`, `double activation`, `time_point last_used`

#### `ActivationLevel` (enum class) — `memory/activation_level.hpp`
- Werte: `LOW` (<0.3), `MEDIUM` (0.3–0.7), `HIGH` (≥0.7)

#### `ActivationClass` (enum class) — `memory/activation_level.hpp`
- Werte: `CORE_KNOWLEDGE` (langsamer Decay), `CONTEXTUAL` (schnellerer Decay)

#### `ShortTermMemory` — `memory/stm.hpp/.cpp`
- **Verantwortung:** Rein mechanische Aktivierungsschicht. Speichert KEIN Wissen, nur Aktivierungszustände.
- **Key Members:**
  - `unordered_map<ContextId, Context> contexts_` (Context enthält Konzept- und Relations-Aktivierungen)
  - Decay-Raten: `core_decay_rate_`, `contextual_decay_rate_`, `relation_decay_rate_`
  - Thresholds: `relation_inactive_threshold_`, `relation_removal_threshold_`, `concept_removal_threshold_`
- **Key Methods:**
  - `create_context()` / `destroy_context()` / `clear_context()`
  - `activate_concept()`, `activate_relation()`, `boost_concept()`, `boost_relation()`
  - `get_concept_activation()`, `get_active_concepts(threshold)`, `get_active_relations(threshold)`
  - `decay_all(context, time_delta)` — exponentieller Decay mit Zwei-Phasen-Relation-Decay
- **Invarianten:** STM speichert NUR Aktivierung, nie Wissensinhalt. STM bewertet nie Korrektheit.
- **Dependencies:** `STMEntry`, `ActiveRelation`, `ActivationClass`

#### `BrainController` — `memory/brain_controller.hpp/.cpp`
- **Verantwortung:** Minimale Orchestrierungsschicht. Kontext-Management und Flow-Koordination.
- **Key Members:** `unique_ptr<ShortTermMemory> stm_`, `bool initialized_`, `map<ContextId, ThinkingState>`
- **Key Methods:**
  - `initialize()` / `shutdown()`
  - `create_context()` / `destroy_context()`
  - `begin_thinking()` / `end_thinking()`
  - `activate_concept_in_context()`, `activate_relation_in_context()`
  - `decay_context()`, `query_concept_activation()`, `query_active_concepts()`
  - `get_stm()` (const), `get_stm_mutable()`
- **NICHT:** Lernen, Schlussfolgern, Bewerten, Entscheidungen über Wichtigkeit.
- **Dependencies:** `ShortTermMemory`

---

### 1.5 cognitive/ (Cognitive Dynamics)

#### Konfigurationsstructs — `cognitive/cognitive_config.hpp`
- `ActivationSpreaderConfig` — max_depth, damping_factor, activation_threshold, trust_weighted, relation_weighted
- `FocusManagerConfig` — max_focus_size (7±2 Miller), decay_rate, focus_threshold, attention_boost
- `SalienceComputerConfig` — Gewichte für activation/trust/connectivity/recency, query_boost_factor
- `ThoughtPathConfig` — max_paths (beam width), depth_penalty, salience/trust/coherence Gewichte
- `CognitiveDynamicsConfig` — Master-Config mit enable-Flags und debug_mode

#### Zustandstypen — `cognitive/cognitive_config.hpp`
- `ActivationEntry`, `FocusEntry`, `SalienceScore`, `ThoughtPathNode`, `ThoughtPath`, `SpreadingStats`

#### `CognitiveDynamics` — `cognitive/cognitive_dynamics.hpp/.cpp`
- **Verantwortung:** Additive kognitive Schicht: Spreading Activation, Salience, Focus, Thought Path Ranking.
- **Key Members:**
  - `CognitiveDynamicsConfig config_`
  - `unordered_map<ContextId, vector<FocusEntry>> focus_sets_`
  - `Stats stats_` (atomic counters)
- **Key Methods:**
  - **Spreading Activation:**
    - `spread_activation(source, activation, context, ltm, stm)` → `SpreadingStats`
    - `spread_activation_multi(sources, activation, context, ltm, stm)`
    - Formel: `activation(B) += activation(A) × relation_weight × trust(A) × damping^depth`
    - Zyklen-Erkennung via visited-Set, Depth-Limited, INVALIDATED werden übersprungen
  - **Salience:**
    - `compute_salience(cid, context, ltm, stm)` → `SalienceScore`
    - `compute_salience_batch()`, `get_top_k_salient()`, `compute_query_salience()`
    - Gewichtete Summe: activation × w_a + trust × w_t + connectivity × w_c + recency × w_r + query_boost
  - **Focus:**
    - `init_focus()`, `focus_on()`, `decay_focus()`, `get_focus_set()`, `is_focused()`, `get_focus_score()`
    - Kapazitätslimit (max_focus_size), exponentieller Decay
  - **Thought Paths:**
    - `find_best_paths(source, context, ltm, stm)` — Beam Search
    - `find_paths_to(source, target, context, ltm, stm)` — Zielgerichtete Suche
    - `score_path()` — salience × w_s + trust × w_t + coherence × w_c − depth_penalty
- **Architekturvertrag:**
  - ✅ READ-ONLY auf LTM und Trust
  - ✅ Schreibt NUR in STM (Aktivierungen) und eigenen Zustand
  - ✅ Deterministisch, bounded [0.0, 1.0], depth-limited
  - ❌ Darf NICHT: Wissen erzeugen, Hypothesen generieren, Trust ändern, epistemische Entscheidungen treffen
- **Dependencies:** `LongTermMemory` (read-only), `ShortTermMemory` (write Aktivierungen)

---

### 1.6 curiosity/

#### `TriggerType` (enum class) — `curiosity/curiosity_trigger.hpp`
- Werte: `SHALLOW_RELATIONS`, `MISSING_DEPTH`, `LOW_EXPLORATION`, `RECURRENT_WITHOUT_FUNCTION`, `UNKNOWN`

#### `CuriosityTrigger` — `curiosity/curiosity_trigger.hpp`
- **Verantwortung:** Reines Datensignal (keine Logik).
- **Key Members:** `TriggerType type`, `ContextId context_id`, `vector<ConceptId> related_concept_ids`, `string description`

#### `SystemObservation` — `curiosity/curiosity_engine.hpp`
- **Members:** `ContextId context_id`, `size_t active_concept_count`, `size_t active_relation_count`

#### `CuriosityEngine` — `curiosity/curiosity_engine.hpp/.cpp`
- **Verantwortung:** Reiner Signalgenerator. Beobachtet Systemzustand, emittiert Trigger.
- **Key Methods:**
  - `observe_and_generate_triggers(observations)` → `vector<CuriosityTrigger>`
  - `set_shallow_relation_threshold()`, `set_low_exploration_threshold()`
- **Constraints:** KEINE Aktionen, KEIN Lernen, KEINE direkten Modifikationen.
- **Dependencies:** Keine (nur eigene Structs)

---

### 1.7 kan/ (Kolmogorov-Arnold Networks)

#### `KANNode` — `kan/kan_node.hpp/.cpp`
- **Verantwortung:** Univariate lernbare Funktion via kubischen B-Splines.
- **Key Members:** `vector<double> knots_`, `vector<double> coefficients_`
- **Key Methods:** `evaluate(x)` → double, `gradient(x)`, `set_coefficients()`, `get_coefficients()`
- **Dependencies:** Keine.

#### `KANLayer` — `kan/kan_layer.hpp/.cpp`
- **Verantwortung:** Kollektion von KANNodes. Additive Kombination.
- **Key Members:** `vector<unique_ptr<KANNode>> nodes_`
- **Key Methods:** `evaluate(inputs)` → `vector<double>`, `input_dim()`
- **Dependencies:** `KANNode`

#### `KANModule` — `kan/kan_module.hpp/.cpp`
- **Verantwortung:** Vollständiger Funktionsapproximator f: ℝⁿ → ℝᵐ.
- **Key Members:** `size_t input_dim_`, `size_t output_dim_`, `vector<unique_ptr<KANLayer>> layers_` (ein Layer pro Output-Dimension)
- **Key Methods:** `evaluate(inputs)`, `train(dataset, config)` → `KanTrainingResult`, `compute_mse(dataset)`
- **Training:** Numerischer Gradient-Descent über Spline-Koeffizienten.
- **Dependencies:** `KANLayer`

#### `FunctionHypothesis` — `kan/function_hypothesis.hpp`
- **Verantwortung:** Reiner Daten-Wrapper für gelernten Funktionszustand.
- **Key Members:** `size_t input/output_dim`, `shared_ptr<KANModule> module`, `size_t training_iterations`, `double training_error`
- **Dependencies:** `KANModule`

#### `DataPoint`, `KanTrainingConfig`, `KanTrainingResult` — `kan/kan_module.hpp`
- Trainings-Datenstrukturen.

---

### 1.8 adapter/

#### `KANAdapter` — `adapter/kan_adapter.hpp/.cpp`
- **Verantwortung:** Saubere Schnittstelle zwischen BrainController und KAN. Explizite Delegation, KEINE Entscheidungslogik.
- **Key Members:** `unordered_map<uint64_t, KANModuleEntry> modules_`, `uint64_t next_module_id_`
- **Key Methods:**
  - `create_kan_module(input_dim, output_dim, num_knots)` → `uint64_t`
  - `train_kan_module(module_id, dataset, config)` → `unique_ptr<FunctionHypothesis>`
  - `evaluate_kan_module(module_id, inputs)` → `vector<double>`
  - `destroy_kan_module(module_id)`, `has_module(module_id)`
- **Dependencies:** `KANModule`, `FunctionHypothesis`

---

### 1.9 micromodel/ (Per-Concept Bilineare Micro-Modelle)

#### `MicroModel` — `micromodel/micro_model.hpp/.cpp`
- **Verantwortung:** Bilineares Modell pro Konzept: berechnet personalisierte Relevanz.
- **Architektur:** v = W·c + b (10D), z = eᵀ·v (Skalar), w = σ(z) ∈ (0,1)
- **Key Members:** `Mat10x10 W_`, `Vec10 b_`, `Vec10 e_init_`, `Vec10 c_init_`, `TrainingState state_` (Adam-Optimizer)
- **Key Methods:**
  - `predict(e, c)` → double ∈ (0,1)
  - `train_step(e, c, target, config)` → loss (Adam-Optimizer)
  - `train(samples, config)` → `MicroTrainingResult`
  - `to_flat() / from_flat()` — Serialisierung in 430 doubles
- **FLAT_SIZE:** 430 doubles (100 W + 10 b + 10 e_init + 10 c_init + 300 TrainingState)
- **Dependencies:** Keine externen.

#### `MicroModelRegistry` — `micromodel/micro_model_registry.hpp/.cpp`
- **Verantwortung:** Ein MicroModel pro ConceptId. Keyed Lookup, Bulk-Operationen.
- **Key Members:** `unordered_map<ConceptId, MicroModel> models_`
- **Key Methods:**
  - `create_model(cid)`, `get_model(cid)`, `has_model(cid)`, `remove_model(cid)`
  - `ensure_models_for(ltm)` — Bulk-Erstellung für alle LTM-Konzepte
- **Dependencies:** `MicroModel`, `LongTermMemory`

#### `EmbeddingManager` — `micromodel/embedding_manager.hpp/.cpp`
- **Verantwortung:** 10D-Embeddings für RelationTypes und Named Contexts.
- **Key Members:**
  - `array<Vec10, 10> relation_embeddings_` — heuristisch initialisiert (hierarchisch, kausal, etc.)
  - `unordered_map<string, Vec10> context_embeddings_` — deterministisch aus Name-Hash generiert
- **Key Methods:**
  - `get_relation_embedding(type)`, `get_context_embedding(name)` (auto-create)
  - `make_target_embedding(context_hash, source_id, target_id)` — ohne String-Allokation
  - Convenience: `query_context()`, `recall_context()`, `creative_context()`, `analytical_context()`
- **Dependencies:** `RelationType`

#### `MicroTrainer` — `micromodel/micro_trainer.hpp/.cpp`
- **Verantwortung:** Bootstrapped Trainingsdaten aus KG-Struktur, trainiert MicroModels.
- **Trainingsdata-Generierung:**
  - Positiv: Ausgehende Relationen (C→T, weight), eingehende Relationen (discount 0.8)
  - Negativ: 3× Negatives pro Positiv, von nicht-verbundenen Konzepten, target ≈ 0.05
- **Key Methods:**
  - `train_all(registry, embeddings, ltm)` → `TrainerStats`
  - `train_single(cid, model, embeddings, ltm)` → `MicroTrainingResult`
  - `generate_samples(cid, embeddings, ltm)` → `vector<TrainingSample>`
- **Dependencies:** `MicroModel`, `MicroModelRegistry`, `EmbeddingManager`, `LongTermMemory`

#### `RelevanceMap` — `micromodel/relevance_map.hpp/.cpp`
- **Verantwortung:** Evaluiert MicroModel eines Konzepts über alle KG-Knoten → scored Relevanz-Map.
- **Key Methods:**
  - `RelevanceMap::compute(source, registry, embeddings, ltm, rel_type, context)` (statisch)
  - `score(cid)`, `top_k(k)`, `above_threshold(threshold)`
  - `overlay(other, mode, weight)` — Kombination mehrerer Perspektiven (Phase 3 Creativity)
  - `combine(maps, mode, weights)` (statisch), `normalize()` → [0,1]
- **OverlayMode:** `ADDITION`, `MAX`, `WEIGHTED_AVERAGE`
- **Dependencies:** `MicroModel`, `MicroModelRegistry`, `EmbeddingManager`, `LongTermMemory`

#### `persistence` (Namespace) — `micromodel/persistence.hpp/.cpp`
- **Verantwortung:** Binäre Serialisierung von MicroModel-Registry + Embeddings.
- **Format:** Header (Magic "BM19", Version, Counts) + Models (3448 Bytes/Stk) + Relation Embeddings + Context Embeddings + XOR-Checksum
- **Key Methods:** `save(filepath, registry, embeddings)`, `load(filepath, registry, embeddings)`, `validate(filepath)`
- **Dependencies:** `MicroModelRegistry`, `EmbeddingManager`

---

### 1.10 importers/

#### `KnowledgeProposal` — `importers/knowledge_proposal.hpp`
- **Verantwortung:** Reine Datenstruktur für Wissens-VORSCHLÄGE (nicht akzeptiertes Wissen).
- **Key Members:** `uint64_t proposal_id`, `SourceType source_type`, `string source_reference`, `string extracted_text/title`, `vector<SuggestedConcept>`, `vector<SuggestedRelation>`, `SuggestedEpistemicType` (NUR Vorschlag!)
- **Epistemic Rule:** Importers DÜRFEN NICHT EpistemicType/Trust/Status zuweisen. Nur Suggestions.
- **Dependencies:** Keine.

#### `SuggestedEpistemicType` (enum class) — `importers/knowledge_proposal.hpp`
- Werte: `FACT_CANDIDATE`, `THEORY_CANDIDATE`, `HYPOTHESIS_CANDIDATE`, `DEFINITION_CANDIDATE`, `UNKNOWN_CANDIDATE`

#### `WikipediaImporter` — `importers/wikipedia_importer.hpp/.cpp`
- **Verantwortung:** Extrahiert strukturierte Übersicht aus Wikipedia. Schreibt NICHT in LTM.
- **Key Methods:** `import_article(title)`, `parse_wikipedia_text(title, html)` → `unique_ptr<KnowledgeProposal>`
- **Interne Extraktion:** Lead-Section, Konzepte (Regex auf Großschreibung), "X is a Y"-Relationen
- **Dependencies:** `KnowledgeProposal`

#### `ScholarImporter` — `importers/scholar_importer.hpp/.cpp`
- **Verantwortung:** Extrahiert Forschungswissen aus Papers. Schreibt NICHT in LTM.
- **Key Methods:** `import_paper_by_doi(doi)`, `parse_paper_text(title, text, authors, year, venue, is_preprint)` → `unique_ptr<KnowledgeProposal>`
- **Uncertainty Detection:** Erkennt Hedging-Sprache (may, might, could, etc.) → `HYPOTHESIS_CANDIDATE`
- **Dependencies:** `KnowledgeProposal`

---

### 1.11 ingestor/ (Ingestion Pipeline)

#### `TextChunker` — `ingestor/text_chunker.hpp/.cpp`
- **Verantwortung:** Teilt Plaintext in satzbasierte Chunks mit optionalem Overlap.
- **Config:** `sentences_per_chunk` (3), `overlap_sentences` (1), `max_chunk_chars` (2000)
- **Key Methods:** `chunk_text(text)` → `vector<TextChunk>`, `split_sentences(text)`
- **Dependencies:** Keine.

#### `EntityExtractor` — `ingestor/entity_extractor.hpp/.cpp`
- **Verantwortung:** Musterbasierte Entity-Extraktion aus Text (ohne externe NLP).
- **Extraktionsstrategien:** Großgeschriebene Phrasen, Zitate, Definitionsmuster ("X is a..."), frequente Terme
- **Key Methods:** `extract_from_text(text)`, `extract_from_chunks(chunks)` → `vector<ExtractedEntity>`
- **Dependencies:** `TextChunk`

#### `RelationExtractor` — `ingestor/relation_extractor.hpp/.cpp`
- **Verantwortung:** Musterbasierte Relationsextraktion. Mapped Patterns auf `RelationType`.
- **Patterns:** "X is a Y" → IS_A, "X causes Y" → CAUSES, "X enables Y" → ENABLES, etc. (15+ Muster)
- **Key Methods:** `extract_relations(text, known_entities)`, `extract_relations_blind(text)` → `vector<ExtractedRelation>`
- **Dependencies:** `ExtractedEntity`, `RelationType`

#### `TrustTagger` — `ingestor/trust_tagger.hpp/.cpp`
- **Verantwortung:** Mappt Trust-Kategorien auf EpistemicMetadata. Heuristisch basiert auf Text-Signalen.
- **TrustCategory → EpistemicType + Trust Mapping:**
  - FACTS → FACT, 0.95–0.99
  - DEFINITIONS → DEFINITION, 0.90–0.99
  - THEORIES → THEORY, 0.85–0.95
  - HYPOTHESES → HYPOTHESIS, 0.50–0.80
  - INFERENCES → INFERENCE, 0.40–0.70
  - SPECULATION → SPECULATION, 0.10–0.40
  - INVALIDATED → INVALIDATED, 0.01–0.10
- **Text-Signale:** Hedging Language, Certainty Language, Definition Patterns, Citation Markers
- **Key Methods:** `assign_trust(category)`, `suggest_from_text(text)`, `suggest_from_source(source)`, `suggest_from_proposal(suggested_type)`
- **Dependencies:** `EpistemicMetadata`

#### `ProposalQueue` — `ingestor/proposal_queue.hpp/.cpp`
- **Verantwortung:** Staging-Area für unvalidierte Wissensvorschläge. Nichts gelangt in LTM ohne diese Queue.
- **ProposalStatus:** `PENDING`, `APPROVED`, `REJECTED`, `MODIFIED`, `EXPIRED`
- **Key Methods:**
  - `enqueue(proposal)`, `enqueue_batch(proposals)`
  - `review(id, decision)`, `review_batch(ids, decision)`, `auto_approve_all()`
  - `get_pending()`, `get_approved()`, `pop_approved()`
  - `expire_old(max_age)`
- **ReviewDecision:** `approve()`, `reject()`, `approve_with_trust(category)`
- **Dependencies:** `IngestProposal`, `TrustTagger`

#### `KnowledgeIngestor` — `ingestor/knowledge_ingestor.hpp/.cpp`
- **Verantwortung:** Parst strukturierten Input (JSON, CSV) in `StructuredInput`.
- **JSON-Format:** `{"source": "...", "concepts": [...], "relations": [...]}`
- **CSV-Format:** `label,definition,trust_category,trust_value`
- **Key Methods:** `parse_json(json_str)`, `parse_csv_concepts(csv)`, `to_proposals(input, tagger)`
- **Kein externer JSON-Parser** — handgeschriebener Minimal-Parser.
- **Dependencies:** `TrustTagger`, `StructuredInput`

#### `IngestionPipeline` — `ingestor/ingestion_pipeline.hpp/.cpp`
- **Verantwortung:** Vollständige Pipeline: Input → Chunking → Extraktion → Trust → Queue → LTM.
- **Key Members:** `LongTermMemory& ltm_`, `ProposalQueue queue_`, `TrustTagger`, `KnowledgeIngestor`, `TextChunker`, `EntityExtractor`, `RelationExtractor`
- **Key Methods:**
  - `ingest_json(json_str, auto_approve)`, `ingest_csv(concepts_csv, relations_csv, auto_approve)`
  - `ingest_text(text, source_ref, auto_approve)` — NLP-Pipeline: Chunk → Extract → Trust → Queue
  - `commit_approved()` — Schreibt approved Proposals in LTM
- **Architekturverträge:**
  - Pipeline schreibt NIE direkt in LTM ohne ProposalQueue
  - Bestehende LTM-Daten werden NIE modifiziert
  - Pipeline ist ADDITIV (kein Delete, kein Modify)
- **Dependencies:** Alle Ingestor-Komponenten + `LongTermMemory`

---

### 1.12 understanding/ (Understanding Layer)

#### `MiniLLM` (Interface) — `understanding/mini_llm.hpp`
- **Verantwortung:** Abstrakte Schnittstelle für semantische Modelle.
- **Architekturvertrag:**
  - ✅ Texte interpretieren, Muster erkennen, Vorschläge generieren
  - ❌ KG modifizieren, Trust setzen, epistemische Entscheidungen treffen, in LTM schreiben
  - Alle Outputs sind HYPOTHESIS
  - READ-ONLY Zugriff auf LTM
- **Pure Virtual Methods:** `extract_meaning()`, `generate_hypotheses()`, `detect_analogies()`, `detect_contradictions()`
- **Dependencies:** `LongTermMemory` (read-only), `ShortTermMemory` (read-only)

#### `StubMiniLLM` — `understanding/mini_llm.hpp/.cpp`
- **Verantwortung:** Test-Placeholder ohne echtes LLM. Verifiziert epistemische Invarianten.

#### `OllamaMiniLLM` — `understanding/ollama_mini_llm.hpp/.cpp`
- **Verantwortung:** Echte semantische Analyse via Ollama-LLM.
- **Key Methods:** Implementiert `MiniLLM`-Interface. Baut Prompts aus Konzepten, parsed LLM-Responses.
- **Confidence-Heuristik:** Uncertainty-Words senken Confidence, Certainty-Words erhöhen sie.
- **Dependencies:** `OllamaClient`, `LongTermMemory` (read-only)

#### `MiniLLMFactory` — `understanding/mini_llm_factory.hpp`
- **Verantwortung:** (TODO/Planned) Erzeugt spezialisierte Mini-LLMs für gelernte Konzepte.
- **Status:** Header deklariert, nicht implementiert.

#### `SpecializedMiniLLM` — `understanding/mini_llm_factory.hpp`
- **Verantwortung:** (TODO/Planned) Mini-LLM mit Expertise in spezifischem Bereich.
- **Status:** Header deklariert, nicht implementiert.

#### Proposal-Typen — `understanding/understanding_proposals.hpp`
- **`MeaningProposal`:** Semantischer Vorschlag. `epistemic_type` ist IMMER `HYPOTHESIS`.
- **`HypothesisProposal`:** Vorgeschlagene Hypothese. `suggested_epistemic.suggested_type` ist IMMER `HYPOTHESIS` (unabhängig vom Input).
- **`AnalogyProposal`:** Strukturelle Analogie zwischen zwei Domänen.
- **`ContradictionProposal`:** Erkannte potenzielle Inkonsistenz.
- Alle Confidence-Werte sind model_confidence [0.0, 1.0] — NICHT epistemic trust.

#### `UnderstandingLayer` — `understanding/understanding_layer.hpp/.cpp`
- **Verantwortung:** Semantische Analyse-Schicht über Cognitive Dynamics.
- **Key Members:** `vector<unique_ptr<MiniLLM>> mini_llms_`, `UnderstandingLayerConfig config_`, `Statistics stats_`
- **Key Methods:**
  - `register_mini_llm(mini_llm)` — übernimmt Ownership
  - `analyze_meaning(concepts, ltm, stm, context)` → `vector<MeaningProposal>`
  - `propose_hypotheses()` → `vector<HypothesisProposal>`
  - `find_analogies()` → `vector<AnalogyProposal>`
  - `check_contradictions()` → `vector<ContradictionProposal>`
  - `perform_understanding_cycle(seed, cognitive_dynamics, ltm, stm, context)` → `UnderstandingResult`
    1. Spreading Activation via CognitiveDynamics
    2. Salience-Berechnung → Top-10 salient Concepts
    3. Meaning + Hypothesis + Contradiction Proposals
    4. Analogy Detection (bei ≥4 Konzepten)
- **Architekturvertrag:** READ-ONLY LTM, alle Proposals sind HYPOTHESIS, kein autonomes Handeln
- **Dependencies:** `MiniLLM`, `CognitiveDynamics`, `LongTermMemory`, `ShortTermMemory`

---

### 1.13 llm/ (LLM Integration)

#### `OllamaClient` — `llm/ollama_client.hpp/.cpp`
- **Verantwortung:** HTTP-Client für Ollama REST API.
- **Key Members:** `OllamaConfig config_` (host, model, temperature, num_predict), `bool initialized_`
- **Key Methods:** `initialize(config)`, `is_available()`, `list_models()`, `chat(messages)` → `OllamaResponse`, `generate(prompt)`
- **Impl:** Verwendet libcurl + nlohmann/json.
- **Dependencies:** Externe: libcurl, nlohmann/json

#### `ChatInterface` — `llm/chat_interface.hpp/.cpp`
- **Verantwortung:** LLM-powered Verbalisierung von Brain19-Wissen. LLM ist ein TOOL, kein Agent.
- **Key Methods:**
  - `ask(question, ltm)` → `ChatResponse` — findet relevante Konzepte, baut epistemischen Kontext, fragt LLM
  - `explain_concept(id, ltm)`, `compare(id1, id2, ltm)`, `list_knowledge(ltm, type)`, `get_summary(ltm)`
- **System Prompt:** Enforced epistemische Rigorosität: LLM MUSS Trust-Level und EpistemicType in Antworten einbauen.
- **Fallback:** Funktioniert ohne LLM mit strukturiertem Output.
- **Dependencies:** `OllamaClient`, `LongTermMemory` (read-only)

---

### 1.14 snapshot_generator*

#### `SnapshotGenerator` — `snapshot_generator.hpp/.cpp`
- **Verantwortung:** Erzeugt JSON-Snapshots für Visualisierung mit epistemischer Metadaten.
- **Key Method:** `generate_json_snapshot(brain, ltm, curiosity, context_id)` → JSON string
- **Epistemic Enforcement:** Jedes Konzept MUSS epistemische Daten enthalten. STM-only Konzepte bekommen `HYPOTHESIS/CONTEXTUAL/0.5`.
- **Dependencies:** `BrainController`, `LongTermMemory`, `CuriosityEngine`

---

### 1.15 tools/

#### `brain19_cli` — `tools/brain19_cli.cpp`
- **Verantwortung:** Interaktives CLI-Tool. Verknüpft Ingestor + MicroModel.
- **Modi:** `--review` (manuelles Approve/Reject) oder Direct (auto-approve)
- **Menü:** [1] JSON ingest, [2] Text ingest, [3] KG anzeigen, [4] Review, [5] Train, [6] Relevance Map, [7] Train All
- **Dependencies:** `IngestionPipeline`, `MicroModelRegistry`, `EmbeddingManager`, `MicroTrainer`, `RelevanceMap`

---

### 1.16 Demo-Programme

| File | Zweck |
|---|---|
| `demo_chat.cpp` | Chat-Interface Demo mit LLM |
| `demo_cognitive_dynamics.cpp` | Cognitive Dynamics Demo |
| `demo_epistemic_complete.cpp` | Vollständiges Epistemic-Enforcement Demo |
| `demo_integrated.cpp` | Integrierte Pipeline Demo |
| `demo_understanding_layer.cpp` | Understanding Layer Demo |

---

## 2. Subsystem-Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        BRAIN19 ARCHITEKTUR                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Importers  │     │   Ingestor       │     │  tools/CLI       │
│  (Wikipedia, │────>│  (Pipeline,      │────>│  (brain19_cli)   │
│   Scholar)   │     │   Chunker,       │     └──────────────────┘
│              │     │   Entity/Relation │
│  Proposals   │     │   Extractor,     │
│  ONLY        │     │   TrustTagger,   │
└──────────────┘     │   ProposalQueue) │
                     └────────┬─────────┘
                              │ commit_approved()
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LongTermMemory (LTM)                          │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐    │
│  │ Concepts │  │  Relations   │  │ EpistemicMetadata      │    │
│  │(ConceptInfo)│(RelationInfo)│  │ (Type, Status, Trust)  │    │
│  └──────────┘  └──────────────┘  └────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
            READ-ONLY ▼   │ READ-ONLY ▼
┌───────────────────────┐ │ ┌─────────────────────────────────┐
│  Cognitive Dynamics   │ │ │     Understanding Layer          │
│  (Spreading, Salience,│ │ │  ┌────────────────────────┐     │
│   Focus, ThoughtPaths)│◄┼─│  │ MiniLLM (Interface)    │     │
│                       │ │ │  │  ├─ StubMiniLLM        │     │
│  SCHREIBT → STM       │ │ │  │  ├─ OllamaMiniLLM      │     │
│  LIEST  ← LTM        │ │ │  │  └─ (SpecializedMiniLLM)│    │
└───────────┬───────────┘ │ │  └────────────────────────┘     │
            │             │ │  Alle Outputs: HYPOTHESIS        │
            ▼             │ │  READ-ONLY LTM                   │
┌───────────────────────┐ │ └─────────────────────────────────┘
│  ShortTermMemory (STM)│ │
│  (Aktivierungen,      │ │
│   Decay, Contexte)    │ │ ┌─────────────────────────────────┐
└───────────┬───────────┘ │ │      MicroModel Subsystem        │
            │             │ │  ┌──────────────────────────┐    │
            ▼             │ │  │ MicroModelRegistry       │    │
┌───────────────────────┐ │ │  │ EmbeddingManager         │    │
│  BrainController      │ │ │  │ MicroTrainer             │    │
│  (Orchestration,      │◄┘ │  │ RelevanceMap             │    │
│   Context Lifecycle)  │   │  │ Persistence (BM19)       │    │
└───────────────────────┘   │  └──────────────────────────┘    │
                            │  LIEST ← LTM (KG-Struktur)      │
┌───────────────────────┐   └─────────────────────────────────┘
│  KAN Subsystem        │
│  ┌──────────┐         │   ┌─────────────────────────────────┐
│  │ KANNode  │         │   │  LLM Integration                │
│  │ KANLayer │         │   │  ┌────────────────────────┐     │
│  │ KANModule│         │   │  │ OllamaClient (curl)    │     │
│  └──────────┘         │   │  │ ChatInterface          │     │
│  KANAdapter ──────────┼──>│  └────────────────────────┘     │
└───────────────────────┘   │  LIEST ← LTM                    │
                            └─────────────────────────────────┘
┌───────────────────────┐   ┌─────────────────────────────────┐
│  CuriosityEngine      │   │  SnapshotGenerator              │
│  (Signalgenerator)    │   │  (JSON Visualisierung)           │
│  READ-ONLY            │   │  LIEST ← Brain, LTM, Curiosity  │
└───────────────────────┘   └─────────────────────────────────┘
```

---

## 3. Datenfluss-Diagramme

### 3.1 Input → Processing → Storage

```
Externer Input
     │
     ├── JSON/CSV ──────────> KnowledgeIngestor.parse_json/csv()
     │                              │
     ├── Plain Text ──────────> TextChunker.chunk_text()
     │                              │
     ├── Wikipedia Article ───> WikipediaImporter.import_article()
     │                              │ KnowledgeProposal (SUGGESTION only)
     └── Research Paper ──────> ScholarImporter.parse_paper_text()
                                    │
                                    ▼
                          EntityExtractor.extract_from_chunks()
                                    │
                                    ▼
                          RelationExtractor.extract_relations()
                                    │
                                    ▼
                          TrustTagger.suggest_from_text()
                                    │ TrustAssignment (SUGGESTION)
                                    ▼
                          ProposalQueue.enqueue()
                                    │ IngestProposal (PENDING)
                                    ▼
                          [Human Review / auto_approve]
                                    │ APPROVED / REJECTED
                                    ▼
                          IngestionPipeline.commit_approved()
                                    │
                                    ▼
                          LTM.store_concept(label, def, EpistemicMetadata)
                          LTM.add_relation(source, target, type, weight)
```

### 3.2 Spreading Activation Flow

```
seed ConceptId + initial_activation
         │
         ▼
CognitiveDynamics.spread_activation()
         │
         ├── LTM.retrieve_concept(source) ──── READ-ONLY
         │   └── Check: is_invalidated()? → SKIP
         │   └── Get trust value
         │
         ├── STM.activate_concept(source, activation)
         │
         ├── LTM.get_outgoing_relations(source) ──── READ-ONLY
         │
         └── For each relation:
              │
              │ propagated = activation × weight × trust × damping^depth
              │
              ├── STM.activate_concept(target, propagated) oder boost_concept()
              │
              └── REKURSIV: spread_recursive(target, propagated, depth+1)
                  │
                  ├── BASE: depth ≥ max_depth → STOP
                  ├── BASE: activation < threshold → STOP
                  └── BASE: already visited → STOP (Zyklen-Erkennung)
```

### 3.3 Salience Flow

```
vector<ConceptId> candidates
         │
         ▼
CognitiveDynamics.compute_salience_batch()
         │
         ├── For each concept:
         │    │
         │    ├── activation_factor ← STM.get_concept_activation()
         │    ├── trust_factor     ← LTM.retrieve_concept().epistemic.trust
         │    ├── connectivity     ← LTM.get_relation_count() / max_connectivity
         │    ├── recency          ← exp(-0.07 × ticks_since_access)
         │    └── query_boost      ← direct_match(0.5) / indirect_match(0.25)
         │
         │ salience = w_a × activation + w_t × trust + w_c × connectivity
         │          + w_r × recency + query_boost
         │
         │ clamp(0.0, max_salience)
         │
         └── Sort descending by salience
              │
              ▼
         vector<SalienceScore>
```

### 3.4 Understanding/LLM Cycle

```
ConceptId seed
    │
    ▼
UnderstandingLayer.perform_understanding_cycle()
    │
    ├── Phase 1: CognitiveDynamics.spread_activation(seed)
    │             → Aktiviert verbundene Konzepte in STM
    │
    ├── Phase 2: CognitiveDynamics.compute_salience_batch()
    │             → Top-10 salient Konzepte identifizieren
    │
    ├── Phase 3: For each registered MiniLLM:
    │    │
    │    ├── MiniLLM.extract_meaning(salient_concepts, ltm, stm)
    │    │   → vector<MeaningProposal>  (ALL HYPOTHESIS)
    │    │
    │    ├── MiniLLM.generate_hypotheses(salient_concepts, ltm, stm)
    │    │   → vector<HypothesisProposal>  (ALL HYPOTHESIS)
    │    │
    │    └── MiniLLM.detect_contradictions(salient_concepts, ltm, stm)
    │        → vector<ContradictionProposal>
    │
    ├── Phase 4: MiniLLM.detect_analogies(set_a, set_b, ltm, stm)
    │             → vector<AnalogyProposal>  (wenn ≥4 Konzepte)
    │
    └── Filter by confidence thresholds
         │
         ▼
    UnderstandingResult
    (meaning + hypothesis + analogy + contradiction proposals)

    ⚠ ALLE Proposals sind HYPOTHESIS
    ⚠ LTM wurde NICHT modifiziert
    ⚠ Trust wurde NICHT verändert
```

### 3.5 Ingestion Pipeline (Detailliert)

```
ingest_text("Cats are mammals...", "source")
    │
    ├── TextChunker.chunk_text()
    │   └── Satz-Splitting → Gruppierung à 3 Sätze mit 1 Overlap
    │       → vector<TextChunk>
    │
    ├── EntityExtractor.extract_from_chunks()
    │   ├── Capitalized Phrases: "Cat", "Mammal"
    │   ├── Quoted Terms: "photosynthesis"
    │   ├── Definition Patterns: "X is a ..."
    │   └── Frequent Terms (≥2 Vorkommen)
    │       → vector<ExtractedEntity> (dedupliziert)
    │
    ├── RelationExtractor.extract_relations(text, entities)
    │   ├── Blind: Regex-Matching ("X is a Y" → IS_A, etc.)
    │   ├── Entity-Boosted: Bekannte Entities × 1.1–1.3 Confidence
    │   └── Proximity: Entities nahe beieinander + Keywords
    │       → vector<ExtractedRelation>
    │
    ├── For each Entity:
    │   ├── TrustTagger.suggest_from_text(context_snippet)
    │   │   └── Hedging → SPECULATION, Definitions → DEFINITIONS, etc.
    │   └── IngestProposal erstellen
    │       → ProposalQueue.enqueue()
    │
    └── If auto_approve:
        ├── ProposalQueue.auto_approve_all()
        └── commit_approved()
            ├── Check existing concepts (Label-Match)
            ├── LTM.store_concept(label, def, TrustAssignment.to_epistemic_metadata())
            └── LTM.add_relation(source_id, target_id, type, confidence)
```

---

## 4. Ownership & Lifecycle

### 4.1 Wer erstellt/besitzt/zerstört was?

| Objekt | Erstellt von | Besitzer | Zerstört von |
|--------|-------------|----------|-------------|
| `ShortTermMemory` | `BrainController::initialize()` | `BrainController` (unique_ptr) | `BrainController::shutdown()` |
| `Context` (in STM) | `STM::create_context()` | `ShortTermMemory` | `STM::destroy_context()` |
| `STMEntry` | `STM::activate_concept()` | `Context.concepts` map | `STM::decay_all()` (bei threshold) |
| `ConceptInfo` | `LTM::store_concept()` | `LongTermMemory.concepts_` map | NIE gelöscht (nur invalidiert) |
| `RelationInfo` | `LTM::add_relation()` | `LongTermMemory.relations_` map | `LTM::remove_relation()` |
| `KANModule` | `KANAdapter::create_kan_module()` | `KANAdapter` (shared_ptr) | `KANAdapter::destroy_kan_module()` |
| `FunctionHypothesis` | `KANAdapter::train_kan_module()` | Aufrufer (unique_ptr) | Aufrufer |
| `MicroModel` | `MicroModelRegistry::create_model()` | `MicroModelRegistry.models_` map | `Registry::remove_model()` oder `clear()` |
| `MiniLLM` | Extern erstellt | `UnderstandingLayer` (unique_ptr) | `UnderstandingLayer` Destruktor |
| `KnowledgeProposal` | Importers | Aufrufer (unique_ptr) | Aufrufer |
| `IngestProposal` | `IngestionPipeline` | `ProposalQueue.proposals_` | `Queue::pop_approved()` oder `clear()` |
| `CuriosityTrigger` | `CuriosityEngine` | Aufrufer (value) | Aufrufer |
| `OllamaClient` | `ChatInterface` | `ChatInterface` (unique_ptr) | `ChatInterface` Destruktor |

### 4.2 Lifecycle-Muster

```
BrainController Lifecycle:
  BrainController() → initialize() → [use] → shutdown() → ~BrainController()

Context Lifecycle:
  create_context() → [activate, decay, query] → destroy_context()

Concept Lifecycle (LTM):
  store_concept() → [retrieve, query] → invalidate_concept()
  ⚠ NIE gelöscht. INVALIDATED bleibt in Storage.

Proposal Lifecycle:
  Importer/Extractor → IngestProposal → enqueue() → [review] → APPROVED/REJECTED
  → pop_approved() → commit_approved() → LTM.store_concept()

MicroModel Lifecycle:
  Registry.create_model() → Trainer.train_single() → RelevanceMap.compute()
  → Persistence.save() → [restart] → Persistence.load()
```

---

## 5. Epistemischer Fluss

### 5.1 Kernprinzipien

1. **Kein Default:** `EpistemicMetadata` hat keinen Default-Konstruktor. Jedes Wissenselement muss explizit typisiert werden.
2. **Keine Inferenz:** Importers und Understanding Layer dürfen NICHT epistemische Entscheidungen treffen.
3. **Kein Löschen:** Wissen wird invalidiert, nie gelöscht. `INVALIDATED` bleibt mit trust < 0.2.
4. **Proposals only:** MiniLLMs und Importers produzieren VORSCHLÄGE, die IMMER `HYPOTHESIS` sind.
5. **Trust ist immutabel durch Kognition:** CognitiveDynamics und UnderstandingLayer ändern Trust NICHT.

### 5.2 Trust-Propagation durch das System

```
Externe Quelle (Wikipedia, Paper, JSON, Text)
        │
        ▼
  SuggestedEpistemicType (NUR Vorschlag)
  ├── FACT_CANDIDATE
  ├── THEORY_CANDIDATE
  ├── HYPOTHESIS_CANDIDATE
  ├── DEFINITION_CANDIDATE
  └── UNKNOWN_CANDIDATE
        │
        ▼
  TrustTagger (konservativer Downgrade)
  ├── FACT_CANDIDATE → THEORIES (0.90)  ← konservativ!
  ├── THEORY_CANDIDATE → HYPOTHESES (0.65)
  ├── HYPOTHESIS_CANDIDATE → HYPOTHESES (0.65)
  └── DEFINITION_CANDIDATE → DEFINITIONS (0.95)
        │
        ▼  TrustAssignment
  ProposalQueue (PENDING)
        │
        ▼  [Human Review / Auto-Approve]
  ReviewDecision
  ├── approve() → TrustAssignment bleibt
  ├── reject() → wird verworfen
  └── approve_with_trust(FACTS) → Trust Override
        │
        ▼
  EpistemicMetadata (final, explizit)
  ├── EpistemicType (FACT/THEORY/HYPOTHESIS/...)
  ├── EpistemicStatus (ACTIVE)
  └── trust (0.0–1.0)
        │
        ▼
  LTM.store_concept(..., EpistemicMetadata)
        │
        ├──────────────────────────────────────────┐
        │                                          │
        ▼  READ-ONLY                               ▼  READ-ONLY
  CognitiveDynamics                        UnderstandingLayer
  ├── Trust gewichtet Spreading            ├── LTM lesen
  │   activation × trust × weight          ├── Vorschläge generieren
  ├── Trust beeinflusst Salience           │   (IMMER HYPOTHESIS)
  │   salience += w_t × trust              └── Trust NICHT ändern
  ├── Trust beeinflusst Path Score
  │   path_score += w_t × avg_trust
  └── INVALIDATED → wird übersprungen
```

### 5.3 Epistemische Barrieren

```
                    KANN NICHT DURCHBROCHEN WERDEN
                    ==============================

  Importer ──X──> LTM     (Importers schreiben NIE direkt)
  MiniLLM  ──X──> LTM     (Understanding ist READ-ONLY)
  MiniLLM  ──X──> Trust    (Trust wird NIE von LLMs geändert)
  CogDyn   ──X──> Trust    (Spreading ändert NIE Trust)
  CogDyn   ──X──> LTM      (CognitiveDynamics schreibt NIE in LTM)

  EINZIGER Schreibpfad in LTM:
  IngestionPipeline.commit_approved() → LTM.store_concept()

  EINZIGER Weg Trust zu ändern:
  LTM.update_epistemic_metadata() / LTM.invalidate_concept()
```

---

## 6. Dependency Graph

```
                          common/types.hpp
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    epistemic/            memory/           memory/
    epistemic_metadata    activation_level   active_relation
              │           stm_entry              │
              │                │                │
              │                ▼                │
              │          memory/stm ◄───────────┘
              │                │
              │                ▼
              │       memory/brain_controller
              │
              ├──────────────────────────────────────┐
              ▼                                      │
    ltm/long_term_memory ◄── ltm/relation            │
              │                                      │
    ┌─────────┼──────────────┬────────────────┐     │
    │         │              │                │     │
    ▼         ▼              ▼                ▼     │
cognitive/  understanding/  micromodel/    ingestor/ │
cognitive   understanding   micro_model    │         │
_dynamics   _layer          │              │         │
    │         │             ▼              │         │
    │         │       micro_model_registry │         │
    │         │             │              │         │
    │         │             ▼              │         │
    │         │       embedding_manager    │         │
    │         │             │              │         │
    │         │             ▼              │         │
    │         │       micro_trainer        │         │
    │         │             │              │         │
    │         │             ▼              │         │
    │         │       relevance_map        │         │
    │         │             │              │         │
    │         │             ▼              │         │
    │         │       persistence          │         │
    │         │                            │         │
    │         ▼                            │         │
    │   mini_llm (interface)               │         │
    │     ├── stub_mini_llm                │         │
    │     └── ollama_mini_llm              │         │
    │              │                       │         │
    │              ▼                       │         │
    │     llm/ollama_client                │         │
    │              │                       │         │
    │              ▼                       │         │
    │     llm/chat_interface               │         │
    │                                      │         │
    │   understanding_proposals            │         │
    │                                      │         │
    │                            ┌─────────┘         │
    │                            ▼                   │
    │                   importers/                   │
    │                   knowledge_proposal            │
    │                     ├── wikipedia_importer      │
    │                     └── scholar_importer        │
    │                            │                   │
    │                            ▼                   │
    │                   ingestor/text_chunker         │
    │                   ingestor/entity_extractor     │
    │                   ingestor/relation_extractor   │
    │                   ingestor/trust_tagger         │
    │                   ingestor/proposal_queue       │
    │                   ingestor/knowledge_ingestor   │
    │                            │                   │
    │                            ▼                   │
    │                   ingestor/ingestion_pipeline ──┘
    │
    ▼
curiosity/curiosity_engine ◄── curiosity/curiosity_trigger

kan/kan_node ──> kan/kan_layer ──> kan/kan_module
                                        │
                                        ▼
                              kan/function_hypothesis
                                        │
                                        ▼
                              adapter/kan_adapter

snapshot_generator ◄── brain_controller + ltm + curiosity_engine

tools/brain19_cli ◄── ingestion_pipeline + micromodel subsystem
```

---

## Anhang: Dateiübersicht

| Subsystem | Header (.hpp) | Impl (.cpp) | Tests |
|-----------|--------------|-------------|-------|
| common | 1 | 0 | — |
| epistemic | 1 | 0 | test_epistemic_enforcement.cpp |
| ltm | 2 | 1 | — |
| memory | 4 | 2 | test_brain.cpp |
| cognitive | 2 | 1 | test_cognitive_dynamics.cpp |
| curiosity | 2 | 1 | — |
| kan | 4 | 3 | — |
| adapter | 1 | 1 | — |
| micromodel | 6 | 6 | test_micromodel.cpp |
| importers | 3 | 2 | test_importers.cpp |
| ingestor | 7 | 7 | test_ingestor.cpp |
| understanding | 4 | 3 | test_understanding_layer.cpp |
| llm | 2 | 2 | — |
| snapshot | 1 | 1 | — |
| tools | 0 | 1 | — |
| demos | 0 | 5 | — |
| **Gesamt** | **40** | **36** | **6** |
