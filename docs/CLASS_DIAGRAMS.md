# Brain19 — Class Diagrams

> Detailed UML class diagrams for all core module hierarchies.
> All class names, member variables, and method signatures match the actual code in `backend/`.
> Updated: 2026-02-12

---

## Table of Contents

1. [KAN Hierarchy](#1-kan-hierarchy)
2. [Memory Subsystem](#2-memory-subsystem)
3. [Epistemic System](#3-epistemic-system)
4. [MicroModel Layer](#4-micromodel-layer)
5. [KAN-LLM Hybrid System](#5-kan-llm-hybrid-system)
6. [Understanding Layer](#6-understanding-layer)
7. [Evolution System](#7-evolution-system)
8. [Ingestion Pipeline](#8-ingestion-pipeline)
9. [Streams & Concurrency](#9-streams--concurrency)
10. [Persistence Layer](#10-persistence-layer)
11. [Core Orchestration](#11-core-orchestration)

---

## 1. KAN Hierarchy

Kolmogorov-Arnold Networks: B-spline-based learnable function approximators.

```mermaid
classDiagram
    class KANNode {
        -size_t num_knots_
        -vector~double~ knots_
        -vector~double~ coefficients_
        +KANNode(num_knots: size_t)
        +evaluate(x: double) double
        +get_coefficients() vector~double~
        +set_coefficients(coefs: vector~double~)
        +gradient(x: double, epsilon: double) vector~double~
        -basis_function(i: size_t, x: double, degree: size_t) double
        -cox_de_boor(i: size_t, k: size_t, x: double) double
    }

    class KANLayer {
        -size_t input_dim_
        -size_t output_dim_
        -vector~unique_ptr~KANNode~~ nodes_
        +KANLayer(input_dim, output_dim, num_knots)
        +evaluate(inputs: vector~double~) vector~double~
        +node(i: size_t, j: size_t) KANNode
        +input_dim() size_t
        +output_dim() size_t
        +num_nodes() size_t
    }

    class KANModule {
        -size_t input_dim_
        -size_t output_dim_
        -vector~size_t~ layer_dims_
        -vector~unique_ptr~KANLayer~~ layers_
        +KANModule(layer_dims: vector~size_t~, num_knots)
        +KANModule(input_dim, output_dim, num_knots)
        +evaluate(inputs: vector~double~) vector~double~
        +train(dataset: vector~DataPoint~, config) KanTrainingResult
        +compute_mse(dataset: vector~DataPoint~) double
        +input_dim() size_t
        +output_dim() size_t
        +num_layers() size_t
        +topology() vector~size_t~
        -compute_loss(dataset) double
        -gradient_descent_step(dataset, lr) void
        -forward_all(inputs) vector~vector~double~~
    }

    class DataPoint {
        +vector~double~ inputs
        +vector~double~ outputs
    }

    class KanTrainingConfig {
        +size_t max_iterations = 1000
        +double learning_rate = 0.01
        +double convergence_threshold = 1e-6
        +bool verbose = false
    }

    class KanTrainingResult {
        +size_t iterations_run
        +double final_loss
        +bool converged
        +milliseconds duration
    }

    class KANAdapter {
        -unordered_map~uint64_t, KANModuleEntry~ modules_
        -uint64_t next_module_id_
        +create_kan_module(input_dim, output_dim, num_knots) uint64_t
        +create_kan_module_multilayer(layer_dims, num_knots) uint64_t
        +train_kan_module(module_id, dataset, config) unique_ptr~FunctionHypothesis~
        +evaluate_kan_module(module_id, inputs) vector~double~
        +destroy_kan_module(module_id) void
        +has_module(module_id) bool
        +get_topology(module_id) vector~size_t~
    }

    KANModule "1" *-- "1..*" KANLayer : layers_
    KANLayer "1" *-- "n_in×n_out" KANNode : nodes_ (row-major)
    KANAdapter "1" o-- "*" KANModule : modules_
    KANModule ..> DataPoint : uses
    KANModule ..> KanTrainingConfig : uses
    KANModule ..> KanTrainingResult : returns
```

---

## 2. Memory Subsystem

LTM (knowledge graph) and STM (activation layer) with BrainController orchestration.

```mermaid
classDiagram
    class LongTermMemory {
        -unordered_map~ConceptId, ConceptInfo~ concepts_
        -ConceptId next_concept_id_
        -unordered_map~RelationId, RelationInfo~ relations_
        -unordered_map~ConceptId, vector~RelationId~~ outgoing_relations_
        -unordered_map~ConceptId, vector~RelationId~~ incoming_relations_
        -RelationId next_relation_id_
        -size_t total_relations_
        +store_concept(label, definition, epistemic) ConceptId
        +retrieve_concept(id) optional~ConceptInfo~
        +exists(id) bool
        +update_epistemic_metadata(id, new_meta) bool
        +invalidate_concept(id, trust) bool
        +get_concepts_by_type(type) vector~ConceptId~
        +get_concepts_by_status(status) vector~ConceptId~
        +get_active_concepts() vector~ConceptId~
        +add_relation(source, target, type, weight) RelationId
        +get_outgoing_relations(source) vector~RelationInfo~
        +get_incoming_relations(target) vector~RelationInfo~
        +total_relation_count() size_t
    }

    class ConceptInfo {
        +ConceptId id
        +string label
        +string definition
        +EpistemicMetadata epistemic
        +ConceptInfo() = delete
        +ConceptInfo(id, label, definition, epistemic)
    }

    class ShortTermMemory {
        -unordered_map~ContextId, Context~ contexts_
        -ContextId next_context_id_
        -double core_decay_rate_
        -double contextual_decay_rate_
        -double relation_decay_rate_
        +create_context() ContextId
        +destroy_context(context_id) void
        +activate_concept(ctx, cid, activation, class) void
        +activate_relation(ctx, source, target, type, activation) void
        +boost_concept(ctx, cid, delta) void
        +get_concept_activation(ctx, cid) double
        +get_active_concepts(ctx, threshold) vector~ConceptId~
        +decay_all(ctx, time_delta) void
        +export_state() STMSnapshotData
        +import_state(data) void
    }

    class BrainController {
        -unique_ptr~ShortTermMemory~ stm_
        -bool initialized_
        -unordered_map~ContextId, ThinkingState~ thinking_states_
        +initialize() bool
        +shutdown() void
        +create_context() ContextId
        +destroy_context(context_id) void
        +begin_thinking(context_id) void
        +end_thinking(context_id) void
        +activate_concept_in_context(ctx, cid, activation, class) void
        +decay_context(ctx, time_delta) void
        +get_stm() ShortTermMemory*
    }

    class STMEntry {
        +ConceptId concept_id
        +double activation
        +ActivationClass classification
        +uint64_t last_access_tick
    }

    class ActiveRelation {
        +ConceptId source
        +ConceptId target
        +RelationType type
        +double activation
    }

    BrainController "1" *-- "1" ShortTermMemory : stm_
    LongTermMemory "1" *-- "*" ConceptInfo : concepts_
    ShortTermMemory "1" *-- "*" STMEntry : contexts_[ctx].concepts
    ShortTermMemory "1" *-- "*" ActiveRelation : contexts_[ctx].relations
    ConceptInfo *-- EpistemicMetadata : epistemic
```

---

## 3. Epistemic System

Compile-time enforced epistemic metadata — the foundation of Brain19's knowledge integrity.

```mermaid
classDiagram
    class EpistemicMetadata {
        +EpistemicType type
        +EpistemicStatus status
        +double trust
        +EpistemicMetadata() = delete
        +EpistemicMetadata(type, status, trust)
        +is_valid() bool
        +is_active() bool
        +is_invalidated() bool
        +is_superseded() bool
        +is_contextual() bool
    }

    class EpistemicType {
        <<enumeration>>
        FACT
        DEFINITION
        THEORY
        HYPOTHESIS
        INFERENCE
        SPECULATION
    }

    class EpistemicStatus {
        <<enumeration>>
        ACTIVE
        CONTEXTUAL
        SUPERSEDED
        INVALIDATED
    }

    EpistemicMetadata --> EpistemicType : type
    EpistemicMetadata --> EpistemicStatus : status

    note for EpistemicMetadata "INVARIANTS:\n- No default constructor\n- trust ∈ [0.0, 1.0]\n- INVALIDATED → trust < 0.2\n- Knowledge NEVER deleted"
    note for EpistemicType "No UNKNOWN type!\nAbsence = compile error"
```

---

## 4. MicroModel Layer

Per-concept bilinear relevance models (430 parameters each).

```mermaid
classDiagram
    class MicroModel {
        -Mat10x10 W_
        -Vec10 b_
        -Vec10 e_init_
        -Vec10 c_init_
        -TrainingState state_
        +predict(e: Vec10, c: Vec10) double
        +train_step(e, c, target, config) double
        +train(samples: vector~TrainingSample~, config) MicroTrainingResult
        +to_flat(out: array~double,430~) void
        +from_flat(in: array~double,430~) void
    }

    class MicroModelRegistry {
        -unordered_map~ConceptId, MicroModel~ models_
        +create_model(cid) bool
        +get_model(cid) MicroModel*
        +has_model(cid) bool
        +remove_model(cid) bool
        +ensure_models_for(ltm) size_t
        +size() size_t
    }

    class EmbeddingManager {
        +get_embedding(cid) Vec10
        +get_relation_embedding(type) Vec10
        +set_embedding(cid, embedding) void
    }

    class MicroTrainer {
        +train_all(registry, embeddings, ltm) TrainerStats
        +train_single(cid, model, embeddings, ltm) MicroTrainingResult
        -generate_samples(cid, embeddings, ltm) vector~TrainingSample~
    }

    class RelevanceMap {
        +compute(cid, registry, embeddings, ltm) RelevanceMap
        +combine(maps, mode, weights) RelevanceMap
    }

    class TrainingSample {
        +Vec10 relation_embedding
        +Vec10 context_embedding
        +double target
    }

    class TrainingState {
        +Mat10x10 dW_momentum
        +Vec10 db_momentum
        +Mat10x10 dW_variance
        +Vec10 db_variance
        +double timestep
        +double last_loss
        +double best_loss
    }

    MicroModelRegistry "1" *-- "*" MicroModel : models_
    MicroModel *-- TrainingState : state_
    MicroTrainer ..> MicroModelRegistry : writes
    MicroTrainer ..> EmbeddingManager : reads
    MicroTrainer ..> LongTermMemory : reads
    RelevanceMap ..> MicroModelRegistry : reads
    RelevanceMap ..> EmbeddingManager : reads

    note for MicroModel "Forward: v = W·c + b, z = eᵀ·v, w = σ(z)\n430 params: 100(W) + 10(b) + 10(e) + 10(c) + 300(state)"
```

---

## 5. KAN-LLM Hybrid System

Phase 7: Linguistic hypotheses validated through KAN function approximation.

```mermaid
classDiagram
    class KanValidator {
        -Config config_
        -HypothesisTranslator translator_
        -EpistemicBridge bridge_
        +validate(proposal: HypothesisProposal) ValidationResult
        +translator() HypothesisTranslator
        +bridge() EpistemicBridge
    }

    class HypothesisTranslator {
        -Config config_
        +translate(proposal: HypothesisProposal) TranslationResult
        +detect_pattern(text) RelationshipPattern
        +detect_pattern_detailed(text) PatternDetectionResult
        +generate_training_data(pattern, n, min, max, hints) vector~DataPoint~
        +extract_numeric_hints(text) NumericHints
        -suggest_topology(pattern) vector~size_t~
        -suggest_config(pattern) KanTrainingConfig
    }

    class EpistemicBridge {
        -Config config_
        +assess(hypothesis, result, config, quality, n) EpistemicAssessment
        +check_interpretability(module) bool
        -compute_trust(mse, converged, speed, interp, quality, iters, n) double
        -determine_type(mse, converged) EpistemicType
        -determine_status(converged) EpistemicStatus
    }

    class DomainManager {
        -Config config_
        -unordered_map~DomainType, DomainInfo~ domains_
        +detect_domain(cid, ltm) DomainType
        +cluster_by_domain(concepts, ltm) map
        +get_domain_info(type) DomainInfo
        +get_domain_validator_config(type) KanValidator::Config
        +find_cross_domain_insights(concepts, ltm) vector~CrossDomainInsight~
    }

    class RefinementLoop {
        -KanValidator validator_
        -Config config_
        +run(initial, refiner) RefinementResult
        -build_residual_feedback(result, iteration) string
    }

    class RelationshipPattern {
        <<enumeration>>
        LINEAR
        POLYNOMIAL
        EXPONENTIAL
        PERIODIC
        THRESHOLD
        CONDITIONAL
        NOT_QUANTIFIABLE
    }

    class DomainType {
        <<enumeration>>
        PHYSICAL
        BIOLOGICAL
        SOCIAL
        ABSTRACT
        TEMPORAL
    }

    class ValidationResult {
        +bool validated
        +EpistemicAssessment assessment
        +RelationshipPattern pattern
        +shared_ptr~KANModule~ trained_module
        +string explanation
    }

    class EpistemicAssessment {
        +EpistemicMetadata metadata
        +double mse
        +bool converged
        +size_t iterations_used
        +double convergence_speed
        +string explanation
        +bool is_interpretable
        +DataQuality data_quality
    }

    class TranslationResult {
        +bool translatable
        +optional~KanTrainingProblem~ problem
        +RelationshipPattern detected_pattern
        +double pattern_confidence
        +string explanation
    }

    KanValidator *-- HypothesisTranslator : translator_
    KanValidator *-- EpistemicBridge : bridge_
    RefinementLoop *-- KanValidator : validator_
    KanValidator ..> ValidationResult : returns
    HypothesisTranslator ..> TranslationResult : returns
    EpistemicBridge ..> EpistemicAssessment : returns
    DomainManager ..> DomainType : uses
    HypothesisTranslator ..> RelationshipPattern : uses
```

---

## 6. Understanding Layer

Semantic analysis via Mini-LLMs with strict epistemic boundaries.

```mermaid
classDiagram
    class UnderstandingLayer {
        -UnderstandingLayerConfig config_
        -vector~unique_ptr~MiniLLM~~ mini_llms_
        -Statistics stats_
        +register_mini_llm(mini_llm: unique_ptr~MiniLLM~) void
        +analyze_meaning(concepts, ltm, stm, ctx) vector~MeaningProposal~
        +propose_hypotheses(evidence, ltm, stm, ctx) vector~HypothesisProposal~
        +find_analogies(set_a, set_b, ltm, stm, ctx) vector~AnalogyProposal~
        +check_contradictions(concepts, ltm, stm, ctx) vector~ContradictionProposal~
        +perform_understanding_cycle(seed, cd, ltm, stm, ctx) UnderstandingResult
    }

    class MiniLLM {
        <<abstract>>
        +extract_meaning(concepts, ltm, stm, ctx)* vector~MeaningProposal~
        +generate_hypotheses(evidence, ltm, stm, ctx)* vector~HypothesisProposal~
        +detect_analogies(a, b, ltm, stm, ctx)* vector~AnalogyProposal~
        +detect_contradictions(concepts, ltm, stm, ctx)* vector~ContradictionProposal~
    }

    class OllamaMiniLLM {
        -OllamaClient* client_
        +extract_meaning(...) vector~MeaningProposal~
        +generate_hypotheses(...) vector~HypothesisProposal~
        +detect_analogies(...) vector~AnalogyProposal~
        +detect_contradictions(...) vector~ContradictionProposal~
    }

    class MeaningProposal {
        +ConceptId concept_id
        +string meaning
        +double confidence
    }

    class HypothesisProposal {
        +uint64_t id
        +string hypothesis_text
        +vector~ConceptId~ evidence
        +double confidence
    }

    class AnalogyProposal {
        +ConceptId source_a
        +ConceptId source_b
        +string description
        +double confidence
    }

    class ContradictionProposal {
        +ConceptId concept_a
        +ConceptId concept_b
        +string description
        +double severity
    }

    UnderstandingLayer "1" *-- "*" MiniLLM : mini_llms_
    MiniLLM <|-- OllamaMiniLLM : implements
    UnderstandingLayer ..> MeaningProposal : generates
    UnderstandingLayer ..> HypothesisProposal : generates
    UnderstandingLayer ..> AnalogyProposal : generates
    UnderstandingLayer ..> ContradictionProposal : generates

    note for UnderstandingLayer "CONTRACTS:\n- READ-ONLY LTM access\n- All outputs are HYPOTHESIS\n- Trust ceiling: max 0.3-0.5\n- No epistemic writes"
```

---

## 7. Evolution System

Dynamic concept generation and epistemic lifecycle management.

```mermaid
classDiagram
    class PatternDiscovery {
        -const LongTermMemory& ltm_
        +find_clusters(min_size) vector~DiscoveredPattern~
        +find_hierarchies(min_depth) vector~DiscoveredPattern~
        +find_bridges() vector~DiscoveredPattern~
        +find_cycles(max_length) vector~DiscoveredPattern~
        +find_gaps() vector~DiscoveredPattern~
        +discover_all() vector~DiscoveredPattern~
        -build_graph() AdjacencyGraph
        -find_components(graph) vector~vector~ConceptId~~
        -dfs_cycles(...) bool
    }

    class EpistemicPromotion {
        -LongTermMemory& ltm_
        +evaluate_all() vector~PromotionCandidate~
        +evaluate(id) optional~PromotionCandidate~
        +promote(id, new_type, new_trust) bool
        +demote(id, new_type, new_trust, reason) bool
        +confirm_as_fact(id, trust, human_note) bool
        +run_maintenance() MaintenanceResult
        -check_speculation_to_hypothesis(id, info)
        -check_hypothesis_to_theory(id, info)
        -check_theory_to_fact(id, info)
        -check_demotion(id, info)
        -count_supporting_relations(id) size_t
        -has_contradictions(id) bool
    }

    class ConceptProposer {
        -const LongTermMemory& ltm_
        +from_curiosity(triggers) vector~ConceptProposal~
        +from_relevance_anomalies(map, threshold) vector~ConceptProposal~
        +from_analogies(analogies) vector~ConceptProposal~
        +rank_proposals(proposals, max_k) vector~ConceptProposal~
        -concept_exists_similar(label) bool
        -compute_quality_score(proposal) double
    }

    class ConceptProposal {
        +string label
        +string description
        +EpistemicType initial_type
        +double initial_trust
        +string source
        +vector~ConceptId~ evidence
        +string reasoning
    }

    class DiscoveredPattern {
        +string description
        +vector~ConceptId~ involved_concepts
        +double confidence
        +string pattern_type
    }

    class PromotionCandidate {
        +ConceptId id
        +EpistemicType current_type
        +EpistemicType proposed_type
        +double current_trust
        +double proposed_trust
        +string reasoning
        +vector~ConceptId~ evidence
        +bool requires_human_review
    }

    PatternDiscovery ..> DiscoveredPattern : returns
    EpistemicPromotion ..> PromotionCandidate : returns
    ConceptProposer ..> ConceptProposal : returns

    note for ConceptProposal "INVARIANT:\ninitial_trust CAPPED at 0.5\ninitial_type must be SPECULATION or HYPOTHESIS"
    note for EpistemicPromotion "INVARIANT:\nTHEORY → FACT requires human review\nNEVER automatic FACT promotion"
```

---

## 8. Ingestion Pipeline

From raw text/JSON/CSV to knowledge graph, with human review gate.

```mermaid
classDiagram
    class IngestionPipeline {
        -LongTermMemory& ltm_
        -ProposalQueue queue_
        -TrustTagger tagger_
        -KnowledgeIngestor ingestor_
        -TextChunker chunker_
        -EntityExtractor entity_extractor_
        -RelationExtractor relation_extractor_
        -TrustCategory default_trust_
        +ingest_json(json_str, auto_approve) IngestionResult
        +ingest_csv(concepts_csv, relations_csv, auto_approve) IngestionResult
        +ingest_text(text, source_ref, auto_approve) IngestionResult
        +get_queue() ProposalQueue&
        +commit_approved() IngestionResult
    }

    class TextChunker {
        +chunk_text(text) vector~TextChunk~
    }

    class EntityExtractor {
        +extract_from_chunks(chunks) vector~ExtractedEntity~
    }

    class RelationExtractor {
        +extract_relations(text, entities) vector~ExtractedRelation~
    }

    class TrustTagger {
        +suggest_from_text(text) TrustAssignment
    }

    class ProposalQueue {
        +enqueue(proposal) void
        +review(id, decision) void
        +pop_approved() vector~IngestProposal~
    }

    class KnowledgeIngestor {
        +ingest_structured(data) vector~IngestProposal~
    }

    class IngestionResult {
        +bool success
        +string error_message
        +size_t chunks_created
        +size_t entities_extracted
        +size_t relations_extracted
        +size_t proposals_created
        +size_t proposals_approved
        +size_t concepts_stored
        +size_t relations_stored
        +vector~ConceptId~ stored_concept_ids
    }

    IngestionPipeline *-- TextChunker
    IngestionPipeline *-- EntityExtractor
    IngestionPipeline *-- RelationExtractor
    IngestionPipeline *-- TrustTagger
    IngestionPipeline *-- ProposalQueue
    IngestionPipeline *-- KnowledgeIngestor
    IngestionPipeline ..> IngestionResult : returns
    IngestionPipeline --> LongTermMemory : writes to
```

---

## 9. Streams & Concurrency

Multi-threaded thinking streams with lock-free communication.

```mermaid
classDiagram
    class StreamOrchestrator {
        -SharedLTM& ltm_
        -SharedSTM& stm_
        -SharedRegistry& registry_
        -SharedEmbeddings& embeddings_
        -StreamConfig config_
        -unordered_map~StreamId, unique_ptr~ThinkStream~~ streams_
        -atomic~StreamId~ next_id_
        -thread monitor_thread_
        -atomic~bool~ monitor_running_
        -OrchestratorMetrics metrics_
        +create_stream() StreamId
        +create_stream(subsystems) StreamId
        +start_stream(id) bool
        +stop_stream(id) void
        +destroy_stream(id) void
        +start_all() void
        +shutdown(timeout) bool
        +auto_scale() void
        +health_check() vector~StreamHealth~
        +distribute_task(task) bool
    }

    class ThinkStream {
        -StreamId id_
        -atomic~ContextId~ context_id_
        -SharedLTM& ltm_
        -SharedSTM& stm_
        -SharedRegistry& registry_
        -SharedEmbeddings& embeddings_
        -StreamConfig config_
        -atomic~StreamState~ state_
        -atomic~bool~ stop_requested_
        -MPMCQueue~ThinkTask~ work_queue_
        -StreamMetrics metrics_
        -thread thread_
        +start() bool
        +stop() void
        +join(timeout) bool
        +push_task(task) bool
        +id() StreamId
        +state() StreamState
        +metrics() StreamMetrics
        -run() void
        -tick() void
        -do_spreading() void
        -do_salience() void
        -do_curiosity() void
        -do_understanding() void
        -backoff(idle_count) void
    }

    class StreamState {
        <<enumeration>>
        Created
        Starting
        Running
        Paused
        Stopping
        Stopped
        Error
    }

    class StreamMetrics {
        +atomic~uint64_t~ total_ticks
        +atomic~uint64_t~ spreading_ticks
        +atomic~uint64_t~ salience_ticks
        +atomic~uint64_t~ curiosity_ticks
        +atomic~uint64_t~ understanding_ticks
        +atomic~uint64_t~ idle_ticks
        +atomic~uint64_t~ errors
        +atomic~int64_t~ last_tick_epoch_us
    }

    class SharedLTM {
        -shared_mutex mutex_
        -LongTermMemory& ltm_
        +shared_lock for reads
        +unique_lock for writes
    }

    class SharedSTM {
        -shared_mutex global_mutex_
        -per-context shared_mutex
        -ShortTermMemory& stm_
    }

    class SharedRegistry {
        -shared_mutex mutex_
        -MicroModelRegistry& registry_
        +ModelGuard RAII per-model lock
    }

    class SharedEmbeddings {
        -shared_mutex mutex_
        -EmbeddingManager& embeddings_
        +fast-path shared_lock
    }

    StreamOrchestrator "1" *-- "*" ThinkStream : streams_
    ThinkStream --> StreamState : state_
    ThinkStream *-- StreamMetrics : metrics_
    ThinkStream --> SharedLTM : reads
    ThinkStream --> SharedSTM : reads/writes
    ThinkStream --> SharedRegistry : reads
    ThinkStream --> SharedEmbeddings : reads
```

---

## 10. Persistence Layer

Binary persistence, WAL, and checkpoint management.

```mermaid
classDiagram
    class PersistentLTM {
        +save(ltm, path) bool
        +load_into(ltm, path) bool
    }

    class WALWriter {
        +log_store_concept(cid, label, def, meta) void
        +log_add_relation(rid, source, target, type, weight) void
        +flush() void
    }

    class WALReader {
        +replay_into(ltm) size_t
        +replay_after(timestamp, ltm) size_t
    }

    class STMSnapshot {
        +save(stm, path) bool
        +load_into(stm, path) bool
    }

    class CheckpointManager {
        -string data_dir_
        -size_t max_checkpoints_
        +create_checkpoint(tag) string
        +list_checkpoints() vector~string~
        +rotate_checkpoints(max) void
    }

    class CheckpointRestore {
        +restore(checkpoint_dir, ltm, stm, registry) bool
    }

    CheckpointManager ..> PersistentLTM : uses
    CheckpointManager ..> STMSnapshot : uses
    CheckpointManager ..> WALWriter : uses
    CheckpointRestore ..> PersistentLTM : uses
    CheckpointRestore ..> STMSnapshot : uses
    CheckpointRestore ..> WALReader : uses
```

---

## 11. Core Orchestration

SystemOrchestrator owns all subsystems; Brain19App provides the user interface.

```mermaid
classDiagram
    class SystemOrchestrator {
        -Config config_
        -atomic~bool~ running_
        -int init_stage_
        -recursive_mutex subsystem_mtx_
        -unique_ptr~LongTermMemory~ ltm_
        -unique_ptr~PersistentLTM~ persistent_ltm_
        -unique_ptr~WALWriter~ wal_
        -unique_ptr~BrainController~ brain_
        -unique_ptr~EmbeddingManager~ embeddings_
        -unique_ptr~MicroModelRegistry~ registry_
        -unique_ptr~MicroTrainer~ trainer_
        -unique_ptr~CognitiveDynamics~ cognitive_
        -unique_ptr~CuriosityEngine~ curiosity_
        -unique_ptr~KANAdapter~ kan_adapter_
        -unique_ptr~UnderstandingLayer~ understanding_
        -unique_ptr~KanValidator~ kan_validator_
        -unique_ptr~DomainManager~ domain_manager_
        -unique_ptr~RefinementLoop~ refinement_loop_
        -unique_ptr~IngestionPipeline~ ingestion_
        -unique_ptr~ChatInterface~ chat_
        -unique_ptr~StreamOrchestrator~ stream_orch_
        -unique_ptr~PatternDiscovery~ pattern_discovery_
        -unique_ptr~EpistemicPromotion~ epistemic_promotion_
        -unique_ptr~ConceptProposer~ concept_proposer_
        -unique_ptr~ThinkingPipeline~ thinking_
        -thread periodic_thread_
        +initialize() bool
        +shutdown() void
        +ask(question) ChatResponse
        +ingest_text(text, auto_approve) IngestionResult
        +ingest_wikipedia(url) IngestionResult
        +create_checkpoint(tag) void
        +restore_checkpoint(dir) bool
        +run_thinking_cycle(seeds) ThinkingResult
        +run_periodic_maintenance() void
        +concept_count() size_t
        +relation_count() size_t
    }

    class Brain19App {
        -SystemOrchestrator orchestrator_
        +run_interactive() int
        +run_command(command, arg) int
        -cmd_ask(question) void
        -cmd_ingest(text) void
        -cmd_import(url) void
        -cmd_status() void
        -cmd_streams() void
        -cmd_checkpoint(tag) void
        -cmd_concepts() void
        -cmd_explain(id) void
        -cmd_think(label) void
        -cmd_help() void
    }

    class ThinkingPipeline {
        -Config config_
        +execute(seeds, ctx, ltm, stm, brain, cognitive, ...) ThinkingResult
        -step_activate_seeds(...)
        -step_spreading(...)
        -step_salience(...)
        -step_relevance(...)
        -step_thought_paths(...)
        -step_curiosity(...)
        -step_understanding(...)
        -step_kan_validation(...)
    }

    class ThinkingResult {
        +vector~ConceptId~ activated_concepts
        +vector~SalienceScore~ top_salient
        +vector~ThoughtPath~ best_paths
        +vector~CuriosityTrigger~ curiosity_triggers
        +RelevanceMap combined_relevance
        +UnderstandingResult understanding
        +vector~ValidationResult~ validated_hypotheses
        +size_t steps_completed
        +double total_duration_ms
    }

    Brain19App *-- SystemOrchestrator : orchestrator_
    SystemOrchestrator *-- ThinkingPipeline : thinking_
    ThinkingPipeline ..> ThinkingResult : returns
```

---

*Generated from actual code in `backend/`. Updated: 2026-02-12.*
