# Brain19 — Architecture Diagrams

> Comprehensive UML diagrams for the Brain19 C++20 Cognitive Architecture.
> All class names, method signatures, and data flows match the actual code in `backend/`.
> Updated: 2026-02-12

---

## Table of Contents

1. [System Architecture — Component Diagram](#1-system-architecture--component-diagram)
2. [Thinking Cycle — Single Tick](#2-thinking-cycle--single-tick)
3. [ThinkingPipeline — 10-Step Orchestrated Cycle](#3-thinkingpipeline--10-step-orchestrated-cycle)
4. [Learning Flow — Knowledge Ingestion](#4-learning-flow--knowledge-ingestion)
5. [Query/Chat Flow — User Question](#5-querychat-flow--user-question)
6. [Understanding Cycle — Semantic Analysis](#6-understanding-cycle--semantic-analysis)
7. [KAN-LLM Hybrid Validation — Phase 7](#7-kan-llm-hybrid-validation--phase-7)
8. [Refinement Loop — Bidirectional LLM↔KAN Dialog](#8-refinement-loop--bidirectional-llmkan-dialog)
9. [Dynamic Concept Evolution — Phase 6](#9-dynamic-concept-evolution--phase-6)
10. [Curiosity → Action](#10-curiosity--action)
11. [Component Dependency — Ownership & Access](#11-component-dependency--ownership--access)
12. [Data Lifecycle — Concepts, MicroModels, STM](#12-data-lifecycle--concepts-micromodels-stm)
13. [Multi-Stream Architecture](#13-multi-stream-architecture)
14. [Multi-Stream Thinking Cycle](#14-multi-stream-thinking-cycle)
15. [System Initialization Sequence](#15-system-initialization-sequence)
16. [Checkpoint & Restore Flow](#16-checkpoint--restore-flow)
17. [Full-Stack Deployment](#17-full-stack-deployment)

---

## 1. System Architecture — Component Diagram

Top-level view of all Brain19 subsystems and their interactions. The SystemOrchestrator owns and coordinates 14 subsystem groups.

```mermaid
flowchart TB
    subgraph External["External Interfaces"]
        USER[User / CLI REPL]
        API[FastAPI Server<br/>:8019 REST + WebSocket]
        VIZ[React Frontend<br/>:3019 Brain19Visualizer]
    end

    subgraph Core["Core Layer"]
        SO[SystemOrchestrator<br/>Lifecycle, Config, Mutex]
        APP[Brain19App<br/>REPL, Commands]
        TP[ThinkingPipeline<br/>10-Step Cycle]
    end

    subgraph Memory["Memory Subsystem"]
        LTM[LongTermMemory<br/>Knowledge Graph<br/>Concepts + Relations]
        STM[ShortTermMemory<br/>Activation Layer<br/>Contexts + Decay]
        BC[BrainController<br/>Context Management]
    end

    subgraph Cognitive["Cognitive Subsystem"]
        CD[CognitiveDynamics<br/>Spreading Activation<br/>Salience + Focus<br/>Thought Paths]
        CE[CuriosityEngine<br/>Trigger Generation]
    end

    subgraph MicroML["MicroModel Layer"]
        MM[MicroModel<br/>430 params each<br/>Bilinear: v=Wc+b]
        MR[MicroModelRegistry<br/>1 model per concept]
        EM[EmbeddingManager<br/>10D embeddings]
        MT[MicroTrainer<br/>Adam optimizer]
        RM[RelevanceMap<br/>Relevance scoring]
    end

    subgraph KAN["KAN Subsystem"]
        KN[KANNode<br/>B-spline univariate]
        KL[KANLayer<br/>n_in × n_out grid]
        KM[KANModule<br/>Multi-layer network]
        KA[KANAdapter<br/>Module lifecycle]
    end

    subgraph Hybrid["KAN-LLM Hybrid (Phase 7)"]
        HT[HypothesisTranslator<br/>NLP → KAN Problem]
        EB[EpistemicBridge<br/>MSE → Trust mapping]
        KV[KanValidator<br/>End-to-end validation]
        DM[DomainManager<br/>Domain clustering]
        RL[RefinementLoop<br/>LLM↔KAN dialog]
    end

    subgraph Understanding["Understanding Layer"]
        UL[UnderstandingLayer<br/>Semantic Analysis]
        MLLM[MiniLLM<br/>Registered models]
        OML[OllamaMiniLLM<br/>Ollama-backed]
    end

    subgraph Evolution["Evolution (Phase 6)"]
        PD[PatternDiscovery<br/>Graph analysis]
        EP[EpistemicPromotion<br/>Status lifecycle]
        CP[ConceptProposer<br/>New concept generation]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        IP[IngestionPipeline<br/>JSON/CSV/Text]
        TC[TextChunker]
        EEX[EntityExtractor]
        REX[RelationExtractor]
        TT[TrustTagger]
        PQ[ProposalQueue]
    end

    subgraph Importers["External Importers"]
        WI[WikipediaImporter<br/>HTTP + parse]
        SI[ScholarImporter<br/>Paper search]
        HC[HttpClient<br/>libcurl wrapper]
    end

    subgraph LLMLayer["LLM Layer"]
        OC[OllamaClient<br/>HTTP to Ollama]
        CI[ChatInterface<br/>Verbalization tool]
    end

    subgraph Persistence["Persistence Layer"]
        PLTM[PersistentLTM<br/>Binary format]
        WAL[WAL Writer/Reader<br/>Write-Ahead Log]
        SNAP[STMSnapshot<br/>Binary snapshot]
        CKPT[CheckpointManager<br/>Full state save]
        REST[CheckpointRestore<br/>Full state load]
    end

    subgraph Streams["Multi-Stream Architecture"]
        SORCH[StreamOrchestrator<br/>N streams, auto-scale]
        TS[ThinkStream<br/>Autonomous thread]
        SCHED[StreamScheduler<br/>Category-based]
        SMON[StreamMonitor<br/>Health + metrics]
    end

    subgraph Concurrent["Concurrency Layer"]
        SLTM[SharedLTM<br/>shared_mutex]
        SSTM[SharedSTM<br/>per-context mutex]
        SREG[SharedRegistry<br/>shared_mutex + ModelGuard]
        SEMB[SharedEmbeddings<br/>shared_mutex]
        LH[LockHierarchy<br/>Deadlock prevention]
    end

    subgraph Bootstrap["Bootstrap"]
        FC[FoundationConcepts<br/>Seed knowledge]
        BI[BootstrapInterface<br/>Init workflow]
        CA[ContextAccumulator<br/>Context building]
    end

    %% External connections
    USER --> APP
    API --> SO
    VIZ --> API

    %% Core connections
    APP --> SO
    SO --> TP
    SO --> Memory
    SO --> Cognitive
    SO --> MicroML
    SO --> KAN
    SO --> Hybrid
    SO --> Understanding
    SO --> Evolution
    SO --> Ingestion
    SO --> LLMLayer
    SO --> Persistence
    SO --> Streams
    SO --> Bootstrap

    %% Memory internals
    BC --> STM

    %% Cognitive reads
    CD -.-> LTM
    CD --> STM

    %% MicroModel connections
    MR --> MM
    MT --> MR
    MT -.-> LTM
    RM --> MR
    RM --> EM

    %% KAN hierarchy
    KM --> KL
    KL --> KN
    KA --> KM

    %% Hybrid connections
    KV --> HT
    KV --> EB
    KV --> KM
    RL --> KV
    DM -.-> LTM

    %% Understanding
    UL --> MLLM
    OML --> OC

    %% Evolution reads
    PD -.-> LTM
    EP --> LTM
    CP -.-> LTM

    %% Ingestion pipeline
    IP --> TC
    IP --> EEX
    IP --> REX
    IP --> TT
    IP --> PQ
    IP --> LTM

    %% Importers
    WI --> HC
    SI --> HC

    %% LLM
    CI --> OC
    CI -.-> LTM

    %% Persistence
    PLTM --> LTM
    WAL --> LTM

    %% Streams use concurrent wrappers
    SORCH --> TS
    TS --> SLTM
    TS --> SSTM
    TS --> SREG
    TS --> SEMB
    SLTM --> LTM
    SSTM --> STM
    SREG --> MR
    SEMB --> EM

    %% Styles
    style SO fill:#2563eb,color:#fff
    style LTM fill:#16a34a,color:#fff
    style STM fill:#d97706,color:#fff
    style CD fill:#7c3aed,color:#fff
    style KV fill:#dc2626,color:#fff
    style TP fill:#2563eb,color:#fff
    style SORCH fill:#0891b2,color:#fff
```

---

## 2. Thinking Cycle — Single Tick

A single thinking tick: BrainController orchestrates, CognitiveDynamics computes, CuriosityEngine observes.

```mermaid
sequenceDiagram
    participant Caller
    participant BC as BrainController
    participant STM as ShortTermMemory
    participant CD as CognitiveDynamics
    participant LTM as LongTermMemory
    participant CE as CuriosityEngine

    Caller->>BC: begin_thinking(context_id)
    Note over BC: ThinkingState.is_thinking = true

    Caller->>BC: activate_concept_in_context(ctx, cid, activation, class)
    BC->>STM: activate_concept(ctx, cid, activation, class)

    Caller->>CD: spread_activation(source, activation, ctx, ltm, stm)
    CD->>LTM: get_outgoing_relations(source)
    LTM-->>CD: vector<RelationInfo>
    loop Depth ≤ max_depth (3), per relation
        CD->>LTM: retrieve_concept(target) → EpistemicMetadata.trust
        Note over CD: activation(T) += activation(S) × weight × trust × damping^depth
        CD->>STM: activate_concept(ctx, target, computed_activation, CONTEXTUAL)
        CD->>CD: spread_recursive(target, ..., depth+1, visited)
    end
    CD-->>Caller: SpreadingStats

    Caller->>CD: compute_salience_batch(concepts, ctx, ltm, stm, tick)
    Note over CD: score = activation×0.4 + recency + connectivity×0.2 + trust×0.3
    CD->>STM: get_concept_activation(ctx, cid)
    CD->>LTM: get_relation_count(cid)
    CD->>LTM: retrieve_concept(cid) → epistemic.trust
    CD-->>Caller: vector<SalienceScore>

    Caller->>CD: get_top_k_salient(candidates, k, ctx, ltm, stm)
    Note over CD: Top-K into working memory (Focus Set)
    Caller->>CD: focus_on(ctx, cid, boost)
    CD-->>Caller: FocusEntry[]

    Caller->>CE: observe_and_generate_triggers(observations)
    Note over CE: detect_shallow_relations() / detect_low_exploration()
    CE-->>Caller: vector<CuriosityTrigger>

    Caller->>BC: decay_context(ctx, time_delta)
    BC->>STM: decay_all(ctx, time_delta)

    Caller->>BC: end_thinking(context_id)
    Note over BC: ThinkingState.is_thinking = false
```

---

## 3. ThinkingPipeline — 10-Step Orchestrated Cycle

The heart of Brain19. SystemOrchestrator calls ThinkingPipeline::execute() which runs all 10 steps sequentially.

```mermaid
sequenceDiagram
    participant SO as SystemOrchestrator
    participant TP as ThinkingPipeline
    participant BC as BrainController
    participant STM as ShortTermMemory
    participant CD as CognitiveDynamics
    participant LTM as LongTermMemory
    participant CE as CuriosityEngine
    participant MR as MicroModelRegistry
    participant EM as EmbeddingManager
    participant UL as UnderstandingLayer
    participant KV as KanValidator

    SO->>TP: execute(seeds, context, ltm, stm, brain, cognitive, ...)
    Note over TP: Step 1: Activate Seeds

    TP->>BC: activate_concept_in_context(ctx, seed, 0.8, CORE)
    BC->>STM: activate_concept(ctx, seed, 0.8, CORE)

    Note over TP: Step 2: Spreading Activation
    TP->>CD: spread_activation_multi(seeds, 0.8, ctx, ltm, stm)
    CD-->>TP: SpreadingStats {concepts_activated, max_depth_reached}

    Note over TP: Step 3: Compute Salience + Focus
    TP->>CD: get_top_k_salient(active, top_k=10, ctx, ltm, stm)
    CD-->>TP: vector<SalienceScore> top_salient

    Note over TP: Step 4: Generate RelevanceMaps (MicroModels)
    loop For each salient concept (max 5)
        TP->>MR: get_model(cid)
        TP->>EM: get_embedding(cid)
        Note over TP: v = W·c + b, z = eᵀ·v, w = σ(z)
    end
    TP-->>TP: individual RelevanceMaps

    Note over TP: Step 5: Combine RelevanceMaps (overlay for creativity)
    TP->>TP: RelevanceMap::combine(maps, WEIGHTED_AVERAGE)
    Note over TP: Cross-perspective overlay enables creative connections

    Note over TP: Step 6: Find ThoughtPaths
    TP->>CD: find_best_paths(seed, ctx, ltm, stm)
    CD-->>TP: vector<ThoughtPath> best_paths

    Note over TP: Step 7: Run CuriosityEngine
    TP->>CE: observe_and_generate_triggers(observations)
    CE-->>TP: vector<CuriosityTrigger>

    Note over TP: Step 8: Run UnderstandingLayer (MiniLLMs)
    alt understanding != nullptr
        TP->>UL: perform_understanding_cycle(seed, cd, ltm, stm, ctx)
        UL-->>TP: UnderstandingResult {meanings, hypotheses, analogies, contradictions}
    end

    Note over TP: Step 9: KAN-LLM Validation (Phase 7 Hybrid)
    alt kan_validator != nullptr
        loop For each HypothesisProposal
            TP->>KV: validate(hypothesis)
            KV-->>TP: ValidationResult {validated, assessment, trained_module}
        end
    end

    Note over TP: Step 10: Return complete result
    TP-->>SO: ThinkingResult {activated, salient, paths, triggers, relevance, understanding, validated}
```

---

## 4. Learning Flow — Knowledge Ingestion

From external sources through the IngestionPipeline to LTM — with human review.

```mermaid
sequenceDiagram
    participant User
    participant WI as WikipediaImporter
    participant SI as ScholarImporter
    participant IP as IngestionPipeline
    participant TC as TextChunker
    participant EE as EntityExtractor
    participant RE as RelationExtractor
    participant TT as TrustTagger
    participant PQ as ProposalQueue
    participant Human
    participant LTM as LongTermMemory
    participant MT as MicroTrainer
    participant MR as MicroModelRegistry

    User->>WI: import_article(title, lang)
    WI-->>User: unique_ptr<KnowledgeProposal>
    Note over WI: Proposals only, NO LTM write access

    User->>SI: search_papers(query, limit)
    SI-->>User: vector<unique_ptr<KnowledgeProposal>>

    User->>IP: ingest_text(text, source_ref, auto_approve=false)
    IP->>TC: chunk_text(text)
    TC-->>IP: vector<TextChunk>

    IP->>EE: extract_from_chunks(chunks)
    Note over EE: Capitalization, Quotes, Definition patterns, Frequency
    EE-->>IP: vector<ExtractedEntity>

    IP->>RE: extract_relations(text, entities)
    Note over RE: "X is a Y" → IS_A, "X causes Y" → CAUSES, etc.
    RE-->>IP: vector<ExtractedRelation>

    IP->>TT: suggest_from_text(text)
    Note over TT: Hedging → SPECULATION, Certainty → FACT, Citations → THEORY
    TT-->>IP: TrustAssignment

    IP->>PQ: enqueue(IngestProposal)
    Note over PQ: ProposalStatus::PENDING

    Human->>PQ: review(id, ReviewDecision::approve())
    Note over PQ: ProposalStatus::APPROVED

    IP->>IP: commit_approved()
    IP->>PQ: pop_approved()
    PQ-->>IP: vector<IngestProposal>

    loop Per approved proposal
        IP->>LTM: store_concept(label, definition, EpistemicMetadata)
        Note over LTM: ConceptInfo() = delete! EpistemicMetadata REQUIRED
        LTM-->>IP: ConceptId
        IP->>LTM: add_relation(source, target, type, weight)
    end

    User->>MT: train_all(registry, embeddings, ltm)
    MT->>MT: generate_samples(cid, embeddings, ltm)
    Note over MT: Positives from relations, 3x negatives per positive
    MT->>MR: get_model(cid) → MicroModel
    MT->>MR: model.train(samples, config)
    Note over MT: Adam optimizer, MSE loss, convergence check
    MT-->>User: TrainerStats
```

---

## 5. Query/Chat Flow — User Question

From user question through keyword search, spreading activation, and MicroModel relevance to LLM answer.

```mermaid
sequenceDiagram
    participant User
    participant CI as ChatInterface
    participant LTM as LongTermMemory
    participant CD as CognitiveDynamics
    participant STM as ShortTermMemory
    participant RM as RelevanceMap
    participant MR as MicroModelRegistry
    participant EM as EmbeddingManager
    participant OC as OllamaClient

    User->>CI: ask(question, ltm)

    CI->>CI: find_relevant_concepts(question, ltm)
    CI->>LTM: get_all_concept_ids()
    Note over CI: Keyword matching against label/definition
    CI-->>CI: vector<ConceptInfo> relevant

    CI->>CD: spread_activation_multi(sources, activation, ctx, ltm, stm)
    Note over CD: Spreading along relations, trust-weighted
    CD->>STM: activate_concept(ctx, target, activation, CONTEXTUAL)
    CD-->>CI: SpreadingStats

    loop Per relevant concept
        CI->>RM: compute(cid, registry, embeddings, ltm, rel_type, context)
        RM->>MR: get_model(cid)
        RM->>EM: get_embedding(cid)
        Note over RM: v = W·c + b, z = eᵀ·v, w = σ(z)
        RM-->>CI: RelevanceMap (scores per concept)
    end

    CI->>RM: combine(maps, mode, weights)
    Note over RM: ADDITION / MAX / WEIGHTED_AVERAGE

    CI->>CI: build_epistemic_context(concepts)
    Note over CI: Trust levels in prompt:<br/>FACT (trust 0.98), THEORY (0.9), HYPOTHESIS (0.5)

    CI->>CI: build_system_prompt()
    Note over CI: Verbalize ONLY existing knowledge.<br/>Mark uncertainties explicitly.

    CI->>OC: generate(prompt, context)
    OC-->>CI: OllamaResponse

    CI-->>User: ChatResponse {answer, referenced_concepts, epistemic_note}
```

---

## 6. Understanding Cycle — Semantic Analysis

UnderstandingLayer uses CognitiveDynamics for focus and Mini-LLMs for semantic proposals.

```mermaid
sequenceDiagram
    participant Caller
    participant UL as UnderstandingLayer
    participant CD as CognitiveDynamics
    participant LTM as LongTermMemory
    participant STM as ShortTermMemory
    participant ML as MiniLLM (registered)

    Caller->>UL: perform_understanding_cycle(seed_concept, cd, ltm, stm, ctx)

    UL->>CD: spread_activation(seed_concept, activation, ctx, ltm, stm)
    CD-->>UL: SpreadingStats

    UL->>CD: get_top_k_salient(candidates, k, ctx, ltm, stm)
    CD-->>UL: vector<SalienceScore> (focused concepts)

    UL->>CD: focus_on(ctx, cid, boost)

    loop For each registered MiniLLM
        UL->>ML: extract_meaning(active_concepts, ltm, stm, ctx)
        ML-->>UL: vector<MeaningProposal>
        Note over ML: All proposals: EpistemicType::HYPOTHESIS

        UL->>ML: generate_hypotheses(evidence_concepts, ltm, stm, ctx)
        ML-->>UL: vector<HypothesisProposal>

        UL->>ML: detect_analogies(set_a, set_b, ltm, stm, ctx)
        ML-->>UL: vector<AnalogyProposal>

        UL->>ML: detect_contradictions(active_concepts, ltm, stm, ctx)
        ML-->>UL: vector<ContradictionProposal>
    end

    Note over UL: Filter thresholds:<br/>meaning ≥ 0.3, hypothesis ≥ 0.2,<br/>analogy ≥ 0.4, contradiction ≥ 0.5

    Note over UL: Trust ceiling: all proposals max 0.3-0.5<br/>All outputs remain HYPOTHESIS

    UL-->>Caller: UnderstandingResult {meaning, hypothesis, analogy, contradiction}

    Note over Caller: BrainController → Epistemic Core<br/>decides acceptance/rejection
```

---

## 7. KAN-LLM Hybrid Validation — Phase 7

The complete LLM → KAN validation pipeline. HypothesisTranslator converts linguistic hypotheses to KAN training problems, KANModule trains, EpistemicBridge assigns trust.

```mermaid
sequenceDiagram
    participant UL as UnderstandingLayer
    participant KV as KanValidator
    participant HT as HypothesisTranslator
    participant KM as KANModule
    participant EB as EpistemicBridge

    UL->>KV: validate(HypothesisProposal)

    Note over KV: Step 1: Translate hypothesis to KAN problem
    KV->>HT: translate(proposal)

    HT->>HT: detect_pattern_detailed(hypothesis_text)
    Note over HT: NLP-lite: keyword matching<br/>LINEAR, POLYNOMIAL, EXPONENTIAL,<br/>PERIODIC, THRESHOLD, CONDITIONAL

    HT->>HT: extract_numeric_hints(text)
    Note over HT: Extract slopes, intercepts, ranges

    alt Pattern is NOT_QUANTIFIABLE
        HT-->>KV: TranslationResult {translatable: false}
        KV-->>UL: ValidationResult {validated: false, "Not quantifiable"}
    else Pattern detected (confidence ≥ 0.3)
        HT->>HT: generate_training_data(pattern, hints)
        Note over HT: Generate 20-100 synthetic data points<br/>with hypothesis-specific parameters

        HT->>HT: suggest_topology(pattern)
        Note over HT: LINEAR → [1,5,1]<br/>POLYNOMIAL → [1,8,5,1]<br/>PERIODIC → [1,10,5,1]

        HT-->>KV: TranslationResult {translatable: true, KanTrainingProblem}
    end

    Note over KV: Step 2: Train KAN
    KV->>KM: KANModule(suggested_topology, num_knots)
    KV->>KM: train(training_data, config)
    KM-->>KV: KanTrainingResult {iterations, final_loss, converged}

    Note over KV: Step 3: Epistemic Assessment
    KV->>EB: assess(hypothesis, training_result, config, data_quality, num_points)

    Note over EB: MSE Mapping:<br/>MSE < 0.01 → THEORY (trust 0.7-0.9)<br/>MSE < 0.1  → HYPOTHESIS (trust 0.4-0.6)<br/>MSE ≥ 0.1  → SPECULATION (trust 0.1-0.3)<br/>Not converged → INVALIDATED (trust 0.05)

    Note over EB: H2 Modifiers:<br/>Synthetic data → max trust 0.6<br/>Trivial convergence (<10 epochs) → penalty 0.15<br/>Trust > 0.5 requires ≥ 50 data points

    EB->>EB: check_interpretability(module)
    Note over EB: Interpretability bonus: +0.05 trust

    EB-->>KV: EpistemicAssessment {metadata, mse, converged, explanation}

    KV-->>UL: ValidationResult {validated, assessment, pattern, trained_module}
```

---

## 8. Refinement Loop — Bidirectional LLM↔KAN Dialog

The iterative refinement process: LLM generates hypothesis, KAN validates, residuals fed back.

```mermaid
sequenceDiagram
    participant Caller
    participant RL as RefinementLoop
    participant KV as KanValidator
    participant LLM as HypothesisRefinerFn

    Caller->>RL: run(initial_hypothesis, refiner_callback)
    Note over RL: max_iterations: 5<br/>mse_threshold: 0.01<br/>improvement_threshold: 0.001

    RL->>KV: validate(initial_hypothesis)
    KV-->>RL: ValidationResult (iteration 0)

    loop iteration 1..max_iterations
        alt MSE < mse_threshold
            Note over RL: CONVERGED — stop loop
        else improvement < improvement_threshold
            Note over RL: NO IMPROVEMENT — stop loop
        else Continue refinement
            RL->>RL: build_residual_feedback(result, iteration)
            Note over RL: Feedback includes:<br/>- Current MSE value<br/>- Pattern detected<br/>- Convergence status<br/>- Suggestions for improvement

            RL->>LLM: refiner(previous_hypothesis, residual_feedback)
            LLM-->>RL: refined HypothesisProposal

            RL->>KV: validate(refined_hypothesis)
            KV-->>RL: ValidationResult (iteration N)

            Note over RL: Record RefinementIteration<br/>in provenance_chain
        end
    end

    RL-->>Caller: RefinementResult {converged, iterations, provenance_chain, final_validation}
```

---

## 9. Dynamic Concept Evolution — Phase 6

Three engines that work together: PatternDiscovery finds structure, ConceptProposer generates new concepts, EpistemicPromotion manages lifecycle.

```mermaid
sequenceDiagram
    participant SO as SystemOrchestrator
    participant PD as PatternDiscovery
    participant CP as ConceptProposer
    participant EP as EpistemicPromotion
    participant LTM as LongTermMemory
    participant CE as CuriosityEngine

    Note over SO: run_periodic_maintenance() — called periodically

    SO->>PD: discover_all()
    PD->>PD: build_graph()
    PD->>PD: find_clusters(min_size=3)
    PD->>PD: find_hierarchies(min_depth=3)
    PD->>PD: find_bridges()
    PD->>PD: find_cycles(max_length=5)
    PD->>PD: find_gaps()
    PD-->>SO: vector<DiscoveredPattern>

    Note over SO: Feed triggers to ConceptProposer

    SO->>CP: from_curiosity(triggers)
    Note over CP: Knowledge gaps → new concepts<br/>Initial type: SPECULATION or HYPOTHESIS<br/>Trust CAPPED at 0.5
    CP-->>SO: vector<ConceptProposal>

    SO->>CP: from_relevance_anomalies(relevance_map, threshold=0.8)
    Note over CP: Unexpected connections → new concepts
    CP-->>SO: vector<ConceptProposal>

    SO->>CP: rank_proposals(all_proposals, max_k=10)
    Note over CP: Deduplicate, check similarity,<br/>compute quality scores, rank
    CP-->>SO: ranked proposals

    loop Per approved proposal
        SO->>LTM: store_concept(label, def, EpistemicMetadata{SPECULATION, ACTIVE, trust≤0.5})
        LTM-->>SO: ConceptId
    end

    Note over SO: Epistemic maintenance

    SO->>EP: run_maintenance()

    EP->>EP: evaluate_all()
    Note over EP: Check each concept for promotion/demotion

    loop Per concept
        alt SPECULATION with ≥3 supporting relations
            EP->>EP: check_speculation_to_hypothesis(id, info)
            EP->>LTM: update_epistemic_metadata(id, {HYPOTHESIS, ACTIVE, new_trust})
        else HYPOTHESIS with ≥5 supports from THEORY+
            EP->>EP: check_hypothesis_to_theory(id, info)
            EP->>LTM: update_epistemic_metadata(id, {THEORY, ACTIVE, new_trust})
        else THEORY → FACT
            Note over EP: ALWAYS requires human review!
            EP-->>SO: PromotionCandidate {requires_human_review: true}
        else Has contradictions
            EP->>EP: check_demotion(id, info)
            EP->>LTM: update_epistemic_metadata(id, demoted_metadata)
        end
    end

    EP-->>SO: MaintenanceResult {promotions, demotions, deprecations, pending_review}
```

---

## 10. Curiosity → Action

CuriosityEngine detects knowledge gaps and triggers follow-up actions.

```mermaid
flowchart TD
    CE[CuriosityEngine<br/>observe_and_generate_triggers]

    CE --> D1{detect_shallow_relations?<br/>ratio < shallow_relation_ratio_}
    CE --> D2{detect_low_exploration?<br/>active < low_exploration_min_concepts_}

    D1 -->|SHALLOW_RELATIONS| T1[CuriosityTrigger<br/>type: SHALLOW_RELATIONS]
    D2 -->|LOW_EXPLORATION| T2[CuriosityTrigger<br/>type: LOW_EXPLORATION]

    T1 --> A1[MicroModel Overlay<br/>RelevanceMap.combine<br/>Creativity through<br/>perspective combination]
    T1 --> A2[Trigger ingestion<br/>WikipediaImporter /<br/>ScholarImporter<br/>→ IngestionPipeline]

    T2 --> A3[Trigger understanding<br/>UnderstandingLayer<br/>.perform_understanding_cycle]
    T2 --> A2

    A1 --> R[New insights<br/>→ ProposalQueue<br/>→ Human Review]
    A2 --> R
    A3 --> R

    style CE fill:#4a90d9,color:#fff
    style T1 fill:#e6a23c,color:#fff
    style T2 fill:#e6a23c,color:#fff
    style R fill:#67c23a,color:#fff
```

---

## 11. Component Dependency — Ownership & Access

Ownership, read-only, and write access between components.

```mermaid
flowchart TD
    subgraph Ownership["Ownership (unique_ptr)"]
        SO[SystemOrchestrator] -->|owns| BC[BrainController]
        SO -->|owns| LTM[LongTermMemory]
        SO -->|owns| CD[CognitiveDynamics]
        SO -->|owns| CE[CuriosityEngine]
        SO -->|owns| MR[MicroModelRegistry]
        SO -->|owns| EM[EmbeddingManager]
        SO -->|owns| MT[MicroTrainer]
        SO -->|owns| KA[KANAdapter]
        SO -->|owns| UL[UnderstandingLayer]
        SO -->|owns| KV[KanValidator]
        SO -->|owns| DM[DomainManager]
        SO -->|owns| RL[RefinementLoop]
        SO -->|owns| IP[IngestionPipeline]
        SO -->|owns| CI[ChatInterface]
        SO -->|owns| SORCH[StreamOrchestrator]
        SO -->|owns| PD[PatternDiscovery]
        SO -->|owns| EP[EpistemicPromotion]
        SO -->|owns| CP[ConceptProposer]
        SO -->|owns| TP[ThinkingPipeline]
        BC -->|owns| STM[ShortTermMemory]
        KM[KANModule] -->|owns| KL["vector<KANLayer>"]
        KL -->|owns| KN[KANNode]
        UL -->|owns| ML["vector<MiniLLM>"]
        IP -->|owns| PQ[ProposalQueue]
        IP -->|owns| TC[TextChunker]
        IP -->|owns| EE[EntityExtractor]
        IP -->|owns| RE[RelationExtractor]
        IP -->|owns| TT[TrustTagger]
        CI -->|owns| OC["OllamaClient"]
    end

    subgraph ReadOnly["Read-Only Access"]
        CD -.->|reads| LTM
        CD -.->|reads| STM
        UL -.->|reads| LTM
        UL -.->|reads| STM
        CE -.->|reads| STM
        CI -.->|reads| LTM
        RM[RelevanceMap] -.->|reads| LTM
        MT -.->|reads| LTM
        PD -.->|reads| LTM
        CP -.->|reads| LTM
        DM -.->|reads| LTM
    end

    subgraph WriteAccess["Write Access"]
        CD ==>|writes| STM
        IP ==>|writes| LTM
        MT ==>|writes| MR
        BC ==>|writes| STM
        EP ==>|writes| LTM
    end

    style SO fill:#2563eb,color:#fff
    style LTM fill:#16a34a,color:#fff
    style STM fill:#d97706,color:#fff
    style MR fill:#6b7280,color:#fff
```

---

## 12. Data Lifecycle — Concepts, MicroModels, STM

Lifecycles of concepts, MicroModels, and STM entries.

```mermaid
flowchart TD
    subgraph Concept["Concept Lifecycle"]
        I1[Input: Text/URL/JSON] --> ING[IngestionPipeline]
        ING --> PQ1[ProposalQueue<br/>PENDING]
        PQ1 -->|Human approve| AP[APPROVED]
        PQ1 -->|Human reject| RJ[REJECTED]
        AP --> SC[LTM.store_concept<br/>EpistemicStatus::ACTIVE]
        SC --> ACT[ACTIVE<br/>trust per TrustTagger]
        ACT -->|update_epistemic_metadata| SUP[SUPERSEDED<br/>replaced by better knowledge]
        ACT -->|invalidate_concept| INV[INVALIDATED<br/>trust = 0.05<br/>remains in LTM!]
        ACT -->|update_epistemic_metadata| CTX[CONTEXTUAL<br/>valid only in specific contexts]
    end

    subgraph MicroML["MicroModel Lifecycle"]
        MM1[MicroModel created<br/>Random W, b init] --> TR[MicroTrainer.train_single<br/>Adam optimizer]
        TR --> INF["Inference<br/>predict(e, c) → w"]
        INF -->|New relations in LTM| TR
        TR -->|converged=true| CONV[Converged<br/>final_loss < threshold]
    end

    subgraph STMLife["STM Entry Lifecycle"]
        ACT2[activate_concept<br/>CORE / CONTEXTUAL] --> LIVE[Active<br/>activation > 0]
        LIVE -->|decay_all| DEC[Decay<br/>activation -= rate × Δt]
        DEC -->|below removal_threshold| PRU[Pruned<br/>Entry removed]
        LIVE -->|boost_concept| LIVE
    end

    style I1 fill:#4a90d9,color:#fff
    style ACT fill:#67c23a,color:#fff
    style INV fill:#f56c6c,color:#fff
    style RJ fill:#909399,color:#fff
    style CONV fill:#67c23a,color:#fff
    style PRU fill:#909399,color:#fff
```

---

## 13. Multi-Stream Architecture

StreamOrchestrator manages N ThinkStreams working in parallel on shared state.

```mermaid
flowchart TD
    subgraph Orchestrator["StreamOrchestrator"]
        SO[StreamOrchestrator<br/>auto-detect hardware_concurrency]
        MON[Monitor Thread<br/>Health + Throughput<br/>stall_threshold: 5s]
    end

    SO -->|owns N| TS1[ThinkStream 0<br/>ContextId: own<br/>StreamState: Running]
    SO -->|owns N| TS2[ThinkStream 1<br/>ContextId: own<br/>StreamState: Running]
    SO -->|owns N| TSN[ThinkStream N-1<br/>ContextId: own<br/>StreamState: Running]

    MON -.->|reads metrics| TS1
    MON -.->|reads metrics| TS2
    MON -.->|reads metrics| TSN

    subgraph SharedState["Shared State (Reader-Writer Locks)"]
        SLTM[SharedLTM<br/>shared_mutex<br/>shared_lock: reads<br/>unique_lock: writes]
        SSTM[SharedSTM<br/>Per-Context shared_mutex<br/>+ global shared_mutex]
        SREG[SharedRegistry<br/>shared_mutex + per-model mutex<br/>ModelGuard RAII]
        SEMB[SharedEmbeddings<br/>shared_mutex<br/>fast-path shared_lock]
    end

    TS1 -->|shared_lock| SLTM
    TS1 -->|per-ctx lock| SSTM
    TS1 -->|shared_lock| SREG
    TS1 -->|shared_lock| SEMB
    TS2 -->|shared_lock| SLTM
    TS2 -->|per-ctx lock| SSTM
    TS2 -->|shared_lock| SREG
    TS2 -->|shared_lock| SEMB
    TSN -->|shared_lock| SLTM
    TSN -->|per-ctx lock| SSTM
    TSN -->|shared_lock| SREG
    TSN -->|shared_lock| SEMB

    subgraph Queue["Inter-Stream Communication"]
        LFQ["MPMCQueue&lt;ThinkTask&gt;<br/>Vyukov bounded MPMC<br/>ABA-safe via sequence counters<br/>alignas(64) head/tail"]
    end

    SO -->|push_task| LFQ
    LFQ -->|try_pop| TS1
    LFQ -->|try_pop| TS2
    LFQ -->|try_pop| TSN

    subgraph LockHierarchy["Lock Hierarchy (Deadlock Prevention)"]
        direction LR
        L1["1. SharedLTM"] --> L2["2. SharedSTM"] --> L3["3. SharedRegistry"] --> L4["4. SharedEmbeddings"]
    end

    style SO fill:#4a90d9,color:#fff
    style MON fill:#e6a23c,color:#fff
    style TS1 fill:#67c23a,color:#fff
    style TS2 fill:#67c23a,color:#fff
    style TSN fill:#67c23a,color:#fff
    style LFQ fill:#f56c6c,color:#fff
    style SLTM fill:#909399,color:#fff
    style SSTM fill:#909399,color:#fff
    style SREG fill:#909399,color:#fff
    style SEMB fill:#909399,color:#fff
```

---

## 14. Multi-Stream Thinking Cycle

Lifecycle of a ThinkStream: start, autonomous tick loop with subsystems, backoff, and graceful shutdown.

```mermaid
sequenceDiagram
    participant SO as StreamOrchestrator
    participant TS as ThinkStream
    participant WQ as MPMCQueue
    participant SLTM as SharedLTM
    participant SSTM as SharedSTM
    participant SREG as SharedRegistry
    participant SEMB as SharedEmbeddings

    SO->>SSTM: create_context()
    SSTM-->>SO: ContextId (per stream)
    SO->>TS: ThinkStream(id, ltm, stm, registry, embeddings, config)
    SO->>TS: start()
    Note over TS: state: Created → Starting → Running

    loop Tick loop (while !stop_requested)
        TS->>WQ: try_pop()
        alt Task available
            WQ-->>TS: ThinkTask {type: Tick, target_concept}
        else Queue empty
            Note over TS: Execute default tick
        end

        Note over TS: tick() begins

        opt has_subsystem(Spreading)
            TS->>TS: do_spreading()
            TS->>SLTM: get_outgoing_relations(source) [shared_lock]
            SLTM-->>TS: vector<RelationInfo>
            TS->>SLTM: retrieve_concept(target) [shared_lock]
            SLTM-->>TS: trust value
            TS->>SSTM: activate_concept(ctx, target, activation) [per-ctx lock]
            Note over TS: metrics.spreading_ticks++
        end

        opt has_subsystem(Salience)
            TS->>TS: do_salience()
            TS->>SSTM: get_concept_activation(ctx, cid) [shared_lock]
            TS->>SLTM: get_relation_count(cid) [shared_lock]
            Note over TS: metrics.salience_ticks++
        end

        opt has_subsystem(Curiosity)
            TS->>TS: do_curiosity()
            Note over TS: metrics.curiosity_ticks++
        end

        opt has_subsystem(Understanding)
            TS->>TS: do_understanding()
            TS->>SREG: get_model(cid) [shared_lock]
            TS->>SEMB: get_relation_embedding(type) [shared_lock]
            Note over TS: metrics.understanding_ticks++
        end

        Note over TS: metrics.total_ticks++<br/>last_tick_epoch_us = now

        alt No progress (idle)
            TS->>TS: backoff(idle_count)
            Note over TS: Tier 1: spin (100x)<br/>Tier 2: yield (10x)<br/>Tier 3: sleep (500µs)
            Note over TS: metrics.idle_ticks++
        end
    end

    SO->>TS: stop()
    Note over TS: stop_requested_ = true<br/>state: Running → Stopping

    Note over TS: Current tick completes

    SO->>TS: join(shutdown_timeout: 5s)
    Note over TS: state: Stopping → Stopped

    SO->>SSTM: destroy_context(ctx)
```

---

## 15. System Initialization Sequence

SystemOrchestrator::initialize() brings up all 14 subsystem groups in dependency order.

```mermaid
sequenceDiagram
    participant Main as main()
    participant APP as Brain19App
    participant SO as SystemOrchestrator
    participant LTM as LongTermMemory
    participant PLTM as PersistentLTM
    participant WAL as WALWriter
    participant BC as BrainController
    participant MR as MicroModelRegistry
    participant EM as EmbeddingManager
    participant MT as MicroTrainer
    participant CD as CognitiveDynamics
    participant CE as CuriosityEngine
    participant KA as KANAdapter
    participant UL as UnderstandingLayer
    participant KV as KanValidator
    participant DM as DomainManager
    participant RL as RefinementLoop
    participant IP as IngestionPipeline
    participant CI as ChatInterface
    participant SORCH as StreamOrchestrator
    participant BI as BootstrapInterface

    Main->>APP: Brain19App(config)
    APP->>SO: SystemOrchestrator(config)
    Main->>APP: run_interactive()
    APP->>SO: initialize()

    Note over SO: Stage 1: LTM
    SO->>LTM: new LongTermMemory()
    SO->>PLTM: new PersistentLTM(data_dir)
    SO->>PLTM: load_into(ltm)
    Note over SO: init_stage_ = 1

    Note over SO: Stage 2: WAL
    SO->>WAL: new WALWriter(data_dir)
    Note over SO: init_stage_ = 2

    Note over SO: Stage 3: BrainController + STM
    SO->>BC: new BrainController()
    SO->>BC: initialize()
    Note over SO: init_stage_ = 3

    Note over SO: Stage 4: MicroModels
    SO->>EM: new EmbeddingManager()
    SO->>MR: new MicroModelRegistry()
    SO->>MR: ensure_models_for(ltm)
    SO->>MT: new MicroTrainer()
    Note over SO: init_stage_ = 4

    Note over SO: Stage 5: Cognitive
    SO->>CD: new CognitiveDynamics(config)
    Note over SO: init_stage_ = 5

    Note over SO: Stage 6: Curiosity
    SO->>CE: new CuriosityEngine()
    Note over SO: init_stage_ = 6

    Note over SO: Stage 7: KAN
    SO->>KA: new KANAdapter()
    Note over SO: init_stage_ = 7

    Note over SO: Stage 8: Understanding
    SO->>UL: new UnderstandingLayer(config)
    Note over SO: init_stage_ = 8

    Note over SO: Stage 9: Hybrid
    SO->>KV: new KanValidator(config)
    SO->>DM: new DomainManager(config)
    SO->>RL: new RefinementLoop(validator)
    Note over SO: init_stage_ = 9

    Note over SO: Stage 10: Ingestion
    SO->>IP: new IngestionPipeline(ltm)
    Note over SO: init_stage_ = 10

    Note over SO: Stage 11: Chat + LLM
    SO->>CI: new ChatInterface()
    SO->>CI: initialize(ollama_config)
    Note over SO: init_stage_ = 11

    Note over SO: Stage 12: Shared Wrappers
    Note over SO: SharedLTM, SharedSTM, SharedRegistry, SharedEmbeddings

    Note over SO: Stage 13: Streams
    SO->>SORCH: new StreamOrchestrator(shared_ltm, shared_stm, ...)
    SO->>SORCH: auto_scale()
    SO->>SORCH: start_all()
    Note over SO: init_stage_ = 13

    Note over SO: Stage 14: Evolution
    Note over SO: PatternDiscovery, EpistemicPromotion, ConceptProposer

    opt config.seed_foundation
        SO->>BI: seed_foundation()
        Note over BI: Foundation concepts:<br/>logic, mathematics, causality, etc.
    end

    SO->>SO: periodic_thread_ = thread(periodic_task_loop)
    Note over SO: running_ = true

    SO-->>APP: true (success)
    Note over APP: Enter interactive REPL
```

---

## 16. Checkpoint & Restore Flow

Full state persistence: checkpoint saves everything, restore rebuilds from checkpoint.

```mermaid
sequenceDiagram
    participant User
    participant SO as SystemOrchestrator
    participant CM as CheckpointManager
    participant PLTM as PersistentLTM
    participant SNAP as STMSnapshot
    participant WAL as WALWriter
    participant LTM as LongTermMemory
    participant STM as ShortTermMemory
    participant MR as MicroModelRegistry
    participant CR as CheckpointRestore

    Note over User: === CHECKPOINT ===
    User->>SO: create_checkpoint(tag)
    SO->>SO: lock subsystem_mtx_

    SO->>CM: create_checkpoint(tag)
    CM->>PLTM: save(ltm)
    Note over PLTM: Binary format: concepts + relations
    CM->>SNAP: save(stm)
    Note over SNAP: Binary format: all contexts + entries
    CM->>CM: save_micromodels(registry)
    Note over CM: 430 doubles per model (flat array)
    CM->>WAL: flush()
    CM->>CM: rotate_checkpoints(max=5)
    CM-->>SO: checkpoint_dir path

    SO->>SO: unlock subsystem_mtx_

    Note over User: === RESTORE ===
    User->>SO: restore_checkpoint(checkpoint_dir)
    SO->>SO: shutdown existing subsystems

    SO->>CR: restore(checkpoint_dir)
    CR->>PLTM: load_into(ltm)
    Note over PLTM: Rebuild concept + relation maps
    CR->>SNAP: load_into(stm)
    Note over SNAP: Rebuild context + entry maps
    CR->>CR: load_micromodels(registry)
    CR->>WAL: replay_after(checkpoint_timestamp)
    Note over WAL: Apply any WAL entries after checkpoint

    CR-->>SO: restore success
    SO->>SO: re-initialize remaining subsystems
    SO-->>User: restore complete
```

---

## 17. Full-Stack Deployment

Three-tier architecture: C++ backend binary, Python API bridge, React frontend.

```mermaid
flowchart TD
    subgraph Frontend["Frontend (:3019)"]
        VIZ[Brain19Visualizer.jsx<br/>React + Vite]
        STMGraph[STMGraph Component<br/>Force-directed SVG graph]
        EPPanel[EpistemicPanel<br/>Trust + Type display]
        CPanel[CuriosityPanel<br/>Trigger display]
    end

    subgraph API["Python API Bridge (:8019)"]
        FA[FastAPI Server<br/>server.py]
        REST["REST Endpoints<br/>GET /api/status<br/>GET /api/snapshot<br/>GET /api/concepts<br/>POST /api/ask<br/>POST /api/ingest"]
        WS["WebSocket /ws<br/>Periodic broadcast<br/>Real-time commands"]
    end

    subgraph Backend["C++ Backend Binary"]
        BIN[brain19 executable<br/>main.cpp → Brain19App]
        SO[SystemOrchestrator<br/>14 subsystem groups]
        DATA[(brain19_data/<br/>LTM binary + WAL<br/>+ Checkpoints)]
    end

    subgraph External["External Services"]
        OLLAMA[Ollama<br/>:11434<br/>llama3.2:1b]
        WIKI[Wikipedia API<br/>Article import]
        SCHOLAR[Google Scholar<br/>Paper search]
    end

    VIZ --> STMGraph
    VIZ --> EPPanel
    VIZ --> CPanel
    VIZ -->|HTTP + WebSocket| FA
    FA --> REST
    FA --> WS
    REST -->|subprocess exec| BIN
    WS -->|subprocess exec| BIN
    BIN --> SO
    SO --> DATA
    SO -->|HTTP| OLLAMA
    SO -->|HTTP| WIKI
    SO -->|HTTP| SCHOLAR

    style VIZ fill:#61dafb,color:#000
    style FA fill:#009688,color:#fff
    style BIN fill:#2563eb,color:#fff
    style OLLAMA fill:#7c3aed,color:#fff
```

---

*Generated from actual code in `backend/`, `api/`, and `frontend/`. Updated: 2026-02-12.*
