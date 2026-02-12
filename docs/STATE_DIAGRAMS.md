# Brain19 — State Diagrams

> UML state machine diagrams for all stateful components.
> All states, transitions, and guards match the actual code in `backend/`.
> Updated: 2026-02-12

---

## Table of Contents

1. [ThinkStream State Machine](#1-thinkstream-state-machine)
2. [EpistemicStatus Lifecycle](#2-epistemicstatus-lifecycle)
3. [EpistemicType Promotion Ladder](#3-epistemictype-promotion-ladder)
4. [System Initialization Stages](#4-system-initialization-stages)
5. [Concept Lifecycle](#5-concept-lifecycle)
6. [Proposal Lifecycle](#6-proposal-lifecycle)
7. [MicroModel Training State](#7-micromodel-training-state)
8. [Refinement Loop State](#8-refinement-loop-state)
9. [STM Entry Activation State](#9-stm-entry-activation-state)

---

## 1. ThinkStream State Machine

Each ThinkStream has a `StreamState` enum that follows this state machine. Transitions are atomic.

```mermaid
stateDiagram-v2
    [*] --> Created : ThinkStream constructor

    Created --> Starting : start() called
    Starting --> Running : thread launched successfully
    Starting --> Error : thread launch failed

    Running --> Stopping : stop() called\nstop_requested_ = true
    Running --> Error : unhandled exception in tick()

    Stopping --> Stopped : join() succeeds\nthread exits cleanly
    Stopping --> Error : join() timeout

    Paused --> Running : resume (not yet implemented)
    Running --> Paused : pause (not yet implemented)

    Stopped --> [*]
    Error --> [*]

    note right of Running
        Autonomous tick loop:
        1. try_pop(work_queue)
        2. do_spreading()
        3. do_salience()
        4. do_curiosity()
        5. do_understanding()
        6. backoff if idle
    end note

    note right of Stopping
        Current tick completes
        before thread exits
        (graceful shutdown)
    end note
```

---

## 2. EpistemicStatus Lifecycle

Every ConceptInfo in LTM has an EpistemicStatus. Knowledge is NEVER deleted — only transitions between statuses.

```mermaid
stateDiagram-v2
    [*] --> ACTIVE : store_concept()

    ACTIVE --> CONTEXTUAL : update_epistemic_metadata()\n"valid only in specific contexts"
    ACTIVE --> SUPERSEDED : update_epistemic_metadata()\n"replaced by better knowledge"
    ACTIVE --> INVALIDATED : invalidate_concept()\ntrust = 0.05

    CONTEXTUAL --> ACTIVE : update_epistemic_metadata()\n"context broadened"
    CONTEXTUAL --> SUPERSEDED : update_epistemic_metadata()
    CONTEXTUAL --> INVALIDATED : invalidate_concept()

    SUPERSEDED --> INVALIDATED : invalidate_concept()

    note right of ACTIVE
        Default state for new concepts
        Full participation in:
        - Spreading activation
        - Salience computation
        - Relevance maps
    end note

    note right of INVALIDATED
        NEVER deleted!
        Remains in LTM with trust < 0.2
        Preserves epistemic history
        Still queryable but low-weighted
    end note

    note right of SUPERSEDED
        Replaced by better knowledge
        but not wrong — just outdated
    end note
```

---

## 3. EpistemicType Promotion Ladder

The epistemic promotion system manages knowledge certainty levels. Each transition has specific requirements.

```mermaid
stateDiagram-v2
    direction TB

    [*] --> SPECULATION : System-generated\n(trust 0.1-0.3)

    SPECULATION --> HYPOTHESIS : EpistemicPromotion\n≥3 supporting relations\ntrust → 0.3-0.5

    HYPOTHESIS --> THEORY : EpistemicPromotion\n≥5 supports from THEORY+\nindependent evidence\ntrust → 0.5-0.8

    THEORY --> FACT : confirm_as_fact()\nHUMAN REVIEW REQUIRED\ntrust → 0.8-1.0

    note right of SPECULATION
        System-generated concepts
        ConceptProposer output
        Trust CAPPED at 0.5
    end note

    note right of HYPOTHESIS
        Testable claim with
        some supporting evidence
        KAN validation candidate
    end note

    note right of THEORY
        Well-supported, falsifiable
        Multiple independent sources
        Automatic promotion possible
    end note

    note right of FACT
        NEVER automatic!
        Always requires human confirmation
        Highest certainty level
    end note

    state demotion <<choice>>
    THEORY --> demotion : has_contradictions()
    HYPOTHESIS --> demotion : has_contradictions()
    demotion --> SPECULATION : demote()

    note left of demotion
        Demotion CAN be automatic
        on contradictions
        (unlike FACT promotion)
    end note
```

**Additional epistemic types** (entered via direct construction, not through promotion):

```mermaid
stateDiagram-v2
    [*] --> DEFINITION : Definitional/tautological\ntrust typically 1.0

    [*] --> INFERENCE : Derived from other knowledge\ntrust depends on source

    [*] --> FACT : Human-provided fact\ntrust 0.8-1.0

    note right of DEFINITION
        Tautological knowledge
        e.g., "A triangle has 3 sides"
        Does not participate in promotion
    end note

    note right of INFERENCE
        Derived from other concepts
        Trust depends on source trust
    end note
```

---

## 4. System Initialization Stages

SystemOrchestrator tracks initialization progress via `init_stage_`. On failure, `cleanup_from_stage()` tears down in reverse order.

```mermaid
stateDiagram-v2
    [*] --> Stage0_Uninit : SystemOrchestrator()

    Stage0_Uninit --> Stage1_LTM : new LongTermMemory\nPersistentLTM.load_into()

    Stage1_LTM --> Stage2_WAL : new WALWriter(data_dir)

    Stage2_WAL --> Stage3_Brain : new BrainController\ninitialize()

    Stage3_Brain --> Stage4_MicroModels : new EmbeddingManager\nnew MicroModelRegistry\nnew MicroTrainer

    Stage4_MicroModels --> Stage5_Cognitive : new CognitiveDynamics(config)

    Stage5_Cognitive --> Stage6_Curiosity : new CuriosityEngine()

    Stage6_Curiosity --> Stage7_KAN : new KANAdapter()

    Stage7_KAN --> Stage8_Understanding : new UnderstandingLayer(config)

    Stage8_Understanding --> Stage9_Hybrid : new KanValidator\nnew DomainManager\nnew RefinementLoop

    Stage9_Hybrid --> Stage10_Ingestion : new IngestionPipeline(ltm)

    Stage10_Ingestion --> Stage11_Chat : new ChatInterface\ninitialize()

    Stage11_Chat --> Stage12_Shared : SharedLTM, SharedSTM\nSharedRegistry, SharedEmbeddings

    Stage12_Shared --> Stage13_Streams : new StreamOrchestrator\nauto_scale(), start_all()

    Stage13_Streams --> Stage14_Evolution : PatternDiscovery\nEpistemicPromotion\nConceptProposer

    Stage14_Evolution --> Running : seed_foundation()\nperiodic_thread_ started\nrunning_ = true

    Running --> Shutdown : shutdown() called

    state Shutdown {
        [*] --> StopPeriodic : periodic_running_ = false
        StopPeriodic --> StopStreams : stream_orch_->shutdown()
        StopStreams --> CleanupReverse : cleanup_from_stage(init_stage_)
        CleanupReverse --> [*] : running_ = false
    }

    note right of Stage0_Uninit
        init_stage_ = 0
        On failure at any stage,
        cleanup_from_stage(stage)
        tears down in reverse
    end note
```

---

## 5. Concept Lifecycle

A concept's complete lifecycle from creation through ingestion to potential invalidation.

```mermaid
stateDiagram-v2
    [*] --> RawInput : Text/URL/JSON/CSV

    RawInput --> TextChunk : TextChunker.chunk_text()
    TextChunk --> ExtractedEntity : EntityExtractor
    ExtractedEntity --> IngestProposal : RelationExtractor\nTrustTagger

    IngestProposal --> PENDING : ProposalQueue.enqueue()

    state PENDING {
        [*] --> AwaitReview
        AwaitReview --> Approved : Human approve
        AwaitReview --> Rejected : Human reject
        AwaitReview --> AutoApproved : auto_approve=true
    }

    Approved --> StoredConcept : LTM.store_concept()\n+ EpistemicMetadata
    AutoApproved --> StoredConcept

    state StoredConcept {
        [*] --> Active
        Active --> MicroModelCreated : MicroModelRegistry.create_model()
        MicroModelCreated --> MicroModelTrained : MicroTrainer.train_single()
        MicroModelTrained --> FullyIntegrated : Available for\nspreading activation\nrelevance maps
    }

    FullyIntegrated --> Promoted : EpistemicPromotion\n(trust increases)
    FullyIntegrated --> Demoted : Contradictions found\n(trust decreases)
    FullyIntegrated --> Superseded : Better knowledge arrives
    FullyIntegrated --> Invalidated : invalidate_concept()\ntrust = 0.05

    Rejected --> [*] : Discarded

    note right of Invalidated
        NEVER deleted
        Remains in LTM
        Low trust weight
    end note
```

---

## 6. Proposal Lifecycle

IngestProposal status transitions in the ProposalQueue.

```mermaid
stateDiagram-v2
    [*] --> PENDING : ProposalQueue.enqueue()

    PENDING --> APPROVED : review(id, approve)
    PENDING --> REJECTED : review(id, reject)
    PENDING --> MODIFIED : review(id, modify)

    MODIFIED --> APPROVED : review(id, approve)
    MODIFIED --> REJECTED : review(id, reject)

    APPROVED --> COMMITTED : commit_approved()\n→ LTM.store_concept()

    REJECTED --> [*] : Removed from queue
    COMMITTED --> [*] : Now in LTM
```

---

## 7. MicroModel Training State

Training lifecycle for a single MicroModel via Adam optimizer.

```mermaid
stateDiagram-v2
    [*] --> Initialized : MicroModel()\nRandom W, b init

    Initialized --> GeneratingSamples : MicroTrainer begins\ngenerate_samples(cid, embeddings, ltm)

    GeneratingSamples --> Training : samples ready\n(positives from relations,\n3x negatives per positive)

    state Training {
        [*] --> Epoch
        Epoch --> ComputeLoss : Forward pass all samples
        ComputeLoss --> ComputeGradients : MSE loss computed
        ComputeGradients --> AdamUpdate : dW, db gradients
        AdamUpdate --> CheckConvergence : W -= lr * m/(sqrt(v)+ε)

        CheckConvergence --> Epoch : loss > threshold\n&& epoch < max_epochs
        CheckConvergence --> Converged : loss < threshold
        CheckConvergence --> MaxEpochs : epoch >= max_epochs
    }

    Converged --> Ready : MicroTrainingResult\n{converged: true}
    MaxEpochs --> Ready : MicroTrainingResult\n{converged: false}

    Ready --> Inference : predict(e, c) → w ∈ (0,1)
    Inference --> Training : New relations added\n(retrain needed)
```

---

## 8. Refinement Loop State

The LLM↔KAN bidirectional refinement process.

```mermaid
stateDiagram-v2
    [*] --> InitialValidation : run(initial_hypothesis, refiner)

    InitialValidation --> CheckConvergence : KanValidator.validate()

    state CheckConvergence <<choice>>
    CheckConvergence --> Converged : MSE < mse_threshold (0.01)
    CheckConvergence --> CheckImprovement : MSE ≥ mse_threshold

    state CheckImprovement <<choice>>
    CheckImprovement --> NoImprovement : improvement < 0.001
    CheckImprovement --> BuildFeedback : improvement ≥ 0.001\n&& iteration < max (5)

    BuildFeedback --> RefineLLM : build_residual_feedback()\n→ refiner callback

    RefineLLM --> Revalidation : refined HypothesisProposal
    Revalidation --> CheckConvergence : KanValidator.validate()

    Converged --> Done : RefinementResult\n{converged: true}
    NoImprovement --> Done : RefinementResult\n{converged: false}

    state Done {
        [*] --> ReturnResult
        note right of ReturnResult
            Includes full provenance_chain:
            all RefinementIterations with
            hypothesis, validation, feedback
        end note
    }
```

---

## 9. STM Entry Activation State

Activation levels in Short-Term Memory follow a decay model with explicit boosting.

```mermaid
stateDiagram-v2
    [*] --> Inactive : Concept not in STM context

    Inactive --> Active : activate_concept(ctx, cid, activation, class)

    state Active {
        [*] --> HighActivation : activation > 0.7
        HighActivation --> MediumActivation : decay_all()\nactivation -= rate × Δt
        MediumActivation --> LowActivation : decay_all()
        LowActivation --> BelowThreshold : activation < removal_threshold

        HighActivation --> HighActivation : boost_concept()\nactivation += delta
        MediumActivation --> HighActivation : boost_concept()
        LowActivation --> MediumActivation : boost_concept()
    }

    BelowThreshold --> Pruned : Entry removed from context

    Pruned --> [*]

    note right of Active
        ActivationClass:
        - CORE: core_decay_rate_
        - CONTEXTUAL: contextual_decay_rate_

        All activations clamped to [0.0, 1.0]
    end note

    note right of HighActivation
        Participates in:
        - Salience computation
        - Focus set candidacy
        - Spreading activation source
    end note
```

---

## Appendix: Domain Type Classification

How DomainManager classifies concepts into knowledge domains based on their relations.

```mermaid
stateDiagram-v2
    [*] --> AnalyzeRelations : detect_domain(cid, ltm)

    AnalyzeRelations --> PHYSICAL : CAUSES, MEASURES relations\ndominate
    AnalyzeRelations --> BIOLOGICAL : PART_OF, PRODUCES relations\ndominate
    AnalyzeRelations --> SOCIAL : INFLUENCES, ASSOCIATED_WITH\ndominate
    AnalyzeRelations --> ABSTRACT : IS_A, IMPLIES relations\ndominate
    AnalyzeRelations --> TEMPORAL : PRECEDES, FOLLOWS relations\ndominate

    note right of PHYSICAL
        num_knots: 15
        hidden_dim: 8
        Higher precision for physics
    end note

    note right of ABSTRACT
        num_knots: 12
        hidden_dim: 6
        Mathematics, logic
    end note
```

---

*Generated from actual code in `backend/`. Updated: 2026-02-12.*
