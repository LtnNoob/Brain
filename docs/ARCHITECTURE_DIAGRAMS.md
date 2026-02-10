# Brain19 — Architecture Diagrams

> Dynamische Flows der C++20 Cognitive Architecture.  
> Alle Methoden- und Klassennamen entsprechen dem tatsächlichen Code in `backend/`.

---

## 1. THINKING CYCLE — Kompletter Tick

Ein einzelner Denk-Tick: BrainController orchestriert, CognitiveDynamics rechnet, CuriosityEngine beobachtet.

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
    loop Depth ≤ max_depth (3), pro Relation
        CD->>LTM: retrieve_concept(target) → EpistemicMetadata.trust
        Note over CD: activation(T) += activation(S) × weight × trust × damping^depth
        CD->>STM: activate_concept(ctx, target, computed_activation, CONTEXTUAL)
        CD->>CD: spread_recursive(target, ..., depth+1, visited)
    end
    CD-->>Caller: SpreadingStats

    Caller->>CD: compute_salience_batch(concepts, ctx, ltm, stm, tick)
    Note over CD: score = activation × 0.4(freq) + recency + connectivity × 0.2 + trust × 0.3
    CD->>STM: get_concept_activation(ctx, cid)
    CD->>LTM: get_relation_count(cid)
    CD->>LTM: retrieve_concept(cid) → epistemic.trust
    CD-->>Caller: vector<SalienceScore>

    Caller->>CD: get_top_k_salient(candidates, k, ctx, ltm, stm)
    Note over CD: Top-K ins Arbeitsgedächtnis (Focus Set)
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

## 2. LEARNING FLOW — Neues Wissen integrieren

Von externen Quellen über die IngestionPipeline bis in LTM — mit Human Review.

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
    Note over WI: Nur Vorschläge, KEIN LTM-Schreibzugriff

    User->>SI: search_papers(query, limit)
    SI-->>User: vector<unique_ptr<KnowledgeProposal>>

    User->>IP: ingest_text(text, source_ref, auto_approve=false)
    IP->>TC: chunk_text(text)
    TC-->>IP: vector<TextChunk>

    IP->>EE: extract_from_chunks(chunks)
    Note over EE: Kapitalisierung, Quotes, Definition-Patterns, Frequenz
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

    loop Pro genehmigtem Proposal
        IP->>LTM: store_concept(label, definition, EpistemicMetadata)
        Note over LTM: ConceptInfo() = delete! EpistemicMetadata PFLICHT
        LTM-->>IP: ConceptId
        IP->>LTM: add_relation(source, target, type, weight)
    end

    User->>MT: train_all(registry, embeddings, ltm)
    MT->>MT: generate_samples(cid, embeddings, ltm)
    Note over MT: Positives aus Relations, 3x Negatives pro Positive
    MT->>MR: get_model(cid) → MicroModel
    MT->>MR: model.train(samples, config)
    Note over MT: Adam-Optimizer, MSE-Loss, Convergence-Check
    MT-->>User: TrainerStats
```

---

## 3. QUERY/CHAT FLOW — User stellt Frage

Von der User-Frage über Keyword-Suche, Spreading Activation und MicroModel-Relevanz bis zur LLM-Antwort.

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
    Note over CI: Keyword-Matching gegen label/definition
    CI-->>CI: vector<ConceptInfo> relevant

    CI->>CD: spread_activation_multi(sources, activation, ctx, ltm, stm)
    Note over CD: Spreading entlang Relations, trust-gewichtet
    CD->>STM: activate_concept(ctx, target, activation, CONTEXTUAL)
    CD-->>CI: SpreadingStats

    loop Pro relevantem Konzept
        CI->>RM: compute(cid, registry, embeddings, ltm, rel_type, context)
        RM->>MR: get_model(cid)
        RM->>EM: get_embedding(cid)
        Note over RM: v = W*c + b, z = e^T*v, w = sigma(z)
        RM-->>CI: RelevanceMap (scores pro Konzept)
    end

    CI->>RM: combine(maps, mode, weights)
    Note over RM: ADDITION / MAX / WEIGHTED_AVERAGE

    CI->>CI: build_epistemic_context(concepts)
    Note over CI: Trust-Levels in Prompt einbauen:<br/>FACT (trust 0.98), THEORY (0.9), HYPOTHESIS (0.5)

    CI->>CI: build_system_prompt()
    Note over CI: Verbalisiere NUR vorhandenes Wissen.<br/>Markiere Unsicherheiten.

    CI->>OC: generate(prompt, context)
    OC-->>CI: OllamaResponse

    CI-->>User: ChatResponse {answer, referenced_concepts, epistemic_note}
```

---

## 4. UNDERSTANDING CYCLE — Semantische Analyse

UnderstandingLayer nutzt CognitiveDynamics für Fokus und Mini-LLMs für semantische Vorschläge.

```mermaid
sequenceDiagram
    participant Caller
    participant UL as UnderstandingLayer
    participant CD as CognitiveDynamics
    participant LTM as LongTermMemory
    participant STM as ShortTermMemory
    participant ML as MiniLLM (registriert)

    Caller->>UL: perform_understanding_cycle(seed_concept, cd, ltm, stm, ctx)

    UL->>CD: spread_activation(seed_concept, activation, ctx, ltm, stm)
    CD-->>UL: SpreadingStats

    UL->>CD: get_top_k_salient(candidates, k, ctx, ltm, stm)
    CD-->>UL: vector<SalienceScore> (fokussierte Konzepte)

    UL->>CD: focus_on(ctx, cid, boost)

    loop Für jede registrierte MiniLLM
        UL->>ML: extract_meaning(active_concepts, ltm, stm, ctx)
        ML-->>UL: vector<MeaningProposal>
        Note over ML: Alle Proposals: EpistemicType::HYPOTHESIS

        UL->>ML: generate_hypotheses(evidence_concepts, ltm, stm, ctx)
        ML-->>UL: vector<HypothesisProposal>

        UL->>ML: detect_analogies(set_a, set_b, ltm, stm, ctx)
        ML-->>UL: vector<AnalogyProposal>

        UL->>ML: detect_contradictions(active_concepts, ltm, stm, ctx)
        ML-->>UL: vector<ContradictionProposal>
    end

    Note over UL: Filter: min_meaning_confidence 0.3,<br/>min_hypothesis_confidence 0.2,<br/>min_analogy_confidence 0.4,<br/>min_contradiction_severity 0.5

    Note over UL: Trust-Ceiling: model_confidence max 0.3-0.5<br/>Alle Proposals bleiben HYPOTHESIS

    UL-->>Caller: UnderstandingResult {meaning, hypothesis, analogy, contradiction}

    Note over Caller: BrainController → Epistemic Core<br/>entscheidet über Akzeptanz/Ablehnung
```

---

## 5. CURIOSITY → ACTION

CuriosityEngine erkennt Wissenslücken und triggert Folgeaktionen.

```mermaid
flowchart TD
    CE[CuriosityEngine<br/>observe_and_generate_triggers]

    CE --> D1{detect_shallow_relations?<br/>ratio < shallow_relation_ratio_}
    CE --> D2{detect_low_exploration?<br/>active < low_exploration_min_concepts_}

    D1 -->|SHALLOW_RELATIONS| T1[CuriosityTrigger<br/>type: SHALLOW_RELATIONS]
    D2 -->|LOW_EXPLORATION| T2[CuriosityTrigger<br/>type: LOW_EXPLORATION]

    T1 --> A1[MicroModel-Overlay<br/>RelevanceMap.combine<br/>Kreativität durch<br/>Perspektiven-Kombination]
    T1 --> A2[Ingestion triggern<br/>WikipediaImporter /<br/>ScholarImporter<br/>→ IngestionPipeline]

    T2 --> A3[Understanding triggern<br/>UnderstandingLayer<br/>.perform_understanding_cycle]
    T2 --> A2

    A1 --> R[Neue Erkenntnisse<br/>→ ProposalQueue<br/>→ Human Review]
    A2 --> R
    A3 --> R

    style CE fill:#4a90d9,color:#fff
    style T1 fill:#e6a23c,color:#fff
    style T2 fill:#e6a23c,color:#fff
    style R fill:#67c23a,color:#fff
```

---

## 6. COMPONENT DEPENDENCY

Ownership, Read-Only und Write-Zugriff zwischen Komponenten.

```mermaid
flowchart TD
    subgraph Ownership["Ownership (unique_ptr)"]
        BC[BrainController] -->|owns| STM[ShortTermMemory]
        KM[KANModule] -->|owns| KL["vector unique_ptr KANLayer"]
        KL -->|owns| KN[KANNode]
        UL[UnderstandingLayer] -->|owns| ML["vector unique_ptr MiniLLM"]
        IP[IngestionPipeline] -->|owns| PQ[ProposalQueue]
        IP -->|owns| TC[TextChunker]
        IP -->|owns| EE[EntityExtractor]
        IP -->|owns| RE[RelationExtractor]
        IP -->|owns| TT[TrustTagger]
        CI[ChatInterface] -->|owns| OC["unique_ptr OllamaClient"]
    end

    subgraph ReadOnly["Read-Only Zugriff"]
        CD[CognitiveDynamics] -.->|reads| LTM[LongTermMemory]
        CD -.->|reads| STM
        UL -.->|reads| LTM
        UL -.->|reads| STM
        CE[CuriosityEngine] -.->|reads| STM
        CI -.->|reads| LTM
        RM[RelevanceMap] -.->|reads| LTM
        MT[MicroTrainer] -.->|reads| LTM
    end

    subgraph WriteAccess["Write-Zugriff"]
        CD ==>|writes| STM
        IP ==>|writes| LTM
        MT ==>|writes| MR[MicroModelRegistry]
        BC ==>|writes| STM
    end

    style BC fill:#4a90d9,color:#fff
    style LTM fill:#67c23a,color:#fff
    style STM fill:#e6a23c,color:#fff
    style MR fill:#909399,color:#fff
```

---

## 7. DATA LIFECYCLE

Lebenszyklen von Konzepten, MicroModels und STM-Einträgen.

```mermaid
flowchart TD
    subgraph Concept["Konzept-Lebenszyklus"]
        I1[Input: Text/URL/JSON] --> ING[IngestionPipeline]
        ING --> PQ1[ProposalQueue<br/>PENDING]
        PQ1 -->|Human approve| AP[APPROVED]
        PQ1 -->|Human reject| RJ[REJECTED]
        AP --> SC[LTM.store_concept<br/>EpistemicStatus::ACTIVE]
        SC --> ACT[ACTIVE<br/>trust gemäß TrustTagger]
        ACT -->|update_epistemic_metadata| SUP[SUPERSEDED<br/>ersetzt durch besseres Wissen]
        ACT -->|invalidate_concept| INV[INVALIDATED<br/>trust = 0.05<br/>bleibt in LTM!]
        ACT -->|update_epistemic_metadata| CTX[CONTEXTUAL<br/>nur in bestimmten Kontexten]
    end

    subgraph MicroML["MicroModel-Lebenszyklus"]
        MM1[MicroModel erstellt<br/>Random W, b Init] --> TR[MicroTrainer.train_single<br/>Adam-Optimizer]
        TR --> INF["Inference<br/>predict(e, c) → w"]
        INF -->|Neue Relations in LTM| TR
        TR -->|converged=true| CONV[Konvergiert<br/>final_loss < threshold]
    end

    subgraph STMLife["STM-Entry-Lebenszyklus"]
        ACT2[activate_concept<br/>CORE / CONTEXTUAL] --> LIVE[Aktiv<br/>activation > 0]
        LIVE -->|decay_all| DEC[Decay<br/>activation -= rate * delta_t]
        DEC -->|below removal_threshold| PRU[Pruned<br/>Entry entfernt]
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

---

## 8. MULTI-STREAM ARCHITECTURE

StreamOrchestrator verwaltet N ThinkStreams die parallel auf geteiltem State arbeiten.

```mermaid
flowchart TD
    subgraph Orchestrator["StreamOrchestrator"]
        SO[StreamOrchestrator<br/>auto-detect hardware_concurrency]
        MON[Monitor-Thread<br/>Health + Throughput<br/>stall_threshold: 5s]
    end

    SO -->|owns N| TS1[ThinkStream 0<br/>ContextId: eigener<br/>StreamState: Running]
    SO -->|owns N| TS2[ThinkStream 1<br/>ContextId: eigener<br/>StreamState: Running]
    SO -->|owns N| TSN[ThinkStream N-1<br/>ContextId: eigener<br/>StreamState: Running]

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

    subgraph Queue["Inter-Stream Kommunikation"]
        LFQ["MPMCQueue&lt;ThinkTask&gt;<br/>Vyukov bounded MPMC<br/>ABA-safe via sequence counters<br/>alignas(64) head/tail"]
    end

    SO -->|push_task| LFQ
    LFQ -->|try_pop| TS1
    LFQ -->|try_pop| TS2
    LFQ -->|try_pop| TSN

    subgraph LockHierarchy["Lock-Hierarchie (Deadlock-Prevention)"]
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

## 9. MULTI-STREAM THINKING CYCLE

Lebenszyklus eines ThinkStream: Start, autonomer Tick-Loop mit Subsystemen, Backoff und Graceful Shutdown.

```mermaid
sequenceDiagram
    participant SO as StreamOrchestrator
    participant TS as ThinkStream
    participant WQ as MPMCQueue<ThinkTask>
    participant SLTM as SharedLTM
    participant SSTM as SharedSTM
    participant SREG as SharedRegistry
    participant SEMB as SharedEmbeddings

    SO->>SSTM: create_context()
    SSTM-->>SO: ContextId (per Stream)
    SO->>TS: ThinkStream(id, ltm, stm, registry, embeddings, config)
    SO->>TS: start()
    Note over TS: state: Created → Starting → Running

    loop Tick-Loop (solange !stop_requested)
        TS->>WQ: try_pop()
        alt Task vorhanden
            WQ-->>TS: ThinkTask {type: Tick, target_concept}
        else Queue leer
            Note over TS: Default-Tick ausführen
        end

        Note over TS: tick() beginnt

        opt has_subsystem(Spreading)
            TS->>TS: do_spreading()
            TS->>SLTM: get_outgoing_relations(source) [shared_lock]
            SLTM-->>TS: vector<RelationInfo>
            TS->>SLTM: retrieve_concept(target) [shared_lock]
            SLTM-->>TS: trust-Wert
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

        alt Kein Fortschritt (idle)
            TS->>TS: backoff(idle_count)
            Note over TS: Tier 1: spin (100x)<br/>Tier 2: yield (10x)<br/>Tier 3: sleep (500µs)
            Note over TS: metrics.idle_ticks++
        end
    end

    SO->>TS: stop()
    Note over TS: stop_requested_ = true<br/>state: Running → Stopping

    Note over TS: Aktueller Tick wird beendet

    SO->>TS: join(shutdown_timeout: 5s)
    Note over TS: state: Stopping → Stopped

    SO->>SSTM: destroy_context(ctx)
```

---

*Generiert aus dem tatsächlichen Code in `backend/`. Stand: 2026-02-10.*
