# Brain19 — Sequence Diagrams (Extended)

> Detailed sequence diagrams for secondary and cross-cutting workflows.
> Complements the primary diagrams in [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md).
> Updated: 2026-02-12

---

## Table of Contents

1. [API Request Lifecycle (REST)](#1-api-request-lifecycle-rest)
2. [WebSocket Real-Time Flow](#2-websocket-real-time-flow)
3. [Cross-Domain Insight Discovery](#3-cross-domain-insight-discovery)
4. [Bootstrap — Foundation Seeding](#4-bootstrap--foundation-seeding)
5. [WAL Recovery on Startup](#5-wal-recovery-on-startup)
6. [Periodic Maintenance Cycle](#6-periodic-maintenance-cycle)
7. [Wikipedia Import Pipeline](#7-wikipedia-import-pipeline)
8. [MicroModel Training Cycle](#8-micromodel-training-cycle)
9. [ThinkStream Tick with All Subsystems](#9-thinkstream-tick-with-all-subsystems)

---

## 1. API Request Lifecycle (REST)

How a REST API call flows through the Python bridge to the C++ binary and back.

```mermaid
sequenceDiagram
    participant Client as Browser / Client
    participant FA as FastAPI (server.py)
    participant Lock as asyncio.Lock (cmd_lock)
    participant BIN as brain19 binary
    participant SO as SystemOrchestrator

    Client->>FA: POST /api/ask {"question": "What is gravity?"}

    FA->>Lock: acquire cmd_lock
    Note over Lock: Serializes all subprocess calls

    FA->>BIN: subprocess_exec(brain19 --data-dir brain19_data ask "What is gravity?")

    BIN->>SO: initialize()
    Note over SO: Full 14-stage init for each request

    SO->>SO: ask("What is gravity?")
    Note over SO: ThinkingPipeline → ChatInterface → Ollama

    SO-->>BIN: ChatResponse {answer, concepts, epistemic_note}

    BIN->>SO: shutdown()
    BIN-->>FA: stdout text (filtered)

    FA->>Lock: release cmd_lock
    FA->>FA: snapshot_cache.ts = 0 (invalidate)

    FA-->>Client: {"answer": "...", "timestamp": ...}
```

---

## 2. WebSocket Real-Time Flow

WebSocket connection for real-time updates and interactive commands.

```mermaid
sequenceDiagram
    participant VIZ as Brain19Visualizer
    participant WS as WebSocket /ws
    participant FA as FastAPI
    participant BIN as brain19 binary

    VIZ->>WS: connect()
    WS->>FA: ws.accept()
    FA->>FA: ws_clients.add(ws)

    FA->>FA: build_snapshot()
    FA-->>VIZ: {"type": "snapshot", "data": {...}}

    Note over FA: Periodic broadcast (every 10s)
    loop Every 10 seconds
        FA->>FA: build_snapshot()
        FA-->>VIZ: {"type": "snapshot", "data": {...}}
    end

    VIZ->>WS: {"command": "ask", "text": "What is entropy?"}
    FA->>BIN: subprocess_exec(brain19 ask "What is entropy?")
    BIN-->>FA: answer text
    FA-->>VIZ: {"type": "answer", "data": "..."}
    FA->>FA: snapshot_cache.ts = 0
    FA->>FA: build_snapshot()
    FA-->>VIZ: {"type": "snapshot", "data": {...}}

    VIZ->>WS: {"command": "ingest", "text": "Photosynthesis converts..."}
    FA->>BIN: subprocess_exec(brain19 ingest "Photosynthesis converts...")
    BIN-->>FA: result text
    FA-->>VIZ: {"type": "ingested", "data": "..."}

    VIZ->>WS: disconnect
    FA->>FA: ws_clients.discard(ws)
```

---

## 3. Cross-Domain Insight Discovery

DomainManager finds creative connections between different knowledge domains.

```mermaid
sequenceDiagram
    participant SO as SystemOrchestrator
    participant DM as DomainManager
    participant LTM as LongTermMemory

    SO->>DM: find_cross_domain_insights(active_concepts, ltm)

    DM->>DM: cluster_by_domain(active_concepts, ltm)

    loop For each active concept
        DM->>LTM: get_outgoing_relations(cid)
        LTM-->>DM: vector<RelationInfo>
        DM->>LTM: get_incoming_relations(cid)
        LTM-->>DM: vector<RelationInfo>
        DM->>DM: classify_relations(relations, ltm)
        Note over DM: CAUSES/MEASURES → PHYSICAL<br/>PART_OF/PRODUCES → BIOLOGICAL<br/>INFLUENCES → SOCIAL<br/>IS_A/IMPLIES → ABSTRACT<br/>PRECEDES/FOLLOWS → TEMPORAL
    end

    DM-->>DM: clusters: Map<DomainType, vector<ConceptId>>

    loop For each pair of domains (A, B)
        DM->>DM: find bridging concepts
        Note over DM: Concepts that have relations<br/>into BOTH domain A and domain B

        alt Bridging concepts found
            DM->>DM: compute novelty_score
            Note over DM: PHYSICAL↔SOCIAL → 0.8 (high)<br/>BIOLOGICAL↔ABSTRACT → 0.7 (medium)<br/>Same domain → 0.5 (default)

            DM->>DM: CrossDomainInsight(A, B, bridges, desc, novelty)
        end
    end

    DM-->>SO: vector<CrossDomainInsight>
```

---

## 4. Bootstrap — Foundation Seeding

How foundation concepts are seeded when Brain19 starts for the first time.

```mermaid
sequenceDiagram
    participant SO as SystemOrchestrator
    participant FC as FoundationConcepts
    participant BI as BootstrapInterface
    participant CA as ContextAccumulator
    participant LTM as LongTermMemory

    SO->>SO: seed_foundation()

    SO->>FC: get_foundation_concepts()
    Note over FC: Returns predefined concepts:<br/>logic, mathematics, causality,<br/>time, space, evidence, etc.

    FC-->>SO: vector<FoundationConcept>

    loop Per foundation concept
        SO->>LTM: store_concept(label, definition, EpistemicMetadata{DEFINITION, ACTIVE, 1.0})
        LTM-->>SO: ConceptId
    end

    loop Per foundation relation
        SO->>LTM: add_relation(source, target, type, weight)
        LTM-->>SO: RelationId
    end

    SO->>BI: initialize_context(ltm)
    BI->>CA: accumulate(foundation_concepts)
    Note over CA: Build initial context<br/>for first thinking cycle

    Note over SO: Foundation concepts provide<br/>the semantic backbone for<br/>all future knowledge
```

---

## 5. WAL Recovery on Startup

Write-Ahead Log replay during system initialization to recover from crashes.

```mermaid
sequenceDiagram
    participant SO as SystemOrchestrator
    participant PLTM as PersistentLTM
    participant WAL as WALReader
    participant LTM as LongTermMemory

    Note over SO: Stage 1: Load persisted LTM
    SO->>PLTM: new PersistentLTM(data_dir)
    SO->>PLTM: load_into(ltm)
    PLTM->>PLTM: read binary file (concepts + relations)
    PLTM->>LTM: store_concept() × N
    PLTM->>LTM: add_relation() × M
    PLTM-->>SO: loaded (N concepts, M relations)

    Note over SO: Stage 2: Replay WAL entries after last checkpoint
    SO->>WAL: new WALReader(data_dir)
    SO->>WAL: replay_into(ltm)

    loop Per WAL entry
        alt Entry: STORE_CONCEPT
            WAL->>LTM: store_concept(label, def, epistemic)
            Note over LTM: Skip if concept already exists<br/>(from persisted LTM)
        else Entry: ADD_RELATION
            WAL->>LTM: add_relation(source, target, type, weight)
        else Entry: UPDATE_METADATA
            WAL->>LTM: update_epistemic_metadata(id, new_meta)
        end
    end

    WAL-->>SO: replayed K entries

    Note over SO: LTM now has complete state:<br/>checkpoint + all WAL entries since
```

---

## 6. Periodic Maintenance Cycle

The background thread that runs evolution, checkpoints, and cleanup.

```mermaid
sequenceDiagram
    participant Thread as periodic_task_loop
    participant SO as SystemOrchestrator
    participant PD as PatternDiscovery
    participant EP as EpistemicPromotion
    participant CP as ConceptProposer
    participant CM as CheckpointManager
    participant LTM as LongTermMemory
    participant MTX as subsystem_mtx_

    loop While periodic_running_
        Note over Thread: Sleep checkpoint_interval_minutes (30)

        Thread->>MTX: lock()

        Note over Thread: === Evolution Maintenance ===

        Thread->>PD: discover_all()
        PD-->>Thread: vector<DiscoveredPattern>

        Thread->>EP: run_maintenance()
        EP->>EP: evaluate_all()
        loop Per concept
            EP->>EP: check promotion/demotion
            alt Promotion eligible
                EP->>LTM: update_epistemic_metadata(id, promoted)
            else Demotion needed
                EP->>LTM: update_epistemic_metadata(id, demoted)
            end
        end
        EP-->>Thread: MaintenanceResult

        Note over Thread: === Checkpoint ===

        alt checkpoint_interval elapsed
            Thread->>CM: create_checkpoint("periodic")
            CM-->>Thread: checkpoint_dir
        end

        Thread->>MTX: unlock()
    end
```

---

## 7. Wikipedia Import Pipeline

Full flow from Wikipedia URL to LTM concepts.

```mermaid
sequenceDiagram
    participant User
    participant SO as SystemOrchestrator
    participant WI as WikipediaImporter
    participant HC as HttpClient
    participant IP as IngestionPipeline
    participant LTM as LongTermMemory
    participant MR as MicroModelRegistry

    User->>SO: ingest_wikipedia(url)
    SO->>WI: import_article(url)

    WI->>HC: get(wikipedia_api_url)
    Note over HC: libcurl HTTP GET
    HC-->>WI: HTML/JSON response

    WI->>WI: parse_article(response)
    Note over WI: Extract title, summary,<br/>sections, categories

    WI-->>SO: KnowledgeProposal {title, text, categories}

    SO->>IP: ingest_text(proposal.text, "wikipedia:" + url, auto_approve=true)

    Note over IP: Full pipeline runs:<br/>TextChunker → EntityExtractor<br/>→ RelationExtractor → TrustTagger<br/>→ ProposalQueue (auto-approved)

    IP->>IP: commit_approved()
    IP->>LTM: store_concept() × N
    IP->>LTM: add_relation() × M
    IP-->>SO: IngestionResult {concepts_stored, relations_stored}

    SO->>MR: ensure_models_for(new_concept_ids)
    Note over MR: Create MicroModels for new concepts

    SO-->>User: IngestionResult
```

---

## 8. MicroModel Training Cycle

Detailed training flow for a single concept's MicroModel.

```mermaid
sequenceDiagram
    participant MT as MicroTrainer
    participant MR as MicroModelRegistry
    participant MM as MicroModel
    participant EM as EmbeddingManager
    participant LTM as LongTermMemory

    MT->>MR: get_model(cid)
    MR-->>MT: MicroModel* (or create new)

    MT->>MT: generate_samples(cid, embeddings, ltm)

    MT->>LTM: get_outgoing_relations(cid)
    LTM-->>MT: vector<RelationInfo>

    loop Per relation (positive samples)
        MT->>EM: get_relation_embedding(relation.type)
        EM-->>MT: Vec10 e
        MT->>EM: get_embedding(relation.target)
        EM-->>MT: Vec10 c
        MT->>MT: TrainingSample{e, c, target=1.0}
    end

    loop 3× per positive (negative samples)
        MT->>MT: random_concept (not related)
        MT->>EM: get_embedding(random_cid)
        EM-->>MT: Vec10 c_neg
        MT->>MT: TrainingSample{e, c_neg, target=0.0}
    end

    MT->>MM: train(samples, config)

    Note over MM: Adam optimizer loop:
    loop Epoch 1..max_epochs
        MM->>MM: Forward: v = W·c + b, z = eᵀ·v, w = σ(z)
        MM->>MM: Loss: MSE = (w - target)²
        MM->>MM: Backward: dW, db gradients
        MM->>MM: Adam update: momentum + variance correction
        alt loss < convergence_threshold
            Note over MM: Converged!
            MM-->>MT: MicroTrainingResult{converged: true}
        end
    end
    MM-->>MT: MicroTrainingResult{converged, final_loss, epochs}
```

---

## 9. ThinkStream Tick with All Subsystems

A single complete tick of a ThinkStream showing all subsystem interactions.

```mermaid
sequenceDiagram
    participant TS as ThinkStream
    participant WQ as MPMCQueue
    participant SLTM as SharedLTM
    participant SSTM as SharedSTM
    participant SREG as SharedRegistry
    participant SEMB as SharedEmbeddings

    TS->>WQ: try_pop()
    alt Task available
        WQ-->>TS: ThinkTask{Tick, target_concept}
    else Queue empty
        Note over TS: Use round-robin concept sampling
    end

    Note over TS: === do_spreading() ===
    TS->>SSTM: get_active_concepts(ctx, threshold) [shared_lock]
    SSTM-->>TS: vector<ConceptId>
    loop Per active concept (round-robin subset)
        TS->>SLTM: get_outgoing_relations(cid) [shared_lock]
        SLTM-->>TS: vector<RelationInfo>
        loop Per relation
            TS->>SLTM: retrieve_concept(target) [shared_lock]
            SLTM-->>TS: ConceptInfo (trust value)
            Note over TS: new_activation = activation × weight × trust × damping
            TS->>SSTM: activate_concept(ctx, target, new_activation) [per-ctx lock]
        end
    end
    Note over TS: metrics.spreading_ticks++

    Note over TS: === do_salience() ===
    TS->>SSTM: get_active_concepts(ctx, 0.0) [shared_lock]
    SSTM-->>TS: vector<ConceptId>
    loop Per active concept
        TS->>SSTM: get_concept_activation(ctx, cid) [shared_lock]
        TS->>SLTM: get_relation_count(cid) [shared_lock]
        TS->>SLTM: retrieve_concept(cid) [shared_lock]
        Note over TS: salience = activation×0.4 + connectivity×0.2 + trust×0.3 + recency×0.1
    end
    Note over TS: metrics.salience_ticks++

    Note over TS: === do_curiosity() ===
    TS->>SSTM: debug_active_concept_count(ctx) [shared_lock]
    TS->>SSTM: debug_active_relation_count(ctx) [shared_lock]
    Note over TS: Build SystemObservation, detect triggers
    Note over TS: metrics.curiosity_ticks++

    Note over TS: === do_understanding() ===
    TS->>SREG: get_model(cid) [shared_lock]
    SREG-->>TS: MicroModel*
    TS->>SEMB: get_relation_embedding(type) [shared_lock]
    SEMB-->>TS: Vec10
    Note over TS: predict(e, c) → relevance score
    Note over TS: metrics.understanding_ticks++

    Note over TS: metrics.total_ticks++
    Note over TS: last_tick_epoch_us = now()

    alt No work done (all subsystems idle)
        TS->>TS: backoff(idle_count)
        Note over TS: idle_count < 100 → spin<br/>idle_count < 110 → yield<br/>else → sleep(500µs)
    end
```

---

*Generated from actual code in `backend/` and `api/`. Updated: 2026-02-12.*
