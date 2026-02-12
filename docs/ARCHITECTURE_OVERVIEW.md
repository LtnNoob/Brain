# Brain19 — Architecture Overview

> **Status:** Complete Architecture Reference (February 2026)
> **Purpose:** Definitive reference for Brain19's architecture and design philosophy
> **Updated:** 2026-02-12

---

## What Brain19 Is

Brain19 is an **externalized working memory** — a C++20 Cognitive Architecture designed for people with ADHD and autism. It persistently understands contexts, proactively recalls relevant information, discovers patterns, and adapts to individual needs.

Brain19 is **not an LLM replacement**. It is an independently thinking system with epistemic integrity — it knows what it knows, and what it doesn't know.

---

## Core Principle: Brain19 Thinks Independently

The most common misconception: Brain19 uses an LLM for thinking. **This is wrong.**

All cognitive work — relevance computation, logical inference, creativity, validation — is performed by **bilinear MicroModels** with 430 parameters per concept. No LLM is in the critical thinking path.

### MicroModel Architecture

Every concept in the Knowledge Graph has its own MicroModel. The forward computation:

```
v = W·c + b        (10D vector)
z = eᵀ · v         (scalar)
w = σ(z)            (relevance ∈ (0,1))
```

Where:
- `e ∈ ℝ¹⁰` — Relation embedding
- `c ∈ ℝ¹⁰` — Context embedding
- `W ∈ ℝ¹⁰ˣ¹⁰` — Weight matrix (100 parameters)
- `b ∈ ℝ¹⁰` — Bias (10 parameters)
- `σ` — Sigmoid activation
- Total flat size: `EMBED_DIM * EMBED_DIM + EMBED_DIM * 2 + EMBED_DIM * EMBED_DIM + EMBED_DIM = 430`

**430 parameters per concept. No overhead. No LLM. Pure mechanics.**

Training uses the Adam optimizer, fully implemented in C++ — no external dependencies.

> See: [`backend/micromodel/micro_model.hpp`](../backend/micromodel/micro_model.hpp)

---

## System Architecture

Brain19 is a three-tier system: C++ backend binary, Python FastAPI bridge, and React frontend.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Frontend (React/Vite :3019)                      │
│   Brain19Visualizer: STMGraph, EpistemicPanel, CuriosityPanel           │
├──────────────────────────┬──────────────────────────────────────────────┤
│    REST API (:8019)      │         WebSocket /ws (:8019)                │
│    POST /api/ask         │         Real-time snapshots (10s)            │
│    POST /api/ingest      │         Interactive commands                 │
│    GET  /api/snapshot    │                                              │
├──────────────────────────┴──────────────────────────────────────────────┤
│                     FastAPI Bridge (api/server.py)                       │
│              asyncio.Lock serializes subprocess calls                    │
│              subprocess_exec(brain19 --data-dir ... <command>)           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    ┌──────────────────────────┐                         │
│                    │   SystemOrchestrator     │                         │
│                    │   (14 Subsystem Groups)  │                         │
│                    └────────────┬─────────────┘                         │
│                                 │                                       │
│     ┌───────────────────────────┼───────────────────────────┐          │
│     │              CORE MEMORY LAYER                         │          │
│     │  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐ │          │
│     │  │   STM   │  │   LTM    │  │  Epistemic System    │ │          │
│     │  │  Short- │  │  Long-   │  │  6 Types, 4 States   │ │          │
│     │  │  Term   │  │  Term    │  │  Compile-Time        │ │          │
│     │  │  Memory │  │  Memory  │  │  Enforcement         │ │          │
│     │  │         │  │  (KG)    │  │                      │ │          │
│     │  └─────────┘  └──────────┘  └──────────────────────┘ │          │
│     └───────────────────────────────────────────────────────┘          │
│                                 │                                       │
│     ┌───────────────────────────┼───────────────────────────┐          │
│     │           COGNITIVE PROCESSING LAYER                   │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Cognitive       │  │  MicroModel Layer          │ │          │
│     │  │  Dynamics        │  │  (430 Params/Concept)      │ │          │
│     │  │  Spreading Act.  │  │  Relevance Maps            │ │          │
│     │  │  + Salience      │  │  Creativity via Overlay    │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Curiosity       │  │  Understanding Layer       │ │          │
│     │  │  Engine          │  │  (Mini-LLMs, OPTIONAL)     │ │          │
│     │  │  Exploration     │  │  Outputs: HYPOTHESIS only  │ │          │
│     │  │  Triggers        │  │  Trust ceiling 0.3-0.5     │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     └───────────────────────────────────────────────────────┘          │
│                                 │                                       │
│     ┌───────────────────────────┼───────────────────────────┐          │
│     │          KAN-LLM HYBRID LAYER (Phase 7)                │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  KAN Adapter     │  │  Hypothesis Translator     │ │          │
│     │  │  B-Spline KAN    │  │  Linguistic → Numeric      │ │          │
│     │  │  Networks         │  │  Pattern Detection         │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Epistemic       │  │  KAN Validator             │ │          │
│     │  │  Bridge          │  │  End-to-End Validation     │ │          │
│     │  │  MSE → Trust     │  │  Hypothesis → Result       │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Domain Manager  │  │  Refinement Loop           │ │          │
│     │  │  5 Domain Types  │  │  Bidirectional LLM↔KAN     │ │          │
│     │  │  Cross-Domain    │  │  max 5 iterations          │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     └───────────────────────────────────────────────────────┘          │
│                                 │                                       │
│     ┌───────────────────────────┼───────────────────────────┐          │
│     │         EVOLUTION LAYER (Phase 6)                      │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Pattern         │  │  Epistemic Promotion       │ │          │
│     │  │  Discovery       │  │  SPECULATION → HYPOTHESIS  │ │          │
│     │  │  Graph Analysis  │  │  → THEORY → FACT           │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  ┌──────────────────────────────────────────────────┐ │          │
│     │  │  Concept Proposer                                │ │          │
│     │  │  System-generated concepts (trust capped ≤ 0.5)  │ │          │
│     │  └──────────────────────────────────────────────────┘ │          │
│     └───────────────────────────────────────────────────────┘          │
│                                 │                                       │
│     ┌───────────────────────────┼───────────────────────────┐          │
│     │        CONCURRENCY & STREAMS LAYER                     │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Stream          │  │  Shared State              │ │          │
│     │  │  Orchestrator    │  │  SharedLTM, SharedSTM      │ │          │
│     │  │  N ThinkStreams  │  │  SharedRegistry            │ │          │
│     │  │  Lock-Free Queue │  │  SharedEmbeddings          │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  Lock Hierarchy: SharedLTM → SharedSTM → SharedReg    │          │
│     │  → SharedEmbeddings (deadlock prevention)              │          │
│     └───────────────────────────────────────────────────────┘          │
│                                 │                                       │
│     ┌───────────────────────────┼───────────────────────────┐          │
│     │        PERSISTENCE & I/O LAYER                         │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  PersistentLTM   │  │  WAL (Write-Ahead Log)     │ │          │
│     │  │  Binary Storage  │  │  Crash Recovery            │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Checkpoint      │  │  Ingestion Pipeline        │ │          │
│     │  │  Manager         │  │  Text → Entities →         │ │          │
│     │  │  Periodic Save   │  │  Relations → LTM           │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  ┌──────────────────┐  ┌────────────────────────────┐ │          │
│     │  │  Wikipedia       │  │  Chat Interface            │ │          │
│     │  │  Importer        │  │  Ollama LLM Verbalization  │ │          │
│     │  └──────────────────┘  └────────────────────────────┘ │          │
│     │  ┌──────────────────────────────────────────────────┐ │          │
│     │  │  Bootstrap Interface                             │ │          │
│     │  │  Foundation Concepts Seeding                     │ │          │
│     │  └──────────────────────────────────────────────────┘ │          │
│     └───────────────────────────────────────────────────────┘          │
│                                                                         │
│                       C++ Backend Binary (brain19)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The 14 Subsystem Groups

SystemOrchestrator owns all subsystems via `unique_ptr` and performs a 14-stage initialization in dependency order. On failure at any stage, `cleanup_from_stage()` tears down in reverse order.

| # | Stage | Components | Purpose |
|---|-------|------------|---------|
| 1 | **LTM** | LongTermMemory, PersistentLTM | Knowledge Graph with epistemic metadata |
| 2 | **WAL** | WALWriter | Write-ahead log for crash recovery |
| 3 | **Brain** | BrainController, STM | Working memory and context management |
| 4 | **MicroModels** | EmbeddingManager, MicroModelRegistry, MicroTrainer | Per-concept relevance models (430 params each) |
| 5 | **Cognitive** | CognitiveDynamics | Spreading activation, salience, focus management |
| 6 | **Curiosity** | CuriosityEngine | Exploration trigger generation |
| 7 | **KAN** | KANAdapter (KANNode → KANLayer → KANModule) | B-spline Kolmogorov-Arnold Networks |
| 8 | **Understanding** | UnderstandingLayer, MiniLLM | Semantic analysis via Mini-LLMs |
| 9 | **Hybrid** | KanValidator, DomainManager, RefinementLoop, EpistemicBridge, HypothesisTranslator | KAN-LLM bidirectional validation |
| 10 | **Ingestion** | IngestionPipeline (TextChunker → EntityExtractor → RelationExtractor → TrustTagger → ProposalQueue) | Knowledge import pipeline |
| 11 | **Chat** | ChatInterface, OllamaClient | LLM verbalization (llama3.2:1b) |
| 12 | **Shared State** | SharedLTM, SharedSTM, SharedRegistry, SharedEmbeddings | Thread-safe wrappers for multi-stream access |
| 13 | **Streams** | StreamOrchestrator, ThinkStreams, StreamScheduler, StreamMonitor | Parallel autonomous thinking threads |
| 14 | **Evolution** | PatternDiscovery, EpistemicPromotion, ConceptProposer | Dynamic knowledge evolution |

After stage 14, foundation concepts are seeded (if first run) and the periodic maintenance thread starts.

> See: [`backend/core/system_orchestrator.hpp`](../backend/core/system_orchestrator.hpp)

---

## ThinkingPipeline: The 10-Step Cognitive Cycle

The ThinkingPipeline orchestrates a complete thinking cycle. Each step builds on the previous:

```
Step 1:  Activate seed concepts in STM
            │
Step 2:  Spreading Activation (trust-weighted, depth-limited)
            │  activation propagates through LTM relations
            │  new_activation = source_activation × weight × trust × damping
            │
Step 3:  Salience Computation
            │  salience = activation×0.4 + recency×0.1 + connectivity×0.2 + trust×0.3
            │
Step 4:  Build Relevance Maps (top-K salient concepts)
            │  Each MicroModel produces a relevance map over all relations
            │
Step 5:  Combine Relevance Maps → emergent creativity
            │  Methods: multiplication, harmonic mean, surprise-based
            │
Step 6:  Rank Thought Paths (multi-hop reasoning chains)
            │
Step 7:  Curiosity Triggers (what should be explored next?)
            │  Detects: shallow_relations, low_exploration
            │
Step 8:  Understanding Layer (optional, LLM-based)
            │  Outputs: MeaningProposal, HypothesisProposal,
            │           AnalogyProposal, ContradictionProposal
            │
Step 9:  KAN Validation (Phase 7 Hybrid)
            │  HypothesisTranslator → KANModule::train → EpistemicBridge
            │
Step 10: Return ThinkingResult
            {activated_concepts, top_salient, best_paths,
             curiosity_triggers, combined_relevance,
             understanding, validated_hypotheses}
```

Configuration: `initial_activation=0.8`, `top_k_salient=10`, `max_relevance_maps=5`.

> See: [`backend/core/thinking_pipeline.hpp`](../backend/core/thinking_pipeline.hpp)

---

## Creativity Without LLM

Brain19 generates creativity through **overlay of MicroModel relevance maps**:

```
Map("Temperature")  ──┐
                      ├──→  Overlay  ──→  Unexpected patterns
Map("Music")       ──┘                    New hypotheses
```

When the relevance map of "Temperature" and that of "Music" are overlaid, unexpected shared relevances emerge — connections that no single concept alone makes visible.

**Combination methods:**
- **Multiplication:** Finds jointly important relations
- **Harmonic Mean:** Emphasizes overlap
- **Surprise-Based:** `|w₁ - w₂| · max(w₁, w₂)` — finds asymmetric relevances

The Curiosity Engine triggers these overlays based on Spreading Activation and Salience. **Emergent creativity, fully deterministic, fully inspectable.**

---

## LLM: Verbalization Interface Only

The LLM in Brain19 has exactly **one** job: translate structured system output into human language. It is a **verbalizer**, not a thinker.

```
Brain19 Thinking Process        LLM Language Interface
━━━━━━━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━━━━━
MicroModel inference    →       "Based on the connections
Spreading Activation    →        between temperature and
Salience Scores         →        pressure, it follows..."
Epistemic Values        →

(THINKING)                      (SPEAKING)
```

### Kahneman Analogy

| Role | Brain19 Component | Function |
|------|-------------------|----------|
| **System 2** (logical, precise) | MicroModels + Epistemic System | Thinking, Validating, Deciding |
| **System 1** (associative, fast) | LLM (optional) | Only suggestions, no authority |

When the LLM is optionally used for creative hypothesis generation, these **always** pass through epistemic validation. LLM proposals receive a trust ceiling of 0.3–0.5 and are never automatically accepted.

The LLM backend is Ollama (model: `llama3.2:1b` on port 11434), configured via `--ollama-host` and `--ollama-model` flags.

> See: [`backend/llm/chat_interface.hpp`](../backend/llm/chat_interface.hpp)

---

## KAN-LLM Hybrid Validation (Phase 7)

Phase 7 introduces bidirectional validation between Kolmogorov-Arnold Networks and the LLM. This is the core innovation for epistemic integrity of system-generated knowledge.

### KAN Architecture

KAN networks use B-spline basis functions (Cox-de Boor algorithm) instead of fixed activation functions:

```
KANNode (B-spline univariate function)
   │  num_knots_, knots_, coefficients_
   │  output = Σᵢ coefficients_[i] × B_i(x)
   │
KANLayer (n_in × n_out grid of KANNodes)
   │  output_j = Σᵢ phi_{i,j}(input_i)
   │
KANModule (multi-layer network)
   topology: e.g. [3,5,2] → R³ → R²
   train(): gradient descent with DataPoint samples
```

### Hybrid Validation Pipeline

```
Linguistic Hypothesis
    │
    ▼
HypothesisTranslator
    │  Detects pattern: LINEAR, POLYNOMIAL, EXPONENTIAL,
    │  PERIODIC, THRESHOLD, CONDITIONAL, NOT_QUANTIFIABLE
    │  Extracts NumericHints (variable bounds, expected ranges)
    │
    ▼
KANModule::train()
    │  B-spline network learns the hypothesized relationship
    │  Returns: KanTrainingResult {final_mse, epochs, converged}
    │
    ▼
EpistemicBridge
    │  Maps KAN results to epistemic trust:
    │  MSE < 0.01 → THEORY (trust 0.7–0.9)
    │  MSE < 0.10 → HYPOTHESIS (trust 0.4–0.6)
    │  MSE ≥ 0.10 → SPECULATION (trust 0.1–0.3)
    │  Modifiers: synthetic_trust_cap = 0.6
    │             trivial_convergence_penalty = 0.15
    │
    ▼
ValidationResult {epistemic_type, trust, evidence}
```

### Domain-Aware Validation

The DomainManager classifies concepts into 5 knowledge domains based on their relation types:

| Domain | Triggering Relations | KAN Config (knots, hidden) |
|--------|---------------------|---------------------------|
| PHYSICAL | CAUSES, MEASURES | 15 knots, hidden_dim 8 |
| BIOLOGICAL | PART_OF, PRODUCES | 10 knots |
| SOCIAL | INFLUENCES, ASSOCIATED_WITH | 8 knots |
| ABSTRACT | IS_A, IMPLIES | 12 knots, hidden_dim 6 |
| TEMPORAL | PRECEDES, FOLLOWS | 10 knots |

Cross-domain insights receive **novelty scores**: PHYSICAL↔SOCIAL = 0.8 (high), BIOLOGICAL↔ABSTRACT = 0.7 (medium).

### Refinement Loop

The RefinementLoop enables bidirectional dialog between LLM and KAN:

1. LLM generates initial hypothesis
2. KAN validates → produces residual feedback
3. LLM refines hypothesis based on feedback
4. KAN re-validates
5. Repeat until convergence (MSE < 0.01) or max 5 iterations

Returns `RefinementResult` with full `provenance_chain` of all iterations.

> See: [`backend/hybrid/`](../backend/hybrid/)

---

## Dynamic Concept Evolution (Phase 6)

The evolution system enables the knowledge graph to grow and self-correct without human intervention (except for FACT promotion).

### Pattern Discovery

Graph analysis algorithms running on the LTM:
- **Cluster detection:** Groups of densely connected concepts
- **Hierarchy detection:** IS_A chains (BFS/DFS)
- **Bridge concepts:** Connect otherwise separate clusters
- **Cycle detection:** Circular dependency identification
- **Gap detection:** Missing expected connections

### Epistemic Promotion Ladder

```
SPECULATION ──→ HYPOTHESIS ──→ THEORY ──→ FACT
   (0.1–0.3)    (0.3–0.5)     (0.5–0.8)   (0.8–1.0)
                                              │
                                    HUMAN REVIEW REQUIRED
```

- **SPECULATION → HYPOTHESIS:** Requires ≥3 supporting relations
- **HYPOTHESIS → THEORY:** Requires ≥5 supports from THEORY+ level, independent evidence
- **THEORY → FACT:** **NEVER automatic** — always requires human confirmation
- **Demotion:** CAN be automatic when contradictions are detected

### Concept Proposer

System-generated concepts from curiosity triggers, relevance anomalies, and analogies.

**Critical invariant:** Initial trust is CAPPED at 0.5. Initial type must be SPECULATION or HYPOTHESIS. System-generated knowledge can never bypass human review for FACT status.

> See: [`backend/evolution/`](../backend/evolution/)

---

## Epistemic Integrity

Brain19's epistemic system is not a feature — it is the **foundation**.

### Compile-Time Enforcement

```cpp
ConceptInfo() = delete;  // No concept without epistemic classification
```

It is **impossible** to insert a concept into the Knowledge Graph without specifying its epistemic status. This is enforced at compile time — not at runtime, not by convention, but by the compiler.

### The 6 Epistemic Types

| Epistemic Type | Trust Range | Meaning | Entry Method |
|----------------|-------------|---------|--------------|
| **FACT** | 0.8–1.0 | Verified, reproducible | Human confirmation only |
| **DEFINITION** | ~1.0 | Tautological (e.g., "triangle has 3 sides") | Direct construction |
| **THEORY** | 0.5–0.8 | Evidence-based, falsifiable | Promotion from HYPOTHESIS |
| **HYPOTHESIS** | 0.3–0.5 | Testable, unconfirmed | Promotion from SPECULATION |
| **INFERENCE** | Varies | Derived from other knowledge | Direct construction |
| **SPECULATION** | 0.1–0.3 | No evidence, idea | System-generated default |

### The 4 Epistemic Statuses

| Status | Meaning | Transitions To |
|--------|---------|---------------|
| **ACTIVE** | Full participation in all computations | CONTEXTUAL, SUPERSEDED, INVALIDATED |
| **CONTEXTUAL** | Valid only in specific contexts | ACTIVE, SUPERSEDED, INVALIDATED |
| **SUPERSEDED** | Replaced by better knowledge (not wrong, just outdated) | INVALIDATED |
| **INVALIDATED** | Trust set to 0.05, but **NEVER deleted** | Terminal state |

**Knowledge is never deleted.** Invalidated concepts remain in LTM with low trust, preserving epistemic history.

### No Hallucinations

Brain19 cannot hallucinate. The system knows exactly:
- What it knows (FACT, THEORY)
- What it suspects (HYPOTHESIS)
- What it doesn't know (missing concepts)
- What is unreliable (SPECULATION, LLM proposals with low trust)

LLM output is **always** epistemically validated before entering the Knowledge Graph. Contradictions to existing FACT/THEORY knowledge lead to automatic rejection.

> See: [`backend/epistemic/epistemic_metadata.hpp`](../backend/epistemic/epistemic_metadata.hpp)

---

## Multi-Stream Architecture

Brain19 scales horizontally through parallel ThinkStreams — autonomous thinking threads that share state through thread-safe wrappers.

### ThinkStream Lifecycle

Each ThinkStream runs an independent tick loop:

```
1. try_pop(work_queue)         — Check for assigned tasks
2. do_spreading()              — Trust-weighted activation propagation
3. do_salience()               — Compute salience scores
4. do_curiosity()              — Generate exploration triggers
5. do_understanding()          — MicroModel inference
6. backoff if idle             — Spin → yield → sleep(500µs)
```

State machine: `Created → Starting → Running → Stopping → Stopped` (or `Error`).

### Shared State with Lock Hierarchy

All ThinkStreams access shared state through reader-writer lock wrappers:

| Wrapper | Protects | Lock Order |
|---------|----------|------------|
| **SharedLTM** | LongTermMemory (Knowledge Graph) | 1st (highest priority) |
| **SharedSTM** | ShortTermMemory (activation state) | 2nd |
| **SharedRegistry** | MicroModelRegistry | 3rd |
| **SharedEmbeddings** | EmbeddingManager | 4th (lowest priority) |

**Lock hierarchy rule:** Locks must always be acquired in descending priority order to prevent deadlocks. A DeadlockDetector monitors for violations.

### Work Distribution

The StreamOrchestrator distributes `ThinkTask` items via a **lock-free MPMC queue** (Vyukov bounded queue with ABA-safe sequence counters):

```
StreamOrchestrator
    │
    ├── auto_scale()        — Scale streams to hardware
    ├── distribute_task()   — Push to MPMC queue
    ├── health_check()      — Detect stalled streams
    │
    └── ThinkStream[0..N]
         └── MPMCQueue<ThinkTask> (shared work queue)
```

### Scaling

| Hardware | Streams | Use Case |
|----------|---------|----------|
| i5-6600K (4 cores) | 4 | Development |
| EPYC 80-core | 80 | Production |
| 10× EPYC cluster | 800 | Massively parallel |

- **Lock-free design:** No mutex bottlenecks in the hot path
- **MicroModels are independent:** Each concept has its own model, no shared weights
- **Config-driven:** System detects available hardware and uses it automatically
- **Linear scaling:** Double cores = double parallel thinking

> See: [`backend/streams/`](../backend/streams/), [`backend/concurrent/`](../backend/concurrent/)

---

## Persistence Layer

Brain19 ensures durability through a multi-layered persistence system.

### PersistentLTM

Binary storage of the complete Knowledge Graph (concepts + relations). Loaded at startup before WAL replay.

### Write-Ahead Log (WAL)

All mutations (store_concept, add_relation, update_metadata) are first written to the WAL before being applied. On crash recovery:

1. Load PersistentLTM snapshot (last checkpoint)
2. Replay WAL entries since last checkpoint
3. Result: complete state recovery

Entry types: `STORE_CONCEPT`, `ADD_RELATION`, `UPDATE_METADATA`.

### Checkpoint Manager

Periodic checkpoints (every 30 minutes by default) write a consistent PersistentLTM snapshot and truncate the WAL. Checkpoint/restore is also available as a CLI command.

### STM Snapshot

Serializable snapshot of the Short-Term Memory state for the FastAPI bridge. Includes all active concepts, activation levels, and context state.

> See: [`backend/persistent/`](../backend/persistent/)

---

## Ingestion Pipeline

Knowledge enters Brain19 through a multi-stage ingestion pipeline:

```
Raw Input (text, URL, JSON, CSV)
    │
    ▼
TextChunker
    │  Split into processable chunks
    ▼
EntityExtractor
    │  Identify concepts/entities
    ▼
RelationExtractor
    │  Identify relationships between entities
    ▼
TrustTagger
    │  Assign initial epistemic metadata
    ▼
ProposalQueue
    │  PENDING → [Human Review] → APPROVED/REJECTED
    │  (or auto_approve=true for Wikipedia imports)
    ▼
LTM.store_concept() + LTM.add_relation()
```

**Critical invariant:** The ingestion pipeline is ADDITIVE only — it NEVER modifies existing knowledge. New knowledge coexists with existing knowledge; contradictions are resolved through the epistemic system.

### Wikipedia Import

The WikipediaImporter fetches articles via the Wikipedia API (libcurl HTTP), extracts title/summary/sections/categories, and feeds them through the full ingestion pipeline with `auto_approve=true`.

> See: [`backend/ingestor/`](../backend/ingestor/), [`backend/importers/`](../backend/importers/)

---

## Bootstrap System

On first startup, Brain19 seeds **foundation concepts** — the semantic backbone for all future knowledge:

- **Foundation concepts:** logic, mathematics, causality, time, space, evidence, etc.
- Each gets `EpistemicMetadata{DEFINITION, ACTIVE, trust=1.0}`
- Foundation relations create the initial graph structure
- The ContextAccumulator builds the initial context for the first thinking cycle

> See: [`backend/bootstrap/`](../backend/bootstrap/)

---

## Three-Tier Deployment

### C++ Backend Binary

The `brain19` binary is a REPL application (`Brain19App`) wrapping `SystemOrchestrator`. CLI commands: `ask`, `ingest`, `import`, `status`, `streams`, `checkpoint`, `restore`, `concepts`, `explain`, `think`, `help`.

Flags: `--data-dir`, `--ollama-host`, `--ollama-model`, `--no-persistence`, `--no-foundation`, `--max-streams`, `--no-monitor`.

### Python FastAPI Bridge (api/server.py)

Port 8019. Translates HTTP/WebSocket requests into subprocess calls to the brain19 binary. An `asyncio.Lock` (`cmd_lock`) serializes all subprocess calls. Snapshot caching with invalidation on mutations.

### React Frontend (frontend/src/Brain19Visualizer.jsx)

Port 3019 (Vite dev server). Components:
- **STMGraph:** Force-directed SVG visualization of active concepts
- **EpistemicPanel:** Trust and type display for selected concepts
- **CuriosityPanel:** Display of current exploration triggers

WebSocket connection receives snapshot updates every 10 seconds.

---

## Periodic Maintenance

A background thread runs every 30 minutes (configurable via `checkpoint_interval_minutes`):

1. **Pattern Discovery:** `discover_all()` — find clusters, bridges, gaps
2. **Epistemic Promotion:** `run_maintenance()` — evaluate all concepts for promotion/demotion
3. **Checkpoint:** Create persistent snapshot if interval elapsed

All maintenance runs under `subsystem_mtx_` to ensure consistency.

---

## Summary

| Property | Implementation |
|----------|---------------|
| Independent Thinking | MicroModels (430 params/concept) |
| Creativity | Overlay of relevance maps |
| Language | LLM as verbalizer (optional, Ollama) |
| Truth | Compile-time epistemic enforcement |
| Validation | KAN-LLM hybrid bidirectional (Phase 7) |
| Evolution | Pattern discovery + epistemic promotion (Phase 6) |
| Scaling | 1 core → N cores, linear via ThinkStreams |
| Hallucinations | Impossible (trust system + epistemic types) |
| Persistence | WAL + PersistentLTM + Checkpoints |
| Knowledge Import | Multi-stage pipeline with human review |
| Purpose | Externalized working memory (ADHD/Autism) |

---

## Related Documentation

- [`ARCHITECTURE_DIAGRAMS.md`](ARCHITECTURE_DIAGRAMS.md) — 17 Mermaid component, flow, and data-flow diagrams
- [`CLASS_DIAGRAMS.md`](CLASS_DIAGRAMS.md) — 11 UML class diagrams for all module hierarchies
- [`SEQUENCE_DIAGRAMS.md`](SEQUENCE_DIAGRAMS.md) — 9 extended sequence diagrams for key workflows
- [`STATE_DIAGRAMS.md`](STATE_DIAGRAMS.md) — 9 state machine diagrams for all stateful components
- [`PROJECT_VISION.md`](PROJECT_VISION.md) — Motivation and purpose
- [`KAN_LLM_HYBRID_THEORY.md`](KAN_LLM_HYBRID_THEORY.md) — KAN-LLM hybrid architecture, epistemic theory
- [`KAN_RELEVANCE_MAPS_ANALYSIS.md`](KAN_RELEVANCE_MAPS_ANALYSIS.md) — MicroModel architecture analysis
- [`DESIGN_THEORY.md`](DESIGN_THEORY.md) — Domain-Auto-LLM theory
- [`technical/DOCUMENTATION_TASK.md`](technical/DOCUMENTATION_TASK.md) — HTML documentation specification

---

*Felix Hirschpek, 2026*
