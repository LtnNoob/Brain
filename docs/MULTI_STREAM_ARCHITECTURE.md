# Brain19 Multi-Stream Architecture

## Phase 3: Parallel Thinking Streams

---

## 1. Executive Summary

Brain19's knowledge graph (LTM), epistemic metadata, micro-model relevance maps, cognitive
dynamics, and short-term memory are all operational — but single-threaded. A human brain
doesn't think one thought at a time. Neither should Brain19.

**Purpose**: Enable multidimensional parallel thinking. Multiple streams simultaneously
query, train, ingest, validate, create, and monitor — each a mechanical worker executing
its assigned task, never an autonomous agent making decisions.

**Targets**:
- **Primary**: 80 parallel streams on AMD EPYC 80-core (1 stream per core)
- **Future**: 100+ streams on Q.ANT photonic NPU (optical processing lanes)

**Design philosophy**: "Tools not Agents." Every stream is a deterministic, bounded,
mechanical delegate. Streams don't decide *what* to think — the orchestrator assigns work,
streams execute it. This matches the existing Brain19 contract: STM never evaluates
importance, CognitiveDynamics never generates hypotheses, BrainController never reasons.
Streams are the same: pure execution, zero autonomy.

**Key constraints**:
- Knowledge is NEVER deleted, only INVALIDATED (epistemic integrity preserved)
- LLM-generated proposals carry permanent origin markers (Provenienz ist unverlierbar)
- All shared state access through thread-safe wrappers (existing code unchanged)
- Lock-free fast paths for reads; locks only on writes
- Design maps cleanly to photonic hardware (no recursive mutexes, no condition variables in hot paths)

---

## 2. Threading Model

### Hybrid Architecture: Dedicated Coordinator + Worker Pool

```
┌─────────────────────────────────────────────────────┐
│                  StreamOrchestrator                  │
│  (1 dedicated jthread — the "conductor")            │
│                                                      │
│  Responsibilities:                                   │
│  - Monitor stream health & metrics (~10Hz)           │
│  - Rebalance work across streams                     │
│  - Spawn/retire streams based on load                │
│  - Aggregate cross-stream insights                   │
│  - Detect stuck streams (heartbeat timeout >5s)      │
└────────────┬──────────────────────────┬──────────────┘
             │                          │
    ┌────────▼────────┐       ┌────────▼────────┐
    │  StreamPool      │       │  TaskQueue       │
    │  (N jthreads)    │◄──────│  (MPMC lock-free)│
    │                  │       │                  │
    │  Each thread:    │       │  Priority lanes: │
    │  - owns 1 STM    │       │  [CRITICAL]      │
    │    ContextId      │       │  [HIGH]          │
    │  - has StreamId   │       │  [NORMAL]        │
    │  - runs task loop │       │  [BACKGROUND]    │
    │  - reports metrics│       │                  │
    └──────────────────┘       └──────────────────┘
```

### Why Hybrid, Not Pure Thread-Pool?

A pure thread-pool treats all work as fungible. Brain19's workloads are not:

- **User queries** need guaranteed latency — they get dedicated, pinned streams that
  never compete for scheduling with batch training work.
- **Training** is embarrassingly parallel and perfectly suited to pooling — any idle
  thread can pick up any training task.
- **The orchestrator** must never block on work. Its job is pure oversight: monitoring
  heartbeats, rebalancing load, detecting anomalies. If the orchestrator were pooled
  with workers, a burst of training tasks could starve monitoring.

**Photonic mapping**: Dedicated streams map to fixed optical processing lanes (always
available, deterministic latency). Pool streams map to reconfigurable lanes (assigned
on demand). The orchestrator maps to the optical control plane.

### C++20 Primitives

| Primitive | Usage | Why |
|-----------|-------|-----|
| `std::jthread` | All threads (RAII, auto-join) | Clean shutdown, no leaked threads |
| `std::stop_token` | Cooperative cancellation | Non-destructive, checkable in task loop |
| `std::atomic<T>` | Lock-free counters, flags, metrics | Zero-contention reads for monitoring |
| `std::latch` | Startup barrier (all streams ready before work begins) | One-shot synchronization |
| `std::barrier` | Shutdown coordination (all streams stopped before teardown) | Reusable sync point |
| `std::shared_mutex` | Reader-writer locks on shared state | Many readers, rare writers |
| Custom MPMC queue | Task distribution (atomic head/tail ring buffer) | No mutex in hot path |

**Not used** (photonic portability constraint):
- `std::recursive_mutex` — not mappable to photonic latches
- `std::condition_variable` in hot paths — use spin-wait with exponential backoff
- Dynamic thread creation after startup — fixed stream count, reconfigurable assignment

---

## 3. Stream Specialization

Six categories of thinking stream, each with distinct behavior and resource access:

| Category | Count | Priority | Owns STM Context? | LTM Access | Description |
|----------|-------|----------|-------------------|------------|-------------|
| **Query** | 4-8 | CRITICAL | Yes | Read-only | Handle user queries: compute relevance maps, spreading activation, salience ranking |
| **Training** | 16-32 | NORMAL | No | Read-only | Background micro-model training via `MicroTrainer` (embarrassingly parallel) |
| **Ingestion** | 4-8 | HIGH | Yes | Read+Write | Parallel `IngestionPipeline` runs: chunk, extract, tag, store |
| **Validation** | 8-16 | BACKGROUND | Yes | Read-only | Trust decay checks, consistency validation, contradiction detection |
| **Creative** | 8-16 | NORMAL | Yes | Read-only | Hypothesis generation: cross-domain `RelevanceMap::overlay()`, novel connections |
| **Meta** | 2-4 | HIGH | No | None | Monitor stream patterns, detect emergent insights from cross-stream metrics |

**Total**: ~80 streams (configurable at runtime via `StreamConfig`)

### Category Design Decisions

**Query streams are pinned** (dedicated `jthread`, never pooled):
- Latency-critical: user is waiting for a response
- Each query stream owns a dedicated STM `ContextId` for its query session
- Only consumes from the CRITICAL task lane — never steals lower-priority work
- Uses `CognitiveDynamics::spread_activation()` + `compute_query_salience()` + `RelevanceMap::compute()`

**Training streams are pooled** (one model per task, no shared state between tasks):
- Each task trains one `MicroModel` for one `ConceptId`
- No STM needed — training uses `TrainingSample` vectors directly
- Perfect parallelism: models are independent, no inter-task communication
- Uses `MicroTrainer::train()` with `TrainingConfig`

**Ingestion streams own STM contexts** for entity/relation extraction workspace:
- Each stream runs a full `IngestionPipeline`: TextChunker → EntityExtractor → RelationExtractor → TrustTagger → ProposalQueue
- Write access to LTM (via `SharedLTM`) for `store_concept()` and `add_relation()`
- ProposalQueue entries carry `EpistemicType::SPECULATION` until human review

**Validation streams are background-priority**:
- Check trust consistency: concepts with `EpistemicStatus::ACTIVE` but stale trust
- Detect contradictions: `RelationType::CONTRADICTS` between active concepts
- Flag superseded knowledge: `EpistemicStatus::SUPERSEDED` candidates
- Never modify trust directly — produce validation reports for human review

**Creative streams explore cross-domain connections**:
- Compute `RelevanceMap` for concept A in domain X
- Compute `RelevanceMap` for concept B in domain Y
- `RelevanceMap::overlay()` with `OverlayMode::ADDITION` or `OverlayMode::WEIGHTED_AVERAGE`
- High-scoring overlay results → hypothesis proposals (trust 0.10-0.40, `EpistemicType::SPECULATION`)
- Bounded: time-limited creative sessions (configurable, default 30s)

**Meta streams are read-only observers**:
- Read `StreamMetrics` from all other streams (atomic counters, zero contention)
- Detect patterns: "training throughput dropped 50%" → alert
- Detect emergent insights: "query streams keep activating the same concept cluster" → flag
- Never touch LTM, STM, or any knowledge data

---

## 4. Shared State Protection

### Problem

The existing subsystems are single-threaded:
- `LongTermMemory` — the knowledge graph (concepts, relations, epistemic metadata)
- `MicroModelRegistry` — per-concept bilinear micro-models (`FLAT_SIZE` = 430 doubles each)
- `EmbeddingManager` — relation type embeddings (10 fixed `Vec10`) + context embeddings
- `ShortTermMemory` — activation state per `ContextId`

### Solution: Thread-Safe Wrapper Layer

Adapter pattern: wrap each subsystem in a thread-safe shell. **No modification to
existing code.** The wrappers borrow non-owning references and add synchronization.

```
┌─────────────────────────────────────────────────────────┐
│                  Thread-Safe Wrappers                    │
│                                                          │
│  SharedLTM                                               │
│  ├── std::shared_mutex rwlock_                           │
│  ├── read_lock()  → shared lock for queries              │
│  │   retrieve_concept(), exists(), get_outgoing_relations│
│  │   get_concepts_by_type(), get_active_concepts()       │
│  ├── write_lock() → exclusive lock for mutations         │
│  │   store_concept(), add_relation(), invalidate_concept │
│  │   update_epistemic_metadata()                         │
│  └── wraps LongTermMemory& (non-owning reference)        │
│                                                          │
│  SharedRegistry                                          │
│  ├── std::shared_mutex registry_lock_                    │
│  │   Protects: create_model(), has_model(), get_model_ids│
│  ├── Per-model std::mutex model_locks_[]                 │
│  │   Lock the model, not the registry.                   │
│  │   predict() and train_step() on different models      │
│  │   run concurrently with zero contention.              │
│  └── wraps MicroModelRegistry& (non-owning reference)    │
│                                                          │
│  SharedEmbeddings                                        │
│  ├── std::shared_mutex rwlock_                           │
│  ├── read_lock()  → get_relation_embedding() (hot path)  │
│  │   get_context_embedding() for existing contexts       │
│  ├── write_lock() → get_context_embedding() new context  │
│  │   (auto-creates on first access — rare, amortized)    │
│  └── wraps EmbeddingManager& (non-owning reference)      │
│                                                          │
│  SharedSTM                                               │
│  ├── Per-context std::mutex context_locks_[]             │
│  │   Each stream owns its ContextId → natural isolation  │
│  │   Zero contention in the common case.                 │
│  ├── std::mutex create_lock_                             │
│  │   Only for create_context() / destroy_context()       │
│  └── wraps ShortTermMemory& (non-owning reference)       │
└─────────────────────────────────────────────────────────┘
```

### Lock Hierarchy

To prevent deadlocks, locks are **always** acquired in this fixed order:

```
Level 1: SharedLTM.rwlock_             (coarsest — knowledge graph)
Level 2: SharedRegistry.registry_lock_ (model registry)
Level 3: SharedEmbeddings.rwlock_      (embeddings)
Level 4: SharedRegistry.model_locks_[i] (per-model, finest)
Level 5: SharedSTM.context_locks_[i]   (per-context, finest)
```

**Rules**:
- Never acquire a higher-level lock while holding a lower-level lock
- Never hold two locks at the same level simultaneously
- Per-model and per-context locks are at the finest level — safe to hold one each
  (they are at the same level but are disjoint lock sets, never cross-acquired)
- Read locks (shared) do not participate in deadlock (multiple readers coexist)

### Lock-Free Fast Paths

The common case for most streams is **reading**:

| Operation | Lock Type | Contention |
|-----------|-----------|------------|
| `retrieve_concept(id)` | shared read on LTM | Near-zero (many readers) |
| `get_relation_embedding(type)` | shared read on Embeddings | Near-zero |
| `get_context_embedding(name)` (existing) | shared read on Embeddings | Near-zero |
| `MicroModel::predict(e, c)` | per-model mutex | Zero (each stream trains different model) |
| STM operations | per-context mutex | Zero (each stream owns its context) |
| Stream metrics reads | `atomic<uint64_t>` load | Zero (lock-free) |

**Write operations** (exclusive locks, infrequent):

| Operation | Lock Type | Frequency |
|-----------|-----------|-----------|
| `store_concept()` | exclusive write on LTM | Ingestion only |
| `add_relation()` | exclusive write on LTM | Ingestion only |
| `invalidate_concept()` | exclusive write on LTM | Rare |
| `create_model()` | exclusive write on Registry | Ingestion only |
| `create_context()` | STM create_lock_ | Stream startup only |

### Photonic Portability Note

- `shared_mutex` maps to optical read/write arbitration (multiple read channels, exclusive write channel)
- Per-model isolation maps to independent photonic channels (one model per optical lane)
- Per-context isolation maps the same way (one context per optical lane)
- The wrapper interface stays identical — only the synchronization primitive changes
- All locks are non-recursive (photonic constraint satisfied)

---

## 5. Task Queue Design

### Multi-Lane MPMC Lock-Free Queue

```
TaskQueue
├── Lane[CRITICAL]    ← user queries (always drained first)
├── Lane[HIGH]        ← ingestion tasks, meta-cognition
├── Lane[NORMAL]      ← training tasks, creative exploration
└── Lane[BACKGROUND]  ← validation, maintenance, trust decay

Each Lane:
├── ring buffer: power-of-2 size, atomic head/tail indices
├── default capacity: configurable (256-1024 per lane)
├── overflow policy: back-pressure signal to producer
│   (producer spins briefly, then returns QUEUE_FULL status)
└── memory layout: cache-line aligned entries (avoid false sharing)
```

### StreamTask: The Unit of Work

```cpp
struct StreamTask {
    uint64_t        task_id;          // Monotonic, from atomic counter
    StreamTaskType  task_type;        // QUERY, TRAIN, INGEST, VALIDATE, CREATE, META
    TaskPriority    priority;         // CRITICAL, HIGH, NORMAL, BACKGROUND
    ConceptId       target_concept;   // 0 = no specific target
    TaskPayload     payload;          // variant<QueryPayload, TrainPayload, ...>
    time_point      created_at;       // steady_clock
    optional<time_point> deadline;    // For latency-sensitive tasks
    shared_ptr<TaskResult> result;    // Caller awaits via atomic flag
    uint64_t        parent_task_id;   // 0 = root task (enables task trees)
};
```

### Task Type Payloads

```cpp
// TaskPayload = variant<QueryPayload, TrainPayload, IngestPayload,
//                       ValidatePayload, CreativePayload, MetaPayload>

struct QueryPayload {
    std::vector<ConceptId> seed_concepts;    // Starting points for spreading activation
    std::string context_name;                // EmbeddingManager context (e.g. "query", "analytical")
    RelationType relation_type;              // For RelevanceMap::compute()
    size_t top_k;                            // How many results to return
};

struct TrainPayload {
    ConceptId model_id;                      // Which MicroModel to train
    std::vector<TrainingSample> samples;     // Training data (Vec10 e, Vec10 c, double target)
    TrainingConfig config;                   // Learning rate, epochs, convergence threshold
};

struct IngestPayload {
    std::string text;                        // Raw text to ingest
    EpistemicType default_type;              // Default epistemic classification
    double default_trust;                    // Default trust level
};

struct ValidatePayload {
    std::vector<ConceptId> targets;          // Concepts to validate
    ValidationType check_type;               // TRUST_DECAY, CONTRADICTION, CONSISTENCY, SUPERSEDED
};

struct CreativePayload {
    ConceptId concept_a;                     // First concept for overlay
    ConceptId concept_b;                     // Second concept for overlay
    OverlayMode overlay_mode;                // ADDITION, MAX, WEIGHTED_AVERAGE
    double overlay_weight;                   // Weight parameter for WEIGHTED_AVERAGE
    std::chrono::seconds time_limit;         // Bounded creative session (default 30s)
};

struct MetaPayload {
    MetaTaskType meta_type;                  // THROUGHPUT_CHECK, PATTERN_DETECT, HEALTH_SCAN
    std::chrono::seconds window;             // Observation window
};
```

### TaskResult: Completion Signal

```cpp
struct TaskResult {
    std::atomic<bool>        completed{false};    // Polled by caller (lock-free)
    bool                     success{false};
    std::string              error_message;
    std::chrono::nanoseconds duration;

    // Type-specific results
    variant<
        QueryResult,       // top-k concepts with scores, RelevanceMap
        TrainResult,       // final loss, epochs trained, convergence status
        IngestResult,      // concepts stored count, relations stored count
        ValidateResult,    // issues found (contradictions, stale trust, etc.)
        CreativeResult,    // overlay RelevanceMap, hypothesis proposals
        MetaResult         // pattern observations, alerts generated
    > data;
};
```

### Work Stealing

Idle streams follow this priority order:

1. **Check local queue** (SPSC, orchestrator → this stream) — highest priority
2. **Check global lane** matching stream category (e.g., Training → NORMAL lane)
3. **Steal from lower-priority lanes** (NORMAL stream may steal from BACKGROUND)

**Exception**: Query streams ONLY consume from CRITICAL lane. They never steal from
lower-priority lanes — this guarantees query latency is never impacted by batch work.

**Meta streams** consume from HIGH lane but may also process META-typed tasks from NORMAL
lane when HIGH is empty.

---

## 6. ThinkStream Anatomy

### Structure

```cpp
class ThinkStream {
    StreamId            stream_id_;        // uint32_t, unique across all streams
    StreamCategory      category_;         // QUERY, TRAINING, INGESTION, ...
    std::jthread        thread_;           // Owns the OS thread (RAII)
    std::optional<ContextId> stm_context_; // Own STM sandbox (if category needs it)
    StreamMetrics       metrics_;          // Atomic counters for monitoring

    // Local queue: orchestrator pushes targeted work here (SPSC)
    SPSCQueue<StreamTask> local_queue_;

    // Borrowed shared state (non-owning pointers, lifetime managed by main())
    SharedLTM*          shared_ltm_;
    SharedRegistry*     shared_registry_;
    SharedEmbeddings*   shared_embeddings_;
    SharedSTM*          shared_stm_;

    // Global task queue reference (for work stealing)
    TaskQueue*          global_queue_;
};
```

### StreamMetrics (Lock-Free)

```cpp
struct StreamMetrics {
    std::atomic<uint64_t> tasks_completed{0};
    std::atomic<uint64_t> tasks_failed{0};
    std::atomic<uint64_t> total_busy_ns{0};
    std::atomic<uint64_t> total_idle_ns{0};
    std::atomic<uint64_t> current_task_id{0};    // 0 = idle
    std::atomic<uint64_t> last_heartbeat_ns{0};  // steady_clock epoch nanoseconds
    std::atomic<uint64_t> steal_count{0};         // work stolen from other lanes
    std::atomic<uint64_t> backoff_count{0};       // times entered backoff
};
```

All metrics use `atomic<uint64_t>` (single-word CAS) — photonic-compatible, zero
contention with the monitoring thread.

### Main Loop

```
ThinkStream::run(std::stop_token stop):
    // Initialization
    if (category needs STM):
        stm_context_ = shared_stm_->create_context()
    metrics_.last_heartbeat_ns = now_ns()

    while (!stop.stop_requested()):
        // 1. Try local queue first (orchestrator-targeted work)
        task = local_queue_.try_pop()

        // 2. Try global queue by priority
        if (!task):
            task = global_queue_->try_pop(preferred_lane(category_))

        // 3. Work stealing (except Query streams)
        if (!task && category_ != QUERY):
            task = global_queue_->try_steal_lower(preferred_lane(category_))

        // 4. Execute or backoff
        if (task):
            metrics_.current_task_id.store(task->task_id)
            start = steady_clock::now()

            execute(task)    // dispatch to category-specific handler

            elapsed = steady_clock::now() - start
            metrics_.total_busy_ns += elapsed.count()
            metrics_.tasks_completed++
            metrics_.current_task_id.store(0)
        else:
            backoff()  // exponential, capped at 1ms
            metrics_.backoff_count++

        // 5. Heartbeat (every iteration)
        metrics_.last_heartbeat_ns.store(now_ns())

    // Shutdown
    if (stm_context_):
        shared_stm_->destroy_context(*stm_context_)
```

### Task Dispatch

```
ThinkStream::execute(StreamTask& task):
    switch (task.task_type):
        case QUERY:
            // 1. Activate seed concepts in own STM context
            // 2. CognitiveDynamics::spread_activation() via shared references
            // 3. RelevanceMap::compute() for each seed x relation_type
            // 4. CognitiveDynamics::compute_query_salience()
            // 5. Return top-k results in TaskResult

        case TRAIN:
            // 1. SharedRegistry::lock_model(model_id) -> per-model mutex
            // 2. MicroModel::train(samples, config) via MicroTrainer
            // 3. Unlock model
            // 4. Return TrainResult (loss, epochs)

        case INGEST:
            // 1. IngestionPipeline::process(text) -> proposals
            // 2. SharedLTM::write_lock() -> store_concept(), add_relation()
            // 3. Return IngestResult (counts)

        case VALIDATE:
            // 1. SharedLTM::read_lock() -> retrieve concepts
            // 2. Check trust, contradictions, consistency
            // 3. Return ValidateResult (issues found)

        case CREATE:
            // 1. RelevanceMap::compute() for concept_a
            // 2. RelevanceMap::compute() for concept_b
            // 3. RelevanceMap::overlay() with specified mode
            // 4. High-scoring results -> hypothesis proposals
            // 5. Return CreativeResult (overlay map, proposals)

        case META:
            // 1. Read StreamMetrics from all streams (atomic loads)
            // 2. Compute aggregates, detect patterns
            // 3. Return MetaResult (observations, alerts)
```

### Backoff Strategy

```
backoff():
    spin_count++
    if spin_count <= 4:
        // Spin: ~10-40ns (L1 cache hit latency)
        std::atomic_thread_fence(memory_order_seq_cst)
    elif spin_count <= 8:
        // Yield: ~1-10us (OS scheduler quantum)
        std::this_thread::yield()
    else:
        // Sleep: 100us increments, capped at 1ms
        sleep_for(min(100us * (spin_count - 8), 1ms))
        // Reset after hitting cap to restart the cycle
        if spin_count > 18: spin_count = 4
```

Three-tier backoff: keeps latency low when work is available (spin), avoids CPU waste
during idle periods (yield/sleep). The 1ms cap ensures streams respond quickly when new
work arrives.

**Photonic note**: Spin maps to optical polling. Sleep maps to optical idle (lane powered
down). No `condition_variable` needed.

---

## 7. Performance Monitoring

### StreamMonitor

Runs inside the orchestrator thread at ~10Hz sample rate:

```
StreamMonitor
├── Per-stream metrics (read from StreamMetrics atomics):
│   ├── Utilization: busy_ns / (busy_ns + idle_ns) — target >70% for workers
│   ├── Throughput: tasks_completed delta / sample_interval
│   ├── Current task: task_id (0 = idle)
│   └── Heartbeat: time since last heartbeat (>5s = STUCK alert)
│
├── Per-lane metrics (read from TaskQueue atomics):
│   ├── Queue depth: pending tasks per priority lane
│   ├── Saturation: depth / capacity (>80% = PRESSURE alert)
│   └── Drain rate: tasks consumed per second
│
├── Aggregate metrics (computed by monitor):
│   ├── Global throughput: sum of per-stream throughput
│   ├── Stream balance: stddev of per-stream utilization
│   │   (high stddev = hot spots, rebalance needed)
│   ├── Lock contention: estimated from busy/idle ratio anomalies
│   └── Task latency: time from task creation to completion
│       (p50, p95, p99 per task type)
│
└── Alerts:
    ├── STUCK_STREAM: heartbeat >5s → log + optional restart
    ├── QUEUE_PRESSURE: lane >80% full → log + scale hint
    ├── THROUGHPUT_DROP: >50% drop in 10s window → log
    └── IMBALANCE: utilization stddev >0.3 → rebalance suggestion
```

### Latency Tracking

Per-task latency is tracked without additional locks:

```cpp
// In StreamTask (set by producer when enqueuing):
time_point created_at;

// In ThinkStream::execute():
time_point started_at = steady_clock::now();
// ... do work ...
time_point finished_at = steady_clock::now();

// Latency decomposition:
auto queue_latency = started_at - task.created_at;   // Time waiting in queue
auto exec_latency  = finished_at - started_at;        // Time executing
auto total_latency = finished_at - task.created_at;   // End-to-end
```

The monitor maintains per-task-type histograms (fixed-bucket, lock-free atomic increment)
for p50/p95/p99 computation.

### CLI Integration

Extend `brain19_cli` with a new menu option:

```
[S] Stream Status
    > Active streams: 70/70 (+ 1 orchestrator)
    > Global throughput: 1,247 tasks/sec
    > Queue depths: CRITICAL=0  HIGH=3  NORMAL=47  BACKGROUND=112
    > Stream utilization: min=0.52  avg=0.78  max=0.95  stddev=0.08
    > Latency (QUERY):    p50=0.3ms   p95=1.2ms   p99=4.7ms
    > Latency (TRAIN):    p50=12ms    p95=45ms    p99=120ms
    > Latency (INGEST):   p50=2.1ms   p95=8.3ms   p99=22ms
    > Latency (VALIDATE): p50=0.8ms   p95=3.1ms   p99=9.5ms
    > Latency (CREATIVE): p50=18ms    p95=85ms    p99=250ms
    > Alerts: none
```

---

## 8. Memory Ownership Model

### Ownership Hierarchy

```
main() owns:
│
├── LongTermMemory                    // The single source of truth (KG)
│   store_concept(), retrieve_concept(), add_relation(),
│   invalidate_concept(), get_outgoing_relations(), ...
│
├── MicroModelRegistry                // All per-concept bilinear models
│   create_model(), get_model(), has_model(), get_model_ids()
│
├── EmbeddingManager                  // Relation + context embeddings
│   get_relation_embedding(), get_context_embedding(), ...
│
├── ShortTermMemory                   // All contexts (streams get ContextId handles)
│   create_context(), activate_concept(), get_active_concepts(), decay_all()
│
├── CognitiveDynamics                 // Spreading activation, salience, focus, paths
│   spread_activation(), compute_salience_batch(), find_best_paths()
│
├── SharedLTM                         // Thread-safe wrapper, borrows &LTM
├── SharedRegistry                    // Thread-safe wrapper, borrows &Registry
├── SharedEmbeddings                  // Thread-safe wrapper, borrows &Embeddings
├── SharedSTM                         // Thread-safe wrapper, borrows &STM
│
├── TaskQueue                         // Owns all lane ring buffers
│
└── StreamOrchestrator                // Owns all ThinkStreams + StreamMonitor
    ├── ThinkStream[0]  (Query)       // Owns jthread + local SPSC queue + metrics
    ├── ThinkStream[1]  (Query)       //   borrows: SharedLTM*, SharedRegistry*,
    ├── ...                           //            SharedEmbeddings*, SharedSTM*,
    ├── ThinkStream[69] (Meta)        //            TaskQueue*
    └── StreamMonitor                 // Borrows orchestrator metrics (read-only)
```

### Lifetime Rules

1. **Construction order**: LTM → Registry → Embeddings → STM → CognitiveDynamics →
   SharedWrappers → TaskQueue → Orchestrator
2. **Destruction order**: Reverse of construction (C++ stack unwinding guarantees this)
3. **Streams borrow, never own**: All shared pointers are non-owning raw pointers.
   Streams' lifetimes are strictly contained within the orchestrator's lifetime,
   which is strictly contained within `main()`'s scope.
4. **No `shared_ptr` for shared state**: Raw pointers are intentional. The ownership
   hierarchy is static and deterministic — reference counting adds overhead and
   obscures the ownership model.
5. **STM contexts are owned by streams**: Each stream with an STM context calls
   `create_context()` on startup and `destroy_context()` on shutdown. The STM
   object persists; only the context data is cleaned up.

### Thread-Safety Guarantee

Every access from a stream to shared state goes through exactly one wrapper:

```
ThinkStream            Wrapper              Actual Subsystem
───────────           ─────────            ──────────────────
shared_ltm_->         SharedLTM::          LongTermMemory::
  retrieve_concept()    read_lock()          retrieve_concept()
                        delegate()
                        unlock()
```

No stream ever holds a direct reference to `LongTermMemory`, `MicroModelRegistry`,
`EmbeddingManager`, or `ShortTermMemory`. The wrappers are the only access path.

---

## 9. Photonic Portability Path (Q.ANT NPU)

### Design Decisions Enabling Future Photonic Hardware

The Q.ANT photonic NPU processes data using optical waveguides rather than electronic
transistors. Key characteristics: massive parallelism (100+ optical lanes), deterministic
latency, but constrained synchronization primitives.

Every design decision in this architecture has been evaluated for photonic portability:

| C++ Concept | Photonic Equivalent | Design Constraint Applied |
|-------------|---------------------|--------------------------|
| `std::jthread` | Optical processing lane | Fixed stream count at startup, no dynamic spawn after init |
| `std::stop_token` | Optical interrupt line | Boolean flag only, no complex state machine |
| `std::atomic<uint64_t>` | Photonic latch/flip-flop | Single-word atomics only (64-bit), no 128-bit CAS |
| `std::shared_mutex` | Optical read/write arbiter | Non-recursive, no `try_lock_for()` timeouts |
| MPMC ring buffer | Optical ring buffer | Power-of-2 sizes, allocated at startup, no runtime resize |
| Per-model `std::mutex` | Optical channel isolation | One model per channel, no cross-channel locking |
| Exponential backoff | Optical polling with power-down | Spin → yield → sleep maps to poll → idle → off |
| `std::latch` | Optical barrier gate | One-shot only, count-down semantics |
| `std::barrier` | Optical sync pulse | Phase-based synchronization |

### Portability Constraints Enforced Now

These constraints are applied in the C++ implementation even though they're not strictly
necessary for CPU execution. They exist solely to ensure the architecture can be ported
to photonic hardware without redesign:

1. **No `std::recursive_mutex`**: Not mappable to photonic latches. All lock acquisitions
   are non-recursive. The lock hierarchy (Section 4) guarantees no thread needs to
   re-acquire a lock it holds.

2. **No `std::condition_variable` in hot paths**: Photonic hardware uses polling, not
   interrupt-driven wake. The backoff strategy (Section 6) uses spin-wait → yield → sleep,
   which maps directly to optical poll → idle → power-down.

3. **Queue sizes fixed at startup**: Optical ring buffers are physically sized at
   fabrication time. Our ring buffers are allocated once in `TaskQueue` construction
   and never resized.

4. **All inter-stream communication through queues**: No shared-memory writes between
   streams. Streams read shared state (through wrappers with read locks) and write
   results to `TaskResult` objects. The only stream-to-stream path is through the
   `TaskQueue` or through the orchestrator's local queue dispatch.

5. **Metrics use `atomic<uint64_t>` only**: Single-word CAS operations. No `atomic<T>`
   where `sizeof(T) > 8`. No `atomic<double>` (not lock-free on all platforms). Durations
   stored as nanosecond `uint64_t` counts.

6. **No thread-local storage for shared data**: Photonic lanes don't have private caches.
   Each stream's `StreamMetrics` is in a shared array, accessed only by the owning stream
   (writes) and the monitor (reads). Cache-line alignment prevents false sharing on CPU.

### Migration Path

When Q.ANT NPU hardware becomes available:

1. Replace `std::jthread` with photonic lane allocation API
2. Replace `std::atomic` with photonic latch read/write
3. Replace `std::shared_mutex` with optical arbiter
4. Replace ring buffer atomics with optical ring buffer DMA
5. Replace `std::stop_token` with optical interrupt line
6. **No architectural changes** — the stream topology, task queue structure, wrapper
   interfaces, and ownership model remain identical

---

## 10. File Plan (Future Implementation)

```
backend/streams/
├── ARCHITECTURE.md                  ← THIS DOCUMENT
│
├── stream_types.hpp                 ← Enums: StreamId, StreamCategory,
│                                      TaskPriority, StreamTaskType,
│                                      ValidationType, MetaTaskType
│
├── stream_config.hpp                ← StreamConfig: stream counts per category,
│                                      queue sizes, backoff params, monitor interval
│
├── stream_task.hpp                  ← StreamTask, TaskPayload (variant),
│                                      TaskResult, all payload structs
│
├── lock_free_queue.hpp              ← MPMC ring buffer (atomic head/tail),
│                                      SPSC ring buffer (single-producer),
│                                      cache-line aligned entries
│
├── task_queue.hpp                   ← TaskQueue: four priority lanes,
│                                      try_push(), try_pop(), try_steal_lower(),
│                                      lane depth metrics (atomic counters)
│
├── shared_ltm.hpp                   ← SharedLTM: shared_mutex wrapper over
│                                      LongTermMemory&, read/write lock methods
│
├── shared_registry.hpp              ← SharedRegistry: shared_mutex + per-model
│                                      mutex, wraps MicroModelRegistry&
│
├── shared_embeddings.hpp            ← SharedEmbeddings: shared_mutex wrapper
│                                      over EmbeddingManager&
│
├── shared_stm.hpp                   ← SharedSTM: per-context mutex + create lock,
│                                      wraps ShortTermMemory&
│
├── think_stream.hpp                 ← ThinkStream class declaration,
│                                      StreamMetrics struct
├── think_stream.cpp                 ← ThinkStream::run() main loop,
│                                      execute() dispatch, backoff logic
│
├── stream_orchestrator.hpp          ← StreamOrchestrator + StreamMonitor
│                                      declarations
├── stream_orchestrator.cpp          ← Orchestrator startup/shutdown,
│                                      monitor loop, rebalancing logic
│
├── Makefile.streams                 ← Build rules (pattern: make -f Makefile.streams)
│
└── test_streams.cpp                 ← Unit tests for all components
```

### Estimated Scope

| Component | Estimated LOC | Complexity |
|-----------|--------------|------------|
| Enums + config | ~150 | Low (header-only) |
| StreamTask + payloads | ~200 | Low (data structures) |
| Lock-free queues (MPMC + SPSC) | ~300 | High (atomics correctness) |
| TaskQueue (multi-lane) | ~150 | Medium |
| Shared wrappers (4 files) | ~400 | Medium (lock discipline) |
| ThinkStream (hpp + cpp) | ~300 | Medium (main loop, dispatch) |
| StreamOrchestrator + Monitor | ~400 | High (lifecycle, monitoring) |
| Tests | ~600 | Medium |
| **Total** | **~2500** | |

### Dependencies

New code depends on (all existing, no modifications needed):

```
backend/ltm/long_term_memory.hpp         ← LTM interface (ConceptId, RelationId, ConceptInfo)
backend/epistemic/epistemic_metadata.hpp ← EpistemicType, EpistemicStatus, EpistemicMetadata
backend/memory/stm.hpp                   ← ShortTermMemory, ContextId
backend/memory/active_relation.hpp       ← RelationType, RelationInfo
backend/memory/activation_level.hpp      ← ActivationLevel, ActivationClass
backend/memory/brain_controller.hpp      ← BrainController (optional integration point)
backend/cognitive/cognitive_dynamics.hpp  ← CognitiveDynamics (query streams use this)
backend/micromodel/micro_model.hpp       ← MicroModel, Vec10, TrainingConfig, FLAT_SIZE=430
backend/micromodel/micro_model_registry.hpp ← MicroModelRegistry
backend/micromodel/embedding_manager.hpp ← EmbeddingManager, Vec10
backend/micromodel/relevance_map.hpp     ← RelevanceMap, OverlayMode
backend/micromodel/micro_trainer.hpp     ← MicroTrainer (training streams)
backend/ingestor/ingestion_pipeline.hpp  ← IngestionPipeline (ingestion streams)
```

---

## 11. Startup / Shutdown Sequence

### Startup

```
Phase A: Subsystem Creation (single-threaded, as today)
─────────────────────────────────────────────────────────
1.  LongTermMemory ltm;
2.  MicroModelRegistry registry;
3.  EmbeddingManager embeddings;
4.  ShortTermMemory stm;
5.  CognitiveDynamics cognitive(config);

Phase B: Wrapper Creation (single-threaded, new)
─────────────────────────────────────────────────────────
6.  SharedLTM shared_ltm(ltm);
7.  SharedRegistry shared_registry(registry);
8.  SharedEmbeddings shared_embeddings(embeddings);
9.  SharedSTM shared_stm(stm);

Phase C: Queue + Orchestrator Creation (single-threaded, new)
─────────────────────────────────────────────────────────
10. StreamConfig stream_config;  // Load from file or use defaults
11. TaskQueue task_queue(stream_config);
12. StreamOrchestrator orchestrator(
        stream_config,
        &shared_ltm, &shared_registry,
        &shared_embeddings, &shared_stm,
        &task_queue
    );

Phase D: Stream Startup (multi-threaded transition)
─────────────────────────────────────────────────────────
13. orchestrator.create_streams();
    // Creates ThinkStream objects, allocates local SPSC queues
    // Assigns StreamId + StreamCategory to each
    // NO threads spawned yet

14. orchestrator.start();
    // Creates std::latch(N+1) where N = total stream count
    // Spawns all jthreads — each thread:
    //   a) Initializes (create STM context if needed)
    //   b) Arrives at latch
    //   c) Waits for latch release
    // Orchestrator arrives at latch (the +1)
    // Latch releases → all streams begin their task loops simultaneously
    // Orchestrator enters its monitor loop

15. // System is now live.
    // External code enqueues tasks via task_queue.push(task)
    // CLI displays stream status via orchestrator.get_metrics()
```

### Shutdown

```
Phase 1: Stop Signal
─────────────────────────────────────────────────────────
1.  orchestrator.request_stop();
    // Calls request_stop() on every jthread's stop_source
    // Streams see stop_requested() == true in their next loop iteration
    // Orchestrator's own stop_token is also set (exits monitor loop)

Phase 2: Drain + Join
─────────────────────────────────────────────────────────
2.  orchestrator.join_all();
    // Each stream: finishes current task → cleans up STM context
    //   → exits run() → jthread destructor joins
    // Timeout: if a stream doesn't join within 10s, log STUCK warning
    // (jthread destructor will request_stop + join regardless — RAII safety)

Phase 3: Final Report
─────────────────────────────────────────────────────────
3.  orchestrator.print_final_metrics();
    // Per-stream summary: tasks_completed, utilization, tasks_failed
    // Per-category summary: total throughput, avg latency
    // Aggregate: total tasks processed, total uptime, peak throughput

Phase 4: Destruction (reverse order, automatic via stack unwinding)
─────────────────────────────────────────────────────────
4.  ~StreamOrchestrator      // ThinkStream destructors (jthreads already joined)
5.  ~TaskQueue               // Free ring buffers
6.  ~SharedSTM               // No-op (non-owning wrapper)
7.  ~SharedEmbeddings        // No-op (non-owning wrapper)
8.  ~SharedRegistry          // No-op (non-owning wrapper)
9.  ~SharedLTM               // No-op (non-owning wrapper)
10. ~CognitiveDynamics       // State cleanup
11. ~ShortTermMemory         // Context cleanup (all contexts already destroyed by streams)
12. ~EmbeddingManager        // Embedding cleanup
13. ~MicroModelRegistry      // Model cleanup
14. ~LongTermMemory          // Knowledge graph cleanup
```

### Graceful Degradation

If shutdown is interrupted (SIGINT/SIGTERM):

1. Signal handler sets an `atomic<bool>` flag (async-signal-safe)
2. Orchestrator checks flag in monitor loop → calls `request_stop()`
3. `jthread` destructors automatically call `request_stop()` + `join()` during stack unwinding
4. **No data loss**: LTM is the source of truth and persists independently. In-flight tasks
   may be lost, but all knowledge committed to LTM before the signal remains intact.
5. **No corruption**: `shared_mutex` + atomic operations guarantee memory consistency
   even with abrupt stop. No partially-written knowledge (LTM writes are atomic at
   the concept/relation level).

---

## Appendix A: Enum Definitions

```cpp
// Stream identification
using StreamId = uint32_t;

enum class StreamCategory : uint8_t {
    QUERY      = 0,    // User query handling
    TRAINING   = 1,    // MicroModel training
    INGESTION  = 2,    // Knowledge ingestion
    VALIDATION = 3,    // Trust/consistency validation
    CREATIVE   = 4,    // Hypothesis generation via RelevanceMap overlay
    META       = 5     // Cross-stream pattern detection
};

enum class TaskPriority : uint8_t {
    CRITICAL   = 0,    // Lowest value = highest priority
    HIGH       = 1,
    NORMAL     = 2,
    BACKGROUND = 3
};

enum class StreamTaskType : uint8_t {
    QUERY      = 0,
    TRAIN      = 1,
    INGEST     = 2,
    VALIDATE   = 3,
    CREATE     = 4,    // Creative/overlay tasks
    META       = 5
};

enum class ValidationType : uint8_t {
    TRUST_DECAY     = 0,   // Check for stale trust values
    CONTRADICTION   = 1,   // Detect RelationType::CONTRADICTS between active concepts
    CONSISTENCY     = 2,   // Verify relation/type consistency
    SUPERSEDED      = 3    // Find EpistemicStatus::SUPERSEDED candidates
};

enum class MetaTaskType : uint8_t {
    THROUGHPUT_CHECK = 0,  // Monitor overall throughput
    PATTERN_DETECT   = 1,  // Detect recurring activation patterns across streams
    HEALTH_SCAN      = 2   // Full health assessment of all streams
};
```

## Appendix B: Configuration Defaults

```cpp
struct StreamConfig {
    // Stream counts per category
    // Default total: 70 workers + 1 orchestrator = 71 threads
    // Remaining 9 cores: OS, background processes, future expansion
    uint32_t query_streams      = 6;      // Range: 4-8
    uint32_t training_streams   = 24;     // Range: 16-32
    uint32_t ingestion_streams  = 6;      // Range: 4-8
    uint32_t validation_streams = 16;     // Range: 8-16
    uint32_t creative_streams   = 16;     // Range: 8-16
    uint32_t meta_streams       = 2;      // Range: 2-4

    // Queue sizes (must be power of 2)
    uint32_t critical_lane_size   = 256;
    uint32_t high_lane_size       = 512;
    uint32_t normal_lane_size     = 1024;
    uint32_t background_lane_size = 1024;
    uint32_t local_queue_size     = 64;   // Per-stream SPSC queue

    // Backoff parameters
    uint32_t spin_iterations      = 4;    // Spins before yielding
    uint32_t yield_iterations     = 4;    // Yields before sleeping
    uint64_t max_sleep_us         = 1000; // 1ms sleep cap

    // Monitor parameters
    double   monitor_hz           = 10.0; // Sample rate (Hz)
    uint64_t stuck_timeout_ms     = 5000; // Heartbeat timeout before STUCK alert
    double   queue_pressure_threshold = 0.8;  // Lane saturation alert threshold
    double   imbalance_threshold  = 0.3;  // Utilization stddev alert threshold

    // Creative session limits
    uint32_t creative_session_timeout_s = 30;  // Max seconds per creative task
    uint32_t max_overlay_depth    = 3;         // Max chained RelevanceMap overlays

    // Shutdown
    uint64_t join_timeout_ms      = 10000;     // Max wait for stream join before warning
};
```

## Appendix C: Concurrency Correctness Arguments

### Deadlock Freedom

The lock hierarchy (Section 4) defines a total order on all locks:

```
SharedLTM.rwlock_ < SharedRegistry.registry_lock_ < SharedEmbeddings.rwlock_
    < SharedRegistry.model_locks_[i] < SharedSTM.context_locks_[j]
```

All code paths acquire locks in ascending order only. Since the order is total and
acyclic, deadlock is impossible by the standard lock-ordering theorem.

**Proof sketch**: Assume deadlock exists. Then there exists a cycle T1→T2→...→T1 in
the wait-for graph, where each edge Ti→Tj means "thread Ti holds lock Li and waits
for lock Lj held by Tj." Following the cycle: L1 < L2 < ... < L1, which contradicts
the total order. Therefore no deadlock cycle can exist. QED.

### No Dangling References

All shared wrappers hold non-owning references to objects owned by `main()`. The
ownership hierarchy guarantees:

1. Wrappers are destroyed before the objects they reference (reverse construction order)
2. Streams are destroyed before wrappers (orchestrator is destroyed first, triggering
   `jthread` joins, ensuring all streams have exited before wrapper destruction)
3. No stream can outlive the orchestrator (`jthread` destructor calls `join()`)

Therefore, no stream can access a wrapper after the wrapper's referent is destroyed.

### Progress Guarantee

- **Reads are obstruction-free**: `shared_mutex` read lock never blocks on other readers.
  `atomic` loads are wait-free. When no writer holds the lock, readers make progress
  without any synchronization overhead.
- **Writes are deadlock-free**: By the lock hierarchy argument above, no write operation
  can deadlock. Writers may wait for readers to drain (standard `shared_mutex` behavior)
  but will eventually acquire the lock.
- **Starvation resistance**: `shared_mutex` implementations typically give pending writers
  priority after a bounded number of readers. Writers (ingestion only) are infrequent
  relative to readers (queries, training, validation), so writer starvation is not a
  practical concern.

### Data Race Freedom

Every shared mutable variable is protected by exactly one synchronization mechanism:

| Variable | Protection | Accessed By |
|----------|-----------|-------------|
| LTM internal state | `SharedLTM.rwlock_` | All streams (read), Ingestion (write) |
| Registry internal state | `SharedRegistry.registry_lock_` | All streams |
| Individual `MicroModel` | `SharedRegistry.model_locks_[i]` | One training stream at a time |
| Embeddings internal state | `SharedEmbeddings.rwlock_` | All streams |
| STM context data | `SharedSTM.context_locks_[i]` | Owning stream only |
| `StreamMetrics` fields | `std::atomic<uint64_t>` | Owning stream (write), Monitor (read) |
| `TaskQueue` lane indices | `std::atomic<uint64_t>` | All streams + orchestrator |
| `TaskResult::completed` | `std::atomic<bool>` | Executing stream (write), Caller (read) |

No variable is accessed from multiple threads without synchronization. The wrapper
layer enforces this by being the sole access path to shared subsystems.
