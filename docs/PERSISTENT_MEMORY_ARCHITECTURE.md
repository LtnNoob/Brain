# Brain19 Persistent Memory Architecture — 3-Layer Design

**Date:** 2025-02-10  
**Version:** 2.0  
**Status:** Architecture Complete — Ready for Implementation  
**Authors:** Brain19 Architecture Team

---

## Executive Summary

Dieses Dokument beschreibt die vollständige Persistence-Architektur für Brain19. Das Design besteht aus drei unabhängigen, aber komplementären Layern:

| Layer | Was | Wann | Datenverlust | Recovery-Zeit |
|-------|-----|------|-------------|---------------|
| **Layer 1: LTM auf mmap** | Langzeitgedächtnis persistent auf SSD | Immer aktiv | 0 (WAL-geschützt) | <1s warm, <30s cold |
| **Layer 2: STM-Snapshots** | Periodische STM-State-Sicherung | Alle 30-60s | Max 30-60s | <2s |
| **Layer 3: Full Brain-State Checkpoint** | Kompletter konsistenter Systemzustand | Manuell/zeitgesteuert | 0 (Zeitpunkt des Checkpoints) | <10s |

**Ziel:** Brain19 überlebt Crashes, Reboots und geplante Restarts ohne signifikanten Zustandsverlust. Bei einem Full Checkpoint kann das System exakt dort weitermachen, wo es aufgehört hat.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Layer 1: LTM auf mmap](#2-layer-1-ltm-auf-mmap)
3. [Layer 2: STM-Snapshots](#3-layer-2-stm-snapshots)
4. [Layer 3: Full Brain-State Checkpoint](#4-layer-3-full-brain-state-checkpoint)
5. [Cross-Layer Integration](#5-cross-layer-integration)
6. [File Layout](#6-file-layout)
7. [API Design](#7-api-design)
8. [Implementation Phases](#8-implementation-phases)
9. [Performance Targets](#9-performance-targets)

---

## 1. Current State Analysis

### 1.1 Existing Data Structures (~16k LOC)

**LongTermMemory** (`ltm/long_term_memory.hpp`):
```cpp
// Heap-only, dies on process exit
std::unordered_map<ConceptId, ConceptInfo> concepts_;         // ConceptId = uint64_t
std::unordered_map<RelationId, RelationInfo> relations_;      // RelationId = uint64_t
std::unordered_map<ConceptId, std::vector<RelationId>> outgoing_relations_;
std::unordered_map<ConceptId, std::vector<RelationId>> incoming_relations_;
ConceptId next_concept_id_;
RelationId next_relation_id_;
```

**ShortTermMemory** (`memory/stm.hpp`):
```cpp
struct Context {
    std::unordered_map<ConceptId, STMEntry> concepts;          // Activation states
    std::unordered_map<uint64_t, ActiveRelation> relations;    // Hashed relation keys
};
std::unordered_map<ContextId, Context> contexts_;
ContextId next_context_id_;
// + decay rates (5 doubles)
```

**STMEntry** (`memory/stm_entry.hpp`):
```cpp
struct STMEntry {
    ConceptId concept_id;
    double activation;                    // [0.0, 1.0]
    ActivationClass classification;       // CORE_KNOWLEDGE | CONTEXTUAL
    std::chrono::steady_clock::time_point last_used;
};
```

**CognitiveDynamics** (`cognitive/cognitive_dynamics.hpp`):
```cpp
std::unordered_map<ContextId, std::vector<FocusEntry>> focus_sets_;
uint64_t current_tick_;
Stats stats_;
CognitiveDynamicsConfig config_;
```

**ConceptInfo** (`ltm/long_term_memory.hpp`):
```cpp
struct ConceptInfo {
    ConceptId id;
    std::string label;
    std::string definition;
    EpistemicMetadata epistemic;  // Deleted default ctor + deleted assignment
};
```

**EpistemicMetadata** (`epistemic/epistemic_metadata.hpp`):
```cpp
struct EpistemicMetadata {
    EpistemicType type;       // FACT, DEFINITION, THEORY, HYPOTHESIS, INFERENCE, SPECULATION
    EpistemicStatus status;   // ACTIVE, CONTEXTUAL, SUPERSEDED, INVALIDATED
    double trust;             // [0.0, 1.0]
    // Deleted: default ctor, assignment operator
    // Allowed: copy ctor, move ctor
};
```

**RelationInfo** (`ltm/relation.hpp`):
```cpp
struct RelationInfo {
    RelationId id;
    ConceptId source, target;
    RelationType type;    // IS_A, HAS_PROPERTY, CAUSES, ENABLES, PART_OF, SIMILAR_TO, CONTRADICTS, SUPPORTS, TEMPORAL_BEFORE, CUSTOM
    double weight;        // [0.0, 1.0]
};
```

**MicroModel Persistence** (`micromodel/persistence.hpp`):
- Bereits existierendes Binary-Format mit Header ("BM19"), Checksums, versioned format
- **Pattern to follow** for Layer 2 & 3

### 1.2 Key Constraints

| Constraint | Impact |
|-----------|--------|
| `EpistemicMetadata` hat deleted assignment operator | Placement-new Pattern bei Deserialisierung |
| `ConceptInfo` hat deleted default constructor | Kann nicht trivial deserialisiert werden |
| Single-threaded Design (kein Threading) | Layer 1 Thread-Safety kann inkrementell kommen |
| Keine File-I/O in LTM | Alles stirbt bei Process-Exit |
| `BrainController` besitzt nur STM, nicht LTM | LTM-Ownership muss ergänzt werden |

---

## 2. Layer 1: LTM auf mmap

### 2.1 Übersicht

Layer 1 macht das Langzeitgedächtnis persistent via Memory-Mapped Files. Die bestehende `LongTermMemory`-Klasse wird zur Facade — die API bleibt identisch, aber die Daten leben auf der SSD statt auf dem Heap.

### 2.2 Memory-Tiering

```
┌─────────────────────────────────────────────────┐
│ HOT TIER — mmap'd RAM (mlock'd)                │
│ ~20-40GB, most-accessed concepts & relations     │
│ Access: ~100ns (memory speed)                    │
│ Policy: Access-counter based promotion           │
├─────────────────────────────────────────────────┤
│ WARM TIER — mmap'd RAM (pageable)               │
│ ~40-80GB, moderately accessed data               │
│ Access: ~100ns-1μs (may page fault)              │
│ Policy: LRU demotion to cold                     │
├─────────────────────────────────────────────────┤
│ COLD TIER — mmap'd SSD-backed files             │
│ ~1TB, infrequently accessed archive              │
│ Access: ~10-100μs (SSD page fault)               │
│ Policy: On-demand page-in, background eviction   │
└─────────────────────────────────────────────────┘
```

### 2.3 Persistent Record Formats

```cpp
// Fixed-size, mmap-able, cache-line aligned
struct PersistentConceptRecord {
    uint64_t concept_id;                // 8B
    uint32_t label_offset;              // 4B — into string pool
    uint32_t label_length;              // 4B
    uint32_t definition_offset;         // 4B
    uint32_t definition_length;         // 4B
    uint8_t  epistemic_type;            // 1B (EpistemicType enum)
    uint8_t  epistemic_status;          // 1B (EpistemicStatus enum)
    uint8_t  _pad1[6];                  // 6B alignment
    double   trust;                     // 8B
    uint64_t access_count;              // 8B — for hot/cold tiering
    uint64_t last_access_epoch_us;      // 8B
    uint64_t created_epoch_us;          // 8B
    uint8_t  flags;                     // 1B (deleted, locked, etc.)
    uint8_t  _reserved[7];             // 7B
    // Total: 72 bytes (pad to 128 for 2x cache line if needed)
};
static_assert(sizeof(PersistentConceptRecord) == 72);

struct PersistentRelationRecord {
    uint64_t relation_id;               // 8B
    uint64_t source;                    // 8B
    uint64_t target;                    // 8B
    uint8_t  type;                      // 1B (RelationType)
    uint8_t  _pad[7];                   // 7B
    double   weight;                    // 8B
    uint8_t  flags;                     // 1B
    uint8_t  _reserved[7];             // 7B
    // Total: 48 bytes
};
static_assert(sizeof(PersistentRelationRecord) == 48);
```

### 2.4 Core Components

**`PersistentStore<T>`** — Template mmap-Manager:
```cpp
template<typename RecordT>
class PersistentStore {
public:
    PersistentStore(const std::string& filepath, size_t initial_capacity);
    bool open();                              // mmap existing or create
    void close();                             // msync + munmap
    const RecordT* get(uint64_t id) const;    // Direct pointer into mmap
    RecordT* get_mutable(uint64_t id);        // Mutable pointer
    uint64_t insert(const RecordT& record);   // Append
    bool remove(uint64_t id);                 // Soft-delete (flag)
    void sync();                              // msync(MS_ASYNC)
    bool grow(size_t new_capacity);           // mremap to expand
};
```

**`StringPool`** — Append-only String Storage:
```cpp
struct StringPoolHeader {
    char magic[4];          // "SP19"
    uint32_t version;
    uint64_t total_bytes;
    uint64_t used_bytes;
};
// After header: raw UTF-8 bytes, referenced by offset+length
```

**`TierManager`** — Hot/Warm/Cold Promotion/Demotion:
```cpp
class TierManager {
    void record_access(ConceptId id);
    void rebalance();                         // Periodic: promote/demote
    void promote_to_hot(ConceptId id);        // mlock pages
    void demote_from_hot(ConceptId id);       // munlock
};
```

**`WriteAheadLog`** — Crash Safety:
```cpp
class WriteAheadLog {
    uint64_t append(OpType op, uint64_t id, const void* data, size_t len);
    void checkpoint();     // Flush to main store, truncate
    void recover();        // Replay after crash
};
```

### 2.5 LongTermMemory Facade

Die bestehende API bleibt **100% identisch**. Internals wechseln von Heap zu Persistent:

```cpp
class LongTermMemory {
public:
    LongTermMemory();                                    // Legacy: heap mode
    LongTermMemory(const PersistentConfig& config);      // NEW: persistent mode
    
    // === UNCHANGED PUBLIC API ===
    ConceptId store_concept(const std::string& label, const std::string& def, EpistemicMetadata ep);
    std::optional<ConceptInfo> retrieve_concept(ConceptId id) const;
    bool exists(ConceptId id) const;
    bool update_epistemic_metadata(ConceptId id, EpistemicMetadata new_metadata);
    bool invalidate_concept(ConceptId id, double invalidation_trust = 0.05);
    // ... all relation methods unchanged ...

private:
    bool persistent_mode_;
    // Persistent backend (when persistent_mode_ = true):
    std::unique_ptr<PersistentConceptStore> concept_store_;
    std::unique_ptr<PersistentRelationStore> relation_store_;
    std::unique_ptr<StringPool> string_pool_;
    std::unique_ptr<TierManager> tier_manager_;
    std::unique_ptr<WriteAheadLog> wal_;
    // Legacy backend (when persistent_mode_ = false):
    std::unordered_map<ConceptId, ConceptInfo> concepts_;
    // ...
};
```

### 2.6 EpistemicMetadata Serialization

Problem: `EpistemicMetadata` hat deleted default ctor und deleted assignment. Lösung:

```cpp
// Serialize: Direct field extraction (trivial)
void serialize_epistemic(const EpistemicMetadata& em, PersistentConceptRecord& rec) {
    rec.epistemic_type   = static_cast<uint8_t>(em.type);
    rec.epistemic_status = static_cast<uint8_t>(em.status);
    rec.trust            = em.trust;
}

// Deserialize: Reconstruct via explicit constructor
EpistemicMetadata deserialize_epistemic(const PersistentConceptRecord& rec) {
    return EpistemicMetadata(
        static_cast<EpistemicType>(rec.epistemic_type),
        static_cast<EpistemicStatus>(rec.epistemic_status),
        rec.trust
    );
}

// ConceptInfo reconstruction (no default ctor):
ConceptInfo reconstruct_concept(const PersistentConceptRecord& rec, const StringPool& sp) {
    return ConceptInfo(
        rec.concept_id,
        sp.get_string(rec.label_offset, rec.label_length),
        sp.get_string(rec.definition_offset, rec.definition_length),
        deserialize_epistemic(rec)
    );
}
```

---

## 3. Layer 2: STM-Snapshots

### 3.1 Übersicht

Layer 2 sichert den aktiven STM-Zustand periodisch (alle 30-60 Sekunden) als Binary Snapshot auf Disk. Bei einem Crash wird der letzte valide Snapshot geladen — maximaler Datenverlust: 30-60 Sekunden an Aktivierungsänderungen.

### 3.2 Was wird gesichert?

Basierend auf den tatsächlichen Datenstrukturen in `stm.hpp`, `cognitive_dynamics.hpp`:

| Komponente | Datenstruktur | Serialisierungs-Strategie |
|------------|--------------|--------------------------|
| STM Contexts | `map<ContextId, Context>` | Iterieren, Entry-weise serialisieren |
| STM Entries | `map<ConceptId, STMEntry>` | 4 Felder: concept_id, activation, classification, last_used |
| Active Relations | `map<uint64_t, ActiveRelation>` | Key + ActiveRelation Felder |
| STM Config | 5 decay-rate doubles | Direkt als Block |
| Focus Sets | `map<ContextId, vector<FocusEntry>>` | Per-Context Focus-Vektor |
| Cognitive Tick | `uint64_t current_tick_` | Single value |
| Context Counter | `ContextId next_context_id_` | Single value |

### 3.3 Binary Snapshot Format

```
+------------------------------------------------------+
| HEADER (64 bytes)                                     |
|   magic:           "B19S" (4B)                        |
|   version:         uint32 (4B) = 1                    |
|   timestamp_us:    uint64 (8B) -- snapshot time       |
|   stm_context_count: uint64 (8B)                      |
|   focus_context_count: uint64 (8B)                    |
|   total_entries:   uint64 (8B) -- total STM entries   |
|   total_relations: uint64 (8B) -- total active rels   |
|   flags:           uint64 (8B) -- reserved            |
|   header_checksum: uint64 (8B)                        |
+------------------------------------------------------+
| STM CONFIG BLOCK (48 bytes)                           |
|   next_context_id:           uint64 (8B)              |
|   core_decay_rate:           double (8B)              |
|   contextual_decay_rate:     double (8B)              |
|   relation_decay_rate:       double (8B)              |
|   relation_inactive_thresh:  double (8B)              |
|   relation_removal_thresh:   double (8B)              |
+------------------------------------------------------+
| COGNITIVE STATE BLOCK (16 bytes)                      |
|   current_tick:              uint64 (8B)              |
|   _reserved:                 uint64 (8B)              |
+------------------------------------------------------+
| Per STM Context (repeated x stm_context_count):       |
|   context_id:    uint64 (8B)                          |
|   entry_count:   uint64 (8B)                          |
|   relation_count: uint64 (8B)                         |
|                                                       |
|   Per Entry (repeated x entry_count):                 |
|     concept_id:      uint64 (8B)                      |
|     activation:      double (8B)                      |
|     classification:  uint8  (1B)                      |
|     _pad:            7 bytes                          |
|     last_used_ns:    uint64 (8B) -- since epoch       |
|     Total: 32 bytes per entry                         |
|                                                       |
|   Per Active Relation (repeated x relation_count):    |
|     hash_key:        uint64 (8B)                      |
|     source:          uint64 (8B)                      |
|     target:          uint64 (8B)                      |
|     type:            uint8  (1B)                      |
|     _pad:            7 bytes                          |
|     activation:      double (8B)                      |
|     Total: 40 bytes per relation                      |
+------------------------------------------------------+
| Per Focus Context (repeated x focus_context_count):   |
|   context_id:    uint64 (8B)                          |
|   focus_count:   uint64 (8B)                          |
|                                                       |
|   Per FocusEntry (repeated x focus_count):            |
|     concept_id:      uint64 (8B)                      |
|     focus_score:     double (8B)                      |
|     last_access_tick: uint64 (8B)                     |
|     Total: 24 bytes per focus entry                   |
+------------------------------------------------------+
| FOOTER (8 bytes)                                      |
|   body_checksum: uint64 (8B) -- XOR of all 8B blocks |
+------------------------------------------------------+
```

### 3.4 Snapshot-Klasse

```cpp
class STMSnapshotManager {
public:
    struct Config {
        std::string snapshot_dir = "/data/brain19/snapshots/stm";
        uint32_t interval_seconds = 30;       // Snapshot-Intervall
        uint32_t max_snapshots = 5;           // Rotation: behalte die letzten N
        bool verify_on_load = true;           // Checksum-Verifikation beim Laden
    };

    STMSnapshotManager(Config config);
    
    // === Snapshot erstellen ===
    std::optional<std::string> create_snapshot(
        const ShortTermMemory& stm,
        const CognitiveDynamics& cognitive
    );
    
    // === Snapshot laden ===
    bool load_latest_snapshot(
        ShortTermMemory& stm,
        CognitiveDynamics& cognitive
    );
    
    bool load_snapshot(
        const std::string& filepath,
        ShortTermMemory& stm,
        CognitiveDynamics& cognitive
    );
    
    // === Validierung ===
    bool validate_snapshot(const std::string& filepath) const;
    
    // === Maintenance ===
    void rotate_snapshots();
    std::vector<std::string> list_snapshots() const;
    
    // === Periodischer Background-Timer ===
    void start_periodic(ShortTermMemory& stm, CognitiveDynamics& cognitive);
    void stop_periodic();
    
private:
    Config config_;
    std::atomic<bool> running_{false};
    std::thread snapshot_thread_;
    std::string generate_filename() const;
    uint64_t compute_checksum(const uint8_t* data, size_t len) const;
};
```

### 3.5 STM Zugriff für Serialisierung

Problem: STM's `Context` struct ist private. Lösung — **friend + Accessor statt Breaking Change**:

```cpp
// In stm.hpp — minimaler Eingriff:
class ShortTermMemory {
    // ... bestehender Code ...
    
    friend class STMSnapshotManager;
    
    // Alternative: Public accessor (weniger invasiv)
public:
    struct SnapshotData {
        struct ContextSnapshot {
            ContextId id;
            std::vector<STMEntry> entries;
            std::vector<std::pair<uint64_t, ActiveRelation>> relations;
        };
        std::vector<ContextSnapshot> contexts;
        ContextId next_context_id;
        double core_decay_rate;
        double contextual_decay_rate;
        double relation_decay_rate;
        double relation_inactive_threshold;
        double relation_removal_threshold;
    };
    
    SnapshotData export_snapshot() const;
    void import_snapshot(const SnapshotData& data);
};
```

### 3.6 Recovery-Ablauf

```
STM CRASH RECOVERY:
1. STMSnapshotManager::load_latest_snapshot()
2. Suche neuesten *.b19s in snapshot_dir (sortiert nach Timestamp im Filename)
3. Validiere Header-Checksum
4. Lese STM Config Block -> setze Decay-Rates
5. Lese Cognitive State Block -> setze Tick-Counter
6. Pro Context:
   a. create_context() mit gespeicherter ContextId
   b. Pro STMEntry: activate_concept() mit gespeicherten Werten
   c. Pro ActiveRelation: activate_relation() mit gespeicherten Werten
7. Pro Focus Context:
   a. init_focus(context_id)
   b. Pro FocusEntry: focus_on() mit gespeicherten Scores
8. Validiere Body-Checksum
9. Log: "STM recovered from snapshot at <timestamp>, age: <seconds>s"
```

### 3.7 Snapshot-Timing & Konsistenz

**Atomizität:** Da Brain19 aktuell single-threaded ist, kann der Snapshot synchron erstellt werden — es gibt keine Concurrent-Writes. Der Snapshot ist automatisch konsistent.

**Wenn Multi-Threading kommt:** Snapshot muss unter einer globalen Read-Lock erstellt werden (alle Writer pausieren für ~1-5ms). Alternative: Copy-on-Write Snapshot via `fork()`:

```cpp
// Future: Atomic snapshot via fork() — zero-pause for writers
std::optional<std::string> create_snapshot_atomic(...) {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: hat Copy-on-Write Kopie des gesamten Adressraums
        write_snapshot_to_disk(stm, cognitive);
        _exit(0);
    }
    // Parent: weiter arbeiten, waitpid() async
    return snapshot_path;
}
```

---

## 4. Layer 3: Full Brain-State Checkpoint

### 4.1 Übersicht

Layer 3 erstellt einen **kompletten, konsistenten Snapshot des gesamten Brain19-Zustands**. Alle Subsysteme werden atomar gesichert. Brain19 kann aus einem Checkpoint exakt dort weitermachen, wo es aufgehört hat — inklusive aller Aktivierungen, Focus-Sets, LTM-Metadata, und Stream-States.

### 4.2 Was wird gesichert?

| Subsystem | Klasse | State |
|-----------|--------|-------|
| LTM Concepts | `LongTermMemory` | Alle Konzepte + Epistemic-Metadata |
| LTM Relations | `LongTermMemory` | Alle Relations + Indexes |
| LTM Counters | `LongTermMemory` | `next_concept_id_`, `next_relation_id_` |
| STM Contexts | `ShortTermMemory` | Alle Contexts mit Entries + Relations |
| STM Config | `ShortTermMemory` | Decay-Rates, Thresholds |
| Cognitive Focus | `CognitiveDynamics` | Focus-Sets per Context |
| Cognitive Tick | `CognitiveDynamics` | `current_tick_`, Stats |
| Cognitive Config | `CognitiveDynamics` | `CognitiveDynamicsConfig` |
| Curiosity State | `CuriosityEngine` | Exploration-Queues, Curiosity-Scores |
| MicroModels | `MicroModelRegistry` | Alle trainierten MicroModels |
| Embeddings | `EmbeddingManager` | Alle Embeddings + Context-Embeddings |
| Brain Controller | `BrainController` | ThinkingStates, initialized-Flag |

### 4.3 Checkpoint Format

Ein Checkpoint ist ein **Directory** mit atomarem Rename:

```
/data/brain19/checkpoints/
+-- checkpoint_20250210_120000/           # Completed checkpoint
|   +-- MANIFEST.json                     # Metadata + component list
|   +-- ltm_concepts.b19c                 # LTM Concepts binary
|   +-- ltm_relations.b19c               # LTM Relations binary
|   +-- ltm_strings.b19c                 # String pool
|   +-- stm_state.b19s                   # STM Snapshot (Layer 2 Format!)
|   +-- cognitive_state.b19c             # CognitiveDynamics full state
|   +-- curiosity_state.b19c             # CuriosityEngine state
|   +-- micromodels.b19m                 # MicroModel persistence (existing format!)
|   +-- brain_controller.b19c           # BrainController state
|   +-- CHECKSUM                         # SHA-256 of all files
+-- checkpoint_latest -> checkpoint_20250210_120000/  # Symlink
+-- _checkpoint_in_progress/             # Temp dir during write (atomic rename)
```

### 4.4 MANIFEST.json

```json
{
    "version": 1,
    "brain19_version": "0.1.0",
    "timestamp_utc": "2025-02-10T12:00:00.000Z",
    "timestamp_us": 1707566400000000,
    "components": [
        {"name": "ltm_concepts",    "file": "ltm_concepts.b19c",    "records": 150000, "checksum": "a1b2c3..."},
        {"name": "ltm_relations",   "file": "ltm_relations.b19c",   "records": 500000, "checksum": "d4e5f6..."},
        {"name": "ltm_strings",     "file": "ltm_strings.b19c",     "bytes": 52428800, "checksum": "..."},
        {"name": "stm_state",       "file": "stm_state.b19s",       "contexts": 3,     "checksum": "..."},
        {"name": "cognitive_state",  "file": "cognitive_state.b19c", "checksum": "..."},
        {"name": "curiosity_state",  "file": "curiosity_state.b19c","checksum": "..."},
        {"name": "micromodels",      "file": "micromodels.b19m",    "models": 42,      "checksum": "..."},
        {"name": "brain_controller", "file": "brain_controller.b19c","checksum": "..."}
    ],
    "total_size_bytes": 1073741824,
    "consistent": true,
    "trigger": "manual"
}
```

### 4.5 Atomic Checkpoint — Konsistenz-Garantie

**Problem:** Während des Schreibens darf kein inkonsistenter Zustand entstehen. Wenn der Prozess während des Checkpoints crasht, darf kein halber Checkpoint als "fertig" gelten.

**Lösung: Write-to-Temp + Atomic Rename:**

```cpp
class BrainCheckpointManager {
public:
    struct Config {
        std::string checkpoint_dir = "/data/brain19/checkpoints";
        uint32_t max_checkpoints = 3;         // Rotation
        bool compress = false;                 // Optional: zstd Kompression
        bool verify_after_write = true;        // Sofort nach Schreiben verifizieren
    };
    
    BrainCheckpointManager(Config config);
    
    // === Checkpoint erstellen (atomar) ===
    // Schritt 1: Schreibe alles in _checkpoint_in_progress/
    // Schritt 2: fsync alle Files
    // Schritt 3: rename() -> checkpoint_<timestamp>/  (ATOMIC auf gleicher FS)
    // Schritt 4: Update symlink checkpoint_latest
    std::optional<std::string> create_checkpoint(
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        const CognitiveDynamics& cognitive,
        const CuriosityEngine& curiosity,
        const MicroModelRegistry& micromodels,
        const EmbeddingManager& embeddings,
        const BrainController& brain
    );
    
    // === Checkpoint laden ===
    bool load_checkpoint(
        const std::string& checkpoint_path,
        LongTermMemory& ltm,
        ShortTermMemory& stm,
        CognitiveDynamics& cognitive,
        CuriosityEngine& curiosity,
        MicroModelRegistry& micromodels,
        EmbeddingManager& embeddings,
        BrainController& brain
    );
    
    bool load_latest_checkpoint(/* same params */);
    
    // === Validierung ===
    bool validate_checkpoint(const std::string& checkpoint_path) const;
    
    // === Maintenance ===
    void rotate_checkpoints();
    std::vector<std::string> list_checkpoints() const;
    
private:
    Config config_;
    bool finalize_checkpoint(const std::string& temp_dir, const std::string& final_dir);
    bool write_ltm_concepts(const std::string& path, const LongTermMemory& ltm);
    bool write_ltm_relations(const std::string& path, const LongTermMemory& ltm);
    bool write_stm_state(const std::string& path, const ShortTermMemory& stm, const CognitiveDynamics& cd);
    bool write_cognitive_state(const std::string& path, const CognitiveDynamics& cd);
    bool write_curiosity_state(const std::string& path, const CuriosityEngine& ce);
    bool write_micromodels(const std::string& path, const MicroModelRegistry& mr, const EmbeddingManager& em);
    bool write_brain_controller(const std::string& path, const BrainController& bc);
    bool write_manifest(const std::string& path, const std::vector<ComponentInfo>& components);
};
```

### 4.6 LTM-Checkpoint Format (b19c)

Für Layer 3 wird LTM **unabhängig** von Layer 1's mmap serialisiert. Das erlaubt Checkpoints auch im Heap-Modus und dient als Backup des mmap-Stores.

```
LTM Concepts (ltm_concepts.b19c):
+---------------------------------------------+
| HEADER (48 bytes)                            |
|   magic: "B19C" (4B)                         |
|   version: uint32 (4B)                       |
|   concept_count: uint64 (8B)                 |
|   next_concept_id: uint64 (8B)               |
|   string_section_offset: uint64 (8B)         |
|   flags: uint64 (8B)                         |
|   header_checksum: uint64 (8B)               |
+---------------------------------------------+
| CONCEPT RECORDS (variable)                   |
|   Per Concept:                               |
|     concept_id: uint64 (8B)                  |
|     epistemic_type: uint8 (1B)               |
|     epistemic_status: uint8 (1B)             |
|     _pad: 6B                                 |
|     trust: double (8B)                       |
|     label_len: uint32 (4B)                   |
|     definition_len: uint32 (4B)              |
|     label: label_len bytes                   |
|     definition: definition_len bytes         |
|     Padding to 8-byte alignment              |
+---------------------------------------------+
| FOOTER                                       |
|   body_checksum: uint64 (8B)                 |
+---------------------------------------------+
```

### 4.7 Trigger-Mechanismen

```cpp
// Manual trigger via CLI oder API
brain19_cli checkpoint create
brain19_cli checkpoint list
brain19_cli checkpoint load <path>
brain19_cli checkpoint validate <path>
```

```toml
# Zeitgesteuert via Config
[checkpoint]
auto_interval_minutes = 60        # Alle 60 Minuten
auto_on_shutdown = true           # Bei graceful shutdown
auto_on_ltm_change_count = 10000  # Nach N LTM-Aenderungen
```

### 4.8 Restore-Ablauf

```
FULL BRAIN-STATE RESTORE:
1. BrainCheckpointManager::load_latest_checkpoint()
2. Validiere MANIFEST.json + CHECKSUM
3. Pro Komponente:
   a. Validiere einzelne File-Checksums
   b. Deserialisiere in Ziel-Objekt
4. Reihenfolge (Abhaengigkeiten beachten!):
   (1) LTM Concepts + Relations + Strings  (Basis fuer alles)
   (2) MicroModels + Embeddings            (unabhaengig)
   (3) STM State                           (referenziert ConceptIds aus LTM)
   (4) Cognitive State + Focus             (referenziert ConceptIds + ContextIds)
   (5) Curiosity State                     (referenziert ConceptIds)
   (6) BrainController State               (referenziert ContextIds)
5. Konsistenz-Check: Alle ConceptIds in STM existieren in LTM
6. Log: "Brain state restored from checkpoint at <timestamp>"
```

---

## 5. Cross-Layer Integration

### 5.1 Zusammenspiel der 3 Layer

```
                    +-------------------------+
                    |  Layer 3: Full Checkpoint|
                    |  (alles, atomar, selten) |
                    |  Trigger: manuell/60min  |
                    +------------+------------+
                                 | enthaelt
                    +------------v------------+
                    |  Layer 2: STM-Snapshots  |
                    |  (STM+Focus, periodisch) |
                    |  Interval: 30-60s        |
                    +------------+------------+
                                 | ergaenzt
                    +------------v------------+
                    |  Layer 1: LTM auf mmap   |
                    |  (Wissen, permanent)      |
                    |  Always-on, WAL-protected |
                    +--------------------------+
```

### 5.2 Recovery-Matrix

| Szenario | Recovery-Strategie |
|----------|-------------------|
| Normaler Restart (graceful shutdown) | Layer 1 mmap remaps instantly, Layer 2 latest snapshot, kein Verlust |
| Crash (SIGKILL, OOM) | Layer 1 WAL replay (<10s), Layer 2 letzter Snapshot (max 30-60s alt) |
| Korrupte mmap-Files | Layer 3 Checkpoint restore, dann Layer 2 Snapshot on top |
| Hardware-Fehler (SSD-Corruption) | Layer 3 von Backup-SSD, Full Restore |
| Geplante Migration | Layer 3 Checkpoint -> Transfer -> Restore auf neuem Host |

### 5.3 Startup-Sequenz mit allen 3 Layern

```
BRAIN19 STARTUP:
1. Parse Config
2. === Layer 1: LTM ===
   a. Check WAL for incomplete operations -> recover()
   b. mmap concept store, relation store, string pool, indexes
   c. Verify checksums
   d. Load tier manifest -> mlock hot-tier pages
   
3. === Layer 2: STM Recovery ===
   a. Suche neuesten validen STM-Snapshot
   b. Wenn gefunden:
      - Load STM state (contexts, activations, decay rates)
      - Load Focus state (focus sets, tick counter)
      - Log: "STM recovered from snapshot (age: Xs)"
   c. Wenn nicht gefunden:
      - Start mit leerem STM (clean boot)
      
4. === Layer 3: Checkpoint (nur wenn Layer 1+2 fehlschlagen) ===
   a. Wenn Layer 1 mmap korrupt ODER Layer 2 kein Snapshot:
      - Suche neuesten validen Checkpoint
      - Full Restore aus Checkpoint
      - Log: "Full restore from checkpoint at <timestamp>"
      
5. Initialize remaining subsystems (CognitiveDynamics, CuriosityEngine, etc.)
6. Validate cross-references (STM ConceptIds in LTM)
7. Signal ready
```

### 5.4 Shutdown-Sequenz

```
BRAIN19 GRACEFUL SHUTDOWN:
1. Stop accepting new work
2. === Layer 2: Final STM Snapshot ===
   - create_snapshot() -- captures aktuellen STM-Zustand
3. === Layer 3: Shutdown Checkpoint (wenn konfiguriert) ===
   - create_checkpoint() -- Full Brain State
4. === Layer 1: LTM Sync ===
   - WAL checkpoint() -- flush pending writes
   - msync(MS_SYNC) all mmap'd regions
   - Update headers, verify checksums
   - munmap all regions, close FDs
5. Log: "Brain19 shutdown complete, all state persisted"
```

---

## 6. File Layout

```
/data/brain19/                              # Root (SSD mount point)
+-- config.toml                             # Master configuration
+-- persistent/                             # Layer 1: LTM mmap files
|   +-- concepts.b19                        # mmap'd concept store
|   +-- relations.b19                       # mmap'd relation store
|   +-- relations_idx_out.b19               # Outgoing relation index
|   +-- relations_idx_in.b19                # Incoming relation index
|   +-- strings.b19                         # String pool
|   +-- concept_index.b19                   # ConceptId -> offset mapping
|   +-- metadata.b19                        # Header, version, counters
+-- wal/                                    # Layer 1: Write-Ahead Log
|   +-- wal_000001.log
|   +-- wal_current.log
+-- tiers/                                  # Layer 1: Hot/Cold metadata
|   +-- hot_manifest.b19
|   +-- access_counters.b19
+-- snapshots/                              # Layer 2: STM Snapshots
|   +-- stm/
|       +-- stm_snapshot_1707566400000000.b19s
|       +-- stm_snapshot_1707566430000000.b19s
|       +-- latest -> stm_snapshot_1707566430000000.b19s
+-- checkpoints/                            # Layer 3: Full Checkpoints
    +-- checkpoint_20250210_120000/
    |   +-- MANIFEST.json
    |   +-- ltm_concepts.b19c
    |   +-- ltm_relations.b19c
    |   +-- ltm_strings.b19c
    |   +-- stm_state.b19s
    |   +-- cognitive_state.b19c
    |   +-- curiosity_state.b19c
    |   +-- micromodels.b19m
    |   +-- brain_controller.b19c
    |   +-- CHECKSUM
    +-- checkpoint_latest -> checkpoint_20250210_120000/
```

---

## 7. API Design

### 7.1 Neue Dateien

```
backend/persistent/                          # Layer 1
+-- persistent_store.hpp/.cpp                # Core mmap template
+-- persistent_concept_store.hpp/.cpp        # Concept-specific store
+-- persistent_relation_store.hpp/.cpp       # Relation-specific store
+-- string_pool.hpp/.cpp                     # Append-only strings
+-- tier_manager.hpp/.cpp                    # Hot/Warm/Cold
+-- access_tracker.hpp/.cpp                  # Per-concept access counting
+-- wal.hpp/.cpp                             # Write-Ahead Log
+-- persistent_config.hpp                    # Config structs

backend/snapshot/                            # Layer 2
+-- stm_snapshot_manager.hpp/.cpp            # STM Snapshot creation/loading
+-- snapshot_format.hpp                      # Binary format structs

backend/checkpoint/                          # Layer 3
+-- brain_checkpoint_manager.hpp/.cpp        # Full checkpoint creation/loading
+-- checkpoint_format.hpp                    # Per-component formats
+-- component_serializers.hpp/.cpp           # Individual component serializers
```

### 7.2 Modifizierte Dateien (Breaking Changes: KEINE)

| Datei | Aenderung | Breaking? |
|-------|----------|-----------|
| `ltm/long_term_memory.hpp` | Neuer Constructor mit `PersistentConfig`, neue private Members | Nein (Default ctor bleibt) |
| `memory/stm.hpp` | `export_snapshot()`/`import_snapshot()` + `SnapshotData` struct | Nein (Additive) |
| `memory/brain_controller.hpp` | LTM ownership, `shutdown()` erweitert, Checkpoint-Support | Nein (Additive) |
| `cognitive/cognitive_dynamics.hpp` | `export_state()`/`import_state()` fuer Focus + Tick | Nein (Additive) |

### 7.3 BrainController — Erweiterte Orchestrierung

```cpp
class BrainController {
public:
    // Existing
    bool initialize();
    void shutdown();
    
    // NEW: Persistent mode
    bool initialize(const PersistentConfig& config);
    
    // NEW: Layer 2 -- STM Snapshots
    void enable_stm_snapshots(uint32_t interval_seconds = 30);
    void disable_stm_snapshots();
    std::optional<std::string> create_stm_snapshot();
    bool load_stm_snapshot(const std::string& path);
    
    // NEW: Layer 3 -- Full Checkpoints
    std::optional<std::string> create_checkpoint();
    bool load_checkpoint(const std::string& path);
    bool load_latest_checkpoint();
    
    // NEW: Owns LTM
    LongTermMemory* get_ltm() { return ltm_.get(); }
    const LongTermMemory* get_ltm() const { return ltm_.get(); }
    
private:
    std::unique_ptr<ShortTermMemory> stm_;
    std::unique_ptr<LongTermMemory> ltm_;                    // NEW
    std::unique_ptr<STMSnapshotManager> snapshot_mgr_;       // NEW: Layer 2
    std::unique_ptr<BrainCheckpointManager> checkpoint_mgr_; // NEW: Layer 3
    bool initialized_;
    bool persistent_mode_;
};
```

---

## 8. Implementation Phases

### Phase 1: Layer 1 — Basic mmap Persistent Store (2-3 Wochen)
**Ziel:** LTM-Daten ueberleben Process-Restart

- `PersistentStore<T>` — mmap, CRUD, grow
- `StringPool` — append-only
- `PersistentConceptStore` + `PersistentRelationStore`
- `LongTermMemory` Dual-Mode (heap/persistent)
- `BrainController` Lifecycle mit LTM
- **Test:** 1M Concepts speichern, kill -9, restart, verify

### Phase 2: Layer 1 — WAL + Crash Recovery (1-2 Wochen)
**Ziel:** Kein Datenverlust bei Crash

- `WriteAheadLog` — append, checkpoint, recover
- Integration in Write-Path
- **Test:** Kill -9 waehrend Writes -> Recovery

### Phase 3: Layer 1 — Hot/Cold Tiering (1-2 Wochen)
**Ziel:** Frequently accessed Concepts in RAM pinnen

- `AccessTracker` + `TierManager`
- `mlock()`/`munlock()` fuer Hot-Tier
- Background Rebalancing

### Phase 4: Layer 2 — STM-Snapshots (1-2 Wochen)
**Ziel:** STM-State ueberlebt Crashes (max 30-60s Verlust)

- `STMSnapshotManager` — create/load/rotate
- `ShortTermMemory::export_snapshot()`/`import_snapshot()`
- `CognitiveDynamics::export_state()`/`import_state()`
- Periodischer Background-Thread
- **Test:** Activate 1000 Concepts in STM, kill -9, restart, verify Snapshot-Recovery

### Phase 5: Layer 3 — Full Brain-State Checkpoint (2-3 Wochen)
**Ziel:** Kompletter State-Snapshot, atomic, restoreable

- `BrainCheckpointManager` — create/load/validate
- Per-Component Serializers (LTM, STM, Cognitive, Curiosity, MicroModels)
- Atomic Write (temp-dir + rename)
- MANIFEST.json + CHECKSUM
- CLI Integration (`brain19_cli checkpoint create/load/list`)
- **Test:** Full Checkpoint -> Kill -> Restore -> Verify alle Subsysteme identisch

### Phase 6: Integration + Thread-Safety (1-2 Wochen)
**Ziel:** Alles zusammen, Cross-Layer Recovery

- Startup-Sequenz (Layer 1 -> 2 -> 3 Fallback)
- Shutdown-Sequenz (Layer 2 Snapshot -> Layer 3 Checkpoint -> Layer 1 Sync)
- Partitioned RW-Locks fuer Multi-Stream
- Stress-Tests

### Phase 7: NUMA + Performance (ongoing)
**Ziel:** Optimierung fuer EPYC 80-Core

- `NumaAllocator` — topology detection, page binding
- Cache-line padding, prefetch hints, huge pages
- Benchmarks: <1us p99 concept lookup

**Geschaetzter Gesamtaufwand:** 10-15 Wochen fuer alle 7 Phasen

---

## 9. Performance Targets

| Operation | Target | Layer |
|-----------|--------|-------|
| Concept lookup (hot, mlock'd) | <100ns | L1 |
| Concept lookup (warm, page cache) | <1us | L1 |
| Concept lookup (cold, SSD) | <100us | L1 |
| Concept insert (WAL + mmap) | <10us | L1 |
| STM Snapshot (1000 entries) | <5ms | L2 |
| STM Snapshot (10000 entries) | <50ms | L2 |
| STM Snapshot Load | <10ms | L2 |
| Full Checkpoint Write (100k concepts) | <5s | L3 |
| Full Checkpoint Write (1M concepts) | <30s | L3 |
| Full Checkpoint Load | <10s | L3 |
| Startup (warm, mmap + snapshot) | <2s | All |
| Startup (cold, checkpoint restore) | <30s | All |
| Crash Recovery (WAL + snapshot) | <5s | L1+L2 |

---

## Appendix A: Configuration Reference

```toml
# /data/brain19/config.toml

[persistent]
enabled = true
data_dir = "/data/brain19/persistent"
wal_dir = "/data/brain19/wal"

[persistent.capacity]
initial_concepts = 10_000_000
initial_relations = 100_000_000
initial_string_pool_gb = 10

[persistent.tiers]
hot_budget_gb = 20
warm_budget_gb = 80
promotion_threshold = 100
demotion_interval_sec = 300

[persistent.wal]
segment_size_mb = 256
checkpoint_interval_sec = 60
sync_mode = "async"

[snapshot]
enabled = true
dir = "/data/brain19/snapshots/stm"
interval_seconds = 30
max_snapshots = 5
verify_on_load = true

[checkpoint]
enabled = true
dir = "/data/brain19/checkpoints"
max_checkpoints = 3
auto_interval_minutes = 60
auto_on_shutdown = true
compress = false
verify_after_write = true

[numa]
enabled = true
auto_detect = true

[threading]
num_partitions = 64
```

---

## Appendix B: Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| mmap corruption on crash | High | WAL ensures consistency + periodic checkpoints |
| STM Snapshot inkonsistent bei Multi-Threading | Medium | fork()-based COW snapshot oder global read-lock |
| Checkpoint zu gross / zu langsam | Medium | Inkrementelle Checkpoints (nur Deltas) als future optimization |
| EpistemicMetadata Serialisierung falsch | High | Roundtrip-Tests: serialize -> deserialize -> compare |
| Disk full waehrend Checkpoint | Medium | Pre-check free space, atomic rename verhindert halbe Checkpoints |
| NUMA falsch konfiguriert | Low | Auto-detection + Fallback auf non-NUMA mode |

---

*Dieses Dokument ist die Living Architecture Specification fuer Brain19's Persistence-System. Es wird mit jeder Implementation-Phase aktualisiert.*
