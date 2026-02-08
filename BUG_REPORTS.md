# Brain19 Bug Reports & Issue Tracker
## Comprehensive Issue Documentation

**Generated:** January 6, 2026  
**Version:** 1.0  
**System:** Brain19 Complete v1.0  

---

## Issue Summary

| ID | Title | Severity | Priority | Status | Effort |
|----|-------|----------|----------|--------|--------|
| BUG-001 | Snapshot relations array always empty | HIGH | HIGH | OPEN | 2-4h |
| BUG-002 | Missing Long-Term Memory implementation | HIGH | HIGH | OPEN | 2-3d |
| BUG-003 | Epistemology integration incomplete | MEDIUM | MEDIUM | OPEN | 1-2d |
| BUG-004 | No persistence layer implemented | MEDIUM | MEDIUM | OPEN | 1-2d |
| ENH-001 | Limited curiosity trigger patterns | LOW | LOW | OPEN | 4-8h |
| ENH-002 | Manual snapshot copy required | LOW | MEDIUM | OPEN | 4-8h |
| ENH-003 | No zoom/pan in visualization | LOW | LOW | OPEN | 2-4h |

---

# BUG-001: Snapshot Relations Array Always Empty

## Metadata
- **Type:** Bug
- **Severity:** HIGH
- **Priority:** HIGH
- **Status:** OPEN
- **Assignee:** Unassigned
- **Reported:** 2026-01-06
- **Component:** backend/snapshot_generator.cpp
- **Affects:** Visualization, Integration

## Description

The `generate_json_snapshot()` function always returns an empty `active_relations` array, even when relations are active in STM. This causes the visualization to display only nodes without any edges, making the graph incomplete and less useful.

## Steps to Reproduce

1. Run the integrated demo:
   ```bash
   cd backend
   ./demo_integrated
   ```

2. Observe that the demo activates 3 relations:
   ```cpp
   brain.activate_relation_in_context(ctx, 100, 200, RelationType::IS_A, 0.88);
   brain.activate_relation_in_context(ctx, 200, 300, RelationType::HAS_PROPERTY, 0.75);
   brain.activate_relation_in_context(ctx, 300, 400, RelationType::CAUSES, 0.62);
   ```

3. Check the generated `snapshot.json`:
   ```json
   "active_relations": []  // Always empty!
   ```

4. Load snapshot in visualization - no edges are displayed

## Expected Behavior

```json
{
  "stm": {
    "active_relations": [
      {"source": 100, "target": 200, "type": "IS_A", "activation": 0.88},
      {"source": 200, "target": 300, "type": "HAS_PROPERTY", "activation": 0.75},
      {"source": 300, "target": 400, "type": "CAUSES", "activation": 0.62}
    ]
  }
}
```

Visualization should display:
- Nodes: 100, 200, 300, 400
- Edges: 3 directed arrows connecting them

## Actual Behavior

```json
{
  "stm": {
    "active_relations": []  // Empty!
  }
}
```

Visualization displays:
- Nodes: 100, 200, 300, 400
- Edges: None

## Root Cause

File: `backend/snapshot_generator.cpp`, Line ~40

```cpp
// Active relations (simplified - would need actual relation data)
json << "    \"active_relations\": [\n";
json << "    ]\n";  // ← Always hardcoded empty!
```

The code contains a comment indicating this is a stub, but was never implemented.

## Proposed Fix

### Option A: Add STM query method (RECOMMENDED)

1. Add to `BrainController` (or STM):
   ```cpp
   struct ActiveRelationInfo {
       ConceptId source;
       ConceptId target;
       std::string type;
       double activation;
   };
   
   std::vector<ActiveRelationInfo> query_active_relations(
       ContextId context_id,
       double min_activation = 0.0
   ) const;
   ```

2. Implement in `snapshot_generator.cpp`:
   ```cpp
   auto relations = brain->query_active_relations(context_id);
   json << "    \"active_relations\": [\n";
   for (size_t i = 0; i < relations.size(); i++) {
       json << "      {";
       json << "\"source\": " << relations[i].source << ", ";
       json << "\"target\": " << relations[i].target << ", ";
       json << "\"type\": \"" << relations[i].type << "\", ";
       json << "\"activation\": " << relations[i].activation;
       json << "}";
       if (i < relations.size() - 1) json << ",";
       json << "\n";
   }
   json << "    ]\n";
   ```

### Option B: Access STM directly

Extract relations directly from STM data structures (less clean).

## Acceptance Criteria

- [ ] `snapshot.json` contains all active relations
- [ ] Relations have correct source/target IDs
- [ ] Activation levels are preserved
- [ ] Relation types are included
- [ ] Frontend visualization displays edges
- [ ] Integration test passes with visible edges
- [ ] No performance regression

## Test Plan

1. **Unit Test:**
   ```cpp
   void test_snapshot_relations() {
       BrainController brain;
       brain.initialize();
       auto ctx = brain.create_context();
       brain.begin_thinking(ctx);
       
       // Activate concepts and relations
       brain.activate_concept_in_context(ctx, 1, 0.9, CORE);
       brain.activate_concept_in_context(ctx, 2, 0.8, CORE);
       brain.activate_relation_in_context(ctx, 1, 2, IS_A, 0.85);
       
       // Generate snapshot
       SnapshotGenerator gen;
       std::string json = gen.generate_json_snapshot(&brain, nullptr, ctx);
       
       // Verify
       assert(json.find("\"source\": 1") != std::string::npos);
       assert(json.find("\"target\": 2") != std::string::npos);
       assert(json.find("\"activation\": 0.85") != std::string::npos);
   }
   ```

2. **Integration Test:**
   - Run `demo_integrated`
   - Verify `snapshot.json` contains relations
   - Load in frontend
   - Verify edges are visible
   - Hover over edges to see details

3. **Regression Test:**
   - Verify concepts still appear
   - Verify performance is acceptable
   - Verify JSON is valid

## Estimated Effort

- **Analysis:** 30 minutes
- **Implementation:** 2-3 hours
- **Testing:** 1-2 hours
- **Documentation:** 30 minutes
- **Total:** 2-4 hours

## Impact Assessment

**Without Fix:**
- Visualization is incomplete
- Users cannot see knowledge structure
- Reduces system usefulness by ~40%
- Creates wrong impression of system

**With Fix:**
- Complete graph visualization
- Clear knowledge structure
- Proper edge weights visible
- System demonstrates full capability

## Dependencies

- Depends on: STM query API
- Blocks: Visualization usability
- Relates to: BUG-003 (epistemology)

## Notes

This is a **critical visual bug**. While the backend works correctly (relations ARE stored and decay properly), the visualization cannot show this. This creates a disconnect between system capability and user perception.

**Recommendation:** Fix immediately before any demo or release.

---

# BUG-002: Missing Long-Term Memory Implementation

## Metadata
- **Type:** Missing Feature
- **Severity:** HIGH
- **Priority:** HIGH
- **Status:** OPEN
- **Assignee:** Unassigned
- **Reported:** 2026-01-06
- **Component:** backend/ltm/ (not implemented)
- **Affects:** Core Functionality, Persistence

## Description

Brain19 currently has NO implementation of Long-Term Memory (LTM). All knowledge exists only in Short-Term Memory (STM), which means:
- No persistent concept storage
- No stable knowledge base
- No epistemological metadata storage
- Knowledge is lost when contexts are destroyed
- No semantic memory

This is a **fundamental missing component** that prevents the system from functioning as a true cognitive architecture.

## Current State

**Exists:**
- ✅ STM (activation tracking)
- ✅ BrainController (orchestration)
- ✅ Epistemology types (enums only)

**Missing:**
- ❌ LTM storage
- ❌ Concept definitions
- ❌ Persistent relations
- ❌ Epistemological metadata
- ❌ Trust tracking
- ❌ Source attribution

## Expected Behavior

### Minimal LTM Implementation

```cpp
// backend/ltm/long_term_memory.hpp

class LongTermMemory {
public:
    // Concept management
    ConceptId store_concept(const std::string& label,
                           const std::string& definition);
    std::optional<ConceptInfo> retrieve_concept(ConceptId id) const;
    
    // Epistemology
    void set_epistemic_type(ConceptId id, EpistemicType type);
    void set_trust(ConceptId id, double trust);
    
    // Relations
    void store_relation(ConceptId source, ConceptId target, 
                       RelationType type);
    std::vector<Relation> get_relations(ConceptId concept_id) const;
    
    // Query
    std::vector<ConceptId> search_by_label(const std::string& query) const;
};

struct ConceptInfo {
    ConceptId id;
    std::string label;
    std::string definition;
    EpistemicType epistemic_type;
    std::optional<double> trust;
    Timestamp created_at;
    std::vector<SourceAttribution> sources;
};
```

### Integration with BrainController

```cpp
class BrainController {
    // Current: Only STM
    std::unique_ptr<ShortTermMemory> stm_;
    
    // Add: LTM integration
    std::unique_ptr<LongTermMemory> ltm_;
    
    // New methods:
    ConceptId create_concept(const std::string& label,
                            const std::string& definition);
    void activate_from_ltm(ContextId ctx, ConceptId concept_id);
};
```

## Actual Behavior

```cpp
// Attempting to use concepts:
brain.activate_concept_in_context(ctx, 100, 0.9, CORE);
// Where does concept 100 come from? ← NOWHERE!
// It's just an arbitrary ID with no meaning
```

**Current Workaround:**
- Developers use arbitrary concept IDs (100, 200, 300...)
- No way to know what these concepts represent
- No persistence between sessions
- No semantic information

## Root Cause

LTM was **never implemented** in the initial architecture. The focus was on:
1. STM (activation) ✅
2. KAN (function learning) ✅
3. Curiosity (pattern detection) ✅
4. Importers (knowledge candidates) ✅

But the **foundational storage layer** was deferred.

## Proposed Implementation

### Phase 1: Basic Storage (Week 1)

**Files:**
- `backend/ltm/long_term_memory.hpp`
- `backend/ltm/long_term_memory.cpp`
- `backend/ltm/concept_store.hpp`
- `backend/ltm/concept_store.cpp`

**Features:**
```cpp
class LongTermMemory {
private:
    std::unordered_map<ConceptId, ConceptInfo> concepts_;
    std::unordered_map<ConceptId, std::vector<Relation>> relations_;
    ConceptId next_concept_id_;
    
public:
    ConceptId store_concept(const std::string& label, 
                           const std::string& definition);
    std::optional<ConceptInfo> retrieve_concept(ConceptId id) const;
    bool exists(ConceptId id) const;
};
```

**Scope:**
- In-memory storage (no persistence yet)
- Basic CRUD operations
- Simple linear search

### Phase 2: Epistemology Integration (Week 2)

**Features:**
```cpp
void set_epistemic_type(ConceptId id, EpistemicType type);
void set_trust(ConceptId id, double trust);
std::optional<EpistemicType> get_epistemic_type(ConceptId id) const;
std::optional<double> get_trust(ConceptId id) const;
```

**Integration:**
- Snapshot generator queries LTM for epistemic data
- Visualization shows proper epistemic types
- Trust values displayed when present

### Phase 3: Source Attribution (Week 3)

**Features:**
```cpp
struct SourceAttribution {
    std::string source_type;  // "wikipedia", "scholar", "manual"
    std::string source_reference;  // URL, DOI, etc.
    Timestamp imported_at;
};

void add_source(ConceptId id, const SourceAttribution& source);
std::vector<SourceAttribution> get_sources(ConceptId id) const;
```

### Phase 4: Relations (Week 4)

**Features:**
```cpp
void store_relation(ConceptId source, ConceptId target, 
                   RelationType type, double weight = 1.0);
std::vector<Relation> get_relations_from(ConceptId source) const;
std::vector<Relation> get_relations_to(ConceptId target) const;
```

## Acceptance Criteria

**Phase 1 (Basic Storage):**
- [ ] Can store concepts with labels and definitions
- [ ] Can retrieve concepts by ID
- [ ] Concepts persist within session
- [ ] Tests: 10+ unit tests passing
- [ ] Integration: BrainController can use LTM
- [ ] Memory: No leaks detected

**Phase 2 (Epistemology):**
- [ ] Can set/get epistemic type
- [ ] Can set/get trust (optional)
- [ ] Snapshot includes correct epistemic data
- [ ] Visualization displays types correctly
- [ ] Tests: 5+ epistemology tests passing

**Phase 3 (Sources):**
- [ ] Can attribute multiple sources per concept
- [ ] Source metadata preserved
- [ ] Importers can add source info
- [ ] Tests: 5+ source attribution tests

**Phase 4 (Relations):**
- [ ] Can store typed relations
- [ ] Can query relations by source/target
- [ ] Relations have weights
- [ ] Tests: 8+ relation tests

## Test Plan

### Unit Tests (per phase)

```cpp
// Phase 1
void test_store_and_retrieve_concept();
void test_concept_exists();
void test_invalid_concept_id();

// Phase 2
void test_set_epistemic_type();
void test_trust_values();
void test_optional_trust();

// Phase 3
void test_add_source();
void test_multiple_sources();
void test_source_timestamp();

// Phase 4
void test_store_relation();
void test_query_relations();
void test_relation_types();
```

### Integration Tests

```cpp
void test_importer_to_ltm_workflow() {
    WikipediaImporter importer;
    auto proposal = importer.parse_wikipedia_text("Cat", "...");
    
    // Human reviews and approves
    LongTermMemory ltm;
    auto concept_id = ltm.store_concept(
        proposal->title,
        proposal->extracted_text
    );
    
    ltm.set_epistemic_type(concept_id, EpistemicType::FACT);
    ltm.set_trust(concept_id, 0.8);
    ltm.add_source(concept_id, {
        .source_type = "wikipedia",
        .source_reference = proposal->source_reference,
        .imported_at = proposal->import_timestamp
    });
    
    // Verify
    auto info = ltm.retrieve_concept(concept_id);
    assert(info.has_value());
    assert(info->epistemic_type == EpistemicType::FACT);
}
```

## Estimated Effort

**Breakdown:**
- Phase 1 (Basic): 16-24 hours
- Phase 2 (Epistemology): 8-16 hours
- Phase 3 (Sources): 8-12 hours
- Phase 4 (Relations): 12-16 hours
- **Total:** 44-68 hours (1-2 weeks full-time, 2-3 weeks part-time)

**Critical Path:**
- Week 1: Basic storage + integration
- Week 2: Epistemology + snapshot integration
- Week 3-4: Sources + relations (can be parallel)

## Impact Assessment

**Without LTM:**
- ❌ System is incomplete
- ❌ No persistent knowledge
- ❌ Cannot store import results
- ❌ No semantic memory
- ❌ Concept IDs are meaningless
- ❌ System is essentially "demo-only"

**With LTM:**
- ✅ Complete cognitive architecture
- ✅ Persistent knowledge base
- ✅ Import workflow functional
- ✅ Epistemology integrated
- ✅ Source attribution working
- ✅ Production-ready foundation

## Dependencies

- Blocks: Full system functionality
- Blocks: Importer integration
- Blocks: Epistemology system
- Enables: Persistence layer (BUG-004)
- Enables: Mindmap logic
- Enables: OutputGate

## Notes

This is the **single most important missing component**. Without LTM, Brain19 is essentially:
- A demonstration of architecture
- A prototype of subsystems
- NOT a functional cognitive system

**Recommendation:** Prioritize above all other enhancements. Implement in phases to get basic functionality quickly, then iterate.

**Alternative:** Could use existing knowledge base systems (e.g., SQLite, embedded graph DB) as backend storage, but custom implementation maintains transparency and control.

---

# BUG-003: Epistemology Integration Incomplete

## Metadata
- **Type:** Missing Integration
- **Severity:** MEDIUM
- **Priority:** MEDIUM
- **Status:** OPEN
- **Assignee:** Unassigned
- **Reported:** 2026-01-06
- **Component:** backend/epistemic/ (partial)
- **Affects:** Snapshot, Visualization, Importers

## Description

Epistemological types are defined as enums throughout the codebase, but there is NO actual epistemology subsystem to:
- Assign types to concepts
- Track trust values
- Validate epistemic consistency
- Query epistemic metadata

This results in:
- All concepts showing "UNKNOWN" in visualization
- Trust always showing `null`
- Importers suggesting types that are never used
- No way to distinguish FACT from HYPOTHESIS

## Current State

**Defined Enums:**
```cpp
// In various files:
enum class EpistemicType {
    FACT,
    DEFINITION,
    THEORY,
    HYPOTHESIS,
    INFERENCE,
    SPECULATION,
    UNKNOWN
};

enum class SuggestedEpistemicType {
    FACT_CANDIDATE,
    THEORY_CANDIDATE,
    HYPOTHESIS_CANDIDATE,
    DEFINITION_CANDIDATE,
    UNKNOWN_CANDIDATE
};
```

**Missing:**
- ❌ Epistemology subsystem
- ❌ Type assignment API
- ❌ Trust calculation API
- ❌ Consistency checking
- ❌ Query interface

## Expected Behavior

```cpp
// backend/epistemic/epistemology.hpp

class Epistemology {
public:
    // Type management
    void set_type(ConceptId id, EpistemicType type);
    std::optional<EpistemicType> get_type(ConceptId id) const;
    
    // Trust management
    void set_trust(ConceptId id, double trust);  // [0.0, 1.0]
    std::optional<double> get_trust(ConceptId id) const;
    
    // Validation
    bool is_consistent(ConceptId id) const;
    std::vector<std::string> get_inconsistencies(ConceptId id) const;
    
    // Query
    std::vector<ConceptId> get_by_type(EpistemicType type) const;
    std::vector<ConceptId> get_by_trust_range(double min, double max) const;
};
```

**Rules:**
```cpp
// Epistemological constraints:
- UNKNOWN cannot have trust value (structurally impossible)
- FACT requires high trust (e.g., > 0.8)
- HYPOTHESIS can have any trust
- SPECULATION should have low trust (< 0.5)
- Trust must be in [0.0, 1.0]
```

## Actual Behavior

**Snapshot Output:**
```json
{
  "concepts": [
    {"id": 100, "epistemic_type": "UNKNOWN", "trust": null},
    {"id": 200, "epistemic_type": "UNKNOWN", "trust": null}
  ]
}
```

**Importer Output:**
```cpp
// Importer suggests:
proposal->suggested_epistemic_type = SuggestedEpistemicType::FACT_CANDIDATE;

// But this is NEVER used anywhere!
// Human sees suggestion, but has no API to actually SET it
```

## Root Cause

1. **Enums defined** in multiple places (inconsistent)
2. **No central epistemology subsystem**
3. **LTM doesn't exist** (nowhere to store epistemic data)
4. **Snapshot generator** has no epistemology to query
5. **BrainController** has no epistemology integration

## Proposed Fix

### Phase 1: Create Epistemology Subsystem (1-2 days)

**File Structure:**
```
backend/epistemic/
  ├── epistemology.hpp
  ├── epistemology.cpp
  ├── epistemic_type.hpp (consolidated enums)
  └── test_epistemology.cpp
```

**Implementation:**
```cpp
class Epistemology {
private:
    std::unordered_map<ConceptId, EpistemicType> types_;
    std::unordered_map<ConceptId, double> trust_;
    
public:
    void set_type(ConceptId id, EpistemicType type) {
        // Validate
        if (type == EpistemicType::UNKNOWN && trust_.count(id)) {
            throw std::invalid_argument("UNKNOWN cannot have trust");
        }
        types_[id] = type;
    }
    
    void set_trust(ConceptId id, double trust) {
        // Validate
        if (trust < 0.0 || trust > 1.0) {
            throw std::out_of_range("Trust must be [0.0, 1.0]");
        }
        if (types_[id] == EpistemicType::UNKNOWN) {
            throw std::invalid_argument("UNKNOWN cannot have trust");
        }
        trust_[id] = trust;
    }
    
    std::optional<EpistemicType> get_type(ConceptId id) const {
        auto it = types_.find(id);
        return (it != types_.end()) ? std::optional(it->second) : std::nullopt;
    }
};
```

### Phase 2: Integrate with LTM (depends on BUG-002)

```cpp
class LongTermMemory {
private:
    std::unique_ptr<Epistemology> epistemology_;
    
public:
    void set_epistemic_type(ConceptId id, EpistemicType type) {
        epistemology_->set_type(id, type);
    }
};
```

### Phase 3: Update Snapshot Generator

```cpp
std::string SnapshotGenerator::generate_json_snapshot(...) {
    // Query epistemology
    auto epistemic_type = brain->get_epistemic_type(concept_id);
    auto trust = brain->get_trust(concept_id);
    
    json << "\"epistemic_type\": \"" << to_string(epistemic_type) << "\", ";
    if (trust.has_value()) {
        json << "\"trust\": " << trust.value();
    } else {
        json << "\"trust\": null";
    }
}
```

## Acceptance Criteria

- [ ] Epistemology subsystem implemented
- [ ] Can set/get epistemic type
- [ ] Can set/get trust (optional)
- [ ] UNKNOWN cannot have trust (enforced)
- [ ] Trust range validated [0.0, 1.0]
- [ ] Integrated with LTM
- [ ] Snapshot shows correct types
- [ ] Visualization displays types
- [ ] Tests: 8+ unit tests
- [ ] Documentation updated

## Test Plan

```cpp
void test_set_epistemic_type() {
    Epistemology epi;
    epi.set_type(1, EpistemicType::FACT);
    assert(epi.get_type(1) == EpistemicType::FACT);
}

void test_trust_validation() {
    Epistemology epi;
    epi.set_type(1, EpistemicType::HYPOTHESIS);
    
    // Valid
    epi.set_trust(1, 0.7);  // OK
    
    // Invalid
    try {
        epi.set_trust(1, 1.5);  // Should throw
        assert(false);
    } catch (std::out_of_range&) {
        // Expected
    }
}

void test_unknown_cannot_have_trust() {
    Epistemology epi;
    epi.set_type(1, EpistemicType::UNKNOWN);
    
    try {
        epi.set_trust(1, 0.5);  // Should throw
        assert(false);
    } catch (std::invalid_argument&) {
        // Expected
    }
}
```

## Estimated Effort

- Epistemology subsystem: 8 hours
- LTM integration: 4 hours (after BUG-002)
- Snapshot integration: 2 hours
- Tests: 4 hours
- Documentation: 2 hours
- **Total:** 20 hours (1-2 days)

## Impact Assessment

**Without Fix:**
- All concepts labeled "UNKNOWN" (meaningless)
- No trust differentiation
- Cannot distinguish facts from speculation
- Importer suggestions wasted
- Epistemic rigor promise unfulfilled

**With Fix:**
- Clear epistemic status for all concepts
- Trust levels visible and meaningful
- FACT vs HYPOTHESIS distinction clear
- Importer integration complete
- System fulfills epistemic promise

## Dependencies

- Depends on: BUG-002 (LTM) for full integration
- Blocks: Full epistemic functionality
- Enables: Proper visualization
- Enables: Trust-based reasoning

## Notes

Can be partially implemented **before LTM** as standalone subsystem, then integrated later. This allows visualization testing while LTM is being built.

**Quick Win:** Implement epistemology subsystem in isolation, use hardcoded concept data for testing, then integrate with LTM when ready.

---

# BUG-004: No Persistence Layer Implemented

## Metadata
- **Type:** Missing Feature
- **Severity:** MEDIUM
- **Priority:** MEDIUM
- **Status:** OPEN
- **Assignee:** Unassigned
- **Reported:** 2026-01-06
- **Component:** backend/ (no persistence)
- **Affects:** All subsystems, Long-term usability

## Description

Brain19 has NO persistence layer. All data exists only in memory and is lost when the process exits:
- STM state lost on shutdown
- LTM lost (when implemented)
- Epistemology lost
- KAN modules lost
- All work is ephemeral

This makes the system unsuitable for any real use beyond demos.

## Expected Behavior

**Save/Load API:**
```cpp
class PersistenceManager {
public:
    // Save entire system state
    void save_to_file(const std::string& filepath,
                     const BrainController& brain);
    
    // Load entire system state
    void load_from_file(const std::string& filepath,
                       BrainController& brain);
    
    // Incremental saves
    void save_ltm(const std::string& filepath, 
                 const LongTermMemory& ltm);
    void save_epistemology(const std::string& filepath,
                          const Epistemology& epi);
};
```

**Usage:**
```cpp
// Save
PersistenceManager pm;
pm.save_to_file("brain19_state.json", brain);

// Later... load
BrainController brain;
pm.load_from_file("brain19_state.json", brain);
// All concepts, relations, epistemic data restored
```

## Actual Behavior

```bash
./demo_integrated
# Creates concepts, trains KAN, etc.
# Exit program
./demo_integrated
# All data lost! Start from scratch.
```

## Root Cause

No serialization implementation. System was designed for in-memory operation only.

## Proposed Implementation

### Phase 1: JSON Serialization (Week 1)

**Format:**
```json
{
  "version": "1.0",
  "timestamp": "2026-01-06T16:00:00Z",
  "ltm": {
    "concepts": [
      {
        "id": 1,
        "label": "Cat",
        "definition": "A small carnivorous mammal",
        "epistemic_type": "FACT",
        "trust": 0.95,
        "sources": [...]
      }
    ],
    "relations": [...]
  },
  "epistemology": {...},
  "kan_modules": [...]
}
```

**Implementation:**
```cpp
class JSONSerializer {
public:
    std::string serialize_ltm(const LongTermMemory& ltm);
    void deserialize_ltm(const std::string& json, LongTermMemory& ltm);
    
private:
    std::string escape_json_string(const std::string& str);
    // Use existing JSON generation code from snapshot_generator
};
```

### Phase 2: Incremental Saves (Week 2)

**Auto-save:**
```cpp
class PersistenceManager {
private:
    std::chrono::seconds auto_save_interval_;
    std::thread auto_save_thread_;
    
public:
    void enable_auto_save(const std::string& filepath,
                         std::chrono::seconds interval = 300s) {
        // Save every 5 minutes
    }
};
```

### Phase 3: Differential Saves (Optional)

Only save changes since last save (optimization).

## Acceptance Criteria

- [ ] Can save entire system state to JSON
- [ ] Can load system state from JSON
- [ ] Data integrity preserved (no corruption)
- [ ] Timestamps included
- [ ] Version number included (for future compatibility)
- [ ] Error handling (file not found, corrupt JSON)
- [ ] Tests: Save/load round-trip successful
- [ ] Performance: < 100ms for typical dataset
- [ ] Auto-save optional feature works

## Test Plan

```cpp
void test_save_and_load_round_trip() {
    // Create system with data
    BrainController brain;
    brain.initialize();
    // ... populate with concepts, relations, etc.
    
    // Save
    PersistenceManager pm;
    pm.save_to_file("test_state.json", brain);
    
    // Load into new instance
    BrainController brain2;
    pm.load_from_file("test_state.json", brain2);
    
    // Verify equivalence
    assert(brain2.query_active_concepts(...).size() == ...);
    // ... more assertions
}

void test_corrupted_file() {
    PersistenceManager pm;
    BrainController brain;
    
    try {
        pm.load_from_file("corrupted.json", brain);
        assert(false);  // Should throw
    } catch (std::runtime_error& e) {
        // Expected
        assert(std::string(e.what()).find("corrupted") != std::string::npos);
    }
}
```

## Estimated Effort

- JSON serialization: 12-16 hours
- Deserialization: 8-12 hours
- Error handling: 4 hours
- Auto-save: 4-6 hours
- Tests: 6-8 hours
- **Total:** 34-46 hours (1-2 weeks)

## Impact Assessment

**Without Persistence:**
- ❌ All work is ephemeral
- ❌ Cannot build knowledge over time
- ❌ System reset on every restart
- ❌ Unsuitable for real use
- ❌ Demo-only quality

**With Persistence:**
- ✅ Knowledge accumulates
- ✅ Work is preserved
- ✅ Can restart and continue
- ✅ Production-quality
- ✅ Long-term usable

## Dependencies

- Depends on: BUG-002 (LTM) for full functionality
- Depends on: BUG-003 (Epistemology) to save epistemic data
- Enables: Long-term system use
- Enables: Knowledge base building

## Notes

**File Format Choice:**
- JSON: Human-readable, debuggable, no dependencies ✅
- Binary: Faster, smaller, harder to debug
- SQLite: Queryable, structured, adds dependency

**Recommendation:** Start with JSON for transparency, optimize later if needed.

---

# ENH-001: Limited Curiosity Trigger Patterns

## Metadata
- **Type:** Enhancement
- **Severity:** LOW
- **Priority:** LOW
- **Status:** OPEN
- **Assignee:** Unassigned
- **Reported:** 2026-01-06
- **Component:** backend/curiosity/curiosity_engine.cpp
- **Affects:** Curiosity detection capability

## Description

Curiosity Engine currently implements only 2 trigger patterns:
- `SHALLOW_RELATIONS` (relations < 30% of concepts)
- `LOW_EXPLORATION` (< 5 active concepts)

The design identified 4 trigger types, but only 2 are implemented:
- ❌ `MISSING_DEPTH` - Not implemented
- ❌ `RECURRENT_WITHOUT_FUNCTION` - Not implemented

## Expected Behavior

**Full Trigger Suite:**
```cpp
enum class TriggerType {
    SHALLOW_RELATIONS,              // Implemented ✅
    LOW_EXPLORATION,                // Implemented ✅
    MISSING_DEPTH,                  // Not implemented ❌
    RECURRENT_WITHOUT_FUNCTION,     // Not implemented ❌
    CONCEPT_DRIFT                   // Not in original design
};
```

**Missing Depth:**
```cpp
// Detects: Same concepts activated repeatedly without
//          deeper structural understanding
bool detect_missing_depth(const std::vector<SystemObservation>& history) {
    // Check if same concepts appear across observations
    // But no new relations or deeper connections form
}
```

**Recurrent Without Function:**
```cpp
// Detects: Pattern repeats but no KAN function learned
bool detect_recurrent_without_function(
    const std::vector<SystemObservation>& history,
    const KANAdapter& kan_adapter
) {
    // Check if activation pattern is recurrent
    // But no function hypothesis exists for it
}
```

## Proposed Implementation

### MISSING_DEPTH Detection

```cpp
struct ObservationHistory {
    std::deque<SystemObservation> recent_observations;
    size_t max_history = 10;
    
    void add(const SystemObservation& obs) {
        recent_observations.push_back(obs);
        if (recent_observations.size() > max_history) {
            recent_observations.pop_front();
        }
    }
};

bool CuriosityEngine::detect_missing_depth(
    const ObservationHistory& history
) const {
    if (history.recent_observations.size() < 3) {
        return false;
    }
    
    // Count concept repetitions
    std::unordered_map<ConceptId, size_t> concept_counts;
    for (const auto& obs : history.recent_observations) {
        // Would need to track which concepts are active
        // Requires extension to SystemObservation
    }
    
    // If same concepts appear >= 3 times
    // But relation count hasn't increased
    // → Missing depth trigger
}
```

### RECURRENT_WITHOUT_FUNCTION Detection

```cpp
bool CuriosityEngine::detect_recurrent_without_function(
    const ObservationHistory& history,
    const KANAdapter* kan_adapter  // Optional dependency
) const {
    // This requires:
    // 1. Pattern detection (statistical)
    // 2. Query if KAN has learned this pattern
    // 3. Complex, may violate "simple patterns only" principle
    
    // Recommendation: DEFER until use case is clear
}
```

## Acceptance Criteria

- [ ] MISSING_DEPTH trigger implemented
- [ ] Detects repeated concepts without deepening
- [ ] Configurable repetition threshold
- [ ] Tests: 3+ scenarios
- [ ] No false positives on legitimate repetition
- [ ] Does not slow down normal operation

## Test Plan

```cpp
void test_missing_depth_detection() {
    CuriosityEngine curiosity;
    ObservationHistory history;
    
    // Add same observation 4 times
    SystemObservation obs;
    obs.context_id = 1;
    obs.active_concept_count = 5;
    obs.active_relation_count = 2;
    
    for (int i = 0; i < 4; i++) {
        history.add(obs);
    }
    
    // Should trigger MISSING_DEPTH
    auto triggers = curiosity.observe_with_history(history);
    
    bool found = false;
    for (const auto& t : triggers) {
        if (t.type == TriggerType::MISSING_DEPTH) {
            found = true;
            break;
        }
    }
    assert(found);
}
```

## Estimated Effort

- MISSING_DEPTH design: 2 hours
- Implementation: 4 hours
- Testing: 2 hours
- **Total:** 8 hours (1 day)

## Impact Assessment

**Without Enhancement:**
- Curiosity is functional but limited
- May miss some learning opportunities
- Still adequate for current use

**With Enhancement:**
- More comprehensive pattern detection
- Better learning signal generation
- Richer BrainController inputs

## Dependencies

- Requires: ObservationHistory structure
- Optional: KAN integration (for RECURRENT_WITHOUT_FUNCTION)
- Does not block: Any core functionality

## Notes

**Design Philosophy Question:**
Adding more trigger types increases complexity. Does this violate "simple patterns only"?

**Recommendation:** 
- Implement MISSING_DEPTH (still simple, threshold-based)
- DEFER RECURRENT_WITHOUT_FUNCTION (too complex)
- DOCUMENT decision to keep Curiosity simple

---

# ENH-002: Manual Snapshot Copy Required

## Metadata
- **Type:** Enhancement
- **Severity:** LOW
- **Priority:** MEDIUM
- **Status:** OPEN
- **Assignee:** Unassigned
- **Reported:** 2026-01-06
- **Component:** Integration, backend/
- **Affects:** Developer experience, Frontend integration

## Description

Currently, to visualize Brain19 state, developers must:
1. Run backend: `./demo_integrated`
2. Manually copy: `cp snapshot.json ../frontend/public/`
3. Start frontend: `npm run dev`

This is cumbersome and prevents live updates.

## Expected Behavior

**Option A: HTTP REST Endpoint**
```bash
# Terminal 1: Backend with HTTP server
./brain19_server --port 8080

# Terminal 2: Frontend (auto-fetches from backend)
npm run dev
# Frontend fetches http://localhost:8080/api/snapshot
```

**Option B: File Watcher**
```bash
# Terminal 1: Backend (auto-saves)
./demo_integrated --output frontend/public/snapshot.json

# Terminal 2: Frontend (auto-reloads)
npm run dev
# Watches snapshot.json, reloads on change
```

## Proposed Implementation

### Option A: HTTP Server (RECOMMENDED)

**backend/http_server.cpp:**
```cpp
#include <httplib.h>  // cpp-httplib (header-only)

class Brain19HTTPServer {
private:
    BrainController* brain_;
    CuriosityEngine* curiosity_;
    SnapshotGenerator snapshot_gen_;
    httplib::Server server_;
    
public:
    Brain19HTTPServer(BrainController* brain, 
                     CuriosityEngine* curiosity)
        : brain_(brain), curiosity_(curiosity) {
        
        // GET /api/snapshot
        server_.Get("/api/snapshot", [this](const auto&, auto& res) {
            std::string json = snapshot_gen_.generate_json_snapshot(
                brain_, curiosity_, /* context_id */ 1
            );
            res.set_content(json, "application/json");
        });
        
        // Health check
        server_.Get("/health", [](const auto&, auto& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
    }
    
    void start(int port = 8080) {
        server_.listen("localhost", port);
    }
};
```

**frontend/src/App.jsx:**
```jsx
useEffect(() => {
  const fetchSnapshot = async () => {
    const response = await fetch('http://localhost:8080/api/snapshot');
    const data = await response.json();
    setSnapshot(data);
  };
  
  fetchSnapshot();
  
  // Poll every 2 seconds (optional)
  const interval = setInterval(fetchSnapshot, 2000);
  return () => clearInterval(interval);
}, []);
```

### Option B: WebSocket (Live Updates)

**For real-time updates:**
```cpp
// Backend broadcasts on state change
void BrainController::activate_concept_in_context(...) {
    // ... normal activation
    
    // Broadcast update
    if (websocket_server_) {
        websocket_server_->broadcast_update();
    }
}
```

**Frontend listens:**
```jsx
useEffect(() => {
  const ws = new WebSocket('ws://localhost:8080/ws');
  ws.onmessage = (event) => {
    const snapshot = JSON.parse(event.data);
    setSnapshot(snapshot);
  };
  return () => ws.close();
}, []);
```

## Acceptance Criteria

**HTTP Implementation:**
- [ ] Backend provides `/api/snapshot` endpoint
- [ ] Frontend can fetch snapshot via HTTP
- [ ] CORS headers configured correctly
- [ ] Health check endpoint works
- [ ] Error handling (backend not running)
- [ ] Documentation updated
- [ ] Still supports file-based workflow (fallback)

**WebSocket (Optional):**
- [ ] Live updates work
- [ ] Connection management robust
- [ ] Reconnection on disconnect
- [ ] Throttling (max 10 updates/sec)

## Test Plan

```bash
# Test HTTP endpoint
curl http://localhost:8080/api/snapshot
# Should return valid JSON

curl http://localhost:8080/health
# Should return {"status":"ok"}

# Test frontend integration
npm run dev
# Should load snapshot automatically
# Should show graph
```

## Estimated Effort

**HTTP Implementation:**
- Add cpp-httplib: 30 minutes
- Implement server: 3 hours
- Frontend integration: 2 hours
- Testing: 2 hours
- Documentation: 1 hour
- **Total:** 8-10 hours (1 day)

**WebSocket (additional):**
- WebSocket server: 4 hours
- Frontend client: 2 hours
- Testing: 2 hours
- **Total:** +8 hours

## Impact Assessment

**Without Enhancement:**
- Manual workflow is tedious
- No live updates
- Developer friction
- Demo setup is cumbersome

**With Enhancement:**
- Seamless integration
- Live updates possible
- Better developer experience
- Professional demo quality

## Dependencies

- External: cpp-httplib (header-only, MIT license)
- Optional: libwebsockets (for WebSocket)
- Does not block: Core functionality

## Notes

**Security Consideration:**
HTTP server should:
- Only bind to localhost (not 0.0.0.0)
- No authentication needed (local only)
- CORS restricted to localhost:3000

**Recommendation:**
- Start with HTTP (simple, effective)
- Add WebSocket later if needed
- Maintain file-based as fallback

---

# ENH-003: No Zoom/Pan in Visualization

## Metadata
- **Type:** Enhancement
- **Severity:** LOW
- **Priority:** LOW
- **Status:** OPEN
- **Assignee:** Unassigned
- **Reported:** 2026-01-06
- **Component:** frontend/src/Brain19Visualizer.jsx
- **Affects:** Visualization usability (large graphs)

## Description

The React visualization does not support zoom or pan. With graphs > 50 nodes, the visualization becomes difficult to use:
- Nodes may overlap
- Cannot focus on specific area
- Cannot see details of dense regions

## Expected Behavior

**Zoom:**
- Mouse wheel to zoom in/out
- Zoom level: 0.1x - 10x
- Zoom centered on mouse position

**Pan:**
- Click and drag background to pan
- Arrow keys for keyboard panning
- Reset button to restore view

## Proposed Implementation

### Using d3-zoom

```jsx
import { zoom } from 'd3-zoom';
import { select } from 'd3-selection';

const STMGraph = ({ data, concepts }) => {
  const svgRef = useRef(null);
  const [transform, setTransform] = useState({ k: 1, x: 0, y: 0 });
  
  useEffect(() => {
    const svg = select(svgRef.current);
    
    const zoomBehavior = zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        setTransform(event.transform);
      });
    
    svg.call(zoomBehavior);
    
    return () => {
      svg.on('.zoom', null);
    };
  }, []);
  
  return (
    <svg ref={svgRef} width="600" height="400">
      <g transform={`translate(${transform.x},${transform.y}) scale(${transform.k})`}>
        {/* Graph content */}
      </g>
    </svg>
  );
};
```

### Controls

```jsx
const ZoomControls = ({ onZoomIn, onZoomOut, onReset }) => (
  <div style={styles.controls}>
    <button onClick={onZoomIn}>+</button>
    <button onClick={onZoomOut}>−</button>
    <button onClick={onReset}>Reset</button>
  </div>
);
```

## Acceptance Criteria

- [ ] Mouse wheel zooms in/out
- [ ] Click-drag pans the view
- [ ] Zoom controls (+/−/Reset) work
- [ ] Smooth animation (not jarring)
- [ ] Zoom level displayed
- [ ] Min/max zoom enforced
- [ ] Still read-only (no node dragging)
- [ ] Performance acceptable (60 FPS)

## Test Plan

**Manual Testing:**
1. Load visualization with 50+ nodes
2. Scroll mouse wheel → Should zoom
3. Click and drag background → Should pan
4. Click controls → Should zoom/reset
5. Verify hover still works when zoomed
6. Verify no controls to modify data

**Performance:**
- Test with 100 nodes
- Measure FPS during zoom/pan
- Should maintain 60 FPS

## Estimated Effort

- d3-zoom integration: 2 hours
- Control buttons: 1 hour
- Styling: 30 minutes
- Testing: 1 hour
- **Total:** 4-5 hours

## Impact Assessment

**Without Enhancement:**
- Large graphs are hard to navigate
- Users cannot focus on areas of interest
- Limited to ~50 nodes practically

**With Enhancement:**
- Can handle 100+ nodes
- Users can explore graph freely
- More professional UX
- Better usability

## Dependencies

- External: d3-zoom, d3-selection (already use d3 style)
- Does not affect: Read-only guarantee

## Notes

**Important:** Zoom/pan must NOT enable node dragging. The visualization remains strictly read-only:
- Pan: drag BACKGROUND only
- Zoom: mouse wheel only
- NO node manipulation

---

# Summary of All Issues

## Priority Matrix

```
            HIGH          MEDIUM         LOW
CRITICAL    BUG-001       BUG-003        
            BUG-002                      
                                         
IMPORTANT                 BUG-004        ENH-002
                                         
MINOR                                   ENH-001
                                         ENH-003
```

## Recommended Action Order

**Sprint 1 (Week 1):**
1. BUG-001 - Fix snapshot relations (4 hours) ← IMMEDIATE
2. BUG-002 Phase 1 - Basic LTM (3 days)

**Sprint 2 (Week 2):**
3. BUG-003 - Epistemology integration (2 days)
4. BUG-002 Phase 2 - LTM epistemology (2 days)

**Sprint 3 (Week 3):**
5. BUG-004 - Persistence (1 week)
6. ENH-002 - HTTP endpoint (1 day)

**Sprint 4 (Week 4+):**
7. ENH-001 - More curiosity triggers (1 day)
8. ENH-003 - Zoom/pan (0.5 days)

**Total Estimated Time:** 4-5 weeks to clear all critical issues

---

*End of Bug Reports*  
*Generated: January 6, 2026*
