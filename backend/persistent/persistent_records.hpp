#pragma once
// Phase 1.1: Fixed-size mmap-able record types for LTM persistence
//
// These structs are designed for direct mmap access:
// - Fixed size, no pointers, no virtual functions
// - Cache-line friendly alignment
// - Strings stored separately in StringPool (offset+length)

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace brain19 {
namespace persistent {

// =============================================================================
// File header for persistent stores
// =============================================================================
struct StoreHeader {
    char     magic[4];          // "B19C" for concepts, "B19R" for relations, "B19S" for strings
    uint32_t version;           // Format version (1)
    uint64_t record_count;      // Number of valid records
    uint64_t capacity;          // Total slots allocated
    uint64_t next_id;           // Next ID to assign
    uint8_t  _reserved[32];     // Future use
    // Total: 64 bytes
};
static_assert(sizeof(StoreHeader) == 64);

// =============================================================================
// PersistentConceptRecord — 128 bytes, 2 cache lines
// =============================================================================
struct PersistentConceptRecord {
    uint64_t concept_id;            // 8B
    uint32_t label_offset;          // 4B — into StringPool
    uint32_t label_length;          // 4B
    uint32_t definition_offset;     // 4B
    uint32_t definition_length;     // 4B
    uint8_t  epistemic_type;        // 1B (EpistemicType enum)
    uint8_t  epistemic_status;      // 1B (EpistemicStatus enum)
    uint8_t  flags;                 // 1B (bit 0 = deleted)
    uint8_t  _pad1[5];             // 5B alignment
    double   trust;                 // 8B
    uint64_t access_count;          // 8B — for future hot/cold tiering
    uint64_t last_access_epoch_us;  // 8B
    uint64_t created_epoch_us;      // 8B
    // === Refactor: 4 doubles carved from _reserved (32B used, 32B remain) ===
    double   activation_score;      // 8B — FocusCursor activation [0,1]
    double   salience_score;        // 8B — Computed salience [0,1]
    double   structural_confidence; // 8B — Graph-structural confidence [0,1]
    double   semantic_confidence;   // 8B — Semantic confidence [0,1]
    uint8_t  _reserved[32];        // 32B — remaining pad to 128
    // Total: 128 bytes (unchanged)
    
    bool is_deleted() const { return flags & 0x01; }
    void mark_deleted() { flags |= 0x01; }
    void clear() { std::memset(this, 0, sizeof(*this)); }
};
static_assert(sizeof(PersistentConceptRecord) == 128);
static_assert(std::is_trivially_copyable_v<PersistentConceptRecord>);

// =============================================================================
// PersistentRelationRecord — 64 bytes, 1 cache line
// =============================================================================
struct PersistentRelationRecord {
    uint64_t relation_id;       // 8B
    uint64_t source;            // 8B
    uint64_t target;            // 8B
    uint8_t  type;              // 1B (RelationType low byte)
    uint8_t  type_high;         // 1B (RelationType high byte — was _pad[0])
    uint8_t  flags;             // 1B (bit 0 = deleted)
    uint8_t  _pad[5];          // 5B alignment (was 6)
    double   weight;            // 8B
    // === Refactor: 3 doubles carved from _reserved (24B used, 0B remain) ===
    double   dynamic_weight;    // 8B — Runtime-adjusted weight [0,1]
    double   inhibition_factor; // 8B — Inhibition from conflict [0,1]
    double   structural_strength; // 8B — Graph-structural strength [0,1]
    // Total: 64 bytes (unchanged)

    // Type accessor helpers (combines type + type_high into uint16_t)
    uint16_t get_type_id() const {
        return static_cast<uint16_t>(type) | (static_cast<uint16_t>(type_high) << 8);
    }
    void set_type_id(uint16_t val) {
        type = static_cast<uint8_t>(val & 0xFF);
        type_high = static_cast<uint8_t>((val >> 8) & 0xFF);
    }
    
    bool is_deleted() const { return flags & 0x01; }
    void mark_deleted() { flags |= 0x01; }
    void clear() { std::memset(this, 0, sizeof(*this)); }
};
static_assert(sizeof(PersistentRelationRecord) == 64);
static_assert(std::is_trivially_copyable_v<PersistentRelationRecord>);

// =============================================================================
// StringPool header — 64 bytes
// =============================================================================
struct StringPoolHeader {
    char     magic[4];          // "B19S"
    uint32_t version;           // 1
    uint64_t used_bytes;        // Current write position (after header)
    uint64_t capacity;          // Total file size
    uint8_t  _reserved[40];    // Future use
    // Total: 64 bytes
};
static_assert(sizeof(StringPoolHeader) == 64);

// Flag constants
constexpr uint8_t RECORD_FLAG_DELETED = 0x01;

} // namespace persistent
} // namespace brain19
