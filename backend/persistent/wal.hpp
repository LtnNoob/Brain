#pragma once
// Phase 1.2: Write-Ahead Log for PersistentLTM crash recovery
//
// WAL ensures durability: operations are logged to an append-only file
// before being applied to mmap stores. On crash, WAL is replayed.

#include "persistent_records.hpp"
#include <string>
#include <cstdint>
#include <cstring>
#include <functional>

namespace brain19 {
namespace persistent {

// Forward declaration
class PersistentLTM;

// =============================================================================
// WAL Operation Types
// =============================================================================
enum class WALOpType : uint8_t {
    STORE_CONCEPT      = 1,
    ADD_RELATION       = 2,
    REMOVE_RELATION    = 3,
    INVALIDATE_CONCEPT = 4,
    UPDATE_METADATA    = 5,
};

// =============================================================================
// WAL Entry Header — 32 bytes
// =============================================================================
struct __attribute__((packed)) WALEntryHeader {
    char     magic[4];          // "WL19"
    uint64_t sequence_number;   // Monotonically increasing
    uint8_t  operation;         // WALOpType
    uint8_t  _pad[3];          // Alignment
    uint32_t payload_size;      // Bytes of payload following this header
    uint32_t checksum;          // CRC32 of header fields + payload
    uint8_t  _reserved[8];     // Future use
};
static_assert(sizeof(WALEntryHeader) == 32);

// =============================================================================
// WAL Payloads — trivially copyable structs
// =============================================================================

struct WALStoreConceptPayload {
    uint64_t concept_id;
    uint32_t label_offset;
    uint32_t label_length;
    uint32_t definition_offset;
    uint32_t definition_length;
    uint8_t  epistemic_type;
    uint8_t  epistemic_status;
    uint8_t  _pad[2];
    double   trust;
    uint64_t created_epoch_us;
};

struct WALAddRelationPayload {
    uint64_t relation_id;
    uint64_t source;
    uint64_t target;
    uint16_t type;
    uint8_t  _pad[6];
    double   weight;
};

struct WALRemoveRelationPayload {
    uint64_t relation_id;
};

struct WALInvalidateConceptPayload {
    uint64_t concept_id;
    double   invalidation_trust;
};

struct WALUpdateMetadataPayload {
    uint64_t concept_id;
    uint8_t  epistemic_type;
    uint8_t  epistemic_status;
    uint8_t  _pad[6];
    double   trust;
};

// =============================================================================
// WALWriter — append entries + checkpoint
// =============================================================================
class WALWriter {
public:
    explicit WALWriter(const std::string& wal_path);
    ~WALWriter();

    WALWriter(const WALWriter&) = delete;
    WALWriter& operator=(const WALWriter&) = delete;

    // Append an operation with payload. Returns sequence number.
    uint64_t append(WALOpType op, const void* payload, uint32_t payload_size);

    // Checkpoint: truncate WAL after stores have been msync'd
    void checkpoint();

    // Current sequence number
    uint64_t sequence() const { return next_seq_; }

    // Is WAL open?
    bool is_open() const { return fd_ >= 0; }

    friend class WALRecovery;
private:
    static uint32_t compute_checksum(const WALEntryHeader& hdr, const void* payload, uint32_t size);

    std::string path_;
    int fd_;
    uint64_t next_seq_;
};

// =============================================================================
// WALRecovery — replay WAL entries into PersistentLTM
// =============================================================================
class WALRecovery {
public:
    struct Stats {
        uint64_t entries_read = 0;
        uint64_t entries_applied = 0;
        uint64_t entries_skipped = 0;  // already present (idempotent)
        uint64_t entries_corrupt = 0;  // checksum failures at tail
    };

    // Recover: replay all valid WAL entries into ltm.
    // Returns recovery stats.
    static Stats recover(const std::string& wal_path, PersistentLTM& ltm);
};

} // namespace persistent
} // namespace brain19
