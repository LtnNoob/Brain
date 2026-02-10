#include "wal.hpp"
#include "persistent_ltm.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <array>
#include <stdexcept>
#include <vector>

namespace brain19 {
namespace persistent {

// =============================================================================
// CRC32 (thread-safe via function-local static initialization)
// =============================================================================
static const std::array<uint32_t, 256>& crc32_table() {
    static const auto table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int j = 0; j < 8; j++) {
                crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
            }
            t[i] = crc;
        }
        return t;
    }();
    return table;
}

static uint32_t crc32_update(uint32_t crc, const void* data, size_t len) {
    const auto& table = crc32_table();
    const uint8_t* buf = static_cast<const uint8_t*>(data);
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

// =============================================================================
// WALWriter
// =============================================================================

uint32_t WALWriter::compute_checksum(const WALEntryHeader& hdr, const void* payload, uint32_t size) {
    // Checksum covers: sequence_number, operation, payload_size, then payload
    uint32_t crc = 0;
    crc = crc32_update(crc, &hdr.sequence_number, sizeof(hdr.sequence_number));
    crc = crc32_update(crc, &hdr.operation, sizeof(hdr.operation));
    crc = crc32_update(crc, &hdr.payload_size, sizeof(hdr.payload_size));
    if (payload && size > 0) {
        crc = crc32_update(crc, payload, size);
    }
    return crc;
}

WALWriter::WALWriter(const std::string& wal_path)
    : path_(wal_path), fd_(-1), next_seq_(1)
{
    fd_ = ::open(wal_path.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) {
        throw std::runtime_error("WALWriter: cannot open " + wal_path + ": " + strerror(errno));
    }

    // Determine next sequence number by scanning existing entries
    struct stat st;
    if (::fstat(fd_, &st) == 0 && st.st_size > 0) {
        // Read from beginning to find max sequence
        int rfd = ::open(wal_path.c_str(), O_RDONLY);
        if (rfd >= 0) {
            WALEntryHeader hdr;
            while (true) {
                ssize_t r = ::read(rfd, &hdr, sizeof(hdr));
                if (r != sizeof(hdr)) break;
                if (std::memcmp(hdr.magic, "WL19", 4) != 0) break;
                // Skip payload
                if (hdr.payload_size > 0) {
                    if (::lseek(rfd, hdr.payload_size, SEEK_CUR) < 0) break;
                }
                if (hdr.sequence_number >= next_seq_) {
                    next_seq_ = hdr.sequence_number + 1;
                }
            }
            ::close(rfd);
        }
    }
}

WALWriter::~WALWriter() {
    if (fd_ >= 0) ::close(fd_);
}

uint64_t WALWriter::append(WALOpType op, const void* payload, uint32_t payload_size) {
    if (fd_ < 0) throw std::runtime_error("WALWriter: not open");

    WALEntryHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    std::memcpy(hdr.magic, "WL19", 4);
    hdr.sequence_number = next_seq_;
    hdr.operation = static_cast<uint8_t>(op);
    hdr.payload_size = payload_size;
    hdr.checksum = compute_checksum(hdr, payload, payload_size);

    // Write header + payload atomically via writev() to prevent torn writes on crash
    struct iovec iov[2] = {
        { &hdr, sizeof(hdr) },
        { const_cast<void*>(payload), payload_size }
    };
    int iovcnt = (payload_size > 0) ? 2 : 1;
    ssize_t written = ::writev(fd_, iov, iovcnt);
    if (written != static_cast<ssize_t>(sizeof(hdr) + payload_size)) {
        throw std::runtime_error("WALWriter: writev failed");
    }

    ::fsync(fd_);
    return next_seq_++;
}

void WALWriter::checkpoint() {
    if (fd_ < 0) return;
    // Truncate the WAL file to zero
    if (::ftruncate(fd_, 0) != 0) {
        throw std::runtime_error("WALWriter: checkpoint truncate failed");
    }
    ::lseek(fd_, 0, SEEK_SET);
    next_seq_ = 1;
}

// =============================================================================
// WALRecovery
// =============================================================================

WALRecovery::Stats WALRecovery::recover(const std::string& wal_path, PersistentLTM& ltm) {
    Stats stats;

    int fd = ::open(wal_path.c_str(), O_RDONLY);
    if (fd < 0) {
        // No WAL file = nothing to recover
        return stats;
    }

    auto read_exact = [fd](void* buf, size_t len) -> bool {
        size_t total = 0;
        while (total < len) {
            ssize_t r = ::read(fd, static_cast<char*>(buf) + total, len - total);
            if (r <= 0) return false;
            total += r;
        }
        return true;
    };

    while (true) {
        WALEntryHeader hdr;
        if (!read_exact(&hdr, sizeof(hdr))) break;

        // Validate magic
        if (std::memcmp(hdr.magic, "WL19", 4) != 0) {
            stats.entries_corrupt++;
            break;  // Corrupt entry at tail, stop
        }

        // Sanity check payload size (max 16MB)
        if (hdr.payload_size > 16 * 1024 * 1024) {
            stats.entries_corrupt++;
            break;
        }

        // Read payload
        std::vector<uint8_t> payload(hdr.payload_size);
        if (hdr.payload_size > 0) {
            if (!read_exact(payload.data(), hdr.payload_size)) {
                stats.entries_corrupt++;
                break;
            }
        }

        // Validate checksum
        uint32_t expected = WALWriter::compute_checksum(hdr, payload.data(), hdr.payload_size);
        if (hdr.checksum != expected) {
            stats.entries_corrupt++;
            break;  // Corrupt at tail, stop
        }

        stats.entries_read++;

        // Replay operation (idempotent)
        WALOpType op = static_cast<WALOpType>(hdr.operation);

        switch (op) {
            case WALOpType::STORE_CONCEPT: {
                if (hdr.payload_size < sizeof(WALStoreConceptPayload)) break;
                auto* p = reinterpret_cast<const WALStoreConceptPayload*>(payload.data());
                // Idempotent: skip if concept already exists
                if (ltm.exists(p->concept_id)) {
                    stats.entries_skipped++;
                } else {
                    // Replay: we use replay_store_concept which takes pre-built data
                    ltm.replay_store_concept(
                        p->concept_id,
                        p->label_offset, p->label_length,
                        p->definition_offset, p->definition_length,
                        p->epistemic_type, p->epistemic_status,
                        p->trust, p->created_epoch_us
                    );
                    stats.entries_applied++;
                }
                break;
            }
            case WALOpType::ADD_RELATION: {
                if (hdr.payload_size < sizeof(WALAddRelationPayload)) break;
                auto* p = reinterpret_cast<const WALAddRelationPayload*>(payload.data());
                // Idempotent: skip if relation exists
                if (ltm.get_relation(p->relation_id).has_value()) {
                    stats.entries_skipped++;
                } else {
                    ltm.replay_add_relation(
                        p->relation_id, p->source, p->target,
                        p->type, p->weight
                    );
                    stats.entries_applied++;
                }
                break;
            }
            case WALOpType::REMOVE_RELATION: {
                if (hdr.payload_size < sizeof(WALRemoveRelationPayload)) break;
                auto* p = reinterpret_cast<const WALRemoveRelationPayload*>(payload.data());
                // Idempotent: remove is safe to call even if already removed
                ltm.remove_relation(p->relation_id);
                stats.entries_applied++;
                break;
            }
            case WALOpType::INVALIDATE_CONCEPT: {
                if (hdr.payload_size < sizeof(WALInvalidateConceptPayload)) break;
                auto* p = reinterpret_cast<const WALInvalidateConceptPayload*>(payload.data());
                ltm.invalidate_concept(p->concept_id, p->invalidation_trust);
                stats.entries_applied++;
                break;
            }
            case WALOpType::UPDATE_METADATA: {
                if (hdr.payload_size < sizeof(WALUpdateMetadataPayload)) break;
                auto* p = reinterpret_cast<const WALUpdateMetadataPayload*>(payload.data());
                EpistemicMetadata meta(
                    static_cast<EpistemicType>(p->epistemic_type),
                    static_cast<EpistemicStatus>(p->epistemic_status),
                    p->trust
                );
                ltm.update_epistemic_metadata(p->concept_id, meta);
                stats.entries_applied++;
                break;
            }
            default:
                stats.entries_corrupt++;
                break;
        }
    }

    ::close(fd);
    return stats;
}

} // namespace persistent
} // namespace brain19
