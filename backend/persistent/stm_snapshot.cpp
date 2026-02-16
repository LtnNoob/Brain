#include "stm_snapshot.hpp"

#include <fstream>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <bit>

namespace brain19 {

// Snapshot format assumes little-endian. Guard against silent corruption on big-endian.
static_assert(std::endian::native == std::endian::little,
              "STM snapshot binary format requires little-endian architecture");

// ── helpers ──

namespace {

// CRC32 for snapshot integrity (same algorithm as WAL)
static const std::array<uint32_t, 256>& crc32_table() {
    static const auto table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j)
                c = (c >> 1) ^ (0xEDB88320u & (-(c & 1)));
            t[i] = c;
        }
        return t;
    }();
    return table;
}

static uint32_t crc32_update(uint32_t crc, const void* data, size_t len) {
    const auto& table = crc32_table();
    auto* p = static_cast<const uint8_t*>(data);
    crc = ~crc;
    for (size_t i = 0; i < len; ++i)
        crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

template<typename T>
void write_pod(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<typename T>
bool read_pod(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return f.good();
}

} // anon

// ── STMSnapshotManager ──

STMSnapshotManager::STMSnapshotManager(size_t max_snapshots)
    : max_snapshots_(max_snapshots)
{}

bool STMSnapshotManager::create_snapshot(ShortTermMemory& stm, const std::string& path) {
    // Export state from STM
    STMSnapshotData data = stm.export_state();

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    // Header
    write_pod(out, STM_SNAPSHOT_MAGIC);
    write_pod(out, STM_SNAPSHOT_VERSION);
    write_pod(out, data.timestamp);

    uint32_t context_count = static_cast<uint32_t>(data.contexts.size());
    uint32_t total_concepts = 0, total_relations = 0;
    for (auto& ctx : data.contexts) {
        total_concepts  += static_cast<uint32_t>(ctx.concepts.size());
        total_relations += static_cast<uint32_t>(ctx.relations.size());
    }
    write_pod(out, context_count);
    write_pod(out, total_concepts);
    write_pod(out, total_relations);

    // Per context
    for (auto& ctx : data.contexts) {
        write_pod(out, ctx.context_id);
        uint32_t cc = static_cast<uint32_t>(ctx.concepts.size());
        uint32_t rc = static_cast<uint32_t>(ctx.relations.size());
        write_pod(out, cc);
        write_pod(out, rc);

        for (auto& c : ctx.concepts) {
            write_pod(out, c.concept_id);
            write_pod(out, c.activation);
            uint8_t cls = static_cast<uint8_t>(c.classification);
            write_pod(out, cls);
        }

        for (auto& r : ctx.relations) {
            write_pod(out, r.source);
            write_pod(out, r.target);
            uint16_t t = static_cast<uint16_t>(r.type);
            write_pod(out, t);
            write_pod(out, r.activation);
        }
    }

    if (!out.good()) return false;

    // Write CRC32 footer over entire file content
    out.flush();
    std::ifstream reread(path, std::ios::binary);
    if (!reread) return false;
    uint32_t crc = 0;
    char buf[4096];
    while (reread.read(buf, sizeof(buf)) || reread.gcount() > 0) {
        crc = crc32_update(crc, buf, static_cast<size_t>(reread.gcount()));
    }
    reread.close();
    std::ofstream append(path, std::ios::binary | std::ios::app);
    if (!append) return false;
    write_pod(append, crc);
    return append.good();
}

bool STMSnapshotManager::load_snapshot(const std::string& path, STMSnapshotData& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t magic; uint16_t version;
    if (!read_pod(in, magic) || magic != STM_SNAPSHOT_MAGIC) return false;
    if (!read_pod(in, version) || version != STM_SNAPSHOT_VERSION) return false;

    if (!read_pod(in, out.timestamp)) return false;

    uint32_t context_count, total_concepts, total_relations;
    if (!read_pod(in, context_count)) return false;
    if (!read_pod(in, total_concepts)) return false;
    if (!read_pod(in, total_relations)) return false;

    out.contexts.resize(context_count);
    for (uint32_t i = 0; i < context_count; ++i) {
        auto& ctx = out.contexts[i];
        if (!read_pod(in, ctx.context_id)) return false;

        uint32_t cc, rc;
        if (!read_pod(in, cc)) return false;
        if (!read_pod(in, rc)) return false;

        ctx.concepts.resize(cc);
        for (uint32_t j = 0; j < cc; ++j) {
            auto& c = ctx.concepts[j];
            if (!read_pod(in, c.concept_id)) return false;
            if (!read_pod(in, c.activation)) return false;
            uint8_t cls;
            if (!read_pod(in, cls)) return false;
            c.classification = static_cast<ActivationClass>(cls);
        }

        ctx.relations.resize(rc);
        for (uint32_t j = 0; j < rc; ++j) {
            auto& r = ctx.relations[j];
            if (!read_pod(in, r.source)) return false;
            if (!read_pod(in, r.target)) return false;
            uint16_t t;
            if (!read_pod(in, t)) return false;
            r.type = static_cast<RelationType>(t);
            if (!read_pod(in, r.activation)) return false;
        }
    }

    // Verify CRC32 footer
    auto data_end = in.tellg();
    uint32_t stored_crc;
    if (!read_pod(in, stored_crc)) return true;  // No CRC = legacy snapshot, accept
    in.close();

    // Recompute CRC over data portion
    std::ifstream verify(path, std::ios::binary);
    if (!verify) return false;
    uint32_t computed_crc = 0;
    size_t remaining = static_cast<size_t>(data_end);
    char buf[4096];
    while (remaining > 0) {
        size_t to_read = std::min(remaining, sizeof(buf));
        verify.read(buf, static_cast<std::streamsize>(to_read));
        if (!verify.good() && !verify.eof()) return false;
        computed_crc = crc32_update(computed_crc, buf, static_cast<size_t>(verify.gcount()));
        remaining -= static_cast<size_t>(verify.gcount());
    }
    if (computed_crc != stored_crc) return false;

    return true;
}

void STMSnapshotManager::apply_snapshot(ShortTermMemory& stm, const STMSnapshotData& data) {
    stm.import_state(data);
}

void STMSnapshotManager::rotate_snapshots(const std::string& directory, const std::string& prefix) {
    namespace fs = std::filesystem;
    if (!fs::exists(directory)) return;

    std::vector<fs::path> matching;
    for (auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().filename().string().rfind(prefix, 0) == 0) {
            matching.push_back(entry.path());
        }
    }

    if (matching.size() <= max_snapshots_) return;

    // Sort by last write time, oldest first
    std::sort(matching.begin(), matching.end(), [](const fs::path& a, const fs::path& b) {
        return fs::last_write_time(a) < fs::last_write_time(b);
    });

    size_t to_remove = matching.size() - max_snapshots_;
    for (size_t i = 0; i < to_remove; ++i) {
        fs::remove(matching[i]);
    }
}

} // namespace brain19
