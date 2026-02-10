#include "persistence.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace brain19 {
namespace persistence {

static constexpr char MAGIC[4] = {'B', 'M', '1', '9'};
static constexpr uint32_t VERSION = 1;
static constexpr size_t HEADER_SIZE = 32;
static constexpr size_t MODEL_SIZE = 8 + FLAT_SIZE * 8;  // 3448 bytes

// XOR checksum over 8-byte blocks
static uint64_t compute_checksum(const std::vector<uint8_t>& data) {
    uint64_t checksum = 0;
    size_t full_blocks = data.size() / 8;
    for (size_t i = 0; i < full_blocks; ++i) {
        uint64_t block = 0;
        std::memcpy(&block, data.data() + i * 8, 8);
        checksum ^= block;
    }
    // Handle remaining bytes
    size_t remaining = data.size() % 8;
    if (remaining > 0) {
        uint64_t block = 0;
        std::memcpy(&block, data.data() + full_blocks * 8, remaining);
        checksum ^= block;
    }
    return checksum;
}

// =============================================================================
// Save
// =============================================================================

bool save(const std::string& filepath,
          const MicroModelRegistry& registry,
          const EmbeddingManager& embeddings) {

    std::vector<uint8_t> buffer;

    auto write_bytes = [&](const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), p, p + size);
    };

    auto model_ids = registry.get_model_ids();
    auto context_names = embeddings.get_context_names();

    // --- Header ---
    write_bytes(MAGIC, 4);
    write_bytes(&VERSION, 4);
    uint64_t model_count = model_ids.size();
    write_bytes(&model_count, 8);
    uint64_t context_count = context_names.size();
    write_bytes(&context_count, 8);
    uint64_t reserved = 0;
    write_bytes(&reserved, 8);

    // --- Models ---
    std::array<double, FLAT_SIZE> flat;
    for (ConceptId cid : model_ids) {
        const MicroModel* model = registry.get_model(cid);
        if (!model) continue;

        uint64_t id_val = cid;
        write_bytes(&id_val, 8);
        model->to_flat(flat);
        write_bytes(flat.data(), FLAT_SIZE * sizeof(double));
    }

    // --- Relation embeddings ---
    const auto& rel_embs = embeddings.relation_embeddings();
    for (size_t i = 0; i < NUM_RELATION_TYPES; ++i) {
        write_bytes(rel_embs[i].data(), EMBED_DIM * sizeof(double));
    }

    // --- Context embeddings ---
    const auto& ctx_map = embeddings.context_embeddings();
    for (const auto& name : context_names) {
        auto it = ctx_map.find(name);
        if (it == ctx_map.end()) continue;

        uint32_t name_len = static_cast<uint32_t>(name.size());
        write_bytes(&name_len, 4);
        write_bytes(name.data(), name_len);
        write_bytes(it->second.data(), EMBED_DIM * sizeof(double));
    }

    // --- Checksum ---
    uint64_t checksum = compute_checksum(buffer);
    write_bytes(&checksum, 8);

    // Write to file
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
    return out.good();
}

// =============================================================================
// Load
// =============================================================================

bool load(const std::string& filepath,
          MicroModelRegistry& registry,
          EmbeddingManager& embeddings) {

    std::ifstream in(filepath, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;

    auto file_size = in.tellg();
    if (file_size < static_cast<std::streamoff>(HEADER_SIZE + 8)) return false;

    in.seekg(0);
    std::vector<uint8_t> buffer(static_cast<size_t>(file_size));
    in.read(reinterpret_cast<char*>(buffer.data()), file_size);
    if (!in.good()) return false;

    size_t pos = 0;
    auto read_bytes = [&](void* dest, size_t size) -> bool {
        if (pos + size > buffer.size()) return false;
        std::memcpy(dest, buffer.data() + pos, size);
        pos += size;
        return true;
    };

    // --- Verify checksum ---
    // Checksum is last 8 bytes, computed over everything before it
    if (buffer.size() < 8) return false;
    size_t data_size = buffer.size() - 8;
    std::vector<uint8_t> data_portion(buffer.begin(), buffer.begin() + data_size);
    uint64_t expected_checksum = compute_checksum(data_portion);
    uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, buffer.data() + data_size, 8);
    if (expected_checksum != stored_checksum) return false;

    // --- Header ---
    char magic[4];
    if (!read_bytes(magic, 4)) return false;
    if (std::memcmp(magic, MAGIC, 4) != 0) return false;

    uint32_t version = 0;
    if (!read_bytes(&version, 4)) return false;
    if (version != VERSION) return false;

    uint64_t model_count = 0;
    if (!read_bytes(&model_count, 8)) return false;

    uint64_t context_count = 0;
    if (!read_bytes(&context_count, 8)) return false;

    uint64_t reserved = 0;
    if (!read_bytes(&reserved, 8)) return false;

    // --- Models ---
    registry.clear();
    std::array<double, FLAT_SIZE> flat;
    for (uint64_t i = 0; i < model_count; ++i) {
        uint64_t cid = 0;
        if (!read_bytes(&cid, 8)) return false;
        if (!read_bytes(flat.data(), FLAT_SIZE * sizeof(double))) return false;

        registry.create_model(static_cast<ConceptId>(cid));
        MicroModel* model = registry.get_model(static_cast<ConceptId>(cid));
        if (model) {
            model->from_flat(flat);
        }
    }

    // --- Relation embeddings ---
    auto& rel_embs = embeddings.relation_embeddings_mut();
    for (size_t i = 0; i < NUM_RELATION_TYPES; ++i) {
        if (!read_bytes(rel_embs[i].data(), EMBED_DIM * sizeof(double))) return false;
    }

    // --- Context embeddings ---
    auto& ctx_map = embeddings.context_embeddings_mut();
    ctx_map.clear();
    for (uint64_t i = 0; i < context_count; ++i) {
        uint32_t name_len = 0;
        if (!read_bytes(&name_len, 4)) return false;
        if (name_len > 1024) return false;  // Sanity check

        std::string name(name_len, '\0');
        if (!read_bytes(name.data(), name_len)) return false;

        Vec10 emb;
        if (!read_bytes(emb.data(), EMBED_DIM * sizeof(double))) return false;

        ctx_map[name] = emb;
    }

    return true;
}

// =============================================================================
// Validate
// =============================================================================

bool validate(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;

    auto file_size = in.tellg();
    if (file_size < static_cast<std::streamoff>(HEADER_SIZE + 8)) return false;

    in.seekg(0);
    std::vector<uint8_t> buffer(static_cast<size_t>(file_size));
    in.read(reinterpret_cast<char*>(buffer.data()), file_size);
    if (!in.good()) return false;

    // Check magic
    if (std::memcmp(buffer.data(), MAGIC, 4) != 0) return false;

    // Check version
    uint32_t version = 0;
    std::memcpy(&version, buffer.data() + 4, 4);
    if (version != VERSION) return false;

    // Verify checksum
    if (buffer.size() < 8) return false;
    size_t data_size = buffer.size() - 8;
    std::vector<uint8_t> data_portion(buffer.begin(), buffer.begin() + data_size);
    uint64_t expected = compute_checksum(data_portion);
    uint64_t stored = 0;
    std::memcpy(&stored, buffer.data() + data_size, 8);

    return expected == stored;
}

} // namespace persistence
} // namespace brain19
