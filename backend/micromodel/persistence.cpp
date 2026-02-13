#include "persistence.hpp"
#include "../memory/relation_type_registry.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace brain19 {
namespace persistence {

static constexpr char MAGIC[4] = {'B', 'M', '1', '9'};
static constexpr uint32_t VERSION = 3;  // v3: FlexEmbedding (16D core + variable detail)
static constexpr uint32_t VERSION_V2 = 2;
static constexpr uint32_t VERSION_V1 = 1;
static constexpr size_t HEADER_SIZE = 32;
static constexpr size_t MODEL_SIZE = 8 + FLAT_SIZE * 8;  // 7528 bytes per model (v3)
static constexpr size_t V1_NUM_RELATION_TYPES = 10;
static constexpr size_t OLD_EMBED_DIM = 10;  // v1/v2 used 10D
static constexpr size_t OLD_FLAT_SIZE = 430; // v1/v2 model size

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

// Helper: write a FlexEmbedding to buffer
static void write_flex_embedding(const FlexEmbedding& emb,
                                  std::vector<uint8_t>& buffer) {
    auto write_bytes = [&](const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), p, p + size);
    };

    // Core (16 doubles = 128 bytes)
    write_bytes(emb.core.data(), CORE_DIM * sizeof(double));
    // Detail dim count
    uint16_t detail_dim = static_cast<uint16_t>(emb.detail.size());
    write_bytes(&detail_dim, sizeof(detail_dim));
    // Detail data
    if (detail_dim > 0) {
        write_bytes(emb.detail.data(), detail_dim * sizeof(double));
    }
}

// Helper: read a FlexEmbedding from buffer
static bool read_flex_embedding(FlexEmbedding& emb,
                                 const std::vector<uint8_t>& buffer,
                                 size_t& pos) {
    auto read_bytes = [&](void* dest, size_t size) -> bool {
        if (pos + size > buffer.size()) return false;
        std::memcpy(dest, buffer.data() + pos, size);
        pos += size;
        return true;
    };

    if (!read_bytes(emb.core.data(), CORE_DIM * sizeof(double))) return false;
    uint16_t detail_dim = 0;
    if (!read_bytes(&detail_dim, sizeof(detail_dim))) return false;
    emb.detail.resize(detail_dim);
    if (detail_dim > 0) {
        if (!read_bytes(emb.detail.data(), detail_dim * sizeof(double))) return false;
    }
    return true;
}

// Helper: read old 10D embedding and migrate to 16D FlexEmbedding
static bool read_old_embedding(FlexEmbedding& emb,
                                const std::vector<uint8_t>& buffer,
                                size_t& pos) {
    auto read_bytes = [&](void* dest, size_t size) -> bool {
        if (pos + size > buffer.size()) return false;
        std::memcpy(dest, buffer.data() + pos, size);
        pos += size;
        return true;
    };

    std::array<double, 10> old_emb{};
    if (!read_bytes(old_emb.data(), OLD_EMBED_DIM * sizeof(double))) return false;
    emb = FlexEmbedding::from_vec10(old_emb);
    return true;
}

// Helper: migrate old 430-double flat model to new 940-double flat
static void migrate_flat_v2_to_v3(const std::array<double, OLD_FLAT_SIZE>& old_flat,
                                   std::array<double, FLAT_SIZE>& new_flat) {
    new_flat.fill(0.0);
    size_t old_idx = 0;
    size_t new_idx = 0;

    // W: old 10x10 -> new 16x16 (embed in top-left corner)
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        for (size_t j = 0; j < OLD_EMBED_DIM; ++j) {
            new_flat[i * CORE_DIM + j] = old_flat[old_idx++];
        }
    }
    new_idx = CORE_DIM * CORE_DIM;  // 256

    // b: old 10 -> new 16 (pad with 0)
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
    new_idx += CORE_DIM;  // 272

    // e_init: old 10 -> new 16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
    new_idx += CORE_DIM;  // 288

    // c_init: old 10 -> new 16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
    new_idx += CORE_DIM;  // 304

    // TrainingState: dW_momentum 10x10 -> 16x16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        for (size_t j = 0; j < OLD_EMBED_DIM; ++j) {
            new_flat[new_idx + i * CORE_DIM + j] = old_flat[old_idx++];
        }
    }
    new_idx += CORE_DIM * CORE_DIM;  // 560

    // db_momentum: 10 -> 16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
    new_idx += CORE_DIM;  // 576

    // dW_variance: 10x10 -> 16x16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        for (size_t j = 0; j < OLD_EMBED_DIM; ++j) {
            new_flat[new_idx + i * CORE_DIM + j] = old_flat[old_idx++];
        }
    }
    new_idx += CORE_DIM * CORE_DIM;  // 832

    // db_variance: 10 -> 16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
    new_idx += CORE_DIM;  // 848

    // e_grad_accum: 10 -> 16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
    new_idx += CORE_DIM;  // 864

    // c_grad_accum: 10 -> 16
    for (size_t i = 0; i < OLD_EMBED_DIM; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
    new_idx += CORE_DIM;  // 880

    // scalars (5) + reserved (55) = 60 doubles, same layout
    for (size_t i = 0; i < 60; ++i) {
        new_flat[new_idx + i] = old_flat[old_idx++];
    }
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

    // --- Relation embeddings (v3: FlexEmbedding) ---
    auto& reg = RelationTypeRegistry::instance();
    auto all_types = reg.all_types();
    uint32_t rel_emb_count = static_cast<uint32_t>(all_types.size());
    write_bytes(&rel_emb_count, 4);

    for (RelationType rt : all_types) {
        uint16_t type_val = static_cast<uint16_t>(rt);
        write_bytes(&type_val, 2);
        write_flex_embedding(reg.get_embedding(rt), buffer);
    }

    // --- Concept embeddings (v3: FlexEmbedding) ---
    const auto& concept_emb_data = embeddings.concept_embeddings().data();
    uint32_t concept_emb_count = static_cast<uint32_t>(concept_emb_data.size());
    write_bytes(&concept_emb_count, 4);
    for (const auto& [cid, emb] : concept_emb_data) {
        uint64_t id_val = cid;
        write_bytes(&id_val, 8);
        write_flex_embedding(emb, buffer);
    }

    // --- Context embeddings (v3: FlexEmbedding) ---
    const auto& ctx_map = embeddings.context_embeddings();
    for (const auto& name : context_names) {
        auto it = ctx_map.find(name);
        if (it == ctx_map.end()) continue;

        uint32_t name_len = static_cast<uint32_t>(name.size());
        write_bytes(&name_len, 4);
        write_bytes(name.data(), name_len);
        write_flex_embedding(it->second, buffer);
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
    if (version != VERSION && version != VERSION_V2 && version != VERSION_V1) return false;

    uint64_t model_count = 0;
    if (!read_bytes(&model_count, 8)) return false;

    uint64_t context_count = 0;
    if (!read_bytes(&context_count, 8)) return false;

    uint64_t reserved = 0;
    if (!read_bytes(&reserved, 8)) return false;

    // --- Models ---
    registry.clear();
    if (version >= VERSION) {
        // v3: 940-double flat format
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
    } else {
        // v1/v2: 430-double flat format — migrate to 940
        std::array<double, OLD_FLAT_SIZE> old_flat;
        std::array<double, FLAT_SIZE> new_flat;
        for (uint64_t i = 0; i < model_count; ++i) {
            uint64_t cid = 0;
            if (!read_bytes(&cid, 8)) return false;
            if (!read_bytes(old_flat.data(), OLD_FLAT_SIZE * sizeof(double))) return false;

            migrate_flat_v2_to_v3(old_flat, new_flat);
            registry.create_model(static_cast<ConceptId>(cid));
            MicroModel* model = registry.get_model(static_cast<ConceptId>(cid));
            if (model) {
                model->from_flat(new_flat);
            }
        }
    }

    // --- Relation embeddings ---
    if (version == VERSION_V1) {
        // v1: fixed 10 relation embeddings (skip — registry has its own defaults)
        for (size_t i = 0; i < V1_NUM_RELATION_TYPES; ++i) {
            pos += OLD_EMBED_DIM * sizeof(double);
        }
    } else if (version == VERSION_V2) {
        // v2: counted relation embeddings with type IDs (10D)
        uint32_t rel_emb_count = 0;
        if (!read_bytes(&rel_emb_count, 4)) return false;
        for (uint32_t i = 0; i < rel_emb_count; ++i) {
            uint16_t type_val = 0;
            if (!read_bytes(&type_val, 2)) return false;
            pos += OLD_EMBED_DIM * sizeof(double);  // Skip — registry manages
        }
    } else {
        // v3: FlexEmbedding relation embeddings
        uint32_t rel_emb_count = 0;
        if (!read_bytes(&rel_emb_count, 4)) return false;
        for (uint32_t i = 0; i < rel_emb_count; ++i) {
            uint16_t type_val = 0;
            if (!read_bytes(&type_val, 2)) return false;
            FlexEmbedding emb;
            if (!read_flex_embedding(emb, buffer, pos)) return false;
            // Read but don't override registry defaults
        }
    }

    // --- Concept embeddings ---
    if (version >= VERSION) {
        // v3: FlexEmbedding
        uint32_t concept_emb_count = 0;
        if (!read_bytes(&concept_emb_count, 4)) return false;
        auto& concept_store = embeddings.concept_embeddings().data_mut();
        concept_store.clear();
        for (uint32_t i = 0; i < concept_emb_count; ++i) {
            uint64_t id_val = 0;
            if (!read_bytes(&id_val, 8)) return false;
            FlexEmbedding emb;
            if (!read_flex_embedding(emb, buffer, pos)) return false;
            concept_store[static_cast<ConceptId>(id_val)] = std::move(emb);
        }
    } else if (version == VERSION_V2) {
        // v2: 10D concept embeddings — migrate to 16D core
        uint32_t concept_emb_count = 0;
        if (!read_bytes(&concept_emb_count, 4)) return false;
        auto& concept_store = embeddings.concept_embeddings().data_mut();
        concept_store.clear();
        for (uint32_t i = 0; i < concept_emb_count; ++i) {
            uint64_t id_val = 0;
            if (!read_bytes(&id_val, 8)) return false;
            FlexEmbedding emb;
            if (!read_old_embedding(emb, buffer, pos)) return false;
            concept_store[static_cast<ConceptId>(id_val)] = std::move(emb);
        }
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

        FlexEmbedding emb;
        if (version >= VERSION) {
            if (!read_flex_embedding(emb, buffer, pos)) return false;
        } else {
            if (!read_old_embedding(emb, buffer, pos)) return false;
        }

        ctx_map[name] = std::move(emb);
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
    if (version != VERSION && version != VERSION_V2 && version != VERSION_V1) return false;

    // Verify checksum
    if (buffer.size() < 8) return false;
    size_t data_size_val = buffer.size() - 8;
    std::vector<uint8_t> data_portion(buffer.begin(), buffer.begin() + data_size_val);
    uint64_t expected = compute_checksum(data_portion);
    uint64_t stored = 0;
    std::memcpy(&stored, buffer.data() + data_size_val, 8);

    return expected == stored;
}

} // namespace persistence
} // namespace brain19
