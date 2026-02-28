#include "concept_persistence.hpp"
#include "../memory/relation_type_registry.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace brain19 {
namespace persistence {

static constexpr char MAGIC[4] = {'B', 'M', '1', '9'};
static constexpr uint32_t VERSION_V8 = 8;
static constexpr uint32_t VERSION_V7 = 7;
static constexpr uint32_t VERSION_V6 = 6;
static constexpr uint32_t VERSION_V5 = 5;
static constexpr uint32_t VERSION_V4 = 4;
static constexpr uint32_t VERSION_V3 = 3;
static constexpr uint32_t VERSION_V2 = 2;
static constexpr uint32_t VERSION_V1 = 1;
static constexpr size_t HEADER_SIZE = 32;
static constexpr size_t V3_FLAT_SIZE = 940;
static constexpr size_t V4_FLAT_SIZE = 1300;
static constexpr size_t V5_FLAT_SIZE = CM_FLAT_SIZE_V5;  // 1900
static constexpr size_t V6_FLAT_SIZE = CM_FLAT_SIZE_V6;  // 5836
static constexpr size_t V7_FLAT_SIZE = CM_FLAT_SIZE_V7;  // 9772

// XOR checksum over 8-byte blocks
static uint64_t compute_checksum(const std::vector<uint8_t>& data) {
    uint64_t checksum = 0;
    size_t full_blocks = data.size() / 8;
    for (size_t i = 0; i < full_blocks; ++i) {
        uint64_t block = 0;
        std::memcpy(&block, data.data() + i * 8, 8);
        checksum ^= block;
    }
    size_t remaining = data.size() % 8;
    if (remaining > 0) {
        uint64_t block = 0;
        std::memcpy(&block, data.data() + full_blocks * 8, remaining);
        checksum ^= block;
    }
    return checksum;
}

static void write_flex_embedding(const FlexEmbedding& emb,
                                  std::vector<uint8_t>& buffer) {
    auto write_bytes = [&](const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), p, p + size);
    };
    write_bytes(emb.core.data(), CORE_DIM * sizeof(double));
    uint16_t detail_dim = static_cast<uint16_t>(emb.detail.size());
    write_bytes(&detail_dim, sizeof(detail_dim));
    if (detail_dim > 0) {
        write_bytes(emb.detail.data(), detail_dim * sizeof(double));
    }
}

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

// Initialize FlexKAN identity coefficients into a flat array at given offset
static void init_flexkan_identity(std::array<double, CM_FLAT_SIZE>& flat, size_t kan_offset) {
    auto safe_logit = [](double p) -> double {
        p = std::max(0.01, std::min(0.99, p));
        return std::log(p / (1.0 - p));
    };

    // FlexKAN constants
    constexpr size_t HIDDEN_DIM = 4;
    constexpr size_t NUM_KNOTS = 10;
    constexpr size_t LAYER0_EDGES = 6 * HIDDEN_DIM; // 24

    // Layer 0: edge from input_4 to hidden_0
    size_t l0_edge = kan_offset + (4 * HIDDEN_DIM + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        flat[l0_edge + k] = safe_logit(x);
    }

    // Layer 1: edge from hidden_0 to output
    size_t l1_edge = kan_offset + (LAYER0_EDGES + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        flat[l1_edge + k] = safe_logit(x);
    }
}

// Migrate v3 (940 doubles) to v7 (9772 doubles):
// Copy bilinear core, zero multihead, FlexKAN identity, default patterns,
// zero convergence port + gate.
static void migrate_v3_to_v7(const double* v3_data,
                              std::array<double, CM_FLAT_SIZE>& v7_flat) {
    v7_flat.fill(0.0);

    // Copy bilinear core (940 doubles at offset 0)
    for (size_t i = 0; i < V3_FLAT_SIZE; ++i) {
        v7_flat[i] = v3_data[i];
    }

    // MultiHeadBilinear (offsets 940..1579): zeros (already filled)

    // FlexKAN (offsets 1580..1859): identity init
    init_flexkan_identity(v7_flat, 1580);

    // Pattern weights (offsets 1860..1874): defaults
    v7_flat[1860] = 1.0;   // shared_parent
    v7_flat[1861] = 1.0;   // transitive_causation
    v7_flat[1862] = 1.0;   // missing_link
    v7_flat[1863] = 1.0;   // weak_strengthening
    v7_flat[1864] = 1.0;   // contradictory_signal
    v7_flat[1865] = 0.85;  // chain_hypothesis

    // Reserved (offsets 1875..1899): already zero
    // ConvergencePort (offsets 1900..5835): already zero (safe: tanh(0)=0)
    // Gate weights (offsets 5836..9771): already zero (sigmoid(0)=0.5 = neutral)
}

// Migrate v4 (1300 doubles) to v7 (9772 doubles):
// Copy bilinear core (940), skip old EmbeddedKAN (288), copy patterns (15),
// zero multihead, FlexKAN identity, zero reserved, zero convergence port + gate.
static void migrate_v4_to_v7(const double* v4_data,
                              std::array<double, CM_FLAT_SIZE>& v7_flat) {
    v7_flat.fill(0.0);

    // Copy bilinear core (940 doubles at offset 0)
    for (size_t i = 0; i < V3_FLAT_SIZE; ++i) {
        v7_flat[i] = v4_data[i];
    }

    // Skip old EmbeddedKAN params at v4 offsets 940..1227 (288 doubles)

    // MultiHeadBilinear (offsets 940..1579): zeros (already filled)

    // FlexKAN (offsets 1580..1859): identity init
    init_flexkan_identity(v7_flat, 1580);

    // Copy pattern weights from v4 offsets 1228..1242 to v7 offsets 1860..1874
    size_t v4_pat_offset = V3_FLAT_SIZE + 288;  // 940 + 288 = 1228
    for (size_t i = 0; i < 15; ++i) {
        v7_flat[1860 + i] = v4_data[v4_pat_offset + i];
    }

    // Reserved (offsets 1875..1899): already zero
    // ConvergencePort (offsets 1900..5835): already zero (safe: tanh(0)=0)
    // Gate weights (offsets 5836..9771): already zero (sigmoid(0)=0.5 = neutral)
}

// Migrate v5 (1900 doubles) to v7 (9772 doubles):
// Copy all 1900 doubles, zero convergence port + gate.
static void migrate_v5_to_v7(const double* v5_data,
                              std::array<double, CM_FLAT_SIZE>& v7_flat) {
    v7_flat.fill(0.0);
    for (size_t i = 0; i < V5_FLAT_SIZE; ++i) {
        v7_flat[i] = v5_data[i];
    }
    // ConvergencePort (offsets 1900..5835): already zero (safe: tanh(0)=0)
    // Gate weights (offsets 5836..9771): already zero (sigmoid(0)=0.5 = neutral)
}

// Migrate v6 (5836 doubles) to v7 (9772 doubles):
// Copy all 5836 doubles, zero-fill gate weights.
// sigmoid(0)=0.5 → half update, half retain = safe neutral behavior.
static void migrate_v6_to_v7(const double* v6_data,
                              std::array<double, CM_FLAT_SIZE>& v7_flat) {
    v7_flat.fill(0.0);
    for (size_t i = 0; i < V6_FLAT_SIZE; ++i) {
        v7_flat[i] = v6_data[i];
    }
    // Gate weights (offsets 5836..9771): already zero (sigmoid(0)=0.5 = neutral)
}

// =============================================================================
// Save v5 (function name kept as save_v4 for API compatibility)
// =============================================================================

bool save_v4(const std::string& filepath,
             const ConceptModelRegistry& registry,
             const EmbeddingManager& embeddings) {

    std::vector<uint8_t> buffer;

    auto write_bytes = [&](const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), p, p + size);
    };

    auto model_ids = registry.get_model_ids();
    auto context_names = embeddings.get_context_names();

    // Header
    write_bytes(MAGIC, 4);
    uint32_t version = VERSION_V8;
    write_bytes(&version, 4);
    uint64_t model_count = model_ids.size();
    write_bytes(&model_count, 8);
    uint64_t context_count = context_names.size();
    write_bytes(&context_count, 8);
    uint64_t reserved = 0;
    write_bytes(&reserved, 8);

    // Models
    std::array<double, CM_FLAT_SIZE> flat;
    for (ConceptId cid : model_ids) {
        const ConceptModel* model = registry.get_model(cid);
        if (!model) continue;

        uint64_t id_val = cid;
        write_bytes(&id_val, 8);
        model->to_flat(flat);
        write_bytes(flat.data(), CM_FLAT_SIZE * sizeof(double));
    }

    // Relation embeddings
    auto& reg = RelationTypeRegistry::instance();
    auto all_types = reg.all_types();
    uint32_t rel_emb_count = static_cast<uint32_t>(all_types.size());
    write_bytes(&rel_emb_count, 4);

    for (RelationType rt : all_types) {
        uint16_t type_val = static_cast<uint16_t>(rt);
        write_bytes(&type_val, 2);
        write_flex_embedding(reg.get_embedding(rt), buffer);
    }

    // Concept embeddings
    const auto& concept_emb_data = embeddings.concept_embeddings().data();
    uint32_t concept_emb_count = static_cast<uint32_t>(concept_emb_data.size());
    write_bytes(&concept_emb_count, 4);
    for (const auto& [cid, emb] : concept_emb_data) {
        uint64_t id_val = cid;
        write_bytes(&id_val, 8);
        write_flex_embedding(emb, buffer);
    }

    // Context embeddings
    const auto& ctx_map = embeddings.context_embeddings();
    for (const auto& name : context_names) {
        auto it = ctx_map.find(name);
        if (it == ctx_map.end()) continue;

        uint32_t name_len = static_cast<uint32_t>(name.size());
        write_bytes(&name_len, 4);
        write_bytes(name.data(), name_len);
        write_flex_embedding(it->second, buffer);
    }

    // Checksum
    uint64_t checksum = compute_checksum(buffer);
    write_bytes(&checksum, 8);

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
    return out.good();
}

// =============================================================================
// Load v5 (with v3/v4 backward compatibility)
// =============================================================================

bool load_v4(const std::string& filepath,
             ConceptModelRegistry& registry,
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

    // Verify checksum
    if (buffer.size() < 8) return false;
    size_t data_size = buffer.size() - 8;
    std::vector<uint8_t> data_portion(buffer.begin(), buffer.begin() + data_size);
    uint64_t expected_checksum = compute_checksum(data_portion);
    uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, buffer.data() + data_size, 8);
    if (expected_checksum != stored_checksum) return false;

    // Header
    char magic[4];
    if (!read_bytes(magic, 4)) return false;
    if (std::memcmp(magic, MAGIC, 4) != 0) return false;

    uint32_t version = 0;
    if (!read_bytes(&version, 4)) return false;
    if (version != VERSION_V8 && version != VERSION_V7 && version != VERSION_V6 &&
        version != VERSION_V5 && version != VERSION_V4 && version != VERSION_V3) return false;

    uint64_t model_count = 0;
    if (!read_bytes(&model_count, 8)) return false;

    uint64_t context_count = 0;
    if (!read_bytes(&context_count, 8)) return false;

    uint64_t reserved = 0;
    if (!read_bytes(&reserved, 8)) return false;

    // Models
    registry.clear();
    if (version == VERSION_V8) {
        // v8: 9933 doubles — V7 + ContextSuperposition (161 params)
        std::array<double, CM_FLAT_SIZE> flat;
        for (uint64_t i = 0; i < model_count; ++i) {
            uint64_t cid = 0;
            if (!read_bytes(&cid, 8)) return false;
            if (!read_bytes(flat.data(), CM_FLAT_SIZE * sizeof(double))) return false;

            registry.create_model(static_cast<ConceptId>(cid));
            ConceptModel* model = registry.get_model(static_cast<ConceptId>(cid));
            if (model) {
                model->from_flat(flat);
            }
        }
    } else if (version == VERSION_V7) {
        // v7: 9772 doubles -> migrate to 9933 (zero superposition = disabled)
        std::array<double, V7_FLAT_SIZE> v7_flat;
        std::array<double, CM_FLAT_SIZE> v8_flat;
        for (uint64_t i = 0; i < model_count; ++i) {
            uint64_t cid = 0;
            if (!read_bytes(&cid, 8)) return false;
            if (!read_bytes(v7_flat.data(), V7_FLAT_SIZE * sizeof(double))) return false;

            v8_flat.fill(0.0);
            for (size_t j = 0; j < V7_FLAT_SIZE; ++j)
                v8_flat[j] = v7_flat[j];
            // Superposition params (offsets 9772..9932) already zero = disabled

            registry.create_model(static_cast<ConceptId>(cid));
            ConceptModel* model = registry.get_model(static_cast<ConceptId>(cid));
            if (model) {
                model->from_flat(v8_flat);
            }
        }
    } else if (version == VERSION_V6) {
        // v6: 5836 doubles -> migrate to 9772 (zero gate weights)
        std::array<double, V6_FLAT_SIZE> v6_flat;
        std::array<double, CM_FLAT_SIZE> v7_flat;
        for (uint64_t i = 0; i < model_count; ++i) {
            uint64_t cid = 0;
            if (!read_bytes(&cid, 8)) return false;
            if (!read_bytes(v6_flat.data(), V6_FLAT_SIZE * sizeof(double))) return false;

            migrate_v6_to_v7(v6_flat.data(), v7_flat);
            registry.create_model(static_cast<ConceptId>(cid));
            ConceptModel* model = registry.get_model(static_cast<ConceptId>(cid));
            if (model) {
                model->from_flat(v7_flat);
            }
        }
    } else if (version == VERSION_V5) {
        // v5: 1900 doubles -> migrate to 9772
        std::array<double, V5_FLAT_SIZE> v5_flat;
        std::array<double, CM_FLAT_SIZE> v7_flat;
        for (uint64_t i = 0; i < model_count; ++i) {
            uint64_t cid = 0;
            if (!read_bytes(&cid, 8)) return false;
            if (!read_bytes(v5_flat.data(), V5_FLAT_SIZE * sizeof(double))) return false;

            migrate_v5_to_v7(v5_flat.data(), v7_flat);
            registry.create_model(static_cast<ConceptId>(cid));
            ConceptModel* model = registry.get_model(static_cast<ConceptId>(cid));
            if (model) {
                model->from_flat(v7_flat);
            }
        }
    } else if (version == VERSION_V4) {
        // v4: 1300 doubles -> migrate to 9772
        std::array<double, V4_FLAT_SIZE> v4_flat;
        std::array<double, CM_FLAT_SIZE> v7_flat;
        for (uint64_t i = 0; i < model_count; ++i) {
            uint64_t cid = 0;
            if (!read_bytes(&cid, 8)) return false;
            if (!read_bytes(v4_flat.data(), V4_FLAT_SIZE * sizeof(double))) return false;

            migrate_v4_to_v7(v4_flat.data(), v7_flat);
            registry.create_model(static_cast<ConceptId>(cid));
            ConceptModel* model = registry.get_model(static_cast<ConceptId>(cid));
            if (model) {
                model->from_flat(v7_flat);
            }
        }
    } else {
        // v3: 940 doubles -> migrate to 9772
        std::array<double, V3_FLAT_SIZE> v3_flat;
        std::array<double, CM_FLAT_SIZE> v7_flat;
        for (uint64_t i = 0; i < model_count; ++i) {
            uint64_t cid = 0;
            if (!read_bytes(&cid, 8)) return false;
            if (!read_bytes(v3_flat.data(), V3_FLAT_SIZE * sizeof(double))) return false;

            migrate_v3_to_v7(v3_flat.data(), v7_flat);
            registry.create_model(static_cast<ConceptId>(cid));
            ConceptModel* model = registry.get_model(static_cast<ConceptId>(cid));
            if (model) {
                model->from_flat(v7_flat);
            }
        }
    }

    // Relation embeddings (v3/v4/v5 same format)
    uint32_t rel_emb_count = 0;
    if (!read_bytes(&rel_emb_count, 4)) return false;
    for (uint32_t i = 0; i < rel_emb_count; ++i) {
        uint16_t type_val = 0;
        if (!read_bytes(&type_val, 2)) return false;
        FlexEmbedding emb;
        if (!read_flex_embedding(emb, buffer, pos)) return false;
    }

    // Concept embeddings
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

    // Context embeddings
    auto& ctx_map = embeddings.context_embeddings_mut();
    ctx_map.clear();
    for (uint64_t i = 0; i < context_count; ++i) {
        uint32_t name_len = 0;
        if (!read_bytes(&name_len, 4)) return false;
        if (name_len > 1024) return false;

        std::string name(name_len, '\0');
        if (!read_bytes(name.data(), name_len)) return false;

        FlexEmbedding emb;
        if (!read_flex_embedding(emb, buffer, pos)) return false;

        ctx_map[name] = std::move(emb);
    }

    return true;
}

// =============================================================================
// Validate
// =============================================================================

bool validate_v4(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;

    auto file_size = in.tellg();
    if (file_size < static_cast<std::streamoff>(HEADER_SIZE + 8)) return false;

    in.seekg(0);
    std::vector<uint8_t> buffer(static_cast<size_t>(file_size));
    in.read(reinterpret_cast<char*>(buffer.data()), file_size);
    if (!in.good()) return false;

    if (std::memcmp(buffer.data(), MAGIC, 4) != 0) return false;

    uint32_t version = 0;
    std::memcpy(&version, buffer.data() + 4, 4);
    if (version != VERSION_V8 && version != VERSION_V7 && version != VERSION_V6 &&
        version != VERSION_V5 && version != VERSION_V4 && version != VERSION_V3) return false;

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
