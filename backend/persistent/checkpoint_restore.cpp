#include "checkpoint_restore.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace brain19 {
namespace persistent {

// =============================================================================
// Component parsing
// =============================================================================

Component parse_component(const std::string& name) {
    if (name == "ltm")       return Component::LTM;
    if (name == "stm")       return Component::STM;
    if (name == "models")    return Component::MODELS;
    if (name == "kan")       return Component::KAN;
    if (name == "cognitive") return Component::COGNITIVE;
    if (name == "config")    return Component::CONFIG;
    if (name == "all")       return Component::ALL;
    return Component::ALL;
}

uint8_t parse_components(const std::string& csv) {
    uint8_t mask = 0;
    std::istringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // trim
        while (!token.empty() && token.front() == ' ') token.erase(token.begin());
        while (!token.empty() && token.back() == ' ') token.pop_back();
        mask |= uint8_t(parse_component(token));
    }
    return mask;
}

// =============================================================================
// Read helpers
// =============================================================================

namespace {
template<typename T>
bool read_pod(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return f.good();
}

std::string read_file_string(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
} // anon

// =============================================================================
// Verify
// =============================================================================

VerifyResult CheckpointRestore::verify(const std::string& checkpoint_dir) {
    VerifyResult result;
    
    std::string manifest_path = checkpoint_dir + "/manifest.json";
    if (!fs::exists(manifest_path)) {
        result.failures.push_back("manifest.json missing");
        return result;
    }
    
    CheckpointManifest manifest;
    try {
        manifest = load_manifest(checkpoint_dir);
    } catch (...) {
        result.failures.push_back("manifest.json: failed to parse");
        return result;
    }
    
    if (manifest.components.empty()) {
        result.failures.push_back("manifest.json: no components listed (corrupt or empty)");
        return result;
    }
    
    for (auto& comp : manifest.components) {
        result.files_checked++;
        std::string fpath = checkpoint_dir + "/" + comp.filename;
        
        if (!fs::exists(fpath)) {
            result.failures.push_back(comp.filename + ": file missing");
            continue;
        }
        
        std::string hash = SHA256::hash_file(fpath);
        if (hash != comp.sha256) {
            result.failures.push_back(comp.filename + ": hash mismatch (expected " + 
                comp.sha256.substr(0, 16) + "... got " + hash.substr(0, 16) + "...)");
            continue;
        }
        
        uint64_t actual_size = fs::file_size(fpath);
        if (actual_size != comp.size_bytes) {
            result.failures.push_back(comp.filename + ": size mismatch");
            continue;
        }
        
        result.files_ok++;
    }
    
    result.valid = result.failures.empty();
    return result;
}

std::string VerifyResult::to_string() const {
    std::ostringstream o;
    if (valid) {
        o << "✓ Checkpoint valid (" << files_ok << "/" << files_checked << " files OK)\n";
    } else {
        o << "✗ Checkpoint INVALID\n";
        for (auto& f : failures) o << "  - " << f << "\n";
        o << "  " << files_ok << "/" << files_checked << " files OK\n";
    }
    return o.str();
}

// =============================================================================
// Load Manifest
// =============================================================================

CheckpointManifest CheckpointRestore::load_manifest(const std::string& checkpoint_dir) {
    std::string json = read_file_string(checkpoint_dir + "/manifest.json");
    return CheckpointManifest::from_json(json);
}

// =============================================================================
// Restore
// =============================================================================

RestoreResult CheckpointRestore::restore(
    const std::string& checkpoint_dir,
    uint8_t components,
    PersistentLTM*              ltm,
    STMSnapshotData*            stm_out,
    ConceptModelRegistry*         models,
    std::vector<std::pair<std::string, KANModule*>>* kan_modules,
    CognitiveState*             cognitive,
    CheckpointConfig*           config
) {
    RestoreResult result;
    
    // Verify first
    auto vr = verify(checkpoint_dir);
    if (!vr.valid) {
        result.error = "Integrity check failed";
        return result;
    }
    
    if (has_component(components, Component::LTM) && ltm) {
        std::string path = checkpoint_dir + "/ltm.bin";
        if (fs::exists(path)) {
            if (!restore_ltm(path, *ltm)) {
                result.error = "LTM restore failed";
                return result;
            }
        }
    }
    
    if (has_component(components, Component::STM) && stm_out) {
        std::string path = checkpoint_dir + "/stm.bin";
        if (fs::exists(path)) {
            if (!restore_stm(path, *stm_out)) {
                result.error = "STM restore failed";
                return result;
            }
        }
    }
    
    if (has_component(components, Component::MODELS) && models) {
        std::string path = checkpoint_dir + "/micromodels.bin";
        if (fs::exists(path)) {
            if (!restore_micromodels(path, *models)) {
                result.error = "MicroModel restore failed";
                return result;
            }
        }
    }
    
    if (has_component(components, Component::KAN) && kan_modules) {
        std::string path = checkpoint_dir + "/kan_modules.bin";
        if (fs::exists(path)) {
            if (!restore_kan_modules(path, *kan_modules)) {
                result.error = "KAN restore failed";
                return result;
            }
        }
    }
    
    if (has_component(components, Component::COGNITIVE) && cognitive) {
        std::string path = checkpoint_dir + "/cognitive.bin";
        if (fs::exists(path)) {
            if (!restore_cognitive(path, *cognitive)) {
                result.error = "Cognitive restore failed";
                return result;
            }
        }
    }
    
    if (has_component(components, Component::CONFIG) && config) {
        std::string path = checkpoint_dir + "/config.json";
        if (fs::exists(path)) {
            if (!restore_config(path, *config)) {
                result.error = "Config restore failed";
                return result;
            }
        }
    }
    
    auto manifest = load_manifest(checkpoint_dir);
    result.success = true;
    result.concepts_restored = manifest.concept_count;
    result.relations_restored = manifest.relation_count;
    result.models_restored = manifest.model_count;
    result.kan_modules_restored = manifest.kan_module_count;
    return result;
}

// =============================================================================
// Individual restore functions
// =============================================================================

bool CheckpointRestore::restore_ltm(const std::string& path, PersistentLTM& ltm) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    
    uint32_t magic; uint16_t version;
    if (!read_pod(f, magic) || magic != 0x4C544D42) return false;
    if (!read_pod(f, version) || version != 1) return false;
    
    uint64_t n;
    if (!read_pod(f, n)) return false;
    
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t id;
        if (!read_pod(f, id)) return false;
        
        uint32_t llen;
        if (!read_pod(f, llen)) return false;
        std::string label(llen, '\0');
        f.read(label.data(), llen);
        
        uint32_t dlen;
        if (!read_pod(f, dlen)) return false;
        std::string def(dlen, '\0');
        f.read(def.data(), dlen);
        
        uint8_t etype, estatus;
        double trust;
        if (!read_pod(f, etype)) return false;
        if (!read_pod(f, estatus)) return false;
        if (!read_pod(f, trust)) return false;
        
        // Only store if not already present
        if (!ltm.exists(id)) {
            EpistemicMetadata meta(
                static_cast<EpistemicType>(etype),
                static_cast<EpistemicStatus>(estatus),
                trust
            );
            ltm.store_concept(label, def, meta);
        }
    }
    
    // Relations
    uint64_t rel_count;
    if (!read_pod(f, rel_count)) return false;
    
    for (uint64_t i = 0; i < rel_count; ++i) {
        uint64_t rid, src, tgt;
        uint8_t t;
        double w;
        if (!read_pod(f, rid)) return false;
        if (!read_pod(f, src)) return false;
        if (!read_pod(f, tgt)) return false;
        if (!read_pod(f, t)) return false;
        if (!read_pod(f, w)) return false;
        
        if (ltm.exists(src) && ltm.exists(tgt)) {
            ltm.add_relation(src, tgt, static_cast<RelationType>(t), w);
        }
    }
    
    return true;
}

bool CheckpointRestore::restore_stm(const std::string& path, STMSnapshotData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    
    uint32_t magic; uint16_t version;
    if (!read_pod(f, magic) || magic != 0x53544D42) return false;
    if (!read_pod(f, version) || version != 1) return false;
    if (!read_pod(f, out.timestamp)) return false;
    
    uint32_t ctx_count;
    if (!read_pod(f, ctx_count)) return false;
    out.contexts.resize(ctx_count);
    
    for (uint32_t i = 0; i < ctx_count; ++i) {
        auto& ctx = out.contexts[i];
        if (!read_pod(f, ctx.context_id)) return false;
        uint32_t cc, rc;
        if (!read_pod(f, cc)) return false;
        if (!read_pod(f, rc)) return false;
        
        ctx.concepts.resize(cc);
        for (uint32_t j = 0; j < cc; ++j) {
            if (!read_pod(f, ctx.concepts[j].concept_id)) return false;
            if (!read_pod(f, ctx.concepts[j].activation)) return false;
            uint8_t cls;
            if (!read_pod(f, cls)) return false;
            ctx.concepts[j].classification = static_cast<ActivationClass>(cls);
        }
        
        ctx.relations.resize(rc);
        for (uint32_t j = 0; j < rc; ++j) {
            if (!read_pod(f, ctx.relations[j].source)) return false;
            if (!read_pod(f, ctx.relations[j].target)) return false;
            uint8_t t;
            if (!read_pod(f, t)) return false;
            ctx.relations[j].type = static_cast<RelationType>(t);
            if (!read_pod(f, ctx.relations[j].activation)) return false;
        }
    }
    
    return true;
}

bool CheckpointRestore::restore_micromodels(const std::string& path, ConceptModelRegistry& reg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic; uint16_t version;
    if (!read_pod(f, magic) || magic != 0x4D4D4442) return false;
    // Accept v1 (legacy 940), v2 (ConceptModel 1300), v3 (ConceptModel 1900), v4 (5836), v5 (9772)
    if (!read_pod(f, version) || (version < 1 || version > 5)) return false;

    uint64_t n;
    if (!read_pod(f, n)) return false;

    // Helper: initialize FlexKAN identity coefficients into flat array
    auto init_flexkan_identity = [](std::array<double, CM_FLAT_SIZE>& flat, size_t kan_offset) {
        auto safe_logit = [](double p) -> double {
            p = std::max(0.01, std::min(0.99, p));
            return std::log(p / (1.0 - p));
        };
        constexpr size_t HIDDEN_DIM = 4;
        constexpr size_t NUM_KNOTS = 10;
        constexpr size_t LAYER0_EDGES = 6 * HIDDEN_DIM;
        size_t l0_edge = kan_offset + (4 * HIDDEN_DIM + 0) * NUM_KNOTS;
        for (size_t k = 0; k < NUM_KNOTS; ++k) {
            double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
            flat[l0_edge + k] = safe_logit(x);
        }
        size_t l1_edge = kan_offset + (LAYER0_EDGES + 0) * NUM_KNOTS;
        for (size_t k = 0; k < NUM_KNOTS; ++k) {
            double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
            flat[l1_edge + k] = safe_logit(x);
        }
    };

    for (uint64_t i = 0; i < n; ++i) {
        uint64_t id;
        if (!read_pod(f, id)) return false;

        if (version == 5) {
            // v5: 9772 doubles — current format with ConvergencePort + gate
            std::array<double, CM_FLAT_SIZE> flat;
            f.read(reinterpret_cast<char*>(flat.data()), sizeof(flat));
            if (!f.good()) return false;
            reg.create_model(id);
            ConceptModel* model = reg.get_model(id);
            if (model) model->from_flat(flat);
        } else if (version == 4) {
            // v4: 5836 doubles — migrate to 9772 (zero gate weights)
            std::array<double, CM_FLAT_SIZE_V6> old_flat;
            f.read(reinterpret_cast<char*>(old_flat.data()), sizeof(old_flat));
            if (!f.good()) return false;
            std::array<double, CM_FLAT_SIZE> flat{};
            std::copy(old_flat.begin(), old_flat.end(), flat.begin());
            // Gate weights (5836..9771): already zero (sigmoid(0)=0.5 = neutral)
            reg.create_model(id);
            ConceptModel* model = reg.get_model(id);
            if (model) model->from_flat(flat);
        } else if (version == 3) {
            // v3: 1900 doubles — migrate to 9772
            std::array<double, CM_FLAT_SIZE_V5> old_flat;
            f.read(reinterpret_cast<char*>(old_flat.data()), sizeof(old_flat));
            if (!f.good()) return false;
            std::array<double, CM_FLAT_SIZE> flat{};
            std::copy(old_flat.begin(), old_flat.end(), flat.begin());
            // ConvergencePort + gate (1900..9771): already zero
            reg.create_model(id);
            ConceptModel* model = reg.get_model(id);
            if (model) model->from_flat(flat);
        } else if (version == 2) {
            // v2: 1300 doubles — migrate to 9772
            constexpr size_t V2_SIZE = 1300;
            std::array<double, V2_SIZE> old_flat;
            f.read(reinterpret_cast<char*>(old_flat.data()), sizeof(old_flat));
            if (!f.good()) return false;
            std::array<double, CM_FLAT_SIZE> flat{};
            // Copy bilinear core (940)
            for (size_t j = 0; j < 940; ++j) flat[j] = old_flat[j];
            // Skip old EmbeddedKAN (288 doubles at offsets 940..1227)
            // MultiHeadBilinear at 940..1579: zeros
            // FlexKAN at 1580..1859: identity init
            init_flexkan_identity(flat, 1580);
            // Copy pattern weights from old offsets 1228..1242 to 1860..1874
            for (size_t j = 0; j < 15; ++j) flat[1860 + j] = old_flat[1228 + j];
            // ConvergencePort + gate (1900..9771): already zero
            reg.create_model(id);
            ConceptModel* model = reg.get_model(id);
            if (model) model->from_flat(flat);
        } else {
            // v1: legacy 940 doubles — migrate to 9772
            std::array<double, FLAT_SIZE> old_flat;
            f.read(reinterpret_cast<char*>(old_flat.data()), sizeof(old_flat));
            if (!f.good()) return false;
            std::array<double, CM_FLAT_SIZE> flat{};
            std::copy(old_flat.begin(), old_flat.end(), flat.begin());
            // FlexKAN identity init at 1580
            init_flexkan_identity(flat, 1580);
            // Default pattern weights at 1860
            flat[1860] = 1.0; flat[1861] = 1.0; flat[1862] = 1.0;
            flat[1863] = 1.0; flat[1864] = 1.0; flat[1865] = 0.85;
            // ConvergencePort + gate (1900..9771): already zero
            reg.create_model(id);
            ConceptModel* model = reg.get_model(id);
            if (model) model->from_flat(flat);
        }
    }

    return true;
}

bool CheckpointRestore::restore_kan_modules(const std::string& path,
    std::vector<std::pair<std::string, KANModule*>>& modules)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    
    uint32_t magic; uint16_t version;
    if (!read_pod(f, magic) || magic != 0x4B414E42) return false;
    if (!read_pod(f, version) || version != 1) return false;
    
    uint64_t n;
    if (!read_pod(f, n)) return false;
    
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t nlen;
        if (!read_pod(f, nlen)) return false;
        std::string name(nlen, '\0');
        f.read(name.data(), nlen);
        
        uint32_t num_dims;
        if (!read_pod(f, num_dims)) return false;
        std::vector<size_t> topo(num_dims);
        for (uint32_t d = 0; d < num_dims; ++d) {
            uint64_t dim;
            if (!read_pod(f, dim)) return false;
            topo[d] = dim;
        }
        
        // Find matching module by name
        KANModule* target = nullptr;
        for (auto& [mname, mod] : modules) {
            if (mname == name) { target = mod; break; }
        }
        
        // Read layers regardless (to advance file position)
        uint32_t nlayers;
        if (!read_pod(f, nlayers)) return false;
        
        for (uint32_t l = 0; l < nlayers; ++l) {
            uint32_t nnodes;
            if (!read_pod(f, nnodes)) return false;
            for (uint32_t ni = 0; ni < nnodes; ++ni) {
                uint32_t nc;
                if (!read_pod(f, nc)) return false;
                std::vector<double> coefs(nc);
                f.read(reinterpret_cast<char*>(coefs.data()), nc * sizeof(double));
                
                if (target && l < target->num_layers()) {
                    auto& layer = target->layer_mutable(l);
                    if (ni < layer.num_nodes()) {
                        layer.get_nodes_mutable()[ni]->set_coefficients(coefs);
                    }
                }
            }
        }
    }
    
    return true;
}

bool CheckpointRestore::restore_cognitive(const std::string& path, CognitiveState& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    
    uint32_t magic; uint16_t version;
    if (!read_pod(f, magic) || magic != 0x434F4742) return false;
    if (!read_pod(f, version) || version != 1) return false;
    
    uint64_t n;
    if (!read_pod(f, n)) return false;
    out.focus_set.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
        if (!read_pod(f, out.focus_set[i])) return false;
    }
    if (!read_pod(f, out.avg_activation)) return false;
    if (!read_pod(f, out.tick_count)) return false;
    if (!read_pod(f, out.epoch_ms)) return false;
    
    return true;
}

bool CheckpointRestore::restore_config(const std::string& path, CheckpointConfig& out) {
    std::string json = read_file_string(path);
    if (json.empty()) return false;
    out = CheckpointConfig::from_json(json);
    return true;
}

// =============================================================================
// Diff
// =============================================================================

DiffResult CheckpointRestore::diff(const std::string& dir_a, const std::string& dir_b) {
    DiffResult result;
    result.checkpoint_a = dir_a;
    result.checkpoint_b = dir_b;
    
    auto ma = load_manifest(dir_a);
    auto mb = load_manifest(dir_b);
    
    result.concept_count_diff  = int64_t(mb.concept_count)  - int64_t(ma.concept_count);
    result.relation_count_diff = int64_t(mb.relation_count) - int64_t(ma.relation_count);
    result.model_count_diff    = int64_t(mb.model_count)    - int64_t(ma.model_count);
    result.kan_module_diff     = int64_t(mb.kan_module_count) - int64_t(ma.kan_module_count);
    result.epoch_ms_diff       = int64_t(mb.epoch_ms)       - int64_t(ma.epoch_ms);
    
    // Build hash maps
    std::unordered_map<std::string, std::string> hashes_a, hashes_b;
    for (auto& c : ma.components) hashes_a[c.filename] = c.sha256;
    for (auto& c : mb.components) hashes_b[c.filename] = c.sha256;
    
    std::set<std::string> all_files;
    for (auto& [k,_] : hashes_a) all_files.insert(k);
    for (auto& [k,_] : hashes_b) all_files.insert(k);
    
    for (auto& f : all_files) {
        auto ia = hashes_a.find(f);
        auto ib = hashes_b.find(f);
        if (ia == hashes_a.end() || ib == hashes_b.end() || ia->second != ib->second) {
            result.changed_files.push_back(f);
        }
    }
    
    return result;
}

std::string DiffResult::to_string() const {
    std::ostringstream o;
    o << "Diff: " << checkpoint_a << " → " << checkpoint_b << "\n";
    auto fmt = [](int64_t v) -> std::string {
        if (v > 0) return "+" + std::to_string(v);
        return std::to_string(v);
    };
    o << "  Concepts:    " << fmt(concept_count_diff)  << "\n";
    o << "  Relations:   " << fmt(relation_count_diff) << "\n";
    o << "  Models:      " << fmt(model_count_diff)    << "\n";
    o << "  KAN Modules: " << fmt(kan_module_diff)     << "\n";
    o << "  Time delta:  " << (epoch_ms_diff / 1000)   << "s\n";
    if (!changed_files.empty()) {
        o << "  Changed files:\n";
        for (auto& f : changed_files) o << "    - " << f << "\n";
    }
    return o.str();
}

// =============================================================================
// List
// =============================================================================

std::vector<CheckpointRestore::ListEntry> CheckpointRestore::list(const std::string& base_dir) {
    std::vector<ListEntry> result;
    if (!fs::exists(base_dir)) return result;
    
    for (auto& entry : fs::directory_iterator(base_dir)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        if (name.rfind("checkpoint_", 0) != 0) continue;
        
        std::string mpath = entry.path().string() + "/manifest.json";
        if (!fs::exists(mpath)) continue;
        
        try {
            auto m = load_manifest(entry.path().string());
            ListEntry le;
            le.path = entry.path().string();
            le.timestamp = m.timestamp;
            le.tag = m.tag;
            le.epoch_ms = m.epoch_ms;
            le.concept_count = m.concept_count;
            le.relation_count = m.relation_count;
            le.model_count = m.model_count;
            result.push_back(le);
        } catch (...) {
            // Skip corrupted manifests
        }
    }
    
    // Sort newest first
    std::sort(result.begin(), result.end(), [](const ListEntry& a, const ListEntry& b) {
        return a.epoch_ms > b.epoch_ms;
    });
    
    return result;
}

} // namespace persistent
} // namespace brain19
