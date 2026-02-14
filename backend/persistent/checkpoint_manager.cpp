#include "checkpoint_manager.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

namespace brain19 {
namespace persistent {

// =============================================================================
// SHA-256 Implementation (FIPS 180-4, compact)
// =============================================================================

namespace {
constexpr uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sig0(uint32_t x) { return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22); }
inline uint32_t sig1(uint32_t x) { return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25); }
inline uint32_t gam0(uint32_t x) { return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }
inline uint32_t gam1(uint32_t x) { return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }

std::string hex_string(const uint8_t* data, size_t len) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out += hex[data[i] >> 4];
        out += hex[data[i] & 0x0f];
    }
    return out;
}
} // anon

void SHA256::init() {
    state_[0] = 0x6a09e667; state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372; state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f; state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab; state_[7] = 0x5be0cd19;
    bitcount_ = 0;
    buflen_ = 0;
}

void SHA256::transform(const uint8_t block[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; ++i)
        W[i] = (uint32_t(block[4*i]) << 24) | (uint32_t(block[4*i+1]) << 16)
              | (uint32_t(block[4*i+2]) << 8) | uint32_t(block[4*i+3]);
    for (int i = 16; i < 64; ++i)
        W[i] = gam1(W[i-2]) + W[i-7] + gam0(W[i-15]) + W[i-16];

    uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
    uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + sig1(e) + ch(e,f,g) + K[i] + W[i];
        uint32_t t2 = sig0(a) + maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
    state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
}

void SHA256::update(const uint8_t* data, size_t len) {
    bitcount_ += len * 8;
    while (len > 0) {
        size_t space = 64 - buflen_;
        size_t copy = std::min(len, space);
        std::memcpy(buffer_ + buflen_, data, copy);
        buflen_ += copy;
        data += copy;
        len -= copy;
        if (buflen_ == 64) {
            transform(buffer_);
            buflen_ = 0;
        }
    }
}

std::string SHA256::finalize() {
    // Save original bit count before padding adds to it
    uint64_t total_bits = bitcount_;
    
    uint8_t pad[64];
    size_t padlen = (buflen_ < 56) ? (56 - buflen_) : (120 - buflen_);
    std::memset(pad, 0, padlen);
    pad[0] = 0x80;
    update(pad, padlen);

    uint8_t bits[8];
    for (int i = 7; i >= 0; --i) { bits[i] = uint8_t(total_bits); total_bits >>= 8; }
    // Bypass update's bitcount increment for the length field
    // We need to write directly into buffer and transform
    std::memcpy(buffer_ + buflen_, bits, 8);
    transform(buffer_);

    uint8_t hash[32];
    for (int i = 0; i < 8; ++i) {
        hash[4*i]   = uint8_t(state_[i] >> 24);
        hash[4*i+1] = uint8_t(state_[i] >> 16);
        hash[4*i+2] = uint8_t(state_[i] >> 8);
        hash[4*i+3] = uint8_t(state_[i]);
    }
    return hex_string(hash, 32);
}

std::string SHA256::hash_bytes(const uint8_t* data, size_t len) {
    SHA256 h;
    h.init();
    h.update(data, len);
    return h.finalize();
}

std::string SHA256::hash_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("SHA256: cannot open " + path);
    SHA256 h;
    h.init();
    uint8_t buf[8192];
    while (f) {
        f.read(reinterpret_cast<char*>(buf), sizeof(buf));
        auto n = f.gcount();
        if (n > 0) h.update(buf, static_cast<size_t>(n));
    }
    return h.finalize();
}

// =============================================================================
// Simple JSON helpers (no external deps)
// =============================================================================

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;
        }
    }
    return out;
}

// Minimal JSON string value extractor: find "key": "value"
std::string json_get_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos = json.find('"', pos + needle.size() + 1); // skip colon, find opening "
    if (pos == std::string::npos) return {};
    ++pos;
    auto end = json.find('"', pos);
    if (end == std::string::npos) return {};
    return json.substr(pos, end - pos);
}

uint64_t json_get_uint64(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return 0;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return 0;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    return std::stoull(json.substr(pos));
}

uint32_t json_get_uint32(const std::string& json, const std::string& key) {
    return static_cast<uint32_t>(json_get_uint64(json, key));
}

} // anon

// =============================================================================
// CheckpointManifest JSON
// =============================================================================

std::string CheckpointManifest::to_json() const {
    std::ostringstream o;
    o << "{\n";
    o << "  \"format_version\": " << format_version << ",\n";
    o << "  \"timestamp\": \"" << json_escape(timestamp) << "\",\n";
    o << "  \"epoch_ms\": " << epoch_ms << ",\n";
    o << "  \"tag\": \"" << json_escape(tag) << "\",\n";
    o << "  \"concept_count\": " << concept_count << ",\n";
    o << "  \"relation_count\": " << relation_count << ",\n";
    o << "  \"model_count\": " << model_count << ",\n";
    o << "  \"kan_module_count\": " << kan_module_count << ",\n";
    o << "  \"components\": [\n";
    for (size_t i = 0; i < components.size(); ++i) {
        auto& c = components[i];
        o << "    {\"filename\": \"" << json_escape(c.filename) 
          << "\", \"sha256\": \"" << c.sha256 
          << "\", \"size_bytes\": " << c.size_bytes << "}";
        if (i + 1 < components.size()) o << ",";
        o << "\n";
    }
    o << "  ]\n";
    o << "}\n";
    return o.str();
}

CheckpointManifest CheckpointManifest::from_json(const std::string& json) {
    CheckpointManifest m;
    m.format_version = json_get_uint32(json, "format_version");
    m.timestamp = json_get_string(json, "timestamp");
    m.epoch_ms = json_get_uint64(json, "epoch_ms");
    m.tag = json_get_string(json, "tag");
    m.concept_count = json_get_uint64(json, "concept_count");
    m.relation_count = json_get_uint64(json, "relation_count");
    m.model_count = json_get_uint64(json, "model_count");
    m.kan_module_count = json_get_uint64(json, "kan_module_count");
    
    // Parse components array
    auto arr_pos = json.find("\"components\"");
    if (arr_pos != std::string::npos) {
        auto start = json.find('[', arr_pos);
        auto end = json.find(']', start);
        if (start != std::string::npos && end != std::string::npos) {
            std::string arr = json.substr(start, end - start + 1);
            // Find each {...} object
            size_t p = 0;
            while (true) {
                auto ob = arr.find('{', p);
                if (ob == std::string::npos) break;
                auto cb = arr.find('}', ob);
                if (cb == std::string::npos) break;
                std::string obj = arr.substr(ob, cb - ob + 1);
                ComponentHash ch;
                ch.filename = json_get_string(obj, "filename");
                ch.sha256 = json_get_string(obj, "sha256");
                ch.size_bytes = json_get_uint64(obj, "size_bytes");
                m.components.push_back(ch);
                p = cb + 1;
            }
        }
    }
    return m;
}

// =============================================================================
// CheckpointConfig JSON
// =============================================================================

std::string CheckpointConfig::to_json() const {
    std::ostringstream o;
    o << "{\n";
    size_t i = 0;
    for (auto& [k, v] : entries) {
        o << "  \"" << json_escape(k) << "\": \"" << json_escape(v) << "\"";
        if (++i < entries.size()) o << ",";
        o << "\n";
    }
    o << "}\n";
    return o.str();
}

CheckpointConfig CheckpointConfig::from_json(const std::string& json) {
    CheckpointConfig cfg;
    // Simple parser: find all "key": "value" pairs at top level
    size_t pos = 0;
    while (true) {
        auto q1 = json.find('"', pos);
        if (q1 == std::string::npos) break;
        auto q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string key = json.substr(q1 + 1, q2 - q1 - 1);
        auto colon = json.find(':', q2 + 1);
        if (colon == std::string::npos) break;
        auto q3 = json.find('"', colon + 1);
        if (q3 == std::string::npos) break;
        auto q4 = json.find('"', q3 + 1);
        if (q4 == std::string::npos) break;
        std::string val = json.substr(q3 + 1, q4 - q3 - 1);
        cfg.entries[key] = val;
        pos = q4 + 1;
    }
    return cfg;
}

// =============================================================================
// CheckpointManager
// =============================================================================

CheckpointManager::CheckpointManager(const Options& opts)
    : opts_(opts)
{}

std::string CheckpointManager::now_iso8601() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

uint64_t CheckpointManager::now_epoch_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );
}

std::string CheckpointManager::make_timestamp_dirname(const std::string& tag) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "checkpoint_%Y%m%d_%H%M%S", &tm);
    std::string name(buf);
    if (!tag.empty()) {
        // Sanitize tag: remove path separators to prevent directory traversal
        std::string safe_tag;
        for (char c : tag) {
            if (c != '/' && c != '\\' && c != '\0') safe_tag += c;
        }
        if (!safe_tag.empty()) name += "_" + safe_tag;
    }
    return name;
}

std::string CheckpointManager::save(
    PersistentLTM*              ltm,
    const STMSnapshotData*      stm_data,
    const ConceptModelRegistry*   models,
    const std::vector<std::pair<std::string, KANModule*>>* kan_modules,
    const CognitiveState*       cognitive,
    const CheckpointConfig*     config
) {
    fs::create_directories(opts_.base_dir);
    
    std::string dirname = make_timestamp_dirname(opts_.tag);
    std::string final_path = opts_.base_dir + "/" + dirname;
    std::string temp_path = opts_.base_dir + "/." + dirname + ".tmp";
    
    // Clean up temp if exists from a previous failed attempt
    if (fs::exists(temp_path)) {
        fs::remove_all(temp_path);
    }
    
    fs::create_directories(temp_path);
    
    CheckpointManifest manifest;
    manifest.format_version = 1;
    manifest.timestamp = now_iso8601();
    manifest.epoch_ms = now_epoch_ms();
    manifest.tag = opts_.tag;
    
    try {
        if (ltm) {
            auto ch = write_ltm(temp_path, *ltm);
            manifest.components.push_back(ch);
            manifest.concept_count = ltm->concept_count();
            manifest.relation_count = ltm->relation_count();
        }
        
        if (stm_data) {
            auto ch = write_stm(temp_path, *stm_data);
            manifest.components.push_back(ch);
        }
        
        if (models) {
            auto ch = write_micromodels(temp_path, *models);
            manifest.components.push_back(ch);
            manifest.model_count = models->size();
        }
        
        if (kan_modules) {
            auto ch = write_kan_modules(temp_path, *kan_modules);
            manifest.components.push_back(ch);
            manifest.kan_module_count = kan_modules->size();
        }
        
        if (cognitive) {
            auto ch = write_cognitive(temp_path, *cognitive);
            manifest.components.push_back(ch);
        }
        
        if (config) {
            auto ch = write_config(temp_path, *config);
            manifest.components.push_back(ch);
        }
        
        // Write manifest
        {
            std::string mpath = temp_path + "/manifest.json";
            std::ofstream f(mpath);
            if (!f) throw std::runtime_error("Cannot write manifest");
            f << manifest.to_json();
        }
        
        // Atomic rename
        fs::rename(temp_path, final_path);
        
    } catch (...) {
        // Cleanup temp on failure
        if (fs::exists(temp_path)) fs::remove_all(temp_path);
        throw;
    }
    
    // Rotate old checkpoints
    rotate();
    
    return final_path;
}

void CheckpointManager::rotate() {
    auto dirs = list_checkpoints();
    if (dirs.size() <= opts_.max_keep) return;
    
    // dirs is sorted newest first, so remove from the end
    for (size_t i = opts_.max_keep; i < dirs.size(); ++i) {
        fs::remove_all(dirs[i]);
    }
}

std::vector<std::string> CheckpointManager::list_checkpoints() const {
    std::vector<std::string> result;
    if (!fs::exists(opts_.base_dir)) return result;
    
    for (auto& entry : fs::directory_iterator(opts_.base_dir)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        if (name.rfind("checkpoint_", 0) == 0) {
            result.push_back(entry.path().string());
        }
    }
    
    // Sort newest first (lexicographic on timestamp-based names works)
    std::sort(result.begin(), result.end(), std::greater<>());
    return result;
}

// ─── Component Writers ───────────────────────────────────────────────────────

namespace {
template<typename T>
void write_pod(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
} // anon

ComponentHash CheckpointManager::write_ltm(const std::string& dir, PersistentLTM& ltm) {
    std::string path = dir + "/ltm.bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write ltm.bin");
    
    constexpr uint32_t MAGIC = 0x4C544D42; // "LTMB"
    constexpr uint16_t VERSION = 1;
    write_pod(f, MAGIC);
    write_pod(f, VERSION);
    
    // Get all concepts
    auto ids = ltm.get_all_concept_ids();
    uint64_t n = ids.size();
    write_pod(f, n);
    
    for (auto id : ids) {
        auto info = ltm.retrieve_concept(id);
        if (!info) continue;
        write_pod(f, id);
        // label
        uint32_t llen = static_cast<uint32_t>(info->label.size());
        write_pod(f, llen);
        f.write(info->label.data(), llen);
        // definition
        uint32_t dlen = static_cast<uint32_t>(info->definition.size());
        write_pod(f, dlen);
        f.write(info->definition.data(), dlen);
        // epistemic
        uint8_t etype = static_cast<uint8_t>(info->epistemic.type);
        uint8_t estatus = static_cast<uint8_t>(info->epistemic.status);
        write_pod(f, etype);
        write_pod(f, estatus);
        write_pod(f, info->epistemic.trust);
    }
    
    // Relations
    uint64_t rel_count = ltm.relation_count();
    write_pod(f, rel_count);
    
    for (auto id : ids) {
        auto rels = ltm.get_outgoing_relations(id);
        for (auto& r : rels) {
            write_pod(f, r.id);
            write_pod(f, r.source);
            write_pod(f, r.target);
            uint8_t t = static_cast<uint8_t>(r.type);
            write_pod(f, t);
            write_pod(f, r.weight);
        }
    }
    
    f.close();
    
    ComponentHash ch;
    ch.filename = "ltm.bin";
    ch.sha256 = SHA256::hash_file(path);
    ch.size_bytes = fs::file_size(path);
    return ch;
}

ComponentHash CheckpointManager::write_stm(const std::string& dir, const STMSnapshotData& data) {
    std::string path = dir + "/stm.bin";
    
    // Reuse STMSnapshotManager's format concept — but we write directly
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write stm.bin");
    
    constexpr uint32_t MAGIC = 0x53544D42; // "STMB"
    constexpr uint16_t VERSION = 1;
    write_pod(f, MAGIC);
    write_pod(f, VERSION);
    write_pod(f, data.timestamp);
    
    uint32_t ctx_count = static_cast<uint32_t>(data.contexts.size());
    write_pod(f, ctx_count);
    
    for (auto& ctx : data.contexts) {
        write_pod(f, ctx.context_id);
        uint32_t cc = static_cast<uint32_t>(ctx.concepts.size());
        uint32_t rc = static_cast<uint32_t>(ctx.relations.size());
        write_pod(f, cc);
        write_pod(f, rc);
        
        for (auto& c : ctx.concepts) {
            write_pod(f, c.concept_id);
            write_pod(f, c.activation);
            uint8_t cls = static_cast<uint8_t>(c.classification);
            write_pod(f, cls);
        }
        for (auto& r : ctx.relations) {
            write_pod(f, r.source);
            write_pod(f, r.target);
            uint8_t t = static_cast<uint8_t>(r.type);
            write_pod(f, t);
            write_pod(f, r.activation);
        }
    }
    
    f.close();
    ComponentHash ch;
    ch.filename = "stm.bin";
    ch.sha256 = SHA256::hash_file(path);
    ch.size_bytes = fs::file_size(path);
    return ch;
}

ComponentHash CheckpointManager::write_micromodels(const std::string& dir, const ConceptModelRegistry& reg) {
    std::string path = dir + "/micromodels.bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write micromodels.bin");

    constexpr uint32_t MAGIC = 0x4D4D4442; // "MMDB"
    constexpr uint16_t VERSION = 3;  // v3: ConceptModel (CM_FLAT_SIZE=1900)
    write_pod(f, MAGIC);
    write_pod(f, VERSION);

    auto ids = reg.get_model_ids();
    uint64_t n = ids.size();
    write_pod(f, n);

    for (auto id : ids) {
        const ConceptModel* model = reg.get_model(id);
        if (!model) continue;
        write_pod(f, id);
        std::array<double, CM_FLAT_SIZE> flat;
        model->to_flat(flat);
        f.write(reinterpret_cast<const char*>(flat.data()), sizeof(flat));
    }
    
    f.close();
    ComponentHash ch;
    ch.filename = "micromodels.bin";
    ch.sha256 = SHA256::hash_file(path);
    ch.size_bytes = fs::file_size(path);
    return ch;
}

ComponentHash CheckpointManager::write_kan_modules(const std::string& dir,
    const std::vector<std::pair<std::string, KANModule*>>& modules)
{
    std::string path = dir + "/kan_modules.bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write kan_modules.bin");
    
    constexpr uint32_t MAGIC = 0x4B414E42; // "KANB"
    constexpr uint16_t VERSION = 1;
    write_pod(f, MAGIC);
    write_pod(f, VERSION);
    
    uint64_t n = modules.size();
    write_pod(f, n);
    
    for (auto& [name, mod] : modules) {
        // Write name
        uint32_t nlen = static_cast<uint32_t>(name.size());
        write_pod(f, nlen);
        f.write(name.data(), nlen);
        
        // Write topology
        auto& topo = mod->topology();
        uint32_t num_dims = static_cast<uint32_t>(topo.size());
        write_pod(f, num_dims);
        for (auto d : topo) {
            uint64_t dim = d;
            write_pod(f, dim);
        }
        
        // Write all layer coefficients
        uint32_t nlayers = static_cast<uint32_t>(mod->num_layers());
        write_pod(f, nlayers);
        for (size_t l = 0; l < mod->num_layers(); ++l) {
            auto& layer = mod->layer(l);
            uint32_t nnodes = static_cast<uint32_t>(layer.num_nodes());
            write_pod(f, nnodes);
            for (size_t ni = 0; ni < nnodes; ++ni) {
                auto& node = layer.get_nodes()[ni];
                auto& coefs = node->get_coefficients();
                uint32_t nc = static_cast<uint32_t>(coefs.size());
                write_pod(f, nc);
                f.write(reinterpret_cast<const char*>(coefs.data()), nc * sizeof(double));
            }
        }
    }
    
    f.close();
    ComponentHash ch;
    ch.filename = "kan_modules.bin";
    ch.sha256 = SHA256::hash_file(path);
    ch.size_bytes = fs::file_size(path);
    return ch;
}

ComponentHash CheckpointManager::write_cognitive(const std::string& dir, const CognitiveState& state) {
    std::string path = dir + "/cognitive.bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write cognitive.bin");
    
    constexpr uint32_t MAGIC = 0x434F4742; // "COGB"
    constexpr uint16_t VERSION = 1;
    write_pod(f, MAGIC);
    write_pod(f, VERSION);
    
    uint64_t n = state.focus_set.size();
    write_pod(f, n);
    for (auto id : state.focus_set) write_pod(f, id);
    write_pod(f, state.avg_activation);
    write_pod(f, state.tick_count);
    write_pod(f, state.epoch_ms);
    
    f.close();
    ComponentHash ch;
    ch.filename = "cognitive.bin";
    ch.sha256 = SHA256::hash_file(path);
    ch.size_bytes = fs::file_size(path);
    return ch;
}

ComponentHash CheckpointManager::write_config(const std::string& dir, const CheckpointConfig& cfg) {
    std::string path = dir + "/config.json";
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write config.json");
    f << cfg.to_json();
    f.close();
    
    ComponentHash ch;
    ch.filename = "config.json";
    ch.sha256 = SHA256::hash_file(path);
    ch.size_bytes = fs::file_size(path);
    return ch;
}

} // namespace persistent
} // namespace brain19
