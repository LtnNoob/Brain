#pragma once
// Phase 4.1: Checkpoint Manager — Full Brain State Checkpointing
//
// Saves the ENTIRE brain state atomically into a versioned directory:
//   manifest.json, ltm.bin, stm.bin, micromodels.bin, kan_modules.bin,
//   cognitive.bin, config.json
//
// Atomic: writes to temp dir, then renames (POSIX atomic rename).
// Integrity: SHA-256 hash per component file.
// Rotation: keeps max N checkpoints, deletes oldest.

#include "persistent_ltm.hpp"
#include "stm_snapshot.hpp"
#include "stm_snapshot_data.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../kan/kan_module.hpp"
#include "../common/types.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <chrono>

namespace brain19 {
namespace persistent {

// ─── SHA-256 (compact, no external deps) ─────────────────────────────────────

struct SHA256 {
    static std::string hash_file(const std::string& path);
    static std::string hash_bytes(const uint8_t* data, size_t len);
private:
    uint32_t state_[8];
    uint64_t bitcount_;
    uint8_t  buffer_[64];
    size_t   buflen_;

    void init();
    void update(const uint8_t* data, size_t len);
    std::string finalize();
    void transform(const uint8_t block[64]);
};

// ─── Checkpoint Metadata ─────────────────────────────────────────────────────

struct ComponentHash {
    std::string filename;
    std::string sha256;
    uint64_t    size_bytes = 0;
};

struct CheckpointManifest {
    uint32_t    format_version = 1;
    std::string timestamp;       // ISO 8601
    uint64_t    epoch_ms = 0;
    std::string tag;             // user-provided label
    
    // Component hashes
    std::vector<ComponentHash> components;
    
    // Summary stats
    uint64_t concept_count   = 0;
    uint64_t relation_count  = 0;
    uint64_t model_count     = 0;
    uint64_t kan_module_count = 0;
    
    // Serialize / deserialize
    std::string to_json() const;
    static CheckpointManifest from_json(const std::string& json);
};

// ─── Cognitive State (serializable) ──────────────────────────────────────────

struct CognitiveState {
    std::vector<ConceptId> focus_set;
    double    avg_activation = 0.0;
    uint64_t  tick_count     = 0;
    uint64_t  epoch_ms       = 0;
};

// ─── Config Data ─────────────────────────────────────────────────────────────

struct CheckpointConfig {
    std::unordered_map<std::string, std::string> entries;
    
    std::string to_json() const;
    static CheckpointConfig from_json(const std::string& json);
};

// ─── Checkpoint Manager ──────────────────────────────────────────────────────

class CheckpointManager {
public:
    struct Options {
        std::string base_dir    = "checkpoints";
        size_t      max_keep    = 5;
        std::string tag;  // optional tag for this checkpoint
    };
    
    CheckpointManager() : opts_{} {}
    explicit CheckpointManager(const Options& opts);
    
    // Save a full brain checkpoint. Returns the checkpoint directory path.
    // All parameters are optional — pass nullptr to skip a component.
    std::string save(
        PersistentLTM*              ltm,
        const STMSnapshotData*      stm_data,
        const MicroModelRegistry*   models,
        const std::vector<std::pair<std::string, KANModule*>>* kan_modules,
        const CognitiveState*       cognitive,
        const CheckpointConfig*     config
    );
    
    // Rotate: delete oldest checkpoints beyond max_keep
    void rotate();
    
    // List all checkpoint dirs sorted by time (newest first)
    std::vector<std::string> list_checkpoints() const;
    
    const Options& options() const { return opts_; }
    
private:
    Options opts_;
    
    // Write individual component files and return their hashes
    ComponentHash write_ltm(const std::string& dir, PersistentLTM& ltm);
    ComponentHash write_stm(const std::string& dir, const STMSnapshotData& data);
    ComponentHash write_micromodels(const std::string& dir, const MicroModelRegistry& reg);
    ComponentHash write_kan_modules(const std::string& dir,
        const std::vector<std::pair<std::string, KANModule*>>& modules);
    ComponentHash write_cognitive(const std::string& dir, const CognitiveState& state);
    ComponentHash write_config(const std::string& dir, const CheckpointConfig& cfg);
    
    static std::string make_timestamp_dirname(const std::string& tag);
    static std::string now_iso8601();
    static uint64_t now_epoch_ms();
};

} // namespace persistent
} // namespace brain19
