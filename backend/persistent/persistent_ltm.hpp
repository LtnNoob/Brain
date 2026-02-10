#pragma once
// Phase 1.1: PersistentLTM — mmap-backed LongTermMemory implementation
// Phase 1.2: WAL integration for crash recovery
//
// Wraps PersistentStore + StringPool to provide the same interface as LongTermMemory
// but with data living on disk via mmap.

#include "persistent_store.hpp"
#include "string_pool.hpp"
#include "persistent_records.hpp"
#include "wal.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include "../ltm/relation.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../common/types.hpp"

#include <string>
#include <optional>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <filesystem>

namespace brain19 {
namespace persistent {

class PersistentLTM {
public:
    // data_dir: directory where .dat files will be stored
    explicit PersistentLTM(const std::string& data_dir);
    ~PersistentLTM();
    
    // === Same API as LongTermMemory ===
    
    ConceptId store_concept(
        const std::string& label,
        const std::string& definition,
        EpistemicMetadata epistemic
    );
    
    std::optional<ConceptInfo> retrieve_concept(ConceptId id) const;
    bool exists(ConceptId id) const;
    bool update_epistemic_metadata(ConceptId id, EpistemicMetadata new_metadata);
    bool invalidate_concept(ConceptId id, double invalidation_trust = 0.05);
    
    std::vector<ConceptId> get_concepts_by_type(EpistemicType type) const;
    std::vector<ConceptId> get_concepts_by_status(EpistemicStatus status) const;
    std::vector<ConceptId> get_active_concepts() const;
    
    RelationId add_relation(ConceptId source, ConceptId target, RelationType type, double weight = 1.0);
    std::optional<RelationInfo> get_relation(RelationId id) const;
    std::vector<RelationInfo> get_outgoing_relations(ConceptId source) const;
    std::vector<RelationInfo> get_incoming_relations(ConceptId target) const;
    std::vector<RelationInfo> get_relations_between(ConceptId source, ConceptId target) const;
    bool remove_relation(RelationId id);
    size_t get_relation_count(ConceptId concept_id) const;
    std::vector<ConceptId> get_all_concept_ids() const;
    
    // Persistence-specific
    void sync();
    void checkpoint();  // msync all stores + truncate WAL
    size_t concept_count() const;
    size_t relation_count() const;
    
    // WAL replay helpers (used by WALRecovery — idempotent)
    void replay_store_concept(
        uint64_t concept_id,
        uint32_t label_offset, uint32_t label_length,
        uint32_t def_offset, uint32_t def_length,
        uint8_t epistemic_type, uint8_t epistemic_status,
        double trust, uint64_t created_epoch_us
    );
    void replay_add_relation(
        uint64_t relation_id, uint64_t source, uint64_t target,
        uint8_t type, double weight
    );
    
private:
    void rebuild_indices();
    
    static uint64_t now_epoch_us() {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
    }
    
    std::unique_ptr<PersistentStore<PersistentConceptRecord>> concepts_;
    std::unique_ptr<PersistentStore<PersistentRelationRecord>> relations_;
    std::unique_ptr<StringPool> strings_;
    std::unique_ptr<WALWriter> wal_;
    std::string data_dir_;
    
    // In-memory indices (rebuilt on load from mmap data)
    std::unordered_map<ConceptId, size_t> concept_index_;   // id -> slot
    std::unordered_map<RelationId, size_t> relation_index_;  // id -> slot
    std::unordered_map<ConceptId, std::vector<RelationId>> outgoing_;
    std::unordered_map<ConceptId, std::vector<RelationId>> incoming_;
};

} // namespace persistent
} // namespace brain19
