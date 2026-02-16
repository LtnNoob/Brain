#include "persistent_ltm.hpp"
#include <algorithm>
#include <filesystem>
#include <cstring>

namespace brain19 {
namespace persistent {

PersistentLTM::PersistentLTM(const std::string& data_dir)
    : data_dir_(data_dir)
{
    std::filesystem::create_directories(data_dir);
    
    concepts_ = std::make_unique<PersistentStore<PersistentConceptRecord>>(
        data_dir + "/concepts.dat", "B19C", 4096);
    relations_ = std::make_unique<PersistentStore<PersistentRelationRecord>>(
        data_dir + "/relations.dat", "B19R", 8192);
    strings_ = std::make_unique<StringPool>(
        data_dir + "/strings.dat", 2 * 1024 * 1024);
    
    // Rebuild in-memory indices from persistent data
    rebuild_indices();
    
    // WAL recovery: replay any pending entries from a previous crash
    std::string wal_path = data_dir + "/brain19.wal";
    {
        WALRecovery::Stats stats = WALRecovery::recover(wal_path, *this);
        if (stats.entries_applied > 0) {
            // Re-sync stores after recovery replay
            sync();
            // Rebuild indices since replay may have added records
            rebuild_indices();
        }
    }
    
    // Open WAL writer for new operations
    wal_ = std::make_unique<WALWriter>(wal_path);
    // Checkpoint: WAL was fully replayed, safe to truncate
    wal_->checkpoint();
}

PersistentLTM::~PersistentLTM() {
    sync();
}

void PersistentLTM::rebuild_indices() {
    concept_index_.clear();
    relation_index_.clear();
    outgoing_.clear();
    incoming_.clear();
    
    // Rebuild concept index
    uint64_t n_concepts = concepts_->count();
    for (uint64_t i = 0; i < n_concepts; ++i) {
        auto* rec = concepts_->record(i);
        if (!rec->is_deleted()) {
            concept_index_[rec->concept_id] = i;
        }
    }
    
    // Rebuild relation index + adjacency
    uint64_t n_relations = relations_->count();
    for (uint64_t i = 0; i < n_relations; ++i) {
        auto* rec = relations_->record(i);
        if (!rec->is_deleted()) {
            relation_index_[rec->relation_id] = i;
            outgoing_[rec->source].push_back(rec->relation_id);
            incoming_[rec->target].push_back(rec->relation_id);
        }
    }
}

ConceptId PersistentLTM::store_concept(
    const std::string& label,
    const std::string& definition,
    EpistemicMetadata epistemic
) {
    ConceptId id = concepts_->next_id();
    concepts_->set_next_id(id + 1);
    
    // Store strings
    auto [lab_off, lab_len] = strings_->append(label);
    auto [def_off, def_len] = strings_->append(definition);
    
    // Build record
    PersistentConceptRecord rec;
    rec.clear();
    rec.concept_id = id;
    rec.label_offset = lab_off;
    rec.label_length = lab_len;
    rec.definition_offset = def_off;
    rec.definition_length = def_len;
    rec.epistemic_type = static_cast<uint8_t>(epistemic.type);
    rec.epistemic_status = static_cast<uint8_t>(epistemic.status);
    rec.trust = epistemic.trust;
    rec.created_epoch_us = now_epoch_us();
    rec.last_access_epoch_us = rec.created_epoch_us;
    rec.access_count = 0;
    rec.flags = 0;
    
    // WAL: log before mmap write
    if (wal_) {
        WALStoreConceptPayload wp;
        std::memset(&wp, 0, sizeof(wp));
        wp.concept_id = id;
        wp.label_offset = lab_off;
        wp.label_length = lab_len;
        wp.definition_offset = def_off;
        wp.definition_length = def_len;
        wp.epistemic_type = rec.epistemic_type;
        wp.epistemic_status = rec.epistemic_status;
        wp.trust = rec.trust;
        wp.created_epoch_us = rec.created_epoch_us;
        wal_->append(WALOpType::STORE_CONCEPT, &wp, sizeof(wp));
    }
    
    size_t slot = concepts_->append(rec);
    concept_index_[id] = slot;
    
    return id;
}

std::optional<ConceptInfo> PersistentLTM::retrieve_concept(ConceptId id) const {
    auto it = concept_index_.find(id);
    if (it == concept_index_.end()) return std::nullopt;
    
    auto* rec = concepts_->record(it->second);
    if (rec->is_deleted()) return std::nullopt;
    
    // Access stats removed from const method to avoid data race under shared_lock
    // Stats can be updated via a separate non-const method if needed.
    
    std::string label = strings_->get(rec->label_offset, rec->label_length);
    std::string def = strings_->get(rec->definition_offset, rec->definition_length);
    
    EpistemicMetadata meta(
        static_cast<EpistemicType>(rec->epistemic_type),
        static_cast<EpistemicStatus>(rec->epistemic_status),
        rec->trust
    );

    ConceptInfo info(id, label, def, meta);
    info.activation_score = rec->activation_score;
    info.salience_score = rec->salience_score;
    info.structural_confidence = rec->structural_confidence;
    info.semantic_confidence = rec->semantic_confidence;
    info.is_anti_knowledge = (rec->is_anti_knowledge != 0);
    info.complexity_score = rec->complexity_score;
    return info;
}

bool PersistentLTM::exists(ConceptId id) const {
    auto it = concept_index_.find(id);
    if (it == concept_index_.end()) return false;
    return !concepts_->record(it->second)->is_deleted();
}

bool PersistentLTM::update_epistemic_metadata(ConceptId id, EpistemicMetadata new_metadata) {
    auto it = concept_index_.find(id);
    if (it == concept_index_.end()) return false;
    
    auto* rec = concepts_->record(it->second);
    if (rec->is_deleted()) return false;
    
    if (wal_) {
        WALUpdateMetadataPayload wp;
        std::memset(&wp, 0, sizeof(wp));
        wp.concept_id = id;
        wp.epistemic_type = static_cast<uint8_t>(new_metadata.type);
        wp.epistemic_status = static_cast<uint8_t>(new_metadata.status);
        wp.trust = new_metadata.trust;
        wal_->append(WALOpType::UPDATE_METADATA, &wp, sizeof(wp));
    }
    
    rec->epistemic_type = static_cast<uint8_t>(new_metadata.type);
    rec->epistemic_status = static_cast<uint8_t>(new_metadata.status);
    rec->trust = new_metadata.trust;
    
    return true;
}

bool PersistentLTM::invalidate_concept(ConceptId id, double invalidation_trust) {
    auto it = concept_index_.find(id);
    if (it == concept_index_.end()) return false;
    
    auto* rec = concepts_->record(it->second);
    if (rec->is_deleted()) return false;
    
    if (invalidation_trust < 0.0 || invalidation_trust > 1.0) {
        invalidation_trust = 0.05;
    }
    
    if (wal_) {
        WALInvalidateConceptPayload wp;
        std::memset(&wp, 0, sizeof(wp));
        wp.concept_id = id;
        wp.invalidation_trust = invalidation_trust;
        wal_->append(WALOpType::INVALIDATE_CONCEPT, &wp, sizeof(wp));
    }
    
    // Directly mutate record instead of calling update_epistemic_metadata()
    // to avoid a second WAL entry (double-log bug)
    rec->epistemic_status = static_cast<uint8_t>(EpistemicStatus::INVALIDATED);
    rec->trust = invalidation_trust;
    
    return true;
}

bool PersistentLTM::set_anti_knowledge(ConceptId id, bool is_ak, float complexity) {
    auto it = concept_index_.find(id);
    if (it == concept_index_.end()) return false;

    auto* rec = concepts_->record(it->second);
    if (rec->is_deleted()) return false;

    if (wal_) {
        WALSetAntiKnowledgePayload wp;
        std::memset(&wp, 0, sizeof(wp));
        wp.concept_id = id;
        wp.is_anti_knowledge = is_ak ? 1 : 0;
        wp.complexity_score = complexity;
        wal_->append(WALOpType::SET_ANTI_KNOWLEDGE, &wp, sizeof(wp));
    }

    rec->is_anti_knowledge = is_ak ? 1 : 0;
    rec->complexity_score = complexity;

    return true;
}

std::vector<ConceptId> PersistentLTM::get_concepts_by_type(EpistemicType type) const {
    std::vector<ConceptId> result;
    uint8_t t = static_cast<uint8_t>(type);
    for (const auto& [id, slot] : concept_index_) {
        auto* rec = concepts_->record(slot);
        if (!rec->is_deleted() && rec->epistemic_type == t) {
            result.push_back(id);
        }
    }
    return result;
}

std::vector<ConceptId> PersistentLTM::get_concepts_by_status(EpistemicStatus status) const {
    std::vector<ConceptId> result;
    uint8_t s = static_cast<uint8_t>(status);
    for (const auto& [id, slot] : concept_index_) {
        auto* rec = concepts_->record(slot);
        if (!rec->is_deleted() && rec->epistemic_status == s) {
            result.push_back(id);
        }
    }
    return result;
}

std::vector<ConceptId> PersistentLTM::get_active_concepts() const {
    return get_concepts_by_status(EpistemicStatus::ACTIVE);
}

std::optional<RelationId> PersistentLTM::add_relation(
    ConceptId source, ConceptId target, RelationType type, double weight
) {
    if (!exists(source) || !exists(target)) return std::nullopt;
    
    RelationId id = relations_->next_id();
    relations_->set_next_id(id + 1);
    
    // Clamp weight
    if (weight < 0.0) weight = 0.0;
    if (weight > 1.0) weight = 1.0;
    
    PersistentRelationRecord rec;
    rec.clear();
    rec.relation_id = id;
    rec.source = source;
    rec.target = target;
    rec.set_type_id(static_cast<uint16_t>(type));
    rec.weight = weight;
    rec.flags = 0;

    // WAL: log before mmap write
    if (wal_) {
        WALAddRelationPayload wp;
        std::memset(&wp, 0, sizeof(wp));
        wp.relation_id = id;
        wp.source = source;
        wp.target = target;
        wp.type = static_cast<uint16_t>(type);
        wp.weight = rec.weight;
        wal_->append(WALOpType::ADD_RELATION, &wp, sizeof(wp));
    }
    
    size_t slot = relations_->append(rec);
    relation_index_[id] = slot;
    outgoing_[source].push_back(id);
    incoming_[target].push_back(id);
    
    return id;
}

std::optional<RelationInfo> PersistentLTM::get_relation(RelationId id) const {
    auto it = relation_index_.find(id);
    if (it == relation_index_.end()) return std::nullopt;
    
    auto* rec = relations_->record(it->second);
    if (rec->is_deleted()) return std::nullopt;
    
    RelationInfo info(
        rec->relation_id, rec->source, rec->target,
        static_cast<RelationType>(rec->get_type_id()), rec->weight
    );
    info.dynamic_weight = rec->dynamic_weight;
    info.inhibition_factor = rec->inhibition_factor;
    info.structural_strength = rec->structural_strength;
    return info;
}

std::vector<RelationInfo> PersistentLTM::get_outgoing_relations(ConceptId source) const {
    std::vector<RelationInfo> result;
    auto it = outgoing_.find(source);
    if (it == outgoing_.end()) return result;
    
    for (RelationId rid : it->second) {
        auto rel = get_relation(rid);
        if (rel) result.push_back(*rel);
    }
    return result;
}

std::vector<RelationInfo> PersistentLTM::get_incoming_relations(ConceptId target) const {
    std::vector<RelationInfo> result;
    auto it = incoming_.find(target);
    if (it == incoming_.end()) return result;
    
    for (RelationId rid : it->second) {
        auto rel = get_relation(rid);
        if (rel) result.push_back(*rel);
    }
    return result;
}

std::vector<RelationInfo> PersistentLTM::get_relations_between(ConceptId source, ConceptId target) const {
    std::vector<RelationInfo> result;
    for (const auto& rel : get_outgoing_relations(source)) {
        if (rel.target == target) result.push_back(rel);
    }
    return result;
}

bool PersistentLTM::remove_relation(RelationId id) {
    auto it = relation_index_.find(id);
    if (it == relation_index_.end()) return false;
    
    auto* rec = relations_->record(it->second);
    if (rec->is_deleted()) return false;
    
    ConceptId source = rec->source;
    ConceptId target = rec->target;
    
    if (wal_) {
        WALRemoveRelationPayload wp;
        std::memset(&wp, 0, sizeof(wp));
        wp.relation_id = id;
        wal_->append(WALOpType::REMOVE_RELATION, &wp, sizeof(wp));
    }
    
    rec->mark_deleted();
    relation_index_.erase(it);
    
    // Remove from adjacency
    auto& out = outgoing_[source];
    out.erase(std::remove(out.begin(), out.end(), id), out.end());
    auto& in = incoming_[target];
    in.erase(std::remove(in.begin(), in.end(), id), in.end());
    
    return true;
}

size_t PersistentLTM::get_relation_count(ConceptId concept_id) const {
    size_t count = 0;
    auto oit = outgoing_.find(concept_id);
    if (oit != outgoing_.end()) count += oit->second.size();
    auto iit = incoming_.find(concept_id);
    if (iit != incoming_.end()) count += iit->second.size();
    return count;
}

std::vector<ConceptId> PersistentLTM::get_all_concept_ids() const {
    std::vector<ConceptId> result;
    result.reserve(concept_index_.size());
    for (const auto& [id, _] : concept_index_) {
        result.push_back(id);
    }
    return result;
}

void PersistentLTM::sync() {
    if (concepts_) concepts_->sync();
    if (relations_) relations_->sync();
    if (strings_) strings_->sync();
}

size_t PersistentLTM::concept_count() const {
    return concept_index_.size();
}

size_t PersistentLTM::relation_count() const {
    return relation_index_.size();
}

void PersistentLTM::checkpoint() {
    sync();
    if (wal_) wal_->checkpoint();
}

void PersistentLTM::replay_store_concept(
    uint64_t concept_id,
    uint32_t label_offset, uint32_t label_length,
    uint32_t def_offset, uint32_t def_length,
    uint8_t epistemic_type, uint8_t epistemic_status,
    double trust, uint64_t created_epoch_us
) {
    // Idempotent: skip if already exists
    if (concept_index_.count(concept_id)) return;
    
    // Ensure next_id stays consistent
    if (concepts_->next_id() <= concept_id) {
        concepts_->set_next_id(concept_id + 1);
    }
    
    PersistentConceptRecord rec;
    rec.clear();
    rec.concept_id = concept_id;
    rec.label_offset = label_offset;
    rec.label_length = label_length;
    rec.definition_offset = def_offset;
    rec.definition_length = def_length;
    rec.epistemic_type = epistemic_type;
    rec.epistemic_status = epistemic_status;
    rec.trust = trust;
    rec.created_epoch_us = created_epoch_us;
    rec.last_access_epoch_us = created_epoch_us;
    rec.access_count = 0;
    rec.flags = 0;
    
    size_t slot = concepts_->append(rec);
    concept_index_[concept_id] = slot;
}

void PersistentLTM::replay_add_relation(
    uint64_t relation_id, uint64_t source, uint64_t target,
    uint16_t type, double weight
) {
    // Idempotent: skip if already exists
    if (relation_index_.count(relation_id)) return;

    if (relations_->next_id() <= relation_id) {
        relations_->set_next_id(relation_id + 1);
    }

    PersistentRelationRecord rec;
    rec.clear();
    rec.relation_id = relation_id;
    rec.source = source;
    rec.target = target;
    rec.set_type_id(type);
    rec.weight = weight;
    rec.flags = 0;
    
    size_t slot = relations_->append(rec);
    relation_index_[relation_id] = slot;
    outgoing_[source].push_back(relation_id);
    incoming_[target].push_back(relation_id);
}

} // namespace persistent
} // namespace brain19
