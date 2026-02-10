#include "persistent_ltm.hpp"
#include <algorithm>
#include <filesystem>

namespace brain19 {
namespace persistent {

PersistentLTM::PersistentLTM(const std::string& data_dir) {
    std::filesystem::create_directories(data_dir);
    
    concepts_ = std::make_unique<PersistentStore<PersistentConceptRecord>>(
        data_dir + "/concepts.dat", "B19C", 4096);
    relations_ = std::make_unique<PersistentStore<PersistentRelationRecord>>(
        data_dir + "/relations.dat", "B19R", 8192);
    strings_ = std::make_unique<StringPool>(
        data_dir + "/strings.dat", 2 * 1024 * 1024);
    
    // Rebuild in-memory indices from persistent data
    rebuild_indices();
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
    
    size_t slot = concepts_->append(rec);
    concept_index_[id] = slot;
    
    return id;
}

std::optional<ConceptInfo> PersistentLTM::retrieve_concept(ConceptId id) const {
    auto it = concept_index_.find(id);
    if (it == concept_index_.end()) return std::nullopt;
    
    auto* rec = concepts_->record(it->second);
    if (rec->is_deleted()) return std::nullopt;
    
    // Update access stats (const_cast acceptable for stats-only mutation)
    auto* mut_rec = const_cast<PersistentConceptRecord*>(rec);
    mut_rec->access_count++;
    mut_rec->last_access_epoch_us = const_cast<PersistentLTM*>(this)->now_epoch_us();
    
    std::string label = strings_->get(rec->label_offset, rec->label_length);
    std::string def = strings_->get(rec->definition_offset, rec->definition_length);
    
    EpistemicMetadata meta(
        static_cast<EpistemicType>(rec->epistemic_type),
        static_cast<EpistemicStatus>(rec->epistemic_status),
        rec->trust
    );
    
    return ConceptInfo(id, label, def, meta);
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
    
    EpistemicMetadata meta(
        static_cast<EpistemicType>(rec->epistemic_type),
        EpistemicStatus::INVALIDATED,
        invalidation_trust
    );
    
    return update_epistemic_metadata(id, meta);
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

RelationId PersistentLTM::add_relation(
    ConceptId source, ConceptId target, RelationType type, double weight
) {
    if (!exists(source) || !exists(target)) return 0;
    
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
    rec.type = static_cast<uint8_t>(type);
    rec.weight = weight;
    rec.flags = 0;
    
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
    
    return RelationInfo(
        rec->relation_id, rec->source, rec->target,
        static_cast<RelationType>(rec->type), rec->weight
    );
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

} // namespace persistent
} // namespace brain19
