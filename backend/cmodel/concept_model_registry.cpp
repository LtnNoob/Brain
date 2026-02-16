#include "concept_model_registry.hpp"

namespace brain19 {

bool ConceptModelRegistry::create_model(ConceptId cid) {
    auto [it, inserted] = models_.try_emplace(cid);
    return inserted;
}

ConceptModel* ConceptModelRegistry::get_model(ConceptId cid) {
    auto it = models_.find(cid);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

const ConceptModel* ConceptModelRegistry::get_model(ConceptId cid) const {
    auto it = models_.find(cid);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

bool ConceptModelRegistry::has_model(ConceptId cid) const {
    return models_.find(cid) != models_.end();
}

bool ConceptModelRegistry::remove_model(ConceptId cid) {
    return models_.erase(cid) > 0;
}

std::vector<ConceptId> ConceptModelRegistry::get_model_ids() const {
    std::vector<ConceptId> ids;
    ids.reserve(models_.size());
    for (const auto& [cid, model] : models_) {
        ids.push_back(cid);
    }
    return ids;
}

size_t ConceptModelRegistry::ensure_models_for(const LongTermMemory& ltm) {
    size_t created = 0;
    auto all_ids = ltm.get_all_concept_ids();
    for (ConceptId cid : all_ids) {
        if (create_model(cid)) {
            ++created;
        }
    }
    return created;
}

size_t ConceptModelRegistry::ensure_models_for(const std::vector<ConceptId>& concept_ids) {
    size_t created = 0;
    for (ConceptId cid : concept_ids) {
        if (create_model(cid)) {
            ++created;
        }
    }
    return created;
}

} // namespace brain19
