#include "micro_model_registry.hpp"

namespace brain19 {

bool MicroModelRegistry::create_model(ConceptId cid) {
    auto [it, inserted] = models_.emplace(cid, MicroModel{});
    return inserted;
}

MicroModel* MicroModelRegistry::get_model(ConceptId cid) {
    auto it = models_.find(cid);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

const MicroModel* MicroModelRegistry::get_model(ConceptId cid) const {
    auto it = models_.find(cid);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

bool MicroModelRegistry::has_model(ConceptId cid) const {
    return models_.find(cid) != models_.end();
}

bool MicroModelRegistry::remove_model(ConceptId cid) {
    return models_.erase(cid) > 0;
}

std::vector<ConceptId> MicroModelRegistry::get_model_ids() const {
    std::vector<ConceptId> ids;
    ids.reserve(models_.size());
    for (const auto& [cid, model] : models_) {
        ids.push_back(cid);
    }
    return ids;
}

size_t MicroModelRegistry::ensure_models_for(const LongTermMemory& ltm) {
    size_t created = 0;
    auto all_ids = ltm.get_all_concept_ids();
    for (ConceptId cid : all_ids) {
        if (create_model(cid)) {
            ++created;
        }
    }
    return created;
}

size_t MicroModelRegistry::ensure_models_for(const std::vector<ConceptId>& concept_ids) {
    size_t created = 0;
    for (ConceptId cid : concept_ids) {
        if (create_model(cid)) {
            ++created;
        }
    }
    return created;
}

} // namespace brain19
