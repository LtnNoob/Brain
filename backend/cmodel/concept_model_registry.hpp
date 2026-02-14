#pragma once

#include "concept_model.hpp"
#include "../ltm/long_term_memory.hpp"

#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// CONCEPT MODEL REGISTRY
// =============================================================================
//
// One ConceptModel per ConceptId. Drop-in replacement for MicroModelRegistry.
//

class ConceptModelRegistry {
public:
    ConceptModelRegistry() = default;

    bool create_model(ConceptId cid);

    ConceptModel* get_model(ConceptId cid);
    const ConceptModel* get_model(ConceptId cid) const;

    bool has_model(ConceptId cid) const;
    bool remove_model(ConceptId cid);

    std::vector<ConceptId> get_model_ids() const;

    size_t ensure_models_for(const LongTermMemory& ltm);
    size_t ensure_models_for(const std::vector<ConceptId>& concept_ids);

    size_t size() const { return models_.size(); }
    void clear() { models_.clear(); }

private:
    std::unordered_map<ConceptId, ConceptModel> models_;
};

} // namespace brain19
