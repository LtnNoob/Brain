#pragma once

#include "micro_model.hpp"
#include "../ltm/long_term_memory.hpp"

#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// MICRO-MODEL REGISTRY
// =============================================================================
//
// One MicroModel per ConceptId. Follows KANAdapter pattern:
// explicit lifecycle, keyed lookup, bulk operations.
//

class MicroModelRegistry {
public:
    MicroModelRegistry() = default;

    // Create a new model for the given concept. Returns false if already exists.
    bool create_model(ConceptId cid);

    // Get pointer to model (nullptr if not found). Non-owning.
    MicroModel* get_model(ConceptId cid);
    const MicroModel* get_model(ConceptId cid) const;

    // Check existence
    bool has_model(ConceptId cid) const;

    // Remove model. Returns false if not found.
    bool remove_model(ConceptId cid);

    // Get all concept IDs that have models
    std::vector<ConceptId> get_model_ids() const;

    // Bulk-create models for all concepts in LTM that don't have one yet.
    // Returns number of newly created models.
    size_t ensure_models_for(const LongTermMemory& ltm);

    // Total model count
    size_t size() const { return models_.size(); }

    // Clear all models
    void clear() { models_.clear(); }

private:
    std::unordered_map<ConceptId, MicroModel> models_;
};

} // namespace brain19
