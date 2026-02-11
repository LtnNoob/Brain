#pragma once

#include "../common/types.hpp"
#include "../memory/activation_level.hpp"
#include "../memory/active_relation.hpp"

#include <vector>
#include <cstdint>

namespace brain19 {

struct SnapshotConcept {
    ConceptId concept_id;
    double activation;
    ActivationClass classification;
};

struct SnapshotRelation {
    ConceptId source;
    ConceptId target;
    RelationType type;
    double activation;
};

struct SnapshotContext {
    ContextId context_id;
    std::vector<SnapshotConcept> concepts;
    std::vector<SnapshotRelation> relations;
};

struct STMSnapshotData {
    uint64_t timestamp = 0;
    std::vector<SnapshotContext> contexts;
};

} // namespace brain19
