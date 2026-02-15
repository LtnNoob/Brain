#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../memory/active_relation.hpp"
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brain19 {

// =============================================================================
// PropertyInheritance: Fixpoint-iteration over IS_A hierarchies
// =============================================================================
//
// Algorithm:
//   1. For each concept: collect properties from parents via IS_A chain
//   2. Inherit what is NOT blocked by CONTRADICTS
//   3. Trust decays 0.9x per hop
//   4. Trust floor 0.3 as cutoff (stop propagating below this)
//   5. Repeat until fixpoint (no new properties added)
//
// Example: Pudel -> Hund -> Carnivore -> Mammal -> Animal
//   Animal HAS_PROPERTY "has_spine" (trust 0.95)
//   => Mammal inherits "has_spine" at 0.855 (1 hop)
//   => Carnivore inherits at 0.7695 (2 hops)
//   => Hund inherits at 0.6926 (3 hops)
//   => Pudel inherits at 0.6233 (4 hops)
//
// CONTRADICTS blocking:
//   If concept C has CONTRADICTS relation with property P,
//   then C does NOT inherit P, and neither do C's descendants.
//

struct InheritedProperty {
    ConceptId property_target;   // The property concept (target of HAS_PROPERTY)
    ConceptId origin_concept;    // Where the property was originally defined
    double inherited_trust;      // Trust after hop-decay
    size_t hop_count;            // Number of IS_A hops from origin
};

class PropertyInheritance {
public:
    explicit PropertyInheritance(LongTermMemory& ltm);

    struct Config {
        double decay_per_hop = 0.9;         // Trust multiplier per IS_A hop
        double trust_floor = 0.3;           // Stop propagating below this
        size_t max_iterations = 50;         // Fixpoint iteration cap
        size_t max_hop_depth = 20;          // Max IS_A chain depth (cycle safety)
        double min_relation_weight = 0.1;   // Min weight for generated relations

        // Which relation types to propagate (besides HAS_PROPERTY)
        bool propagate_requires = true;
        bool propagate_uses = true;
        bool propagate_produces = true;

        Config() {}  // C++20 aggregate init workaround
    };

    struct Result {
        size_t properties_inherited = 0;    // Total new HAS_PROPERTY (etc.) added
        size_t contradictions_blocked = 0;  // Properties blocked by CONTRADICTS
        size_t trust_floor_cutoffs = 0;     // Stopped due to trust floor
        size_t duplicates_skipped = 0;      // Already existed
        size_t iterations_run = 0;          // Number of fixpoint iterations
        bool converged = false;             // True if fixpoint reached
        size_t concepts_processed = 0;      // Total concepts examined
        std::map<std::string, size_t> type_distribution;  // By relation type
    };

    Result propagate(const Config& config = Config{});

    // Query: what properties did concept inherit (after propagate())?
    std::vector<InheritedProperty> get_inherited(ConceptId cid) const;

private:
    LongTermMemory& ltm_;

    // Per-concept: set of inherited properties (property_target -> InheritedProperty)
    std::unordered_map<ConceptId,
        std::unordered_map<ConceptId, InheritedProperty>> inherited_;

    // CONTRADICTS index: concept -> set of concepts it contradicts
    std::unordered_map<ConceptId, std::unordered_set<ConceptId>> contradicts_;

    // Existing (source, target, type) triples for dedup
    std::unordered_set<uint64_t> existing_triples_;

    // Which relation types are inheritable
    std::vector<RelationType> inheritable_types(const Config& cfg) const;

    // Build indices
    void build_contradicts_index();
    void build_existing_triples_index(const std::vector<RelationType>& types);

    // Check if concept contradicts a property target
    bool is_contradicted(ConceptId cid, ConceptId property_target) const;

    // Hash for (source, target) dedup
    static uint64_t triple_key(ConceptId src, ConceptId tgt);

    // Single iteration: returns number of new properties propagated
    size_t iterate_once(const Config& cfg, Result& result,
                        const std::vector<RelationType>& types);
};

} // namespace brain19
