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
// GraphDensifier: Generate new typed relations from graph topology
// =============================================================================
//
// Phase 1: Property inheritance (A IS_A B, B HAS_PROP P → A HAS_PROP P)
// Phase 2: Transitive IS_A (A IS_A B, B IS_A C → A IS_A C)
// Phase 3: Co-activation (shared neighbors → ASSOCIATED_WITH)
// Phase 4: PART_OF transitivity
// Phase 5: Causal transitivity (A CAUSES B, B CAUSES C → A ENABLES C)
// Phase 6: PART_OF property inheritance (A PART_OF B, B HAS_PROP P → A HAS_PROP P)
// Iterative: multiple passes compound these phases
//

class GraphDensifier {
public:
    explicit GraphDensifier(LongTermMemory& ltm);

    struct Config {
        bool enable_property_inheritance = true;
        bool enable_transitive_isa = true;
        bool enable_coactivation = true;
        bool enable_partof_transitive = true;
        bool enable_causal_transitive = false;  // disabled: too noisy on large graphs
        bool enable_partof_property = true;

        size_t min_common_neighbors = 5;
        double jaccard_threshold = 0.15;
        size_t max_new_rels_per_concept = 30;
        double base_weight = 0.6;

        size_t iterations = 3;  // iterative passes (each builds on previous)

        Config() {}  // C++20 aggregate init workaround
    };

    struct Result {
        size_t relations_added = 0;
        size_t duplicates_skipped = 0;
        size_t cap_limited = 0;
        double density_before = 0.0;
        double density_after = 0.0;
        size_t concepts = 0;
        std::map<std::string, size_t> type_distribution;
        std::map<std::string, size_t> phase_counts;
    };

    struct SampledRelation {
        std::string source_label;
        std::string target_label;
        std::string type_name;
        double weight;
    };

    Result densify(const Config& config = Config{});

    // Sample N random generated relations for quality inspection
    std::vector<SampledRelation> sample_generated(size_t n = 100) const;

private:
    LongTermMemory& ltm_;
    std::vector<RelationId> generated_ids_;

    // Per-concept new relation counter (for cap enforcement)
    std::unordered_map<ConceptId, size_t> new_rel_count_;

    // Fast duplicate check: set of existing (source, target) pair keys
    std::unordered_set<uint64_t> existing_pairs_;

    void build_existing_pairs_index();
    static uint64_t pair_key(ConceptId a, ConceptId b);
    bool pair_exists(ConceptId a, ConceptId b) const;
    bool try_add(ConceptId src, ConceptId tgt, RelationType type,
                 double weight, size_t cap, Result& result,
                 const std::string& type_name);

    size_t phase_property_inheritance(const Config& cfg, Result& r);
    size_t phase_transitive_isa(const Config& cfg, Result& r);
    size_t phase_coactivation(const Config& cfg, Result& r);
    size_t phase_partof_transitive(const Config& cfg, Result& r);
    size_t phase_causal_transitive(const Config& cfg, Result& r);
    size_t phase_partof_property(const Config& cfg, Result& r);
};

} // namespace brain19
