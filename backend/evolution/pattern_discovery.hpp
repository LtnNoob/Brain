#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/active_relation.hpp"
#include <string>
#include <unordered_set>
#include <vector>

namespace brain19 {

// =============================================================================
// PATTERN DISCOVERY
// =============================================================================
//
// Discovers structural patterns in the knowledge graph:
// - Clusters (densely connected subgraphs)
// - Hierarchies (IS_A chains)
// - Bridges (concepts connecting separate clusters)
// - Cycles (potential contradictions or feedback loops)
// - Gaps (expected relations that don't exist)
//
// Graph Feature aware:
// - Filters anti-knowledge concepts (is_anti_knowledge=true)
// - Uses epistemic trust as pattern weight
// - Includes all relation categories (LINGUISTIC, FUNCTIONAL, etc.)
//

struct DiscoveredPattern {
    std::string description;
    std::vector<ConceptId> involved_concepts;
    double confidence;
    std::string pattern_type;  // "cluster", "hierarchy", "cycle", "bridge", "gap"
    RelationType gap_rel_type = RelationType::ASSOCIATED_WITH;  // For gaps: the missing relation type

    DiscoveredPattern(
        const std::string& desc,
        const std::vector<ConceptId>& concepts,
        double conf,
        const std::string& type
    ) : description(desc)
      , involved_concepts(concepts)
      , confidence(conf)
      , pattern_type(type)
    {}

    DiscoveredPattern(
        const std::string& desc,
        const std::vector<ConceptId>& concepts,
        double conf,
        const std::string& type,
        RelationType rel
    ) : description(desc)
      , involved_concepts(concepts)
      , confidence(conf)
      , pattern_type(type)
      , gap_rel_type(rel)
    {}
};

class PatternDiscovery {
public:
    explicit PatternDiscovery(const LongTermMemory& ltm);

    // Find concept clusters (densely connected subgraphs)
    std::vector<DiscoveredPattern> find_clusters(size_t min_size = 3);

    // Find hierarchies (IS_A chains)
    std::vector<DiscoveredPattern> find_hierarchies(size_t min_depth = 3);

    // Find bridge concepts (connect otherwise separate clusters)
    std::vector<DiscoveredPattern> find_bridges();

    // Find cycles (A→B→C→A)
    std::vector<DiscoveredPattern> find_cycles(size_t max_length = 5);

    // Find gaps (expected relations that don't exist)
    std::vector<DiscoveredPattern> find_gaps();

    // Run full discovery
    std::vector<DiscoveredPattern> discover_all();

private:
    const LongTermMemory& ltm_;

    // Build adjacency for active, non-anti-knowledge concepts (all relation types)
    struct AdjacencyGraph {
        std::vector<ConceptId> nodes;
        std::unordered_map<ConceptId, std::vector<ConceptId>> adj;       // all relations
        std::unordered_map<ConceptId, std::vector<ConceptId>> adj_typed;  // IS_A only
        std::unordered_set<ConceptId> node_set;  // fast membership check
    };
    AdjacencyGraph build_graph() const;

    // Average epistemic trust of a set of concepts
    double avg_trust(const std::vector<ConceptId>& concepts) const;

    // DFS for cycle detection
    bool dfs_cycles(ConceptId node, ConceptId start,
                    std::vector<ConceptId>& path,
                    std::unordered_set<ConceptId>& visited,
                    const AdjacencyGraph& graph,
                    size_t max_length,
                    std::vector<std::vector<ConceptId>>& found_cycles) const;

    // Connected components via BFS
    std::vector<std::vector<ConceptId>> find_components(const AdjacencyGraph& graph) const;
};

} // namespace brain19
