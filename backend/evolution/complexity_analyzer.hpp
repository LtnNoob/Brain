#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../evolution/graph_densifier.hpp"
#include <unordered_set>
#include <vector>

namespace brain19 {

struct ComplexityMetrics {
    size_t causal_chain_length = 0;    // Longest CAUSES chain through this concept
    size_t involved_concepts = 0;      // Unique concepts in the dependency subgraph
    size_t relation_depth = 0;         // Max BFS distance from concept
    size_t inference_steps = 0;        // Number of INFERENCE-typed concepts in chain
    float normalized_score = 0.0f;     // Weighted combination [0.0, 1.0]
};

struct RetentionConfig {
    float complexity_threshold = 0.4f;       // Score >= this → anti-knowledge
    size_t min_causal_chain = 3;             // Minimum chain length
    size_t min_involved_concepts = 5;        // Minimum involved concepts
    size_t max_traversal_depth = 10;         // BFS limit

    // Weights for normalized_score
    float weight_causal_chain = 0.35f;
    float weight_involved_concepts = 0.25f;
    float weight_relation_depth = 0.20f;
    float weight_inference_steps = 0.20f;

    RetentionConfig() {}
};

class ComplexityAnalyzer {
public:
    ComplexityAnalyzer(LongTermMemory& ltm,
                       GraphDensifier& densifier,
                       RetentionConfig config = {});

    // Compute complexity metrics for a concept
    ComplexityMetrics analyze(ConceptId id) const;

    // Should this invalidated concept be retained as anti-knowledge?
    bool should_retain(ConceptId invalidated) const;

    // Evaluate all INVALIDATED concepts, mark as anti-knowledge if complex
    // Returns: number of newly marked anti-knowledge concepts
    size_t evaluate_all_invalidated();

    // Extract the longest CAUSES chain through this concept
    std::vector<ConceptId> extract_causal_chain(ConceptId id) const;

    // Extract all concepts transitively dependent (BFS up to max_depth)
    std::unordered_set<ConceptId> extract_dependency_subgraph(
        ConceptId id, size_t max_depth) const;

private:
    LongTermMemory& ltm_;
    GraphDensifier& densifier_;
    RetentionConfig config_;

    // BFS for longest causal chain starting from concept
    size_t longest_causal_chain(ConceptId start) const;

    // Count INFERENCE-typed concepts in a chain
    size_t count_inference_steps(const std::vector<ConceptId>& chain) const;

    // Normalize value to [0,1] with saturation cap
    static float norm(size_t value, size_t cap);
};

} // namespace brain19
