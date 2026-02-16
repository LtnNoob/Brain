#pragma once

#include "convergence_config.hpp"
#include "../common/types.hpp"
#include <vector>
#include <utility>
#include <unordered_map>

namespace brain19 {
namespace convergence {

// =============================================================================
// CENTROID ROUTER — Top-K concept selection via dot-product routing
// =============================================================================
//
// Routes input queries to the K most relevant ConceptModels using
// centroid-based dot-product similarity.
//
// Each concept has a centroid vector in ℝ^QUERY_DIM. The router selects
// the top-K concepts by computing h · centroid for all concepts.
//

struct RouteResult {
    ConceptId concept_id;
    double weight;      // Normalized routing weight (softmax of scores)
    double raw_score;   // Raw dot-product score
};

class CentroidRouter {
public:
    CentroidRouter();

    // Route input to top-K concepts
    std::vector<RouteResult> route(const std::vector<double>& h, size_t k = ROUTER_TOP_K) const;

    // Register a concept with its centroid
    void set_centroid(ConceptId id, const std::vector<double>& centroid);

    // Remove a concept's centroid
    void remove_centroid(ConceptId id);

    // Initialize centroids from a map of concept embeddings
    // Extracts first ROUTER_DIM dimensions from each embedding
    void init_from_embeddings(const std::unordered_map<ConceptId, std::vector<double>>& embeddings);

    // K-Means style centroid update: given (concept_id, new_centroid) pairs
    void update_centroids(const std::vector<std::pair<ConceptId, std::vector<double>>>& updates,
                          double momentum = 0.9);

    // Backward: compute gradient w.r.t. centroids given d_weight for each routed concept
    // Used during joint training
    void backward(const std::vector<double>& h,
                  const std::vector<RouteResult>& routes,
                  const std::vector<double>& d_weights,
                  double lr);

    // Load balancing: track activation frequency per concept
    void record_activation(const std::vector<RouteResult>& routes);
    double get_activation_frequency(ConceptId id) const;
    double compute_balance_loss() const;

    size_t num_concepts() const { return centroids_.size(); }

private:
    struct CentroidEntry {
        std::vector<double> centroid;  // [ROUTER_DIM]
        double activation_count = 0.0;
    };

    std::unordered_map<ConceptId, CentroidEntry> centroids_;
    double total_activations_ = 0.0;

    double dot_product(const std::vector<double>& a, const std::vector<double>& b) const;
};

} // namespace convergence
} // namespace brain19
