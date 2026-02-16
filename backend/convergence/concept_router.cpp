#include "concept_router.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <numeric>

namespace brain19 {
namespace convergence {

CentroidRouter::CentroidRouter() = default;

std::vector<RouteResult> CentroidRouter::route(
    const std::vector<double>& h, size_t k) const
{
    assert(h.size() >= ROUTER_DIM);

    // Compute dot product with all centroids
    std::vector<RouteResult> scores;
    scores.reserve(centroids_.size());

    for (const auto& [id, entry] : centroids_) {
        double score = dot_product(h, entry.centroid);
        scores.push_back({id, 0.0, score});
    }

    // Partial sort to get top-K
    k = std::min(k, scores.size());
    if (k == 0) return {};

    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const RouteResult& a, const RouteResult& b) {
            return a.raw_score > b.raw_score;
        });
    scores.resize(k);

    // Softmax normalization over top-K scores
    double max_score = scores[0].raw_score;
    double sum_exp = 0.0;
    for (auto& r : scores) {
        r.weight = std::exp(r.raw_score - max_score);
        sum_exp += r.weight;
    }
    if (sum_exp > 0.0) {
        for (auto& r : scores) {
            r.weight /= sum_exp;
        }
    }

    return scores;
}

void CentroidRouter::set_centroid(ConceptId id, const std::vector<double>& centroid) {
    assert(centroid.size() == ROUTER_DIM);
    centroids_[id].centroid = centroid;
}

void CentroidRouter::remove_centroid(ConceptId id) {
    centroids_.erase(id);
}

void CentroidRouter::init_from_embeddings(
    const std::unordered_map<ConceptId, std::vector<double>>& embeddings)
{
    centroids_.clear();
    for (const auto& [id, emb] : embeddings) {
        CentroidEntry entry;
        entry.centroid.resize(ROUTER_DIM);
        size_t copy_dim = std::min(emb.size(), static_cast<size_t>(ROUTER_DIM));
        std::copy_n(emb.begin(), copy_dim, entry.centroid.begin());
        // Zero-pad if embedding is shorter
        for (size_t i = copy_dim; i < ROUTER_DIM; ++i) {
            entry.centroid[i] = 0.0;
        }
        centroids_[id] = std::move(entry);
    }
}

void CentroidRouter::update_centroids(
    const std::vector<std::pair<ConceptId, std::vector<double>>>& updates,
    double momentum)
{
    for (const auto& [id, new_centroid] : updates) {
        auto it = centroids_.find(id);
        if (it == centroids_.end()) {
            set_centroid(id, new_centroid);
            continue;
        }
        // EMA update: centroid = momentum * old + (1-momentum) * new
        auto& c = it->second.centroid;
        assert(new_centroid.size() == ROUTER_DIM);
        for (size_t i = 0; i < ROUTER_DIM; ++i) {
            c[i] = momentum * c[i] + (1.0 - momentum) * new_centroid[i];
        }
    }
}

void CentroidRouter::backward(
    const std::vector<double>& h,
    const std::vector<RouteResult>& routes,
    const std::vector<double>& d_weights,
    double lr)
{
    assert(routes.size() == d_weights.size());

    // d_score_i / d_centroid_i = h  (dot product derivative)
    // d_weight_i / d_score_i = softmax jacobian (simplified: w_i * (1 - w_i) for diagonal)
    // d_L / d_centroid_i = d_L/d_weight_i * d_weight_i/d_score_i * h

    for (size_t idx = 0; idx < routes.size(); ++idx) {
        auto it = centroids_.find(routes[idx].concept_id);
        if (it == centroids_.end()) continue;

        double w = routes[idx].weight;
        double d_score = d_weights[idx] * w * (1.0 - w);

        auto& c = it->second.centroid;
        for (size_t i = 0; i < ROUTER_DIM && i < h.size(); ++i) {
            c[i] -= lr * d_score * h[i];
        }
    }
}

void CentroidRouter::record_activation(const std::vector<RouteResult>& routes) {
    for (const auto& r : routes) {
        auto it = centroids_.find(r.concept_id);
        if (it != centroids_.end()) {
            it->second.activation_count += r.weight;
        }
    }
    total_activations_ += 1.0;
}

double CentroidRouter::get_activation_frequency(ConceptId id) const {
    auto it = centroids_.find(id);
    if (it == centroids_.end() || total_activations_ == 0.0) return 0.0;
    return it->second.activation_count / total_activations_;
}

double CentroidRouter::compute_balance_loss() const {
    if (centroids_.empty() || total_activations_ == 0.0) return 0.0;

    // Load balancing loss: sum(f_i * P_i) where f_i = fraction of queries
    // and P_i = fraction of routing probability
    double N = static_cast<double>(centroids_.size());
    double loss = 0.0;
    for (const auto& [id, entry] : centroids_) {
        double f = entry.activation_count / total_activations_;
        loss += f * f * N;  // Encourage uniform distribution
    }
    return loss;
}

double CentroidRouter::dot_product(
    const std::vector<double>& a,
    const std::vector<double>& b) const
{
    size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

} // namespace convergence
} // namespace brain19
