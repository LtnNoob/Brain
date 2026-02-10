#pragma once

// Lock hierarchy (always acquire in this order to prevent deadlock):
// 1. SharedLTM
// 2. SharedSTM
// 3. SharedRegistry
// 4. SharedEmbeddings

#include "../micromodel/embedding_manager.hpp"
#include <shared_mutex>

namespace brain19 {

// Thread-safe wrapper around EmbeddingManager.
// shared_mutex: mostly reads, rare writes on auto-create contexts.
// OPT-IN: single-threaded code can use EmbeddingManager directly.
class SharedEmbeddings {
public:
    explicit SharedEmbeddings(EmbeddingManager& em) : em_(em) {}

    SharedEmbeddings(const SharedEmbeddings&) = delete;
    SharedEmbeddings& operator=(const SharedEmbeddings&) = delete;

    // === READ operations (shared_lock) ===

    const Vec10& get_relation_embedding(RelationType type) const {
        std::shared_lock lock(mtx_);
        return em_.get_relation_embedding(type);
    }

    bool has_context(const std::string& name) const {
        std::shared_lock lock(mtx_);
        return em_.has_context(name);
    }

    std::vector<std::string> get_context_names() const {
        std::shared_lock lock(mtx_);
        return em_.get_context_names();
    }

    Vec10 make_context_embedding(const std::string& name) const {
        std::shared_lock lock(mtx_);
        return em_.make_context_embedding(name);
    }

    Vec10 make_target_embedding(size_t context_hash, uint64_t source_id, uint64_t target_id) const {
        std::shared_lock lock(mtx_);
        return em_.make_target_embedding(context_hash, source_id, target_id);
    }

    // === WRITE operations (unique_lock) — auto-create on first access ===

    const Vec10& get_context_embedding(const std::string& name) {
        // Fast path: check with shared lock first
        {
            std::shared_lock lock(mtx_);
            if (em_.has_context(name)) {
                return em_.get_context_embedding(name);
            }
        }
        // Slow path: need to create — upgrade to unique lock
        std::unique_lock lock(mtx_);
        return em_.get_context_embedding(name);
    }

    const Vec10& query_context() { return get_context_embedding("query"); }
    const Vec10& recall_context() { return get_context_embedding("recall"); }
    const Vec10& creative_context() { return get_context_embedding("creative"); }
    const Vec10& analytical_context() { return get_context_embedding("analytical"); }

    // === Direct access (unique_lock — for persistence) ===

    const std::array<Vec10, NUM_RELATION_TYPES>& relation_embeddings() const {
        std::shared_lock lock(mtx_);
        return em_.relation_embeddings();
    }

    std::array<Vec10, NUM_RELATION_TYPES>& relation_embeddings_mut() {
        std::unique_lock lock(mtx_);
        return em_.relation_embeddings_mut();
    }

    const std::unordered_map<std::string, Vec10>& context_embeddings() const {
        std::shared_lock lock(mtx_);
        return em_.context_embeddings();
    }

    std::unordered_map<std::string, Vec10>& context_embeddings_mut() {
        std::unique_lock lock(mtx_);
        return em_.context_embeddings_mut();
    }

private:
    EmbeddingManager& em_;
    mutable std::shared_mutex mtx_;
};

} // namespace brain19
