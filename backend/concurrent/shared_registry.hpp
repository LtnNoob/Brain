#pragma once

// Lock hierarchy (always acquire in this order to prevent deadlock):
// 1. SharedLTM
// 2. SharedSTM
// 3. SharedRegistry
// 4. SharedEmbeddings

#include "../micromodel/micro_model_registry.hpp"
#include <shared_mutex>
#include <mutex>
#include <memory>
#include <unordered_map>

namespace brain19 {

// Thread-safe wrapper around MicroModelRegistry.
// shared_mutex at registry level; per-model mutexes for training.
// OPT-IN: single-threaded code can use MicroModelRegistry directly.
class SharedRegistry {
public:
    explicit SharedRegistry(MicroModelRegistry& reg) : reg_(reg) {}

    SharedRegistry(const SharedRegistry&) = delete;
    SharedRegistry& operator=(const SharedRegistry&) = delete;

    // === WRITE operations (unique_lock on registry) ===

    bool create_model(ConceptId cid) {
        std::unique_lock lock(mtx_);
        bool ok = reg_.create_model(cid);
        if (ok) {
            model_mutexes_[cid] = std::make_unique<std::mutex>();
        }
        return ok;
    }

    bool remove_model(ConceptId cid) {
        std::unique_lock lock(mtx_);
        bool ok = reg_.remove_model(cid);
        if (ok) {
            model_mutexes_.erase(cid);
        }
        return ok;
    }

    size_t ensure_models_for(const LongTermMemory& ltm) {
        std::unique_lock lock(mtx_);
        size_t created = reg_.ensure_models_for(ltm);
        // Ensure mutexes for any new models
        for (auto id : reg_.get_model_ids()) {
            if (!model_mutexes_.count(id))
                model_mutexes_[id] = std::make_unique<std::mutex>();
        }
        return created;
    }

    void clear() {
        std::unique_lock lock(mtx_);
        reg_.clear();
        model_mutexes_.clear();
    }

    // === READ operations (shared_lock on registry) ===

    const MicroModel* get_model(ConceptId cid) const {
        std::shared_lock lock(mtx_);
        return reg_.get_model(cid);
    }

    bool has_model(ConceptId cid) const {
        std::shared_lock lock(mtx_);
        return reg_.has_model(cid);
    }

    std::vector<ConceptId> get_model_ids() const {
        std::shared_lock lock(mtx_);
        return reg_.get_model_ids();
    }

    size_t size() const {
        std::shared_lock lock(mtx_);
        return reg_.size();
    }

    // === Per-model access for training (lock model individually) ===
    // Usage: auto* model = registry.lock_model_for_training(cid);
    //        // ... train model ...
    //        registry.unlock_model(cid);
    //
    // Or use the RAII helper: auto guard = registry.model_guard(cid);

    MicroModel* lock_model_for_training(ConceptId cid) {
        std::shared_lock lock(mtx_);
        auto* model = reg_.get_model(cid);
        if (!model) return nullptr;
        model_mutexes_.at(cid)->lock();
        return model;
    }

    void unlock_model(ConceptId cid) {
        model_mutexes_.at(cid)->unlock();
    }

    // RAII guard for per-model locking — holds registry shared_lock to prevent
    // use-after-free if remove_model() is called concurrently
    class ModelGuard {
    public:
        ModelGuard(SharedRegistry& reg, ConceptId cid)
            : reg_(reg), cid_(cid), reg_lock_(reg.mtx_), model_(nullptr) {
            model_ = reg_.reg_.get_model(cid);
            if (model_) {
                reg_.model_mutexes_.at(cid)->lock();
            }
        }
        ~ModelGuard() {
            if (model_) reg_.model_mutexes_.at(cid_)->unlock();
            // reg_lock_ released here — after model mutex unlock
        }
        ModelGuard(const ModelGuard&) = delete;
        ModelGuard& operator=(const ModelGuard&) = delete;
        MicroModel* operator->() { return model_; }
        MicroModel* get() { return model_; }
        explicit operator bool() const { return model_ != nullptr; }
    private:
        SharedRegistry& reg_;
        ConceptId cid_;
        std::shared_lock<std::shared_mutex> reg_lock_;
        MicroModel* model_;
    };

    ModelGuard model_guard(ConceptId cid) {
        return ModelGuard(*this, cid);
    }

private:
    MicroModelRegistry& reg_;
    mutable std::shared_mutex mtx_;
    mutable std::unordered_map<ConceptId, std::unique_ptr<std::mutex>> model_mutexes_;
};

} // namespace brain19
