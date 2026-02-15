#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace brain19 {

// =============================================================================
// FLEX EMBEDDING — Variable-Dimensional Embeddings
// =============================================================================
//
// Core (16D, stack-allocated, always present) + Detail (0-496D, heap-allocated).
// Two-Phase Similarity: core_similarity for fast filtering, full_similarity for reranking.
// Growth/Shrink logic for organic dimension management.
//

namespace FlexConfig {
    static constexpr size_t CORE_DIM         = 16;
    static constexpr size_t MIN_DIM          = 16;    // = Core only
    static constexpr size_t MAX_DIM          = 512;
    static constexpr size_t INITIAL_DETAIL   = 16;    // 16 detail dims from creation
    static constexpr size_t GROWTH_MIN       = 4;     // mindestens 4 Dims auf einmal
    static constexpr size_t GROWTH_MAX       = 64;    // maximal 64 Dims auf einmal
    static constexpr double SHRINK_THRESHOLD = 0.01;  // Dim gilt als "tot" unter diesem Wert
    static constexpr double GROWTH_GRAD_MIN  = 0.05;  // Min Gradient-Magnitude fuer Growth
    static constexpr float  UTILIZATION_SHRINK = 0.3f; // Shrink wenn <30% Dims aktiv
    static constexpr uint32_t RESIZE_COOLDOWN = 100;  // Ticks zwischen Resizes
    static constexpr uint32_t EVAL_INTERVAL   = 500;  // Ticks zwischen Evaluations
    static constexpr size_t   MAX_RESIZE_BATCH = 50;  // Max Konzepte pro Eval-Runde
    static constexpr size_t REL_MAX_DIM = 128;        // Relations brauchen weniger Dims
}

static constexpr size_t CORE_DIM = FlexConfig::CORE_DIM;

using CoreVec = std::array<double, CORE_DIM>;

// =============================================================================
// FlexEmbedding
// =============================================================================

struct FlexEmbedding {
    CoreVec core{};                    // immer 16D, stack-allocated
    std::vector<double> detail;        // 0..496D, heap-allocated

    FlexEmbedding() = default;

    // Construct from initializer list (for core values, detail empty)
    FlexEmbedding(std::initializer_list<double> init) {
        size_t i = 0;
        for (double v : init) {
            if (i < CORE_DIM) {
                core[i] = v;
            } else {
                detail.push_back(v);
            }
            ++i;
        }
    }

    size_t dim() const { return CORE_DIM + detail.size(); }

    double& operator[](size_t i) {
        return i < CORE_DIM ? core[i] : detail[i - CORE_DIM];
    }
    double operator[](size_t i) const {
        return i < CORE_DIM ? core[i] : detail[i - CORE_DIM];
    }

    // Grow by n dimensions (initialized near zero with small noise)
    void grow(size_t n, std::mt19937& rng) {
        std::normal_distribution<double> noise(0.0, 0.01);
        size_t new_size = std::min(detail.size() + n, FlexConfig::MAX_DIM - CORE_DIM);
        detail.reserve(new_size);
        while (detail.size() < new_size) {
            detail.push_back(noise(rng));
        }
    }

    // Shrink: remove trailing dimensions with magnitude < threshold
    size_t shrink(double threshold = FlexConfig::SHRINK_THRESHOLD) {
        while (!detail.empty() && std::abs(detail.back()) < threshold) {
            detail.pop_back();
        }
        detail.shrink_to_fit();
        return detail.size();
    }

    // Serialize to binary stream
    void write(std::ostream& os) const {
        uint16_t detail_dim = static_cast<uint16_t>(detail.size());
        os.write(reinterpret_cast<const char*>(core.data()), CORE_DIM * sizeof(double));
        os.write(reinterpret_cast<const char*>(&detail_dim), sizeof(detail_dim));
        if (detail_dim > 0) {
            os.write(reinterpret_cast<const char*>(detail.data()), detail_dim * sizeof(double));
        }
    }

    // Deserialize from binary stream
    void read(std::istream& is) {
        uint16_t detail_dim;
        is.read(reinterpret_cast<char*>(core.data()), CORE_DIM * sizeof(double));
        is.read(reinterpret_cast<char*>(&detail_dim), sizeof(detail_dim));
        detail.resize(detail_dim);
        if (detail_dim > 0) {
            is.read(reinterpret_cast<char*>(detail.data()), detail_dim * sizeof(double));
        }
    }

    // Migration helper: create FlexEmbedding from old 10D array
    static FlexEmbedding from_vec10(const std::array<double, 10>& old) {
        FlexEmbedding e;
        for (size_t i = 0; i < 10; ++i) e.core[i] = old[i];
        for (size_t i = 10; i < CORE_DIM; ++i) e.core[i] = 0.0;
        return e;
    }
};

// =============================================================================
// Similarity Functions
// =============================================================================

// Phase 1: Core-only similarity (fast, for candidate filtering)
inline double core_similarity(const FlexEmbedding& a, const FlexEmbedding& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        dot += a.core[i] * b.core[i];
        na  += a.core[i] * a.core[i];
        nb  += b.core[i] * b.core[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-12 ? dot / denom : 0.0;
}

// Phase 2: Full similarity (for Top-K candidates)
// Computes over shared dimensions; extra dims increase norm -> similarity naturally decreases
inline double full_similarity(const FlexEmbedding& a, const FlexEmbedding& b) {
    size_t shared = std::min(a.dim(), b.dim());

    double dot = 0, na = 0, nb = 0;

    // Shared dimensions
    for (size_t i = 0; i < shared; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }

    // Remaining dimensions of the larger vector count toward its norm
    if (a.dim() > shared) {
        for (size_t i = shared; i < a.dim(); ++i) na += a[i] * a[i];
    }
    if (b.dim() > shared) {
        for (size_t i = shared; i < b.dim(); ++i) nb += b[i] * b[i];
    }

    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-12 ? dot / denom : 0.0;
}

// =============================================================================
// EmbeddingMeta — tracks concept usage for Growth/Shrink decisions
// =============================================================================

struct EmbeddingMeta {
    uint64_t concept_id = 0;
    uint32_t activation_count = 0;    // wie oft aktiviert
    uint32_t relation_count = 0;      // Anzahl Relations
    float    avg_grad_magnitude = 0;  // durchschnittliche Gradient-Groesse
    float    dim_utilization = 1.0;   // Anteil Dimensionen > threshold
    uint32_t last_resize_tick = 0;    // letzter Growth/Shrink Zeitpunkt
    uint32_t last_access_tick = 0;    // letzte Nutzung
};

// =============================================================================
// DimensionManager — evaluates and applies growth/shrink
// =============================================================================

class DimensionManager {
public:
    // Compute how many dimensions to grow by (0 = no growth needed)
    static size_t growth_amount(const EmbeddingMeta& /*meta*/, size_t current_dim) {
        size_t base = std::max<size_t>(FlexConfig::GROWTH_MIN, current_dim / 4);
        return std::min<size_t>(base, FlexConfig::MAX_DIM - current_dim);
    }

    // Check if embedding should grow
    static bool should_grow(const EmbeddingMeta& meta, size_t current_dim, uint32_t current_tick) {
        if (current_dim >= FlexConfig::MAX_DIM) return false;
        if (meta.activation_count < 20) return false;
        if (meta.avg_grad_magnitude < FlexConfig::GROWTH_GRAD_MIN) return false;
        if (meta.relation_count < static_cast<uint32_t>(current_dim * 0.8)) return false;
        if (current_tick - meta.last_resize_tick < FlexConfig::RESIZE_COOLDOWN) return false;
        return true;
    }

    // Check if embedding should shrink
    static bool should_shrink(const EmbeddingMeta& meta, uint32_t current_tick) {
        // Aggressive shrink for very old unused embeddings
        if (current_tick - meta.last_access_tick > 50000) return true;
        // Normal shrink
        if (meta.dim_utilization < FlexConfig::UTILIZATION_SHRINK &&
            current_tick - meta.last_access_tick > 5000 &&
            current_tick - meta.last_resize_tick > FlexConfig::RESIZE_COOLDOWN * 2) {
            return true;
        }
        return false;
    }

    // Compute dimension utilization (fraction of detail dims above threshold)
    static float compute_utilization(const FlexEmbedding& emb) {
        if (emb.detail.empty()) return 1.0f;
        size_t alive = 0;
        for (double d : emb.detail) {
            if (std::abs(d) > FlexConfig::SHRINK_THRESHOLD) ++alive;
        }
        return static_cast<float>(alive) / static_cast<float>(emb.detail.size());
    }
};

} // namespace brain19
