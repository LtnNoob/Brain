#include "flex_embedding.hpp"
#include "concept_embedding_store.hpp"
#include "embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/active_relation.hpp"
#include "../memory/relation_type_registry.hpp"
#include "../epistemic/epistemic_metadata.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <numeric>

using namespace brain19;

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { \
    tests_total++; \
    std::cout << "  TEST " << tests_total << ": " << name << "... "; \
} while(0)

#define PASS() do { \
    tests_passed++; \
    std::cout << "PASS" << std::endl; \
} while(0)

static constexpr double EPS = 1e-6;

static ConceptId add_cpt(LongTermMemory& ltm,
                         const std::string& label) {
    return ltm.store_concept(label, label,
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
}

// =============================================================================
// Test 1: INITIAL_DETAIL is 16 (not 0)
// =============================================================================
void test_initial_detail_config() {
    TEST("FlexConfig::INITIAL_DETAIL is 16");
    assert(FlexConfig::INITIAL_DETAIL == 16);
    PASS();
}

// =============================================================================
// Test 2: hash_init creates non-empty detail dims
// =============================================================================
void test_hash_init_detail_dims() {
    TEST("hash_init creates 16 detail dimensions");

    ConceptEmbeddingStore store;
    const auto& emb = store.get(42);

    assert(emb.detail.size() == 16);
    assert(emb.dim() == 32);  // 16 core + 16 detail

    // Detail dims should be non-zero (not all zeros)
    double sum = 0.0;
    for (double d : emb.detail) sum += std::abs(d);
    assert(sum > 0.01);  // non-trivial

    PASS();
}

// =============================================================================
// Test 3: Different concepts get different detail dims
// =============================================================================
void test_unique_detail_per_concept() {
    TEST("Different concepts have different detail dims");

    ConceptEmbeddingStore store;
    const auto& emb1 = store.get(1);
    const auto& emb2 = store.get(2);
    const auto& emb3 = store.get(1000);

    // At least some detail dims should differ
    bool any_differ_12 = false;
    bool any_differ_13 = false;
    for (size_t i = 0; i < 16; ++i) {
        if (std::abs(emb1.detail[i] - emb2.detail[i]) > EPS) any_differ_12 = true;
        if (std::abs(emb1.detail[i] - emb3.detail[i]) > EPS) any_differ_13 = true;
    }
    assert(any_differ_12);
    assert(any_differ_13);

    PASS();
}

// =============================================================================
// Test 4: dim_fraction is now non-zero
// =============================================================================
void test_dim_fraction_nonzero() {
    TEST("dim_fraction is non-zero with detail dims");

    ConceptEmbeddingStore store;
    const auto& emb1 = store.get(1);
    const auto& emb2 = store.get(2);

    double dim_fraction = static_cast<double>(
        std::min(emb1.detail.size(), emb2.detail.size())) / 496.0;

    // 16 / 496 = ~0.0323
    assert(dim_fraction > 0.03);
    assert(dim_fraction < 0.04);

    PASS();
}

// =============================================================================
// Test 5: cyclic_compress returns non-zero values
// =============================================================================
void test_cyclic_compress_nonzero() {
    TEST("cyclic_compress returns non-zero for populated detail");

    ConceptEmbeddingStore store;
    const auto& emb = store.get(42);

    // Replicate cyclic_compress logic
    std::array<double, 4> compressed{};
    for (size_t d = 0; d < emb.detail.size(); ++d) {
        compressed[d % 4] += emb.detail[d];
    }

    double total = 0.0;
    for (double c : compressed) total += std::abs(c);
    assert(total > 0.01);  // non-trivial compressed values

    PASS();
}

// =============================================================================
// Test 6: full_similarity differs from core_similarity with detail
// =============================================================================
void test_full_vs_core_similarity() {
    TEST("full_similarity differs from core_similarity when detail present");

    ConceptEmbeddingStore store;
    FlexEmbedding a = store.get(1);
    FlexEmbedding b = store.get(2);

    double cs = core_similarity(a, b);
    double fs = full_similarity(a, b);

    // They should be different (detail dims contribute to full)
    assert(std::abs(cs - fs) > EPS);

    PASS();
}

// =============================================================================
// Test 7: Nudge works with detail dims
// =============================================================================
void test_nudge_with_detail() {
    TEST("Nudge blends detail dimensions");

    ConceptEmbeddingStore store;
    FlexEmbedding before = store.get(1);

    // Create a target with specific detail values
    FlexEmbedding target;
    target.detail.resize(16, 1.0);  // all ones
    for (size_t i = 0; i < CORE_DIM; ++i) target.core[i] = 0.5;

    store.nudge(1, target, 0.5);

    FlexEmbedding after = store.get(1);

    // Detail should be blended: (1-0.5)*before + 0.5*1.0
    for (size_t i = 0; i < 16; ++i) {
        double expected = 0.5 * before.detail[i] + 0.5 * 1.0;
        assert(std::abs(after.detail[i] - expected) < EPS);
    }

    PASS();
}

// =============================================================================
// Test 8: Nudge expands detail dims when target has more
// =============================================================================
void test_nudge_expands_detail() {
    TEST("Nudge expands detail dims when target has more");

    ConceptEmbeddingStore store;
    store.get(1);  // 16 detail dims

    FlexEmbedding big_target;
    big_target.detail.resize(32, 0.5);  // 32 detail dims
    for (size_t i = 0; i < CORE_DIM; ++i) big_target.core[i] = 0.0;

    store.nudge(1, big_target, 0.2);

    FlexEmbedding after = store.get(1);
    assert(after.detail.size() == 32);

    // New dims 16-31 should be 0.2 * 0.5 = 0.1
    for (size_t i = 16; i < 32; ++i) {
        assert(std::abs(after.detail[i] - 0.1) < EPS);
    }

    PASS();
}

// =============================================================================
// Test 9: learn_from_graph makes connected concepts more similar
// =============================================================================
void test_learn_from_graph_clustering() {
    TEST("learn_from_graph clusters connected concepts");

    LongTermMemory ltm;
    auto cat     = add_cpt(ltm, "Cat");
    auto dog     = add_cpt(ltm, "Dog");
    auto mammal  = add_cpt(ltm, "Mammal");
    auto rock    = add_cpt(ltm, "Rock");  // unconnected

    // Cat and Dog both IS_A Mammal
    ltm.add_relation(cat, mammal, RelationType::IS_A, 1.0);
    ltm.add_relation(dog, mammal, RelationType::IS_A, 1.0);

    ConceptEmbeddingStore store;

    // Measure similarity before learning
    double sim_cat_dog_before = store.similarity(cat, dog);
    (void)store.similarity(cat, rock);  // ensure rock embedding exists

    // Learn from graph (multiple iterations for convergence)
    auto result = store.learn_from_graph(ltm, 0.1, 10);

    assert(result.concepts_updated > 0);

    // Measure similarity after learning
    double sim_cat_dog_after = store.similarity(cat, dog);
    double sim_cat_rock_after = store.similarity(cat, rock);

    // Cat-Dog similarity should increase (connected via Mammal)
    assert(sim_cat_dog_after > sim_cat_dog_before);

    // Cat-Rock similarity should stay roughly the same or decrease
    // (Rock is unconnected, not pulled toward Cat/Dog cluster)
    // NOTE: Since Rock has no relations, it doesn't move, but Cat does.
    // The key test: cat-dog should be MORE similar than cat-rock after learning
    assert(sim_cat_dog_after > sim_cat_rock_after);

    PASS();
}

// =============================================================================
// Test 10: learn_from_graph result statistics
// =============================================================================
void test_learn_result_stats() {
    TEST("learn_from_graph returns correct statistics");

    LongTermMemory ltm;
    auto a = add_cpt(ltm, "A");
    auto b = add_cpt(ltm, "B");
    ltm.add_relation(a, b, RelationType::IS_A, 1.0);

    ConceptEmbeddingStore store;
    auto result = store.learn_from_graph(ltm, 0.05, 3);

    assert(result.iterations == 3);
    assert(result.concepts_updated > 0);
    assert(result.total_neighbors > 0);

    PASS();
}

// =============================================================================
// Test 11: Relation type embeddings have non-zero dims 10-15
// =============================================================================
void test_relation_type_dims_10_15() {
    TEST("Relation type embeddings have non-zero dims 10-15");

    auto& reg = RelationTypeRegistry::instance();

    // Check all built-in types have non-zero in dims 10-15
    auto builtins = reg.builtin_types();
    for (auto type : builtins) {
        const auto& emb = reg.get_embedding(type);
        double sum = 0.0;
        for (size_t i = 10; i < 16; ++i) {
            sum += std::abs(emb.core[i]);
        }
        assert(sum > 0.1);  // at least some non-zero values
    }

    // IS_A should have high transitivity (dim 10) and inheritability (dim 11)
    const auto& isa_emb = reg.get_embedding(RelationType::IS_A);
    assert(isa_emb.core[10] > 0.5);  // high transitivity
    assert(isa_emb.core[11] > 0.5);  // high inheritability

    // CONTRADICTS should have high exclusivity (dim 13)
    const auto& contr_emb = reg.get_embedding(RelationType::CONTRADICTS);
    assert(contr_emb.core[13] > 0.5);

    // SIMILAR_TO should have high symmetry (dim 12)
    const auto& sim_emb = reg.get_embedding(RelationType::SIMILAR_TO);
    assert(sim_emb.core[12] > 0.5);

    PASS();
}

// =============================================================================
// Test 12: EmbeddingManager.train_embeddings works end-to-end
// =============================================================================
void test_embedding_manager_train() {
    TEST("EmbeddingManager.train_embeddings end-to-end");

    LongTermMemory ltm;
    auto a = add_cpt(ltm, "A");
    auto b = add_cpt(ltm, "B");
    auto c = add_cpt(ltm, "C");
    ltm.add_relation(a, b, RelationType::IS_A, 0.9);
    ltm.add_relation(b, c, RelationType::IS_A, 0.8);

    EmbeddingManager em;
    auto result = em.train_embeddings(ltm, 0.05, 5);

    assert(result.iterations == 5);
    assert(result.concepts_updated > 0);

    // Concepts should now have embeddings
    assert(em.concept_embeddings().has(a));
    assert(em.concept_embeddings().has(b));
    assert(em.concept_embeddings().has(c));

    // All should have 16 detail dims
    assert(em.concept_embeddings().get_or_default(a).detail.size() == 16);

    PASS();
}

// =============================================================================
// Test 13: Hierarchy creates expected embedding structure
// =============================================================================
void test_hierarchy_embedding_structure() {
    TEST("IS_A hierarchy creates structured embeddings");

    LongTermMemory ltm;
    auto animal  = add_cpt(ltm, "Animal");
    auto mammal  = add_cpt(ltm, "Mammal");
    auto cat     = add_cpt(ltm, "Cat");
    auto dog     = add_cpt(ltm, "Dog");
    auto fish    = add_cpt(ltm, "Fish");

    ltm.add_relation(mammal, animal, RelationType::IS_A, 1.0);
    ltm.add_relation(cat, mammal, RelationType::IS_A, 1.0);
    ltm.add_relation(dog, mammal, RelationType::IS_A, 1.0);
    ltm.add_relation(fish, animal, RelationType::IS_A, 1.0);

    ConceptEmbeddingStore store;
    store.learn_from_graph(ltm, 0.1, 20);

    // Cat and Dog (siblings under Mammal) should be more similar
    // than Cat and Fish (distant: Mammal vs Fish under Animal)
    double sim_cat_dog = store.similarity(cat, dog);
    double sim_cat_fish = store.similarity(cat, fish);

    assert(sim_cat_dog > sim_cat_fish);

    PASS();
}

// =============================================================================
// Test 14: Context embeddings still work
// =============================================================================
void test_context_embeddings_unchanged() {
    TEST("Context embeddings still work (backward compat)");

    EmbeddingManager em;
    const auto& query = em.query_context();
    const auto& recall = em.recall_context();

    // Should have 16 core dims
    assert(query.dim() >= CORE_DIM);
    assert(recall.dim() >= CORE_DIM);

    // Should be different
    bool differ = false;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        if (std::abs(query.core[i] - recall.core[i]) > EPS) { differ = true; break; }
    }
    assert(differ);

    PASS();
}

// =============================================================================
// Test 15: get_or_default returns embedding with detail dims
// =============================================================================
void test_get_or_default_has_detail() {
    TEST("get_or_default returns embedding with 16 detail dims");

    ConceptEmbeddingStore store;
    auto emb = store.get_or_default(999);

    assert(emb.detail.size() == 16);
    assert(emb.dim() == 32);

    PASS();
}

// =============================================================================
// Test 16: Deterministic - same concept ID always gives same embedding
// =============================================================================
void test_deterministic_hash_init() {
    TEST("hash_init is deterministic for same concept ID");

    ConceptEmbeddingStore store1;
    ConceptEmbeddingStore store2;

    auto emb1 = store1.get_or_default(42);
    auto emb2 = store2.get_or_default(42);

    for (size_t i = 0; i < CORE_DIM; ++i) {
        assert(std::abs(emb1.core[i] - emb2.core[i]) < EPS);
    }
    for (size_t i = 0; i < 16; ++i) {
        assert(std::abs(emb1.detail[i] - emb2.detail[i]) < EPS);
    }

    PASS();
}

// =============================================================================
// Test 17: grow() still works on embeddings with initial detail
// =============================================================================
void test_grow_from_initial() {
    TEST("grow() works on embeddings that already have 16 detail dims");

    ConceptEmbeddingStore store;
    FlexEmbedding emb = store.get(1);
    assert(emb.detail.size() == 16);

    std::mt19937 rng(42);
    emb.grow(8, rng);
    assert(emb.detail.size() == 24);

    // New dims should be near zero (noise)
    for (size_t i = 16; i < 24; ++i) {
        assert(std::abs(emb.detail[i]) < 0.1);
    }

    PASS();
}

// =============================================================================
// Test 18: record_activation tracks activation count and tick
// =============================================================================
void test_record_activation() {
    TEST("record_activation tracks activation count");

    ConceptEmbeddingStore store;
    store.get(1);  // ensure embedding exists

    // Initially no metadata
    auto m0 = store.meta(1);
    assert(m0.activation_count == 0);

    // Record activations
    store.record_activation(1, 10);
    store.record_activation(1, 11);
    store.record_activation(1, 12);

    auto m3 = store.meta(1);
    assert(m3.activation_count == 3);
    assert(m3.last_access_tick == 12);

    PASS();
}

// =============================================================================
// Test 19: record_gradient tracks EMA of gradient magnitude
// =============================================================================
void test_record_gradient() {
    TEST("record_gradient tracks EMA of gradient magnitude");

    ConceptEmbeddingStore store;
    store.get(1);

    // Record several gradients
    store.record_gradient(1, 1.0f);  // EMA: 0.0*0.9 + 0.1*1.0 = 0.1
    assert(std::abs(store.meta(1).avg_grad_magnitude - 0.1f) < 0.001f);

    store.record_gradient(1, 1.0f);  // EMA: 0.1*0.9 + 0.1*1.0 = 0.19
    assert(std::abs(store.meta(1).avg_grad_magnitude - 0.19f) < 0.001f);

    PASS();
}

// =============================================================================
// Test 20: learn_from_graph now tracks gradient and relation count
// =============================================================================
void test_learn_from_graph_tracks_meta() {
    TEST("learn_from_graph populates EmbeddingMeta");

    LongTermMemory ltm;
    auto a = add_cpt(ltm, "A");
    auto b = add_cpt(ltm, "B");
    auto c = add_cpt(ltm, "C");
    ltm.add_relation(a, b, RelationType::IS_A, 0.9);
    ltm.add_relation(a, c, RelationType::HAS_PROPERTY, 0.8);
    ltm.add_relation(b, c, RelationType::IS_A, 0.7);

    ConceptEmbeddingStore store;
    store.learn_from_graph(ltm, 0.05, 3);

    // Concept A has 2 outgoing + 0 incoming = 2 relations
    auto ma = store.meta(a);
    assert(ma.avg_grad_magnitude > 0.0f);  // should have been nudged
    assert(ma.relation_count >= 2);        // at least outgoing neighbors

    // Concept B has 1 outgoing + 1 incoming = 2 relations
    auto mb = store.meta(b);
    assert(mb.avg_grad_magnitude > 0.0f);

    PASS();
}

// =============================================================================
// Test 21: DimensionManager should_grow requires all conditions
// =============================================================================
void test_dimension_manager_should_grow() {
    TEST("DimensionManager::should_grow requires all conditions");

    EmbeddingMeta m;
    m.activation_count = 5;   // < 20: too few
    m.avg_grad_magnitude = 0.1f;
    m.relation_count = 30;    // >= 32*0.8=25.6
    m.last_resize_tick = 0;
    assert(!DimensionManager::should_grow(m, 32, 200));

    m.activation_count = 25;  // >= 20: enough
    assert(DimensionManager::should_grow(m, 32, 200));

    m.avg_grad_magnitude = 0.01f;  // < GROWTH_GRAD_MIN: too low
    assert(!DimensionManager::should_grow(m, 32, 200));

    m.avg_grad_magnitude = 0.1f;
    m.last_resize_tick = 150;  // cooldown not passed (200 - 150 < 100)
    assert(!DimensionManager::should_grow(m, 32, 200));

    m.last_resize_tick = 50;   // cooldown passed (200 - 50 >= 100)
    assert(DimensionManager::should_grow(m, 32, 200));

    // At MAX_DIM: no growth possible
    assert(!DimensionManager::should_grow(m, FlexConfig::MAX_DIM, 200));

    PASS();
}

// =============================================================================
// Test 22: evaluate_and_resize grows qualifying embeddings
// =============================================================================
void test_evaluate_and_resize() {
    TEST("evaluate_and_resize grows qualifying embeddings");

    LongTermMemory ltm;
    auto a = add_cpt(ltm, "A");
    auto b = add_cpt(ltm, "B");
    // Create enough relations for concept A
    for (int i = 0; i < 30; ++i) {
        auto x = add_cpt(ltm, "X" + std::to_string(i));
        ltm.add_relation(a, x, RelationType::HAS_PROPERTY, 0.8);
    }

    ConceptEmbeddingStore store;
    store.learn_from_graph(ltm, 0.1, 3);

    size_t initial_dim_a = store.get(a).detail.size();

    // Simulate high activation and gradient for concept A
    for (int i = 0; i < 25; ++i) {
        store.record_activation(a, static_cast<uint32_t>(i));
    }
    // Push avg_grad_magnitude above threshold
    for (int i = 0; i < 20; ++i) {
        store.record_gradient(a, 0.5f);
    }
    // Ensure relation_count is high enough
    store.update_relation_count(a, 40);

    // Evaluate at tick 500 (EVAL_INTERVAL)
    auto result = store.evaluate_and_resize(500);

    // Concept A should have grown
    assert(result.grown >= 1);
    assert(store.get(a).detail.size() > initial_dim_a);

    // Concept B should NOT have grown (low activation)
    assert(store.get(b).detail.size() == 16);

    PASS();
}

// =============================================================================
// Test 23: DimensionManager should_shrink on old unused embeddings
// =============================================================================
void test_should_shrink_old_unused() {
    TEST("DimensionManager shrinks old unused embeddings");

    EmbeddingMeta m;
    m.last_access_tick = 0;
    // Very old (current_tick - last_access > 50000)
    assert(DimensionManager::should_shrink(m, 60000));
    // Recently used
    m.last_access_tick = 59000;
    assert(!DimensionManager::should_shrink(m, 60000));

    PASS();
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n=== FlexEmbedding Activation Tests ===\n\n";

    test_initial_detail_config();
    test_hash_init_detail_dims();
    test_unique_detail_per_concept();
    test_dim_fraction_nonzero();
    test_cyclic_compress_nonzero();
    test_full_vs_core_similarity();
    test_nudge_with_detail();
    test_nudge_expands_detail();
    test_learn_from_graph_clustering();
    test_learn_result_stats();
    test_relation_type_dims_10_15();
    test_embedding_manager_train();
    test_hierarchy_embedding_structure();
    test_context_embeddings_unchanged();
    test_get_or_default_has_detail();
    test_deterministic_hash_init();
    test_grow_from_initial();
    test_record_activation();
    test_record_gradient();
    test_learn_from_graph_tracks_meta();
    test_dimension_manager_should_grow();
    test_evaluate_and_resize();
    test_should_shrink_old_unused();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_total
              << " passed ===\n";

    if (tests_passed == tests_total) {
        std::cout << "ALL TESTS PASSED\n\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED\n\n";
        return 1;
    }
}
