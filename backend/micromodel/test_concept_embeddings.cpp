#include "concept_embedding_store.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

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

void test_auto_create() {
    TEST("Auto-create from hash");
    ConceptEmbeddingStore store;
    assert(!store.has(42));
    const auto& emb = store.get(42);
    assert(store.has(42));
    assert(store.size() == 1);

    // Should be non-zero
    double sum = 0.0;
    for (double v : emb) sum += std::abs(v);
    assert(sum > 0.0);
    PASS();
}

void test_deterministic() {
    TEST("Hash initialization is deterministic");
    ConceptEmbeddingStore s1, s2;
    const auto& e1 = s1.get(100);
    const auto& e2 = s2.get(100);
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        assert(e1[i] == e2[i]);
    }
    PASS();
}

void test_different_concepts_different_embeddings() {
    TEST("Different concepts get different embeddings");
    ConceptEmbeddingStore store;
    const auto& e1 = store.get(1);
    const auto& e2 = store.get(2);
    bool all_same = true;
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        if (e1[i] != e2[i]) { all_same = false; break; }
    }
    assert(!all_same);
    PASS();
}

void test_set() {
    TEST("Explicit set");
    ConceptEmbeddingStore store;
    Vec10 custom{};
    custom[0] = 1.0;
    store.set(99, custom);
    assert(store.has(99));
    assert(store.get(99)[0] == 1.0);
    PASS();
}

void test_nudge() {
    TEST("Nudge moves embedding toward target");
    ConceptEmbeddingStore store;
    Vec10 target{};
    target.fill(1.0);

    store.get(50);  // auto-create
    auto before = store.get(50);
    store.nudge(50, target, 0.5);
    const auto& after = store.get(50);

    // After nudge, values should be closer to 1.0
    double dist_before = 0.0, dist_after = 0.0;
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        dist_before += (before[i] - 1.0) * (before[i] - 1.0);
        dist_after += (after[i] - 1.0) * (after[i] - 1.0);
    }
    assert(dist_after < dist_before);
    PASS();
}

void test_similarity_self() {
    TEST("Self-similarity is 1.0");
    ConceptEmbeddingStore store;
    store.get(10);
    double sim = store.similarity(10, 10);
    assert(std::abs(sim - 1.0) < 1e-6);
    PASS();
}

void test_similarity_range() {
    TEST("Similarity in [-1, 1] range");
    ConceptEmbeddingStore store;
    for (ConceptId c = 1; c <= 20; ++c) store.get(c);

    for (ConceptId a = 1; a <= 20; ++a) {
        for (ConceptId b = 1; b <= 20; ++b) {
            double s = store.similarity(a, b);
            assert(s >= -1.0 - 1e-6 && s <= 1.0 + 1e-6);
        }
    }
    PASS();
}

void test_most_similar() {
    TEST("most_similar returns top-k");
    ConceptEmbeddingStore store;

    // Create 10 concepts
    for (ConceptId c = 1; c <= 10; ++c) store.get(c);

    auto top3 = store.most_similar(1, 3);
    assert(top3.size() == 3);

    // Results should be sorted by descending similarity
    for (size_t i = 1; i < top3.size(); ++i) {
        assert(top3[i-1].second >= top3[i].second);
    }

    // No self in results
    for (auto& [cid, _] : top3) {
        assert(cid != 1);
    }
    PASS();
}

void test_most_similar_small_store() {
    TEST("most_similar with k > store size");
    ConceptEmbeddingStore store;
    store.get(1);
    store.get(2);
    auto results = store.most_similar(1, 100);
    assert(results.size() == 1);  // Only concept 2
    PASS();
}

void test_clear() {
    TEST("Clear removes all");
    ConceptEmbeddingStore store;
    store.get(1);
    store.get(2);
    assert(store.size() == 2);
    store.clear();
    assert(store.size() == 0);
    assert(!store.has(1));
    PASS();
}

void test_normalized_init() {
    TEST("Hash-initialized embeddings are unit-length");
    ConceptEmbeddingStore store;
    for (ConceptId c = 1; c <= 50; ++c) {
        const auto& emb = store.get(c);
        double norm = 0.0;
        for (double v : emb) norm += v * v;
        norm = std::sqrt(norm);
        assert(std::abs(norm - 1.0) < 1e-6);
    }
    PASS();
}

int main() {
    std::cout << "=== ConceptEmbeddingStore Tests ===" << std::endl;

    test_auto_create();
    test_deterministic();
    test_different_concepts_different_embeddings();
    test_set();
    test_nudge();
    test_similarity_self();
    test_similarity_range();
    test_most_similar();
    test_most_similar_small_store();
    test_clear();
    test_normalized_init();

    std::cout << "\n=== " << tests_passed << "/" << tests_total << " PASSED ===" << std::endl;
    return (tests_passed == tests_total) ? 0 : 1;
}
