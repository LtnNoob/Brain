#include "micro_model.hpp"
#include "micro_model_registry.hpp"
#include "embedding_manager.hpp"
#include "micro_trainer.hpp"
#include "relevance_map.hpp"
#include "persistence.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../epistemic/epistemic_metadata.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdio>

using namespace brain19;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  TEST: " << name << "... "; \
    try {

#define END_TEST \
        std::cout << "PASSED" << std::endl; \
        tests_passed++; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << std::endl; \
        tests_failed++; \
    }

#define ASSERT(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond);

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) throw std::runtime_error( \
        std::string("Assertion failed: ") + #a + " == " + std::to_string(a) + " != " + #b + " == " + std::to_string(b));

#define ASSERT_GT(a, b) \
    if (!((a) > (b))) throw std::runtime_error( \
        std::string("Assertion failed: ") + #a + " not > " + #b);

#define ASSERT_LT(a, b) \
    if (!((a) < (b))) throw std::runtime_error( \
        std::string("Assertion failed: ") + #a + " not < " + #b);

#define ASSERT_NEAR(a, b, eps) \
    if (std::abs((a) - (b)) > (eps)) throw std::runtime_error( \
        std::string("Assertion failed: |") + #a + " - " + #b + "| > " + #eps + \
        " (got " + std::to_string(a) + " vs " + std::to_string(b) + ")");

// =============================================================================
// MicroModel unit tests
// =============================================================================

void test_micro_model() {
    std::cout << "\n=== MicroModel Tests ===" << std::endl;

    TEST("Default model predicts in (0,1)")
    {
        MicroModel model;
        Vec10 e, c;
        e.fill(0.1);
        c.fill(0.1);
        double pred = model.predict(e, c);
        ASSERT_GT(pred, 0.0);
        ASSERT_LT(pred, 1.0);
    }
    END_TEST

    TEST("Predict with zero embeddings")
    {
        MicroModel model;
        Vec10 e, c;
        e.fill(0.0);
        c.fill(0.0);
        double pred = model.predict(e, c);
        // With zero e, dot product is 0 regardless of v, so sigmoid(0) = 0.5
        ASSERT_NEAR(pred, 0.5, 0.01);
    }
    END_TEST

    TEST("Predict is deterministic")
    {
        MicroModel model;
        Vec10 e = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        Vec10 c = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
        double pred1 = model.predict(e, c);
        double pred2 = model.predict(e, c);
        ASSERT_NEAR(pred1, pred2, 1e-15);
    }
    END_TEST

    TEST("Single train_step reduces loss")
    {
        MicroModel model;
        Vec10 e = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
        Vec10 c = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
        MicroTrainingConfig config;
        config.learning_rate = 0.1;

        double loss1 = model.train_step(e, c, 0.9, config);
        model.train_step(e, c, 0.9, config);  // intermediate step
        double loss3 = model.train_step(e, c, 0.9, config);
        ASSERT_LT(loss3, loss1);
    }
    END_TEST

    TEST("Training converges on simple target")
    {
        MicroModel model;
        Vec10 e = {0.8, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0};
        Vec10 c = {0.1, 0.5, 0.0, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0, 0.4};

        std::vector<TrainingSample> samples;
        samples.push_back({e, c, 0.8});

        MicroTrainingConfig config;
        config.learning_rate = 0.05;
        config.max_epochs = 500;
        config.convergence_threshold = 1e-8;

        auto result = model.train(samples, config);
        ASSERT_LT(result.final_loss, 0.01);

        double pred = model.predict(e, c);
        ASSERT_NEAR(pred, 0.8, 0.1);
    }
    END_TEST

    TEST("Training with multiple samples")
    {
        MicroModel model;
        Vec10 e1 = {0.9, 0.0, 0.1, 0.3, 0.0, 0.1, 0.7, 0.8, 0.5, 0.7};
        Vec10 e2 = {0.0, 0.9, 0.0, 0.1, 0.7, 0.1, 0.6, 0.9, 0.4, 0.8};
        Vec10 c  = {0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1};

        std::vector<TrainingSample> samples;
        samples.push_back({e1, c, 0.7});
        samples.push_back({e2, c, 0.3});

        MicroTrainingConfig config;
        config.learning_rate = 0.02;
        config.max_epochs = 1000;
        config.convergence_threshold = 1e-8;

        auto result = model.train(samples, config);
        ASSERT_LT(result.final_loss, 0.05);
    }
    END_TEST

    TEST("Empty training set converges immediately")
    {
        MicroModel model;
        std::vector<TrainingSample> samples;
        MicroTrainingConfig config;
        auto result = model.train(samples, config);
        ASSERT(result.converged);
        ASSERT_EQ(result.epochs_run, 0u);
    }
    END_TEST

    TEST("Flat roundtrip preserves model state")
    {
        MicroModel original;
        Vec10 e = {0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4};
        Vec10 c = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

        // Train a bit to get non-trivial state
        MicroTrainingConfig config;
        config.learning_rate = 0.01;
        original.train_step(e, c, 0.7, config);
        original.train_step(e, c, 0.3, config);

        // Serialize
        std::array<double, FLAT_SIZE> flat;
        original.to_flat(flat);

        // Deserialize
        MicroModel restored;
        restored.from_flat(flat);

        // Compare predictions
        double pred_orig = original.predict(e, c);
        double pred_rest = restored.predict(e, c);
        ASSERT_NEAR(pred_orig, pred_rest, 1e-12);

        // Compare weights directly
        for (size_t i = 0; i < EMBED_DIM * EMBED_DIM; ++i) {
            ASSERT_NEAR(original.weights()[i], restored.weights()[i], 1e-12);
        }
        for (size_t i = 0; i < EMBED_DIM; ++i) {
            ASSERT_NEAR(original.bias()[i], restored.bias()[i], 1e-12);
        }
    }
    END_TEST

    TEST("Sigmoid helper is correct")
    {
        ASSERT_NEAR(sigmoid(0.0), 0.5, 1e-10);
        ASSERT_GT(sigmoid(10.0), 0.999);
        ASSERT_LT(sigmoid(-10.0), 0.001);
        ASSERT_NEAR(sigmoid(1.0), 1.0 / (1.0 + std::exp(-1.0)), 1e-10);
        // Numerically stable for very negative values
        ASSERT_GT(sigmoid(-100.0), 0.0);
        ASSERT_LT(sigmoid(-100.0), 1e-30);
    }
    END_TEST
}

// =============================================================================
// Registry tests
// =============================================================================

void test_registry() {
    std::cout << "\n=== MicroModelRegistry Tests ===" << std::endl;

    TEST("Create and retrieve model")
    {
        MicroModelRegistry reg;
        ASSERT(reg.create_model(1));
        ASSERT(reg.has_model(1));
        ASSERT(reg.get_model(1) != nullptr);
        ASSERT_EQ(reg.size(), 1u);
    }
    END_TEST

    TEST("Duplicate create returns false")
    {
        MicroModelRegistry reg;
        ASSERT(reg.create_model(42));
        ASSERT(!reg.create_model(42));
        ASSERT_EQ(reg.size(), 1u);
    }
    END_TEST

    TEST("Get non-existent model returns nullptr")
    {
        MicroModelRegistry reg;
        ASSERT(reg.get_model(999) == nullptr);
        ASSERT(!reg.has_model(999));
    }
    END_TEST

    TEST("Remove model")
    {
        MicroModelRegistry reg;
        reg.create_model(1);
        reg.create_model(2);
        ASSERT(reg.remove_model(1));
        ASSERT(!reg.has_model(1));
        ASSERT(reg.has_model(2));
        ASSERT_EQ(reg.size(), 1u);
    }
    END_TEST

    TEST("Remove non-existent returns false")
    {
        MicroModelRegistry reg;
        ASSERT(!reg.remove_model(999));
    }
    END_TEST

    TEST("Get model IDs")
    {
        MicroModelRegistry reg;
        reg.create_model(10);
        reg.create_model(20);
        reg.create_model(30);
        auto ids = reg.get_model_ids();
        ASSERT_EQ(ids.size(), 3u);
    }
    END_TEST

    TEST("Ensure models for LTM")
    {
        LongTermMemory ltm;
        auto c1 = ltm.store_concept("Dog", "A domesticated canine",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
        auto c2 = ltm.store_concept("Cat", "A domesticated feline",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));

        MicroModelRegistry reg;
        size_t created = reg.ensure_models_for(ltm);
        ASSERT_EQ(created, 2u);
        ASSERT(reg.has_model(c1));
        ASSERT(reg.has_model(c2));

        // Second call creates none
        size_t created2 = reg.ensure_models_for(ltm);
        ASSERT_EQ(created2, 0u);
    }
    END_TEST

    TEST("Clear removes all models")
    {
        MicroModelRegistry reg;
        reg.create_model(1);
        reg.create_model(2);
        reg.clear();
        ASSERT_EQ(reg.size(), 0u);
    }
    END_TEST
}

// =============================================================================
// Embedding manager tests
// =============================================================================

void test_embedding_manager() {
    std::cout << "\n=== EmbeddingManager Tests ===" << std::endl;

    TEST("Relation embeddings are initialized")
    {
        EmbeddingManager emb;
        const auto& is_a = emb.get_relation_embedding(RelationType::IS_A);
        // IS_A should have strong hierarchical component (dim 0)
        ASSERT_GT(is_a[0], 0.5);

        const auto& causes = emb.get_relation_embedding(RelationType::CAUSES);
        // CAUSES should have strong causal component (dim 1)
        ASSERT_GT(causes[1], 0.5);
    }
    END_TEST

    TEST("All relation types return valid embeddings")
    {
        EmbeddingManager emb;
        for (int i = 0; i < 10; ++i) {
            const auto& e = emb.get_relation_embedding(static_cast<RelationType>(i));
            // Should have non-zero norm
            double norm = 0.0;
            for (size_t j = 0; j < EMBED_DIM; ++j) {
                norm += e[j] * e[j];
            }
            ASSERT_GT(norm, 0.01);
        }
    }
    END_TEST

    TEST("Context embedding auto-creation")
    {
        EmbeddingManager emb;
        ASSERT(!emb.has_context("test_ctx"));
        const auto& ctx = emb.get_context_embedding("test_ctx");
        ASSERT(emb.has_context("test_ctx"));
        // Should be non-zero
        double norm = 0.0;
        for (size_t i = 0; i < EMBED_DIM; ++i) {
            norm += ctx[i] * ctx[i];
        }
        ASSERT_GT(norm, 0.0);
    }
    END_TEST

    TEST("Context embedding is deterministic")
    {
        EmbeddingManager emb1;
        EmbeddingManager emb2;
        const auto& c1 = emb1.get_context_embedding("myctx");
        const auto& c2 = emb2.get_context_embedding("myctx");
        for (size_t i = 0; i < EMBED_DIM; ++i) {
            ASSERT_NEAR(c1[i], c2[i], 1e-15);
        }
    }
    END_TEST

    TEST("Different contexts have different embeddings")
    {
        EmbeddingManager emb;
        const auto& c1 = emb.get_context_embedding("query");
        const auto& c2 = emb.get_context_embedding("recall");
        bool any_diff = false;
        for (size_t i = 0; i < EMBED_DIM; ++i) {
            if (std::abs(c1[i] - c2[i]) > 1e-10) {
                any_diff = true;
                break;
            }
        }
        ASSERT(any_diff);
    }
    END_TEST

    TEST("Convenience accessors work")
    {
        EmbeddingManager emb;
        const auto& q = emb.query_context();
        const auto& r = emb.recall_context();
        const auto& cr = emb.creative_context();
        const auto& a = emb.analytical_context();
        ASSERT(emb.has_context("query"));
        ASSERT(emb.has_context("recall"));
        ASSERT(emb.has_context("creative"));
        ASSERT(emb.has_context("analytical"));
        (void)q; (void)r; (void)cr; (void)a;
    }
    END_TEST

    TEST("Get context names")
    {
        EmbeddingManager emb;
        emb.get_context_embedding("alpha");
        emb.get_context_embedding("beta");
        auto names = emb.get_context_names();
        ASSERT_EQ(names.size(), 2u);
    }
    END_TEST
}

// =============================================================================
// Trainer tests
// =============================================================================

static LongTermMemory build_test_kg() {
    LongTermMemory ltm;
    auto dog = ltm.store_concept("Dog", "Domesticated canine",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto animal = ltm.store_concept("Animal", "Living organism",
        EpistemicMetadata(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.99));
    auto cat = ltm.store_concept("Cat", "Domesticated feline",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto pet = ltm.store_concept("Pet", "Domesticated animal companion",
        EpistemicMetadata(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.98));
    auto fur = ltm.store_concept("Fur", "Body hair of mammals",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.96));

    ltm.add_relation(dog, animal, RelationType::IS_A, 0.95);
    ltm.add_relation(cat, animal, RelationType::IS_A, 0.95);
    ltm.add_relation(dog, pet, RelationType::IS_A, 0.90);
    ltm.add_relation(cat, pet, RelationType::IS_A, 0.90);
    ltm.add_relation(dog, fur, RelationType::HAS_PROPERTY, 0.85);
    ltm.add_relation(cat, fur, RelationType::HAS_PROPERTY, 0.85);
    ltm.add_relation(dog, cat, RelationType::SIMILAR_TO, 0.70);

    return ltm;
}

void test_trainer() {
    std::cout << "\n=== MicroTrainer Tests ===" << std::endl;

    TEST("Generate samples for connected concept")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;
        MicroTrainer trainer;

        auto all_ids = ltm.get_all_concept_ids();
        ConceptId dog_id = all_ids[0];  // First stored concept

        auto samples = trainer.generate_samples(dog_id, emb, ltm);
        ASSERT_GT(samples.size(), 0u);

        // Should have positives (outgoing + incoming) and negatives
        // Dog has: 4 outgoing, some incoming -> multiple positives + 3x negatives
        size_t num_outgoing = ltm.get_outgoing_relations(dog_id).size();
        size_t num_incoming = ltm.get_incoming_relations(dog_id).size();
        size_t expected_positives = num_outgoing + num_incoming;
        ASSERT_GT(expected_positives, 0u);
        ASSERT_GT(samples.size(), expected_positives);  // Should have negatives too
    }
    END_TEST

    TEST("Generate samples for isolated concept returns empty")
    {
        LongTermMemory ltm;
        ltm.store_concept("Lonely", "No connections",
            EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.30));
        EmbeddingManager emb;
        MicroTrainer trainer;

        auto all_ids = ltm.get_all_concept_ids();
        auto samples = trainer.generate_samples(all_ids[0], emb, ltm);
        ASSERT_EQ(samples.size(), 0u);
    }
    END_TEST

    TEST("Train single model from KG")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;
        MicroTrainer trainer;
        MicroModel model;

        auto all_ids = ltm.get_all_concept_ids();
        ConceptId dog_id = all_ids[0];

        auto result = trainer.train_single(dog_id, model, emb, ltm);
        ASSERT_GT(result.epochs_run, 0u);
        ASSERT_LT(result.final_loss, 0.5);  // Should make some progress
    }
    END_TEST

    TEST("Train all models")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;
        MicroModelRegistry reg;
        reg.ensure_models_for(ltm);

        TrainerConfig tc;
        tc.model_config.max_epochs = 200;
        tc.model_config.learning_rate = 0.02;
        MicroTrainer trainer(tc);

        auto stats = trainer.train_all(reg, emb, ltm);
        ASSERT_GT(stats.models_trained, 0u);
        ASSERT_GT(stats.total_samples, 0u);
        ASSERT_GT(stats.total_epochs, 0u);
    }
    END_TEST

    TEST("Negative samples have low target")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;

        TrainerConfig tc;
        tc.neg_target = 0.02;
        MicroTrainer trainer(tc);

        auto all_ids = ltm.get_all_concept_ids();
        auto samples = trainer.generate_samples(all_ids[0], emb, ltm);

        // Check that some samples have low targets (negatives)
        bool has_low = false;
        for (const auto& s : samples) {
            if (s.target < 0.1) {
                has_low = true;
                break;
            }
        }
        ASSERT(has_low);
    }
    END_TEST
}

// =============================================================================
// RelevanceMap tests
// =============================================================================

void test_relevance_map() {
    std::cout << "\n=== RelevanceMap Tests ===" << std::endl;

    TEST("Compute relevance map")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;
        MicroModelRegistry reg;
        reg.ensure_models_for(ltm);

        auto all_ids = ltm.get_all_concept_ids();
        ConceptId dog_id = all_ids[0];

        auto rmap = RelevanceMap::compute(dog_id, reg, emb, ltm,
                                          RelationType::IS_A, "recall");
        ASSERT(!rmap.empty());
        ASSERT_EQ(rmap.size(), all_ids.size() - 1);  // Excludes self
        ASSERT_EQ(rmap.source(), dog_id);
    }
    END_TEST

    TEST("Scores are in (0,1) range")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;
        MicroModelRegistry reg;
        reg.ensure_models_for(ltm);

        auto all_ids = ltm.get_all_concept_ids();
        auto rmap = RelevanceMap::compute(all_ids[0], reg, emb, ltm,
                                          RelationType::IS_A, "recall");

        for (const auto& [cid, score] : rmap.scores()) {
            ASSERT_GT(score, 0.0);
            ASSERT_LT(score, 1.0);
        }
    }
    END_TEST

    TEST("Top-k returns correct count")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;
        MicroModelRegistry reg;
        reg.ensure_models_for(ltm);

        auto all_ids = ltm.get_all_concept_ids();
        auto rmap = RelevanceMap::compute(all_ids[0], reg, emb, ltm,
                                          RelationType::IS_A, "recall");

        auto top2 = rmap.top_k(2);
        ASSERT_EQ(top2.size(), 2u);
        // Should be sorted descending
        ASSERT(top2[0].second >= top2[1].second);
    }
    END_TEST

    TEST("Above threshold filters correctly")
    {
        auto ltm = build_test_kg();
        EmbeddingManager emb;
        MicroModelRegistry reg;
        reg.ensure_models_for(ltm);

        auto all_ids = ltm.get_all_concept_ids();
        auto rmap = RelevanceMap::compute(all_ids[0], reg, emb, ltm,
                                          RelationType::IS_A, "recall");

        auto above = rmap.above_threshold(0.5);
        for (const auto& [cid, score] : above) {
            ASSERT(score >= 0.5);
        }
    }
    END_TEST

    TEST("Score of absent concept is 0")
    {
        RelevanceMap rmap(1);
        ASSERT_NEAR(rmap.score(999), 0.0, 1e-15);
    }
    END_TEST

    TEST("Overlay ADDITION")
    {
        RelevanceMap m1(1);
        RelevanceMap m2(2);
        // Manually set scores via compute-like approach
        // We'll test the overlay logic directly
        auto& s1 = const_cast<std::unordered_map<ConceptId, double>&>(m1.scores());
        auto& s2 = const_cast<std::unordered_map<ConceptId, double>&>(m2.scores());
        s1[10] = 0.3;
        s1[20] = 0.5;
        s2[10] = 0.4;
        s2[30] = 0.6;

        m1.overlay(m2, OverlayMode::ADDITION);
        ASSERT_NEAR(m1.score(10), 0.7, 1e-10);
        ASSERT_NEAR(m1.score(20), 0.5, 1e-10);
        ASSERT_NEAR(m1.score(30), 0.6, 1e-10);
    }
    END_TEST

    TEST("Overlay MAX")
    {
        RelevanceMap m1(1);
        RelevanceMap m2(2);
        auto& s1 = const_cast<std::unordered_map<ConceptId, double>&>(m1.scores());
        auto& s2 = const_cast<std::unordered_map<ConceptId, double>&>(m2.scores());
        s1[10] = 0.3;
        s2[10] = 0.8;

        m1.overlay(m2, OverlayMode::MAX);
        ASSERT_NEAR(m1.score(10), 0.8, 1e-10);
    }
    END_TEST

    TEST("Overlay WEIGHTED_AVERAGE")
    {
        RelevanceMap m1(1);
        RelevanceMap m2(2);
        auto& s1 = const_cast<std::unordered_map<ConceptId, double>&>(m1.scores());
        auto& s2 = const_cast<std::unordered_map<ConceptId, double>&>(m2.scores());
        s1[10] = 0.4;
        s2[10] = 0.8;

        m1.overlay(m2, OverlayMode::WEIGHTED_AVERAGE, 0.5);
        // 0.4 * 0.5 + 0.8 * 0.5 = 0.6
        ASSERT_NEAR(m1.score(10), 0.6, 1e-10);
    }
    END_TEST

    TEST("Normalize scales to [0,1]")
    {
        RelevanceMap rmap(1);
        auto& scores = const_cast<std::unordered_map<ConceptId, double>&>(rmap.scores());
        scores[10] = 0.2;
        scores[20] = 0.8;
        scores[30] = 0.5;

        rmap.normalize();
        ASSERT_NEAR(rmap.score(10), 0.0, 1e-10);   // min -> 0
        ASSERT_NEAR(rmap.score(20), 1.0, 1e-10);   // max -> 1
        ASSERT_NEAR(rmap.score(30), 0.5, 1e-10);   // mid -> 0.5
    }
    END_TEST

    TEST("Combine multiple maps")
    {
        RelevanceMap m1(1);
        RelevanceMap m2(2);
        auto& s1 = const_cast<std::unordered_map<ConceptId, double>&>(m1.scores());
        auto& s2 = const_cast<std::unordered_map<ConceptId, double>&>(m2.scores());
        s1[10] = 0.3;
        s2[10] = 0.7;

        auto combined = RelevanceMap::combine({m1, m2}, OverlayMode::ADDITION);
        ASSERT_NEAR(combined.score(10), 1.0, 1e-10);
    }
    END_TEST
}

// =============================================================================
// Persistence tests
// =============================================================================

void test_persistence() {
    std::cout << "\n=== Persistence Tests ===" << std::endl;

    const std::string test_file = "/tmp/brain19_test_micromodel.bin";

    TEST("Save and load roundtrip")
    {
        MicroModelRegistry reg;
        EmbeddingManager emb;

        reg.create_model(1);
        reg.create_model(2);
        reg.create_model(3);

        // Train models a bit
        MicroModel* m1 = reg.get_model(1);
        Vec10 e = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
        Vec10 c = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
        MicroTrainingConfig config;
        m1->train_step(e, c, 0.8, config);

        // Create some contexts
        emb.get_context_embedding("test_ctx_1");
        emb.get_context_embedding("test_ctx_2");

        // Save
        ASSERT(persistence::save(test_file, reg, emb));

        // Load into fresh instances
        MicroModelRegistry reg2;
        EmbeddingManager emb2;
        ASSERT(persistence::load(test_file, reg2, emb2));

        // Verify models
        ASSERT_EQ(reg2.size(), 3u);
        ASSERT(reg2.has_model(1));
        ASSERT(reg2.has_model(2));
        ASSERT(reg2.has_model(3));

        // Verify trained model predictions match
        MicroModel* m1_loaded = reg2.get_model(1);
        ASSERT(m1_loaded != nullptr);
        double pred_orig = m1->predict(e, c);
        double pred_loaded = m1_loaded->predict(e, c);
        ASSERT_NEAR(pred_orig, pred_loaded, 1e-12);

        // Verify contexts
        ASSERT(emb2.has_context("test_ctx_1"));
        ASSERT(emb2.has_context("test_ctx_2"));

        // Verify relation embeddings match
        for (int i = 0; i < 10; ++i) {
            auto rt = static_cast<RelationType>(i);
            const auto& orig = emb.get_relation_embedding(rt);
            const auto& loaded = emb2.get_relation_embedding(rt);
            for (size_t j = 0; j < EMBED_DIM; ++j) {
                ASSERT_NEAR(orig[j], loaded[j], 1e-12);
            }
        }

        std::remove(test_file.c_str());
    }
    END_TEST

    TEST("Validate correct file")
    {
        MicroModelRegistry reg;
        EmbeddingManager emb;
        reg.create_model(1);

        ASSERT(persistence::save(test_file, reg, emb));
        ASSERT(persistence::validate(test_file));
        std::remove(test_file.c_str());
    }
    END_TEST

    TEST("Validate rejects corrupt file")
    {
        MicroModelRegistry reg;
        EmbeddingManager emb;
        reg.create_model(1);

        ASSERT(persistence::save(test_file, reg, emb));

        // Corrupt a byte
        std::fstream f(test_file, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(40);
        char corrupt_byte = static_cast<char>(0xFF);
        f.write(&corrupt_byte, 1);
        f.close();

        ASSERT(!persistence::validate(test_file));
        std::remove(test_file.c_str());
    }
    END_TEST

    TEST("Load rejects non-existent file")
    {
        MicroModelRegistry reg;
        EmbeddingManager emb;
        ASSERT(!persistence::load("/tmp/nonexistent_brain19.bin", reg, emb));
    }
    END_TEST

    TEST("Validate rejects non-existent file")
    {
        ASSERT(!persistence::validate("/tmp/nonexistent_brain19.bin"));
    }
    END_TEST

    TEST("Empty registry roundtrip")
    {
        MicroModelRegistry reg;
        EmbeddingManager emb;

        ASSERT(persistence::save(test_file, reg, emb));

        MicroModelRegistry reg2;
        EmbeddingManager emb2;
        ASSERT(persistence::load(test_file, reg2, emb2));
        ASSERT_EQ(reg2.size(), 0u);

        std::remove(test_file.c_str());
    }
    END_TEST
}

// =============================================================================
// Integration test
// =============================================================================

void test_integration() {
    std::cout << "\n=== Integration Tests ===" << std::endl;

    TEST("Full pipeline: LTM -> models -> train -> relevance")
    {
        // Build KG
        auto ltm = build_test_kg();

        // Create models
        MicroModelRegistry reg;
        size_t created = reg.ensure_models_for(ltm);
        ASSERT_EQ(created, 5u);

        // Train models
        EmbeddingManager emb;
        TrainerConfig tc;
        tc.model_config.max_epochs = 300;
        tc.model_config.learning_rate = 0.02;
        MicroTrainer trainer(tc);

        auto stats = trainer.train_all(reg, emb, ltm);
        ASSERT_GT(stats.models_trained, 0u);

        // Compute relevance maps
        auto all_ids = ltm.get_all_concept_ids();
        ConceptId dog_id = all_ids[0];

        auto rmap = RelevanceMap::compute(dog_id, reg, emb, ltm,
                                          RelationType::IS_A, "recall");
        ASSERT(!rmap.empty());

        // Get top results
        auto top3 = rmap.top_k(3);
        ASSERT_GT(top3.size(), 0u);

        // All scores should be valid
        for (const auto& [cid, score] : top3) {
            ASSERT_GT(score, 0.0);
            ASSERT_LT(score, 1.0);
        }
    }
    END_TEST

    TEST("Full pipeline with persistence")
    {
        const std::string test_file = "/tmp/brain19_integration_test.bin";

        auto ltm = build_test_kg();

        MicroModelRegistry reg;
        reg.ensure_models_for(ltm);

        EmbeddingManager emb;
        TrainerConfig tc;
        tc.model_config.max_epochs = 200;
        tc.model_config.learning_rate = 0.02;
        MicroTrainer trainer(tc);
        trainer.train_all(reg, emb, ltm);

        // Save
        ASSERT(persistence::save(test_file, reg, emb));

        // Load
        MicroModelRegistry reg2;
        EmbeddingManager emb2;
        ASSERT(persistence::load(test_file, reg2, emb2));

        // Verify same relevance scores
        auto all_ids = ltm.get_all_concept_ids();
        ConceptId dog_id = all_ids[0];

        auto rmap1 = RelevanceMap::compute(dog_id, reg, emb, ltm,
                                           RelationType::IS_A, "recall");
        auto rmap2 = RelevanceMap::compute(dog_id, reg2, emb2, ltm,
                                           RelationType::IS_A, "recall");

        for (const auto& [cid, score] : rmap1.scores()) {
            ASSERT_NEAR(score, rmap2.score(cid), 1e-10);
        }

        std::remove(test_file.c_str());
    }
    END_TEST

    TEST("Overlay two concept perspectives")
    {
        auto ltm = build_test_kg();
        MicroModelRegistry reg;
        reg.ensure_models_for(ltm);
        EmbeddingManager emb;

        auto all_ids = ltm.get_all_concept_ids();
        // Compute two relevance maps from different concepts
        auto rmap1 = RelevanceMap::compute(all_ids[0], reg, emb, ltm,
                                           RelationType::IS_A, "recall");
        auto rmap2 = RelevanceMap::compute(all_ids[1], reg, emb, ltm,
                                           RelationType::IS_A, "recall");

        // Overlay them
        rmap1.overlay(rmap2, OverlayMode::ADDITION);
        rmap1.normalize();

        // Should still have valid scores
        for (const auto& [cid, score] : rmap1.scores()) {
            ASSERT(score >= 0.0);
            ASSERT(score <= 1.0);
        }
    }
    END_TEST
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Brain19 Bilinear Micro-Model Test Suite" << std::endl;
    std::cout << "Phase 2: Per-Concept Learned Models" << std::endl;
    std::cout << "========================================" << std::endl;

    test_micro_model();
    test_registry();
    test_embedding_manager();
    test_trainer();
    test_relevance_map();
    test_persistence();
    test_integration();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed == 0 ? 0 : 1;
}
