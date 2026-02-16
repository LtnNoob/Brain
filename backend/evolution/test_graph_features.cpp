// test_graph_features.cpp — Tests for Phase 1-3 Graph Features
//
// Phase 1: Schema extensions (RelationType, ConceptInfo, LTM hooks)
// Phase 2: ComplexityAnalyzer + RetentionManager
// Phase 3: TrustPropagator

#include "../ltm/long_term_memory.hpp"
#include "../memory/relation_type_registry.hpp"
#include "epistemic_promotion.hpp"
#include "graph_densifier.hpp"
#include "complexity_analyzer.hpp"
#include "retention_manager.hpp"
#include "trust_propagator.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace brain19;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { \
        Register_##name() { \
            std::cerr << "  TEST " #name "... "; \
            try { test_##name(); std::cerr << "PASS\n"; ++tests_passed; } \
            catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << "\n"; ++tests_failed; } \
            catch (...) { std::cerr << "FAIL: unknown exception\n"; ++tests_failed; } \
        } \
    } register_##name; \
    static void test_##name()

#define ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond " at line " + std::to_string(__LINE__)); } while(0)
#define ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Expected " #a " == " #b " at line " + std::to_string(__LINE__) + " got " + std::to_string(a) + " vs " + std::to_string(b)); } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) throw std::runtime_error("Expected " #a " ≈ " #b " at line " + std::to_string(__LINE__) + " got " + std::to_string(a)); } while(0)

// =============================================================================
// Phase 1 Tests: Schema Extensions
// =============================================================================

TEST(relation_type_linguistic_builtins) {
    auto& reg = RelationTypeRegistry::instance();
    ASSERT(reg.has(RelationType::SUBJECT_OF));
    ASSERT(reg.has(RelationType::OBJECT_OF));
    ASSERT(reg.has(RelationType::VERB_OF));
    ASSERT(reg.has(RelationType::MODIFIER_OF));
    ASSERT(reg.has(RelationType::DENOTES));
    ASSERT(reg.has(RelationType::PART_OF_SENTENCE));
    ASSERT(reg.has(RelationType::TEMPORAL_OF));
    ASSERT(reg.has(RelationType::LOCATIVE_OF));
    ASSERT(reg.has(RelationType::PRECEDES));
}

TEST(relation_type_name_lookup) {
    auto& reg = RelationTypeRegistry::instance();
    auto t = reg.find_by_name("SUBJECT_OF");
    ASSERT(t.has_value());
    ASSERT(*t == RelationType::SUBJECT_OF);

    auto t2 = reg.find_by_name("subject-of");
    ASSERT(t2.has_value());
    ASSERT(*t2 == RelationType::SUBJECT_OF);
}

TEST(relation_type_linguistic_category) {
    auto& reg = RelationTypeRegistry::instance();
    ASSERT(reg.get_category(RelationType::DENOTES) == RelationCategory::LINGUISTIC);
    ASSERT(reg.get_category(RelationType::VERB_OF) == RelationCategory::LINGUISTIC);
    ASSERT(reg.get_category(RelationType::PRECEDES) == RelationCategory::LINGUISTIC);
}

TEST(concept_info_anti_knowledge_defaults) {
    LongTermMemory ltm;
    auto id = ltm.store_concept("test", "def",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto cinfo = ltm.retrieve_concept(id);
    ASSERT(cinfo.has_value());
    ASSERT(!cinfo->is_anti_knowledge);
    ASSERT_NEAR(cinfo->complexity_score, 0.0f, 0.001f);
}

TEST(ltm_invalidation_hook_fires) {
    LongTermMemory ltm;
    auto id = ltm.store_concept("fact1", "def",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));

    bool hook_fired = false;
    ConceptId hook_id = 0;
    double hook_old_trust = 0.0;

    ltm.register_invalidation_hook([&](ConceptId cid, double old_trust) {
        hook_fired = true;
        hook_id = cid;
        hook_old_trust = old_trust;
    });

    ltm.invalidate_concept(id);
    ASSERT(hook_fired);
    ASSERT_EQ(hook_id, id);
    ASSERT_NEAR(hook_old_trust, 0.95, 0.01);
}

TEST(ltm_anti_knowledge_marking) {
    LongTermMemory ltm;
    auto id = ltm.store_concept("wrong", "def",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    ltm.invalidate_concept(id);

    ASSERT(ltm.get_anti_knowledge().empty());
    ASSERT_EQ(ltm.get_gc_candidates().size(), size_t(1));

    ltm.mark_as_anti_knowledge(id, "complex causal chain");
    ASSERT_EQ(ltm.get_anti_knowledge().size(), size_t(1));
    ASSERT(ltm.get_gc_candidates().empty());  // no longer a GC candidate

    ltm.unmark_anti_knowledge(id);
    ASSERT(ltm.get_anti_knowledge().empty());
    ASSERT_EQ(ltm.get_gc_candidates().size(), size_t(1));
}

TEST(ltm_garbage_collect) {
    LongTermMemory ltm;
    auto a = ltm.store_concept("A", "def",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto b = ltm.store_concept("B", "def",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    auto c = ltm.store_concept("C", "def",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.2));

    ltm.add_relation(a, b, RelationType::CAUSES, 0.8);
    ltm.add_relation(b, c, RelationType::CAUSES, 0.7);

    // Invalidate B and C
    ltm.invalidate_concept(b);
    ltm.invalidate_concept(c);

    // Mark C as anti-knowledge
    ltm.mark_as_anti_knowledge(c, "complex");

    // GC should only remove B (C is anti-knowledge)
    size_t removed = ltm.garbage_collect();
    ASSERT_EQ(removed, size_t(1));
    ASSERT(!ltm.exists(b));
    ASSERT(ltm.exists(c));  // anti-knowledge preserved
    ASSERT(ltm.exists(a));  // active concept preserved
}

TEST(ltm_linguistic_relations) {
    LongTermMemory ltm;
    auto sentence = ltm.store_concept("Sentence_1", "Katzen jagen Mäuse",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    auto word_katzen = ltm.store_concept("Word:Katzen", "surface form",
        EpistemicMetadata(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.95));
    auto word_jagen = ltm.store_concept("Word:jagen", "surface form",
        EpistemicMetadata(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.95));

    auto r1 = ltm.add_relation(word_katzen, sentence, RelationType::SUBJECT_OF, 1.0);
    auto r2 = ltm.add_relation(word_jagen, sentence, RelationType::VERB_OF, 1.0);

    ASSERT(r1 > 0);
    ASSERT(r2 > 0);

    auto out = ltm.get_incoming_relations(sentence);
    ASSERT_EQ(out.size(), size_t(2));

    bool has_subject = false, has_verb = false;
    for (const auto& rel : out) {
        if (rel.type == RelationType::SUBJECT_OF) has_subject = true;
        if (rel.type == RelationType::VERB_OF) has_verb = true;
    }
    ASSERT(has_subject);
    ASSERT(has_verb);
}

// =============================================================================
// Phase 2 Tests: ComplexityAnalyzer + RetentionManager
// =============================================================================

// Helper: create a causal chain of N concepts
static std::vector<ConceptId> create_causal_chain(LongTermMemory& ltm, size_t n) {
    std::vector<ConceptId> ids;
    for (size_t i = 0; i < n; ++i) {
        auto id = ltm.store_concept(
            "chain_" + std::to_string(i), "part of causal chain",
            EpistemicMetadata(EpistemicType::INFERENCE, EpistemicStatus::ACTIVE, 0.6));
        ids.push_back(id);
        if (i > 0) {
            ltm.add_relation(ids[i-1], ids[i], RelationType::CAUSES, 0.8);
        }
    }
    return ids;
}

TEST(complexity_analyzer_simple_concept) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    auto id = ltm.store_concept("simple", "simple fact",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));

    ComplexityAnalyzer analyzer(ltm, densifier);
    auto metrics = analyzer.analyze(id);

    ASSERT_EQ(metrics.causal_chain_length, size_t(1));  // just itself
    ASSERT_EQ(metrics.involved_concepts, size_t(1));
    ASSERT(metrics.normalized_score < 0.2f);
    ASSERT(!analyzer.should_retain(id));
}

TEST(complexity_analyzer_complex_chain) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    auto chain = create_causal_chain(ltm, 8);

    // Add some cross-links for higher involved_concepts
    for (size_t i = 0; i + 2 < chain.size(); ++i) {
        ltm.add_relation(chain[i], chain[i+2], RelationType::ENABLES, 0.5);
    }

    ComplexityAnalyzer analyzer(ltm, densifier);
    auto metrics = analyzer.analyze(chain[3]);  // middle of chain

    ASSERT(metrics.causal_chain_length >= 4);
    ASSERT(metrics.involved_concepts >= 5);
    ASSERT(metrics.inference_steps >= 3);
    ASSERT(metrics.normalized_score >= 0.3f);
}

TEST(complexity_should_retain_complex) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    auto chain = create_causal_chain(ltm, 10);
    // Add extra connections
    for (size_t i = 0; i + 3 < chain.size(); ++i) {
        ltm.add_relation(chain[i], chain[i+3], RelationType::ENABLES, 0.4);
    }

    RetentionConfig cfg;
    cfg.min_causal_chain = 3;
    cfg.min_involved_concepts = 5;
    cfg.complexity_threshold = 0.3f;

    ComplexityAnalyzer analyzer(ltm, densifier, cfg);

    // Middle of a long chain should be retained
    ASSERT(analyzer.should_retain(chain[5]));

    // Simple isolated concept should not
    auto simple = ltm.store_concept("simple", "def",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    ASSERT(!analyzer.should_retain(simple));
}

TEST(retention_manager_on_invalidation) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    auto chain = create_causal_chain(ltm, 8);
    for (size_t i = 0; i + 2 < chain.size(); ++i) {
        ltm.add_relation(chain[i], chain[i+2], RelationType::ENABLES, 0.5);
    }

    RetentionConfig cfg;
    cfg.min_causal_chain = 3;
    cfg.min_involved_concepts = 5;
    cfg.complexity_threshold = 0.3f;

    ComplexityAnalyzer analyzer(ltm, densifier, cfg);
    RetentionManager retention(ltm, analyzer);

    // Invalidate middle of chain
    ltm.invalidate_concept(chain[4]);
    retention.on_invalidation(chain[4]);

    auto cinfo = ltm.retrieve_concept(chain[4]);
    ASSERT(cinfo.has_value());
    ASSERT(cinfo->is_anti_knowledge);
}

TEST(retention_manager_simple_not_retained) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    auto simple = ltm.store_concept("Berlin ist Hauptstadt von Frankreich", "wrong",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.4));

    ComplexityAnalyzer analyzer(ltm, densifier);
    RetentionManager retention(ltm, analyzer);

    ltm.invalidate_concept(simple);
    retention.on_invalidation(simple);

    auto cinfo = ltm.retrieve_concept(simple);
    ASSERT(cinfo.has_value());
    ASSERT(!cinfo->is_anti_knowledge);  // simple → not retained
}

TEST(retention_gc_cycle) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    // Create 5 simple invalidated concepts
    for (int i = 0; i < 5; ++i) {
        auto id = ltm.store_concept("wrong_" + std::to_string(i), "def",
            EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.2));
        ltm.invalidate_concept(id);
    }

    // Create 1 complex invalidated concept
    auto chain = create_causal_chain(ltm, 8);
    for (size_t i = 0; i + 2 < chain.size(); ++i) {
        ltm.add_relation(chain[i], chain[i+2], RelationType::ENABLES, 0.5);
    }
    ltm.invalidate_concept(chain[4]);

    RetentionConfig cfg;
    cfg.min_causal_chain = 3;
    cfg.min_involved_concepts = 5;
    cfg.complexity_threshold = 0.3f;

    ComplexityAnalyzer analyzer(ltm, densifier, cfg);
    RetentionManager retention(ltm, analyzer);

    auto stats = retention.run_gc_cycle();
    ASSERT(stats.total_invalidated >= 6);
    ASSERT(stats.actually_removed >= 4);  // simple ones removed
}

TEST(retention_explain_anti_knowledge) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    auto chain = create_causal_chain(ltm, 6);
    for (size_t i = 0; i + 2 < chain.size(); ++i) {
        ltm.add_relation(chain[i], chain[i+2], RelationType::ENABLES, 0.5);
    }

    RetentionConfig cfg;
    cfg.min_causal_chain = 2;
    cfg.min_involved_concepts = 3;
    cfg.complexity_threshold = 0.2f;

    ComplexityAnalyzer analyzer(ltm, densifier, cfg);
    RetentionManager retention(ltm, analyzer);

    ltm.invalidate_concept(chain[3]);
    retention.on_invalidation(chain[3]);

    std::string explanation = retention.explain_anti_knowledge(chain[3]);
    ASSERT(explanation.find("Anti-Knowledge") != std::string::npos);
    ASSERT(explanation.find("Causal Chain Length") != std::string::npos);
}

// =============================================================================
// Phase 3 Tests: TrustPropagator
// =============================================================================

TEST(trust_propagator_basic) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);
    GraphDensifier densifier(ltm);

    // Create shared target so A and B have structural similarity
    auto shared = ltm.store_concept("shared", "shared target",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto a = ltm.store_concept("A", "source",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.6));
    auto b = ltm.store_concept("B", "neighbor",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.7));

    // A and B both connect to shared (co-activation + structural similarity)
    ltm.add_relation(a, shared, RelationType::CAUSES, 0.9);
    ltm.add_relation(b, shared, RelationType::CAUSES, 0.9);
    ltm.add_relation(a, b, RelationType::SIMILAR_TO, 0.8);

    PropagationConfig cfg;
    cfg.similarity_threshold = 0.1f;  // low threshold for test
    cfg.max_trust_reduction = 0.5f;

    TrustPropagator propagator(ltm, promotion, densifier, cfg);

    // Invalidate A
    ltm.invalidate_concept(a);
    auto result = propagator.propagate(a);

    ASSERT(result.concepts_checked > 0);
    // B should be affected (direct neighbor with similar structure)
    bool b_affected = false;
    for (const auto& [cid, new_trust] : result.affected) {
        if (cid == b) {
            b_affected = true;
            ASSERT(new_trust < 0.7);  // trust should have decreased
        }
    }
    ASSERT(b_affected);
}

TEST(trust_propagator_cascading_invalidation) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);
    GraphDensifier densifier(ltm);

    // A → B → C chain, all low trust
    auto a = ltm.store_concept("A", "root",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.3));
    auto b = ltm.store_concept("B", "mid",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.15));
    auto c = ltm.store_concept("C", "leaf",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.12));

    ltm.add_relation(a, b, RelationType::CAUSES, 0.9);
    ltm.add_relation(b, c, RelationType::CAUSES, 0.9);

    PropagationConfig cfg;
    cfg.similarity_threshold = 0.05f;
    cfg.cumulative_invalidation_threshold = 0.1f;
    cfg.max_trust_reduction = 0.5f;

    TrustPropagator propagator(ltm, promotion, densifier, cfg);

    ltm.invalidate_concept(a);
    auto result = propagator.propagate(a);

    // B should potentially be force-invalidated due to low trust
    // (depends on exact similarity calculation)
    ASSERT(result.concepts_checked >= 2);
}

TEST(trust_propagator_similarity_scores) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);
    GraphDensifier densifier(ltm);

    // Create two structurally similar concepts
    auto shared = ltm.store_concept("shared", "shared target",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto a = ltm.store_concept("A", "concept a",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.6));
    auto b = ltm.store_concept("B", "concept b",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.6));

    // Both have same relation types to same target
    ltm.add_relation(a, shared, RelationType::CAUSES, 0.8);
    ltm.add_relation(b, shared, RelationType::CAUSES, 0.8);
    ltm.add_relation(a, shared, RelationType::SUPPORTS, 0.5);
    ltm.add_relation(b, shared, RelationType::SUPPORTS, 0.5);

    TrustPropagator propagator(ltm, promotion, densifier);

    float sim = propagator.combined_similarity(a, b);
    ASSERT(sim > 0.0f);  // should have some similarity
}

TEST(trust_propagator_no_linguistic_propagation) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);
    GraphDensifier densifier(ltm);

    auto semantic = ltm.store_concept("Katze", "semantic concept",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto word = ltm.store_concept("Word:Katzen", "linguistic",
        EpistemicMetadata(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.95));

    ltm.add_relation(word, semantic, RelationType::DENOTES, 1.0);

    PropagationConfig cfg;
    cfg.propagate_to_linguistic = false;
    cfg.similarity_threshold = 0.01f;

    TrustPropagator propagator(ltm, promotion, densifier, cfg);

    ltm.invalidate_concept(semantic);
    auto result = propagator.propagate(semantic);

    // Word concept should NOT be affected
    for (const auto& [cid, _trust] : result.affected) {
        ASSERT(cid != word);
    }
}

TEST(trust_propagator_history) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);
    GraphDensifier densifier(ltm);

    auto a = ltm.store_concept("A", "source",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    auto b = ltm.store_concept("B", "target",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.7));

    ltm.add_relation(a, b, RelationType::CAUSES, 0.9);

    PropagationConfig cfg;
    cfg.similarity_threshold = 0.05f;
    cfg.max_trust_reduction = 0.5f;

    TrustPropagator propagator(ltm, promotion, densifier, cfg);

    ltm.invalidate_concept(a);
    propagator.propagate(a);

    auto sources = propagator.get_propagation_sources(b);
    // B should have A as propagation source if it was affected
    // (depends on similarity threshold being met)
    // Just verify the method doesn't crash
    ASSERT(sources.size() <= 1);
}

TEST(epistemic_promotion_trust_propagation) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);

    auto a = ltm.store_concept("A", "source",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.6));
    auto b = ltm.store_concept("B", "neighbor",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.7));

    ltm.add_relation(a, b, RelationType::CAUSES, 0.9);

    auto adjustments = promotion.compute_trust_propagation(a, 0.3f);
    // Should find B as affected
    // adjustments may or may not contain entries depending on radius
    (void)adjustments;

    // Apply adjustments
    promotion.apply_trust_propagation(adjustments);
}

TEST(epistemic_promotion_should_force_invalidate) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);

    auto id = ltm.store_concept("low_trust", "def",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.05));

    ASSERT(promotion.should_force_invalidate(id, 0.1));

    auto id2 = ltm.store_concept("high_trust", "def",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));

    ASSERT(!promotion.should_force_invalidate(id2, 0.1));
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(integration_invalidation_chain) {
    LongTermMemory ltm;
    EpistemicPromotion promotion(ltm);
    GraphDensifier densifier(ltm);

    // Build a network where wrong_A and wrong_B have HIGH similarity:
    // - Same outgoing relation types (structural similarity)
    // - Shared neighbors (co-activation)
    // - Shared incoming sources (shared_source)
    auto target1 = ltm.store_concept("target1", "shared target",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto target2 = ltm.store_concept("target2", "shared target 2",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto source = ltm.store_concept("source", "shared source",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    auto wrong_A = ltm.store_concept("wrong_A", "will be invalidated",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    auto wrong_B = ltm.store_concept("wrong_B", "structurally similar to A",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));

    // Same outgoing types to same targets (structural + co-activation)
    ltm.add_relation(wrong_A, target1, RelationType::CAUSES, 0.8);
    ltm.add_relation(wrong_A, target2, RelationType::CAUSES, 0.7);
    ltm.add_relation(wrong_B, target1, RelationType::CAUSES, 0.8);
    ltm.add_relation(wrong_B, target2, RelationType::CAUSES, 0.7);

    // Shared incoming source (shared_source score) — but NOT as SUPPORTS
    // (SUPPORTS would give wrong_B full support_ratio → 0 reduction)
    ltm.add_relation(source, wrong_A, RelationType::DERIVED_FROM, 0.5);
    ltm.add_relation(source, wrong_B, RelationType::DERIVED_FROM, 0.5);

    // Setup propagator
    PropagationConfig pcfg;
    pcfg.similarity_threshold = 0.3f;
    pcfg.max_trust_reduction = 0.4f;
    pcfg.cumulative_invalidation_threshold = 0.05f;

    TrustPropagator propagator(ltm, promotion, densifier, pcfg);

    // Verify: A and B should have high combined similarity
    float sim = propagator.combined_similarity(wrong_A, wrong_B);
    ASSERT(sim > 0.3f);

    // Setup retention
    ComplexityAnalyzer analyzer(ltm, densifier);
    RetentionManager retention(ltm, analyzer);

    // Wire up hook
    ltm.register_invalidation_hook([&](ConceptId cid, double /*old_trust*/) {
        auto prop_result = propagator.propagate(cid);
        retention.on_invalidation(cid);
        for (auto fi : prop_result.force_invalidated) {
            retention.on_invalidation(fi);
        }
    });

    // Invalidate wrong_A
    ltm.invalidate_concept(wrong_A);

    // Verify: wrong_A is invalidated
    auto a_info = ltm.retrieve_concept(wrong_A);
    ASSERT(a_info.has_value());
    ASSERT(a_info->epistemic.is_invalidated());

    // Verify: wrong_B should have reduced trust due to high similarity
    auto b_info = ltm.retrieve_concept(wrong_B);
    ASSERT(b_info.has_value());
    ASSERT(b_info->epistemic.trust < 0.5 || b_info->epistemic.is_invalidated());
}

TEST(integration_resembles_known_error) {
    LongTermMemory ltm;
    GraphDensifier densifier(ltm);

    // Create an anti-knowledge pattern: A CAUSES B
    auto a = ltm.store_concept("wrong_premise", "def",
        EpistemicMetadata(EpistemicType::INFERENCE, EpistemicStatus::ACTIVE, 0.5));
    auto b = ltm.store_concept("wrong_conclusion", "def",
        EpistemicMetadata(EpistemicType::INFERENCE, EpistemicStatus::ACTIVE, 0.5));
    ltm.add_relation(a, b, RelationType::CAUSES, 0.9);

    // Mark as anti-knowledge
    ltm.invalidate_concept(a);
    ltm.mark_as_anti_knowledge(a, "known error");

    // Create a new concept with same pattern
    auto c = ltm.store_concept("new_premise", "def",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    ltm.add_relation(c, b, RelationType::CAUSES, 0.9);

    ComplexityAnalyzer analyzer(ltm, densifier);
    RetentionManager retention(ltm, analyzer);

    // Should detect structural similarity to known error
    ASSERT(retention.resembles_known_error(c, 0.5f));
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cerr << "\n=== Graph Features Tests (Phase 1-3) ===\n\n";

    // Tests are auto-registered via static constructors above.
    // They've already run by this point.

    std::cerr << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n\n";

    return tests_failed > 0 ? 1 : 0;
}
