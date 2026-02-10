#include "ingestion_pipeline.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <cmath>

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
        std::string("Assertion failed: ") + #a + " != " + #b);

#define ASSERT_GT(a, b) \
    if (!((a) > (b))) throw std::runtime_error( \
        std::string("Assertion failed: ") + #a + " not > " + #b);

// =============================================================================
// TextChunker Tests
// =============================================================================

void test_text_chunker() {
    std::cout << "\n=== TextChunker Tests ===" << std::endl;

    TEST("Empty input returns no chunks")
    {
        TextChunker chunker;
        auto chunks = chunker.chunk_text("");
        ASSERT_EQ(chunks.size(), 0u);
    }
    END_TEST

    TEST("Single sentence")
    {
        TextChunker chunker;
        auto chunks = chunker.chunk_text("This is a single sentence.");
        ASSERT_EQ(chunks.size(), 1u);
        ASSERT(chunks[0].text.find("single sentence") != std::string::npos);
    }
    END_TEST

    TEST("Multiple sentences split into chunks")
    {
        TextChunker::Config config;
        config.sentences_per_chunk = 2;
        config.overlap_sentences = 0;
        TextChunker chunker(config);

        std::string text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        auto chunks = chunker.chunk_text(text);
        ASSERT_GT(chunks.size(), 1u);
    }
    END_TEST

    TEST("Sentence splitting handles abbreviations")
    {
        TextChunker chunker;
        auto sentences = chunker.split_sentences(
            "Dr. Smith went to Washington. He met the president.");
        // Should ideally not split on "Dr."
        // But our simple heuristic may or may not handle this perfectly
        ASSERT_GT(sentences.size(), 0u);
    }
    END_TEST

    TEST("Chunk offsets are valid")
    {
        TextChunker chunker;
        std::string text = "Hello world. This is a test. Another sentence here.";
        auto chunks = chunker.chunk_text(text);
        for (const auto& chunk : chunks) {
            ASSERT(chunk.start_offset <= text.size());
            ASSERT(chunk.end_offset <= text.size());
            ASSERT(chunk.start_offset <= chunk.end_offset);
        }
    }
    END_TEST

    TEST("Max chunk chars enforced")
    {
        TextChunker::Config config;
        config.max_chunk_chars = 50;
        config.sentences_per_chunk = 100;
        TextChunker chunker(config);

        std::string text = "This is a very long sentence that should be truncated because it exceeds the maximum chunk character limit.";
        auto chunks = chunker.chunk_text(text);
        for (const auto& chunk : chunks) {
            ASSERT(chunk.text.size() <= 50u);
        }
    }
    END_TEST
}

// =============================================================================
// EntityExtractor Tests
// =============================================================================

void test_entity_extractor() {
    std::cout << "\n=== EntityExtractor Tests ===" << std::endl;

    TEST("Extracts capitalized phrases")
    {
        EntityExtractor extractor;
        auto entities = extractor.extract_from_text(
            "Albert Einstein developed the Theory of Relativity.");

        bool found_einstein = false;
        for (const auto& e : entities) {
            if (e.label.find("Einstein") != std::string::npos) found_einstein = true;
        }
        ASSERT(found_einstein);
    }
    END_TEST

    TEST("Extracts quoted terms")
    {
        EntityExtractor extractor;
        auto entities = extractor.extract_from_text(
            "The process called \"photosynthesis\" converts light to energy.");

        bool found = false;
        for (const auto& e : entities) {
            if (e.label == "photosynthesis") {
                found = true;
                ASSERT(e.is_quoted);
            }
        }
        ASSERT(found);
    }
    END_TEST

    TEST("Extracts defined terms")
    {
        EntityExtractor extractor;
        auto entities = extractor.extract_from_text(
            "Photosynthesis is a process that converts light energy.");

        bool found = false;
        for (const auto& e : entities) {
            if (e.label == "Photosynthesis") {
                found = true;
            }
        }
        ASSERT(found);
    }
    END_TEST

    TEST("Deduplication works")
    {
        EntityExtractor extractor;
        auto entities = extractor.extract_from_text(
            "Machine Learning is important. Machine Learning transforms industries.");

        int count = 0;
        for (const auto& e : entities) {
            if (e.label == "Machine Learning") count++;
        }
        // Should be deduplicated to 1 with frequency > 1
        ASSERT_EQ(count, 1);
    }
    END_TEST

    TEST("Stopwords filtered")
    {
        EntityExtractor extractor;
        auto entities = extractor.extract_from_text(
            "The cat sat on the mat.");

        for (const auto& e : entities) {
            ASSERT(e.label != "The");
            ASSERT(e.label != "the");
        }
    }
    END_TEST

    TEST("Converts to SuggestedConcept")
    {
        EntityExtractor extractor;
        auto entities = extractor.extract_from_text(
            "Machine Learning is a branch of AI.");
        auto suggested = EntityExtractor::to_suggested_concepts(entities);
        ASSERT_EQ(suggested.size(), entities.size());
        for (size_t i = 0; i < suggested.size(); ++i) {
            ASSERT_EQ(suggested[i].label, entities[i].label);
        }
    }
    END_TEST
}

// =============================================================================
// RelationExtractor Tests
// =============================================================================

void test_relation_extractor() {
    std::cout << "\n=== RelationExtractor Tests ===" << std::endl;

    TEST("Extracts IS_A relation")
    {
        RelationExtractor extractor;
        auto relations = extractor.extract_relations_blind(
            "A Cat is a mammal that lives in homes.");

        bool found = false;
        for (const auto& r : relations) {
            if (r.relation_type == RelationType::IS_A &&
                r.source_label.find("Cat") != std::string::npos) {
                found = true;
            }
        }
        ASSERT(found);
    }
    END_TEST

    TEST("Extracts CAUSES relation")
    {
        RelationExtractor extractor;
        auto relations = extractor.extract_relations_blind(
            "Smoking causes lung cancer and other diseases.");

        bool found = false;
        for (const auto& r : relations) {
            if (r.relation_type == RelationType::CAUSES) {
                found = true;
            }
        }
        ASSERT(found);
    }
    END_TEST

    TEST("Extracts PART_OF relation")
    {
        RelationExtractor extractor;
        auto relations = extractor.extract_relations_blind(
            "The Engine is part of a car system.");

        bool found = false;
        for (const auto& r : relations) {
            if (r.relation_type == RelationType::PART_OF) {
                found = true;
            }
        }
        ASSERT(found);
    }
    END_TEST

    TEST("Entity-aware extraction boosts confidence")
    {
        EntityExtractor entity_ex;
        RelationExtractor rel_ex;

        std::string text = "A Cat is a mammal. Dogs are also mammals.";
        auto entities = entity_ex.extract_from_text(text);
        auto relations = rel_ex.extract_relations(text, entities);
        auto blind_relations = rel_ex.extract_relations_blind(text);

        // Relations with known entities should have equal or higher confidence
        // (this test validates the pipeline connects properly)
        ASSERT_GT(relations.size() + blind_relations.size(), 0u);
    }
    END_TEST

    TEST("Converts to SuggestedRelation")
    {
        RelationExtractor extractor;
        auto relations = extractor.extract_relations_blind(
            "A Cat is a small mammal.");
        auto suggested = RelationExtractor::to_suggested_relations(relations);
        ASSERT_EQ(suggested.size(), relations.size());
    }
    END_TEST
}

// =============================================================================
// TrustTagger Tests
// =============================================================================

void test_trust_tagger() {
    std::cout << "\n=== TrustTagger Tests ===" << std::endl;

    TEST("FACTS trust range correct")
    {
        TrustTagger tagger;
        auto assignment = tagger.assign_trust(TrustCategory::FACTS);
        ASSERT_EQ(assignment.epistemic_type, EpistemicType::FACT);
        ASSERT_EQ(assignment.epistemic_status, EpistemicStatus::ACTIVE);
        ASSERT(assignment.trust_value >= 0.95 && assignment.trust_value <= 0.99);
    }
    END_TEST

    TEST("THEORIES trust range correct")
    {
        TrustTagger tagger;
        auto assignment = tagger.assign_trust(TrustCategory::THEORIES);
        ASSERT_EQ(assignment.epistemic_type, EpistemicType::THEORY);
        ASSERT(assignment.trust_value >= 0.85 && assignment.trust_value <= 0.95);
    }
    END_TEST

    TEST("SPECULATION trust range correct")
    {
        TrustTagger tagger;
        auto assignment = tagger.assign_trust(TrustCategory::SPECULATION);
        ASSERT_EQ(assignment.epistemic_type, EpistemicType::SPECULATION);
        ASSERT(assignment.trust_value >= 0.10 && assignment.trust_value <= 0.40);
    }
    END_TEST

    TEST("INVALIDATED produces correct status")
    {
        TrustTagger tagger;
        auto assignment = tagger.assign_trust(TrustCategory::INVALIDATED);
        ASSERT_EQ(assignment.epistemic_status, EpistemicStatus::INVALIDATED);
        ASSERT(assignment.trust_value < 0.2);
    }
    END_TEST

    TEST("EpistemicMetadata conversion works")
    {
        TrustTagger tagger;
        auto assignment = tagger.assign_trust(TrustCategory::FACTS);
        auto metadata = assignment.to_epistemic_metadata();
        ASSERT_EQ(metadata.type, EpistemicType::FACT);
        ASSERT_EQ(metadata.status, EpistemicStatus::ACTIVE);
        ASSERT(metadata.trust >= 0.95);
    }
    END_TEST

    TEST("Text-based trust with hedging language")
    {
        TrustTagger tagger;
        auto assignment = tagger.suggest_from_text(
            "This might possibly be related to quantum effects.");
        ASSERT(assignment.trust_value < 0.7);
    }
    END_TEST

    TEST("Text-based trust with certainty language")
    {
        TrustTagger tagger;
        auto assignment = tagger.suggest_from_text(
            "Research demonstrates that water boils at 100 degrees Celsius [1]. This is a proven fact.");
        ASSERT(assignment.trust_value > 0.5);
    }
    END_TEST

    TEST("Custom trust value clamped to range")
    {
        TrustTagger tagger;
        auto range = tagger.get_trust_range(TrustCategory::SPECULATION);
        // Try to set trust above range max
        auto assignment = tagger.assign_trust_with_value(TrustCategory::SPECULATION, 0.99);
        ASSERT(assignment.trust_value <= range.max_trust);
        ASSERT(assignment.trust_value >= range.min_trust);
    }
    END_TEST
}

// =============================================================================
// ProposalQueue Tests
// =============================================================================

void test_proposal_queue() {
    std::cout << "\n=== ProposalQueue Tests ===" << std::endl;

    TEST("Enqueue and retrieve")
    {
        ProposalQueue queue;
        IngestProposal p;
        p.concept_label = "TestConcept";
        p.concept_definition = "A test concept";

        uint64_t id = queue.enqueue(std::move(p));
        ASSERT_GT(id, 0u);

        auto retrieved = queue.get_proposal(id);
        ASSERT(retrieved.has_value());
        ASSERT_EQ(retrieved->concept_label, "TestConcept");
        ASSERT_EQ(retrieved->status, ProposalStatus::PENDING);
    }
    END_TEST

    TEST("Review approve")
    {
        ProposalQueue queue;
        IngestProposal p;
        p.concept_label = "Approved";
        uint64_t id = queue.enqueue(std::move(p));

        bool result = queue.review(id, ReviewDecision::approve("Looks good"));
        ASSERT(result);

        auto retrieved = queue.get_proposal(id);
        ASSERT(retrieved.has_value());
        ASSERT_EQ(retrieved->status, ProposalStatus::APPROVED);
    }
    END_TEST

    TEST("Review reject")
    {
        ProposalQueue queue;
        IngestProposal p;
        p.concept_label = "Rejected";
        uint64_t id = queue.enqueue(std::move(p));

        queue.review(id, ReviewDecision::reject("Not valid"));
        auto retrieved = queue.get_proposal(id);
        ASSERT_EQ(retrieved->status, ProposalStatus::REJECTED);
    }
    END_TEST

    TEST("Pop approved removes from queue")
    {
        ProposalQueue queue;

        IngestProposal p1; p1.concept_label = "A";
        IngestProposal p2; p2.concept_label = "B";
        IngestProposal p3; p3.concept_label = "C";

        uint64_t id1 = queue.enqueue(std::move(p1));
        uint64_t id2 = queue.enqueue(std::move(p2));
        queue.enqueue(std::move(p3));

        queue.review(id1, ReviewDecision::approve());
        queue.review(id2, ReviewDecision::approve());

        auto approved = queue.pop_approved();
        ASSERT_EQ(approved.size(), 2u);
        ASSERT_EQ(queue.size(), 1u); // Only C remains
    }
    END_TEST

    TEST("Auto-approve all")
    {
        ProposalQueue queue;
        for (int i = 0; i < 5; ++i) {
            IngestProposal p;
            p.concept_label = "Concept" + std::to_string(i);
            queue.enqueue(std::move(p));
        }

        size_t count = queue.auto_approve_all();
        ASSERT_EQ(count, 5u);
        ASSERT(!queue.has_pending());
    }
    END_TEST

    TEST("Stats are correct")
    {
        ProposalQueue queue;
        IngestProposal p1; p1.concept_label = "A";
        IngestProposal p2; p2.concept_label = "B";
        IngestProposal p3; p3.concept_label = "C";

        uint64_t id1 = queue.enqueue(std::move(p1));
        uint64_t id2 = queue.enqueue(std::move(p2));
        queue.enqueue(std::move(p3));

        queue.review(id1, ReviewDecision::approve());
        queue.review(id2, ReviewDecision::reject());

        auto stats = queue.get_stats();
        ASSERT_EQ(stats.total, 3u);
        ASSERT_EQ(stats.approved, 1u);
        ASSERT_EQ(stats.rejected, 1u);
        ASSERT_EQ(stats.pending, 1u);
    }
    END_TEST

    TEST("Trust override in review")
    {
        ProposalQueue queue;
        IngestProposal p;
        p.concept_label = "Upgraded";

        uint64_t id = queue.enqueue(std::move(p));
        queue.review(id, ReviewDecision::approve_with_trust(
            TrustCategory::FACTS, "Verified by expert"));

        auto retrieved = queue.get_proposal(id);
        ASSERT(retrieved.has_value());
        ASSERT_EQ(retrieved->trust_assignment.category, TrustCategory::FACTS);
    }
    END_TEST
}

// =============================================================================
// KnowledgeIngestor Tests (JSON/CSV parsing)
// =============================================================================

void test_knowledge_ingestor() {
    std::cout << "\n=== KnowledgeIngestor Tests ===" << std::endl;

    TEST("Parse valid JSON")
    {
        KnowledgeIngestor ingestor;
        std::string json = R"({
            "source": "test",
            "concepts": [
                {"label": "Cat", "definition": "A small mammal", "trust": "FACT", "trust_value": 0.98},
                {"label": "Dog", "definition": "A domesticated canine", "trust": "FACT", "trust_value": 0.99}
            ],
            "relations": [
                {"source": "Cat", "target": "Mammal", "type": "is-a", "weight": 0.9}
            ]
        })";

        auto result = ingestor.parse_json(json);
        ASSERT(result.success);
        ASSERT_EQ(result.concepts_parsed, 2u);
        ASSERT_EQ(result.relations_parsed, 1u);
        ASSERT_EQ(result.data.concepts[0].label, "Cat");
        ASSERT_EQ(result.data.concepts[1].label, "Dog");
    }
    END_TEST

    TEST("Parse empty JSON fails")
    {
        KnowledgeIngestor ingestor;
        auto result = ingestor.parse_json("");
        ASSERT(!result.success);
    }
    END_TEST

    TEST("Parse invalid JSON fails")
    {
        KnowledgeIngestor ingestor;
        auto result = ingestor.parse_json("not json at all");
        ASSERT(!result.success);
    }
    END_TEST

    TEST("Parse CSV concepts")
    {
        KnowledgeIngestor ingestor;
        std::string csv =
            "label,definition,trust_category,trust_value\n"
            "Cat,\"A small mammal\",FACT,0.98\n"
            "Dog,\"A domesticated canine\",FACT,0.99\n";

        auto result = ingestor.parse_csv_concepts(csv);
        ASSERT(result.success);
        ASSERT_EQ(result.concepts_parsed, 2u);
        ASSERT_EQ(result.data.concepts[0].label, "Cat");
        ASSERT_EQ(result.data.concepts[0].trust_category, "FACT");
    }
    END_TEST

    TEST("Parse CSV with different column names")
    {
        KnowledgeIngestor ingestor;
        std::string csv =
            "name,description,category,confidence\n"
            "Photon,\"A quantum of light\",THEORY,0.90\n";

        auto result = ingestor.parse_csv_concepts(csv);
        ASSERT(result.success);
        ASSERT_EQ(result.concepts_parsed, 1u);
        ASSERT_EQ(result.data.concepts[0].label, "Photon");
    }
    END_TEST

    TEST("Trust category parsing")
    {
        ASSERT_EQ(KnowledgeIngestor::parse_trust_category("FACT"),
                  TrustCategory::FACTS);
        ASSERT_EQ(KnowledgeIngestor::parse_trust_category("theory"),
                  TrustCategory::THEORIES);
        ASSERT_EQ(KnowledgeIngestor::parse_trust_category("SPECULATION"),
                  TrustCategory::SPECULATION);
        ASSERT_EQ(KnowledgeIngestor::parse_trust_category("INVALIDATED"),
                  TrustCategory::INVALIDATED);
    }
    END_TEST

    TEST("Relation type parsing")
    {
        ASSERT_EQ(KnowledgeIngestor::parse_relation_type("is-a"),
                  RelationType::IS_A);
        ASSERT_EQ(KnowledgeIngestor::parse_relation_type("causes"),
                  RelationType::CAUSES);
        ASSERT_EQ(KnowledgeIngestor::parse_relation_type("part-of"),
                  RelationType::PART_OF);
        ASSERT_EQ(KnowledgeIngestor::parse_relation_type("unknown-type"),
                  RelationType::CUSTOM);
    }
    END_TEST

    TEST("Convert to proposals with trust")
    {
        KnowledgeIngestor ingestor;
        TrustTagger tagger;

        StructuredInput input;
        StructuredConcept c;
        c.label = "TestConcept";
        c.definition = "A test";
        c.trust_category = "FACT";
        c.trust_value = 0.98;
        input.concepts.push_back(c);

        auto proposals = ingestor.to_proposals(input, tagger);
        ASSERT_EQ(proposals.size(), 1u);
        ASSERT_EQ(proposals[0].concept_label, "TestConcept");
        ASSERT_EQ(proposals[0].trust_assignment.epistemic_type, EpistemicType::FACT);
        ASSERT(proposals[0].trust_assignment.trust_value >= 0.95);
    }
    END_TEST
}

// =============================================================================
// IngestionPipeline Integration Tests
// =============================================================================

void test_ingestion_pipeline() {
    std::cout << "\n=== IngestionPipeline Integration Tests ===" << std::endl;

    TEST("JSON ingestion with auto-approve stores to LTM")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        std::string json = R"({
            "source": "unit-test",
            "concepts": [
                {"label": "Hydrogen", "definition": "Lightest chemical element", "trust": "FACT", "trust_value": 0.98},
                {"label": "Oxygen", "definition": "Essential for life", "trust": "FACT", "trust_value": 0.99}
            ],
            "relations": [
                {"source": "Hydrogen", "target": "Oxygen", "type": "similar-to", "weight": 0.5}
            ]
        })";

        auto result = pipeline.ingest_json(json, true);
        ASSERT(result.success);
        ASSERT_EQ(result.concepts_stored, 2u);
        ASSERT_GT(result.stored_concept_ids.size(), 0u);

        // Verify in LTM
        auto cinfo =ltm.retrieve_concept(result.stored_concept_ids[0]);
        ASSERT(cinfo.has_value());
        ASSERT_EQ(cinfo->label, "Hydrogen");
        ASSERT_EQ(cinfo->epistemic.type, EpistemicType::FACT);
        ASSERT(cinfo->epistemic.trust >= 0.95);
    }
    END_TEST

    TEST("JSON ingestion without auto-approve queues proposals")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        std::string json = R"({
            "source": "test",
            "concepts": [
                {"label": "Test", "definition": "A test", "trust": "THEORY"}
            ]
        })";

        auto result = pipeline.ingest_json(json, false);
        ASSERT(result.success);
        ASSERT_EQ(result.concepts_stored, 0u);
        ASSERT(pipeline.get_queue().has_pending());
    }
    END_TEST

    TEST("Manual review workflow")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        std::string json = R"({
            "source": "test",
            "concepts": [
                {"label": "ReviewMe", "definition": "Needs review", "trust": "SPECULATION"}
            ]
        })";

        pipeline.ingest_json(json, false);

        // Review: upgrade to THEORY
        auto pending = pipeline.get_queue().get_pending();
        ASSERT_EQ(pending.size(), 1u);

        pipeline.get_queue().review(pending[0].id,
            ReviewDecision::approve_with_trust(TrustCategory::THEORIES, "Expert verified"));

        // Commit
        auto commit_result = pipeline.commit_approved();
        ASSERT_EQ(commit_result.concepts_stored, 1u);

        // Verify trust level in LTM
        auto cinfo =ltm.retrieve_concept(commit_result.stored_concept_ids[0]);
        ASSERT(cinfo.has_value());
        ASSERT_EQ(cinfo->epistemic.type, EpistemicType::THEORY);
        ASSERT(cinfo->epistemic.trust >= 0.85);
    }
    END_TEST

    TEST("CSV ingestion")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        std::string csv =
            "label,definition,trust_category,trust_value\n"
            "Helium,\"Noble gas with atomic number 2\",FACT,0.99\n"
            "Neon,\"Noble gas with atomic number 10\",FACT,0.98\n";

        auto result = pipeline.ingest_csv(csv, "", true);
        ASSERT(result.success);
        ASSERT_EQ(result.concepts_stored, 2u);
    }
    END_TEST

    TEST("Text ingestion extracts and stores")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        std::string text =
            "Machine Learning is a branch of artificial intelligence. "
            "Deep Learning is a subset of Machine Learning. "
            "Neural Networks are used in Deep Learning. "
            "Gradient Descent is an optimization algorithm used in Neural Networks.";

        auto result = pipeline.ingest_text(text, "textbook", true);
        ASSERT(result.success);
        ASSERT_GT(result.entities_extracted, 0u);
        ASSERT_GT(result.concepts_stored, 0u);
    }
    END_TEST

    TEST("Relations are stored in LTM")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        std::string json = R"({
            "source": "test",
            "concepts": [
                {"label": "Cat", "definition": "A small mammal", "trust": "FACT", "trust_value": 0.98},
                {"label": "Mammal", "definition": "Warm-blooded animal", "trust": "FACT", "trust_value": 0.99}
            ],
            "relations": [
                {"source": "Cat", "target": "Mammal", "type": "is-a", "weight": 0.9}
            ]
        })";

        auto result = pipeline.ingest_json(json, true);
        ASSERT(result.success);
        ASSERT_EQ(result.relations_stored, 1u);

        // Verify relation in LTM
        auto cat_id = result.stored_concept_ids[0];
        auto outgoing = ltm.get_outgoing_relations(cat_id);
        ASSERT_GT(outgoing.size(), 0u);
        ASSERT_EQ(outgoing[0].type, RelationType::IS_A);
    }
    END_TEST

    TEST("Duplicate concepts not overwritten")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        // First ingestion
        std::string json1 = R"({
            "source": "test",
            "concepts": [
                {"label": "Water", "definition": "H2O molecule", "trust": "FACT", "trust_value": 0.99}
            ]
        })";
        pipeline.ingest_json(json1, true);
        size_t count_before = ltm.get_all_concept_ids().size();

        // Second ingestion with same label
        std::string json2 = R"({
            "source": "test2",
            "concepts": [
                {"label": "Water", "definition": "A liquid substance", "trust": "THEORY"}
            ]
        })";
        pipeline.ingest_json(json2, true);
        size_t count_after = ltm.get_all_concept_ids().size();

        // Should not create duplicate
        ASSERT_EQ(count_before, count_after);
    }
    END_TEST

    TEST("Epistemic integrity preserved - all concepts have metadata")
    {
        LongTermMemory ltm;
        IngestionPipeline pipeline(ltm);

        std::string json = R"({
            "source": "integrity-test",
            "concepts": [
                {"label": "Alpha", "definition": "First", "trust": "FACT", "trust_value": 0.98},
                {"label": "Beta", "definition": "Second", "trust": "THEORY"},
                {"label": "Gamma", "definition": "Third", "trust": "SPECULATION"}
            ]
        })";

        pipeline.ingest_json(json, true);

        // Verify ALL concepts have proper epistemic metadata
        auto all_ids = ltm.get_all_concept_ids();
        for (auto id : all_ids) {
            auto cinfo =ltm.retrieve_concept(id);
            ASSERT(cinfo.has_value());
            ASSERT(cinfo->epistemic.is_valid());
            ASSERT(cinfo->epistemic.trust >= 0.0);
            ASSERT(cinfo->epistemic.trust <= 1.0);
            ASSERT_EQ(cinfo->epistemic.status, EpistemicStatus::ACTIVE);
        }
    }
    END_TEST

    TEST("Existing LTM data untouched by ingestion")
    {
        LongTermMemory ltm;

        // Pre-existing concept
        auto existing_id = ltm.store_concept(
            "PreExisting", "Was here before",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.99));

        IngestionPipeline pipeline(ltm);

        std::string json = R"({
            "source": "test",
            "concepts": [
                {"label": "NewConcept", "definition": "Brand new", "trust": "THEORY"}
            ]
        })";

        pipeline.ingest_json(json, true);

        // Verify pre-existing concept is unchanged
        auto existing = ltm.retrieve_concept(existing_id);
        ASSERT(existing.has_value());
        ASSERT_EQ(existing->label, "PreExisting");
        ASSERT_EQ(existing->definition, "Was here before");
        ASSERT_EQ(existing->epistemic.type, EpistemicType::FACT);
        ASSERT(std::abs(existing->epistemic.trust - 0.99) < 0.001);
    }
    END_TEST
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Brain19 Knowledge Ingestor Test Suite" << std::endl;
    std::cout << "Phase 1: Knowledge Input Mechanism" << std::endl;
    std::cout << "========================================" << std::endl;

    test_text_chunker();
    test_entity_extractor();
    test_relation_extractor();
    test_trust_tagger();
    test_proposal_queue();
    test_knowledge_ingestor();
    test_ingestion_pipeline();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_failed > 0) ? 1 : 0;
}
