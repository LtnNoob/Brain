#pragma once

#include "../ltm/long_term_memory.hpp"
#include "text_chunker.hpp"
#include "entity_extractor.hpp"
#include "relation_extractor.hpp"
#include "trust_tagger.hpp"
#include "proposal_queue.hpp"
#include "knowledge_ingestor.hpp"
#include <string>
#include <vector>
#include <functional>
#include <map>

namespace brain19 {

// IngestionResult: Result of a complete ingestion pipeline run
struct IngestionResult {
    bool success;
    std::string error_message;

    // Pipeline statistics
    size_t chunks_created;
    size_t entities_extracted;
    size_t relations_extracted;
    size_t proposals_created;
    size_t proposals_approved;
    size_t concepts_stored;
    size_t relations_stored;

    // IDs of stored concepts (for caller to use)
    std::vector<ConceptId> stored_concept_ids;
    // IDs of stored relations
    std::vector<RelationId> stored_relation_ids;

    IngestionResult()
        : success(false), chunks_created(0), entities_extracted(0),
          relations_extracted(0), proposals_created(0), proposals_approved(0),
          concepts_stored(0), relations_stored(0) {}
};

// IngestionPipeline: Complete pipeline connecting all components to LTM
//
// ARCHITECTURE:
//
//   Input (JSON/CSV/Text)
//          |
//   KnowledgeIngestor (structured) / TextChunker (plain text)
//          |
//   EntityExtractor + RelationExtractor
//          |
//   TrustTagger (assigns trust suggestions)
//          |
//   ProposalQueue (staging area for review)
//          |
//   [Human Review] (approve/reject/modify)
//          |
//   LTM.store_concept() + LTM.add_relation()
//
// CRITICAL CONTRACTS:
// - Pipeline NEVER writes to LTM without going through ProposalQueue
// - All trust assignments are SUGGESTIONS until review
// - Existing LTM data is NEVER modified by ingestion
// - New concepts get new IDs (no ID collision with existing)
// - Pipeline is ADDITIVE only (no delete, no modify)
//
// MODES:
// - Interactive: proposals go to queue, wait for human review
// - Auto-approve: proposals are automatically approved (for trusted sources)
class IngestionPipeline {
public:
    explicit IngestionPipeline(LongTermMemory& ltm);

    // =========================================================================
    // STRUCTURED INPUT (JSON/CSV)
    // =========================================================================

    // Ingest from JSON string
    IngestionResult ingest_json(const std::string& json_str, bool auto_approve = false);

    // Ingest from CSV (concepts + optional relations)
    IngestionResult ingest_csv(const std::string& concepts_csv,
                              const std::string& relations_csv = "",
                              bool auto_approve = false);

    // =========================================================================
    // PLAIN TEXT INPUT
    // =========================================================================

    // Ingest from plain text (uses chunker + extractors)
    IngestionResult ingest_text(const std::string& text,
                               const std::string& source_ref = "",
                               bool auto_approve = false);

    // =========================================================================
    // PROPOSAL QUEUE ACCESS
    // =========================================================================

    // Get the proposal queue for manual review
    ProposalQueue& get_queue() { return queue_; }
    const ProposalQueue& get_queue() const { return queue_; }

    // Commit approved proposals to LTM
    IngestionResult commit_approved();

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    // Configure pipeline components
    void set_chunker_config(const TextChunker::Config& config);
    void set_entity_config(const EntityExtractor::Config& config);
    void set_relation_config(const RelationExtractor::Config& config);

    // Set default trust category for text ingestion
    void set_default_trust(TrustCategory category);

private:
    LongTermMemory& ltm_;
    ProposalQueue queue_;
    TrustTagger tagger_;
    KnowledgeIngestor ingestor_;

    TextChunker chunker_;
    EntityExtractor entity_extractor_;
    RelationExtractor relation_extractor_;

    TrustCategory default_trust_;

    // Internal: process text through NLP pipeline
    std::vector<IngestProposal> process_text_pipeline(
        const std::string& text,
        const std::string& source_ref);

    // Internal: commit a single proposal to LTM
    struct CommitResult {
        ConceptId concept_id;
        std::vector<RelationId> relation_ids;
        bool success;
    };
    CommitResult commit_proposal(const IngestProposal& proposal);

    // Resolve concept labels to IDs (find existing or use newly created)
    ConceptId resolve_concept_id(const std::string& label,
                                std::map<std::string, ConceptId>& label_map);
};

} // namespace brain19
