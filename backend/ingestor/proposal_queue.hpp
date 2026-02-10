#pragma once

#include "../importers/knowledge_proposal.hpp"
#include "trust_tagger.hpp"
#include "entity_extractor.hpp"
#include "relation_extractor.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <optional>
#include <chrono>

namespace brain19 {

// ProposalStatus: Lifecycle of a proposal in the queue
enum class ProposalStatus {
    PENDING,        // Awaiting review
    APPROVED,       // Accepted for LTM ingestion
    REJECTED,       // Rejected by reviewer
    MODIFIED,       // Approved with modifications
    EXPIRED         // Timed out without review
};

// IngestProposal: A single concept or relation proposed for ingestion
//
// Extends KnowledgeProposal with trust assignment and review metadata.
// This is the unit of work in the ProposalQueue.
struct IngestProposal {
    uint64_t id;
    ProposalStatus status;

    // What is being proposed
    std::string concept_label;
    std::string concept_definition;
    std::vector<ExtractedRelation> proposed_relations;

    // Trust assignment (from TrustTagger)
    TrustAssignment trust_assignment;

    // Source information
    std::string source_text;            // Original text evidence
    std::string source_reference;       // URL, file path, etc.

    // Review metadata
    std::string reviewer_notes;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point reviewed_at;

    IngestProposal()
        : id(0)
        , status(ProposalStatus::PENDING)
        , trust_assignment(TrustTagger().assign_trust(TrustCategory::SPECULATION))
        , created_at(std::chrono::system_clock::now())
        , reviewed_at(std::chrono::system_clock::time_point{})
    {}
};

// ReviewDecision: Result of reviewing a proposal
struct ReviewDecision {
    ProposalStatus new_status;
    std::optional<TrustCategory> override_trust;   // Override trust category
    std::optional<double> override_trust_value;     // Override specific trust value
    std::string notes;

    // Convenience constructors
    static ReviewDecision approve(const std::string& notes = "") {
        ReviewDecision d;
        d.new_status = ProposalStatus::APPROVED;
        d.notes = notes;
        return d;
    }

    static ReviewDecision reject(const std::string& notes = "") {
        ReviewDecision d;
        d.new_status = ProposalStatus::REJECTED;
        d.notes = notes;
        return d;
    }

    static ReviewDecision approve_with_trust(TrustCategory cat, const std::string& notes = "") {
        ReviewDecision d;
        d.new_status = ProposalStatus::MODIFIED;
        d.override_trust = cat;
        d.notes = notes;
        return d;
    }

private:
    ReviewDecision() : new_status(ProposalStatus::PENDING) {}
};

// QueueStats: Summary of queue state
struct QueueStats {
    size_t total;
    size_t pending;
    size_t approved;
    size_t rejected;
    size_t modified;
    size_t expired;
};

// ProposalQueue: Queue for unvalidated knowledge proposals awaiting review
//
// DESIGN:
// - Stores IngestProposals before they enter LTM
// - Supports review operations (approve/reject/modify)
// - Provides batch operations for efficiency
// - Preserves audit trail of all decisions
// - Thread-safe is NOT guaranteed (single-threaded design)
//
// EPISTEMIC RULE:
// - Nothing enters LTM without going through this queue
// - All trust assignments are SUGGESTIONS until review
// - Reviewer has final authority over epistemic metadata
class ProposalQueue {
public:
    ProposalQueue();

    // Add proposals to queue
    uint64_t enqueue(IngestProposal proposal);

    // Batch enqueue
    std::vector<uint64_t> enqueue_batch(std::vector<IngestProposal> proposals);

    // Review operations
    bool review(uint64_t proposal_id, const ReviewDecision& decision);
    size_t review_batch(const std::vector<uint64_t>& ids, const ReviewDecision& decision);

    // Auto-approve all pending (for trusted sources)
    size_t auto_approve_all();

    // Retrieve proposals
    std::optional<IngestProposal> get_proposal(uint64_t id) const;
    std::vector<IngestProposal> get_pending() const;
    std::vector<IngestProposal> get_approved() const;
    std::vector<IngestProposal> get_rejected() const;
    std::vector<IngestProposal> get_all() const;

    // Pop approved proposals (removes from queue)
    std::vector<IngestProposal> pop_approved();

    // Query
    bool has_pending() const;
    QueueStats get_stats() const;
    size_t size() const;
    bool empty() const;

    // Cleanup
    void clear();
    size_t expire_old(std::chrono::seconds max_age);

private:
    std::vector<IngestProposal> proposals_;
    uint64_t next_id_;

    void apply_trust_override(IngestProposal& proposal, const ReviewDecision& decision);
};

} // namespace brain19
