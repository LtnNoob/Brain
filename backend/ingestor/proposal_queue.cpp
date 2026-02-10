#include "proposal_queue.hpp"
#include <algorithm>

namespace brain19 {

ProposalQueue::ProposalQueue()
    : next_id_(1)
{
}

uint64_t ProposalQueue::enqueue(IngestProposal proposal) {
    proposal.id = next_id_++;
    proposal.status = ProposalStatus::PENDING;
    proposal.created_at = std::chrono::system_clock::now();
    proposals_.push_back(std::move(proposal));
    return proposals_.back().id;
}

std::vector<uint64_t> ProposalQueue::enqueue_batch(std::vector<IngestProposal> proposals) {
    std::vector<uint64_t> ids;
    ids.reserve(proposals.size());
    for (auto& p : proposals) {
        ids.push_back(enqueue(std::move(p)));
    }
    return ids;
}

bool ProposalQueue::review(uint64_t proposal_id, const ReviewDecision& decision) {
    for (auto& p : proposals_) {
        if (p.id == proposal_id && p.status == ProposalStatus::PENDING) {
            p.status = decision.new_status;
            p.reviewer_notes = decision.notes;
            p.reviewed_at = std::chrono::system_clock::now();

            if (decision.override_trust.has_value() || decision.override_trust_value.has_value()) {
                apply_trust_override(p, decision);
            }

            return true;
        }
    }
    return false;
}

size_t ProposalQueue::review_batch(const std::vector<uint64_t>& ids, const ReviewDecision& decision) {
    size_t count = 0;
    for (uint64_t id : ids) {
        if (review(id, decision)) {
            ++count;
        }
    }
    return count;
}

size_t ProposalQueue::auto_approve_all() {
    size_t count = 0;
    auto decision = ReviewDecision::approve("Auto-approved");
    for (auto& p : proposals_) {
        if (p.status == ProposalStatus::PENDING) {
            p.status = ProposalStatus::APPROVED;
            p.reviewer_notes = "Auto-approved";
            p.reviewed_at = std::chrono::system_clock::now();
            ++count;
        }
    }
    return count;
}

std::optional<IngestProposal> ProposalQueue::get_proposal(uint64_t id) const {
    for (const auto& p : proposals_) {
        if (p.id == id) {
            return p;
        }
    }
    return std::nullopt;
}

std::vector<IngestProposal> ProposalQueue::get_pending() const {
    std::vector<IngestProposal> result;
    for (const auto& p : proposals_) {
        if (p.status == ProposalStatus::PENDING) {
            result.push_back(p);
        }
    }
    return result;
}

std::vector<IngestProposal> ProposalQueue::get_approved() const {
    std::vector<IngestProposal> result;
    for (const auto& p : proposals_) {
        if (p.status == ProposalStatus::APPROVED || p.status == ProposalStatus::MODIFIED) {
            result.push_back(p);
        }
    }
    return result;
}

std::vector<IngestProposal> ProposalQueue::get_rejected() const {
    std::vector<IngestProposal> result;
    for (const auto& p : proposals_) {
        if (p.status == ProposalStatus::REJECTED) {
            result.push_back(p);
        }
    }
    return result;
}

std::vector<IngestProposal> ProposalQueue::get_all() const {
    return proposals_;
}

std::vector<IngestProposal> ProposalQueue::pop_approved() {
    std::vector<IngestProposal> approved;

    // Partition: move approved to result, keep rest
    auto it = std::stable_partition(proposals_.begin(), proposals_.end(),
        [](const IngestProposal& p) {
            return p.status != ProposalStatus::APPROVED &&
                   p.status != ProposalStatus::MODIFIED;
        });

    for (auto jt = it; jt != proposals_.end(); ++jt) {
        approved.push_back(std::move(*jt));
    }
    proposals_.erase(it, proposals_.end());

    return approved;
}

bool ProposalQueue::has_pending() const {
    return std::any_of(proposals_.begin(), proposals_.end(),
        [](const IngestProposal& p) { return p.status == ProposalStatus::PENDING; });
}

QueueStats ProposalQueue::get_stats() const {
    QueueStats stats{};
    stats.total = proposals_.size();
    for (const auto& p : proposals_) {
        switch (p.status) {
            case ProposalStatus::PENDING: ++stats.pending; break;
            case ProposalStatus::APPROVED: ++stats.approved; break;
            case ProposalStatus::REJECTED: ++stats.rejected; break;
            case ProposalStatus::MODIFIED: ++stats.modified; break;
            case ProposalStatus::EXPIRED: ++stats.expired; break;
        }
    }
    return stats;
}

size_t ProposalQueue::size() const {
    return proposals_.size();
}

bool ProposalQueue::empty() const {
    return proposals_.empty();
}

void ProposalQueue::clear() {
    proposals_.clear();
}

size_t ProposalQueue::expire_old(std::chrono::seconds max_age) {
    auto now = std::chrono::system_clock::now();
    size_t count = 0;

    for (auto& p : proposals_) {
        if (p.status == ProposalStatus::PENDING) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(now - p.created_at);
            if (age > max_age) {
                p.status = ProposalStatus::EXPIRED;
                p.reviewer_notes = "Expired after " + std::to_string(max_age.count()) + " seconds";
                ++count;
            }
        }
    }

    return count;
}

void ProposalQueue::apply_trust_override(IngestProposal& proposal, const ReviewDecision& decision) {
    TrustTagger tagger;

    if (decision.override_trust.has_value()) {
        if (decision.override_trust_value.has_value()) {
            proposal.trust_assignment = tagger.assign_trust_with_value(
                decision.override_trust.value(),
                decision.override_trust_value.value());
        } else {
            proposal.trust_assignment = tagger.assign_trust(decision.override_trust.value());
        }
    } else if (decision.override_trust_value.has_value()) {
        // Override value but keep category
        auto range = tagger.get_trust_range(proposal.trust_assignment.category);
        double val = std::clamp(decision.override_trust_value.value(),
                               range.min_trust, range.max_trust);
        proposal.trust_assignment.trust_value = val;
    }
}

} // namespace brain19
