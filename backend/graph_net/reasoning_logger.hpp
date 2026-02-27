#pragma once

#include "epistemic_trace.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../micromodel/embedding_manager.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace brain19 {

// =============================================================================
// ReasoningLogger --- JSONL Training Data for Orchestrator
// =============================================================================
//
// Appends one JSONL line per reason_from() call with all orchestrator-relevant
// features: seed info, chain metrics, per-step dual-neuron ratios, pain/reward
// signals, termination reason, timestamps.
//
// Thread-safe (mutex-guarded file writes). Zero overhead when not attached
// to a GraphReasoner (nullptr check in reasoner).
//

class ReasoningLogger {
public:
    explicit ReasoningLogger(const std::string& path)
        : file_(path, std::ios::app) {}

    ~ReasoningLogger() {
        if (file_.is_open()) file_.close();
    }

    // Non-copyable
    ReasoningLogger(const ReasoningLogger&) = delete;
    ReasoningLogger& operator=(const ReasoningLogger&) = delete;

    bool is_open() const { return file_.is_open(); }

    void log_chain(ConceptId seed,
                   const GraphChain& chain,
                   double chain_quality,
                   const EmbeddingManager& embeddings,
                   const LongTermMemory& ltm,
                   int feedback_round = -1) const
    {
        if (!file_.is_open() || chain.steps.empty()) return;

        std::string line = build_json(seed, chain, chain_quality,
                                       embeddings, ltm, feedback_round);

        std::lock_guard<std::mutex> lock(mutex_);
        file_ << line << '\n';
        file_.flush();
    }

private:
    mutable std::ofstream file_;
    mutable std::mutex mutex_;

    std::string build_json(ConceptId seed,
                           const GraphChain& chain,
                           double chain_quality,
                           const EmbeddingManager& embeddings,
                           const LongTermMemory& ltm,
                           int fb_round) const
    {
        std::ostringstream j;
        j << std::fixed;
        j.precision(4);

        // Timestamp
        j << "{\"ts\":" << now_epoch_ms();

        // Seed info
        j << ",\"seed\":" << seed;
        auto seed_info = ltm.retrieve_concept(seed);
        if (seed_info) {
            j << ",\"seed_label\":\"" << json_escape(seed_info->label) << "\"";
            j << ",\"seed_trust\":" << seed_info->epistemic.trust;
            j << ",\"seed_type\":\"" << epistemic_type_to_string(seed_info->epistemic.type) << "\"";
        } else {
            j << ",\"seed_label\":\"?\"";
            j << ",\"seed_trust\":0.5,\"seed_type\":\"HYPOTHESIS\"";
        }
        j << ",\"seed_rel_count\":" << ltm.get_relation_count(seed);

        // Seed embedding (16D core)
        auto seed_emb = embeddings.concept_embeddings().get_or_default(seed);
        j << ",\"seed_emb\":[";
        for (size_t i = 0; i < CORE_DIM; ++i) {
            if (i > 0) j << ',';
            j << seed_emb.core[i];
        }
        j << ']';

        // Chain-level metrics
        j << ",\"chain_len\":" << chain.length();
        j << ",\"chain_trust\":" << chain.chain_trust;
        j << ",\"chain_quality\":" << chain_quality;
        j << ",\"termination\":\"" << termination_reason_to_string(chain.termination) << "\"";
        j << ",\"mag_ratio\":" << chain.magnitude_ratio;

        // Chain pain (inline ChainSignal::chain_pain() logic)
        double cpain = 0.0;
        switch (chain.termination) {
            case TerminationReason::MAX_STEPS_REACHED:    cpain = 0.0; break;
            case TerminationReason::SEED_DRIFT:           cpain = 0.4; break;
            case TerminationReason::NO_VIABLE_CANDIDATES: cpain = 0.7; break;
            case TerminationReason::ACTIVATION_DECAY:     cpain = 0.5; break;
            case TerminationReason::TRUST_TOO_LOW:        cpain = 0.6; break;
            case TerminationReason::COHERENCE_GATE:       cpain = 0.3; break;
            default: cpain = 0.0; break;
        }
        j << ",\"chain_pain\":" << cpain;

        // Concept sequence (IDs + labels)
        auto concepts = chain.concept_sequence();
        j << ",\"concepts\":[";
        for (size_t i = 0; i < concepts.size(); ++i) {
            if (i > 0) j << ',';
            j << concepts[i];
        }
        j << "],\"labels\":[";
        for (size_t i = 0; i < concepts.size(); ++i) {
            if (i > 0) j << ',';
            auto info = ltm.retrieve_concept(concepts[i]);
            j << '"' << json_escape(info ? info->label : "?") << '"';
        }
        j << ']';

        // Relation sequence
        auto relations = chain.relation_sequence();
        j << ",\"relations\":[";
        for (size_t i = 0; i < relations.size(); ++i) {
            if (i > 0) j << ',';
            j << '"' << relation_type_to_string(relations[i]) << '"';
        }
        j << ']';

        // Per-step details + running aggregates
        double sum_nn = 0.0, sum_kan = 0.0;
        size_t nn_dom = 0, kan_dom = 0;
        double total_reward = 0.0, total_pain = 0.0;

        j << ",\"steps\":[";
        bool first = true;
        for (size_t i = 1; i < chain.steps.size(); ++i) {
            const auto& s = chain.steps[i];

            if (!first) j << ',';
            first = false;

            // Embedding similarity for pain/reward computation
            auto src_emb = embeddings.concept_embeddings().get_or_default(s.source_id);
            auto tgt_emb = embeddings.concept_embeddings().get_or_default(s.target_id);
            double emb_sim = core_similarity(src_emb, tgt_emb);

            auto src_info = ltm.retrieve_concept(s.source_id);
            auto tgt_info = ltm.retrieve_concept(s.target_id);
            j << "{\"s\":" << s.source_id
              << ",\"s_label\":\"" << json_escape(src_info ? src_info->label : "?") << "\""
              << ",\"t\":" << s.target_id
              << ",\"t_label\":\"" << json_escape(tgt_info ? tgt_info->label : "?") << "\""
              << ",\"r\":\"" << relation_type_to_string(s.relation) << "\""
              << ",\"nn\":" << s.nn_quality
              << ",\"kan\":" << s.kan_quality
              << ",\"gate\":" << s.kan_gate
              << ",\"tq\":" << s.transform_quality
              << ",\"coh\":" << s.coherence
              << ",\"cs\":" << s.composite_score
              << ",\"st\":" << s.step_trust
              << ",\"ss\":" << s.seed_similarity
              << '}';

            // Aggregates
            sum_nn += s.nn_quality;
            sum_kan += s.kan_quality;
            if (s.nn_quality > s.kan_quality) ++nn_dom;
            else if (s.kan_quality > s.nn_quality) ++kan_dom;

            // Pain/reward (inline EdgeSignal formulas)
            bool is_positive = (s.composite_score >= 0.5);
            if (is_positive) {
                total_reward += 0.4 * s.transform_quality + 0.3 * s.coherence
                              + 0.2 * s.epistemic_alignment + 0.1 * emb_sim;
            } else {
                double q = 0.5 * s.transform_quality + 0.3 * s.coherence + 0.2 * emb_sim;
                total_pain += std::max(0.0, 1.0 - q);
            }
        }
        j << ']';

        // Aggregated metrics
        size_t step_count = chain.steps.size() > 1 ? chain.steps.size() - 1 : 0;
        double sc = static_cast<double>(step_count);
        j << ",\"avg_nn\":" << (step_count > 0 ? sum_nn / sc : 0.0);
        j << ",\"avg_kan\":" << (step_count > 0 ? sum_kan / sc : 0.0);
        j << ",\"nn_dom\":" << nn_dom;
        j << ",\"kan_dom\":" << kan_dom;
        j << ",\"total_reward\":" << total_reward;
        j << ",\"total_pain\":" << total_pain;
        j << ",\"fb_round\":" << fb_round;

        j << '}';
        return j.str();
    }

    static std::string json_escape(const std::string& s) {
        std::string result;
        result.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '"':  result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\t': result += "\\t"; break;
                case '\r': result += "\\r"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x",
                                      static_cast<unsigned char>(c));
                        result += buf;
                    } else {
                        result += c;
                    }
                    break;
            }
        }
        return result;
    }

    static uint64_t now_epoch_ms() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch());
        return static_cast<uint64_t>(ms.count());
    }
};

} // namespace brain19
