// Brain19 Interactive CLI Tool
// Links Phase 1 (Ingestor) + Phase 2 (MicroModel) for interactive testing.
//
// Usage:
//   ./tools/brain19_cli           # Direct mode (auto-approve, default)
//   ./tools/brain19_cli --review  # Review mode (manual approve/reject)

#include "../ingestor/ingestion_pipeline.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../micromodel/micro_trainer.hpp"
#include "../micromodel/relevance_map.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>

using namespace brain19;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string read_multiline(const std::string& prompt) {
    std::cout << prompt << " (end with empty line):\n";
    std::string result;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) break;
        if (!result.empty()) result += '\n';
        result += line;
    }
    return result;
}

static std::string read_line(const std::string& prompt) {
    std::cout << prompt << ": ";
    std::string line;
    std::getline(std::cin, line);
    return line;
}

static void print_result(const IngestionResult& r) {
    if (!r.success) {
        std::cout << "  ERROR: " << r.error_message << "\n";
        return;
    }
    std::cout << "  Chunks created:      " << r.chunks_created << "\n"
              << "  Entities extracted:  " << r.entities_extracted << "\n"
              << "  Relations extracted: " << r.relations_extracted << "\n"
              << "  Proposals created:   " << r.proposals_created << "\n"
              << "  Proposals approved:  " << r.proposals_approved << "\n"
              << "  Concepts stored:     " << r.concepts_stored << "\n"
              << "  Relations stored:    " << r.relations_stored << "\n";
}

// ---------------------------------------------------------------------------
// Menu Actions
// ---------------------------------------------------------------------------

static void action_ingest_json(IngestionPipeline& pipeline, bool review_mode) {
    std::string json = read_multiline("Enter JSON");
    if (json.empty()) { std::cout << "  (empty input, skipped)\n"; return; }

    auto r = pipeline.ingest_json(json, !review_mode);
    print_result(r);
    if (review_mode && r.proposals_created > 0) {
        std::cout << "  " << r.proposals_created
                  << " proposal(s) queued for review (menu option 4)\n";
    }
}

static void action_ingest_text(IngestionPipeline& pipeline, bool review_mode) {
    std::string text = read_multiline("Enter text");
    if (text.empty()) { std::cout << "  (empty input, skipped)\n"; return; }

    auto r = pipeline.ingest_text(text, "cli", !review_mode);
    print_result(r);
    if (review_mode && r.proposals_created > 0) {
        std::cout << "  " << r.proposals_created
                  << " proposal(s) queued for review (menu option 4)\n";
    }
}

static void action_show_kg(const LongTermMemory& ltm) {
    auto ids = ltm.get_all_concept_ids();
    if (ids.empty()) {
        std::cout << "  (knowledge graph is empty)\n";
        return;
    }

    std::cout << "\n  --- Concepts (" << ids.size() << ") ---\n";
    for (auto cid : ids) {
        auto cpt = ltm.retrieve_concept(cid);
        if (!cpt) continue;
        std::cout << "  [" << cpt->id << "] " << cpt->label << "\n"
                  << "       def:   " << cpt->definition << "\n"
                  << "       type:  " << epistemic_type_to_string(cpt->epistemic.type)
                  << "  status: " << epistemic_status_to_string(cpt->epistemic.status)
                  << "  trust: " << std::fixed << std::setprecision(2)
                  << cpt->epistemic.trust << "\n";
    }

    std::cout << "\n  --- Relations ---\n";
    bool any = false;
    for (auto cid : ids) {
        auto rels = ltm.get_outgoing_relations(cid);
        for (auto& rel : rels) {
            any = true;
            auto src = ltm.retrieve_concept(rel.source);
            auto tgt = ltm.retrieve_concept(rel.target);
            std::cout << "  " << (src ? src->label : "?")
                      << " --[" << relation_type_to_string(rel.type)
                      << " w=" << std::fixed << std::setprecision(2) << rel.weight
                      << "]--> " << (tgt ? tgt->label : "?") << "\n";
        }
    }
    if (!any) std::cout << "  (no relations)\n";
}

static void action_review(IngestionPipeline& pipeline) {
    auto& queue = pipeline.get_queue();
    auto pending = queue.get_pending();
    if (pending.empty()) {
        std::cout << "  (no pending proposals)\n";
        return;
    }

    std::cout << "\n  " << pending.size() << " pending proposal(s):\n\n";
    for (auto& p : pending) {
        std::cout << "  Proposal #" << p.id << "\n"
                  << "    Label:      " << p.concept_label << "\n"
                  << "    Definition: " << p.concept_definition << "\n"
                  << "    Trust:      "
                  << TrustTagger::category_to_string(p.trust_assignment.category)
                  << " (" << std::fixed << std::setprecision(2)
                  << p.trust_assignment.trust_value << ")\n";

        if (!p.proposed_relations.empty()) {
            std::cout << "    Relations:\n";
            for (auto& rel : p.proposed_relations) {
                std::cout << "      " << rel.source_label
                          << " --[" << relation_type_to_string(rel.relation_type)
                          << "]--> " << rel.target_label
                          << " (conf=" << std::fixed << std::setprecision(2)
                          << rel.confidence << ")\n";
            }
        }

        std::cout << "    [a]pprove / [r]eject / [s]kip ? ";
        std::string choice;
        std::getline(std::cin, choice);

        if (!choice.empty()) {
            char c = choice[0];
            if (c == 'a' || c == 'A') {
                queue.review(p.id, ReviewDecision::approve("CLI approved"));
                std::cout << "    -> approved\n";
            } else if (c == 'r' || c == 'R') {
                queue.review(p.id, ReviewDecision::reject("CLI rejected"));
                std::cout << "    -> rejected\n";
            } else {
                std::cout << "    -> skipped\n";
            }
        } else {
            std::cout << "    -> skipped\n";
        }
        std::cout << "\n";
    }

    // Commit approved proposals to LTM
    auto r = pipeline.commit_approved();
    std::cout << "  Committed: " << r.concepts_stored << " concept(s), "
              << r.relations_stored << " relation(s)\n";
}

static void action_train(MicroModelRegistry& registry,
                         EmbeddingManager& embeddings,
                         MicroTrainer& trainer,
                         LongTermMemory& ltm) {
    std::string input = read_line("Concept ID (or 'all')");
    if (input.empty()) { std::cout << "  (skipped)\n"; return; }

    if (input == "all") {
        size_t created = registry.ensure_models_for(ltm);
        std::cout << "  Models created: " << created
                  << "  (total: " << registry.size() << ")\n";

        auto stats = trainer.train_all(registry, embeddings, ltm);
        std::cout << "  Models trained:  " << stats.models_trained << "\n"
                  << "  Total samples:   " << stats.total_samples << "\n"
                  << "  Total epochs:    " << stats.total_epochs << "\n"
                  << "  Avg final loss:  " << std::fixed << std::setprecision(6)
                  << stats.avg_final_loss << "\n"
                  << "  Converged:       " << stats.models_converged << "\n";
    } else {
        ConceptId cid = 0;
        try { cid = std::stoull(input); }
        catch (...) {
            std::cout << "  Invalid concept ID\n";
            return;
        }

        if (!ltm.exists(cid)) {
            std::cout << "  Concept " << cid << " not found in LTM\n";
            return;
        }

        if (!registry.has_model(cid)) {
            registry.create_model(cid);
            std::cout << "  Created model for concept " << cid << "\n";
        }

        MicroModel* model = registry.get_model(cid);
        auto result = trainer.train_single(cid, *model, embeddings, ltm);
        std::cout << "  Epochs: " << result.epochs_run
                  << "  Loss: " << std::fixed << std::setprecision(6) << result.final_loss
                  << "  Converged: " << (result.converged ? "yes" : "no") << "\n";
    }
}

static void action_relevance_map(MicroModelRegistry& registry,
                                 EmbeddingManager& embeddings,
                                 LongTermMemory& ltm) {
    std::string cid_str = read_line("Source concept ID");
    if (cid_str.empty()) { std::cout << "  (skipped)\n"; return; }

    ConceptId cid = 0;
    try { cid = std::stoull(cid_str); }
    catch (...) {
        std::cout << "  Invalid concept ID\n";
        return;
    }

    if (!ltm.exists(cid)) {
        std::cout << "  Concept " << cid << " not found\n";
        return;
    }

    std::string rel_str = read_line("Relation type [IS_A]");
    RelationType rel_type = RelationType::IS_A;

    if (!rel_str.empty()) {
        if      (rel_str == "HAS_PROPERTY")    rel_type = RelationType::HAS_PROPERTY;
        else if (rel_str == "CAUSES")          rel_type = RelationType::CAUSES;
        else if (rel_str == "ENABLES")         rel_type = RelationType::ENABLES;
        else if (rel_str == "PART_OF")         rel_type = RelationType::PART_OF;
        else if (rel_str == "SIMILAR_TO")      rel_type = RelationType::SIMILAR_TO;
        else if (rel_str == "CONTRADICTS")     rel_type = RelationType::CONTRADICTS;
        else if (rel_str == "SUPPORTS")        rel_type = RelationType::SUPPORTS;
        else if (rel_str == "TEMPORAL_BEFORE") rel_type = RelationType::TEMPORAL_BEFORE;
        else if (rel_str == "CUSTOM")          rel_type = RelationType::CUSTOM;
        else if (rel_str != "IS_A") {
            std::cout << "  Unknown relation type, using IS_A\n";
        }
    }

    if (!registry.has_model(cid)) {
        std::cout << "  No model for concept " << cid << ". Train first.\n";
        return;
    }

    auto rmap = RelevanceMap::compute(cid, registry, embeddings, ltm,
                                      rel_type, "query");
    auto top = rmap.top_k(10);

    auto src = ltm.retrieve_concept(cid);
    std::cout << "\n  Relevance map for [" << cid << "] "
              << (src ? src->label : "?")
              << " (" << relation_type_to_string(rel_type) << "):\n\n";

    if (top.empty()) {
        std::cout << "  (no results)\n";
    } else {
        for (auto& [tid, score] : top) {
            auto tgt = ltm.retrieve_concept(tid);
            std::cout << "    " << std::fixed << std::setprecision(4) << score
                      << "  [" << tid << "] "
                      << (tgt ? tgt->label : "?") << "\n";
        }
    }
}

static void action_train_all_stats(MicroModelRegistry& registry,
                                   EmbeddingManager& embeddings,
                                   MicroTrainer& trainer,
                                   LongTermMemory& ltm) {
    size_t created = registry.ensure_models_for(ltm);
    std::cout << "  Models created: " << created
              << "  (total: " << registry.size() << ")\n";

    auto stats = trainer.train_all(registry, embeddings, ltm);
    std::cout << "\n  === Training Summary ===\n"
              << "  Models trained:  " << stats.models_trained << "\n"
              << "  Total samples:   " << stats.total_samples << "\n"
              << "  Total epochs:    " << stats.total_epochs << "\n"
              << "  Avg final loss:  " << std::fixed << std::setprecision(6)
              << stats.avg_final_loss << "\n"
              << "  Converged:       " << stats.models_converged
              << " / " << stats.models_trained << "\n";

    // Per-model summary
    auto model_ids = registry.get_model_ids();
    if (!model_ids.empty()) {
        std::cout << "\n  Per-model predictions (self, IS_A context):\n";
        for (auto cid : model_ids) {
            auto* model = registry.get_model(cid);
            if (!model) continue;
            auto cpt = ltm.retrieve_concept(cid);
            auto& e = embeddings.get_relation_embedding(RelationType::IS_A);
            auto& c = embeddings.query_context();
            double pred = model->predict(e, c);
            std::cout << "    [" << cid << "] "
                      << (cpt ? cpt->label : "?")
                      << " -> " << std::fixed << std::setprecision(4)
                      << pred << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    bool review_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--review") == 0) {
            review_mode = true;
        }
    }

    LongTermMemory ltm;
    IngestionPipeline pipeline(ltm);
    MicroModelRegistry registry;
    EmbeddingManager embeddings;
    MicroTrainer trainer;

    std::cout << "=== Brain19 CLI ===\n"
              << "Mode: " << (review_mode ? "review" : "direct") << "\n\n";

    while (true) {
        std::cout << "\n[1] Ingest knowledge (JSON)\n"
                  << "[2] Ingest knowledge (free text)\n"
                  << "[3] Show KG contents\n";
        if (review_mode) {
            std::cout << "[4] Review proposals\n";
        }
        std::cout << "[5] Train micro-models\n"
                  << "[6] Show relevance map\n"
                  << "[7] Train ALL + show stats\n"
                  << "[0] Exit\n"
                  << "> ";

        std::string choice;
        if (!std::getline(std::cin, choice)) break;
        if (choice.empty()) continue;

        switch (choice[0]) {
            case '1': action_ingest_json(pipeline, review_mode); break;
            case '2': action_ingest_text(pipeline, review_mode); break;
            case '3': action_show_kg(ltm); break;
            case '4':
                if (review_mode) action_review(pipeline);
                else std::cout << "  (review mode not enabled, use --review)\n";
                break;
            case '5': action_train(registry, embeddings, trainer, ltm); break;
            case '6': action_relevance_map(registry, embeddings, ltm); break;
            case '7': action_train_all_stats(registry, embeddings, trainer, ltm); break;
            case '0': std::cout << "Bye.\n"; return 0;
            default:  std::cout << "  Unknown option\n"; break;
        }
    }

    return 0;
}
