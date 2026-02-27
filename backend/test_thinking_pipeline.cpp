// Standalone test: ThinkingPipeline on full foundation graph
// Build manually (without language_training.o / system_orchestrator.o)

#include "core/thinking_pipeline.hpp"
#include "ltm/long_term_memory.hpp"
#include "ltm/relation.hpp"
#include "memory/stm.hpp"
#include "memory/brain_controller.hpp"
#include "cognitive/cognitive_dynamics.hpp"
#include "curiosity/curiosity_engine.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "micromodel/embedding_manager.hpp"
#include "understanding/understanding_layer.hpp"
#include "hybrid/kan_validator.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "cursor/template_engine.hpp"
#include "reasoning/concept_reasoner.hpp"

#include <iostream>
#include <chrono>
#include <string>

using namespace brain19;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << ts << "] " << msg << "\n";
}

int main() {
    log("=== ThinkingPipeline Full-Graph Test ===");
    log("");

    // ── Load KB ──
    log("[1/5] Loading foundation KB...");
    LongTermMemory ltm;
    bool loaded = false;
    for (const auto& path : {"../data/foundation_full.json", "data/foundation_full.json",
                              "../data/foundation.json", "data/foundation.json"}) {
        if (FoundationConcepts::seed_from_file(ltm, path, true)) {
            log("  Loaded from: " + std::string(path));
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        log("  FALLBACK: using hardcoded seeds");
        FoundationConcepts::seed_all(ltm);
    }
    log("  Concepts: " + std::to_string(ltm.get_all_concept_ids().size()));
    log("  Relations: " + std::to_string(ltm.total_relation_count()));

    // ── Property Inheritance ──
    log("");
    log("[2/5] PropertyInheritance...");
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        auto r = prop.propagate(cfg);
        log("  Inherited: " + std::to_string(r.properties_inherited));
        log("  Relations now: " + std::to_string(ltm.total_relation_count()));
    }

    // ── Train Embeddings + ConceptModels ──
    log("");
    log("[3/5] Training embeddings & ConceptModels...");
    EmbeddingManager embeddings;
    ConceptModelRegistry registry;
    {
        auto t0 = std::chrono::steady_clock::now();
        embeddings.train_embeddings(ltm, 0.05, 10);
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto stats = trainer.train_all(registry, embeddings, ltm);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        log("  Models: " + std::to_string(stats.models_trained)
            + " (" + std::to_string(stats.models_converged) + " converged)");
        log("  Time: " + std::to_string(ms) + "ms");
    }

    // ── Setup Pipeline Subsystems ──
    log("");
    log("[4/5] Setting up pipeline subsystems...");
    BrainController brain;
    brain.initialize();
    CognitiveDynamics cognitive;
    CuriosityEngine curiosity;
    UnderstandingLayer understanding;
    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

    ContextId ctx = brain.create_context();

    // ── Run ThinkingPipeline ──
    log("");
    log("[5/5] Running ThinkingPipeline on queries...");
    log("");

    ThinkingPipeline::Config pcfg;
    pcfg.enable_focus_cursor = true;
    pcfg.enable_understanding = true;
    pcfg.enable_kan_validation = false;  // no KanValidator setup
    pcfg.enable_topology_a = false;
    pcfg.enable_topology_c = false;
    pcfg.enable_curiosity = true;

    ThinkingPipeline pipeline(pcfg);
    TemplateEngine tpl(ltm);

    for (const auto& query : {"Photosynthesis", "Gravity", "Water", "Evolution",
                                "Electricity", "DNA", "Mathematics", "Oxygen"}) {
        auto seeds = ltm.find_by_label(query);
        if (seeds.empty()) { log("  " + std::string(query) + ": not found"); continue; }

        // Fresh context per query
        ContextId qctx = brain.create_context();
        ShortTermMemory* stm = brain.get_stm_mutable();

        auto t0 = std::chrono::steady_clock::now();
        auto result = pipeline.execute(
            seeds, qctx, ltm, *stm, brain,
            cognitive, curiosity, registry, embeddings,
            &understanding, nullptr  // no KanValidator
        );
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        std::string line = "  [" + std::string(query) + "] " + std::to_string(ms) + "ms";
        line += " | steps=" + std::to_string(result.steps_completed) + "/10";
        line += " | activated=" + std::to_string(result.activated_concepts.size());
        line += " | salient=" + std::to_string(result.top_salient.size());
        line += " | paths=" + std::to_string(result.best_paths.size());
        line += " | curiosity=" + std::to_string(result.curiosity_triggers.size());
        log(line);

        // Top salient concepts
        if (!result.top_salient.empty()) {
            std::string sal = "    Salient: ";
            for (size_t i = 0; i < std::min(result.top_salient.size(), size_t(5)); ++i) {
                auto c = ltm.retrieve_concept(result.top_salient[i].concept_id);
                if (c) sal += c->label + "(" + std::to_string(result.top_salient[i].salience).substr(0,4) + ") ";
            }
            log(sal);
        }

        // FocusCursor result
        if (result.cursor_result.has_value() && !result.cursor_result->concept_sequence.empty()) {
            std::string chain = "    Cursor chain (" + std::to_string(result.cursor_result->concept_sequence.size())
                + " steps, score=" + std::to_string(result.cursor_result->chain_score).substr(0,4) + "): ";
            for (size_t i = 0; i < result.cursor_result->concept_sequence.size(); ++i) {
                auto c = ltm.retrieve_concept(result.cursor_result->concept_sequence[i]);
                if (c) chain += c->label;
                if (i + 1 < result.cursor_result->concept_sequence.size()) chain += " -> ";
            }
            log(chain);

            // Template output
            auto tmpl = tpl.generate(*result.cursor_result);
            if (!tmpl.text.empty()) {
                log("    Template: \"" + tmpl.text + "\"");
            }
        } else {
            log("    Cursor: no chain");
        }

        // Understanding results
        auto& u = result.understanding;
        if (u.total_proposals_generated > 0) {
            log("    Understanding: " + std::to_string(u.meaning_proposals.size()) + " meanings, "
                + std::to_string(u.hypothesis_proposals.size()) + " hypotheses, "
                + std::to_string(u.analogy_proposals.size()) + " analogies, "
                + std::to_string(u.contradiction_proposals.size()) + " contradictions");
        }

        // Thought paths
        if (!result.best_paths.empty()) {
            std::string paths = "    Best path: ";
            const auto& bp = result.best_paths[0];
            for (size_t i = 0; i < bp.nodes.size(); ++i) {
                auto c = ltm.retrieve_concept(bp.nodes[i].concept_id);
                if (c) paths += c->label;
                if (i + 1 < bp.nodes.size()) paths += " -> ";
            }
            paths += " (score=" + std::to_string(bp.total_score).substr(0,4) + ")";
            log(paths);
        }

        log("");
        brain.destroy_context(qctx);
    }

    // ── Comparison: ConceptReasoner ──
    log("--- ConceptReasoner comparison ---");
    log("");
    {
        ReasonerConfig rcfg;
        rcfg.max_steps = 10;
        ConceptReasoner reasoner(ltm, registry, embeddings, rcfg);

        for (const auto& query : {"Photosynthesis", "Gravity", "Water", "Evolution"}) {
            auto seeds = ltm.find_by_label(query);
            if (seeds.empty()) continue;

            auto chain = reasoner.reason_from(seeds);
            if (chain.empty()) continue;

            std::string line = "  [reason] " + std::string(query)
                + " (" + std::to_string(chain.steps.size()) + " steps, avg_conf="
                + std::to_string(chain.avg_confidence).substr(0,4) + "): ";
            for (size_t i = 0; i < chain.steps.size(); ++i) {
                const auto& s = chain.steps[i];
                auto cinfo = ltm.retrieve_concept(s.concept_id);
                std::string label = cinfo ? cinfo->label : ("#" + std::to_string(s.concept_id));
                if (i > 0) {
                    std::string dir = s.is_outgoing ? "--" : "<-";
                    std::string conf = std::to_string(s.confidence).substr(0,4);
                    line += " " + dir + relation_type_to_string(s.relation_type)
                          + "(" + conf + ")" + dir + "> ";
                }
                line += label;
            }
            log(line);
        }
    }

    brain.destroy_context(ctx);
    brain.shutdown();
    log("");
    log("=== Done ===");
    return 0;
}
