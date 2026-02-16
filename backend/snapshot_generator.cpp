#include "snapshot_generator.hpp"
#include "memory/brain_controller.hpp"
#include "memory/stm.hpp"
#include "ltm/long_term_memory.hpp"
#include "epistemic/epistemic_metadata.hpp"
#include "cognitive/cognitive_dynamics.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_model.hpp"
#include <sstream>
#include <iomanip>

namespace brain19 {

SnapshotGenerator::SnapshotGenerator() = default;

SnapshotGenerator::~SnapshotGenerator() = default;

std::string SnapshotGenerator::generate_json_snapshot(
    const BrainController* brain,
    const LongTermMemory* ltm,
    const CuriosityEngine* /*curiosity*/,
    ContextId context_id,
    const CognitiveDynamics* cognitive,
    const ConceptModelRegistry* micro_models
) const {
    if (!brain) {
        return "{}";
    }
    
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    
    json << "{\n";
    
    // STM Layer
    json << "  \"stm\": {\n";
    json << "    \"context_id\": " << context_id << ",\n";
    
    // Active concepts
    json << "    \"active_concepts\": [\n";
    auto active_concepts = brain->query_active_concepts(context_id, 0.0);
    for (size_t i = 0; i < active_concepts.size(); i++) {
        double activation = brain->query_concept_activation(context_id, active_concepts[i]);
        json << "      {\"concept_id\": " << active_concepts[i] 
             << ", \"activation\": " << activation << "}";
        if (i < active_concepts.size() - 1) json << ",";
        json << "\n";
    }
    json << "    ],\n";
    
    // Active relations from STM
    json << "    \"active_relations\": [\n";
    if (brain->get_stm()) {
        auto active_relations = brain->get_stm()->get_active_relations(context_id, 0.01);
        for (size_t i = 0; i < active_relations.size(); i++) {
            json << "      {\"source\": " << active_relations[i].source
                 << ", \"target\": " << active_relations[i].target
                 << ", \"activation\": " << active_relations[i].activation << "}";
            if (i < active_relations.size() - 1) json << ",";
            json << "\n";
        }
    }
    json << "    ]\n";
    
    json << "  },\n";
    
    // Concepts layer WITH EPISTEMIC METADATA
    // CRITICAL: Every concept MUST have epistemic data verbalized
    json << "  \"concepts\": [\n";
    for (size_t i = 0; i < active_concepts.size(); i++) {
        json << "    {";
        json << "\"id\": " << active_concepts[i] << ", ";
        
        // Query LTM for epistemic metadata
        if (ltm) {
            auto concept_info = ltm->retrieve_concept(active_concepts[i]);
            if (concept_info.has_value()) {
                // EPISTEMIC ENFORCEMENT: Always verbalize epistemic metadata
                json << "\"label\": \"" << escape_json_string(concept_info->label) << "\", ";
                json << "\"epistemic_type\": \"" << epistemic_type_to_string(concept_info->epistemic.type) << "\", ";
                json << "\"epistemic_status\": \"" << epistemic_status_to_string(concept_info->epistemic.status) << "\", ";
                json << "\"trust\": " << concept_info->epistemic.trust;
                
                // Special marking for INVALIDATED knowledge
                if (concept_info->epistemic.is_invalidated()) {
                    json << ", \"invalidated\": true";
                }
            } else {
                // Concept not in LTM (STM-only)
                // ENFORCEMENT: Still must verbalize epistemic uncertainty
                json << "\"label\": \"Concept_" << active_concepts[i] << "\", ";
                json << "\"epistemic_type\": \"HYPOTHESIS\", ";
                json << "\"epistemic_status\": \"CONTEXTUAL\", ";
                json << "\"trust\": 0.5, ";
                json << "\"note\": \"STM-only, not in LTM\"";
            }
        } else {
            // No LTM available
            // ENFORCEMENT: Must still verbalize epistemic status
            json << "\"label\": \"Concept_" << active_concepts[i] << "\", ";
            json << "\"epistemic_type\": \"HYPOTHESIS\", ";
            json << "\"epistemic_status\": \"CONTEXTUAL\", ";
            json << "\"trust\": 0.5, ";
            json << "\"note\": \"LTM not available\"";
        }
        
        json << "}";
        if (i < active_concepts.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";
    
    // Curiosity triggers
    json << "  \"curiosity_triggers\": [],\n";
    
    // Cognitive Dynamics: Focus-Sets and Tick
    json << "  \"cognitive_dynamics\": {\n";
    if (cognitive) {
        auto focus_set = cognitive->get_focus_set(context_id);
        // Note: current_tick_ is private in CognitiveDynamics, not exposed via public API
        json << "    \"focus_set\": [\n";
        for (size_t i = 0; i < focus_set.size(); i++) {
            json << "      {\"concept_id\": " << focus_set[i].concept_id
                 << ", \"focus_score\": " << focus_set[i].focus_score
                 << ", \"last_accessed_tick\": " << focus_set[i].last_accessed_tick << "}";
            if (i < focus_set.size() - 1) json << ",";
            json << "\n";
        }
        json << "    ],\n";
        auto stats = cognitive->get_stats();
        json << "    \"stats\": {"
             << "\"total_spreads\": " << stats.total_spreads.load()
             << ", \"total_salience_computations\": " << stats.total_salience_computations.load()
             << ", \"total_focus_updates\": " << stats.total_focus_updates.load()
             << ", \"total_path_searches\": " << stats.total_path_searches.load()
             << "}\n";
    } else {
        json << "    \"focus_set\": [],\n";
        json << "    \"stats\": null\n";
    }
    json << "  },\n";
    
    // MicroModel Metrics
    json << "  \"micromodel_metrics\": {\n";
    if (micro_models) {
        size_t model_count = micro_models->size();
        json << "    \"trained_model_count\": " << model_count << ",\n";
        
        // Compute average loss across all models
        double total_loss = 0.0;
        size_t loss_count = 0;
        auto model_ids = micro_models->get_model_ids();
        for (auto cid : model_ids) {
            const auto* model = micro_models->get_model(cid);
            if (model) {
                // Access training state via to_flat - extract last_loss
                std::array<double, CM_FLAT_SIZE> flat;
                model->to_flat(flat);
                // last_loss at: 256(W) + 16(b) + 16(e) + 16(c) + 256(dW_mom) + 16(db_mom)
                //             + 256(dW_var) + 16(db_var) + 16(e_grad) + 16(c_grad) + 1(timestep) = 881
                double last_loss = flat[881]; // TrainingState::last_loss
                total_loss += last_loss;
                loss_count++;
            }
        }
        double avg_loss = (loss_count > 0) ? (total_loss / static_cast<double>(loss_count)) : 0.0;
        json << "    \"average_loss\": " << avg_loss << ",\n";
        json << "    \"model_ids\": [";
        for (size_t i = 0; i < model_ids.size(); i++) {
            json << model_ids[i];
            if (i < model_ids.size() - 1) json << ", ";
        }
        json << "]\n";
    } else {
        json << "    \"trained_model_count\": 0,\n";
        json << "    \"average_loss\": 0.0,\n";
        json << "    \"model_ids\": []\n";
    }
    json << "  }\n";
    
    json << "}\n";
    
    return json.str();
}

std::string SnapshotGenerator::escape_json_string(const std::string& str) const {
    std::ostringstream escaped;
    for (char c : str) {
        switch (c) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << c; break;
        }
    }
    return escaped.str();
}

} // namespace brain19
