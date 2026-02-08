#include "snapshot_generator.hpp"
#include "memory/brain_controller.hpp"
#include "memory/stm.hpp"
#include "ltm/long_term_memory.hpp"
#include "epistemic/epistemic_metadata.hpp"
#include "curiosity/curiosity_engine.hpp"
#include <sstream>
#include <iomanip>

namespace brain19 {

SnapshotGenerator::SnapshotGenerator() {
}

SnapshotGenerator::~SnapshotGenerator() {
}

std::string SnapshotGenerator::generate_json_snapshot(
    const BrainController* brain,
    const LongTermMemory* ltm,
    const CuriosityEngine* curiosity,
    ContextId context_id
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
    
    // Active relations (simplified - would need actual relation data)
    json << "    \"active_relations\": [\n";
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
                json << "\"epistemic_type\": \"" << to_string(concept_info->epistemic.type) << "\", ";
                json << "\"epistemic_status\": \"" << to_string(concept_info->epistemic.status) << "\", ";
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
    json << "  \"curiosity_triggers\": [\n";
    if (curiosity) {
        // Would get actual observations from STM
        std::vector<SystemObservation> observations;
        // auto triggers = curiosity->observe_and_generate_triggers(observations);
        // Simplified for now
    }
    json << "  ]\n";
    
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
