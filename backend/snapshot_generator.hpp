#pragma once

#include "common/types.hpp"

#include <string>
#include <sstream>
#include <memory>

namespace brain19 {


// Forward declarations
class BrainController;
class CuriosityEngine;
class LongTermMemory;  // Added for epistemic metadata access
struct SystemObservation;

// SnapshotGenerator: Creates JSON snapshots for visualization
// 
// EPISTEMIC ENFORCEMENT:
// - MUST expose epistemic metadata in snapshot
// - MUST verbalize EpistemicType and EpistemicStatus
// - MUST include trust values
// - NO epistemically neutral representations
class SnapshotGenerator {
public:
    SnapshotGenerator();
    ~SnapshotGenerator();
    
    // Generate complete JSON snapshot
    // REQUIRES: LTM reference for epistemic metadata
    std::string generate_json_snapshot(
        const BrainController* brain,
        const LongTermMemory* ltm,        // REQUIRED for epistemic data
        const CuriosityEngine* curiosity,
        ContextId context_id
    ) const;
    
private:
    std::string escape_json_string(const std::string& str) const;
};

} // namespace brain19
