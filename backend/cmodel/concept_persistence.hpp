#pragma once

#include "concept_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"

#include <string>

namespace brain19 {
namespace persistence {

// =============================================================================
// BINARY PERSISTENCE — v5 (ConceptModel with MultiHeadBilinear + FlexKAN)
// =============================================================================
//
// Format v5:
//   Header (32 bytes): magic "BM19", version=5, model_count, context_count, reserved
//   Per model (15208 bytes): concept_id(8) + params(1900 x 8 = 15200)
//   Relation embeddings: same as v3/v4
//   Concept embeddings: same as v3/v4
//   Context embeddings: same as v3/v4
//   Footer: XOR checksum
//
// Backward compatible: reads v3 (940-double) and v4 (1300-double) models,
// migrating to 1900-double layout with identity-init new params.
//

bool save_v4(const std::string& filepath,
             const ConceptModelRegistry& registry,
             const EmbeddingManager& embeddings);

bool load_v4(const std::string& filepath,
             ConceptModelRegistry& registry,
             EmbeddingManager& embeddings);

bool validate_v4(const std::string& filepath);

} // namespace persistence
} // namespace brain19
