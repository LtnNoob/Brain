#pragma once

#include "micro_model.hpp"
#include "micro_model_registry.hpp"
#include "embedding_manager.hpp"

#include <string>

namespace brain19 {
namespace persistence {

// =============================================================================
// BINARY PERSISTENCE
// =============================================================================
//
// Format:
//   Header (32 bytes):
//     magic    "BM19"  (4 bytes)
//     version  uint32  (4 bytes) = 1
//     model_count uint64 (8 bytes)
//     context_count uint64 (8 bytes)
//     reserved (8 bytes)
//
//   Per model (3448 bytes each):
//     concept_id  uint64 (8 bytes)
//     params      430 × double (3440 bytes)
//
//   Relation embeddings (800 bytes):
//     10 × Vec10 (10 × 10 × 8 bytes)
//
//   Per context embedding:
//     name_len    uint32 (4 bytes)
//     name        name_len bytes
//     embedding   Vec10 (80 bytes)
//
//   Footer:
//     checksum    uint64 (8 bytes) - XOR of all preceding 8-byte blocks
//

bool save(const std::string& filepath,
          const MicroModelRegistry& registry,
          const EmbeddingManager& embeddings);

bool load(const std::string& filepath,
          MicroModelRegistry& registry,
          EmbeddingManager& embeddings);

bool validate(const std::string& filepath);

} // namespace persistence
} // namespace brain19
