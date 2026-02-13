#pragma once

#include "micro_model.hpp"
#include "micro_model_registry.hpp"
#include "embedding_manager.hpp"

#include <string>

namespace brain19 {
namespace persistence {

// =============================================================================
// BINARY PERSISTENCE — v3 (FlexEmbedding)
// =============================================================================
//
// Format v3:
//   Header (32 bytes):
//     magic    "BM19"  (4 bytes)
//     version  uint32  (4 bytes) = 3
//     model_count uint64 (8 bytes)
//     context_count uint64 (8 bytes)
//     reserved (8 bytes)
//
//   Per model (7528 bytes each):
//     concept_id  uint64 (8 bytes)
//     params      940 x double (7520 bytes)
//
//   Relation embeddings:
//     rel_emb_count uint32 (4 bytes)
//     Per relation:
//       type_id     uint16 (2 bytes)
//       core        16 x double (128 bytes)
//       detail_dim  uint16 (2 bytes)
//       detail      detail_dim x double (variable)
//
//   Concept embeddings:
//     concept_emb_count uint32 (4 bytes)
//     Per concept:
//       concept_id  uint64 (8 bytes)
//       core        16 x double (128 bytes)
//       detail_dim  uint16 (2 bytes)
//       detail      detail_dim x double (variable)
//
//   Context embeddings:
//     Per context:
//       name_len    uint32 (4 bytes)
//       name        name_len bytes
//       core        16 x double (128 bytes)
//       detail_dim  uint16 (2 bytes)
//       detail      detail_dim x double (variable)
//
//   Footer:
//     checksum    uint64 (8 bytes) - XOR of all preceding 8-byte blocks
//
// Backward compatible: reads v1 (10D fixed) and v2 (10D counted) with migration to 16D core.
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
