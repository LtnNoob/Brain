#pragma once

#include "language_config.hpp"
#include "bpe_tokenizer.hpp"
#include "kan_encoder.hpp"
#include "kan_decoder.hpp"
#include "semantic_scorer.hpp"
#include "fusion_layer.hpp"
#include "dimensional_context.hpp"
#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../cursor/template_engine.hpp"
#include "../reasoning/concept_reasoner.hpp"
#include "../graph_net/graph_reasoner.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// LANGUAGE RESULT — output of KANLanguageEngine::generate()
// =============================================================================

struct LanguageResult {
    std::string text;                              // Generated text
    std::vector<ConceptId> activated_concepts;     // Active concepts
    std::vector<ConceptId> causal_chain;           // Extracted causal chain
    double confidence;                             // Overall confidence
    bool used_template;                            // True if decoder fell back to templates
    size_t template_type;                          // Template type index
    size_t tokens_generated;                       // Number of tokens generated

    // Rich reasoning chain (from ConceptReasoner, if available)
    ReasoningChain reasoning_chain;                // Full chain with coherence scores + chain state

    // Graph reasoning chain (from GraphReasoner, if enabled)
    GraphChain graph_chain;                        // Full audit trail with activation vectors
};

// =============================================================================
// KAN LANGUAGE ENGINE — Main orchestrator
// =============================================================================
//
// Pipeline: Query → Tokenize → KAN-Encode → Seed Selection → Reasoning →
//           Semantic Scoring → Gated Fusion → KAN-Decode → Text
//
// Template fallback when decoder confidence < threshold.
//

class KANLanguageEngine {
public:
    KANLanguageEngine(
        const LanguageConfig& config,
        LongTermMemory& ltm,
        ConceptModelRegistry& registry,
        EmbeddingManager& embeddings
    );

    // Main function: query → answer
    LanguageResult generate(const std::string& query, size_t max_tokens = 30) const;

    // Concept-based generation (uses concept prediction instead of token prediction)
    LanguageResult generate_concept_response(const std::string& query) const;

    // Initialize tokenizer from LTM
    void initialize();

    // Is engine ready for generation?
    bool is_ready() const { return initialized_; }

    // Rebuild dimensional context from current LTM state.
    // Call this after LTM has been populated (e.g., after foundation seeding).
    void rebuild_dimensional_context();

    // Individual pipeline phases (exposed for testing)
    std::vector<double> encode(const std::string& text) const;
    std::vector<ConceptId> find_seeds(const std::string& text) const;

    // Access components
    const BPETokenizer& tokenizer() const { return tokenizer_; }
    KANEncoder& encoder() { return encoder_; }
    KANDecoder& decoder() { return decoder_; }
    SemanticScorer& scorer() { return scorer_; }
    FusionLayer& fusion() { return fusion_; }
    EmbeddingManager& embeddings() { return embeddings_; }
    const DimensionalContext& dim_context() const { return dim_context_; }
    ConceptReasoner* reasoner() { return reasoner_.get(); }
    const ConceptReasoner* reasoner() const { return reasoner_.get(); }
    GraphReasoner* graph_reasoner() { return graph_reasoner_.get(); }
    const GraphReasoner* graph_reasoner() const { return graph_reasoner_.get(); }

    // Persistence
    void save(const std::string& dir) const;
    bool load(const std::string& dir);

private:
    LanguageConfig config_;
    bool initialized_ = false;

    // Components
    BPETokenizer tokenizer_;
    KANEncoder encoder_;
    KANDecoder decoder_;
    SemanticScorer scorer_;
    FusionLayer fusion_;
    DimensionalContext dim_context_;

    // References to Brain19 core (not owned)
    LongTermMemory& ltm_;
    ConceptModelRegistry& registry_;
    EmbeddingManager& embeddings_;

    // Composition-guided reasoner (initialized in initialize())
    std::unique_ptr<ConceptReasoner> reasoner_;
    mutable ReasoningChain last_reasoning_chain_;  // cached from extract_causal_chain()

    // Graph-based reasoner (alternative, config-switched)
    std::unique_ptr<GraphReasoner> graph_reasoner_;
    mutable GraphChain last_graph_chain_;          // cached from extract_causal_chain()
    bool use_graph_reasoner_ = false;              // Config switch

    // Seed selection: find LTM concepts matching query text
    std::vector<ConceptId> label_search(const std::string& text) const;

    // Build concept activations from ConceptModels
    std::unordered_map<ConceptId, std::vector<double>> build_activations(
        const std::vector<ConceptId>& seeds,
        const std::vector<double>& query_embedding
    ) const;

    // Extract causal chain from relations
    std::vector<ConceptId> extract_causal_chain(
        const std::vector<ConceptId>& seeds
    ) const;

    // Build relation embeddings for causal pairs
    std::unordered_map<std::string, std::vector<double>> build_relation_embeddings(
        const std::vector<std::pair<ConceptId, ConceptId>>& pairs
    ) const;

    // Fluent text generation from reasoning chain + supplementary relations
    std::string generate_fluent_text(
        const std::string& query,
        const std::vector<ConceptId>& seeds,
        const std::vector<ConceptId>& ordered_concepts
    ) const;

    // Query classification helpers
    static bool is_causal_query(const std::string& query);
    static bool is_definitional_query(const std::string& query);
};

} // namespace brain19
