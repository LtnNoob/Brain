#include "ingestion_pipeline.hpp"
#include <algorithm>

namespace brain19 {

IngestionPipeline::IngestionPipeline(LongTermMemory& ltm)
    : ltm_(ltm)
    , default_trust_(TrustCategory::HYPOTHESES)
{
}

// =============================================================================
// STRUCTURED INPUT
// =============================================================================

IngestionResult IngestionPipeline::ingest_json(
    const std::string& json_str, bool auto_approve)
{
    IngestionResult result;

    // Parse JSON
    auto parse_result = ingestor_.parse_json(json_str);
    if (!parse_result.success) {
        result.error_message = "JSON parse error: " + parse_result.error_message;
        return result;
    }

    // Convert to proposals
    auto proposals = ingestor_.to_proposals(parse_result.data, tagger_);
    result.proposals_created = proposals.size();
    result.entities_extracted = parse_result.concepts_parsed;
    result.relations_extracted = parse_result.relations_parsed;

    // Enqueue
    for (auto& p : proposals) {
        queue_.enqueue(std::move(p));
    }

    // Auto-approve if requested
    if (auto_approve) {
        result.proposals_approved = queue_.auto_approve_all();
        auto commit_result = commit_approved();
        result.concepts_stored = commit_result.concepts_stored;
        result.relations_stored = commit_result.relations_stored;
        result.stored_concept_ids = commit_result.stored_concept_ids;
        result.stored_relation_ids = commit_result.stored_relation_ids;
    }

    result.success = true;
    return result;
}

IngestionResult IngestionPipeline::ingest_csv(
    const std::string& concepts_csv,
    const std::string& relations_csv,
    bool auto_approve)
{
    IngestionResult result;

    // Parse concepts CSV
    auto parse_result = ingestor_.parse_csv_concepts(concepts_csv);
    if (!parse_result.success) {
        result.error_message = "CSV parse error: " + parse_result.error_message;
        return result;
    }

    // Parse relations CSV if provided
    if (!relations_csv.empty()) {
        auto rel_result = ingestor_.parse_csv_relations(relations_csv, parse_result.data);
        if (!rel_result.success) {
            result.error_message = "Relations CSV parse error: " + rel_result.error_message;
            return result;
        }
    }

    // Convert to proposals
    auto proposals = ingestor_.to_proposals(parse_result.data, tagger_);
    result.proposals_created = proposals.size();
    result.entities_extracted = parse_result.concepts_parsed;
    result.relations_extracted = parse_result.relations_parsed;

    // Enqueue
    for (auto& p : proposals) {
        queue_.enqueue(std::move(p));
    }

    if (auto_approve) {
        result.proposals_approved = queue_.auto_approve_all();
        auto commit_result = commit_approved();
        result.concepts_stored = commit_result.concepts_stored;
        result.relations_stored = commit_result.relations_stored;
        result.stored_concept_ids = commit_result.stored_concept_ids;
        result.stored_relation_ids = commit_result.stored_relation_ids;
    }

    result.success = true;
    return result;
}

// =============================================================================
// PLAIN TEXT INPUT
// =============================================================================

IngestionResult IngestionPipeline::ingest_text(
    const std::string& text,
    const std::string& source_ref,
    bool auto_approve)
{
    IngestionResult result;

    if (text.empty()) {
        result.error_message = "Empty text input";
        return result;
    }

    // Process through NLP pipeline
    auto proposals = process_text_pipeline(text, source_ref);
    result.proposals_created = proposals.size();

    // Count chunks
    auto chunks = chunker_.chunk_text(text);
    result.chunks_created = chunks.size();

    // Count entities and relations from proposals
    for (const auto& p : proposals) {
        result.entities_extracted++;
        result.relations_extracted += p.proposed_relations.size();
    }

    // Enqueue
    for (auto& p : proposals) {
        queue_.enqueue(std::move(p));
    }

    if (auto_approve) {
        result.proposals_approved = queue_.auto_approve_all();
        auto commit_result = commit_approved();
        result.concepts_stored = commit_result.concepts_stored;
        result.relations_stored = commit_result.relations_stored;
        result.stored_concept_ids = commit_result.stored_concept_ids;
        result.stored_relation_ids = commit_result.stored_relation_ids;
    }

    result.success = true;
    return result;
}

std::vector<IngestProposal> IngestionPipeline::process_text_pipeline(
    const std::string& text,
    const std::string& source_ref)
{
    std::vector<IngestProposal> proposals;

    // Step 1: Chunk text
    auto chunks = chunker_.chunk_text(text);
    if (chunks.empty()) {
        // Text too short for chunking, treat as single chunk
        chunks.emplace_back(text, 0, text.size(), 0);
    }

    // Step 2: Extract entities from all chunks
    auto entities = entity_extractor_.extract_from_chunks(chunks);

    // Step 3: Extract relations using known entities
    auto relations = relation_extractor_.extract_relations(text, entities);

    // Step 4: Create proposals for each entity
    for (const auto& entity : entities) {
        IngestProposal proposal;
        proposal.concept_label = entity.label;
        proposal.source_text = entity.context_snippet;
        proposal.source_reference = source_ref;

        // Build definition from context
        if (entity.is_defined) {
            proposal.concept_definition = entity.context_snippet;
        } else {
            proposal.concept_definition = "Extracted from: \"" + entity.context_snippet + "\"";
        }

        // Assign trust based on text analysis
        auto trust = tagger_.suggest_from_text(entity.context_snippet);
        // But also consider the default trust category
        if (trust.category == TrustCategory::HYPOTHESES &&
            default_trust_ != TrustCategory::HYPOTHESES) {
            trust = tagger_.assign_trust(default_trust_);
        }
        proposal.trust_assignment = trust;

        // Attach relations that involve this entity
        for (const auto& rel : relations) {
            if (rel.source_label == entity.label || rel.target_label == entity.label) {
                proposal.proposed_relations.push_back(rel);
            }
        }

        proposals.push_back(std::move(proposal));
    }

    return proposals;
}

// =============================================================================
// COMMIT TO LTM
// =============================================================================

IngestionResult IngestionPipeline::commit_approved() {
    IngestionResult result;
    result.success = true;

    auto approved = queue_.pop_approved();
    if (approved.empty()) {
        return result;
    }

    // Label → ConceptId map for resolving relations
    std::map<std::string, ConceptId> label_map;

    // First pass: check existing concepts in LTM
    auto all_ids = ltm_.get_all_concept_ids();
    for (ConceptId id : all_ids) {
        auto cinfo = ltm_.retrieve_concept(id);
        if (cinfo.has_value()) {
            label_map[cinfo->label] = id;
        }
    }

    // Second pass: store new concepts
    for (const auto& proposal : approved) {
        if (proposal.concept_label.empty()) continue;

        // Check if concept already exists
        auto it = label_map.find(proposal.concept_label);
        if (it != label_map.end()) {
            // Concept already in LTM, skip (don't overwrite)
            result.stored_concept_ids.push_back(it->second);
            continue;
        }

        auto commit = commit_proposal(proposal);
        if (commit.success) {
            result.concepts_stored++;
            result.stored_concept_ids.push_back(commit.concept_id);
            label_map[proposal.concept_label] = commit.concept_id;
        }
    }

    // Third pass: store relations (now that all concepts have IDs)
    for (const auto& proposal : approved) {
        for (const auto& rel : proposal.proposed_relations) {
            auto src_it = label_map.find(rel.source_label);
            auto tgt_it = label_map.find(rel.target_label);

            if (src_it != label_map.end() && tgt_it != label_map.end()) {
                // Check if relation already exists
                auto existing = ltm_.get_relations_between(src_it->second, tgt_it->second);
                bool already_exists = false;
                for (const auto& ex : existing) {
                    if (ex.type == rel.relation_type) {
                        already_exists = true;
                        break;
                    }
                }

                if (!already_exists) {
                    RelationId rid = ltm_.add_relation(
                        src_it->second, tgt_it->second,
                        rel.relation_type, rel.confidence);
                    if (rid != 0) {
                        result.relations_stored++;
                        result.stored_relation_ids.push_back(rid);
                    }
                }
            }
        }
    }

    result.proposals_approved = approved.size();
    return result;
}

IngestionPipeline::CommitResult IngestionPipeline::commit_proposal(
    const IngestProposal& proposal)
{
    CommitResult result;
    result.success = false;

    if (proposal.concept_label.empty()) {
        return result;
    }

    // Create EpistemicMetadata from trust assignment
    // This is the critical bridge to the existing epistemic system
    EpistemicMetadata metadata = proposal.trust_assignment.to_epistemic_metadata();

    // Store concept in LTM with full epistemic metadata
    result.concept_id = ltm_.store_concept(
        proposal.concept_label,
        proposal.concept_definition,
        metadata
    );

    result.success = (result.concept_id != 0);
    return result;
}

ConceptId IngestionPipeline::resolve_concept_id(
    const std::string& label,
    std::map<std::string, ConceptId>& label_map)
{
    auto it = label_map.find(label);
    if (it != label_map.end()) {
        return it->second;
    }
    return 0; // Not found
}

// =============================================================================
// CONFIGURATION
// =============================================================================

void IngestionPipeline::set_chunker_config(const TextChunker::Config& config) {
    chunker_ = TextChunker(config);
}

void IngestionPipeline::set_entity_config(const EntityExtractor::Config& config) {
    entity_extractor_ = EntityExtractor(config);
}

void IngestionPipeline::set_relation_config(const RelationExtractor::Config& config) {
    relation_extractor_ = RelationExtractor(config);
}

void IngestionPipeline::set_default_trust(TrustCategory category) {
    default_trust_ = category;
}

} // namespace brain19
