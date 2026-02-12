# Brain19 Technical Documentation Task

## Goal
Generate a COMPLETE HTML technical documentation for the Brain19 C++ backend.
Output: `/home/hirschpekf/brain19/docs/technical/index.html`

## CRITICAL RULES
- Process module-by-module in dependency order (bottom-up)
- Document EVERY class, EVERY struct, EVERY function, EVERY member variable
- Do NOT summarize — explain internal logic
- Do NOT infer missing functionality — if unclear, state "Insufficient information"
- Include: responsibility, data flow, invariants, cross-module dependencies, weak points

## Per-Class Documentation Must Include:
1. **Purpose** — what it does and why it exists
2. **Architecture Contract** — what it MAY and MAY NOT do
3. **All Member Variables** — name, type, purpose, constraints
4. **All Methods** — signature, parameters, return value, internal logic step-by-step
5. **Invariants** — what must always be true
6. **Thread Safety** — is it thread-safe? How?
7. **Dependencies** — what it uses
8. **Weak Points** — potential issues

## Module Order (dependency order, bottom-up):
1. common/types.hpp
2. epistemic/epistemic_metadata.hpp  
3. memory/ (activation_level, active_relation, stm_entry, stm, brain_controller)
4. ltm/ (relation, long_term_memory)
5. micromodel/ (micro_model, micro_model_registry, embedding_manager, micro_trainer, relevance_map, persistence)
6. kan/ (kan_node, kan_layer, kan_module, function_hypothesis)
7. adapter/kan_adapter
8. cognitive/ (cognitive_config, cognitive_dynamics)
9. curiosity/ (curiosity_trigger, curiosity_engine)
10. understanding/ (understanding_proposals, mini_llm, mini_llm_factory, understanding_layer)
11. hybrid/ (hypothesis_translator, epistemic_bridge, kan_validator, domain_manager, refinement_loop)
12. ingestor/ (text_chunker, entity_extractor, relation_extractor, trust_tagger, proposal_queue, knowledge_ingestor, ingestion_pipeline)
13. importers/ (knowledge_proposal, http_client, wikipedia_importer, scholar_importer)
14. bootstrap/ (foundation_concepts, context_accumulator, bootstrap_interface)
15. llm/ (chat_interface)
16. persistent/ (persistent_records, persistent_store, string_pool, wal, persistent_ltm, stm_snapshot, checkpoint_manager, checkpoint_restore)
17. concurrent/ (shared_ltm, shared_stm, shared_registry, shared_embeddings, lock_hierarchy, deadlock_detector)
18. streams/ (stream_config, lock_free_queue, think_stream, stream_orchestrator, stream_categories, stream_scheduler, stream_monitor, stream_monitor_cli)
19. core/ (thinking_pipeline, system_orchestrator, brain19_app)
20. evolution/ (pattern_discovery, epistemic_promotion, concept_proposal)

## HTML Style
- Dark theme (GitHub-style, bg:#0d1117)
- Collapsible sections per module
- Tables for member variables
- Code blocks for signatures
- Color-coded tags: READ-ONLY (green), READ-WRITE (yellow), CORE (blue), EXTERNAL (red)
- Weak points in red-bordered boxes
- Invariants in green-bordered boxes

## Source Code Location
All source files: `/home/hirschpekf/brain19/backend/`

## Existing (to replace)
Current stub: `/home/hirschpekf/brain19/docs/technical/index.html` — replace with full version
