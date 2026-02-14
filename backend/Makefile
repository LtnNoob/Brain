# Brain19 Complete Backend Build

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O2 -I.
LDFLAGS = -lcurl -pthread

# === Source Groups ===
MEMORY_SOURCES = memory/stm.cpp memory/brain_controller.cpp memory/relation_type_registry.cpp
KAN_SOURCES = kan/kan_node.cpp kan/kan_layer.cpp kan/kan_module.cpp
ADAPTER_SOURCES = adapter/kan_adapter.cpp
CURIOSITY_SOURCES = curiosity/curiosity_engine.cpp curiosity/goal_generator.cpp
SNAPSHOT_SOURCES = snapshot_generator.cpp
LTM_SOURCES = ltm/long_term_memory.cpp
MICROMODEL_SOURCES = micromodel/micro_model.cpp micromodel/micro_model_registry.cpp \
                     micromodel/embedding_manager.cpp micromodel/micro_trainer.cpp \
                     micromodel/relevance_map.cpp micromodel/persistence.cpp \
                     micromodel/concept_embedding_store.cpp
COGNITIVE_SOURCES = cognitive/cognitive_dynamics.cpp cognitive/global_dynamics_operator.cpp
UNDERSTANDING_SOURCES = understanding/understanding_layer.cpp understanding/mini_llm.cpp
CMODEL_SOURCES = cmodel/concept_model.cpp cmodel/concept_model_registry.cpp \
                 cmodel/concept_trainer.cpp cmodel/concept_pattern_engine.cpp \
                 cmodel/concept_persistence.cpp
LLM_SOURCES = llm/chat_interface.cpp
IMPORTER_SOURCES = importers/wikipedia_importer.cpp importers/scholar_importer.cpp \
                   importers/http_client.cpp
INGESTOR_SOURCES = ingestor/entity_extractor.cpp ingestor/relation_extractor.cpp \
                   ingestor/text_chunker.cpp ingestor/trust_tagger.cpp \
                   ingestor/knowledge_ingestor.cpp ingestor/proposal_queue.cpp \
                   ingestor/ingestion_pipeline.cpp
PERSISTENT_SOURCES = persistent/persistent_ltm.cpp persistent/wal.cpp \
                     persistent/stm_snapshot.cpp persistent/checkpoint_manager.cpp \
                     persistent/checkpoint_restore.cpp
HYBRID_SOURCES = hybrid/kan_validator.cpp hybrid/domain_manager.cpp \
                 hybrid/refinement_loop.cpp hybrid/epistemic_bridge.cpp \
                 hybrid/hypothesis_translator.cpp hybrid/kan_graph_monitor.cpp
EVOLUTION_SOURCES = evolution/pattern_discovery.cpp evolution/epistemic_promotion.cpp \
                    evolution/concept_proposal.cpp
BOOTSTRAP_SOURCES = bootstrap/foundation_concepts.cpp bootstrap/bootstrap_interface.cpp \
                    bootstrap/context_accumulator.cpp bootstrap/json_parser.cpp
STREAMS_SOURCES = streams/think_stream.cpp streams/stream_orchestrator.cpp \
                  streams/stream_scheduler.cpp streams/stream_monitor.cpp
CURSOR_SOURCES = cursor/focus_cursor.cpp cursor/focus_cursor_manager.cpp \
                 cursor/template_engine.cpp
LANGUAGE_SOURCES = language/bpe_tokenizer.cpp language/kan_encoder.cpp \
                   language/semantic_scorer.cpp language/fusion_layer.cpp \
                   language/kan_decoder.cpp language/kan_language_engine.cpp \
                   language/language_training.cpp language/dimensional_context.cpp
CORE_APP_SOURCES = core/system_orchestrator.cpp core/brain19_app.cpp \
                   core/thinking_pipeline.cpp

# Core library (everything except mains)
CORE_SOURCES = $(MEMORY_SOURCES) $(KAN_SOURCES) $(ADAPTER_SOURCES) \
               $(CURIOSITY_SOURCES) $(SNAPSHOT_SOURCES) $(LTM_SOURCES) \
               $(MICROMODEL_SOURCES) $(CMODEL_SOURCES) $(COGNITIVE_SOURCES) \
               $(UNDERSTANDING_SOURCES) \
               $(LLM_SOURCES) $(IMPORTER_SOURCES) $(INGESTOR_SOURCES) \
               $(PERSISTENT_SOURCES) $(HYBRID_SOURCES) $(EVOLUTION_SOURCES) \
               $(BOOTSTRAP_SOURCES) $(STREAMS_SOURCES) $(CORE_APP_SOURCES) \
               $(CURSOR_SOURCES) $(LANGUAGE_SOURCES)

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)

# === Targets ===
TARGETS = brain19 \
          demo_integrated demo_epistemic_complete demo_understanding_layer \
          demo_cognitive_dynamics demo_chat \
          test_cognitive_dynamics test_epistemic_enforcement \
          test_understanding_layer test_importers \
          test_brain test_micromodel test_ingestor test_persistent_ltm \
          test_stm_snapshot brain19_cli \
          test_focus_cursor test_termination_conflict \
          test_template_engine test_pipeline_cursor \
          test_relation_registry test_json_parser \
          test_concept_embeddings test_gdo test_goal_generator

.PHONY: all clean run

all: $(TARGETS)

# --- Main Binary ---
brain19: main.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# --- Demos ---
demo_integrated: demo_integrated.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

demo_epistemic_complete: demo_epistemic_complete.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

demo_understanding_layer: demo_understanding_layer.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

demo_cognitive_dynamics: demo_cognitive_dynamics.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

demo_chat: demo_chat.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# --- Tests ---
test_cognitive_dynamics: test_cognitive_dynamics.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_epistemic_enforcement: test_epistemic_enforcement.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_understanding_layer: test_understanding_layer.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_importers: test_importers.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_brain: memory/test_brain.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_micromodel: micromodel/test_micromodel.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_ingestor: ingestor/test_ingestor.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_persistent_ltm: test_persistent_ltm.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_stm_snapshot: test_stm_snapshot.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

brain19_cli: tools/brain19_cli.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_focus_cursor: cursor/test_focus_cursor.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_termination_conflict: cursor/test_termination_conflict.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_template_engine: cursor/test_template_engine.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_pipeline_cursor: core/test_pipeline_cursor.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_relation_registry: memory/test_relation_registry.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_json_parser: bootstrap/test_json_parser.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_concept_embeddings: micromodel/test_concept_embeddings.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_gdo: cognitive/test_gdo.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_goal_generator: curiosity/test_goal_generator.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# --- Pattern Rule ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	find . -name '*.o' -delete
	rm -f $(TARGETS) snapshot.json brain19

run: brain19
	./brain19
