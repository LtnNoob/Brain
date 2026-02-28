# Brain19 Complete Backend Build

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O1 -g -I.
LDFLAGS = -lcurl -pthread

# CUDA support: make GPU=1 to enable
ifdef GPU
NVCC = nvcc
NVCCFLAGS = -std=c++20 -O1 -g -I. -DUSE_CUDA --expt-relaxed-constexpr -arch=sm_75
CXXFLAGS += -DUSE_CUDA
LDFLAGS += -lcudart -L/usr/local/cuda/lib64
CUDA_OBJ = cuda/cuda_ops.o cuda/cuda_training.o
cuda/cuda_ops.o: cuda/cuda_ops.cu cuda/cuda_ops.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
cuda/cuda_training.o: cuda/cuda_training.cu cuda/cuda_training.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
CUDA_OBJ = cuda/cuda_ops_cpu.o cuda/cuda_training_cpu.o
cuda/cuda_ops_cpu.o: cuda/cuda_ops_cpu.cpp cuda/cuda_ops.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
cuda/cuda_training_cpu.o: cuda/cuda_training_cpu.cpp cuda/cuda_training.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

# LibTorch support: make TORCH=1 to enable
ifdef TORCH
TORCH_DIR = /home/claude/.openclaw/workspace/venv-sd/lib/python3.12/site-packages/torch
CXXFLAGS += -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include \
            -DUSE_LIBTORCH -D_GLIBCXX_USE_CXX11_ABI=1
NVSHMEM_DIR = /home/claude/.local/share/pipx/venvs/openai-whisper/lib/python3.12/site-packages/nvidia/nvshmem/lib
LDFLAGS += -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
           -Wl,--disable-new-dtags -Wl,-rpath,$(TORCH_DIR)/lib -Wl,-rpath,$(NVSHMEM_DIR) -Wl,--allow-shlib-undefined
TORCH_OBJ = libtorch/torch_kan.o libtorch/torch_training.o
libtorch/torch_kan.o: libtorch/torch_kan.cpp libtorch/torch_kan.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
libtorch/torch_training.o: libtorch/torch_training.cpp libtorch/torch_training.hpp libtorch/torch_kan.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
else
TORCH_OBJ =
endif

# === Source Groups ===
MEMORY_SOURCES = memory/stm.cpp memory/brain_controller.cpp memory/relation_type_registry.cpp
KAN_SOURCES = kan/kan_node.cpp kan/kan_layer.cpp kan/kan_module.cpp
ADAPTER_SOURCES = adapter/kan_adapter.cpp
CURIOSITY_SOURCES = curiosity/curiosity_engine.cpp curiosity/goal_generator.cpp curiosity/trend_tracker.cpp
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
                 cmodel/concept_persistence.cpp cmodel/multihop_sampler.cpp
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
                    evolution/concept_proposal.cpp evolution/graph_densifier.cpp \
                    evolution/property_inheritance.cpp \
                    evolution/complexity_analyzer.cpp evolution/retention_manager.cpp \
                    evolution/trust_propagator.cpp
BOOTSTRAP_SOURCES = bootstrap/foundation_concepts.cpp bootstrap/bootstrap_interface.cpp \
                    bootstrap/context_accumulator.cpp bootstrap/json_parser.cpp
STREAMS_SOURCES = streams/think_stream.cpp streams/stream_orchestrator.cpp \
                  streams/stream_scheduler.cpp streams/stream_monitor.cpp
CURSOR_SOURCES = cursor/focus_cursor.cpp cursor/focus_cursor_manager.cpp \
                 cursor/template_engine.cpp
REASONING_SOURCES = reasoning/concept_reasoner.cpp reasoning/chain_kan.cpp \
                    graph_net/graph_reasoner.cpp
COLEARN_SOURCES = colearn/episodic_memory.cpp colearn/knowledge_extractor.cpp \
                  colearn/colearn_loop.cpp colearn/error_collector.cpp
LANGUAGE_SOURCES = language/bpe_tokenizer.cpp language/kan_encoder.cpp \
                   language/semantic_scorer.cpp language/fusion_layer.cpp \
                   language/kan_decoder.cpp language/kan_language_engine.cpp \
                   language/language_training.cpp language/dimensional_context.cpp \
                   language/deep_kan.cpp language/sentence_parser.cpp language/sentence_parser.cpp
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
               $(CURSOR_SOURCES) $(LANGUAGE_SOURCES) $(REASONING_SOURCES) \
               $(COLEARN_SOURCES)

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o) $(CUDA_OBJ) $(TORCH_OBJ)

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
          test_concept_embeddings test_gdo test_goal_generator \
          test_property_inheritance \
          test_flex_activation \
          test_graph_features \
          test_concept_reasoner \
          test_language_reasoner \
          test_graph_reasoner \
          train_v10 \
          train_v11 \
          train_v12 \
          train_v12_cpu \
          analyze_graph \
          test_colearn \
          test_logger_system \
          test_curiosity_engine \
          test_parallel_performance \
          test_superposition

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

test_property_inheritance: evolution/test_property_inheritance.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_flex_activation: micromodel/test_flex_activation.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_graph_features: evolution/test_graph_features.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# test_concept_reasoner: exclude language_training + system_orchestrator + brain19_app (libtorch deps)
REASONER_EXCLUDE = language/language_training.o core/system_orchestrator.o core/brain19_app.o
REASONER_TEST_OBJECTS = $(filter-out $(REASONER_EXCLUDE),$(CORE_OBJECTS))
test_concept_reasoner: test_concept_reasoner.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_language_reasoner: test_language_reasoner.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_graph_reasoner: test_graph_reasoner.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

train_v10: train_v10.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

train_v11: train_v11.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

train_v12: train_v12.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

train_v12_cpu: train_v12_cpu.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_colearn: test_colearn.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_logger_system: test_logger_system.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_curiosity_engine: curiosity/test_curiosity_engine.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_parallel_performance: test_parallel_performance.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_superposition: test_superposition.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Graph structure analysis tool (no libtorch deps)
analyze_graph: analyze_graph.o $(REASONER_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# --- Pattern Rule ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	find . -name '*.o' -delete
	rm -f $(TARGETS) snapshot.json brain19

run: brain19
	./brain19
