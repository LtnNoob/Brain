# Brain19 Complete Backend Build

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O2 -I.
LDFLAGS = -lcurl

# === Source Groups ===
MEMORY_SOURCES = memory/stm.cpp memory/brain_controller.cpp
KAN_SOURCES = kan/kan_node.cpp kan/kan_layer.cpp kan/kan_module.cpp
ADAPTER_SOURCES = adapter/kan_adapter.cpp
CURIOSITY_SOURCES = curiosity/curiosity_engine.cpp
SNAPSHOT_SOURCES = snapshot_generator.cpp
LTM_SOURCES = ltm/long_term_memory.cpp
MICROMODEL_SOURCES = micromodel/micro_model.cpp micromodel/micro_model_registry.cpp \
                     micromodel/embedding_manager.cpp micromodel/micro_trainer.cpp \
                     micromodel/relevance_map.cpp micromodel/persistence.cpp
COGNITIVE_SOURCES = cognitive/cognitive_dynamics.cpp
UNDERSTANDING_SOURCES = understanding/understanding_layer.cpp understanding/mini_llm.cpp \
                        understanding/ollama_mini_llm.cpp
LLM_SOURCES = llm/chat_interface.cpp llm/ollama_client.cpp
IMPORTER_SOURCES = importers/wikipedia_importer.cpp importers/scholar_importer.cpp
INGESTOR_SOURCES = ingestor/entity_extractor.cpp ingestor/relation_extractor.cpp \
                   ingestor/text_chunker.cpp ingestor/trust_tagger.cpp \
                   ingestor/knowledge_ingestor.cpp ingestor/proposal_queue.cpp \
                   ingestor/ingestion_pipeline.cpp

# Core library (everything except mains)
CORE_SOURCES = $(MEMORY_SOURCES) $(KAN_SOURCES) $(ADAPTER_SOURCES) \
               $(CURIOSITY_SOURCES) $(SNAPSHOT_SOURCES) $(LTM_SOURCES) \
               $(MICROMODEL_SOURCES) $(COGNITIVE_SOURCES) $(UNDERSTANDING_SOURCES) \
               $(LLM_SOURCES) $(IMPORTER_SOURCES) $(INGESTOR_SOURCES)

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)

# === Targets ===
TARGETS = demo_integrated demo_epistemic_complete demo_understanding_layer \
          demo_cognitive_dynamics demo_chat \
          test_cognitive_dynamics test_epistemic_enforcement \
          test_understanding_layer test_importers \
          test_brain test_micromodel test_ingestor brain19_cli

.PHONY: all clean run

all: $(TARGETS)

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

brain19_cli: tools/brain19_cli.o $(CORE_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# --- Pattern Rule ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	find . -name '*.o' -delete
	rm -f $(TARGETS) snapshot.json

run: demo_integrated
	./demo_integrated
