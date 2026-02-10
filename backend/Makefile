# Brain19 Complete Backend Build

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O2 -I.
LDFLAGS = 

# Source files
MEMORY_SOURCES = memory/stm.cpp memory/brain_controller.cpp
KAN_SOURCES = kan/kan_node.cpp kan/kan_layer.cpp kan/kan_module.cpp
ADAPTER_SOURCES = adapter/kan_adapter.cpp
CURIOSITY_SOURCES = curiosity/curiosity_engine.cpp
SNAPSHOT_SOURCES = snapshot_generator.cpp
LTM_SOURCES = ltm/long_term_memory.cpp

ALL_SOURCES = $(MEMORY_SOURCES) $(KAN_SOURCES) $(ADAPTER_SOURCES) \
              $(CURIOSITY_SOURCES) $(SNAPSHOT_SOURCES) $(LTM_SOURCES) demo_integrated.cpp

OBJECTS = $(ALL_SOURCES:.cpp=.o)
TARGET = demo_integrated

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) snapshot.json

run: $(TARGET)
	./$(TARGET)
