CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -pthread
CXXFLAGS_DBG = -std=c++20 -Wall -Wextra -g -O0 -pthread -fsanitize=thread

BACKEND = backend

# Source files needed for the stream system tests (full backend)
STREAM_SRCS = \
	$(BACKEND)/streams/think_stream.cpp \
	$(BACKEND)/streams/stream_orchestrator.cpp \
	$(BACKEND)/ltm/long_term_memory.cpp \
	$(BACKEND)/memory/stm.cpp \
	$(BACKEND)/micromodel/micro_model.cpp \
	$(BACKEND)/micromodel/micro_model_registry.cpp \
	$(BACKEND)/micromodel/embedding_manager.cpp \
	$(BACKEND)/micromodel/micro_trainer.cpp \
	$(BACKEND)/micromodel/persistence.cpp \
	$(BACKEND)/micromodel/relevance_map.cpp \
	$(BACKEND)/kan/kan_node.cpp \
	$(BACKEND)/kan/kan_layer.cpp \
	$(BACKEND)/kan/kan_module.cpp \
	$(BACKEND)/persistent/stm_snapshot.cpp \
	$(BACKEND)/persistent/persistent_ltm.cpp \
	$(BACKEND)/persistent/wal.cpp

# Queue-only test (no backend deps)
test_streams_lite: tests/test_streams.cpp $(BACKEND)/streams/lock_free_queue.hpp
	$(CXX) $(CXXFLAGS) -I$(BACKEND) -o test_streams tests/test_streams.cpp
	./test_streams

# Full test with backend
test_streams: tests/test_streams.cpp $(STREAM_SRCS)
	$(CXX) $(CXXFLAGS) -DHAS_FULL_BACKEND -I$(BACKEND) -o test_streams \
		tests/test_streams.cpp $(STREAM_SRCS)
	./test_streams

# Debug build with ThreadSanitizer
test_streams_tsan: tests/test_streams.cpp $(STREAM_SRCS)
	$(CXX) $(CXXFLAGS_DBG) -DHAS_FULL_BACKEND -I$(BACKEND) -o test_streams_tsan \
		tests/test_streams.cpp $(STREAM_SRCS)
	./test_streams_tsan

clean:
	rm -f test_streams test_streams_tsan

.PHONY: test_streams test_streams_lite test_streams_tsan clean
