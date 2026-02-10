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


# Lock hierarchy test (standalone, no backend deps)
test_lock_hierarchy: tests/test_lock_hierarchy.cpp backend/concurrent/lock_hierarchy.hpp backend/concurrent/deadlock_detector.hpp
	$(CXX) -std=c++20 -Wall -Wextra -g -O0 -pthread -DBRAIN19_DEBUG \
		-o test_lock_hierarchy tests/test_lock_hierarchy.cpp
	./test_lock_hierarchy

SCHEDULER_SRCS = \
	$(BACKEND)/streams/stream_scheduler.cpp

MONITOR_SRCS = \
	$(BACKEND)/streams/stream_monitor.cpp \
	$(BACKEND)/streams/stream_monitor_cli.cpp

# Stream categories test (Phase 5.2)
test_stream_categories: tests/test_stream_categories.cpp $(STREAM_SRCS) $(SCHEDULER_SRCS)
	$(CXX) $(CXXFLAGS) -DHAS_FULL_BACKEND -I$(BACKEND) -o test_stream_categories \
		tests/test_stream_categories.cpp $(STREAM_SRCS) $(SCHEDULER_SRCS)
	./test_stream_categories

# ─── Phase 4: Checkpoint Manager ───────────────────────────────────────────

CHECKPOINT_SRCS = \
	$(BACKEND)/persistent/checkpoint_manager.cpp \
	$(BACKEND)/persistent/checkpoint_restore.cpp \
	$(BACKEND)/persistent/persistent_ltm.cpp \
	$(BACKEND)/persistent/wal.cpp \
	$(BACKEND)/persistent/stm_snapshot.cpp \
	$(BACKEND)/micromodel/micro_model.cpp \
	$(BACKEND)/micromodel/micro_model_registry.cpp \
	$(BACKEND)/micromodel/embedding_manager.cpp \
	$(BACKEND)/micromodel/micro_trainer.cpp \
	$(BACKEND)/micromodel/persistence.cpp \
	$(BACKEND)/micromodel/relevance_map.cpp \
	$(BACKEND)/kan/kan_node.cpp \
	$(BACKEND)/kan/kan_layer.cpp \
	$(BACKEND)/kan/kan_module.cpp \
	$(BACKEND)/ltm/long_term_memory.cpp \
	$(BACKEND)/memory/stm.cpp

brain19_checkpoint: $(BACKEND)/checkpoint_cli.cpp $(CHECKPOINT_SRCS)
	$(CXX) $(CXXFLAGS) -I$(BACKEND) -o brain19_checkpoint \
		$(BACKEND)/checkpoint_cli.cpp $(CHECKPOINT_SRCS)

test_checkpoint: tests/test_checkpoint.cpp $(CHECKPOINT_SRCS)
	$(CXX) $(CXXFLAGS) -I$(BACKEND) -o test_checkpoint \
		tests/test_checkpoint.cpp $(CHECKPOINT_SRCS)
	./test_checkpoint

# ─── Phase 5.3: Stream Monitor ──────────────────────────────────────────────

brain19_monitor: $(BACKEND)/brain19_monitor.cpp $(STREAM_SRCS) $(SCHEDULER_SRCS) $(MONITOR_SRCS)
	$(CXX) $(CXXFLAGS) -DHAS_FULL_BACKEND -I$(BACKEND) -o brain19_monitor \
		$(BACKEND)/brain19_monitor.cpp $(STREAM_SRCS) $(SCHEDULER_SRCS) $(MONITOR_SRCS)

test_stream_monitor: tests/test_stream_monitor.cpp $(STREAM_SRCS) $(SCHEDULER_SRCS) $(MONITOR_SRCS)
	$(CXX) $(CXXFLAGS) -DHAS_FULL_BACKEND -I$(BACKEND) -o test_stream_monitor \
		tests/test_stream_monitor.cpp $(STREAM_SRCS) $(SCHEDULER_SRCS) $(MONITOR_SRCS)
	./test_stream_monitor

# ─── Phase 7: KAN-LLM Hybrid ───────────────────────────────────────────

HYBRID_SRCS = \
	$(BACKEND)/hybrid/hypothesis_translator.cpp \
	$(BACKEND)/hybrid/epistemic_bridge.cpp \
	$(BACKEND)/hybrid/kan_validator.cpp \
	$(BACKEND)/hybrid/domain_manager.cpp \
	$(BACKEND)/hybrid/refinement_loop.cpp

KAN_SRCS = \
	$(BACKEND)/kan/kan_node.cpp \
	$(BACKEND)/kan/kan_layer.cpp \
	$(BACKEND)/kan/kan_module.cpp

LTM_SRCS = \
	$(BACKEND)/ltm/long_term_memory.cpp \
	$(BACKEND)/memory/stm.cpp

test_kan_llm_hybrid: tests/test_kan_llm_hybrid.cpp $(HYBRID_SRCS) $(KAN_SRCS) $(LTM_SRCS)
	$(CXX) $(CXXFLAGS) -I$(BACKEND) -o test_kan_llm_hybrid \
		tests/test_kan_llm_hybrid.cpp $(HYBRID_SRCS) $(KAN_SRCS) $(LTM_SRCS)
	./test_kan_llm_hybrid

clean:
	rm -f test_streams test_streams_tsan test_lock_hierarchy test_stream_categories brain19_checkpoint test_checkpoint test_kan_llm_hybrid brain19_monitor test_stream_monitor

.PHONY: test_streams test_streams_lite test_streams_tsan test_lock_hierarchy test_stream_categories brain19_checkpoint test_checkpoint brain19_monitor test_stream_monitor test_kan_llm_hybrid clean
