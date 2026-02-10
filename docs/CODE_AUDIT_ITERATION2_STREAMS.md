# Code Audit — Iteration 2: Multi-Stream System

**Date:** 2026-02-10  
**Reviewer:** Senior C++ Code Review (automated)  
**Scope:** `backend/streams/*`, `tests/test_streams.cpp`, `tests/test_lock_hierarchy.cpp`  
**Integration check against:** `brain_controller.hpp`, `cognitive_dynamics.hpp`, `relevance_map.hpp`

---

## Gesamtscore: 7.2 / 10

| Kriterium | Score | Kommentar |
|-----------|-------|-----------|
| Korrektheit | 7 | Solide Queue-Implementierung, aber Shutdown-Race und consumer drain-Bug |
| Thread-Safety | 7 | Memory orders korrekt für Queues, aber Cell-Struct hat false sharing |
| Error Handling | 6 | Exception in run() wird gefangen, aber Shutdown-Edge-Cases unvollständig |
| API Design | 8 | Konsistent mit Core, saubere Trennung Stream/Orchestrator |
| Performance | 6 | False sharing in Cell/Metrics, do_curiosity iteriert alle Concepts |
| Integration | 8 | Shared-Wrappers korrekt genutzt, eigener STM-Context pro Stream |
| Code Quality | 8 | Modern C++20, RAII, gutes Naming |

---

## Per-File Findings

### 1. `lock_free_queue.hpp` — Score: 7/10

**🔴 CRITICAL: False sharing in Cell struct**

Cell enthält atomic<size_t> sequence und T data im selben Cache-Line. Bei hohem Contention schreiben Producer in data und dann in sequence — benachbarte Cells können false-sharing verursachen.

```cpp
// VORHER:
struct Cell {
    std::atomic<size_t> sequence;
    T data{};
};

// FIX:
struct alignas(64) Cell {  // Hardware destructive interference size
    std::atomic<size_t> sequence;
    T data{};
};
```

Alternativ mit std::hardware_destructive_interference_size (C++17):
```cpp
struct alignas(std::hardware_destructive_interference_size) Cell {
    std::atomic<size_t> sequence;
    T data{};
};
```

**🟡 MEDIUM: std::vector<Cell> Alignment nicht garantiert**

std::vector allokiert über den Default-Allocator, der alignas(64) auf den Cells nicht garantiert respektiert. Fix: Custom aligned allocator oder std::aligned_alloc + manuelles Lifetime-Management.

**🟡 MEDIUM: SPSC Queue verliert 1 Slot Kapazität**

SPSCQueue::capacity() gibt capacity_ - 1 zurück, was korrekt ist. Aber der Konstruktor nimmt den User-Wert, rundet auf pow2, und der User bekommt weniger als erwartet. Sollte dokumentiert werden oder die Kapazität intern um 1 erhöht werden.

**🟢 LOW: next_pow2(0) ergibt 0 → Division by Zero via mask_**

```cpp
// FIX: Guard im Konstruktor
explicit MPMCQueue(size_t capacity)
    : capacity_(next_pow2(std::max(capacity, size_t(2))))  // minimum 2
    ...
```

**🟢 LOW: empty() und size_approx() sind racy**

Dokumentiert als "approx" — OK für Monitoring, aber empty() wird im Test für Consumer-Termination verwendet (siehe test_streams.cpp).

---

### 2. `stream_config.hpp` — Score: 9/10

**🟢 LOW: Keine Validierung**

spin_count, yield_count etc. werden nie validiert. Ein spin_count=0 mit SpinYieldSleep überspringt direkt zu yield — funktioniert, aber ist verwirrend.

Saubere Implementierung, gutes Design.

---

### 3. `think_stream.hpp` + `.cpp` — Score: 7/10

**🔴 CRITICAL: Restart-Race in start()**

```cpp
bool ThinkStream::start() {
    StreamState expected = StreamState::Created;
    if (!state_.compare_exchange_strong(expected, StreamState::Starting, ...)) {
        expected = StreamState::Stopped;
        if (!state_.compare_exchange_strong(expected, StreamState::Starting, ...)) {
            return false;
        }
    }
    // ...
    thread_ = std::thread(&ThinkStream::run, this);  // alte thread_ noch joinable?
```

Wenn start() aus Stopped aufgerufen wird, ist thread_ noch joinable (wurde nicht gejoint). Der std::thread-Konstruktor ruft std::terminate() auf, wenn das Ziel-Objekt noch joinable ist.

```cpp
// FIX:
bool ThinkStream::start() {
    // ... CAS logic ...
    
    // Join old thread if needed
    if (thread_.joinable()) {
        thread_.join();
    }
    
    stop_requested_.store(false, std::memory_order_relaxed);
    metrics_.reset();
    context_id_ = stm_.create_context();
    thread_ = std::thread(&ThinkStream::run, this);
    return true;
}
```

**🟡 HIGH: join() Timeout ohne Detach → Resource Leak**

```cpp
bool ThinkStream::join(std::chrono::milliseconds timeout) {
    // ...
    // Timeout — thread still running.
    return false;  // thread_ bleibt joinable, wird nie gejoint
}
```

Wenn join() false zurückgibt und dann ~ThinkStream() aufgerufen wird, wird thread_.join() im Destruktor blockierend aufgerufen — potenziell endlos.

```cpp
// FIX (im Destruktor):
~ThinkStream() {
    stop();
    if (thread_.joinable()) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            auto s = state_.load(std::memory_order_acquire);
            if (s == StreamState::Stopped || s == StreamState::Error) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (state_.load() == StreamState::Stopped || state_.load() == StreamState::Error) {
            thread_.join();
        } else {
            thread_.detach();  // letzte Option, da stop_requested_ true ist
        }
    }
    if (context_id_ != 0) {
        try { stm_.destroy_context(context_id_); } catch (...) {}
    }
}
```

**🟡 HIGH: context_id_ bei Restart wird nicht aufgeräumt**

Beim Restart via start() wird stm_.create_context() erneut aufgerufen, aber der alte context_id_ wird nicht destroyed → Context-Leak in STM.

```cpp
// FIX in start():
if (context_id_ != 0) {
    try { stm_.destroy_context(context_id_); } catch (...) {}
}
context_id_ = stm_.create_context();
```

**🟡 MEDIUM: do_curiosity() iteriert ALLE Concepts**

ltm_.get_all_concept_ids() kopiert potenziell tausende IDs. Bei 10ms tick interval ist das ein Performance-Problem.

```cpp
// FIX: Sample statt iterate-all
void ThinkStream::do_curiosity() {
    auto sample = ltm_.get_random_concept_ids(20);  // nur Stichprobe
    for (auto cid : sample) { ... }
}
```

Falls get_random_concept_ids nicht existiert: Batch mit Offset/Limit oder Round-Robin pro Tick.

**🟡 MEDIUM: run() idle_count nie erhöht im Normalfall**

```cpp
// In run():
} else {
    tick();
    idle_count = 0;  // immer 0 gesetzt, backoff wird nie erreicht
```

Der backoff() wird nur im catch-Block aufgerufen. Im normalen Betrieb wird idle_count permanent auf 0 gesetzt. Das ist kein Bug (sleep_for wird direkt aufgerufen), aber der backoff-Mechanismus ist dann nur für Error-Recovery, was nicht dem Config-Design entspricht.

**🟢 LOW: StreamMetrics hat potentielles false sharing**

Alle atomics liegen in einer Struct ohne Padding. Bei hoher Metrik-Update-Frequenz teilen sie Cache-Lines.

---

### 4. `stream_orchestrator.hpp` + `.cpp` — Score: 7/10

**🟡 HIGH: const_cast in health_check() const**

```cpp
const_cast<OrchestratorMetrics&>(metrics_).total_ticks_all.store(...);
```

Das ist UB wenn das StreamOrchestrator-Objekt const ist. Besser: OrchestratorMetrics members als mutable deklarieren.

```cpp
// FIX:
mutable OrchestratorMetrics metrics_;
```

**🟡 HIGH: shutdown() Timeout wird pro Stream verbraucht**

```cpp
for (auto& [id, stream] : streams_) {
    if (!stream->join(timeout)) {  // jeder Stream bekommt volles Timeout!
        all_stopped = false;
    }
}
```

Bei 10 Streams × 5s Timeout = 50s maximale Shutdown-Dauer.

```cpp
// FIX: Deadline statt per-Stream Timeout
auto deadline = std::chrono::steady_clock::now() + timeout;
for (auto& [id, stream] : streams_) {
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) remaining = std::chrono::milliseconds(1);
    if (!stream->join(remaining)) {
        all_stopped = false;
    }
}
```

**🟡 MEDIUM: set_alert_callback nicht thread-safe**

alert_cb_ wird ohne Lock gesetzt, aber von monitor_loop() gelesen. Data race.

```cpp
// FIX:
void StreamOrchestrator::set_alert_callback(AlertCallback cb) {
    std::lock_guard lock(streams_mtx_);  // oder eigener mutex
    alert_cb_ = std::move(cb);
}
```

**🟡 MEDIUM: destroy_stream active_streams Decrement auch für nicht-laufende Streams**

Wenn ein Stream nie gestartet wurde (state=Created), decrementiert destroy_stream trotzdem active_streams.

**🟢 LOW: least_loaded_stream nutzt total_ticks als Load-Metrik**

Total ticks ist kumulativ und steigt monoton — ein neu gestarteter Stream bekommt immer alle Tasks. Besser: Queue-Größe oder recent tick rate.

---

### 5. `tests/test_streams.cpp` — Score: 6/10

**🟡 HIGH: Consumer drain-Race in MPMC concurrent Test**

```cpp
done.store(true, std::memory_order_release);
std::this_thread::sleep_for(std::chrono::milliseconds(100));  // hoffen auf drain
for (auto& t : consumers) t.join();
```

Consumers prüfen `done.load(relaxed) || !q.empty()` — aber empty() ist racy. Ein Consumer könnte terminieren während Items noch in der Queue sind.

**🟡 MEDIUM: TEST Macro definiert aber nie benutzt**

Das TEST Macro und test_funcs Vector werden definiert, aber main() ruft Tests direkt auf statt über die Registry.

**🟡 MEDIUM: Keine Integration-Tests ohne HAS_FULL_BACKEND**

6 von 10 Tests werden als SKIPPED/FAILED gemeldet. Exit Code 1, obwohl nur Skips vorliegen.

---

### 6. `tests/test_lock_hierarchy.cpp` — Score: 8/10

**🟢 LOW: Kein echter Integrationstest mit ThinkStream**

Test 6 simuliert das Pattern manuell. Ein echter Integrationstest mit ThinkStream + HierarchicalMutex wäre wertvoller.

Insgesamt solide Test-Suite für die Lock-Hierarchie.

---

## Integration mit Core

### ✅ Positiv

1. **Shared Wrappers korrekt genutzt** — SharedLTM, SharedSTM, SharedRegistry, SharedEmbeddings werden by-reference gehalten
2. **Eigener Context pro Stream** — stm_.create_context() in start(), stm_.destroy_context() in Destruktor. Korrekte Isolation
3. **Read-only LTM** — do_spreading, do_curiosity, do_understanding lesen LTM nur, schreiben nur in STM. Konsistent mit CognitiveDynamics-Architekturvertrag
4. **API-Pattern passt** — ThinkStream's Subsystem-Operationen spiegeln CognitiveDynamics' API sauber wider

### ⚠️ Bedenken

1. **Duplizierung mit CognitiveDynamics** — ThinkStream::do_spreading() implementiert eigene Spreading-Logic, statt CognitiveDynamics::spread_activation() zu nutzen. Kann zu Drift führen
2. **Keine RelevanceMap-Nutzung** — Der Stream-Layer nutzt RelevanceMap nicht. Bei Phase 3 Creativity wird das Integration brauchen
3. **BrainController kennt StreamOrchestrator nicht** — Es gibt keinen klaren Integrationspoint

---

## Priorisierte Fix-Liste

| # | Priorität | File | Issue | Aufwand |
|---|-----------|------|-------|---------|
| 1 | 🔴 CRITICAL | think_stream.cpp | Restart-Race: thread_ noch joinable → std::terminate | 5 min |
| 2 | 🔴 CRITICAL | lock_free_queue.hpp | False sharing in Cell — alignas(64) | 5 min |
| 3 | 🟡 HIGH | think_stream.cpp | Context-Leak bei Restart | 5 min |
| 4 | 🟡 HIGH | think_stream.cpp | join() timeout → Destruktor blockiert endlos | 15 min |
| 5 | 🟡 HIGH | stream_orchestrator.cpp | const_cast UB → mutable | 2 min |
| 6 | 🟡 HIGH | stream_orchestrator.cpp | Shutdown-Timeout kumuliert pro Stream | 10 min |
| 7 | 🟡 HIGH | test_streams.cpp | Consumer drain-Race im MPMC-Test | 15 min |
| 8 | 🟡 MEDIUM | stream_orchestrator.cpp | set_alert_callback Data Race | 5 min |
| 9 | 🟡 MEDIUM | think_stream.cpp | do_curiosity iteriert alle Concepts | 20 min |
| 10 | 🟡 MEDIUM | stream_orchestrator.cpp | destroy_stream decrementiert falsch | 5 min |
| 11 | 🟡 MEDIUM | lock_free_queue.hpp | next_pow2(0) → mask=0 → UB | 2 min |
| 12 | 🟢 LOW | stream_orchestrator.cpp | least_loaded_stream Metrik ungeeignet | 15 min |
| 13 | 🟢 LOW | think_stream.hpp | StreamMetrics false sharing | 10 min |
| 14 | 🟢 LOW | test_streams.cpp | SKIP vs FAIL Unterscheidung | 10 min |

**Geschätzter Gesamtaufwand für CRITICAL+HIGH: ~1h**

---

## Fazit

Der Multi-Stream Layer ist architektonisch sauber aufgebaut. Die Vyukov MPMC Queue ist korrekt implementiert (ABA-safe via Sequenznummern), die Stream-Lifecycle-State-Machine ist nachvollziehbar, und die Integration mit dem Core über Shared-Wrappers ist konsistent.

Die kritischsten Issues sind der Restart-Race in ThinkStream::start() (sofortiger std::terminate) und das false sharing in der Queue. Beides ist mit wenigen Zeilen fixbar.

Mittelfristig sollte die Duplizierung von Spreading-Logic zwischen ThinkStream und CognitiveDynamics aufgelöst werden, und ein klarer Integrationspoint mit BrainController geschaffen werden.
