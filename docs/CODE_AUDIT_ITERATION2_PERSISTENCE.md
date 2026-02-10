# Code Audit — Iteration 2: Persistence + Concurrent Layer

**Date:** 2026-02-10  
**Reviewer:** Senior C++ Code Review (automated)  
**Scope:** `backend/persistent/` + `backend/concurrent/`

---

## Gesamtscore: 7.2 / 10

| Kriterium | Score |
|---|---|
| Korrektheit | 6.5 |
| Thread-Safety | 7.0 |
| Error Handling | 6.5 |
| API Design | 8.0 |
| Performance | 7.5 |
| Code Quality | 8.0 |

---

## Per-File Findings

### 1. `persistent_ltm.hpp` + `.cpp` — Score: 7/10

**[CRITICAL] Double WAL-log in `invalidate_concept()`**
`invalidate_concept()` logs a WAL entry, then calls `update_epistemic_metadata()` which logs *another* WAL entry. During normal operation, **two WAL entries are written for one logical op**, wasting IO and producing redundant replay.

```cpp
// FIX: Direct mutation instead of calling update_epistemic_metadata
bool PersistentLTM::invalidate_concept(ConceptId id, double invalidation_trust) {
    auto it = concept_index_.find(id);
    if (it == concept_index_.end()) return false;
    auto* rec = concepts_->record(it->second);
    if (rec->is_deleted()) return false;
    if (invalidation_trust < 0.0 || invalidation_trust > 1.0)
        invalidation_trust = 0.05;
    if (wal_) {
        WALInvalidateConceptPayload wp{};
        wp.concept_id = id;
        wp.invalidation_trust = invalidation_trust;
        wal_->append(WALOpType::INVALIDATE_CONCEPT, &wp, sizeof(wp));
    }
    rec->epistemic_status = static_cast<uint8_t>(EpistemicStatus::INVALIDATED);
    rec->trust = invalidation_trust;
    return true;
}
```

**[HIGH] `const_cast` in `retrieve_concept()` — Data Race**
`retrieve_concept()` is `const` but mutates `access_count` and `last_access_epoch_us` via `const_cast`. Under `SharedLTM`'s `shared_lock`, multiple readers race on these writes — **undefined behavior**.

```cpp
// FIX Option A: Use std::atomic<uint64_t> for access_count/last_access_epoch_us
// FIX Option B: Move stats update to a separate non-const method
// FIX Option C: Remove stats update from retrieve (preferred — keep reads pure)
```

**[MEDIUM] `add_relation()` returns 0 on failure — ambiguous**
ID 0 could be valid. Use `std::optional<RelationId>`.

**[LOW] `now_epoch_us()` is static member — should be free function.

---

### 2. `persistent_store.hpp` — Score: 8/10

**[HIGH] No move constructor despite deleted copy**
Should explicitly `= delete` move ops for clarity.

**[MEDIUM] `record()` bounds-checks against `capacity` not `record_count`**
Allows reading uninitialized memory between `record_count` and `capacity`.

```cpp
// FIX: Check record_count (or provide unchecked variant for internal use)
RecordT* record(size_t index) {
    if (index >= header()->record_count)
        throw std::out_of_range("PersistentStore: index beyond record_count");
    // ...
}
```

**[MEDIUM] `grow()` invalidates header pointer — external callers may hold stale pointers**
Consider making `grow()` private.

**[LOW] No `MAP_POPULATE` hint.

---

### 3. `persistent_records.hpp` — Score: 9/10

✅ Clean struct layout, proper `static_assert`s, cache-aligned.

**[LOW] `clear()` via `memset` — safe due to `is_trivially_copyable` assert. Good.

---

### 4. `string_pool.hpp` — Score: 7/10

**[HIGH] `uint32_t` offset — no overflow check at 4GB boundary**

```cpp
// FIX:
if (hdr->used_bytes > UINT32_MAX)
    throw std::overflow_error("StringPool: exceeded 4GB offset limit");
```

**[MEDIUM] `fstat` return value unchecked in existing-file path**

```cpp
// FIX:
if (::fstat(fd_, &st) != 0) {
    ::close(fd_); throw std::runtime_error("StringPool: fstat failed");
}
```

**[MEDIUM] No string deduplication — duplicates waste mmap space.

**[LOW] Missing explicit move deletion.

---

### 5. `wal.hpp` + `.cpp` — Score: 7.5/10

**[HIGH] CRC32 table init is not thread-safe**
Non-atomic `bool` flag + static array — data race on first concurrent access.

```cpp
// FIX: constexpr or function-local static with lambda
static const auto crc32_table = []() {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        t[i] = crc;
    }
    return t;
}();
```

**[HIGH] Non-atomic WAL writes (two `write()` calls)**
Torn writes on crash between header and payload. CRC detects it on recovery — acceptable but `writev()` is better:

```cpp
struct iovec iov[2] = {
    {&hdr, sizeof(hdr)},
    {const_cast<void*>(payload), payload_size}
};
::writev(fd_, iov, payload_size > 0 ? 2 : 1);
```

**[MEDIUM] `__attribute__((packed))` on WALEntryHeader — potential unaligned access UB. Safe here since only used with `read()`/`write()` syscalls, but fragile.

**[MEDIUM] `checkpoint()` resets sequence to 1 — monotonicity lost across crashes.

---

### 6. `stm_snapshot.hpp` + `.cpp` — Score: 7.5/10

**[MEDIUM] No checksum on snapshot files — silent corruption.
**[MEDIUM] `rotate_snapshots` not thread-safe with concurrent creation.
**[LOW] Little-endian assumption in `write_pod`/`read_pod`.
**[LOW] No migration path for version bumps.

---

### 7. `stm_snapshot_data.hpp` — Score: 9/10

✅ Clean data structs. No issues.

---

### 8. `shared_ltm.hpp` — Score: 8.5/10

**[HIGH] Inherits `retrieve_concept` data race from underlying LTM**
`shared_lock` + `const_cast` mutation = UB with concurrent readers.

✅ Otherwise clean shared_mutex read/write separation.

---

### 9. `shared_stm.hpp` — Score: 6.5/10

**[CRITICAL] `std::shared_mutex` is not movable — `unordered_map` rehash = UB**
`context_mutexes_` stores `std::shared_mutex` by value. Map rehash moves elements — but `shared_mutex` is not movable.

```cpp
// FIX: Use unique_ptr
mutable std::unordered_map<ContextId, std::unique_ptr<std::shared_mutex>> context_mutexes_;

// create_context():
context_mutexes_[cid] = std::make_unique<std::shared_mutex>();

// get_context_mutex():
return *context_mutexes_.at(cid);
```

**[HIGH] `get_context_mutex()` accesses map without any lock**
Concurrent `destroy_context()` can erase while another thread calls `.at()` — data race.

```cpp
// FIX: Hold global shared lock
std::shared_mutex& get_context_mutex(ContextId cid) const {
    std::shared_lock lock(global_mtx_);
    return *context_mutexes_.at(cid);
}
```

**[MEDIUM] Config setters don't protect against concurrent reads of config values.

---

### 10. `shared_registry.hpp` — Score: 6.5/10

**[CRITICAL] `lock_model_for_training()` — use-after-free**
Returns raw pointer + holds model mutex, but releases registry shared_lock. Between return and `unlock_model()`, `remove_model()` can destroy both the model and its mutex.

```cpp
// FIX: ModelGuard must hold registry shared_lock
class ModelGuard {
    std::shared_lock<std::shared_mutex> reg_lock_;
    SharedRegistry& reg_;
    ConceptId cid_;
    MicroModel* model_;
public:
    ModelGuard(SharedRegistry& reg, ConceptId cid)
        : reg_lock_(reg.mtx_), reg_(reg), cid_(cid),
          model_(nullptr) {
        model_ = reg_.reg_.get_model(cid);
        if (model_) reg_.model_mutexes_.at(cid).lock();
    }
    ~ModelGuard() { if (model_) reg_.model_mutexes_.at(cid_).unlock(); }
    // ...
};
```

**[HIGH] `model_mutexes_` — `std::mutex` not movable, same map-rehash UB as SharedSTM.**
Use `std::unique_ptr<std::mutex>`.

**[MEDIUM] `ensure_models_for(const LongTermMemory&)` — not safe if LTM is shared concurrently.

---

### 11. `shared_embeddings.hpp` — Score: 7/10

**[HIGH] Returns `const Vec10&` — dangling reference after lock release**

```cpp
// FIX: Return by value
Vec10 get_context_embedding(const std::string& name) {
    { std::shared_lock lock(mtx_);
      if (em_.has_context(name)) return em_.get_context_embedding(name); }
    std::unique_lock lock(mtx_);
    return em_.get_context_embedding(name);
}
```

**[HIGH] `*_mut()` methods return mutable references — unprotected after lock release.**
Consider removing these or using a callback pattern.

---

### 12. `lock_hierarchy.hpp` — Score: 8/10

**[MEDIUM] Not integrated with SharedLTM/SharedSTM — exists as standalone utility.
**[MEDIUM] Only active in `BRAIN19_DEBUG` — no production safety.
**[LOW] `thread_local` in inline function — verify ODR across TUs.

✅ Good RAII guards. Clean design.

---

### 13. `deadlock_detector.hpp` — Score: 8/10

**[MEDIUM] Global mutex creates contention bottleneck in debug builds.
**[MEDIUM] `log_` grows unbounded — add rotation or max size.
**[MEDIUM] Not wired into any concurrent wrapper.

✅ Correct DFS cycle detection.

---

## Priorisierte Fix-Liste

### 🔴 CRITICAL (fix immediately)
| # | Issue | File |
|---|---|---|
| 1 | `shared_mutex`/`mutex` in `unordered_map` — not movable, rehash = UB | `shared_stm.hpp`, `shared_registry.hpp` |
| 2 | `lock_model_for_training` — use-after-free on model removal | `shared_registry.hpp` |
| 3 | Double WAL entry in `invalidate_concept` | `persistent_ltm.cpp` |

### 🟠 HIGH (fix before production)
| # | Issue | File |
|---|---|---|
| 4 | `retrieve_concept` data race via `const_cast` under shared_lock | `persistent_ltm.cpp` |
| 5 | Dangling references from `SharedEmbeddings` getters | `shared_embeddings.hpp` |
| 6 | CRC32 table init data race | `wal.cpp` |
| 7 | `get_context_mutex()` unprotected map access | `shared_stm.hpp` |
| 8 | WAL torn writes — use `writev()` | `wal.cpp` |
| 9 | StringPool `uint32_t` overflow at 4GB | `string_pool.hpp` |

### 🟡 MEDIUM (next iteration)
| # | Issue | File |
|---|---|---|
| 10 | StringPool `fstat` unchecked | `string_pool.hpp` |
| 11 | `record()` bounds check against capacity vs record_count | `persistent_store.hpp` |
| 12 | STM snapshot missing checksum | `stm_snapshot.cpp` |
| 13 | Lock hierarchy not integrated with wrappers | `lock_hierarchy.hpp` |
| 14 | Deadlock detector unbounded log | `deadlock_detector.hpp` |
| 15 | Config setter race in SharedSTM | `shared_stm.hpp` |

### 🟢 LOW (nice-to-have)
| # | Issue | File |
|---|---|---|
| 16 | Explicit move deletion on non-movable types | multiple |
| 17 | Endianness portability in snapshots | `stm_snapshot.cpp` |
| 18 | `MAP_POPULATE` hint for mmap | `persistent_store.hpp` |
| 19 | String deduplication | `string_pool.hpp` |
| 20 | `now_epoch_us()` as free function | `persistent_ltm.hpp` |
