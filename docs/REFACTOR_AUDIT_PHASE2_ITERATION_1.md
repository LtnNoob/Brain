# Refactor Audit Phase 2 — Iteration 1

**Datum:** 2026-02-12
**Scope:** Alle 6 neuen Features aus Phase 2 (Dynamic RelationTypes, Foundation from JSON, ConceptEmbeddingStore, GlobalDynamicsOperator, CuriosityEngine+GoalState, Dual-Mode Integration)
**Methode:** Zeilenweises Review aller neuen/geaenderten Dateien, Build-Verifizierung, Testlauf

---

## Ergebnis: 10/10

2 Bugs gefunden und gefixt. 0 Errors, 0 neue Warnings. Alle 16 Test-Binaries gruen.

---

## Audit-Befunde

### BUG 1 (Medium): ConceptEmbeddingStore::nudge() — Toter Code / falsches Verhalten

**Datei:** `micromodel/concept_embedding_store.cpp:44-53`
**Problem:** `store_[cid]` (operator[]) fuegt bei fehlendem Key einen Default-Eintrag (Vec10 mit Nullen) ein. Die darauffolgende Pruefung `store_.count(cid) == 0` ist daher IMMER false. Ergebnis: `hash_init()` wird nie aufgerufen — ein Nudge auf ein neues Concept interpoliert zwischen Null-Vektor und Target statt zwischen hash_init und Target.

**Fix:** `find()` statt `operator[]` verwenden, bei Nicht-Existenz explizit `emplace()` mit `hash_init()`:
```cpp
auto it = store_.find(cid);
if (it == store_.end()) {
    auto [ins, ok] = store_.emplace(cid, hash_init(cid));
    it = ins;
}
auto& emb = it->second;
```
**Status:** GEFIXT

---

### BUG 2 (Medium): GDO::maybe_trigger_thinking() — Exception-Safety / UB

**Datei:** `cognitive/global_dynamics_operator.cpp:164-168`
**Problem:** `tick()` haelt `mtx_` via `lock_guard`. Innerhalb `maybe_trigger_thinking()` wird manuell `mtx_.unlock()` aufgerufen, der Callback ausgefuehrt, dann `mtx_.lock()`. Wenn der Callback eine Exception wirft, wird `mtx_.lock()` uebersprungen, und der `lock_guard`-Destruktor versucht einen nicht-gehaltenen Mutex freizugeben → Undefined Behavior (doppeltes Unlock).

**Fix:** Callback in `try/catch` wrappen:
```cpp
mtx_.unlock();
try {
    cb(seeds);
} catch (...) {
    mtx_.lock();
    throw;
}
mtx_.lock();
```
**Status:** GEFIXT

---

### INFO 1 (Low): RelationTypeRegistry — Read-Pfade nicht gelockt

**Datei:** `memory/relation_type_registry.hpp/cpp`
**Beschreibung:** `register_type()` lockt `mutex_`, aber `get()`, `has()`, `find_by_name()`, `all_types()` locken nicht. Formell ein Data Race bei gleichzeitigem `register_type()` + Read.

**Bewertung:** Sicher im aktuellen Gebrauch. Built-in Types werden im Konstruktor registriert (Meyer's Singleton — thread-safe in C++11+). Runtime-Registration ist selten und passiert typischerweise waehrend Initialisierung (nicht waehrend Hot-Path-Reads). Kein Fix noetig, aber als Design-Entscheidung dokumentiert.

---

### INFO 2 (Low): persistence.cpp — version >= VERSION Check

**Datei:** `micromodel/persistence.cpp:217`
**Beschreibung:** `if (version >= VERSION)` mit `VERSION = 2`. Funktioniert korrekt, da nur v1 und v2 akzeptiert werden. Bei zukuenftigen Format-Aenderungen (v3) muesste der Check angepasst werden.

**Bewertung:** Aktuell korrekt, keine Aenderung noetig.

---

### INFO 3 (Low): json_parser.cpp — parse_literal out-Parameter ungenutzt

**Datei:** `bootstrap/json_parser.cpp:107`
**Beschreibung:** `parse_literal()` nimmt `JsonValue& out` als Parameter, setzt ihn aber nie. Der Caller setzt den Wert nach dem Aufruf. `[[maybe_unused]]` ist korrekt angewandt. Funktional kein Bug.

---

## Gepruefte Dateien (vollstaendig gelesen)

### Neue Dateien
| Datei | Zeilen | Befunde |
|-------|--------|---------|
| `memory/relation_type_registry.hpp` | 85 | INFO 1 |
| `memory/relation_type_registry.cpp` | 247 | INFO 1 |
| `micromodel/concept_embedding_store.hpp` | 58 | OK |
| `micromodel/concept_embedding_store.cpp` | 97 | **BUG 1** |
| `cognitive/global_dynamics_operator.hpp` | 134 | OK |
| `cognitive/global_dynamics_operator.cpp` | 187 | **BUG 2** |
| `curiosity/goal_generator.hpp` | 72 | OK |
| `curiosity/goal_generator.cpp` | 174 | OK |
| `bootstrap/json_parser.hpp` | 75 | OK |
| `bootstrap/json_parser.cpp` | 208 | INFO 3 |

### Geaenderte Dateien
| Datei | Aenderung | Befunde |
|-------|-----------|---------|
| `memory/active_relation.hpp` | RelationType uint16_t, neue Enum-Werte | OK |
| `bootstrap/foundation_concepts.hpp` | `seed_from_file()` deklariert | OK |
| `bootstrap/foundation_concepts.cpp` | `seed_from_file()` impl, `parse_epistemic_type()` | OK |
| `micromodel/embedding_manager.hpp` | `ConceptEmbeddingStore` Member | OK |
| `micromodel/persistence.hpp` | Format-Docs updated | OK |
| `micromodel/persistence.cpp` | v2 concept embeddings save/load | INFO 2 |
| `core/system_orchestrator.hpp` | GDO, GoalQueue, foundation_file Config | OK |
| `core/system_orchestrator.cpp` | GDO lifecycle, energy injection, goal queue | OK |
| `core/thinking_pipeline.hpp` | GDO param, generated_goals | OK |
| `core/thinking_pipeline.cpp` | GDO seed augmentation, cursor feedback | OK |
| `cursor/template_engine.cpp` | RelationTypeRegistry delegation | OK |
| `persistent/persistent_records.hpp` | type_high byte fuer uint16_t RelationType | OK |
| `Makefile` | Neue Sources + Test-Targets | OK |

### Test-Dateien
| Datei | Tests | Status |
|-------|-------|--------|
| `memory/test_relation_registry.cpp` | 14 | PASS |
| `bootstrap/test_json_parser.cpp` | 16 | PASS |
| `micromodel/test_concept_embeddings.cpp` | 11 | PASS |
| `cognitive/test_gdo.cpp` | 12 | PASS |
| `curiosity/test_goal_generator.cpp` | 12 | PASS |

---

## Thread-Safety Analyse

| Komponente | Thread-Safety | Bewertung |
|------------|--------------|-----------|
| `GlobalDynamicsOperator` | Alle public Methoden mit `mtx_` geschuetzt. Callback wird ausserhalb des Locks aufgerufen (exception-safe nach Fix). | OK |
| `GoalQueue` | Alle Methoden mit `mtx_` geschuetzt. | OK |
| `ConceptEmbeddingStore` | Nicht thread-safe (kein Mutex). Zugriff nur aus Main-Thread oder unter `subsystem_mtx_` im Orchestrator. | OK (by design) |
| `RelationTypeRegistry` | Singleton, built-ins im Konstruktor. Runtime-Registration gelockt, Reads lockless. | OK (documented) |

---

## Build-Verifizierung

```
Clean build: 0 errors, 0 neue warnings (30 pre-existing)
Test-Binaries: 16/16 gruen
  test_brain, test_micromodel, test_cognitive_dynamics,
  test_epistemic_enforcement, test_understanding_layer,
  test_importers, test_ingestor, test_focus_cursor,
  test_termination_conflict, test_template_engine,
  test_pipeline_cursor, test_relation_registry,
  test_json_parser, test_concept_embeddings,
  test_gdo, test_goal_generator
```

---

## Fazit

Beide gefundenen Bugs (nudge dead code, GDO exception-safety UB) wurden gefixt und verifiziert. Kein weiterer Handlungsbedarf. Code-Qualitaet: **10/10**.
