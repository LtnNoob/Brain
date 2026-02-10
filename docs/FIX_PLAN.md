# Brain19 — Strukturierter Fix-Plan

> **Stand:** 2026-02-10  
> **Grundlage:** CODE_AUDIT_2026-02-10.md, Re-Audit Runde 2  
> **Methode:** Dependency-Analyse → Root-Cause → Fix-Gruppen → Reihenfolge  
> **Regel:** Kein Code in diesem Dokument — nur Analyse und Plan.

---

## 1. Dependency-Graph

```
Bug 1 (update_access_time nicht implementiert)
  │
  ├──→ Bug 2 (compute_recency_factor sucht alle Kontexte)
  │       │
  │       └──→ Bug 9 (recency_weight = 0.0)
  │
  └──→ Bug 3 (Salience Inkonsistenz single vs batch)
          │
          └──→ Bug 9 (Weights summieren auf 0.9)

Bug 4 (RelevanceMap Embedding-Leak) ──── steht allein

Bug 5 (KAN const_cast UB) ──── steht allein

Bug 6 (LTM update_epistemic nicht atomisch) ──── steht allein

Bug 7 (KANAdapter dangling ptr) ──── steht allein

Bug 8 (Makefile nur demo_integrated) ──── steht allein

Bug 10 (Snapshot Relations leer) ──── steht allein

Bug 11 (STM Concepts unbegrenzt) ──── steht allein

Bug 12 (ConceptId/ContextId mehrfach definiert) ──→ beeinflusst alles bei Refactoring
```

**Kern-Erkenntnis:** Bugs 1, 2, 3, 9 bilden einen zusammenhängenden Cluster. Alle anderen sind isoliert.

---

## 2. Root-Cause-Analyse

### Root Cause A: "Cognitive Dynamics Recency ist unfertig"
**Symptome:** Bug 1, Bug 2, Bug 3, Bug 9

`update_access_time()` wurde als Methode deklariert (cognitive_dynamics.hpp Zeile ~240) aber nie implementiert. Ohne funktionierende Access-Time-Tracking kann `compute_recency_factor()` keine sinnvollen Werte liefern. Der Fix von Runde 1 (Suche über focus_sets_) ist ein Workaround der über ALLE Kontexte iteriert statt den relevanten. Da das alles nicht funktioniert, wurde `recency_weight` auf 0.0 gesetzt → die Weights summieren auf 0.9 statt 1.0, und die Inkonsistenz zwischen single/batch Salience bei Connectivity-Normalisierung wird sichtbar.

### Root Cause B: "RelevanceMap BUG-C1 Fix hat Seiteneffekt"
**Symptome:** Bug 4

Der Fix von BUG-C1 erzeugt pro Target-Concept einen neuen Context-Embedding-String (`context + "_target_" + cid`). `EmbeddingManager::get_context_embedding()` cached diese per Name in `context_embeddings_` Map — bei N Konzepten entstehen N neue Einträge pro compute()-Aufruf, die nie aufgeräumt werden.

### Root Cause C: "KAN nutzt numerische statt analytische Gradienten"
**Symptome:** Bug 5

`KANNode::gradient()` mutiert temporär `coefficients_` per `const_cast` um finite differences zu berechnen. Da die Methode `const` ist, ist das UB. Bei paralleler Nutzung = Race Condition. Die eigentliche Lösung sind analytische Gradienten (∂f/∂c_i = B_i(x)), da die B-Spline Basisfunktionen bereits implementiert sind.

### Root Cause D: "EpistemicMetadata hat deleted assignment"
**Symptome:** Bug 6

`ConceptInfo::operator=` ist gelöscht wegen `EpistemicMetadata`. `update_epistemic_metadata()` arbeitet darum herum mit erase+re-insert. Bei OOM zwischen erase und emplace = Datenverlust. Die Architektur-Entscheidung (deleted assignment) ist korrekt — das Workaround nicht.

### Root Cause E: "Ownership-Mismatch bei KAN-Integration"
**Symptome:** Bug 7

`KANAdapter::train_kan_module()` erstellt `shared_ptr<KANModule>` mit No-Op-Deleter auf `unique_ptr`-verwalteten Speicher. Wenn `FunctionHypothesis` den `KANAdapter` überlebt (oder das Modul per `destroy_kan_module()` gelöscht wird), wird der Pointer dangling.

---

## 3. Fix-Gruppen

### Gruppe 1: Cognitive Dynamics Salience-Cluster (Bugs 1, 2, 3, 9)
**Zusammenhang:** Alle vier Bugs sind Facetten desselben unfertigen Features.
**Muss zusammen gefixt werden** weil:
- Bug 1 fixen ohne Bug 9 zu fixen → recency_weight bleibt 0.0, Feature wirkungslos
- Bug 3 fixen ohne Bug 1 → Normalisierung stimmt, aber Recency bleibt kaputt
- Bug 9 fixen ohne Bug 1+2 → Weights summieren auf 1.0 aber Recency liefert Müll

### Gruppe 2: RelevanceMap Memory Leak (Bug 4)
**Isoliert.** Unabhängig von allem anderen.

### Gruppe 3: KAN UB-Fix (Bug 5)
**Isoliert.** Rein interne KAN-Angelegenheit.

### Gruppe 4: LTM Atomizität (Bug 6)
**Isoliert.** Rein interne LTM-Angelegenheit.

### Gruppe 5: KANAdapter Ownership (Bug 7)
**Isoliert.** Betrifft nur KAN-Adapter Interface.

### Gruppe 6: Build-System (Bug 8)
**Isoliert.** Aber Voraussetzung um alle anderen Fixes testen zu können.

### Gruppe 7: Snapshot Completeness (Bug 10)
**Isoliert.**

### Gruppe 8: STM Garbage Collection (Bug 11)
**Isoliert.**

### Gruppe 9: Type-Alias Zentralisierung (Bug 12)
**Isoliert.** Beeinflusst viele Files, sollte zuletzt kommen.

---

## 4. Reihenfolge

```
Phase 1: Bug 8  (Makefile)          — Voraussetzung: Kompilieren + Testen
Phase 2: Bug 6  (LTM Atomizität)    — Safety-Critical, isoliert, klein
Phase 3: Bug 5  (KAN const_cast)    — UB entfernen, isoliert
Phase 4: Bug 7  (KANAdapter Ptr)    — Use-after-free entfernen, isoliert
Phase 5: Bugs 1+2+3+9 (Salience)   — Größter Cluster, braucht Sorgfalt
Phase 6: Bug 4  (Embedding Leak)    — Braucht Design-Entscheidung
Phase 7: Bug 11 (STM GC)           — Wichtig für Langzeitbetrieb
Phase 8: Bug 10 (Snapshot)          — Nice-to-have
Phase 9: Bug 12 (Type-Alias)       — Refactoring, niedrig-prioritär
```

**Begründung:**
- Phase 1 zuerst: Ohne Build kann nichts getestet werden
- Phase 2-4: Isolierte Safety-Fixes, schnelle Wins
- Phase 5: Der Kern-Cluster, braucht die meiste Arbeit
- Phase 6-9: Absteigend nach Impact

---

## 5. Detaillierte Fix-Beschreibungen

---

### Phase 1: Bug 8 — Makefile-Targets

**Was ändern:**
- `backend/Makefile`: Fehlende Source-Gruppen hinzufügen (micromodel/*.cpp, cognitive/*.cpp, understanding/*.cpp, llm/*.cpp)
- Targets für alle Demos definieren (demo_integrated, demo_epistemic_complete, demo_understanding_layer, demo_chat, test_*)
- `all` Target das alle Targets baut

**Was NICHT ändern:**
- Compiler-Flags (sind gut)
- Grundstruktur des Makefile

**Seiteneffekte:** Keine. Rein additiv.

**Test:** `make all` kompiliert ohne Fehler. Jedes Target ist einzeln baubar.

**Aufwand:** Klein (30 Min)

**Risiko:** ⚪ Kein Risiko

---

### Phase 2: Bug 6 — LTM update_epistemic_metadata() Atomizität

**Was ändern:**
- `backend/ltm/long_term_memory.cpp`: `update_epistemic_metadata()` Zeile 45-67
- Statt erase+emplace: In-place Update per placement-new auf dem Value im Map-Iterator
- Konkreter Ansatz: `ConceptInfo` braucht eine `update_epistemic(EpistemicMetadata)` Methode die intern placement-new nutzt, OHNE den Map-Eintrag zu löschen

**Alternative (sicherer):** Erst neues ConceptInfo allozieren, dann im Map den alten Wert ersetzen per move. Bei OOM der Allokation bleibt der alte Eintrag intakt. Konkret: Neues ConceptInfo per Konstruktor erzeugen, dann move-assign in den Map-Slot (ConceptInfo hat move-ctor).

**Was NICHT ändern:**
- `EpistemicMetadata` deleted assignment (Architektur-Entscheidung, korrekt)
- `ConceptInfo` deleted default ctor (korrekt)
- Die API von `update_epistemic_metadata()`

**Seiteneffekte:** Minimal, nur interne Änderung.

**Test:**
1. Concept erstellen, Epistemic updaten, verifizieren dass alter State weg und neuer da ist
2. Update mit gleichem Wert → Concept unverändert vorhanden

**Aufwand:** Klein bis Mittel

**Risiko:** 🟡 Mittel — Berührt Kern-Datenstruktur, aber isoliert

---

### Phase 3: Bug 5 — KAN const_cast UB

**Was ändern:**
- `backend/kan/kan_node.cpp`: `gradient()` Methode (Zeile 56-70)
- `backend/kan/kan_node.hpp`: `gradient()` Signatur

**Konkreter Ansatz (analytische Gradienten):**
∂f/∂c_i = B_i(x) — der Gradient bezüglich des i-ten Koeffizienten ist einfach der Wert der i-ten Basisfunktion an Stelle x. `basis_function(i, x, 3)` ist bereits implementiert.

Die gesamte gradient()-Methode wird: Für jeden Koeffizienten i: grad[i] = basis_function(i, x, 3). Keine Mutation, kein const_cast.

**Zusätzlich:** Rand-Bug fixen in `cox_de_boor()`: `x < knots_[i+1]` → `x <= knots_[i+1]` für das letzte Intervall, sonst ergibt x=1.0 immer 0.0.

**Was NICHT ändern:**
- `evaluate()` Logik
- `basis_function()` / `cox_de_boor()` Kern-Rekursion
- Knot-Vektor Initialisierung

**Seiteneffekte:** Gradient-Werte ändern sich numerisch leicht. Training wird schneller.

**Test:**
1. `gradient(0.5)` liefert Vektor mit mind. einem Nicht-Null-Eintrag
2. `gradient(1.0)` liefert Nicht-Null (Rand-Bug Fix)
3. Numerischer Vergleich: analytisch ≈ alte finite-difference (Toleranz 1e-5)

**Aufwand:** Klein (1h)

**Risiko:** ⚪ Kein Risiko

---

### Phase 4: Bug 7 — KANAdapter Dangling Pointer

**Was ändern:**
- `backend/adapter/kan_adapter.cpp`: `train_kan_module()` Zeile 32-37
- `backend/adapter/kan_adapter.hpp`: `KANModuleEntry::module` von `unique_ptr` auf `shared_ptr` umstellen

**Was NICHT ändern:**
- `FunctionHypothesis` Interface
- `KANModule` selbst

**Seiteneffekte:** `destroy_kan_module()` entfernt nur aus der Map. Tatsächliche Freigabe wenn letzter shared_ptr weg.

**Test:**
1. `train_kan_module()` aufrufen, FunctionHypothesis erhalten
2. `destroy_kan_module()` aufrufen
3. Auf FunctionHypothesis evaluieren → kein Crash

**Aufwand:** Klein (30 Min)

**Risiko:** ⚪ Kein Risiko

---

### Phase 5: Bugs 1+2+3+9 — Salience-Cluster

#### Phase 5a: Bug 1 — update_access_time() implementieren

**Was ändern:**
- `backend/cognitive/cognitive_dynamics.cpp`: Implementierung hinzufügen

**Konkreter Ansatz:** In `focus_sets_[context]` nach `cid` suchen. Wenn gefunden: `last_accessed_tick = current_tick_`. Wenn nicht: optional neuen FocusEntry erstellen.

**Aufwand:** Klein (15 Min)

#### Phase 5b: Bug 2 — compute_recency_factor() Context-Parameter

**Was ändern:**
- `backend/cognitive/cognitive_dynamics.hpp`: Signatur erweitern um `ContextId context`
- `backend/cognitive/cognitive_dynamics.cpp`: Nur im relevanten Context suchen

Statt `for (const auto& [ctx, focus_set] : focus_sets_)` → `auto it = focus_sets_.find(context)`.

**Seiteneffekte:** 3 interne Aufrufer müssen Parameter übergeben — alle haben `context` bereits.

**Aufwand:** Klein (30 Min)

#### Phase 5c: Bug 3 — Salience Inkonsistenz single vs batch

**Problem:** Single-Version: `max_conn = max(1, eigene_count)` → Connectivity immer 1.0. Batch: max über alle → relativ normalisiert.

**Empfehlung:** Single-Version bekommt optionalen `max_connectivity` Parameter (Default 0 = self-normalize). Dokumentieren.

**Aufwand:** Klein (30 Min)

#### Phase 5d: Bug 9 — Weights auf 1.0

**Was ändern:** `backend/cognitive/cognitive_config.hpp`: `recency_weight` von 0.0 auf 0.1. NUR nach Fix von 5a+5b!

**Aufwand:** Klein (5 Min)

#### Phase 5 Test:

1. Konzepte A (5 Rels), B (1 Rel), C (3 Rels) erstellen
2. Single vs Batch Salience vergleichen → konsistent
3. Focus + Ticks → Recency-Unterschiede messbar
4. Weights: 0.4 + 0.3 + 0.2 + 0.1 = 1.0

**Gesamtaufwand Phase 5:** Mittel (~3h)

**Risiko:** 🟡 Mittel — Signaturänderung, 4 koordinierte Fixes

---

### Phase 6: Bug 4 — RelevanceMap Embedding Memory Leak

**Was ändern:**
- `backend/micromodel/relevance_map.cpp`: `compute()` Methode

**Ansatz:** Eine Zeile ändern. Statt `embeddings.get_context_embedding(...)` → `embeddings.make_context_embedding(...)`. Letztere ist bereits public+const, erzeugt temporäres Embedding ohne Cache.

**Test:** 10x `compute()` → `get_context_names().size()` wächst nicht

**Aufwand:** Klein (15 Min) | **Risiko:** ⚪ Safe

---

### Phase 7: Bug 11 — STM Concepts unbegrenzt

**Was ändern:** `backend/memory/stm.cpp`: In `decay_all()` nach Decay Concepts unter Threshold entfernen.

**Test:** 100 Concepts aktivieren, 100x decay → weniger als 100 aktiv

**Aufwand:** Klein (30 Min) | **Risiko:** ⚪ Safe

---

### Phase 8: Bug 10 — Snapshot Relations leer

**Was ändern:** `backend/snapshot_generator.cpp`: Active Relations abfragen und in JSON serialisieren.

**Test:** Snapshot nach Spreading Activation → Relations nicht-leer

**Aufwand:** Klein (30 Min) | **Risiko:** ⚪ Safe

---

### Phase 9: Bug 12 — ConceptId/ContextId Zentralisierung

**Was ändern:** Neue `backend/common/types.hpp`, alle Headers umstellen (5+ Files).

**Test:** `make all` kompiliert

**Aufwand:** Klein (30 Min) | **Risiko:** ⚪ Safe

---

## 6. Risiko-Bewertung

| Phase | Bug(s) | Risiko | Begründung |
|-------|--------|--------|------------|
| 1 | 8 (Makefile) | ⚪ Safe | Rein additiv |
| 2 | 6 (LTM Atomizität) | 🟡 Mittel | Berührt Kern-Datenstruktur |
| 3 | 5 (KAN UB) | ⚪ Safe | Mathematisch beweisbar |
| 4 | 7 (Dangling Ptr) | ⚪ Safe | Klares Ownership-Problem |
| 5 | 1,2,3,9 (Salience) | 🟡 Mittel | 4 koordinierte Fixes |
| 6 | 4 (Embedding Leak) | ⚪ Safe | Eine Zeile |
| 7 | 11 (STM GC) | ⚪ Safe | Additiver Cleanup |
| 8 | 10 (Snapshot) | ⚪ Safe | Additiver Output |
| 9 | 12 (Type-Alias) | ⚪ Safe | Mechanisch |

---

## 7. Geschätzter Gesamtaufwand

| Phase | Aufwand | Kumulativ |
|-------|---------|-----------|
| Phase 1 | 30 Min | 30 Min |
| Phase 2 | 1h | 1h 30 |
| Phase 3 | 1h | 2h 30 |
| Phase 4 | 30 Min | 3h |
| Phase 5 | 3h | 6h |
| Phase 6 | 15 Min | 6h 15 |
| Phase 7 | 30 Min | 6h 45 |
| Phase 8 | 30 Min | 7h 15 |
| Phase 9 | 30 Min | 7h 45 |

**Gesamt: ~2 Abende** (je 4h) für alle 12 Bugs.

---

*Erstellt 2026-02-10 basierend auf vollständiger Source-Code-Analyse aller betroffenen Files.*
