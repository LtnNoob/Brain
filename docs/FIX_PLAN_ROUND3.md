# Fix-Plan Re-Audit Runde 3

**Erstellt:** 2026-02-10  
**Status:** Offen  
**Autor:** Architektur-Analyse (automatisiert)

---

## 1. Dependency-Analyse

Die 7 Bugs sind **weitgehend unabhängig** voneinander. Keine kausalen Abhängigkeiten.

| Bug | Abhängig von | Blockiert |
|-----|-------------|-----------|
| #1 ODR TrainingConfig/Result | — | — |
| #2 Placement-new operator= | — | — |
| #3 hash_relation Overflow | — | — |
| #4 Embedding Asymmetrie | — | — |
| #5 curl_global_init | — | — |
| #6 Doppelter #include | — | — |
| #7 Switch incomplete | — | — |

**Fazit:** Alle Fixes können parallel oder in beliebiger Reihenfolge umgesetzt werden.

---

## 2. Root-Cause-Analyse

| Root Cause | Betroffene Bugs |
|------------|----------------|
| Copy-Paste bei Struct-Design zwischen Modulen | #1 |
| Workaround für deleted operator= statt Design-Überdenken | #2 |
| Fehlende Berücksichtigung von Wertebereichen (ConceptId > 32 bit) | #3 |
| Off-by-one in Mathematik (modulo-basiertes Mapping) | #4 |
| Keine RAII/Singleton-Absicherung für globale Ressource | #5 |
| Flüchtigkeit / fehlender Linter | #6, #7 |

**Gemeinsame Muster:**
- **#1, #2:** Beide entstehen durch Design-Entscheidungen rund um EpistemicMetadata's gelöschte Operatoren. Das rigide Immutabilitäts-Design erzwingt Workarounds an anderer Stelle.
- **#6, #7:** Beide sind typische Aufmerksamkeitsfehler, keine Architektur-Probleme.

---

## 3. Fix-Details pro Bug

### Bug #1: ODR-Verletzung TrainingConfig / TrainingResult

**Schwere:** HIGH  
**Aufwand:** Mittel (ca. 1–2h)

**Ist-Zustand:**
- `kan/kan_module.hpp` definiert `brain19::TrainingConfig` mit Feldern: `max_iterations`, `learning_rate`, `convergence_threshold`, `verbose`
- `micromodel/micro_model.hpp` definiert `brain19::TrainingConfig` mit Feldern: `learning_rate`, `max_epochs`, `convergence_threshold`, `adam_beta1/2`, `epsilon`
- Beide im Namespace `brain19`, beide in Header-Files → ODR-Verletzung beim Linken
- Dasselbe für `TrainingResult` (unterschiedliche Felder)
- Aktuell includiert keine TU beide Header gleichzeitig → Compiler meldet nichts, aber Linker darf UB erzeugen

**Was ändern:**
- **Option A (empfohlen):** Sub-Namespaces → `brain19::kan::TrainingConfig` und `brain19::micro::TrainingConfig`
- **Option B:** Umbenennen → `KanTrainingConfig` / `MicroTrainingConfig`
- `adapter/kan_adapter.hpp` hat Forward-Deklaration von `TrainingConfig` → muss angepasst werden
- `demo_integrated.cpp` und `demo_epistemic_complete.cpp` prüfen

**Was NICHT ändern:**
- Keine Vereinheitlichung der Structs — unterschiedliche Semantik (KAN vs MicroModel)
- Keine Änderung an interner Trainer-Logik

**Seiteneffekte:**
- Alle Stellen die `TrainingConfig` ohne Qualifier nutzen brauchen Anpassung
- Bei Option A: `using`-Deklarationen in bestehenden .cpp-Files nötig

**Test:** TU erstellen das beide Header einbindet → kompiliert fehlerfrei. Bestehende Unit-Tests grün.

---

### Bug #2: ConceptInfo placement-new operator=

**Schwere:** HIGH  
**Aufwand:** Mittel (ca. 1h)

**Ist-Zustand:**
- `ConceptInfo::operator=` ruft `epistemic.~EpistemicMetadata()` auf, dann `new (&epistemic) EpistemicMetadata(other.epistemic)`
- Wenn Copy-Ctor wirft (z.B. `std::out_of_range` bei ungültigem Trust): `this->epistemic` bereits zerstört → Objekt in ungültigem Zustand → UB
- Move-Operator hat dasselbe Problem

**Was ändern:**
- **Bevorzugt:** `EpistemicMetadata::operator=` wieder erlauben (delete entfernen). Die "Immutabilität" ist Over-Engineering — Felder sind public und nicht const
- **Alternative:** Copy-and-Swap in `ConceptInfo::operator=` — neues Objekt komplett konstruieren, dann swappen → Exception-safe
- **Minimaler Fix:** Da EpistemicMetadata Copy-Ctor aus validem Objekt nie wirft, ist UB theoretisch unerreichbar. Trotzdem fixen für Zukunftssicherheit

**Was NICHT ändern:**
- `EpistemicMetadata() = delete` bleibt (sinnvoll)
- `ConceptInfo() = delete` bleibt (sinnvoll)

**Seiteneffekte:**
- Wenn `operator=` erlaubt: Immutabilitäts-Invariante aufgeweicht → Dokumentation anpassen
- Kein Einfluss auf andere Module

**Test:** ConceptInfo-Zuweisung mit gültigen/ungültigen Szenarien. ASan/Valgrind-Lauf.

---

### Bug #3: hash_relation 32-bit Overflow

**Schwere:** MEDIUM  
**Aufwand:** Gering (ca. 15min)

**Ist-Zustand:**
- `hash_relation(source, target)` → `(uint64_t(source) << 32) | uint64_t(target)`
- Wenn `ConceptId` 64-bit und Wert > 2³²: obere Hälfte von `source` geht verloren → Hash-Kollisionen
- Aktuell unrealistisch (wenige Konzepte), aber architekturell unsauber

**Was ändern:**
- Wenn ConceptId 32-bit: Assert + Kommentar hinzufügen → kein Bug
- Wenn ConceptId 64-bit: Richtiges Hash-Combining (z.B. Fibonacci-Hashing oder `std::hash`-Kombination)

**Was NICHT ändern:** STM-Datenstruktur bleibt gleich.

**Seiteneffekte:** Keine — reine Hash-Berechnung, keine persistierten Werte.

**Test:** Unit-Test mit ConceptIds > 2³² (falls 64-bit). Bestehende STM-Tests grün.

---

### Bug #4: EmbeddingManager Werte-Asymmetrie

**Schwere:** MEDIUM  
**Aufwand:** Gering (ca. 15min)

**Ist-Zustand:**
- `(mixed % 10000) / 50000.0 - 0.1` → Range [-0.1, +0.09998] statt [-0.1, +0.1]

**Was ändern:**
- Formel korrigieren, z.B.: `(double(mixed % 20001) / 10000.0 - 1.0) * 0.1` → exakte Range [-0.1, +0.1]

**Was NICHT ändern:** Seed/Hash-Mechanismus, EMBED_DIM.

**Seiteneffekte:**
- ⚠️ **Ändert deterministische Embeddings!** Bestehende persistierte Modelle werden inkompatibel
- Migration oder Versionsnummer nötig

**Test:** Statistischer Test: N Embeddings generieren, min/max prüfen.

---

### Bug #5: curl_global_init nicht thread-safe

**Schwere:** LOW  
**Aufwand:** Gering (ca. 30min)

**Ist-Zustand:**
- Each HTTP client constructor calls `curl_global_init()`, destructor `curl_global_cleanup()`
- Bei mehreren Instanzen: Race Condition + vorzeitiges Cleanup

**Was ändern:**
- `std::call_once` / `std::once_flag` für Init
- Reference-Counting für Cleanup (oder: nie cleanupen)

**Was NICHT ändern:** HTTP-Logik, `curl_easy_*` Aufrufe.

**Seiteneffekte:** Keine funktionalen.

**Test:** Mehrere HTTP-Client-Instanzen parallel erzeugen/zerstören → kein Crash unter TSan.

---

### Bug #6: Doppelter #include in snapshot_generator.hpp

**Schwere:** LOW  
**Aufwand:** Trivial (< 5min)

**Ist-Zustand:** `#include "common/types.hpp"` steht zweimal (Zeile 3 und 8).

**Was ändern:** Eine Zeile entfernen.

**Seiteneffekte:** Keine.

---

### Bug #7: Unvollständiger switch in mini_llm_factory.hpp

**Schwere:** LOW  
**Aufwand:** Gering (ca. 10min)

**Ist-Zustand:**
- Switch über `EpistemicType`: FACT, THEORY, HYPOTHESIS, SPECULATION vorhanden
- Fehlt: **DEFINITION**, **INFERENCE**
- Fällt in `default: "UNKNOWN"`

**Was ändern:**
- Cases für DEFINITION und INFERENCE hinzufügen
- Optional: `default` entfernen → Compiler warnt bei neuen Enum-Werten

**Seiteneffekte:** Keine — rein kosmetisch für LLM-Prompt.

---

## 4. Empfohlene Reihenfolge

| Schritt | Bug | Begründung |
|---------|-----|-----------|
| 1 | #6 Doppelter Include | Trivial, Warmup |
| 2 | #7 Switch incomplete | Trivial, schnell erledigt |
| 3 | #4 Embedding Asymmetrie | Einfach, aber Achtung Persistenz |
| 4 | #3 hash_relation | Einfach, erst ConceptId-Typ prüfen |
| 5 | #5 curl_global_init | Braucht etwas Design (Singleton) |
| 6 | #1 ODR-Verletzung | Namespace-Refactoring, mehrere Files |
| 7 | #2 Placement-new | Design-Entscheidung zur Immutabilität |

**Logik:** Triviales zuerst (schnelle Wins), HIGH-Bugs am Ende weil sie Design-Entscheidungen erfordern.

---

## 5. Risiko-Bewertung

| Bug | Fix-Risiko | Begründung |
|-----|-----------|-----------|
| #1 ODR | 🟡 Mittel | Namespace-Änderung berührt mehrere Files + Forward-Declarations |
| #2 Placement-new | 🟡 Mittel | Design-Entscheidung nötig; operator= erlauben ändert Invariante |
| #3 hash_relation | 🟢 Niedrig | Isolierte Funktion |
| #4 Embedding | 🟡 Mittel | Ändert deterministische Werte → Persistenz-Inkompatibilität |
| #5 curl_global | 🟢 Niedrig | Additive Änderung |
| #6 Doppelter Include | 🟢 Trivial | Zeile entfernen |
| #7 Switch | 🟢 Trivial | Cases hinzufügen |

---

## 6. Zusammenfassung

- **7 Bugs**, davon 2 HIGH, 2 MEDIUM, 3 LOW
- **Keine Abhängigkeiten** untereinander → parallele Bearbeitung möglich
- **Geschätzter Gesamtaufwand:** ~4–5 Stunden
- **Gefährlichster Fix:** #4 (Embedding) wegen Persistenz-Impact, #1 (ODR) wegen Breite
- **Wichtigster Fix:** #2 (Placement-new) — einziger Bug mit echtem UB-Potenzial zur Laufzeit
