# Brain19 - BUG-001 Closure: Epistemic Enforcement

## ✅ STATUS: BUG-001 CLOSED

**BUG-001** ("No trust differentiation / Cannot distinguish facts from speculation") wurde durch **Compile-Time-Enforcement** geschlossen.

---

## Schnellstart

```bash
cd backend

# Kompilieren des vollständigen Demos
make -f Makefile.complete

# Ausführen
./demo_epistemic_complete

# Testen
make -f Makefile.epistemic test
```

---

## Was wurde implementiert

### 1. **Epistemic Metadata (KERN-ENFORCEMENT)**

**Datei:** `backend/epistemic/epistemic_metadata.hpp`

```cpp
struct EpistemicMetadata {
    EpistemicType type;      // FACT, THEORY, HYPOTHESIS, SPECULATION
    EpistemicStatus status;  // ACTIVE, INVALIDATED, SUPERSEDED, CONTEXTUAL
    double trust;            // [0.0, 1.0]
    
    // ✓ ENFORCEMENT: Kein Default-Konstruktor
    EpistemicMetadata() = delete;
    
    // ✓ EINZIGER WEG: Alle Felder explizit
    explicit EpistemicMetadata(EpistemicType t, EpistemicStatus s, double trust);
};
```

**Was dies verhindert:**
```cpp
// ✗ COMPILE ERROR
EpistemicMetadata meta;

// ✓ EINZIG GÜLTIG
EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95);
```

---

### 2. **Long-Term Memory (LTM)**

**Dateien:** 
- `backend/ltm/long_term_memory.hpp`
- `backend/ltm/long_term_memory.cpp`

```cpp
class LongTermMemory {
public:
    // ENFORCEMENT: Kein Default-Parameter für epistemic
    ConceptId store_concept(
        const std::string& label,
        const std::string& definition,
        EpistemicMetadata epistemic  // REQUIRED - kein Bypass möglich
    );
    
    // Wissen wird NIEMALS gelöscht, nur invalidiert
    bool invalidate_concept(ConceptId id, double trust = 0.05);
};
```

**Was dies verhindert:**
```cpp
// ✗ COMPILE ERROR: Fehlt epistemic metadata
ltm.store_concept("Cat", "A mammal");

// ✓ EINZIG GÜLTIG: Explizite epistemic metadata
EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95);
ltm.store_concept("Cat", "A mammal", meta);
```

---

### 3. **Importer Rules**

**Dateien:**
- `backend/importers/wikipedia_importer.cpp`
- `backend/importers/scholar_importer.cpp`

```cpp
// CRITICAL: Dies ist nur eine SUGGESTION, KEINE Zuweisung
// Importers dürfen NICHT epistemic metadata zuweisen
// Mensch muss explizit entscheiden
proposal->suggested_epistemic_type = SuggestedEpistemicType::DEFINITION_CANDIDATE;
```

**Workflow:**
1. Importer extrahiert Text → erstellt `KnowledgeProposal`
2. Vorschlag enthält `suggested_epistemic_type` (nur Vorschlag!)
3. **Mensch reviewed** → entscheidet epistemic metadata
4. **Mensch erstellt** `EpistemicMetadata` explizit
5. **Mensch speichert** in LTM mit `store_concept()`

---

### 4. **Invalidierung (KEINE Löschung)**

```cpp
// Wissen wird NIEMALS gelöscht
ltm.invalidate_concept(phlogiston_id);

// Was passiert:
// - Status → INVALIDATED
// - Trust → < 0.2 (sehr niedrig)
// - Typ bleibt erhalten
// - Wissen bleibt abfragbar
// - Epistemische Historie erhalten
```

**Philosophie:** Keine stillen Datenverluste, Geschichte bleibt sichtbar.

---

## Kompilieren & Ausführen

### Demo 1: Vollständiges System (EMPFOHLEN)

```bash
cd backend

# Kompilieren
make -f Makefile.complete

# Ausführen
./demo_epistemic_complete
```

**Was demonstriert wird:**
- Speichern von Wissen mit expliziter epistemic metadata
- Unterscheidung FACT vs THEORY vs HYPOTHESIS vs SPECULATION
- Trust-Werte differenzieren
- Invalidierung (ohne Löschung)
- Importer-Workflow (nur Vorschläge)
- Snapshot mit epistemic metadata

**Erwartete Ausgabe:**
```
═══════════════════════════════════════════════════════
  EPISTEMIC ENFORCEMENT DEMONSTRATION COMPLETE
═══════════════════════════════════════════════════════

KEY POINTS DEMONSTRATED:
  ✓ All knowledge has explicit epistemic metadata
  ✓ Facts distinguishable from speculation
  ✓ Trust values differentiate certainty
  ✓ Invalidation preserves knowledge (no deletion)
  ✓ Importers only suggest, humans decide
  ✓ Snapshot exposes epistemic metadata

BUG-001 STATUS: CLOSED ✅
```

---

### Demo 2: Enforcement Tests

```bash
cd backend

# Kompilieren
make -f Makefile.epistemic

# Ausführen
./test_epistemic_enforcement
```

**Was getestet wird:**
- ✅ Kein Default-Konstruktor (compile-time)
- ✅ Alle Felder required (compile-time)
- ✅ Trust-Validierung (runtime)
- ✅ ConceptInfo-Enforcement
- ✅ LTM-Enforcement
- ✅ Invalidierung statt Löschung
- ✅ Importer nur Vorschläge
- ✅ Vollständiger Workflow
- ✅ Query nach epistemic type

**Erwartete Ausgabe:**
```
═══════════════════════════════════════════════════════
  BUG-001 STATUS: CLOSED
═══════════════════════════════════════════════════════

ENFORCEMENT SUMMARY:
  ✓ No default construction (compile-time)
  ✓ All fields required (compile-time)
  ✓ Trust validation (runtime)
  ✓ Importers cannot assign epistemic metadata
  ✓ LTM requires explicit epistemic metadata
  ✓ Knowledge never deleted, only invalidated
  ✓ Facts distinguishable from speculation

It is now TECHNICALLY IMPOSSIBLE to:
  • Create knowledge without epistemic metadata
  • Use implicit defaults
  • Have silent fallbacks
  • Infer epistemic state
```

---

## Enforcement-Mechanismen

### 1. Compile-Time (Gelöschte Konstruktoren)

```cpp
// Diese Zeilen führen zu COMPILE ERROR:
EpistemicMetadata meta;                      // ✗ Gelöschter default constructor
ConceptInfo concept;                         // ✗ Gelöschter default constructor
ltm.store_concept("Label", "Definition");   // ✗ Fehlt required parameter
```

### 2. Runtime (Validierung)

```cpp
// Trust-Bereich-Validierung
EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, -0.1);
// Wirft: std::out_of_range("Trust must be in [0.0, 1.0], got: -0.100000")

EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 1.5);
// Wirft: std::out_of_range("Trust must be in [0.0, 1.0], got: 1.500000")
```

### 3. Debug (Assertions)

```cpp
#ifndef NDEBUG
// INVALIDATED mit hohem Trust triggert Assertion
EpistemicMetadata meta(
    EpistemicType::HYPOTHESIS,
    EpistemicStatus::INVALIDATED,
    0.8  // Zu hoch!
);
// Assertion: "WARNING: INVALIDATED knowledge should have trust < 0.2"
#endif
```

---

## Dateistruktur

```
backend/
├── epistemic/
│   └── epistemic_metadata.hpp          ← KERN-ENFORCEMENT
│
├── ltm/
│   ├── long_term_memory.hpp            ← LTM mit Enforcement
│   └── long_term_memory.cpp
│
├── importers/
│   ├── wikipedia_importer.cpp          ← Nur Vorschläge
│   └── scholar_importer.cpp            ← Nur Vorschläge
│
├── snapshot_generator.cpp              ← Exposiert epistemic metadata
│
├── demo_epistemic_complete.cpp         ← VOLLSTÄNDIGES DEMO
├── test_epistemic_enforcement.cpp      ← 11 TESTS
│
├── Makefile.complete                   ← Build: Demo
├── Makefile.epistemic                  ← Build: Tests
│
└── BUG-001-CLOSURE.md                  ← Technische Dokumentation
```

---

## Vollständiger Workflow-Beispiel

```cpp
// 1. Externes Wissen importieren
WikipediaImporter importer;
auto proposal = importer.parse_wikipedia_text(
    "Quantum Mechanics",
    "Quantum mechanics is a fundamental theory..."
);

// ⚠ Importer weist NICHT zu, nur Vorschlag
assert(proposal->suggested_epistemic_type == 
       SuggestedEpistemicType::DEFINITION_CANDIDATE);

// 2. Mensch reviewed Vorschlag
std::cout << "Proposal: " << proposal->title << "\n";
std::cout << "Suggested: DEFINITION_CANDIDATE\n";
std::cout << "Human decides: THEORY (well-established)\n";

// 3. Mensch erstellt explizite epistemic metadata
EpistemicMetadata human_decision(
    EpistemicType::THEORY,      // Mensch entscheidet
    EpistemicStatus::ACTIVE,    // Mensch entscheidet
    0.95                        // Mensch entscheidet
);

// 4. Speichern in LTM mit epistemic metadata
LongTermMemory ltm;
ConceptId id = ltm.store_concept(
    proposal->title,
    proposal->extracted_text,
    human_decision  // REQUIRED - kein Bypass
);

// 5. Verifizieren
auto retrieved = ltm.retrieve_concept(id);
assert(retrieved->epistemic.type == EpistemicType::THEORY);
assert(retrieved->epistemic.trust == 0.95);

// 6. Nach Typ abfragen
auto theories = ltm.get_concepts_by_type(EpistemicType::THEORY);
auto facts = ltm.get_concepts_by_type(EpistemicType::FACT);
// ✓ Kann Fakten von Spekulationen unterscheiden!
```

---

## Was jetzt UNMÖGLICH ist

### ❌ Wissen ohne Metadata erstellen
```cpp
// Alle diese Zeilen sind COMPILE ERRORS:
EpistemicMetadata meta;
ConceptInfo concept;
ltm.store_concept("Label", "Definition");
```

### ❌ Implizite Defaults verwenden
- Kein "UNKNOWN" epistemic type
- Keine null trust-Werte  
- Keine stillen Fallbacks

### ❌ Enforcement umgehen
- Importers weisen nur zu → nur Vorschläge
- LTM erfordert explizite metadata → keine defaults
- Keine versteckten Parameter

---

## Breaking Changes

**JA - Absichtlich:**

Alle bestehenden Code-Stellen müssen aktualisiert werden:
- Alle `store_concept()`-Aufrufe brauchen `EpistemicMetadata`
- Keine Migration für "UNKNOWN"-Zustand
- Zwingt explizite epistemische Entscheidungen
- **Das IST das Enforcement**

---

## Snapshot-Ausgabe

**Vorher (BUG-001 OFFEN):**
```json
{
  "concepts": [
    {"id": 1, "epistemic_type": "UNKNOWN", "trust": null}
  ]
}
```

**Nachher (BUG-001 GESCHLOSSEN):**
```json
{
  "concepts": [
    {
      "id": 1,
      "label": "Cat",
      "epistemic_type": "FACT",
      "epistemic_status": "ACTIVE",
      "trust": 0.98
    },
    {
      "id": 2,
      "label": "Dark Matter",
      "epistemic_type": "HYPOTHESIS",
      "epistemic_status": "ACTIVE",
      "trust": 0.65
    },
    {
      "id": 5,
      "label": "Phlogiston Theory",
      "epistemic_type": "THEORY",
      "epistemic_status": "INVALIDATED",
      "trust": 0.05,
      "invalidated": true
    }
  ]
}
```

✅ Expliziter Typ  
✅ Expliziter Status  
✅ Expliziter Trust  
✅ Fakten unterscheidbar von Spekulation

---

## Testergebnisse

```bash
make -f Makefile.epistemic run
```

```
Test 1: No Default Construction           ✅ PASS
Test 2: All Fields Required                ✅ PASS
Test 3: Trust Validation                   ✅ PASS
Test 4: INVALIDATED Trust Warning          ✅ PASS
Test 5: ConceptInfo No Default             ✅ PASS
Test 6: ConceptInfo Requires Epistemic     ✅ PASS
Test 7: LTM Requires Epistemic             ✅ PASS
Test 8: Invalidation NOT Deletion          ✅ PASS
Test 9: Importers No Assignment            ✅ PASS
Test 10: Complete Workflow                 ✅ PASS
Test 11: Query By Epistemic Type           ✅ PASS

═══════════════════════════════════════════════════════
ALL TESTS PASSED (11/11)
═══════════════════════════════════════════════════════
```

---

## FAQ

### Q: Warum kann ich kein Wissen ohne epistemic metadata speichern?

**A:** Das ist **Absicht**. BUG-001 wurde geschlossen durch **Enforcement by Construction**. Es ist jetzt **compile-time unmöglich**, Wissen ohne explizite epistemische Entscheidung zu erstellen.

### Q: Was passiert, wenn ich alten Code kompiliere?

**A:** **COMPILE ERROR**. Sie müssen für jedes Konzept explizit `EpistemicMetadata` erstellen. Das ist ein **Breaking Change by Design**.

### Q: Kann ich "UNKNOWN" als Typ verwenden?

**A:** **NEIN**. Es gibt keinen "UNKNOWN"-Typ mehr. Sie müssen explizit entscheiden: FACT, THEORY, HYPOTHESIS, oder SPECULATION.

### Q: Kann ich Wissen löschen?

**A:** **NEIN**. Wissen wird **niemals gelöscht**, nur **invalidiert**. Verwenden Sie `ltm.invalidate_concept(id)`. Das setzt:
- Status → INVALIDATED
- Trust → < 0.2
- Typ bleibt erhalten
- Wissen bleibt abfragbar

### Q: Können Importers automatisch epistemic metadata zuweisen?

**A:** **NEIN**. Importers dürfen nur **Vorschläge** (`suggested_epistemic_type`) machen. Der **Mensch muss explizit entscheiden** und `EpistemicMetadata` erstellen.

---

## Zusammenfassung

**BUG-001 ist geschlossen durch Konstruktion.**

**Enforcement:**
- ✅ Gelöschte Default-Konstruktoren (compile-time)
- ✅ Required-Parameter ohne defaults (compile-time)
- ✅ Trust-Bereich-Validierung (runtime)
- ✅ Debug-Assertions (development-time)
- ✅ Explizite Dokumentation (immer)

**Ergebnis:**
- Fakten sind unterscheidbar von Spekulation
- Trust-Differenzierung ist erzwungen
- Jedes Wissens-Item hat epistemic metadata
- KEINE impliziten defaults
- KEINE stillen Fallbacks
- KEIN inferierter epistemischer Zustand

**Es ist jetzt TECHNISCH UNMÖGLICH**, epistemische Anforderungen zu verletzen.

---

## Kontakt

**Closure Method:** Enforcement by Construction  
**Status:** CLOSED ✅  
**Tests:** 11/11 PASSED ✅  
**Demo:** Vollständig funktional ✅

---

*README Generated: 2026-01-11*  
*BUG-001 Closure Documentation*
