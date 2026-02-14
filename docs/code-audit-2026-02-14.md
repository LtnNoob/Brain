# Brain19 Code-Audit Report — 2026-02-14

## Status: Offen
**Erstellt:** 2026-02-14 12:05 UTC
**Auditors:** 4× Opus Sub-Agents

---

## 🔴 KRITISCH

### 1. Segfault: Race Condition in parallelem Training
- **Datei:** `cmodel/concept_trainer.cpp` (train_all)
- **Problem:** Multi-threaded Training crasht mit Segfault (NULL-Pointer Deref in libc)
- **Ursache:** Worker-Threads greifen auf shared mutable state zu (vermutlich `make_target_embedding()`)
- **Fix:** Single-threaded als Sofortfix, dann thread-safety in EmbeddingManager

### 2. SGD statt Adam für MultiHead+KAN
- **Datei:** `cmodel/concept_model.cpp:607-613`
- **Problem:** 920 Params werden mit nacktem SGD trainiert (kein Momentum, kein adaptives LR)
- **Impact:** Training-Qualität massiv eingeschränkt
- **Fix:** Adam-State für MultiHead und FlexKAN hinzufügen

### 3. Zero-Init MultiHead → symmetrische Heads
- **Datei:** `cmodel/concept_model.hpp:47`
- **Problem:** Alle 4 Heads starten identisch bei 0 → produzieren identische Outputs
- **Fix:** Xavier/He-Init oder kleine Zufallswerte

### 4. predict() statt predict_refined()
- **Dateien:** `micromodel/relevance_map.cpp:36`, `tools/brain19_cli.cpp:314`
- **Problem:** Ignoriert Multi-Head + FlexKAN komplett, nutzt nur Bilinear-Score
- **Fix:** `predict()` → `predict_refined()`

### 5. Kein Quality Gate für unconverged Models
- **Problem:** 99.3% Models nicht konvergiert, Scores gleichwertig behandelt
- **Fix:** Convergence-Flag prüfen, unconverged Models default 0.5 oder skip

---

## 🟡 MITTEL

### 6. MicroTrainingConfig Kopplung
- **Dateien:** `cmodel/concept_model.hpp:3,77,79`, `cmodel/concept_trainer.hpp:21`
- **Problem:** cmodel/ hängt hart von micromodel/ ab, keine separaten LRs möglich
- **Fix:** Eigene ConceptTrainingConfig mit separaten LRs (bilinear 0.003, multihead 0.001, kan 0.005)

### 7. ~50 MiniLLM-Referenzen in 20+ Dateien
- **Dateien:** core/system_orchestrator, core/thinking_pipeline, understanding/understanding_layer, llm/chat_interface, etc.
- **Problem:** Veraltete API-Namen, Kommentare, Includes nach Refactoring
- **Fix:** Systematisches Rename MiniLLM → ConceptEngine/ConceptPatternEngine

### 8. Stopword-Liste Deutsch unvollständig
- **Datei:** `llm/chat_interface.cpp:23-46`
- **Fehlend:** "mir", "für", "über", "aber", "auch", "noch", "schon", "dann", "kann", "wird", "hat", "habe", "haben", "bin", "bist", "dem", "den", "des", etc.

### 9. Keine IDF-Gewichtung im Query-Parsing
- **Datei:** `llm/chat_interface.cpp`
- **Problem:** Keywords nur nach String-Länge gewichtet, nicht nach Seltenheit
- **Fix:** IDF-Map beim Init berechnen, bei LTM-Änderungen invalidieren

### 10. User-sichtbar "MiniLLMs" in Chat-Output
- **Datei:** `llm/chat_interface.cpp:722`
- **Fix:** "Semantische Analyse (MiniLLMs):" → "Semantische Analyse (ConceptModels):"

### 11. save_v4 Funktion schreibt V5 Format
- **Datei:** `cmodel/concept_persistence.hpp:30-33`
- **Fix:** Umbenennen zu save_v5/load_v5

### 12. LR=0.01 Default zu hoch
- **Datei:** `micromodel/micro_model.hpp:48`
- **Fix:** ConceptTrainerConfig soll eigene LR setzen (0.003)

---

## 🟢 NIEDRIG

### 13. Verwaiste Dateien (nicht im Makefile)
- test_wal_recovery.cpp, checkpoint_cli.cpp, brain19_monitor.cpp, streams/stream_monitor_cli.cpp

### 14. Tote Headers
- concurrent/deadlock_detector.hpp, concurrent/lock_hierarchy.hpp

### 15. Fehlende ConceptModel Unit-Tests
- Kein test_concept_model.cpp existiert

### 16. KANAdapter architektonisch getrennt von FlexKAN
- Info: Zwei KAN-Systeme koexistieren (standalone vs embedded) — kein Bug, aber dokumentieren

---

## Architektur-Empfehlungen

1. **common/types.hpp** erstellen: Vec10, FlexEmbedding, CoreVec, CoreMat, CORE_DIM dorthin verschieben
2. **ConceptTrainingConfig** in cmodel/: Separate LRs für Bilinear/MultiHead/KAN
3. **MiniLLM → ConceptEngine** umbenennen (Interface + alle Referenzen)
4. **micromodel/** langfristig: Nur noch Shared-Types, aktive Logik nach cmodel/
5. **RelationCategory::LOGICAL** für Meta-Relations (siehe meta-relations-plan.md)
