# Brain19 Refactor — Executive Summary

**Datum:** 2026-02-12 (aktualisiert: Ollama-Entscheidung)
**Basis:** REFACTOR_REVIEW.md, INTEGRATION_PLAN.md, RISK_MITIGATION_PLAN.md
**Zweck:** Nüchterne Bewertung für den Entwickler. Keine Schönfärberei.

---

## 1. Ziel des Refactors

Brain19 soll von einem Spreading-Activation-System zu einer Dual-Mode-Architektur erweitert werden: **Global Dynamics** (Hintergrund-Aktivierung via bestehendem CognitiveDynamics) + **FocusCursor** (gezielte Graph-Traversierung mit MicroModel-gestützter Kantenbewertung). Zusätzlich soll eine **KAN-basierte Language Engine** (BPE-Tokenizer, KAN-Encoder/Decoder, SemanticScorer, FusionLayer) eigene Textgenerierung ermöglichen — ohne externe LLMs wie Ollama. Das Ziel ist ein System, das selbstständig Wissensgraph-Ketten traversiert und daraus deutsche Sätze generiert, komplett on-device mit ~1M Parametern.

---

## 2. Kritische Findings — Top 5 Probleme

Der Refactor-Plan wurde **ohne detaillierte Code-Analyse** geschrieben. Das ist das Kernproblem.

### #1: Phantom-APIs und falsche Struct-Namen
Der Plan referenziert `Concept` (heißt `ConceptInfo`), `Relation` (heißt `RelationInfo`), `source_id`/`target_id` (heißen `source`/`target`), `VecN` (existiert nicht, es gibt nur `Vec10`), und `get_concept_embedding()` (existiert nicht). **14 falsche Bezeichner** wurden im Review identifiziert, davon hängen Kernalgorithmen (`compute_weight()`, `accumulate_context()`) von der nicht-existenten `get_concept_embedding()` ab.

### #2: "Keine Breaking Changes" ist falsch
§8 des Refactor-Plans behauptet "only extensions, no breaking changes". Tatsächlich: `ConceptInfo` und `RelationInfo` haben `= delete` Default-Konstruktoren. Jede Feldänderung betrifft Persistence (WAL, Checkpoints, PersistentLTM). **Rettungsanker:** Die `_reserved`-Bytes in PersistentRecords reichen exakt für die neuen Felder (32B für ConceptInfo, 24B für RelationInfo), sodass kein Format-Breaking entsteht. Aber der Plan wusste das nicht.

### #3: ThinkingPipeline und CognitiveDynamics — Koexistenz ungeklärt
Der Plan führt `Brain19ControlLoop` und `GlobalDynamicsOperator` ein, erwähnt aber die bestehende 10-Schritte `ThinkingPipeline` **kein einziges Mal**. Ebenso ist unklar, ob `CognitiveDynamics` ersetzt, gewrappt oder parallel betrieben wird. Der Integration Plan löst das pragmatisch (ThinkingPipeline erweitern statt ersetzen, GDO als CognitiveDynamics-Wrapper), aber diese Entscheidungen standen im Originalplan nicht.

### #4: KAN-Decoder wird wahrscheinlich Müll generieren
Mit ~134K Parametern im Decoder und Hidden-State Vec10 (10 Dimensionen!) ist autoregressive Textgenerierung in akzeptabler Qualität extrem unwahrscheinlich. Der Risk Mitigation Plan adressiert das mit einem 3-Stufen-Fallback (Template → Hybrid → Pure Decoder), aber **realistisch wird das System lange auf Stufe 1 (Templates) bleiben**. Die Language Engine ist das ambitionierteste und riskanteste Teilprojekt.

### #5: Trainingsdaten-Problem
Der Plan sieht ~500 manuell kuratierte QA-Paare vor. Das reicht nicht. Der Mitigation-Plan schlägt synthetische Generierung aus dem Wissensgraph vor (~18K Paare aus 1000 Concepts), was clever ist — aber die Qualität synthetischer Daten für Sprachgenerierung ist fragwürdig. Overfitting auf Templates ist vorprogrammiert.

---

## 3. Architektur-Änderungen

### Wird HINZUGEFÜGT (neu)

| Komponente | Beschreibung | Dateien |
|-----------|-------------|--------|
| `backend/cursor/` | FocusCursor, FocusCursorManager, GoalState, Termination, Conflict Resolution | 8 neue Dateien |
| `KANTraversalPolicy` | KAN-gesteuerte Cursor-Policy (wraps bestehenden KANAdapter) | 2 Dateien |
| `GlobalDynamicsOperator` | CognitiveDynamics-Wrapper für Hintergrund-Aktivierung + Inhibition | 2 Dateien |
| `ConceptEmbeddingStore` | Per-Concept Embeddings (in EmbeddingManager integriert) | Erweiterung |
| `backend/hybrid/` (Language) | BPETokenizer, KANEncoder, KANDecoder, SemanticScorer, FusionLayer, KANLanguageEngine | 14 neue Dateien |
| Quality/Safety | QualityGate, MicroModelGuard, SyntheticDataGenerator | 6 Dateien |

### Wird ENTFERNT

| Komponente | Grund |
|-----------|-------|
| **Ollama komplett** | Entscheidung Felix: Wird vollständig entfernt, kein Fallback. `OllamaClient`, `OllamaMiniLLM`, `ChatInterface` werden gelöscht. Template-Engine braucht kein LLM. |
| `Brain19ControlLoop` (geplant) | Wird NICHT gebaut — ThinkingPipeline wird stattdessen erweitert |
| `ReasoningLayer` (aus Language Engine Design) | Wird eliminiert — FocusCursorManager übernimmt |

### Wird GEÄNDERT (erweitert)

| Komponente | Änderung |
|-----------|---------|
| `ConceptInfo` | +4 Felder (activation_score, salience_score, structural/semantic_confidence) |
| `RelationInfo` | +3 Felder (dynamic_weight, inhibition_factor, structural_strength) |
| `PersistentConceptRecord` | 4 doubles aus _reserved (128B bleibt gleich) |
| `PersistentRelationRecord` | 3 doubles aus _reserved (64B bleibt gleich) |
| `CuriosityTrigger` | +priority Feld, +3 TriggerTypes |
| `CuriosityEngine` | +analyze_confidence_gaps(), +pending queue |
| `ThinkingPipeline` | +FocusCursor-Step (2.5), +cursor_result in ThinkingResult |
| `SystemOrchestrator` | +GlobalDynamicsOperator, +KANTraversalPolicy, +KANLanguageEngine |
| `RelationType` enum | +5 neue Types (PRODUCES, REQUIRES, USES, SOURCE_OF, HAS_PART) |
| `EmbeddingManager` | +concept_embeddings_ Map, +get/store/compute_concept_embedding() |

---

## 4. Risiken + Mitigationen

| Risiko | Impact | Mitigation | Status |
|--------|--------|-----------|--------|
| KAN-Decoder generiert Müll (30% Wahrscheinlichkeit) | **Hoch** | QualityGate mit 3-Stufen-Fallback (Template → Hybrid → Pure) | Geplant, Code entworfen |
| Trainingsdaten zu wenig (40%) | **Hoch** | SyntheticDataGenerator aus Wissensgraph (~18K Paare) | Geplant, Code entworfen |
| Training korrumpiert MicroModels (15%) | **Kritisch** | MicroModelGuard mit Snapshot/Rollback + Integrity Probes | Geplant, Code entworfen |
| Vec10 zu klein für Query-Encoding | Mittel | Encoder intern ℝ⁶⁴, nur Output Vec10; Dual-Encoding als Fallback | Geplant |
| 8K BPE-Vocab zu klein für Deutsch (20%) | Mittel | VocabDiagnostics + expand_vocab() auf 16K | Geplant |
| FocusCursor-Chain zu kurz für komplexe Fragen | Mittel | Adaptive Depth + Branch-Merge | Geplant |
| KAN-Policy braucht nicht-existente Trainingsdaten | Mittel | Hardcoded Policy zuerst, KAN-Policy später | Offen |
| RelationType-Erweiterung bricht Serialisierung | Hoch | Separate Map statt Fixed-Size-Array | Designentscheidung offen |
| Thread-Safety: FocusCursor nutzt LTM& direkt | Hoch | Immer unter subsystem_mtx_ laufen | Architektur-Constraint |
| Phantom-APIs im Refactor-Plan | Mittel | Integration Plan korrigiert alle 14 Bezeichner | ✅ Erledigt |

---

## 5. Aufwand — Realistische Gesamtschätzung

Basis: Abendarbeit, ~2-3 Stunden pro Session.

| Phase | Inhalt | Aufwand | Kumulativ |
|-------|--------|---------|-----------|
| **Phase 0** | Vorbereitungen (ConceptEmbeddings, RelationType) | 1.0 Tage | 1.0d |
| **Phase 1** | ConceptInfo + RelationInfo erweitern | 0.75 Tage | 1.75d |
| **Phase 2** | Persistence-Migration (Records, PersistentLTM) | 1.25 Tage | 3.0d |
| **Phase 3** | FocusCursor + Manager (Kern des Refactors) | 4.5 Tage | 7.5d |
| **Phase 4** | Termination + Conflict Resolution | 0.75 Tage | 8.25d |
| **Phase 5** | CuriosityEngine API-Erweiterung | 1.0 Tage | 9.25d |
| **Phase 6** | KANTraversalPolicy | 1.5 Tage | 10.75d |
| **Phase 7** | GlobalDynamicsOperator | 1.5 Tage | 12.25d |
| **Phase 8** | ThinkingPipeline + SystemOrchestrator Integration | 2.5 Tage | 14.75d |
| **Tests 0-8** | Unit + Integration Tests (Cursor, GoalState) | 4.0 Tage | 18.75d |
| | | | |
| **Phase 9** | Language Engine komplett | 16.75 Tage | 35.5d |
| **Mitigationen** | QualityGate, Guard, SyntheticData, Tests | 10.0 Tage | 45.5d |

**Zusammenfassung:**
- **Nur Cursor + Graph (Phase 0-8):** ~19 Abende (≈ 4 Wochen)
- **+ Language Engine (Phase 9):** ~35.5 Abende (≈ 7-8 Wochen)
- **+ Alle Mitigationen:** ~45.5 Abende (≈ 9-10 Wochen)

**Vergleich zum Originalplan:** 11-14 Tage geschätzt → real 19-46 Tage je nach Scope. **Faktor 1.7x bis 3.3x Unterschätzung.**

---

## 6. Empfehlung — Reihenfolge

### Sofort starten (Phase 0-3): FocusCursor ist der Kern
1. **ConceptEmbeddingStore** in EmbeddingManager — ohne das ist kein Cursor testbar
2. **ConceptInfo/RelationInfo** erweitern — nutzt _reserved-Bytes, kein Format-Breaking
3. **FocusCursor + Manager** — das ist der eigentliche Wert des Refactors
4. **Tests schreiben** — bevor irgendetwas anderes dazukommt

### Danach (Phase 4-8): Alles um den Cursor herum
5. GoalState, Termination, CuriosityEngine-Erweiterung
6. KANTraversalPolicy — **mit hardcoded Policy starten**, KAN-Training später
7. GlobalDynamicsOperator + ThinkingPipeline Integration

### Language Engine: Template-First-Strategie (kein Ollama)
8. **Phase 9 erst starten wenn Phase 0-8 stabil läuft und getestet ist**
9. **Template-Engine sofort** — RelationType-basierte Satzmuster direkt aus FocusCursor-Ketten:
   - `CAUSES` → "X verursacht Y"
   - `IS_A` → "X ist ein Y"
   - `HAS_PROPERTY` → "X hat die Eigenschaft Y"
   - Multi-Hop: Kette ausschreiben ("X verursacht Y. Y verursacht Z.")
10. Tokenizer + Encoder parallel zu Phase 4-8 entwickeln
11. KAN-Decoder als Upgrade — wenn Qualität besser als Templates, automatisch umschalten (QualityGate)

**Progression: Template → Hybrid → Pure KAN-Decoder. Templates brauchen kein LLM.**

### Was ich NICHT empfehle:
- ❌ Alles gleichzeitig anfangen
- ❌ Language Engine vor stabilem Cursor
- ❌ KAN-Policy vor hardcoded Baseline

---

## 7. Offene Fragen — Müssen VOR Implementierung entschieden werden

| # | Frage | Optionen | Empfehlung |
|---|-------|---------|-----------|
| 1 | Concept-Embeddings: Woher? | Label-Hash-Heuristik vs. extern zuweisen vs. trainieren | Label-Hash als Start, später trainierbar |
| 2 | RelationType: Array vergrößern oder separate Map? | `NUM_RELATION_TYPES` auf 15 + Array resize vs. `unordered_map` für neue Types | Separate Map (sicherer für Serialisierung) |
| 3 | KAN-Policy: Sofort oder hardcoded first? | KAN-Training braucht Episoden-Daten die nicht existieren | Hardcoded first |
| 4 | WAL: Neue Ops für dynamische Felder? | Neue WALOpTypes vs. nur via Checkpoint | Nur Checkpoint (dynamische Felder sind flüchtig) |
| 5 | Thread-Safety-Modell für FocusCursor? | SharedLTM& vs. LTM& unter Lock | SharedLTM& ist sauberer, aber mehr Refactoring |
| 6 | EMBED_DIM: Bei 10 bleiben oder auf 16? | 10 = kompatibel, 16 = besser für Language Engine | Bei 10 bleiben, Language Engine intern mit ℝ³² arbeiten |
| 7 | ~~Ollama~~ | **Entschieden: Wird komplett entfernt.** Kein Fallback. Template-Engine funktioniert ohne LLM. | ✅ Entschieden |
| 8 | Template-Fallback: Wie detailliert? | Einfache String-Konkatenation vs. elaborierte Templates | Einfach starten, iterieren |
| 9 | Vocab-Größe: 8K oder gleich 16K? | 8K = weniger Params, 16K = besser für Deutsch | 8K starten, VocabDiagnostics entscheidet |
| 10 | MicroModel Joint Fine-Tuning (Stage 3): Überhaupt machen? | Risiko der Korrumpierung vs. bessere Integration | Überspringen, MicroModels frozen lassen |

---

## Fazit

Der Refactor beschreibt eine **gute Zielarchitektur**, aber der Weg dorthin ist deutlich komplexer als im Originalplan dargestellt. Der **Integration Plan** hat die meisten Probleme des Reviews gelöst (korrigierte APIs, Persistence-Strategie via _reserved-Bytes, ThinkingPipeline-Erweiterung statt Ersatz). Die **Risk Mitigation** ist solide mit konkretem C++-Code für jeden Risiko-Fall.

**Ehrliche Einschätzung:** Phase 0-8 (FocusCursor) ist machbar und wertvoll. Die Language Engine (Phase 9) ist ein ambitioniertes Experiment — der KAN-Decoder mit 134K Params wird lange auf Template-Stufe bleiben. Das ist okay: Templates direkt aus FocusCursor-Ketten (RelationType → Satzmuster) liefern korrekte, nachvollziehbare Antworten. Kein LLM nötig.

**Empfehlung:** Cursor zuerst, testen, stabilisieren. Template-Engine parallel aufbauen — funktioniert vom ersten Tag ohne externe Abhängigkeiten. KAN-Decoder als langfristiges Upgrade. **Ollama wird komplett entfernt** (Entscheidung Felix).
