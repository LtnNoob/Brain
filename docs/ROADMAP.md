# Brain19 — Priorisierte Entwicklungs-Roadmap

> **Stand:** 2026-02-10  
> **Kontext:** Solo-Entwickler (Felix), ADHS, Schule Mo-Fr 08-16 (bis 11.05.2026)  
> **Hardware:** i5-6600K + RTX 2070 (EPYC 80-Core offline verfügbar)  
> **Prinzip:** Persistence before Performance  

---

## Legende

- 🔴 **CRITICAL** — System ist ohne diesen Fix kaputt
- 🟠 **HIGH** — Kernfunktionalität systematisch falsch
- 🟡 **MEDIUM** — Wichtig, aber workaround-fähig
- 🟢 **LOW** — Nice-to-have
- ⏱️ Zeitschätzung = effektive Arbeitszeit (nicht Kalenderwochen)
- 📦 = Abgeschlossenes, pushbares Paket (ADHS-Win)

---

## Phase 0: Stabilisierung — SOFORT

> **Ziel:** Code kompiliert, Demos laufen, Kernalgorithmen geben korrekte Ergebnisse.  
> **Warum zuerst:** Alles was auf kaputtem Fundament aufbaut ist Zeitverschwendung.  
> **Geschätzt:** 3-5 Abende (je 2-3h)

### 0.1 🔴 RelevanceMap::compute() fixt (1 Abend)
- **Bug:** Target-Concept wird ignoriert → alle Scores identisch
- **Impact:** MicroModel-Inferenz komplett kaputt, Creativity-Overlay sinnlos
- **Fix:** Target-Concept-Embedding in die Berechnung einbauen
- 📦 Commit: `fix: RelevanceMap::compute() uses target concept`

### 0.2 🔴 Build-System reparieren (1 Abend)
- **Bug:** `demo_integrated.cpp` kompiliert nicht, Makefile kaputt
- **Fix:** Makefile fixen, alle Demos kompilierbar machen
- **Test:** `make all && ./demo_integrated` läuft ohne Crash
- 📦 Commit: `fix: repair Makefile, all demos compile`

### 0.3 🟠 Salience-Berechnung korrigieren (1 Abend)
- **Bug:** Recency hardcoded 0.5 (Placeholder), Connectivity=0 bei Einzelaufrufen
- **Impact:** Salience-Scores systematisch falsch → Focus Management unzuverlässig
- **Fix:** Recency aus `last_used` Timestamp berechnen, Connectivity korrekt normalisieren
- 📦 Commit: `fix: salience uses real recency + correct connectivity`

### 0.4 🟠 Understanding Layer: Echte Concept-IDs (1 Abend)
- **Bug:** Concept-IDs werden erfunden (1..N statt echte LTM-IDs)
- **Impact:** Proposals referenzieren nicht-existente Konzepte
- **Fix:** LTM-Lookup bei Concept-Referenzen, Fehler wenn ID nicht existiert
- 📦 Commit: `fix: understanding layer uses real concept IDs`

### 0.5 🟠 KAN: const_cast UB + Rand-Bug (1 Abend)
- **Bug:** `const_cast` in `KANNode::gradient()` = Undefined Behavior, Rand-Initialisierung nicht geseeded
- **Fix:** `mutable` für internen State, deterministische Seed-Strategie
- 📦 Commit: `fix: KAN remove UB const_cast, deterministic seeding`

### Phase-0-Meilenstein ✅
> Alle Demos kompilieren und laufen. RelevanceMap gibt unterschiedliche Scores. Salience-Ranking ist plausibel. Kein UB mehr.

---

## Phase 1: Persistence — Kernfeature

> **Ziel:** Brain19 überlebt Neustarts. LTM-Wissen ist persistent.  
> **Warum jetzt:** Felix' Entscheidung "Persistence before Performance". Ohne Persistenz ist jeder Testlauf flüchtig.  
> **Dependency:** Phase 0 muss fertig sein.  
> **Geschätzt:** 3-4 Wochen (Abende + Wochenenden)  
> **Referenz:** `docs/PERSISTENT_MEMORY_ARCHITECTURE.md` (Layer 1-3 Design existiert komplett)

### 1.1 Layer 1: LTM auf mmap — Basic (2 Wochen)
- `PersistentStore<T>` Template mit mmap
- `PersistentConceptRecord` + `PersistentRelationRecord` (Formate definiert in Doku)
- `StringPool` für Labels/Definitionen
- `LongTermMemory` Dual-Mode (Heap/Persistent)
- **Test:** 10.000 Concepts speichern, `kill -9`, restart, alles da
- 📦 Commit: `feat: LTM persistence via mmap`

### 1.2 Layer 1: WAL + Crash Recovery (1 Woche)
- Write-Ahead-Log für atomare Writes
- Recovery-Replay nach Crash
- **Test:** Kill während Write → kein Datenverlust
- 📦 Commit: `feat: WAL crash recovery for LTM`

### 1.3 Layer 2: STM-Snapshots (1 Woche)
- Binary Snapshot alle 30-60s (Format in Doku definiert)
- `STMSnapshotManager::create_snapshot()` / `load_latest_snapshot()`
- `ShortTermMemory::export_snapshot()` / `import_snapshot()`
- 📦 Commit: `feat: periodic STM snapshots`

### Phase-1-Meilenstein ✅
> Brain19 startet, du arbeitest damit, du killst den Prozess, du startest neu — LTM ist da, STM ist maximal 60s alt. **Das ist der Game-Changer.**

---

## Phase 2: Snapshot & MicroModel Fixes

> **Ziel:** Der Snapshot zeigt den echten Brain-State. MicroModels trainieren korrekt.  
> **Dependency:** Phase 0.  
> **Parallelisierbar mit:** Phase 1 (andere Dateien).  
> **Geschätzt:** 2-3 Abende

### 2.1 🟠 Snapshot: Fehlende 50% ergänzen (1 Abend)
- **Bug:** Snapshot zeigt nur STM-Aktivierungen, fehlt: Focus-Sets, Cognitive Tick, CuriosityEngine-State, MicroModel-Metriken
- **Fix:** `SnapshotGenerator` um alle Subsysteme erweitern
- 📦 Commit: `fix: snapshot includes full brain state`

### 2.2 🟡 KAN: Multi-Layer + Analytische Gradienten (2 Abende)
- **Bug:** Nur 1 Schicht, numerische Gradienten (langsam + ungenau)
- **Fix:** Multi-Layer-Support, analytische B-Spline-Gradienten
- **Nice-to-have:** Grid Extension für progressive Verfeinerung
- 📦 Commit: `feat: KAN multi-layer + analytical gradients`

### Phase-2-Meilenstein ✅
> Snapshot zeigt alles. KAN trainiert richtig.

---

## Phase 3: Thread-Safety Grundlagen

> **Ziel:** Subsysteme sind thread-safe, auch wenn noch single-threaded genutzt.  
> **Warum jetzt:** Persistence (Phase 1) braucht Background-Threads für STM-Snapshots. Multi-Stream (Phase 5) braucht es sowieso.  
> **Dependency:** Phase 0, idealerweise Phase 1.  
> **Geschätzt:** 2 Wochen  
> **Referenz:** `docs/MULTI_STREAM_ARCHITECTURE.md` (Wrapper-Design existiert komplett)

### 3.1 Shared-State Wrappers (1 Woche)
- `SharedLTM` (shared_mutex, read/write)
- `SharedRegistry` (shared_mutex + per-model mutex)
- `SharedEmbeddings` (shared_mutex)
- `SharedSTM` (per-context mutex)
- **Kein bestehender Code wird modifiziert** — Adapter-Pattern
- 📦 Commit: `feat: thread-safe wrappers for all subsystems`

### 3.2 Lock-Hierarchie + Tests (1 Woche)
- Deadlock-freie Lock-Reihenfolge (in Doku definiert)
- Stress-Tests mit mehreren Threads
- 📦 Commit: `test: thread-safety stress tests`

### Phase-3-Meilenstein ✅
> Alle Subsysteme können sicher von mehreren Threads genutzt werden. Fundament für Multi-Stream.

---

## Phase 4: Full Checkpoint (Layer 3)

> **Ziel:** Kompletter Brain-State als atomarer Checkpoint.  
> **Dependency:** Phase 1 (LTM persistent) + Phase 2 (Snapshot komplett).  
> **Geschätzt:** 2 Wochen

### 4.1 Checkpoint-Manager (1 Woche)
- `BrainCheckpointManager` — atomic write via temp-dir + rename
- MANIFEST.json + SHA-256 Checksums
- Per-Component Serializer (LTM, STM, Cognitive, Curiosity, MicroModels, Embeddings)
- 📦 Commit: `feat: full brain-state checkpoint`

### 4.2 Restore + CLI (1 Woche)
- `brain19_cli checkpoint create/load/list/validate`
- Startup-Sequenz: Layer 1 → Layer 2 → Layer 3 Fallback
- Graceful Shutdown mit auto-Checkpoint
- 📦 Commit: `feat: checkpoint restore + CLI`

### Phase-4-Meilenstein ✅
> `brain19_cli checkpoint create` → alles gesichert. `brain19_cli checkpoint load latest` → exakt dort weitermachen. **Brain19 ist jetzt unsterblich.**

---

## Phase 5: Multi-Stream Architecture

> **Ziel:** Paralleles Denken auf allen verfügbaren Cores.  
> **Dependency:** Phase 3 (Thread-Safety).  
> **Geschätzt:** 4-6 Wochen  
> **Referenz:** `docs/MULTI_STREAM_ARCHITECTURE.md` (2424 LOC Design, komplett)  
> **Hardware-Trigger:** EPYC-Server online bringen lohnt sich AB HIER.

### 5.1 Task Queue + Stream Infrastructure (2 Wochen)
- MPMC Lock-Free Queue (4 Priority Lanes)
- `ThinkStream` Klasse mit Main-Loop, Backoff, Heartbeat
- `StreamOrchestrator` + Lifecycle Management
- 📦 Commit: `feat: stream infrastructure + task queue`

### 5.2 Stream-Kategorien implementieren (2 Wochen)
- Query Streams (latency-critical, dedicated)
- Training Streams (embarrassingly parallel MicroModel training)
- Creative Streams (RelevanceMap overlay)
- Validation Streams (trust decay, contradictions)
- 📦 Commit: `feat: specialized thinking streams`

### 5.3 Monitoring + CLI (1 Woche)
- `StreamMonitor` (10Hz sampling, atomic metrics)
- CLI: Stream Status, Throughput, Latency Histogramme
- 📦 Commit: `feat: stream monitoring + CLI`

### Phase-5-Meilenstein ✅
> Brain19 denkt auf 4 Cores parallel (i5-6600K) oder 80 Cores (EPYC). Training von 10.000 MicroModels in unter 1 Minute.

---

## Phase 6: Dynamische Konzept-Erweiterung

> **Ziel:** Brain19 entdeckt selbständig neue Konzepte.  
> **Dependency:** Phase 1 (Persistence) + Phase 2 (korrekte MicroModels).  
> **Geschätzt:** 4-6 Wochen  
> **Referenz:** `docs/DYNAMIC_CONCEPT_EXTENSION.md` (3-Schicht-Modell definiert)

### 6.1 ConceptProposal + Epistemische Promotion-Pipeline (2 Wochen)
- `ConceptProposal` Klasse (tentative Klassifikation)
- 4-Stufen-Validierung: Pattern → Coherence → Utility → Temporal Stability
- `ConceptDiscoveryLayer` mit read-only LTM-Zugriff
- 📦 Commit: `feat: concept proposal + promotion pipeline`

### 6.2 Pattern-basierte Discovery (2 Wochen)
- Ko-Aktivierungs-Clustering aus STM-History
- Semantic Gap Detection im Knowledge Graph
- Curiosity-getriggerte Bridge-Concept-Vorschläge
- 📦 Commit: `feat: automatic concept discovery`

### Phase-6-Meilenstein ✅
> Brain19 schlägt selbständig neue Konzepte vor. Jedes wird epistemisch validiert bevor es in den KG kommt. **Das System lernt.**

---

## Phase 7: KAN-LLM Hybrid Integration

> **Ziel:** KAN validiert LLM-Hypothesen mathematisch.  
> **Dependency:** Phase 2 (KAN korrekt) + Phase 6 (Concept Discovery).  
> **Geschätzt:** 6-8 Wochen  
> **Referenz:** `docs/KAN_LLM_HYBRID_THEORY.md` (Teile I-VIII, vollständiges Design)

### 7.1 Topologie B: LLM → KAN Validation (3 Wochen)
- `HypothesisTranslator`: LLM-Hypothese → KAN-Trainingsproblem
- `EpistemicBridge`: KAN-Metriken (MSE, Konvergenz) → Trust/Type
- KAN als epistemischer Validator (Mechanik prüft Magie)
- 📦 Commit: `feat: KAN validates LLM hypotheses`

### 7.2 Domain-Clustering + Domain-Paare (2 Wochen)
- Relationstyp-basierte Domain-Erkennung
- Pro Domain: KAN-LLM-Paar (lazy instantiation)
- Cross-Domain Queries für kreative Insights
- 📦 Commit: `feat: domain-specific KAN-LLM pairs`

### 7.3 Bidirektionaler Dialog (Topologie C) (2 Wochen)
- Iteratives Refinement: LLM → KAN → Residuum → LLM → ...
- Terminierungsbedingung + Iterations-Limit
- Provenienz-Kette für jede Iteration
- 📦 Commit: `feat: bidirectional KAN-LLM refinement`

### Phase-7-Meilenstein ✅
> Brain19 generiert kreative Hypothesen (LLM) und validiert sie mathematisch (KAN). Inspizierbare B-Spline-Plots als Erklärungen. **Kein existierendes System kann das.**

---

## Phase 8: Hot/Cold Tiering + NUMA (Optional/Ongoing)

> **Ziel:** Performance-Optimierung für große Knowledge Graphs.  
> **Dependency:** Phase 1 + Phase 5.  
> **Hardware-Requirement:** EPYC Server mit 120GB RAM.  
> **Geschätzt:** 2-3 Wochen  

- `TierManager`: Hot (mlock), Warm (pageable), Cold (SSD)
- `AccessTracker` + automatische Promotion/Demotion
- NUMA-aware Allocation für EPYC
- Huge Pages für Hot Tier

### Phase-8-Meilenstein ✅
> 100.000+ Konzepte, <100ns Lookup für Hot-Tier.

---

## Was NICHT auf der Roadmap steht (bewusst weggelassen)

| Feature | Grund |
|---------|-------|
| **HTTP/WebSocket Server** | Nice-to-have, aber Persistence + Korrektheit wichtiger |
| **Frontend-Redesign** | Funktioniert, ist read-only, Priorität ist Backend |
| **Wikipedia/Scholar Importer** | Existiert, funktioniert, keine Bugs bekannt |
| **Sleep Cycle** | Spannend, aber abhängig von Phase 6+7 |
| **Q.ANT Photonic NPU** | Spekulativ, Design ist photonic-ready (Phase 5), Hardware existiert nicht |
| **Publication** | Kommt wenn Phase 7 fertig ist, nicht vorher |

---

## Dependency-Graph

```
Phase 0 ──────────────┬──────────────────────────────────────────────┐
(Bugfixes)            │                                              │
                      ▼                                              ▼
               Phase 1                                        Phase 2
               (Persistence)                                  (Snapshot+KAN Fix)
                      │                                              │
                      ├──────────────┐                               │
                      ▼              ▼                               │
               Phase 3        Phase 4                                │
               (Thread-Safety) (Full Checkpoint)                     │
                      │                                              │
                      ▼                                              │
               Phase 5                                               │
               (Multi-Stream)                                        │
                      │              ┌───────────────────────────────┘
                      │              ▼
                      │       Phase 6
                      │       (Dynamic Concepts)
                      │              │
                      │              ▼
                      │       Phase 7
                      │       (KAN-LLM Hybrid)
                      ▼
               Phase 8
               (Performance)
```

**Parallelisierbar:** Phase 1 + Phase 2 können gleichzeitig laufen.

---

## Realistische Timeline

| Phase | Effektive Zeit | Kalender (Schule Mo-Fr) | Kalender (nach Schule ab 12.05) |
|-------|---------------|------------------------|--------------------------------|
| **Phase 0** | 3-5 Abende | 1 Woche | 2-3 Tage |
| **Phase 1** | 3-4 Wochen | 6-8 Wochen | 3-4 Wochen |
| **Phase 2** | 2-3 Abende | 1 Woche | 2-3 Tage |
| **Phase 3** | 2 Wochen | 4-5 Wochen | 2 Wochen |
| **Phase 4** | 2 Wochen | 4-5 Wochen | 2 Wochen |
| **Phase 5** | 4-6 Wochen | 8-12 Wochen | 4-6 Wochen |
| **Phase 6** | 4-6 Wochen | 8-12 Wochen | 4-6 Wochen |
| **Phase 7** | 6-8 Wochen | 12-16 Wochen | 6-8 Wochen |

**Phase 0+1+2 bis Schulende (11.05):** Realistisch. Das ist der wichtigste Meilenstein.  
**Phase 3-5 Sommer 2026:** Wenn Schule vorbei ist, beschleunigt sich alles.  
**Phase 6-7 Herbst/Winter 2026:** Das sind die Research-Phasen.

---

## EPYC Server: Wann lohnt sich das?

- **Phase 0-4:** Nicht nötig. i5-6600K reicht.
- **Phase 5:** EPYC holen. 80 Cores → 80 Streams. Der Unterschied ist der zwischen "Demo" und "Produktiv".
- **Alternative:** Mietserver (Hetzner AX161: 64 Cores, ~€60/Monat) überbrückt bis EPYC Strom hat.

---

## ADHS-Strategie

1. **Kleine Pakete:** Jeder Punkt ist max 1-2 Abende. Jeder endet mit einem pushbaren Commit.
2. **Phase 0 zuerst:** Bugfixes sind die einfachsten Wins. Sofortige Dopamin-Belohnung.
3. **Tests als Fortschrittsanzeige:** Jeder Fix hat einen Test. Grün = geschafft.
4. **Ein Ding gleichzeitig:** Nicht Phase 1 und Phase 5 mischen. Sequentiell.
5. **Pausen einplanen:** Wenn du bei Phase 1 steckst → Phase 2 machen (andere Dateien, frischer Kontext).

---

*Erstellt 2026-02-10 basierend auf Code-Audit und allen docs/*.md Dokumenten.*
