# Brain19 Architecture Analysis

**Datum:** 2026-02-11  
**Autor:** Automatische Codeanalyse  
**Version:** 1.0

---

## Inhaltsverzeichnis

1. [Executive Summary](#executive-summary)
2. [Ist-Zustand: Was funktioniert, was ist Stub](#ist-zustand)
3. [Datenfluss-Diagramme](#datenfluss)
4. [Analyse der 6 Kernfragen](#kernfragen)
5. [Lücken und fehlende Verbindungen](#luecken)
6. [Empfehlungen: KAN-MiniLLM Hybrid Engine](#empfehlungen)

---

## 1. Executive Summary <a name="executive-summary"></a>

Brain19 ist ein ambitioniertes kognitives Architektur-System mit ~15 Subsystemen. Die **Grundinfrastruktur ist solide implementiert**: LTM mit epistemischem Tracking, Ingestion Pipeline, KAN-basierte Funktionsapproximation, MicroModel-basierte Relevanzberechnung, und ein Ollama-basiertes LLM-Interface.

**Kritische Erkenntnis:** Die Systeme arbeiten weitgehend **parallel aber isoliert**. Die zentrale Schwäche ist das Fehlen einer echten **bidirektionalen Wissensverankerung** — MicroModels lernen aus KG-Struktur, aber das gelernte Wissen fließt nicht zurück in den Graphen. MiniLLMs generieren Hypothesen, aber diese werden nicht systematisch durch KAN validiert und verankert.

### Reifegrad-Übersicht

| Subsystem | Status | Reifegrad |
|-----------|--------|-----------|
| LTM (Knowledge Graph) | ✅ Voll funktional | 95% |
| Epistemic Metadata | ✅ Voll funktional | 95% |
| Ingestion Pipeline | ✅ Voll funktional | 90% |
| MicroModel + Training | ✅ Voll funktional | 85% |
| RelevanceMap | ✅ Voll funktional | 85% |
| KAN (B-Spline Module) | ✅ Voll funktional | 85% |
| KAN Validator (Hybrid) | ✅ Voll funktional | 80% |
| Epistemic Promotion | ✅ Voll funktional | 80% |
| Pattern Discovery | ✅ Voll funktional | 75% |
| Ollama Client | ✅ Voll funktional | 85% |
| ChatInterface | ✅ Voll funktional | 80% |
| UnderstandingLayer | ⚠️ Teilweise | 60% |
| OllamaMiniLLM | ⚠️ Basis-Impl. | 55% |
| MiniLLMFactory | ❌ Nur Header/TODO | 10% |
| SpecializedMiniLLM | ❌ Nur Header/TODO | 10% |
| KAN↔MiniLLM Brücke | ❌ Nicht verbunden | 5% |
| InteractionLayer | ❌ Fehlt komplett | 0% |

---

## 2. Ist-Zustand <a name="ist-zustand"></a>

### 2.1 Voll Funktional

#### LTM + Epistemic System
- In-Memory Knowledge Graph mit `ConceptInfo` (Label, Definition, EpistemicMetadata)
- **Compile-time enforced** epistemische Explizitheit: `ConceptInfo()` = deleted, EpistemicMetadata MUSS übergeben werden
- Vollständiges Relationsmodell: 10 Typen (IS_A, CAUSES, PART_OF, etc.) mit Gewichtung [0,1]
- Bidirektionale Indexierung (outgoing/incoming Relations)
- Epistemische Promotion-Pipeline: SPECULATION → HYPOTHESIS → THEORY → FACT (letzteres nur mit Human Review)

#### Ingestion Pipeline
- JSON/CSV Parser für strukturierte Wissensdaten
- Text-Ingestion mit Entity Extraction (Capitalization, Quotes, Definition Patterns)
- Relation Extraction (regelbasiert: IS_A, CAUSES, PART_OF Patterns)
- Trust Tagger mit Hedging/Certainty Language Detection
- ProposalQueue mit Review-Workflow (Approve/Reject/Auto-Approve)

#### MicroModels
- **Bilineare Micro-Models** (10D): `w = σ(eᵀ · (W·c + b))` pro Konzept
- Adam Optimizer mit vollständigem Training-State (Momentum, Variance)
- **MicroTrainer**: Generiert Training-Samples aus KG-Struktur
  - Positive: Outgoing Relations (weight), Incoming (0.8× discount)
  - Negative: 3× pro Positive, aus nicht-verbundenen Konzepten
- Binary Persistence mit Checksum-Validierung
- RelevanceMap: Evaluiert MicroModel über alle KG-Nodes

#### KAN System
- B-Spline basierte KANNodes (Cox-de-Boor Rekursion)
- Multi-Layer KANModule mit beliebiger Topologie
- Gradient Descent Training mit Backpropagation durch alle Layer
- KANAdapter: Clean Interface mit Module-Lifecycle Management

#### KAN-LLM Hybrid Pipeline
- **HypothesisTranslator**: NLP-lite Pattern Detection (LINEAR, POLYNOMIAL, EXPONENTIAL, PERIODIC, THRESHOLD, CONDITIONAL)
  - Confidence-scored Detection mit Negation und Quantifier-Handling
  - Numerische Hint-Extraktion aus Hypothesentexten
  - Domänenspezifische Training-Data Generation
- **EpistemicBridge**: MSE → Trust-Score Mapping
  - MSE < 0.01 → THEORY (Trust 0.7-0.9)
  - MSE < 0.1 → HYPOTHESIS (Trust 0.4-0.6)
  - MSE ≥ 0.1 → SPECULATION (Trust 0.1-0.3)
  - Trust-Inflation-Caps für synthetische Daten
- **RefinementLoop**: Bidirektionaler LLM↔KAN Dialog (max 5 Iterationen)
- **DomainManager**: Domänen-Detection aus Relations-Mustern (PHYSICAL, BIOLOGICAL, SOCIAL, ABSTRACT, TEMPORAL)

### 2.2 Teilweise Implementiert

#### UnderstandingLayer
- Framework steht: registriert MiniLLMs, aggregiert Proposals, filtert nach Confidence
- `perform_understanding_cycle()`: Spreading Activation → Salience → MiniLLM Proposals
- **Aber**: Outputs werden im SystemOrchestrator nur im ThinkingPipeline-Kontext genutzt, nicht systematisch in den KG zurückgeschrieben

#### OllamaMiniLLM
- Funktionale Ollama-Anbindung für alle 4 Proposal-Typen (Meaning, Hypothesis, Analogy, Contradiction)
- Prompt-Engineering mit epistemischem Kontext
- **Aber**: Kein Fine-Tuning, keine Spezialisierung, keine Wissensverankerung

### 2.3 Nur Stubs / Nicht Implementiert

#### MiniLLMFactory + SpecializedMiniLLM
- Headers existieren mit vollständiger API-Definition
- Kommentare: "TODO: Not yet implemented — planned for KAN-LLM Hybrid Layer"
- Konzept: Dynamisch spezialisierte MiniLLMs pro Wissensbereich
- **Null Implementierung vorhanden**

#### InteractionLayer (MicroModel↔MicroModel)
- **Existiert nicht**. MicroModels kommunizieren ausschließlich über passive RelevanceMap-Aggregation
- Kein Message-Passing, kein Attention-Mechanismus zwischen Models

---

## 3. Datenfluss-Diagramme <a name="datenfluss"></a>

### 3.1 Hauptdatenfluss: Ingestion → LTM → MicroModel

```
                    ┌─────────────────────────────────────────┐
                    │         INGESTION PIPELINE              │
                    │                                         │
  JSON/CSV ────────►│  KnowledgeIngestor                      │
                    │    ├─ parse_json() / parse_csv()        │
                    │    └─ to_proposals() + TrustTagger       │
                    │         │                               │
  Free Text ───────►│  TextChunker → EntityExtractor          │
                    │              → RelationExtractor         │
                    │              → TrustTagger               │
                    │         │                               │
                    │    ProposalQueue                         │
                    │    ├─ PENDING → Review → APPROVED        │
                    │    └─ auto_approve_all()                 │
                    └────────────┬────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────────────┐
                    │              LTM                         │
                    │  store_concept(label, def, Epistemic)   │
                    │  add_relation(src, tgt, type, weight)    │
                    │                                         │
                    │  Concepts: {id, label, def, epistemic}  │
                    │  Relations: {id, src, tgt, type, weight}│
                    └────────────┬────────────────────────────┘
                                 │
                    ┌────────────┴────────────────────────────┐
                    │       MICROMODEL PIPELINE               │
                    │                                         │
                    │  MicroModelRegistry                     │
                    │    └─ ensure_models_for(ltm)            │
                    │         └─ 1 MicroModel pro ConceptId   │
                    │                                         │
                    │  MicroTrainer                            │
                    │    └─ generate_samples(cid, emb, ltm)   │
                    │         ├─ Positive: Relations (weight)  │
                    │         └─ Negative: Non-connected (0.05)│
                    │    └─ train(samples, Adam)               │
                    │                                         │
                    │  RelevanceMap                            │
                    │    └─ compute(src, registry, emb, ltm)  │
                    │         └─ model.predict(e, c) ∀ nodes  │
                    └─────────────────────────────────────────┘
```

### 3.2 Chat-Datenfluss: Frage → Antwort

```
  User Question
       │
       ▼
  SystemOrchestrator::ask()
       │
       ├─── 1. Label-Search in LTM (case-insensitive)
       │         → seed concepts
       │
       ├─── 2. ThinkingPipeline::execute()
       │         ├─ Spreading Activation (CognitiveDynamics)
       │         ├─ Salience Computation
       │         ├─ RelevanceMap Computation
       │         ├─ UnderstandingLayer Proposals
       │         └─ KAN Validation (optional)
       │         → salient_ids, thought_paths
       │
       ├─── 3. run_evolution_after_thinking()
       │         ├─ ConceptProposer::from_curiosity()
       │         └─ ConceptProposer::from_relevance_anomalies()
       │         → new speculative concepts in LTM
       │
       └─── 4. ChatInterface::ask_with_context()
                 ├─ build_epistemic_context(concepts)
                 ├─ Ollama chat(system_prompt + context + question)
                 └─ → ChatResponse (answer, referenced_concepts, speculation_flag)
```

### 3.3 KAN Validation Flow (Hybrid)

```
  HypothesisProposal (from MiniLLM)
       │
       ▼
  KanValidator::validate()
       │
       ├─── 1. HypothesisTranslator::translate()
       │         ├─ detect_pattern_detailed() → Pattern + Confidence
       │         ├─ extract_numeric_hints() → slope, range, scale
       │         └─ generate_training_data() → synthetic DataPoints
       │         → KanTrainingProblem
       │
       ├─── 2. KANModule::train()
       │         ├─ Multi-layer B-Spline Network
       │         └─ Gradient Descent → KanTrainingResult
       │
       └─── 3. EpistemicBridge::assess()
                 ├─ MSE → EpistemicType (THEORY/HYPOTHESIS/SPECULATION)
                 ├─ Trust-Inflation Caps (synthetic data max 0.6)
                 └─ → EpistemicAssessment → ValidationResult
```

### 3.4 Evolution / Promotion Cycle

```
  Periodic (every 5 min)                    After Thinking
       │                                         │
       ▼                                         ▼
  EpistemicPromotion                    ConceptProposer
  ::run_maintenance()                   ::from_curiosity()
       │                                ::from_relevance_anomalies()
       ├─ evaluate_all()                      │
       │   ├─ SPEC→HYP: 3 supports,          ├─ Generate proposals
       │   │   validation>0.3,                │   (max trust 0.5)
       │   │   no contradictions              │
       │   ├─ HYP→THEORY: 5 theory+          ├─ rank_proposals()
       │   │   supports, 2 independent,       │
       │   │   validation>0.6                 └─ Store as SPECULATION
       │   └─ THEORY→FACT: human review            in LTM
       │       (NEVER automatic)
       │
       ├─ check_demotion()
       │   └─ Contradictions → demote
       │
       └─ PatternDiscovery
           ├─ find_clusters()
           ├─ find_hierarchies()
           ├─ find_bridges()
           ├─ find_cycles()
           └─ find_gaps()
```

---

## 4. Analyse der 6 Kernfragen <a name="kernfragen"></a>

### 4.1 Vernetzung MiniLLM ↔ Wissensgraph (LTM)

**Ist-Zustand: Einseitig, READ-ONLY**

Die Verbindung ist bewusst asymmetrisch designed:

- **LTM → MiniLLM**: MiniLLMs erhalten READ-ONLY Zugriff auf LTM. Der `OllamaMiniLLM` baut Prompts aus `ConceptInfo` (Label, Definition, EpistemicMetadata) und gibt sie als Kontext an Ollama. Dies funktioniert.

- **MiniLLM → LTM**: Es gibt **keinen direkten Schreibweg**. MiniLLM-Outputs sind `Proposals` (HYPOTHESIS). Diese gehen an die `UnderstandingLayer`, werden im `ThinkingPipeline` verarbeitet, und nur indirekt über `ConceptProposer::from_curiosity()` werden neue Konzepte in LTM geschrieben — aber als SPECULATION mit max Trust 0.5.

- **Fehlende Brücke**: Die `HypothesisProposal`s aus dem MiniLLM werden **nicht systematisch** durch den `KanValidator` geschickt. Im `ThinkingPipeline` wird der Validator optional aufgerufen, aber es gibt keinen automatischen Loop: MiniLLM-Hypothese → KAN-Validation → Trust-Update → LTM-Speicherung.

**Bewertung: 3/10** — Die Architektur-Invariante "MiniLLMs sind READ-ONLY" ist korrekt, aber der Rückkanal (Proposals → Validation → Verankerung) fehlt.

### 4.2 Tiefe der Wissensverankerung

**Ist-Zustand: Strukturell, nicht semantisch**

MicroModels werden **tatsächlich auf die Relationen trainiert** — aber nur strukturell:

```cpp
// MicroTrainer::generate_samples()
// Positive: outgoing relations (weight as target)
// Negative: non-connected concepts (0.05 as target)
```

Das Training lernt: "Welche Konzepte sind über welche Relationstypen verbunden?" via Embedding-Scores. Die MicroModels lernen die **Graphtopologie**, nicht die **Semantik** der Relationen.

**Was fehlt:**
- Kein Training auf **Definitionstexte** oder semantische Ähnlichkeit
- Kein Training auf **epistemische Qualität** (Trust-Werte fließen nicht ins Training)
- Kein **Cross-Concept Training**: MicroModel A weiß nicht, was MicroModel B gelernt hat
- Die RelevanceMap ist **passiv** — sie wird computed aber nicht für Lernentscheidungen genutzt

**Bewertung: 4/10** — Relationsstruktur wird gelernt, aber keine tiefe semantische Verankerung.

### 4.3 Datenfluss: Training Data → LTM → MicroModel Training

Der Datenfluss ist **klar und funktional**, aber **einmalig statt iterativ**:

```
JSON/Text → IngestionPipeline → LTM.store_concept() + add_relation()
         → MicroModelRegistry.ensure_models_for(ltm)  [creates empty models]
         → MicroTrainer.train_all()                     [trains from KG structure]
```

**Vollständiger Fluss existiert**, aber:
1. Training passiert nur beim `initialize()` oder explizit über `ensure_models_for()`
2. **Kein inkrementelles Training**: Neue Konzepte bekommen ein frisches MicroModel, aber existierende Models werden nicht re-trainiert wenn neue Relationen hinzukommen
3. **Kein Online-Learning**: Der periodische Maintenance-Loop (`run_periodic_maintenance()`) macht Epistemic Promotion und Pattern Discovery, aber **kein MicroModel Re-Training**

**Bewertung: 6/10** — Fluss existiert end-to-end, aber ist nicht-iterativ.

### 4.4 InteractionLayer vs. Passive Aggregation

**Ist-Zustand: Nur passive Aggregation**

MicroModels kommunizieren **ausschließlich** über `RelevanceMap`:

```cpp
// RelevanceMap::compute() — berechnet einen Score pro Ziel-Konzept
// RelevanceMap::overlay() — kombiniert mehrere Maps (ADDITION/MAX/WEIGHTED_AVG)
// RelevanceMap::combine() — statische Aggregation
```

Es gibt **keine InteractionLayer**:
- Kein Message-Passing zwischen MicroModels
- Kein Attention-Mechanismus
- Kein kompetitives oder kooperatives Verhalten
- Kein "Model A beeinflusst Training von Model B"

Die einzige "Interaktion" passiert über den Wissensgraph selbst (Spreading Activation in CognitiveDynamics), nicht zwischen den Models direkt.

**Bewertung: 2/10** — Rein passive Aggregation, keine echte Interaktion.

### 4.5 Was fehlt für echte interaktive Emergenz?

1. **Active InteractionLayer**: MicroModels müssen Signale senden/empfangen können
   - Attention-basiertes Message-Passing: Model_A sendet "Ich bin aktiv mit Score X" → benachbarte Models reagieren
   - Kompetitive Inhibition: Widersprüchliche Models supprimieren sich gegenseitig
   - Kooperative Verstärkung: Unterstützende Models boosten sich

2. **Feedback Loop MiniLLM → KAN → LTM**:
   - MiniLLM generiert Hypothese
   - KAN validiert mathematisch → Trust-Score
   - Trust-Score wird als neue Relation in LTM geschrieben
   - MicroModels werden inkrementell re-trainiert
   - Nächste MiniLLM-Runde hat besseren Kontext

3. **Emergente Konzeptbildung**:
   - Wenn mehrere MicroModels konsistent auf ein "Phantom-Konzept" zeigen (hohe Relevanz ohne LTM-Eintrag), sollte automatisch ein neues Konzept vorgeschlagen werden
   - Aktuell passiert das nur über `ConceptProposer::from_relevance_anomalies()`, aber ohne MicroModel-Feedback

4. **Temporal Dynamics**:
   - Aktivierungsmuster über Zeit tracken
   - Rekurrente Muster erkennen → stabile Attraktoren = emergente Konzepte
   - Aktuell ist alles zustandslos pro Thinking-Cycle

5. **Self-Modification**:
   - MicroModels sollten ihre eigene Architektur anpassen können (z.B. Embedding-Dimension, Learning Rate)
   - KAN-Module sollten ihre Topologie basierend auf Fehleranalyse erweitern/reduzieren

### 4.6 Ollama-Einbindung und KAN-MiniLLM Hybrid Engine

**Aktuelle Ollama-Einbindung:**

Ollama wird an **zwei Stellen** genutzt:

1. **ChatInterface** (`backend/llm/`): Verbalisierung von LTM-Wissen für den User
   - System-Prompt mit epistemischen Regeln
   - Konzept-Kontext aus LTM
   - Modell: `llama3.2:1b` (konfigurierbar)

2. **OllamaMiniLLM** (`backend/understanding/`): Semantische Analyse
   - Meaning Extraction, Hypothesis Generation, Analogy Detection, Contradiction Detection
   - Gleicher Ollama-Client, gleiche API

**Wo Ollama durch KAN-MiniLLM Hybrid ersetzen:**

| Aufgabe | Aktuell (Ollama) | Vorschlag (KAN-Hybrid) |
|---------|------------------|------------------------|
| Hypothesis Generation | LLM-Prompt → Text → Parse | KAN-Pattern-Detection + LLM für Textualierung |
| Analogy Detection | LLM-Prompt → Text → Parse | RelevanceMap-Overlay + Strukturvergleich |
| Contradiction Detection | LLM-Prompt → Text → Parse | Epistemischer Graph-Check + KAN-Validierung |
| Meaning Extraction | Braucht LLM | LLM bleibt, aber mit KAN-Context |
| Chat Verbalization | Braucht LLM | LLM bleibt (= User-Interface) |

**Konkret ersetzbar durch KAN:**
- `detect_analogies()`: Strukturelle Analogie ist ein Graph-Problem, kein LLM-Problem. RelevanceMap-Vergleich + KAN-Funktionsvergleich (Topologie-Match) wäre präziser und schneller.
- `detect_contradictions()`: Epistemische Widersprüche sind im Graph codiert (CONTRADICTS-Relations, Trust-Konflikte). Ein KAN-Modul könnte lernen, welche Konzeptpaare typischerweise widersprüchlich sind.
- `generate_hypotheses()`: Teilweise. Pattern Detection passiert bereits via `HypothesisTranslator`. Der LLM-Teil könnte auf Textgenerierung für die Hypothesenformulierung reduziert werden.

---

## 5. Lücken und fehlende Verbindungen <a name="luecken"></a>

### Lücke 1: MiniLLM-Hypothesen werden nicht KAN-validiert

```
AKTUELL:
  OllamaMiniLLM → HypothesisProposal → UnderstandingLayer → [Ende]
  
SOLL:
  OllamaMiniLLM → HypothesisProposal → KanValidator::validate()
                                      → EpistemicBridge::assess()
                                      → LTM.update_epistemic_metadata()
                                      → MicroModel Re-Training
```

Die gesamte KAN-Validation-Pipeline existiert und ist getestet, wird aber **nicht automatisch** auf MiniLLM-Outputs angewendet.

### Lücke 2: MiniLLMFactory nicht implementiert

Das Design sieht vor, dass pro Wissensbereich ein spezialisiertes MiniLLM erzeugt wird. Die Header sind definiert, die Implementierung fehlt komplett. Ohne spezialisierte MiniLLMs bleibt das System ein "ein LLM für alles".

### Lücke 3: Kein inkrementelles MicroModel-Training

```
AKTUELL:
  ensure_models_for(ltm) → create fresh model → train_all() [einmalig]
  
SOLL:
  Neue Relation in LTM → betroffene Models identifizieren
                        → inkrementelles Re-Training
                        → RelevanceMap Update
```

### Lücke 4: RelevanceMap → LTM Rückfluss fehlt

RelevanceMaps werden berechnet aber nicht persistent genutzt. Anomalien führen über `ConceptProposer` zu neuen Konzepten, aber die Maps selbst werden nicht gespeichert oder für Training genutzt.

### Lücke 5: KAN-Module nicht an MicroModels gebunden

KAN-Module (mathematische Funktionsapproximation) und MicroModels (Relevanzberechnung) sind **völlig separate Systeme**. Es gibt keine Brücke, die ein MicroModel mit einem KAN-validierten Funktionsmodell verbindet.

### Lücke 6: Thinking-Ergebnisse nicht persistent

`ThinkingResult` enthält wertvolle Daten (Salience-Scores, Thought-Paths, Curiosity-Triggers), die nach dem Thinking-Cycle verworfen werden. Nur die ConceptProposer-Ergebnisse werden in LTM geschrieben.

---

## 6. Empfehlungen: KAN-MiniLLM Hybrid Engine <a name="empfehlungen"></a>

### 6.1 Architektur-Vorschlag: KAN-MiniLLM Brücke

```
┌─────────────────────────────────────────────────────────────┐
│                    KAN-MINILLM HYBRID ENGINE                 │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  MiniLLM Pool │    │  KAN Cluster  │    │  MicroModel  │  │
│  │  (Ollama +    │◄──►│  (Validatoren │◄──►│  Ring        │  │
│  │   Specialized)│    │   pro Domain) │    │  (Interaction│  │
│  └──────┬───────┘    └──────┬───────┘    │   Layer)     │  │
│         │                    │            └──────┬───────┘  │
│         ▼                    ▼                   ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              WISSENSVERANKERUNGS-BUS                  │  │
│  │  Proposals → Validation → Trust-Score → LTM Write    │  │
│  │  + MicroModel Re-Training + RelevanceMap Update      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Konkrete Implementierungsschritte

#### Phase 1: Validation Loop schließen (Aufwand: ~2-3 Tage)

```cpp
// In SystemOrchestrator oder ThinkingPipeline:
void validate_and_anchor_hypotheses(
    const std::vector<HypothesisProposal>& proposals
) {
    for (const auto& proposal : proposals) {
        // 1. KAN-Validation
        auto result = kan_validator_->validate(proposal);
        
        if (result.validated) {
            // 2. Store validated hypothesis in LTM
            auto cid = ltm_->store_concept(
                proposal.hypothesis_statement,
                proposal.supporting_reasoning,
                result.assessment.metadata  // Trust from KAN
            );
            
            // 3. Add SUPPORTS relations to evidence concepts
            for (auto evidence_id : proposal.evidence_concepts) {
                ltm_->add_relation(evidence_id, cid, 
                    RelationType::SUPPORTS, 
                    result.assessment.metadata.trust);
            }
            
            // 4. Create + train MicroModel
            registry_->create_model(cid);
            trainer_->train_single(cid, *registry_->get_model(cid),
                                   *embeddings_, *ltm_);
        }
    }
}
```

#### Phase 2: Inkrementelles MicroModel-Training (Aufwand: ~1-2 Tage)

```cpp
// In MicroTrainer:
void retrain_affected(ConceptId new_concept,
                      MicroModelRegistry& registry,
                      EmbeddingManager& embeddings,
                      const LongTermMemory& ltm) {
    // Re-train the new concept's model
    if (auto* model = registry.get_model(new_concept)) {
        train_single(new_concept, *model, embeddings, ltm);
    }
    
    // Re-train neighbors (1-hop)
    for (const auto& rel : ltm.get_outgoing_relations(new_concept)) {
        if (auto* model = registry.get_model(rel.target)) {
            train_single(rel.target, *model, embeddings, ltm);
        }
    }
    for (const auto& rel : ltm.get_incoming_relations(new_concept)) {
        if (auto* model = registry.get_model(rel.source)) {
            train_single(rel.source, *model, embeddings, ltm);
        }
    }
}
```

#### Phase 3: InteractionLayer (Aufwand: ~3-5 Tage)

```cpp
class InteractionLayer {
public:
    struct Signal {
        ConceptId source;
        double activation;
        Vec10 embedding_delta;  // Wie das Model seine Umgebung beeinflusst
    };
    
    // Broadcast: Aktives MicroModel sendet Signal
    void broadcast(ConceptId source, double activation,
                   const MicroModel& model, const EmbeddingManager& emb);
    
    // Receive: Benachbarte Models empfangen und reagieren
    void propagate(MicroModelRegistry& registry,
                   const LongTermMemory& ltm,
                   EmbeddingManager& emb);
    
    // Kompetitive Inhibition
    void lateral_inhibition(const std::vector<ConceptId>& active,
                            MicroModelRegistry& registry);
    
    // Kooperative Verstärkung
    void cooperative_boost(const std::vector<ConceptId>& supporting,
                           MicroModelRegistry& registry);
};
```

#### Phase 4: KAN-MiniLLM Spezialisierung (Aufwand: ~3-5 Tage)

Statt generisches Ollama für alles:

1. **Domain-KAN**: Pro `DomainType` ein trainiertes KAN-Modul, das domänenspezifische Relationen validiert
2. **KAN-gesteuerte Prompts**: KAN-Analyseergebnisse fließen in MiniLLM-Prompts ein
3. **MiniLLMFactory implementieren**: Pro Konzeptcluster ein spezialisiertes MiniLLM mit fokussiertem System-Prompt

```cpp
// Statt: OllamaMiniLLM (generisch für alles)
// Neu:
class KANAugmentedMiniLLM : public MiniLLM {
    // KAN validiert Hypothesen VOR der LLM-Formulierung
    // LLM formuliert nur noch den Text, KAN entscheidet über Plausibilität
    
    std::vector<HypothesisProposal> generate_hypotheses(...) override {
        // 1. Graph-basierte Muster erkennen (ohne LLM)
        auto patterns = detect_graph_patterns(evidence_concepts, ltm);
        
        // 2. KAN-Validation der Muster
        auto validated = kan_validate(patterns);
        
        // 3. NUR für die validierten: LLM zur Textformulierung
        return llm_formulate(validated);
    }
};
```

### 6.3 Priorisierung

| Priorität | Maßnahme | Impact | Aufwand |
|-----------|----------|--------|---------|
| 🔴 P0 | Validation Loop schließen | Hoch | 2-3d |
| 🔴 P0 | Inkrementelles Re-Training | Hoch | 1-2d |
| 🟡 P1 | InteractionLayer | Mittel-Hoch | 3-5d |
| 🟡 P1 | KAN-Augmented MiniLLM | Mittel | 3-5d |
| 🟢 P2 | MiniLLMFactory implementieren | Mittel | 2-3d |
| 🟢 P2 | Thinking-Persistenz | Niedrig | 1d |
| 🔵 P3 | Self-Modifying Topology | Niedrig | 5-7d |

### 6.4 Ollama-Ersetzungsstrategie

**Nicht ersetzen, sondern reduzieren:**

Ollama bleibt für:
- Chat-Verbalisierung (User-Interface)
- Meaning Extraction (semantisches Verständnis braucht Sprachmodell)
- Textformulierung von Hypothesen

Ollama wird ersetzt durch KAN/Graph-Analyse für:
- Analogy Detection → RelevanceMap-Strukturvergleich
- Contradiction Detection → Graph-basierte epistemische Prüfung
- Pattern Detection → bereits implementiert im `HypothesisTranslator`
- Hypothesis Plausibility → KAN-Validation (bereits implementiert, nur nicht verbunden)

---

## Appendix: Dateistruktur der analysierten Komponenten

```
backend/
├── ltm/
│   ├── long_term_memory.{hpp,cpp}     # Knowledge Graph + Epistemic Store
│   └── relation.hpp                    # RelationInfo + RelationType
├── micromodel/
│   ├── micro_model.{hpp,cpp}           # Bilinear 10D Model (W·c+b, Adam)
│   ├── micro_model_registry.{hpp,cpp}  # 1:1 Model↔Concept Registry
│   ├── micro_trainer.{hpp,cpp}         # KG→Training Samples→Train
│   ├── embedding_manager.{hpp,cpp}     # Relation + Context Embeddings
│   ├── relevance_map.{hpp,cpp}         # Per-Concept Relevance Scoring
│   └── persistence.{hpp,cpp}           # Binary Save/Load + Checksum
├── kan/
│   ├── kan_node.{hpp,cpp}              # B-Spline Univariate Function
│   ├── kan_layer.{hpp,cpp}             # n_in × n_out Grid of Nodes
│   ├── kan_module.{hpp,cpp}            # Multi-Layer KAN + Training
│   └── function_hypothesis.hpp         # Pure Data Wrapper
├── understanding/
│   ├── mini_llm.{hpp,cpp}              # Abstract Interface + StubMiniLLM
│   ├── ollama_mini_llm.{hpp,cpp}       # Ollama-backed Semantic Analysis
│   ├── understanding_layer.{hpp,cpp}   # Aggregation + Filtering
│   ├── understanding_proposals.hpp     # Proposal Types (Meaning/Hyp/Analogy/Contra)
│   └── mini_llm_factory.hpp            # TODO: Dynamic Specialization
├── hybrid/
│   ├── kan_validator.{hpp,cpp}         # End-to-End LLM→KAN Validation
│   ├── hypothesis_translator.{hpp,cpp} # NLP-lite Pattern→KAN Problem
│   ├── epistemic_bridge.{hpp,cpp}      # MSE→Trust Mapping
│   ├── refinement_loop.{hpp,cpp}       # Bidirectional LLM↔KAN Dialog
│   └── domain_manager.{hpp,cpp}        # Domain Detection + Cross-Domain
├── llm/
│   ├── ollama_client.{hpp,cpp}         # HTTP REST Client for Ollama
│   └── chat_interface.{hpp,cpp}        # LLM Verbalization with Epistemic Context
├── evolution/
│   ├── epistemic_promotion.{hpp,cpp}   # SPEC→HYP→THEORY→FACT Pipeline
│   ├── concept_proposal.{hpp,cpp}      # System-Generated Concept Proposals
│   └── pattern_discovery.{hpp,cpp}     # Graph Patterns (Cluster/Hierarchy/Gap)
├── adapter/
│   └── kan_adapter.{hpp,cpp}           # Clean KAN Module Lifecycle
├── ingestor/
│   ├── ingestion_pipeline.{hpp,cpp}    # End-to-End JSON/CSV/Text→LTM
│   ├── knowledge_ingestor.{hpp,cpp}    # JSON/CSV Parser
│   ├── entity_extractor.{hpp,cpp}      # NER from Free Text
│   ├── relation_extractor.{hpp,cpp}    # Rule-based Relation Detection
│   ├── text_chunker.{hpp,cpp}          # Sentence Splitting + Chunking
│   ├── trust_tagger.{hpp,cpp}          # Text→TrustCategory Analysis
│   └── proposal_queue.{hpp,cpp}        # Review Workflow
└── core/
    ├── system_orchestrator.{hpp,cpp}   # 15-Stage Init, All Subsystem Wiring
    └── thinking_pipeline.{hpp,cpp}     # Thinking Cycle Orchestration
```
