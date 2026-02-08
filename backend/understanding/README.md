# Understanding Layer

## Überblick

Der **Understanding Layer** ist die semantische Analyse-Schicht von Brain19, die über dem Cognitive Dynamics Layer arbeitet. Er nutzt Mini-LLMs, um semantische Muster zu erkennen und Vorschläge zu generieren - **ohne jemals epistemische Autorität zu besitzen**.

## 🚨 KRITISCHE ARCHITEKTUR-PRINZIPIEN

### Epistemische Grenzen (NICHT VERHANDELBAR)

```
┌─────────────────────────────────────────────┐
│  EPISTEMIC CORE                             │
│  ✅ Einzige Wahrheitsinstanz                │
│  ✅ Setzt Trust-Werte                       │
│  ✅ Entscheidet FACT vs HYPOTHESIS          │
│  ✅ Validiert Wissen                        │
└─────────────────────────────────────────────┘
                    ▲
                    │ proposals (HYPOTHESIS only)
                    │
┌─────────────────────────────────────────────┐
│  UNDERSTANDING LAYER                        │
│  ✅ Generiert Vorschläge                    │
│  ✅ Erkennt Muster                          │
│  ❌ KEINE epistemische Autorität           │
│  ❌ KEIN Wissens-Schreiben                 │
└─────────────────────────────────────────────┘
                    ▲
                    │ READ-ONLY
                    │
┌─────────────────────────────────────────────┐
│  KNOWLEDGE GRAPH (LTM)                      │
│  📖 Explizites Wissen                       │
│  📖 Mit epistemischen Metadaten             │
└─────────────────────────────────────────────┘
```

### Harte Verbote

Der Understanding Layer **DARF NIEMALS**:

❌ Knowledge Graph modifizieren
❌ Trust-Werte setzen oder ändern
❌ Epistemische Entscheidungen treffen
❌ FACT-Promotion durchführen
❌ Wissen als Wahrheit deklarieren
❌ Regeln generieren
❌ Autonome Aktionen ausführen

### Erlaubte Funktionen

Der Understanding Layer **DARF**:

✅ Texte interpretieren
✅ Muster erkennen
✅ Analogien vorschlagen
✅ Hypothesen formulieren
✅ Konflikte markieren
✅ Mini-LLMs parallel ausführen

**ABER**: Alle Outputs sind **HYPOTHESIS** und epistemisch unverbindlich.

## Komponenten

### 1. Proposal-Strukturen

#### MeaningProposal

Semantischer Vorschlag basierend auf aktivierten Konzepten.

```cpp
struct MeaningProposal {
    uint64_t proposal_id;
    std::vector<ConceptId> source_concepts;  // READ-ONLY references
    std::string interpretation;
    std::string reasoning;
    double model_confidence;  // [0.0, 1.0]
    std::string source_model;
    EpistemicType epistemic_type = EpistemicType::HYPOTHESIS;  // ALWAYS!
};
```

**Enforcement**: `epistemic_type` ist IMMER `HYPOTHESIS`, unabhängig vom Input.

#### HypothesisProposal

Vorgeschlagene Hypothese basierend auf Evidenz.

```cpp
struct HypothesisProposal {
    uint64_t proposal_id;
    std::vector<ConceptId> evidence_concepts;  // READ-ONLY references
    std::string hypothesis_statement;
    std::string supporting_reasoning;
    std::vector<std::string> detected_patterns;
    double model_confidence;  // [0.0, 1.0]
    std::string source_model;

    struct SuggestedEpistemic {
        EpistemicType suggested_type = EpistemicType::HYPOTHESIS;  // ALWAYS!
        double suggested_trust = 0.5;
    };
    SuggestedEpistemic suggested_epistemic;
};
```

**Enforcement**: `suggested_type` ist IMMER `HYPOTHESIS`.

#### AnalogyProposal

Strukturelle Analogie zwischen Konzept-Domänen.

```cpp
struct AnalogyProposal {
    uint64_t proposal_id;
    std::vector<ConceptId> source_domain;
    std::vector<ConceptId> target_domain;
    std::string mapping_description;
    double structural_similarity;  // [0.0, 1.0]
    double model_confidence;  // [0.0, 1.0]
    std::string source_model;
};
```

#### ContradictionProposal

Potenzielle Inkonsistenz (keine Wahrheitsentscheidung!).

```cpp
struct ContradictionProposal {
    uint64_t proposal_id;
    ConceptId concept_a;
    ConceptId concept_b;
    std::string contradiction_description;
    std::string reasoning;
    double severity;  // [0.0, 1.0]
    double model_confidence;  // [0.0, 1.0]
    std::string source_model;
};
```

### 2. MiniLLM Interface

Abstrakte Schnittstelle für semantische Modelle.

```cpp
class MiniLLM {
public:
    virtual ~MiniLLM() = default;
    virtual std::string get_model_id() const = 0;

    // All methods have READ-ONLY LTM access
    virtual std::vector<MeaningProposal> extract_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    ) const = 0;

    virtual std::vector<HypothesisProposal> generate_hypotheses(
        const std::vector<ConceptId>& evidence_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    ) const = 0;

    // ... mehr Methoden
};
```

**Enforcement**: Alle Methoden sind `const` und haben nur READ-ONLY LTM/STM-Zugriff.

### 3. UnderstandingLayer

Hauptklasse für semantische Analyse.

```cpp
class UnderstandingLayer {
public:
    explicit UnderstandingLayer(UnderstandingLayerConfig config);

    // Mini-LLM Management
    void register_mini_llm(std::unique_ptr<MiniLLM> mini_llm);

    // Semantic Analysis (ALL READ-ONLY)
    std::vector<MeaningProposal> analyze_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    );

    std::vector<HypothesisProposal> propose_hypotheses(
        const std::vector<ConceptId>& evidence_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    );

    // ... mehr Methoden
};
```

**Enforcement**: Alle Methoden haben nur READ-ONLY LTM-Zugriff.

## Verwendung

### Demo

```bash
cd backend
make -f Makefile.understanding
./demo_understanding_layer
```

**Output:**
```
╔══════════════════════════════════════════════════════╗
║  Understanding Layer Integration - SUCCESS           ║
╚══════════════════════════════════════════════════════╝

ARCHITECTURAL GUARANTEES:
  ✓ Understanding Layer generated semantic proposals
  ✓ All proposals are HYPOTHESIS (never FACT)
  ✓ LTM accessed READ-ONLY (no knowledge writes)
  ✓ Trust values unchanged
  ✓ EpistemicType unchanged
  ✓ EpistemicStatus unchanged
  ✓ No autonomous actions
  ✓ Epistemic Core remains sole truth arbiter
```

### Tests

```bash
./test_understanding_layer
```

**Test Suite (8 Tests):**

1. **No Knowledge Writes** - Verif iziert, dass kein Wissen erstellt wird
2. **No Trust Manipulation** - Verifiziert, dass Trust-Werte unverändert bleiben
3. **All Proposals are HYPOTHESIS** - Alle Vorschläge sind HYPOTHESIS
4. **Deterministic Behavior** - Gleicher Input → gleicher Output
5. **No Autonomous Actions** - Understanding Layer ist passiv
6. **READ-ONLY LTM Access** - LTM-Relationen bleiben unverändert
7. **System Functions Without Understanding** - System funktioniert ohne Understanding Layer
8. **Bounded Values** - Alle Werte ∈ [0.0, 1.0]

**Erwartung:** Alle 8 Tests bestehen ✅

## Integration mit Cognitive Dynamics

Der Understanding Layer nutzt Cognitive Dynamics für Fokus und Salience:

```cpp
// 1. Use Cognitive Dynamics for spreading activation
auto spread_stats = cognitive_dynamics.spread_activation(
    seed_concept, 1.0, context, ltm, stm
);

// 2. Compute salience to identify important concepts
auto salience_scores = cognitive_dynamics.compute_salience_batch(
    all_concepts, context, ltm, stm, 0
);

// 3. Apply Understanding Layer to salient concepts
auto result = understanding_layer.perform_understanding_cycle(
    seed_concept, cognitive_dynamics, ltm, stm, context
);

// 4. Forward proposals to BrainController → Epistemic Core
// BrainController decides which proposals to accept
```

## Workflow

```
1. Cognitive Dynamics
   ↓ (aktiviert Konzepte, berechnet Salience)

2. Understanding Layer
   ↓ (analysiert Muster, generiert Proposals)

3. BrainController
   ↓ (sammelt Proposals)

4. Epistemic Core
   ↓ (ENTSCHEIDET: Akzeptieren oder Ablehnen)

5. LTM (falls akzeptiert)
   (speichert mit korrekten epistemischen Metadaten)
```

**KRITISCH**: Nur Epistemic Core darf Proposals in LTM schreiben!

## Konfiguration

```cpp
UnderstandingLayerConfig config{
    .enable_parallel_llms = false,       // Parallel LLMs (experimental)
    .min_meaning_confidence = 0.3,       // Threshold für MeaningProposals
    .min_hypothesis_confidence = 0.2,    // Threshold für HypothesisProposals
    .min_analogy_confidence = 0.4,       // Threshold für AnalogyProposals
    .min_contradiction_severity = 0.5,   // Threshold für ContradictionProposals
    .max_proposals_per_cycle = 10,       // Rate limiting
    .verbose_logging = false             // Debug-Ausgabe
};
```

## Architektur-Garantien

### Enforcement-Mechanismen

1. **Compile-Time Enforcement**
   - `const` Methoden für LTM-Zugriff
   - Gelöschte Assignment-Operatoren
   - Struct-Konstruktoren erzwingen `HYPOTHESIS`

2. **Runtime Enforcement**
   - Verifikation in `analyze_meaning()`:
     ```cpp
     for (const auto& proposal : proposals) {
         if (proposal.epistemic_type != EpistemicType::HYPOTHESIS) {
             throw std::logic_error("EPISTEMIC VIOLATION");
         }
     }
     ```
   - Bounded values via `clamp()`
   - Vollständiges Logging

3. **Test Enforcement**
   - 8 umfassende Tests
   - Verifizieren epistemische Invarianten
   - Automatisierte Verifikation

### Determinismus

**Garantie:** Gleicher Input → gleicher Output

**Ausnahme:** Wenn `enable_parallel_llms = true`, kann Reihenfolge variieren, aber Inhalt bleibt deterministisch.

### Nebenläufigkeit

Mini-LLMs dürfen parallel laufen, aber:
- Epistemische Entscheidungen bleiben seriell
- Nur im Epistemic Core
- Understanding Layer bleibt zustandslos

## Stub Mini-LLM

Für Tests und Entwicklung ohne echtes LLM:

```cpp
auto understanding = UnderstandingLayer();
understanding.register_mini_llm(std::make_unique<StubMiniLLM>());
```

**StubMiniLLM** generiert einfache Vorschläge zum Testen:
- Verifiziert epistemische Invarianten
- Testet READ-ONLY Zugriff
- Keine echte Semantik

## Erweiterung mit echten Mini-LLMs

### Eigenes Mini-LLM implementieren

```cpp
class MyMiniLLM : public MiniLLM {
public:
    std::string get_model_id() const override {
        return "my-custom-llm-v1.0";
    }

    std::vector<MeaningProposal> extract_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    ) const override {
        std::vector<MeaningProposal> proposals;

        // CRITICAL: Only READ-ONLY LTM access
        for (ConceptId cid : active_concepts) {
            auto concept_info = ltm.retrieve_concept(cid);  // READ-ONLY
            if (concept_info.has_value()) {
                // Generate semantic interpretation
                // ...
            }
        }

        // CRITICAL: All proposals MUST be HYPOTHESIS
        proposals.emplace_back(
            next_id++,
            active_concepts,
            "interpretation",
            "reasoning",
            0.7,  // confidence
            get_model_id()
        );

        return proposals;
    }

    // Implement other methods...
};
```

### Integration mit Ollama/llama.cpp

```cpp
class OllamaMiniLLM : public MiniLLM {
private:
    std::string model_name_;
    std::string endpoint_;

public:
    OllamaMiniLLM(const std::string& model, const std::string& endpoint)
        : model_name_(model), endpoint_(endpoint) {}

    std::vector<MeaningProposal> extract_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context
    ) const override {
        // 1. Build prompt from activated concepts (READ-ONLY)
        std::string prompt = build_prompt(active_concepts, ltm);

        // 2. Call Ollama API
        std::string response = call_ollama_api(prompt);

        // 3. Parse response and create proposals
        auto proposals = parse_response_to_proposals(response);

        // 4. CRITICAL: Ensure all are HYPOTHESIS
        for (auto& prop : proposals) {
            prop.epistemic_type = EpistemicType::HYPOTHESIS;
        }

        return proposals;
    }
};
```

## Dateien

```
backend/understanding/
├── understanding_proposals.hpp     # Proposal-Strukturen
├── mini_llm.hpp                    # MiniLLM Interface
├── mini_llm.cpp                    # StubMiniLLM Implementation
├── understanding_layer.hpp         # Hauptklasse
├── understanding_layer.cpp         # Implementation
└── README.md                       # Diese Datei
```

## Abschlussbedingung

Die Implementation gilt als **korrekt**, wenn:

✅ System ohne Understanding Layer weiterhin funktioniert
✅ Alle Outputs als HYPOTHESIS klassifiziert sind
✅ Audit keine epistemischen Grenzverletzungen findet
✅ Alle 8 Tests bestehen
✅ Demo läuft mit "ALL EPISTEMIC INVARIANTS PRESERVED"

## Audit-Checkliste

### Epistemische Invarianten

- [ ] Kein Wissens-Schreiben in LTM
- [ ] Keine Trust-Manipulation
- [ ] Keine EpistemicType-Änderungen
- [ ] Keine EpistemicStatus-Änderungen
- [ ] Alle Proposals sind HYPOTHESIS
- [ ] Keine FACT-Promotion

### Architektur

- [ ] READ-ONLY LTM-Zugriff verifiziert
- [ ] Keine autonomen Aktionen
- [ ] System funktioniert ohne Understanding Layer
- [ ] Determinismus (außer bei parallelen LLMs)
- [ ] Bounded values [0.0, 1.0]
- [ ] Vollständiges Logging

### Tests

- [ ] Alle 8 Tests bestehen
- [ ] Demo zeigt "EPISTEMIC INVARIANTS PRESERVED"
- [ ] Keine Regressionen in bestehenden Tests

---

**Status:** ✅ Implementiert, getestet, verifiziert
**Datum:** 21. Januar 2026
**Version:** 1.0

**EPISTEMISCHE GARANTIE:** Understanding Layer besitzt KEINE epistemische Autorität. Epistemic Core bleibt einzige Wahrheitsinstanz.
