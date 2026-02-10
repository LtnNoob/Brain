# Brain19 — Architecture Overview

> **Status:** Korrigierte Gesamtarchitektur (Februar 2026)  
> **Zweck:** Definitive Referenz für Brain19's Architektur und Designphilosophie

---

## Was Brain19 ist

Brain19 ist ein **externalisiertes Arbeitsgedächtnis** — eine C++20 Cognitive Architecture, entwickelt für Menschen mit ADHS und Autismus. Es versteht Kontexte dauerhaft, erinnert proaktiv, erkennt Muster und passt sich an individuelle Bedürfnisse an.

Brain19 ist **kein LLM-Replacement**. Es ist ein eigenständig denkendes System mit epistemischer Integrität — es weiß, was es weiß, und was es nicht weiß.

---

## Kernprinzip: Brain19 denkt selbständig

Das häufigste Missverständnis: Brain19 nutze ein LLM zum Denken. **Das ist falsch.**

Alles kognitive Arbeiten — Relevanzberechnung, logische Inferenz, Kreativität, Validierung — geschieht durch **bilineare MicroModels** mit je 430 Parametern pro Konzept. Kein LLM ist im kritischen Denkpfad.

### MicroModel-Architektur

Jedes Konzept im Knowledge Graph besitzt ein eigenes MicroModel. Die Forward-Berechnung:

```
v = W·c + b        (10D Vektor)
z = eᵀ · v         (Skalar)
w = σ(z)            (Relevanz ∈ (0,1))
```

Wobei:
- `e ∈ ℝ¹⁰` — Relation-Embedding
- `c ∈ ℝ¹⁰` — Kontext-Embedding  
- `W ∈ ℝ¹⁰ˣ¹⁰` — Gewichtsmatrix (100 Parameter)
- `b ∈ ℝ¹⁰` — Bias (10 Parameter)
- `σ` — Sigmoid-Aktivierung

**430 Parameter pro Konzept. Kein Overhead. Kein LLM. Reine Mechanik.**

Training erfolgt mit Adam-Optimizer, komplett in C++ implementiert — keine externen Dependencies.

→ Siehe: [`backend/micromodel/micro_model.hpp`](../backend/micromodel/micro_model.hpp)

---

## Systemarchitektur

```
┌──────────────────────────────────────────────────────────┐
│                     BrainController                       │
│              (Orchestrierung, Delegation)                  │
├─────────┬──────────┬──────────────┬──────────────────────┤
│   STM   │   LTM    │  Cognitive   │   Epistemic System   │
│  Short- │  Long-   │  Dynamics    │   6 Types, 4 States  │
│  Term   │  Term    │  Spreading   │   Compile-Time       │
│  Memory │  Memory  │  Activation  │   Enforcement        │
│         │  (KG)    │  + Salience  │                      │
├─────────┴──────────┴──────────────┴──────────────────────┤
│                                                           │
│  ┌─────────────────┐    ┌────────────────────────────┐   │
│  │  Curiosity      │    │  MicroModel Layer          │   │
│  │  Engine         │    │  (430 Params/Konzept)      │   │
│  │  Trigger für    │───→│  Relevanz-Maps             │   │
│  │  Exploration    │    │  Kreativität durch         │   │
│  └─────────────────┘    │  Map-Überlagerung          │   │
│                          └────────────────────────────┘   │
│                                                           │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Understanding Layer (OPTIONAL)                    │   │
│  │  LLM als Sprachinterface — NICHT zum Denken       │   │
│  │  Alle Outputs: HYPOTHESIS (max Trust 0.3-0.5)     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                           │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Snapshot Generator                                │   │
│  │  Vollständige Systemzustands-Inspektion            │   │
│  └────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

### Die 9 Subsysteme

1. **STM** — Short-Term Memory (Arbeitsgedächtnis)
2. **LTM** — Long-Term Memory (Knowledge Graph)
3. **Cognitive Dynamics** — Spreading Activation + Salience
4. **Curiosity Engine** — Trigger für Exploration und Kreativität
5. **Epistemic System** — Wahrheitsbewertung mit Compile-Time-Enforcement
6. **MicroModel Layer** — Bilineare Relevanz-Maps (430 Params/Konzept)
7. **Understanding Layer** — LLM-Sprachinterface (optional)
8. **Sleep Cycle** — Offline-Konsolidierung und Selbstverbesserung
9. **Snapshot Generator** — Systemzustands-Inspektion

---

## Kreativität ohne LLM

Brain19 erzeugt Kreativität durch **Überlagerung von MicroModel-Relevanz-Maps**:

```
Map("Temperatur")  ──┐
                      ├──→  Überlagerung  ──→  Unerwartete Muster
Map("Musik")       ──┘                         Neue Hypothesen
```

Wenn die Relevanz-Map von "Temperatur" und die von "Musik" überlagert werden, zeigen sich unerwartete gemeinsame Relevanzen — Verbindungen, die kein einzelnes Konzept allein sichtbar macht.

**Kombinations-Methoden:**
- Multiplikation: Findet gemeinsam wichtige Relationen
- Harmonisches Mittel: Betont Überlappung
- Überraschungsbasiert: `|w₁ - w₂| · max(w₁, w₂)` — findet asymmetrische Relevanzen

Die Curiosity Engine triggert diese Überlagerungen basierend auf Spreading Activation und Salience. **Emergente Kreativität, vollständig deterministisch, vollständig inspizierbar.**

---

## LLM: Nur Sprachinterface

Das LLM in Brain19 hat genau **eine** Aufgabe: strukturierten System-Output in menschliche Sprache übersetzen. Es ist ein **Verbalizer**, kein Denker.

```
Brain19 Denkprozess        LLM Sprachinterface
━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━
MicroModel-Inferenz   →    "Basierend auf den
Spreading Activation  →     Zusammenhängen zwischen
Salience Scores       →     Temperatur und Druck
Epistemische Werte    →     ergibt sich..."
                             
(DENKEN)                    (SPRECHEN)
```

### Kahneman-Analogie

| Rolle | Brain19-Komponente | Funktion |
|-------|-------------------|----------|
| **System 2** (logisch, präzise) | MicroModels + Epistemic System | Denken, Validieren, Entscheiden |
| **System 1** (assoziativ, schnell) | LLM (optional) | Nur Vorschläge, keine Autorität |

Wenn das LLM optional für kreative Hypothesen-Generierung eingesetzt wird, durchlaufen diese **immer** die epistemische Validierung. LLM-Proposals erhalten ein Trust-Ceiling von 0.3–0.5 und werden nie automatisch akzeptiert.

→ Siehe: [`docs/DESIGN_THEORY.md`](DESIGN_THEORY.md), [`docs/KAN_LLM_HYBRID_THEORY.md`](KAN_LLM_HYBRID_THEORY.md)

---

## Epistemische Integrität

Brain19's epistemisches System ist kein Feature — es ist das **Fundament**.

### Compile-Time-Enforcement

```cpp
ConceptInfo() = delete;  // Kein Konzept ohne epistemische Klassifikation
```

Es ist **unmöglich**, ein Konzept in den Knowledge Graph einzufügen, ohne seinen epistemischen Status anzugeben. Dies wird zur Compile-Zeit erzwungen — nicht zur Laufzeit, nicht durch Konvention, sondern durch den Compiler.

### Trust-Scores

| Epistemischer Typ | Trust-Bereich | Bedeutung |
|-------------------|---------------|-----------|
| **FACT** | 0.98–0.99 | Verifiziert, reproduzierbar |
| **THEORY** | ~0.95 | Evidenzbasiert, falsifizierbar |
| **HYPOTHESIS** | ~0.50 | Testbar, noch unbestätigt |
| **SPECULATION** | ~0.30 | Keine Evidenz, Idee |
| **LLM-Proposal** | max 0.30–0.50 | Trust-Ceiling, nie höher ohne Validierung |

### Keine Halluzinationen

Brain19 kann nicht halluzinieren. Das System weiß exakt:
- Was es weiß (FACT, THEORY)
- Was es vermutet (HYPOTHESIS)
- Was es nicht weiß (fehlende Konzepte)
- Was unzuverlässig ist (SPECULATION, LLM-Proposals mit niedrigem Trust)

LLM-Output wird **immer** epistemisch validiert bevor er in den Knowledge Graph kommt. Widersprüche zu existierendem FACT/THEORY-Wissen führen zur automatischen Ablehnung.

→ Siehe: [`docs/KAN_LLM_HYBRID_THEORY.md`](KAN_LLM_HYBRID_THEORY.md) (Teil I: Epistemische Integrität)

---

## Unendliche Skalierung

Brain19's Architektur hat **kein hardcoded Limit**. Das System skaliert mit der verfügbaren Hardware.

### Streams = Parallele Denkprozesse

Jeder Stream ist unabhängig: eigene MicroModel-Inferenz, eigene Relevanz-Map-Kombinationen, eigene Spreading Activation. Streams teilen sich den Knowledge Graph (read-only), aber ihre Berechnungen sind vollständig parallel.

```
┌────────────────────────────────────────────────┐
│              Knowledge Graph (shared)            │
├──────┬──────┬──────┬──────┬────────────────────┤
│ Core │ Core │ Core │ Core │  ...               │
│  1   │  2   │  3   │  4   │                    │
│  ↓   │  ↓   │  ↓   │  ↓   │                    │
│Stream│Stream│Stream│Stream│  N Streams         │
│  1   │  2   │  3   │  4   │  = N Cores         │
└──────┴──────┴──────┴──────┴────────────────────┘
```

| Hardware | Streams | Einsatz |
|----------|---------|---------|
| i5-6600K (4 Cores) | 4 | Entwicklung |
| EPYC 80-Core | 80 | Produktiv |
| 10× EPYC Cluster | 800 | Massiv parallel |
| + Photonische NPU | + optische Streams | Hybrid |

### Warum das funktioniert

- **Lock-free Design:** Keine Mutexe, keine Bottlenecks
- **MicroModels sind unabhängig:** Jedes Konzept hat sein eigenes Modell, keine shared weights
- **Config-driven:** System erkennt verfügbare Hardware und nutzt sie automatisch
- **Lineare Skalierung:** Doppelte Cores = doppeltes paralleles Denken

---

## Hardware

### Aktuell (Entwicklung)

| Komponente | Spezifikation |
|------------|--------------|
| CPU | Intel i5-6600K |
| GPU | NVIDIA RTX 2070 |
| RAM | 16 GB |

### Produktiv (vorhanden, nicht aktiv)

| Komponente | Spezifikation |
|------------|--------------|
| CPU | AMD EPYC ~80 Cores |
| RAM | 120 GB |
| Storage | 1 TB NVMe SSD |

Der EPYC-Server existiert, steht im Studio, braucht nur Strom. Auf 80 Kernen können 10.000 MicroModels in unter 1 Sekunde trainiert werden.

### Langfristig (spekulativ)

| Komponente | Status |
|------------|--------|
| Q.ANT Photonische NPU | Einzige spekulative Komponente |

→ Siehe: [`docs/KAN_RELEVANCE_MAPS_ANALYSIS.md`](KAN_RELEVANCE_MAPS_ANALYSIS.md) (Abschnitt 6: Praktische Machbarkeit)

---

## Zusammenfassung

| Eigenschaft | Realisierung |
|-------------|-------------|
| Selbständiges Denken | MicroModels (430 Params/Konzept) |
| Kreativität | Überlagerung von Relevanz-Maps |
| Sprache | LLM als Verbalizer (optional) |
| Wahrheit | Compile-Time epistemische Enforcement |
| Skalierung | 1 Core → ∞ Cores, linear |
| Halluzinationen | Unmöglich (Trust-System) |
| Zweck | Externalisiertes Arbeitsgedächtnis (ADHS/Autismus) |

---

## Referenzen

- [`PROJECT_VISION.md`](PROJECT_VISION.md) — Motivation und Zweck
- [`KAN_LLM_HYBRID_THEORY.md`](KAN_LLM_HYBRID_THEORY.md) — KAN-LLM Hybridarchitektur, epistemische Theorie
- [`KAN_RELEVANCE_MAPS_ANALYSIS.md`](KAN_RELEVANCE_MAPS_ANALYSIS.md) — MicroModel-Architekturanalyse
- [`DESIGN_THEORY.md`](DESIGN_THEORY.md) — Domain-Auto-LLM Theorie
- [`backend/micromodel/micro_model.hpp`](../backend/micromodel/micro_model.hpp) — MicroModel Implementation

---

*Felix Hirschpek, 2026*
