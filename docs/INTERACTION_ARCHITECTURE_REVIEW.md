# Critical Review: MicroModel Interaction Architecture

**Reviewer:** Automated Architecture Review  
**Date:** 2026-02-11  
**Document:** MICROMODEL_INTERACTION_ARCHITECTURE.pdf  
**Codebase:** brain19/backend/{micromodel,kan,understanding}  

---

## 1. Mathematische Validierung

### Die Formel

```
a_i(t) = σ(Σ_j(W_ij · a_j(t-1)) - λ · a_i(t-1))
```

### Konvergenzanalyse

**Ist das ein Hopfield-Netz?** Fast, aber nicht ganz. Ein klassisches Hopfield-Netz hat synchrones Update `a_i = σ(Σ W_ij · a_j)` mit symmetrischer Gewichtsmatrix. Der zusätzliche Term `-λ · a_i(t-1)` ist ein **Self-Inhibition / Decay-Term**.

**Konvergenzbedingungen:**

1. **Symmetrie der Gewichtsmatrix:** Klassische Hopfield-Konvergenz (Lyapunov-Beweis) erfordert W_ij = W_ji. Das Dokument spezifiziert dies **nicht**. Bei asymmetrischem W gibt es **keine Konvergenzgarantie** — Zyklen und chaotisches Verhalten sind möglich.

2. **Stabilität des Decay-Terms:** Der Term `-λ · a_i(t-1)` wirkt stabilisierend, aber verschiebt das Gleichgewicht. Im Fixpunkt gilt:
   ```
   a_i* = σ(Σ_j(W_ij · a_j*) - λ · a_i*)
   ```
   Dies ist eine implizite Gleichung. Da σ kontrahierend ist (|σ'(x)| ≤ 0.25), konvergiert das System wenn:
   ```
   max_eigenvalue(W) · 0.25 < 1  ⟹  ρ(W) < 4
   ```
   Wobei ρ(W) der Spektralradius ist.

3. **Praktisches Problem:** Die W_ij werden im Dokument nicht definiert. Sind es die 10x10 MicroModel-Gewichte? Die KG-Kantengewichte? Neue Inter-Modell-Gewichte? **Das Dokument lässt dies offen.**

**Energie-Funktion:**
Das vorgeschlagene `E = -0.5 · Σ W_ij · a_i · a_j` gilt nur für symmetrisches W ohne Decay. Mit dem λ-Term wäre die korrekte Energie:
```
E = -0.5 · Σ W_ij · a_i · a_j + λ · Σ_i ∫σ⁻¹(a_i) da_i
```
Das Dokument ignoriert dies.

### Bewertung: ⚠️ Bedingtes OK

Die Formel konvergiert unter der Bedingung ρ(W) < 4, was bei kleinen Subgraphen (20-100 Knoten) mit normalisierten Gewichten realistisch ist. Aber:
- **Symmetrie muss erzwungen werden** (W_ij = W_ji), sonst keine Konvergenzgarantie
- **λ muss korrekt gewählt werden**: zu groß → alles wird unterdrückt, zu klein → Oszillation
- **Empfehlung:** λ ∈ [0.1, 0.5], adaptive basierend auf Subgraph-Größe

---

## 2. Dimensionalitäts-Check: Vec10 (10D)

### Ist 10D ausreichend?

**Kapazitätsanalyse:** Ein 10D-Vektor kann maximal 2^10 = 1024 orthogonale Richtungen unterscheiden (binär). Bei reellen Werten mit Cosine-Similarity können ~50-100 semantisch distinkte Konzepte sinnvoll separiert werden (empirische Faustregel: ~5-10× Dimensionen).

**Problem:** Bei 1000+ Konzepten reichen 10D **definitiv nicht** für globale Embeddings. Allerdings sind die Embeddings hier **lokal** — jedes MicroModel hat seine eigene 10x10 Matrix. Die Frage ist also: Reichen 10D für die *lokale Interaktion* eines Konzepts mit seinen ~5-20 Nachbarn?

**Antwort:** Für lokale Relationsscores (binäre Relevanz) sind 10D **gerade noch akzeptabel**. Für semantische Nuancen beim Sprachverständnis sind sie **zu wenig**.

### Vergleichswerte

| System | Embedding-Dim | Konzepte |
|--------|:------------:|:--------:|
| Word2Vec | 300 | 3M |
| GloVe | 50-300 | 400K |
| Brain19 MicroModel | **10** | 1000+ |
| ConceptNet Numberbatch | 300 | 500K |
| Minimale brauchbare Embeddings | ~32-64 | 1000-10K |

### Empfehlung: 🔴 Erhöhen auf mindestens 32D

- **Minimum für Sprachverständnis:** 32D (mit dediziertem Training)
- **Komfortabel:** 64D
- **Kostensteigerung:** 10→32 bedeutet 10× mehr Matrixoperationen (32² vs 10²), aber bei 1000 Modellen à 32×32 = 1024 params sind das nur ~1M Parameter total — trivial

---

## 3. Skalierbarkeit

### Komplexitätsanalyse der Interaction Phase

**Pro Zyklus:**
```
Für jeden Knoten i ∈ S (|S| = n):
    Summiere über Nachbarn j ∈ neighbors(i)
    → O(degree(i)) pro Knoten
Total pro Zyklus: O(Σ degree(i)) = O(|E_S|) = O(edges in subgraph)
```

**Das ist NICHT O(n²)!** Sondern O(|E_S|), also proportional zur Anzahl der Kanten im Subgraph. Bei sparse Graphen (typisch degree ~5-15) ist das O(n · d̄) ≈ O(n · 10).

**Für T Zyklen à n = 100 Knoten, d̄ = 10:**
```
100 · 10 · T = 1000T Operationen pro Propagation
Bei T = 10 Zyklen: 10.000 Operationen
```

**Jede "Operation" involviert aber ein MicroModel-predict():**
```
predict(e, c) = σ(eᵀ · (W·c + b))
  = 10×10 Matmul + 10D Dot Product
  = 100 + 10 = 110 FLOPs
```

**Total:** 10.000 × 110 = **1.1M FLOPs pro Interaction Phase**

### Auf Hardware:
| Hardware | FLOPS (double) | Zeit für 1.1M FLOPs |
|----------|:--------------:|:-------------------:|
| i5-6600K (1 core) | ~10 GFLOPS | **~0.1 μs** |
| i5-6600K (4 cores) | ~40 GFLOPS | ~0.03 μs |
| EPYC 80C | ~800 GFLOPS | ~0.001 μs |
| RTX 2070 | ~200 GFLOPS (fp64) | ~0.005 μs |

### Bewertung: ✅ Absolut kein Problem

Selbst bei pessimistischen Annahmen (Cache-Misses, Branch-Misprediction, 100× Overhead) bleibt die Interaction Phase **unter 1ms**. Das Bottleneck wird die **Relevance Phase** sein (compute() iteriert über ALLE Konzepte pro Source).

**Tatsächliches Skalierungsproblem:** `RelevanceMap::compute()` ist O(N) über alle Konzepte im LTM. Bei 10.000 Konzepten → 10.000 predict()-Aufrufe pro Source-Konzept. Das ist 1.1M FLOPs pro RelevanceMap — immer noch schnell, aber der wahre Bottleneck.

---

## 4. Architektur-Lücken

### 4.1 W_ij ist undefiniert 🔴

Das größte Problem: **Was sind die Inter-Modell-Gewichte W_ij?** Das Dokument sagt:
- "input_i = sum(W_ij * a_j(t-1)) for j in neighbors(i)"
- Aber jedes MicroModel hat seine eigene 10×10 Matrix W

**Optionen:**
1. W_ij = KG-Kantengewicht (skalar) → dann ist die Formel trivial und die MicroModels werden nicht genutzt
2. W_ij = predict(e_ij, a_j) → dann ist die Aktivierung der Output des MicroModels i → sinnvoll aber teuer
3. W_ij = gelernte Inter-Modell-Gewichte → völlig neues System, nicht im Code

**Empfehlung:** Option 2 ist die einzig sinnvolle: `a_i(t) = σ(Σ_j predict_i(e_rel, a_j(t-1)) - λ · a_i(t-1))`. Aber dann ist die Aktivierung ein **Skalar** (predict gibt Skalar zurück), und die Aktivierungsvektoren sind keine Vec10 mehr.

### 4.2 Aktivierung = Skalar vs. Vektor 🔴

Das Dokument sagt "Activation Vector" aber `predict()` gibt einen **Skalar** zurück. Es gibt zwei mögliche Interpretationen:
1. **a_i ist Skalar** → einfach, aber dann kein "activation vector" für Analogie-Erkennung
2. **a_i ist Vec10** → braucht neuen Forward-Pass der 10D zurückgibt (z.B. `v = W·c + b` ohne das finale Dot-Product)

**Empfehlung:** Neuen Modus `MicroModel::activate(c) → Vec10` einführen, der den Zwischenvektor `v = W·c + b` zurückgibt. Dann sind Aktivierungsvektoren vergleichbar.

### 4.3 Subgraph-Selektion nicht spezifiziert 🟡

"Top-K relevant concepts" — basierend auf was? Initial braucht man bereits eine RelevanceMap, die ihrerseits die Interaction Phase motiviert. Zirkuläre Abhängigkeit.

**Empfehlung:** Seed-Selektion über einfache String-Matching oder pre-computed TF-IDF, nicht über MicroModel-Relevanz.

### 4.4 Kein Lernmechanismus für Inter-Modell-Gewichte 🟡

Das Dokument beschreibt Propagation, aber nicht wie die Interaktionsgewichte gelernt werden. Die bestehenden MicroModels lernen `predict(e, c)` über Trainingssamples. Aber wer generiert die Trainingssamples für die Interaction Phase?

### 4.5 HYPOTHESIS-Modus Edges: Memory-Leak-Gefahr 🟡

Temporäre Kanten im HYPOTHESIS-Modus müssen aktiv aufgeräumt werden. Kein GC-Mechanismus beschrieben.

---

## 5. Gegenvorschläge

### 5.1 Sparse Attention statt Hopfield Propagation ✨

**Problem:** Hopfield-Netze konvergieren zu lokalen Minima und haben begrenzte Kapazität (~0.14N Muster für N Knoten).

**Alternative:** Graph Attention Network (GAT)-ähnlicher Mechanismus:
```
α_ij = softmax_j(LeakyReLU(a^T · [W·h_i || W·h_j]))
h_i' = σ(Σ_j α_ij · W · h_j)
```
- Keine Symmetrie-Anforderung
- Lernbare Attention-Gewichte
- Besser für heterogene Graphen

**Empfehlung:** Nicht sofort umsetzen, aber als Phase-2-Upgrade einplanen.

### 5.2 Sparse Propagation mit Top-K Neighbors

Statt über alle Nachbarn zu propagieren: nur Top-K (3-5) stärkste Verbindungen nutzen. Reduziert Rauschen und beschleunigt Konvergenz.

### 5.3 Message-Passing statt Activation Propagation

Modern Graph Neural Network (GNN) Paradigma:
```
m_ij = MLP(h_i, h_j, e_ij)        // Message
h_i' = AGG({m_ij : j ∈ N(i)})      // Aggregate
h_i'' = UPDATE(h_i, h_i')           // Update
```
Passt besser zu den vorhandenen MicroModels, da `predict(e, c)` natürlich als Message-Funktion dient.

---

## 6. Integration mit bestehendem Code

### Aktueller Stand

```cpp
// MicroModel: predict(e, c) → σ(eᵀ · (W·c + b)) → scalar ∈ (0,1)
// Registry: ConceptId → MicroModel (1:1 mapping)
// EmbeddingManager: RelationType → Vec10, Context → Vec10
// RelevanceMap: Source × All_Targets → scores (scalar map)
```

### Was muss geändert werden

| Änderung | Aufwand | Priorität |
|----------|:-------:|:---------:|
| `MicroModel::activate(c) → Vec10` (neuer Modus) | Klein | 🔴 Hoch |
| `InteractionSubgraph` Klasse (Knoten + Kanten) | Mittel | 🔴 Hoch |
| `ActivationPropagator` Klasse (Iterationslogik) | Mittel | 🔴 Hoch |
| `EMBED_DIM` 10→32 Migration | Groß* | 🟡 Mittel |
| Inter-Modell-Gewichte W_ij Datenstruktur | Mittel | 🟡 Mittel |
| Convergence-Monitor | Klein | 🟢 Niedrig |

*\*EMBED_DIM-Änderung: Betrifft ALLE MicroModels, Serialisierung, Tests. Sollte als Template-Parameter gemacht werden: `MicroModel<DIM>` statt hartcodiertem `EMBED_DIM=10`.*

### Konkrete Code-Änderung für activate():

```cpp
// In micro_model.hpp — neuer Modus
Vec10 activate(const Vec10& c) const {
    Vec10 v;
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        double sum = b_[i];
        for (size_t j = 0; j < EMBED_DIM; ++j) {
            sum += W_[i * EMBED_DIM + j] * c[j];
        }
        v[i] = sigmoid(sum);  // Element-wise sigmoid
    }
    return v;
}
```

### Interaction Phase Pseudocode (integriert):

```cpp
class ActivationPropagator {
    std::unordered_map<ConceptId, Vec10> activations_;
    double lambda_ = 0.3;
    double epsilon_ = 1e-4;
    size_t max_cycles_ = 10;

    void propagate(const KnowledgeGraph& kg,
                   MicroModelRegistry& registry,
                   EmbeddingManager& embeddings,
                   const std::vector<ConceptId>& subgraph) {
        // Initialize from RelevanceMap scores
        for (auto cid : subgraph) {
            activations_[cid] = {}; // from initial relevance
        }

        for (size_t t = 0; t < max_cycles_; ++t) {
            auto prev = activations_;
            for (auto cid : subgraph) {
                auto* model = registry.get_model(cid);
                if (!model) continue;
                Vec10 input{};
                for (auto [neighbor, rel] : kg.neighbors(cid)) {
                    auto& e = embeddings.get_relation_embedding(rel);
                    double w = model->predict(e, prev[neighbor]);
                    for (size_t d = 0; d < EMBED_DIM; ++d)
                        input[d] += w * prev[neighbor][d];
                }
                for (size_t d = 0; d < EMBED_DIM; ++d) {
                    activations_[cid][d] = sigmoid(
                        input[d] - lambda_ * prev[cid][d]
                    );
                }
            }
            if (converged(prev, activations_, epsilon_)) break;
        }
    }
};
```

---

## 7. Realismus-Check

### Hardware-Anforderungen

**Szenario:** 1000 Konzepte, 1500 Relationen, Subgraph = 50 Knoten, 10 Zyklen

| Komponente | FLOPs | RAM |
|-----------|------:|----:|
| 1000 MicroModels (10×10) | — | 430 × 8B × 1000 = **3.4 MB** |
| Interaction Phase (50 nodes, 10 cycles) | ~55K | negligible |
| RelevanceMap (1000 targets) | ~110K | ~8 KB |
| KG-Adjazenzliste | — | ~50 KB |
| **Total** | **<200K FLOPs** | **<5 MB** |

### Auf den Zielplattformen:

| Platform | Machbar? | Anmerkung |
|----------|:--------:|-----------|
| i5-6600K (4C/4T) | ✅ Ja | Unter 1ms für alles. CPU-bound kein Problem. |
| EPYC 80C | ✅ Absolut | Kann 80 Interaction Phases parallel ausführen |
| RTX 2070 (8GB) | ✅ Overkill | GPU lohnt erst ab >100K Konzepten. Kernel-Launch-Overhead dominiert bei 1000 Konzepten. **Empfehlung: Bleibt auf CPU.** |

### Wenn EMBED_DIM → 32:

| Komponente | RAM Δ |
|-----------|------:|
| 1000 MicroModels (32×32) | 32² × 8B × 1000 ≈ **8 MB** |
| FLOPs pro predict() | 32² + 32 = 1056 (10× mehr) |
| Interaction Phase | ~500K FLOPs → immer noch <1ms |

**Fazit: Auch mit 32D auf i5-6600K kein Problem.**

---

## Zusammenfassung

| Aspekt | Bewertung | Kritisch? |
|--------|:---------:|:---------:|
| Mathematische Korrektheit | ⚠️ Bedingt | Symmetrie erzwingen |
| Konvergenz | ⚠️ Bedingt | ρ(W) < 4 sicherstellen |
| Energie-Funktion | 🔴 Falsch | Decay-Term fehlt |
| 10D Dimensionalität | 🔴 Zu wenig | Min. 32D |
| Skalierbarkeit | ✅ OK | O(n·d̄·T), nicht O(n²) |
| Hardware-Machbarkeit | ✅ OK | Selbst auf i5 trivial |
| W_ij Definition | 🔴 Fehlt | Kritische Lücke |
| Skalar vs. Vektor | 🔴 Unklar | activate() → Vec nötig |
| Integration mit Code | 🟡 Machbar | Moderate Änderungen |
| Lernmechanismus | 🟡 Fehlt | Wer trainiert Interaktion? |

### Top-3 Empfehlungen

1. **Sofort:** W_ij klar definieren — `predict(e_rel, activation_neighbor)` als Inter-Modell-Gewicht verwenden
2. **Kurzfristig:** `activate() → Vec10` Modus einführen, EMBED_DIM als Template-Parameter
3. **Mittelfristig:** EMBED_DIM auf 32 erhöhen, Message-Passing-Paradigma evaluieren

### Gesamturteil

Die Architektur ist **konzeptionell solide aber unvollständig spezifiziert**. Die Kernidee — graph-constrained activation propagation über unabhängige MicroModels — ist originell und computationally günstig. Die Hauptprobleme sind:

1. Mathematische Ungenauigkeiten (Energie-Funktion, Konvergenzbedingungen)
2. Fehlende Spezifikation der Inter-Modell-Kopplung (W_ij)
3. 10D ist zu restriktiv für Sprachverständnis

Keines dieser Probleme ist ein Showstopper. Mit den oben genannten Korrekturen ist die Architektur **realistisch implementierbar** auf der beschriebenen Hardware.
