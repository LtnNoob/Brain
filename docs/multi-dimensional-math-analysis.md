# Multi-Dimensionale Mind-Map Architektur — Mathematische Analyse

**Datum:** 2026-02-14  
**Autor:** Mathematische Tiefenanalyse (Subagent)  
**Status:** FORMALE ANALYSE  
**Kontext:** Brain19 KG (29K Concepts, 87K Relations, 5.9M Patterns), aktuell ~3 Rel/Concept, ~50% mono-dimensional

---

## Inhaltsverzeichnis

1. [Formale Definition des Multi-Dimensionalen Concept-Raums](#1)
2. [Dimensionale Distanzmetrik](#2)
3. [Multi-Dimensionale Graph-Traversierung (FocusCursor)](#3)
4. [Integration mit ConceptModels (KAN)](#4)
5. [Pattern Discovery → Dimensionale Relations](#5)
6. [Informationstheoretische Analyse](#6)
7. [Auswirkung auf Language Generation](#7)
8. [Vergleich mit existierenden Formalismen](#8)

---

## 1. Formale Definition des Multi-Dimensionalen Concept-Raums <a name="1"></a>

### 1.1 Grunddefinitionen

Sei G = (C, R, τ, w) der Brain19 Knowledge Graph mit:
- C = {c₁, ..., cₙ}, N = 29.675 (Concept-Menge)
- R ⊆ C × C × T (Relationsmenge, |R| = 87.581)
- T = {t₁, ..., t₂₀} (Relationstypen, registriert in RelationTypeRegistry)
- τ: R → T (Typfunktion)
- w: R → [0,1] (Gewichtsfunktion, Trust/dynamic_weight)

Die existierende RelationTypeRegistry weist jedem Typ t ∈ T ein 16D-Embedding e(t) ∈ ℝ¹⁶ zu. Die Relationstypen sind in Kategorien K = {HIERARCHICAL, CAUSAL, COMPOSITIONAL, SIMILARITY, FUNCTIONAL, TEMPORAL, OPPOSITION, EPISTEMIC} gruppiert.

### 1.2 Dimensionale Zerlegung des Graphen

**Definition 1.1 (Semantische Dimension).** Eine semantische Dimension d ∈ D ist eine Partition der Relationsmenge R in dimensionale Schichten:

$$R = \bigsqcup_{d \in D} R_d \quad \text{wobei} \quad R_d = \{r \in R : \tau(r) \in T_d\}$$

und T_d ⊆ T die Relationstypen sind, die Dimension d zugeordnet werden.

Aus der bestehenden Kategorisierung ergibt sich eine natürliche Zerlegung mit |D| = 8 Kategorien. Wir verfeinern dies unten.

### 1.3 Gewichtete Adjazenzmatrizen pro Dimension

Für jede Dimension d ∈ D definieren wir die gewichtete Adjazenzmatrix:

$$A_d \in \mathbb{R}^{N \times N}, \quad (A_d)_{ij} = \sum_{\substack{r=(c_i, c_j, t) \in R \\ t \in T_d}} w(r)$$

Die Gesamt-Adjazenzmatrix ist:

$$A = \sum_{d \in D} A_d$$

**Fakten aus dem Code:**
- Aktueller mittlerer Grad: δ̄ = 2|R|/N = 2 × 87.581/29.675 ≈ 5.9 (ungerichtet)
- Gerichteter Out-Degree: δ̄_out ≈ 2.95
- Graph-Dichte: ρ = |R|/(N(N-1)) ≈ 10⁻⁴

### 1.4 Positionsvektor via Spectral Embedding

**Definition 1.2 (Dimensionaler Positionsvektor).** Für jedes Concept cᵢ definieren wir P(cᵢ) ∈ ℝᵏ, wobei k = |D|, durch Spectral Embedding des dimensionalen Graph-Laplacians.

Für Dimension d sei Lₐ der normalisierte Graph-Laplacian:

$$L_d = I - D_d^{-1/2} A_d D_d^{-1/2}$$

wobei D_d = diag(A_d · 𝟙) die Gradmatrix ist. Falls (D_d)ᵢᵢ = 0 (Concept hat keine Relation in Dimension d), setze (D_d)ᵢᵢ = ε = 10⁻⁶ (Regularisierung).

Die d-te Komponente von P(cᵢ) ist die **Fiedler-Koordinate** (zweiter kleinster Eigenwert):

$$L_d \, v_d^{(2)} = \lambda_d^{(2)} \, v_d^{(2)} \quad \Rightarrow \quad P(c_i)_d = (v_d^{(2)})_i$$

**Interpretation:** v_d^(2) partitioniert den Graphen optimal in zwei Hälften entlang Dimension d. Concepts nahe beieinander in v_d^(2) sind in dieser Dimension stark verbunden.

**Warum Fiedler-Vektor und nicht Random Walk?**

Random-Walk-Embedding (z.B. node2vec) erzeugt k-dimensionale Vektoren ohne klare semantische Zuordnung pro Dimension. Der Laplacian-Ansatz ordnet jeder semantischen Dimension eine eigene Achse zu — das ist für die Mind-Map-Navigation essentiell.

### 1.5 Integration der 5.9M Patterns

Die Patterns liefern eine Co-Aktivierungs-Matrix. Sei P = {p₁, ..., p_M}, M ≈ 5.9 × 10⁶, wobei jedes Pattern pⱼ eine Menge involved_concepts(pⱼ) ⊆ C und eine Confidence conf(pⱼ) ∈ [0,1] hat.

**Definition 1.3 (Pattern-gewichtete Co-Aktivierungsmatrix).**

$$M_{ij} = \sum_{\substack{p \in P \\ c_i, c_j \in \text{involved}(p)}} \text{conf}(p)$$

Diese Matrix wird als zusätzliche Gewichtung in die dimensionalen Adjazenzmatrizen eingebaut:

$$\tilde{A}_d = A_d + \alpha \cdot \Pi_d(M)$$

wobei Π_d die **dimensionale Projektion** von M ist. Die Projektion wird berechnet als:

$$(\Pi_d(M))_{ij} = M_{ij} \cdot \max_{t \in T_d} \text{sim}(e(t), \bar{e}_{ij})$$

Hier ist ē_ij das mittlere Relationstyp-Embedding über alle expliziten Relationen zwischen cᵢ und cⱼ (bzw. deren 1-Hop-Nachbarschaft falls keine direkte Relation existiert), und sim die Cosine-Similarity der 16D-Embeddings.

**Parameter α:** Steuert den Pattern-Einfluss. Aus dem Graph-Clustering-Plan: α = 0.4/0.4 = 1.0 (Pattern-Gewicht gleich stark wie strukturelles Gewicht).

### 1.6 Optimale Dimensionszahl k

**Methode 1: Aus der Kategoriestruktur.** Die 20 Relationstypen gruppieren sich in 8 Kategorien. Also k ≤ 8.

**Methode 2: Effektive Dimensionalität via Eigenwert-Analyse.**

Betrachte die symmetrisierte Gesamt-Adjazenzmatrix Ã = (A + Aᵀ)/2 und deren Laplacian L. Die Eigenwerte λ₁ ≤ λ₂ ≤ ... ≤ λₙ des Laplacians zeigen die Cluster-Struktur. Die **Spectral Gap** nach dem k-ten Eigenwert bestimmt die natürliche Dimensionszahl:

$$k^* = \arg\max_k \frac{\lambda_{k+1} - \lambda_k}{\lambda_k}$$

**Methode 3: Kaiser-Kriterium auf der Pattern-Matrix.**

Sei M = UΣVᵀ die SVD der Co-Aktivierungsmatrix. Das Kaiser-Kriterium behält Dimensionen mit σᵢ > σ̄ (Mittelwert der Singulärwerte):

$$k^* = |\{i : \sigma_i > \bar{\sigma}\}| \quad \text{mit} \quad \bar{\sigma} = \frac{1}{N}\sum_{i=1}^{N} \sigma_i$$

**Schätzung für Brain19:** Bei 8 Kategorien und einer Power-Law-Verteilung der Relations (~40% HIERARCHICAL) erwarten wir:
- Methode 1: k = 8
- Methode 2: k ≈ 5-7 (die seltenen Kategorien OPPOSITION/EPISTEMIC erzeugen keinen signifikanten Spectral Gap)
- Methode 3: k ≈ 6-12 (Pattern-Co-Aktivierung ist feiner als explizite Kategorien)

**Empfehlung: k = 8** (aus Kategorien), mit Option auf datengetriebene Verfeinerung zu k ∈ [6, 12] nach SVD-Analyse.

### 1.7 Berechnung von P(cᵢ) — Algorithmische Komplexität

Für jeden der k Laplacianer L_d den Fiedler-Vektor berechnen:
- **Exakt (Eigen-Dekomposition):** O(N³) = O(29.675³) ≈ 2.6 × 10¹³ → **unbrauchbar**
- **Lanczos-Iteration (sparse):** O(|R_d| × iter) mit iter ≈ 100 → O(10⁴ × 100) = O(10⁶) pro Dimension → **machbar**
- **Gesamt für k Dimensionen:** O(k × |R| × iter/k) = O(|R| × iter) ≈ O(8.7 × 10⁶) → **~50ms auf i5-6600K**

Die Sparse-Eigenschaft ist kritisch: |R_d| ≪ N² (Dichte 10⁻⁴).

---

## 2. Dimensionale Distanzmetrik <a name="2"></a>

### 2.1 Distanz innerhalb einer Dimension

**Definition 2.1 (Dimensionale Distanz).** Für Concepts cᵢ, cⱼ und Dimension d:

$$d_{\text{dim}}(c_i, c_j; d) = \left| P(c_i)_d - P(c_j)_d \right|$$

Dies ist der Abstand der Fiedler-Koordinaten in Dimension d. **Problem:** Skalierung — verschiedene Dimensionen haben verschiedene Eigenwert-Magnitudes.

**Normalisierung:** Nutze den zugehörigen Eigenwert als Skalierungsfaktor:

$$\hat{d}_{\text{dim}}(c_i, c_j; d) = \sqrt{\lambda_d^{(2)}} \cdot |P(c_i)_d - P(c_j)_d|$$

**Interpretation:** λ_d^(2) (algebraische Konnektivität) misst, wie gut Dimension d den Graphen verbindet. Dimensionen mit hoher Konnektivität (großes λ) produzieren informativere Distanzen.

### 2.2 Multi-Dimensionale Gesamtdistanz

**Definition 2.2 (Gewichtete Multi-Dimensionale Distanz).**

$$d_{\text{multi}}(c_i, c_j) = \sqrt{\sum_{d=1}^{k} w_d \cdot \hat{d}_{\text{dim}}(c_i, c_j; d)^2}$$

wobei w_d ∈ [0,1] die Dimensionsgewichte sind mit Σ_d w_d = 1.

### 2.3 Lernen der Dimensionsgewichte via Attention

Die Gewichte w_d sollen **kontextabhängig** sein — bei einer kausalen Frage ist die CAUSAL-Dimension wichtiger.

**Definition 2.3 (Dimensionale Attention).** Gegeben einen Query-Vektor q ∈ ℝ¹⁶ (aus dem KAN-Encoder):

$$w_d(q) = \frac{\exp(q^\top \cdot \bar{e}_d / \sqrt{16})}{\sum_{d'=1}^{k} \exp(q^\top \cdot \bar{e}_{d'} / \sqrt{16})}$$

wobei ē_d = (1/|T_d|) Σ_{t ∈ T_d} e(t) das mittlere Relationstyp-Embedding der Dimension d ist.

**Berechnung der Gesamt-Ähnlichkeit:**

$$\text{sim}_{\text{multi}}(c_i, c_j; q) = \exp\left(-d_{\text{multi}}(c_i, c_j; q)^2 / (2\sigma^2)\right)$$

mit Bandbreite σ als Hyperparameter (z.B. mittlere paarweise Distanz im Trainingsset).

### 2.4 Verbindung zu existierenden ConceptModel-Scores

Das ConceptModel berechnet `predict_refined()` mit Score ∈ (0,1). Die aktuelle Pipeline:

```
score = σ(eᵀ · (W·c + b))  →  MultiHead Bilinear  →  FlexKAN  →  refined_score
```

Die Verbindung zur dimensionalen Distanz ist:

**Proposition 2.1.** Sei s_CM(cᵢ, cⱼ, t) der ConceptModel-Score für Relation (cᵢ, cⱼ) vom Typ t ∈ T_d. Dann gilt approximativ:

$$s_{CM}(c_i, c_j, t) \approx \sigma\left(-\gamma \cdot \hat{d}_{\text{dim}}(c_i, c_j; d) + \beta\right)$$

wobei γ, β lernbare Parameter sind.

**Begründung:** Der bilineare Score σ(eᵀ(W·c+b)) misst implizit die "Passung" zweier Concepts relativ zu einem Relationstyp. Wenn die dimensionale Distanz klein ist (cᵢ, cⱼ nahe in Dimension d), sollte der Score für Relationstypen t ∈ T_d hoch sein. Die Sigmoid-Funktion σ transformiert den linearen Abstand in [0,1].

**Empirische Validierung:** Die aktuellen CM-Scores von 0.96-0.97 auf validierten Relationen implizieren, dass γ · d̂_dim + β ≈ 3.2-3.5 (da σ(3.2) ≈ 0.96). Für nicht-existierende Relationen erwarten wir d̂_dim groß → Score ≈ 0.

### 2.5 Metrische Eigenschaften

**Proposition 2.2.** d_multi(·, ·; q) ist eine Pseudometrik auf C für jedes feste q.

*Beweis:*
1. d_multi(cᵢ, cᵢ; q) = 0 ✓ (P(cᵢ) - P(cᵢ) = 0)
2. Symmetrie: d_multi(cᵢ, cⱼ; q) = d_multi(cⱼ, cᵢ; q) ✓ (|·| symmetrisch)
3. Dreiecksungleichung: Folgt aus der gewichteten euklidischen Norm ✓

Es ist eine Pseudometrik (nicht Metrik), weil d_multi(cᵢ, cⱼ; q) = 0 nicht cᵢ = cⱼ impliziert — verschiedene Concepts können die gleichen Fiedler-Koordinaten haben. □

---

## 3. Multi-Dimensionale Graph-Traversierung (FocusCursor) <a name="3"></a>

### 3.1 Aktuelles Modell

Aus `focus_cursor.cpp`: Der FocusCursor evaluiert jede ausgehende Kante mit `evaluate_edge()`:

```
score = ConceptModel.predict_refined(e_rel, c_mixed, concept_from, concept_to)
```

Dann wird der Kandidat mit maximalem Score gewählt (Greedy). Die Komplexität pro Hop:

$$T_{\text{step}} = O(\delta_{\text{out}}(c) \cdot C_{\text{predict}})$$

wobei δ_out(c) der Ausgangsgrad von c und C_predict die Kosten eines predict_refined-Calls sind. Mit δ̄_out ≈ 3 und der MultiHead+FlexKAN-Pipeline: C_predict = O(4 × 20 × 4 + 28 × 10) = O(600) FLOPs.

**Gesamt für eine Traversierung der Tiefe D:**

$$T_{\text{traverse}} = O(D \cdot \bar{\delta}_{\text{out}} \cdot C_{\text{predict}}) \approx O(D \cdot 1800)$$

### 3.2 Dimensionaler Transition-Tensor

**Definition 3.1 (Dimensionaler Transition-Tensor).** Definiere T ∈ ℝ^{N×N×k}:

$$(T_d)_{ij} = \frac{(\tilde{A}_d)_{ij}}{\sum_j (\tilde{A}_d)_{ij}} = \frac{(\tilde{A}_d)_{ij}}{(\tilde{D}_d)_{ii}}$$

wobei T_d die zeilenweise normalisierte Adjazenzmatrix in Dimension d ist (stochastische Matrix).

**Eigenschaft:** T_d ist die Übergangsmatrix eines Random Walk auf dem dimensionalen Subgraph. Für jedes d: Σⱼ (T_d)ᵢⱼ = 1 (falls Concept i in Dimension d Nachbarn hat).

### 3.3 Dimensionale Spreading Activation

**Definition 3.2 (Dimensionale Spreading Activation).** Sei a(t) ∈ ℝᴺ der Aktivierungsvektor zum Zeitpunkt t. Die Dynamik mit dimensionaler Gewichtung:

$$a^{(t+1)} = \sigma\left(\sum_{d=1}^{k} w_d(q) \cdot T_d \cdot a^{(t)} + \beta \cdot b_{\text{ext}}\right)$$

wobei:
- w_d(q) die Query-abhängigen Dimensionsgewichte (Def. 2.3)
- σ die elementweise Sigmoid-Funktion
- b_ext ∈ ℝᴺ ein externer Bias (Seeds, Query-Relevanz)
- β die Bias-Stärke

**Matrixform:** Definiere die gewichtete Transitionsmatrix

$$T_w(q) = \sum_{d=1}^{k} w_d(q) \cdot T_d \in \mathbb{R}^{N \times N}$$

Dann: a^(t+1) = σ(T_w(q) · a^(t) + β · b_ext).

### 3.4 Konvergenz-Beweis

**Theorem 3.1 (Konvergenz der dimensionalen Spreading Activation).** Unter den Bedingungen:
1. Jede T_d ist substochastisch (Σⱼ (T_d)ᵢⱼ ≤ 1)
2. Die Gewichte w_d ≥ 0 mit Σ_d w_d = 1
3. σ ist die Sigmoid-Funktion

konvergiert die Iteration a^(t) → a* für t → ∞.

*Beweis:*

**Schritt 1.** T_w(q) = Σ_d w_d T_d ist eine konvexe Kombination substochastischer Matrizen, also selbst substochastisch. Daher ‖T_w(q)‖_∞ ≤ 1.

**Schritt 2.** Definiere f(a) = σ(T_w · a + β · b_ext). Die Sigmoid-Funktion σ hat Lipschitz-Konstante L_σ = 1/4 (Maximum von σ'(x) = σ(x)(1-σ(x)) bei x=0).

**Schritt 3.** Für a, a' ∈ [0,1]ᴺ:

$$\|f(a) - f(a')\|_\infty \leq L_\sigma \cdot \|T_w \cdot (a - a')\|_\infty \leq \frac{1}{4} \cdot \|T_w\|_\infty \cdot \|a - a'\|_\infty \leq \frac{1}{4} \|a - a'\|_\infty$$

Da 1/4 < 1, ist f eine **Kontraktion** mit Kontraktionsrate 1/4.

**Schritt 4.** Nach dem Banachschen Fixpunktsatz existiert ein eindeutiger Fixpunkt a* = f(a*), und die Iteration konvergiert geometrisch:

$$\|a^{(t)} - a^*\|_\infty \leq \left(\frac{1}{4}\right)^t \cdot \|a^{(0)} - a^*\|_\infty$$

Nach t ≈ 14 Iterationen: (1/4)^14 ≈ 3.7 × 10⁻⁹ → Konvergenz bis Maschinengenauigkeit.

**Vergleich mit InteractionLayer:** Die bestehende InteractionLayer konvergiert in ~14 Zyklen (aus der Simulation in KAN_MINILLM_LANGUAGE_ENGINE.md, δ = 9.2×10⁻⁵ bei t=14). Dies stimmt mit unserer Analyse überein: (1/4)^14 ≈ 10⁻⁸, also reichen ~14 Iterationen für Konvergenz bis ε = 10⁻⁴. □

### 3.5 Attractor-Dynamik im dimensionalen Raum

**Definition 3.3 (Dimensionaler Attractor).** Ein Attractor A ⊆ C ist eine Menge von Concepts für die gilt:

$$\forall c_i \in A: \quad a^*_i > \theta \quad \text{und} \quad \exists d \in D: (T_d \cdot \mathbb{1}_A)_i > \eta$$

wobei θ die Aktivierungsschwelle und η die dimensionale Kohärenz-Schwelle ist.

**Interpretation:** Ein Attractor ist eine Gruppe von Concepts die sich gegenseitig in mindestens einer Dimension verstärken und alle überschwellig aktiviert sind.

**Proposition 3.1 (Dimensionale Attractor-Separation).** Wenn zwei Dimensionen d₁, d₂ schwach gekoppelt sind (formell: ‖T_{d₁} ∘ T_{d₂}‖_F < ε, wobei ∘ das Hadamard-Produkt ist), dann sind die Attractors in d₁ und d₂ **disjunkt** (bis auf Hub-Concepts).

*Beweis-Skizze:* Schwache Kopplung bedeutet, dass die Kanten in d₁ und d₂ verschiedene Concept-Paare verbinden. Ein Fixpunkt der Spreading Activation in d₁ aktiviert daher andere Concepts als in d₂. Hub-Concepts mit hohem Grad in beiden Dimensionen können in beiden Attractors auftreten, aber der Schnitt hat Maß O(ε). □

### 3.6 Komplexitätsreduktion

**Claim:** Dimensionale Filterung reduziert die effektive Kantenzahl pro Hop.

**Analyse:** Sei E_d = |R_d| die Kantenzahl in Dimension d. Bei uniformer Verteilung: E_d ≈ |R|/k. Aber die Verteilung ist nicht uniform — aus der Analyse in graph-density-analysis.md:

| Dimension | Geschätzter Anteil an |R| | E_d |
|:----------|:----------------------|:----|
| HIERARCHICAL | 40% | 35.032 |
| CAUSAL | 12% | 10.510 |
| COMPOSITIONAL | 17% | 14.889 |
| SIMILARITY | 7% | 6.131 |
| FUNCTIONAL | 10% | 8.758 |
| TEMPORAL | 4% | 3.503 |
| OPPOSITION | 2% | 1.752 |
| EPISTEMIC | 3% | 2.627 |
| CUSTOM | 5% | 4.379 |

**Traversierung in einer einzelnen Dimension d:**

$$T_{\text{step}}^{(d)} = O(\delta_{\text{out}}^{(d)}(c) \cdot C_{\text{predict}})$$

wobei δ_out^(d)(c) der dimensionale Ausgangsgrad ist. Im Mittel:

$$\bar{\delta}_{\text{out}}^{(d)} = \frac{E_d}{N}$$

Für CAUSAL: δ̄_out^(CAUSAL) = 10.510/29.675 ≈ 0.35. Vergleich mit gesamt δ̄_out ≈ 2.95.

**Speedup-Faktor für dimensionale Traversierung:**

$$S_d = \frac{\bar{\delta}_{\text{out}}}{\bar{\delta}_{\text{out}}^{(d)}} = \frac{|R|}{E_d}$$

| Dimension | Speedup |
|:----------|:--------|
| HIERARCHICAL | 2.5× |
| CAUSAL | 8.3× |
| SIMILARITY | 14.3× |
| TEMPORAL | 25× |

**Aber:** Dies ist ein Speedup pro Hop. Die Traversierungstiefe D kann sich erhöhen (da dimensionale Subgraphen sparser sind und mehr Hops brauchen, um relevante Concepts zu erreichen). Der Netto-Speedup hängt vom Query-Typ ab.

**Worst Case:** Wenn die Ziel-Information über mehrere Dimensionen verteilt ist, muss sequentiell durch mehrere Dimensionen traversiert werden. Dann: T_multi = Σ_d T^(d) ≈ T_gesamt (kein Speedup).

**Best Case:** Query betrifft nur eine Dimension. Speedup = |R|/E_d ∈ [2.5, 25].

**Erwarteter mittlerer Speedup:** Unter der Annahme dass 70% der Queries primär eine Dimension betreffen:

$$\bar{S} = 0.7 \cdot \mathbb{E}[|R|/E_{d^*}] + 0.3 \cdot 1.0 \approx 0.7 \cdot 8 + 0.3 = 5.9\times$$

---

## 4. Integration mit ConceptModels (KAN) <a name="4"></a>

### 4.1 Aktuelles ConceptModel

Aus `concept_model.hpp`:

```
Input: rel_emb ∈ ℝ²⁰, ctx_emb ∈ ℝ²⁰, concept_from ∈ ℝ²⁰, concept_to ∈ ℝ²⁰
Pipeline: Bilinear → MultiHeadBilinear (K=4 Heads, D_PROJ=4) → FlexKAN [6,4,1] → score ∈ (0,1)
```

Die 4 Heads des MultiHeadBilinear berechnen:

$$s_h = (P_h \cdot x_q)^\top \cdot (Q_h \cdot x_k) \quad h = 0, \ldots, 3$$

wobei P_h, Q_h ∈ ℝ^{4×20} die gelernten Projektionsmatrizen und x_q, x_k ∈ ℝ²⁰ die Input-Embeddings sind. Das FlexKAN bekommt [s₀, s₁, s₂, s₃, bilinear_base, dim_fraction] als 6D-Input.

### 4.2 Interpretation der 4 Heads als Dimensionale Experten

**Hypothese 4.1.** Die 4 Heads spezialisieren sich im Training implizit auf verschiedene semantische Aspekte.

**Formale Analyse:** Jeder Head h berechnet ein Bilinear-Produkt im Unterraum ℝ⁴. Die Gesamtkapazität ist 4 × 4 = 16 Dimensionen — exakt die 16D der Relation-Embeddings.

**Proposition 4.1.** Seien die 4 Projektionsmatrizen P₀, ..., P₃ ∈ ℝ^{4×20}. Falls nach dem Training:

$$\text{span}(P_h^\top) \approx \text{span}\{e(t) : t \in T_{d(h)}\}$$

d.h. jeder Head h projiziert auf den Unterraum der Relationstyp-Embeddings einer Dimension d(h), dann sind die Heads dimensionale Experten.

**Test:** Nach dem Training berechne für jeden Head h:

$$\text{alignment}(h, d) = \frac{\|P_h^\top \cdot \bar{e}_d\|_F}{\|P_h^\top\|_F \cdot \|\bar{e}_d\|}$$

Falls es eine Permutation π gibt mit alignment(h, π(h)) > 0.7 für alle h, bestätigt dies die Hypothese.

**Problem:** 4 Heads < 8 Dimensionen. Die Heads können maximal 4 Dimensionen abdecken. Lösung: Entweder K auf 8 erhöhen (Kosten: 640 → 1280 Parameter, +2.2 KB pro Concept) oder 2 Heads für die 2 wichtigsten Dimensionen (HIERARCHICAL, CAUSAL) und 2 Heads als "Residual" für den Rest.

### 4.3 Dimension-Aware Scoring: Mathematische Formulierung

**Variante A: Erweiterter Input.** ConceptModel bekommt zusätzlich den dimensionalen Positionsvektor:

$$x'_q = [x_q \| P(c_{\text{from}})] \in \mathbb{R}^{20+k}, \quad x'_k = [x_k \| P(c_{\text{to}})] \in \mathbb{R}^{20+k}$$

Neue Dimension des MultiHeadBilinear: INPUT_DIM' = 20 + k = 28 (bei k=8).
Neue Parameterzahl: 2 × 4 × 4 × 28 = 896 (vs. aktuell 640, +40%).

**Variante B: Score pro Dimension.** Das FlexKAN gibt statt 1 Score k Scores aus:

$$\text{FlexKAN}: \mathbb{R}^6 \to \mathbb{R}^k, \quad \text{d.h. } [6, 4, k] \text{ statt } [6, 4, 1]$$

LAYER1_EDGES = 4 × k = 32 (statt 4). TOTAL_PARAMS = (24 + 32) × 10 = 560 (statt 280, +100%).

Der Gesamtscore wird dann dimensional gewichtet:

$$\text{score}(c_i, c_j; q) = \sum_{d=1}^{k} w_d(q) \cdot \text{FlexKAN}_d(s_0, s_1, s_2, s_3, s_{\text{bilinear}}, f_{\text{dim}})$$

**Variante C: Mixture of Experts (MoE).** K separate FlexKANs, eines pro Dimension:

$$\text{score}(c_i, c_j; q) = \sum_{d=1}^{k} w_d(q) \cdot \text{FlexKAN}_d(\text{input}_d)$$

Parameter: k × 280 = 2240 (bei k=8, +700%). **Zu teuer** bei 29K Concepts × 2240 = 65MB zusätzlich.

**Empfehlung: Variante B** — moderater Parameteranstieg (+280 Doubles = +2.2KB pro Concept), dimensionale Scores ohne separate Modelle.

### 4.4 Serialisierung

Aktuell: CM_FLAT_SIZE = 1900 Doubles. Mit Variante B:

| Komponente | Aktuell | Neu (Var. B) |
|:-----------|:--------|:-------------|
| Bilinear Core | 940 | 940 |
| MultiHeadBilinear | 640 | 640 |
| FlexKAN | 280 | 560 (+280) |
| PatternWeights | 15 | 15 |
| Reserved | 25 | 25 - 280 → **negativ** |

**Problem:** CM_FLAT_SIZE reicht nicht! Muss auf 2180 erhöht werden. Das betrifft die Persistenz (`persistent_records.hpp`).

**Migration:** CM_FLAT_SIZE von 1900 auf 2200 erhöhen (mit Headroom). Reserved von 25 auf 45.

**Speicher-Impact:** 29.675 × 300 × 8 Bytes = 71 MB zusätzlich. Bei aktuellem 29.675 × 1900 × 8 = 451 MB → Gesamt ~522 MB. Machbar auf einem System mit ≥2GB RAM.

---

## 5. Pattern Discovery → Dimensionale Relations <a name="5"></a>

### 5.1 Co-Aktivierungs-Matrix aus Patterns

Sei P die Menge der 5.9M Patterns. Konstruiere die symmetrische Co-Aktivierungsmatrix:

$$M \in \mathbb{R}^{N \times N}, \quad M_{ij} = \sum_{\substack{p \in P \\ c_i, c_j \in \text{involved}(p)}} \text{conf}(p)$$

**Sparsity:** Die meisten Concept-Paare tauchen in keinem gemeinsamen Pattern auf. Geschätzte Nicht-Null-Einträge: ~1-5% von N² ≈ 4.4-22 Millionen. Da |P| = 5.9M und avg. |involved(p)| ≈ 3-5, erwarten wir ≈ C(4,2) × 5.9M = 35.4M Paar-Beiträge, verteilt über ≈ 5M einzigartige Paare.

### 5.2 SVD-Zerlegung

Die Truncated SVD der Co-Aktivierungsmatrix:

$$M \approx U_k \Sigma_k V_k^\top$$

wobei U_k, V_k ∈ ℝ^{N×k} und Σ_k = diag(σ₁, ..., σ_k) mit σ₁ ≥ σ₂ ≥ ... ≥ σ_k.

**Interpretation der Singulärwerte:**
- σᵢ² ist der Anteil der Varianz in M, der durch Dimension i erklärt wird
- Die zugehörigen Spalten von U_k sind die "dimensionalen Profile" der Concepts
- (U_k)_{:,d} ordnet jedem Concept einen Wert in der latenten Dimension d zu

**Alternatives NMF (Non-Negative Matrix Factorization):**

Da M ≥ 0 (Konfidenz-Summen), ist NMF angemessener als SVD:

$$M \approx W \cdot H, \quad W \in \mathbb{R}_{\geq 0}^{N \times k}, \quad H \in \mathbb{R}_{\geq 0}^{k \times N}$$

NMF hat den Vorteil, dass die Faktoren nicht-negativ und damit **interpretierbar** sind: W_{i,d} = "Zugehörigkeitsgrad von Concept i zu Dimension d".

**Komplexität:**
- Truncated SVD (Randomized): O(N × nnz(M) × k) ≈ O(29K × 5M × 8) ≈ O(10¹²) → **~10 Minuten**, einmalig
- NMF (Multiplicative Updates, 200 Iterationen): O(iter × N × nnz(M) × k) → **~30 Minuten**, einmalig

### 5.3 Automatische Dimensions-Identifikation

Nach SVD/NMF: Korreliere die latenten Dimensionen mit den bekannten Relationstypen.

**Methode:** Für jede latente Dimension d und Relationskategorie κ ∈ K:

$$\rho(d, \kappa) = \text{cor}\left((U_k)_{:,d}, \quad \mathbf{r}_\kappa\right)$$

wobei r_κ ∈ ℝᴺ der Vektor ist mit (r_κ)ᵢ = |{r ∈ R : source(r) = cᵢ ∧ τ(r) ∈ T_κ}| (Anzahl Relations der Kategorie κ von Concept cᵢ).

**Zuordnungsregel:**

$$\text{category}(d) = \arg\max_{\kappa \in K} |\rho(d, \kappa)|$$

Falls max_κ |ρ(d,κ)| < 0.3 → Dimension d ist eine "emergente" Dimension ohne klare Kategorie-Entsprechung (möglicherweise Cross-Domain-Brücke).

### 5.4 Scree Plot und Kaiser-Kriterium

**Scree Plot:** Plotte σ₁, σ₂, ..., σ_min(N,50) auf log-Skala. Der "Ellenbogen" (stärkster Abfall) markiert die signifikante Dimensionszahl.

**Kaiser-Kriterium:** Behalte Dimensionen mit σᵢ > σ̄.

**Broken-Stick-Modell (konservativer):** Dimension i ist signifikant wenn:

$$\sigma_i^2 > \frac{\|M\|_F^2}{N} \sum_{j=i}^{N} \frac{1}{j}$$

**Schätzung für Brain19:** Bei Power-Law-Grad-Verteilung und 8 semantischen Kategorien erwarten wir:
- σ₁ dominiert (HIERARCHICAL, ~40% der Varianz)
- σ₂ (COMPOSITIONAL, ~17%)
- σ₃ (CAUSAL, ~12%)
- σ₄-σ₆ (FUNCTIONAL, SIMILARITY, TEMPORAL, je 3-10%)
- σ₇-σ₈ (OPPOSITION, EPISTEMIC, <3%)

**Erwartete signifikante Dimensionen: k* = 5-7** (Kaiser), k* = 4-5 (Broken-Stick), k* = 8 (kategorienbasiert).

### 5.5 Aus Singulärwerten → Neue Relations

Die SVD liefert nicht nur Dimensionen, sondern auch Relationsvorschläge:

**Rekonstruktions-Residuum:**

$$\Delta = M - U_k \Sigma_k V_k^\top$$

Paare (cᵢ, cⱼ) mit hohem Δᵢⱼ > 0 sind **unerwartet stark co-aktiviert** gegeben die k-dimensionale Approximation. Dies deutet auf eine fehlende explizite Relation hin.

**Paare mit hohem rekonstruiertem M̂ᵢⱼ aber ohne explizite Relation** sind Relations-Kandidaten:

$$\text{candidates} = \{(c_i, c_j) : \hat{M}_{ij} > \theta_M \quad \wedge \quad (c_i, c_j) \notin R\}$$

Die **Dimensionszugehörigkeit** bestimmt den Relationstyp:

$$\text{type}(c_i, c_j) = \text{category}\left(\arg\max_d (U_k)_{i,d} \cdot \sigma_d \cdot (V_k)_{j,d}\right)$$

---

## 6. Informationstheoretische Analyse <a name="6"></a>

### 6.1 Dimensionen als Zufallsvariablen

Für die informationstheoretische Analyse behandeln wir die dimensionale Zugehörigkeit als diskrete Zufallsvariable. Für ein zufällig gezogenes Concept cᵢ ∈ C:

**Definition 6.1.** Die dimensionale Zugehörigkeitsverteilung:

$$p_d(c_i) = \frac{|\{r \in R : \text{source}(r) = c_i, \tau(r) \in T_d\}|}{\delta_{\text{out}}(c_i)}$$

Falls δ_out(cᵢ) = 0, setze p_d(cᵢ) = 1/k (uniform).

### 6.2 Entropy pro Dimension

**Definition 6.2.** Die Entropy der Dimension d:

$$H(D_d) = -\sum_{i=1}^{N} \frac{1}{N} \log_2 p_d(c_i)$$

**Interpretation:** Hohe Entropy → Dimension d ist gleichmäßig über Concepts verteilt (informativ). Niedrige Entropy → wenige Concepts dominieren Dimension d.

**Erwartungswerte für Brain19:**
- H(HIERARCHICAL) ≈ log₂(N × 0.4) ≈ 13.5 bits (hoch, weil viele Concepts IS_A-Relationen haben)
- H(OPPOSITION) ≈ log₂(N × 0.02) ≈ 9.2 bits (niedrig, wenige Concepts haben CONTRADICTS)

### 6.3 Mutual Information zwischen Dimensionen

$$I(D_{d_1}; D_{d_2}) = \sum_{c \in C} \frac{1}{N} \left[ p_{d_1}(c) \log \frac{p_{d_1}(c)}{P(D_{d_1})} + p_{d_2}(c) \log \frac{p_{d_2}(c)}{P(D_{d_2})} \right]$$

Exakter:

$$I(D_{d_1}; D_{d_2}) = H(D_{d_1}) + H(D_{d_2}) - H(D_{d_1}, D_{d_2})$$

wobei H(D_{d₁}, D_{d₂}) die gemeinsame Entropy über die 2D-Verteilung (p_{d₁}(c), p_{d₂}(c)) ist.

**Erwartung:** I(HIERARCHICAL; CAUSAL) sollte niedrig sein (hierarchische und kausale Struktur sind relativ unabhängig). I(HIERARCHICAL; COMPOSITIONAL) sollte höher sein (IS_A und HAS_PART korrelieren — wenn etwas ein Tier IS_A, hat es oft PART_OF Beine).

**Wenn I(D_i; D_j) < ε für alle i ≠ j**, sind die Dimensionen **quasi-unabhängig**, was das Faktorisierungsmodell M ≈ WH rechtfertigt.

### 6.4 Bedingte Entropy: Informationsgewinn durch dimensionale Filterung

**Definition 6.3 (Traversierungs-Entropy).**

Betrachte eine Query q und den FocusCursor bei Concept c. Die Unsicherheit über den nächsten Hop ohne dimensionale Information:

$$H(\text{next} | c) = -\sum_{j \in \mathcal{N}(c)} p(j|c) \log_2 p(j|c)$$

wobei p(j|c) = score(c, cⱼ) / Σ_j' score(c, c_{j'}).

Mit dimensionaler Filterung auf Dimension d:

$$H(\text{next} | c, d) = -\sum_{j \in \mathcal{N}_d(c)} p_d(j|c) \log_2 p_d(j|c)$$

wobei N_d(c) ⊆ N(c) nur die Nachbarn in Dimension d enthält.

**Informationsgewinn:**

$$\text{IG}(c, d) = H(\text{next} | c) - H(\text{next} | c, d)$$

**Erwarteter Informationsgewinn bei Filterung:**

Bei δ̄_out = 3 und δ̄_out^(d) ≈ 1 (bei k=3 Dimensionen im Durchschnitt pro Concept):

$$H(\text{next} | c) \approx \log_2(3) = 1.58 \text{ bits}$$
$$H(\text{next} | c, d) \approx \log_2(1) = 0 \text{ bits}$$

**Informationsgewinn ≈ 1.58 bits** — die dimensionale Filterung reduziert die Unsicherheit fast vollständig, wenn der Graph dünn genug ist und die dimensionale Zuordnung klar.

### 6.5 KL-Divergenz: Uniforme vs. Dimensionale Traversierung

Die uniforme Traversierung nutzt p_unif(j|c) = 1/|N(c)| (alle Nachbarn gleich wahrscheinlich).
Die dimensionale Traversierung nutzt p_dim(j|c, q) = w_{d(j)}(q) · score(c, cⱼ) / Z.

$$D_{\text{KL}}(p_{\text{dim}} \| p_{\text{unif}}) = \sum_j p_{\text{dim}}(j|c,q) \log \frac{p_{\text{dim}}(j|c,q)}{p_{\text{unif}}(j|c)}$$

**Interpretation:** Hohe KL-Divergenz → dimensionale Traversierung weicht stark von uniformer ab → dimensionale Information ist wertvoll.

**Schätzung:** Bei 3 Nachbarn, einem "richtigen" Nachbar (in der relevanten Dimension) mit w = 0.8 und zwei "falschen" mit w = 0.1:

$$p_{\text{dim}} = (0.8, 0.1, 0.1), \quad p_{\text{unif}} = (1/3, 1/3, 1/3)$$

$$D_{\text{KL}} = 0.8 \log \frac{0.8}{0.33} + 0.1 \log \frac{0.1}{0.33} + 0.1 \log \frac{0.1}{0.33} = 0.8 \cdot 0.88 + 0.2 \cdot (-1.22) = 0.46 \text{ nats}$$

### 6.6 Gesamtanalyse: Wieviel Information steckt in den Dimensionen?

**Obere Schranke des Informationsgewinns durch k Dimensionen:**

$$\text{IG}_{\max} = \log_2 k \text{ bits}$$

Bei k = 8: IG_max = 3 bits. Das bedeutet: Dimensionale Information kann die Traversierungsentscheidung um bis zu **3 bits** informieren — äquivalent zu einer 8-fachen Reduktion des Suchraums.

**Untere Schranke (unter Annahme partieller Dimension-Query-Korrelation):**

Wenn die Query nur zu 50% eine einzelne Dimension identifiziert:

$$\text{IG}_{\text{eff}} \approx 0.5 \cdot \log_2 k + 0.5 \cdot 0 = 0.5 \cdot 3 = 1.5 \text{ bits}$$

**Fazit:** Dimensionale Filterung liefert **1.5-3 bits** Informationsgewinn pro Traversierungsschritt. Bei einer typischen Traversierung von D = 5 Hops: **7.5-15 bits kumulativer Gewinn** — der Unterschied zwischen 180 gleichwahrscheinlichen Pfaden und 1-2 dominanten Pfaden.

---

## 7. Auswirkung auf Language Generation <a name="7"></a>

### 7.1 Aktuelles Modell

Aus KAN_MINILLM_LANGUAGE_ENGINE.md:

```
FusedRepresentation f ∈ ℝ⁶⁴ = concat(top-3 a*_i [3×16], gate_scores [5], template [4], padding [10])
KAN-Decoder: f → h₀ ∈ ℝ¹⁶ → autoregressive Token-Generierung
```

Das Token-Generierungsmodell:

$$P(\text{token}_t | f) = \text{softmax}(h_t^\top \cdot E_{\text{vocab}})$$

wobei h_t = KAN_update(h_{t-1}, embed(token_{t-1})) und h₀ = KAN_init(f).

### 7.2 Erweitertes Modell mit dimensionalem Kontext

Neue FusedRepresentation:

$$f' = [f \| P(c_{\text{seed}}) \| w(q)] \in \mathbb{R}^{64 + k + k} = \mathbb{R}^{80} \quad (\text{bei } k=8)$$

wobei P(c_seed) ∈ ℝᵏ der dimensionale Positionsvektor des Seed-Concepts und w(q) ∈ ℝᵏ die dimensionale Attention-Gewichtung ist.

**Token-Generierung mit dimensionalem Kontext:**

$$P(\text{token}_t | f', d_{\text{context}}) = \text{softmax}\left(h_t'^\top \cdot E_{\text{vocab}} + \lambda \cdot b_d\right)$$

wobei b_d ∈ ℝ^{|V|} ein dimensionaler Bias-Vektor ist, der Tokens bevorzugt, die zur aktiven Dimension passen (z.B. kausale Konnektoren wie "verursacht", "führt zu" in der CAUSAL-Dimension).

### 7.3 Erwartete Perplexity-Reduktion

**Aktuell:** Loss stuck bei 2.45 → Perplexity = e^{2.45} ≈ 11.6.

**Analyse des Loss-Plateaus:**

Cross-Entropy-Loss:

$$\mathcal{L} = -\frac{1}{|S|} \sum_{(f, y) \in S} \sum_{t=1}^{|y|} \log P(y_t | f, y_{<t})$$

Das Plateau bei 2.45 bedeutet: Im Mittel hat das Modell e^{2.45} ≈ 11.6 gleichwahrscheinliche Kandidaten pro Token-Position.

**Hypothese:** Das Plateau entsteht durch **dimensionale Ambiguität** — das Modell weiß nicht, ob es einen hierarchischen, kausalen oder kompositionalen Satz generieren soll.

**Formale Zerlegung:**

$$\mathcal{L} = \underbrace{H(\text{token} | \text{meaning})}_{\text{sprachliche Ambiguität}} + \underbrace{H(\text{meaning} | f)}_{\text{dimensionale Ambiguität}}$$

Der erste Term ist irreduzibel (verschiedene Formulierungen des gleichen Sachverhalts). Der zweite Term kann durch dimensionalen Kontext reduziert werden.

**Schätzung von H(meaning|f):**

Wenn f ohne dimensionalen Kontext ~3 verschiedene "Satztypen" gleich wahrscheinlich macht (z.B. definitional, kausal, kompositionell):

$$H(\text{meaning} | f) \approx \log_2(3) = 1.58 \text{ bits} \approx 1.10 \text{ nats}$$

**Neuer erwarteter Loss:**

$$\mathcal{L}' \approx \mathcal{L} - H(\text{meaning} | f) + H(\text{meaning} | f, d) \approx 2.45 - 1.10 + 0.2 = 1.55$$

Neue Perplexity: e^{1.55} ≈ 4.7 (vs. aktuell 11.6) — **Reduktion um ~60%**.

**Kaveat:** Dies ist eine obere Schranke. Die tatsächliche Reduktion hängt davon ab, wie viel der dimensionale Kontext wirklich zur Token-Vorhersage beiträgt. Konservative Schätzung: Perplexity-Reduktion um 30-50%.

### 7.4 Mechanismus der Konvergenz-Verbesserung

Das Loss-Plateau bei 2.45 kann als **Sattelpunkt** der Loss-Landschaft interpretiert werden, an dem:

1. Verschiedene dimensionale Generierungsmodi sich gegenseitig neutralisieren
2. Der Gradient für jeden einzelnen Modus zu klein ist, um dem anderen zu entkommen
3. Das Modell "averaged" über Dimensionen statt sich zu spezialisieren

**Dimensionaler Kontext als Symmetry-Breaking:**

$$\nabla_\theta \mathcal{L}(f') = \nabla_\theta \mathcal{L}(f) + \underbrace{\nabla_\theta \mathcal{L}_{\text{dim}}(w(q), P)}_{\text{dimensionaler Gradient}}$$

Der zusätzliche Gradient-Term bricht die Symmetrie zwischen den Generierungsmodi und ermöglicht dem Optimizer, aus dem Sattelpunkt zu entkommen.

**Konkret:** Wenn der dimensionale Kontext "CAUSAL" signalisiert, werden die Gradienten für kausale Token-Vorhersagen verstärkt und für hierarchische/kompositionelle Vorhersagen gedämpft. Das Modell kann nun in jeder Dimension unabhängig konvergieren.

---

## 8. Vergleich mit existierenden Formalismen <a name="8"></a>

### 8.1 TransR (Lin et al., 2015)

**Modell:** h + r ≈ t im relationstyp-spezifischen Raum:

$$\mathbf{M}_r \mathbf{h} + \mathbf{r} \approx \mathbf{M}_r \mathbf{t}$$

wobei M_r ∈ ℝ^{k×d} eine Projektionsmatrix pro Relationstyp ist.

**Vergleich mit Brain19:**

| Aspekt | TransR | Brain19 Multi-Dim |
|:-------|:-------|:-------------------|
| Embedding-Raum | Einheitlich ℝᵈ | Faktorisiert ℝᵈ = ⊕_d ℝ^{d/k} |
| Relations | Als Translation | Als dimensionale Transition-Matrix |
| Scoring | ‖M_r h + r - M_r t‖ | σ(eᵀ(Wc+b)) + MultiHead + FlexKAN |
| Training | Ranking Loss | MSE + Pattern Discovery |
| Traversierung | Nicht vorgesehen | FocusCursor mit dim. Filterung |

**Mathematischer Unterschied:** TransR projiziert alle Concepts in einen neuen Raum pro Relationstyp. Brain19 belässt Concepts an ihrem Platz und gewichtet Dimensionen abhängig vom Query. Formell:

$$\text{TransR: } \text{score}(h,r,t) = -\|\mathbf{M}_r(\mathbf{h}-\mathbf{t}) + \mathbf{r}\|$$
$$\text{Brain19: } \text{score}(h,t;q) = \sum_d w_d(q) \cdot \text{FlexKAN}_d(P_d h, Q_d t)$$

Brain19's Ansatz ist **kontextabhängig** (w_d hängt von q ab), während TransR statisch pro Relationstyp projiziert.

### 8.2 RotatE (Sun et al., 2019)

**Modell:** Relationen als Rotationen im komplexen Raum:

$$\mathbf{t} = \mathbf{h} \circ \mathbf{r}, \quad r_i = e^{i\theta_i}$$

**Vergleich:** RotatE modelliert Relationen als **Rotationen** — kompositionell (r₁ ∘ r₂ ist wieder eine Rotation). Brain19's dimensionale Transition-Tensoren T_d sind **nicht kompositionell** im gleichen Sinne: T_{d₁} · T_{d₂} ≠ T_{d₃} für ein d₃.

**Vorteil Brain19:** Die InteractionLayer mit Attractor-Dynamik kann nicht-kommutative, nicht-assoziative Reasoning-Muster darstellen (z.B. A CAUSES B CONTRADICTS C → nicht auf eine einzelne Transformation reduzierbar).

**Nachteil Brain19:** Kein geschlossenes algebraisches Modell → schwerer formal zu analysieren, keine Compositionality-Garantien.

### 8.3 Hyperbolic Embeddings (Nickel & Kiela, 2017)

**Modell:** Concepts im hyperbolischen Raum ℍᵈ (Poincaré-Ball):

$$d_{\mathbb{H}}(\mathbf{u}, \mathbf{v}) = \text{arcosh}\left(1 + 2\frac{\|\mathbf{u}-\mathbf{v}\|^2}{(1-\|\mathbf{u}\|^2)(1-\|\mathbf{v}\|^2)}\right)$$

Hierarchische Strukturen werden natürlich abgebildet: abstrakte Concepts nahe am Zentrum, spezifische am Rand.

**Vergleich:** Brain19's HIERARCHICAL-Dimension (Fiedler-Vektor des IS_A-Subgraphen) approximiert dies in ℝ¹ statt ℍᵈ.

**Proposition 8.1.** Der Fiedler-Vektor v₂ des IS_A-Laplacians korreliert mit dem hyperbolischen Radius:

$$\text{cor}((v_2)_i, d_{\mathbb{H}}(c_i, c_{\text{root}})) > 0.7$$

falls der IS_A-Graph annähernd baumförmig ist (was für Brain19's Taxonomie plausibel ist).

*Begründung:* In einem Baum ordnet der Fiedler-Vektor die Knoten monoton nach Tiefe (bis auf Degenerierungen bei symmetrischen Bäumen). Die hyperbolische Distanz zum Root wächst ebenfalls monoton mit der Tiefe. □

**Schlussfolgerung:** Brain19 muss nicht zu hyperbolischen Embeddings wechseln — die Fiedler-basierte HIERARCHICAL-Dimension erfasst die wesentliche Information für Baumstrukturen. Für nicht-baumförmige Hierarchien (Multivererbung) wäre hyperbolisch besser, aber Brain19's IS_A-Graph ist vermutlich überwiegend baumförmig.

### 8.4 Brain19's Unique Positioning

| Eigenschaft | TransR/RotatE | Hyperbolic | Brain19 Multi-Dim |
|:------------|:--------------|:-----------|:-------------------|
| Hierarchie | ⚠️ Implizit | ✅ Nativ | ✅ Via Fiedler |
| Multi-Relation | ✅ Pro Typ | ❌ | ✅ Pro Dimension |
| Traversierung | ❌ | ❌ | ✅ FocusCursor |
| Attractor-Dynamik | ❌ | ❌ | ✅ InteractionLayer |
| Epistemische Transparenz | ❌ | ❌ | ✅ Trust-Scores |
| Pattern-Integration | ❌ | ❌ | ✅ Co-Aktivierung |
| Inkrementell | ❌ (Retrain) | ❌ (Retrain) | ✅ (ConceptModel online) |

**Brain19's mathematischer Kern-Unterschied:** Die Kombination aus (1) dimensionaler Graph-Faktorisierung, (2) Attractor-basiertem Reasoning, und (3) Query-abhängiger Dimensionsgewichtung existiert in keinem der Standard-Formalismen. Am nächsten kommt **Relational Graph Attention Networks (R-GAT)**, die ebenfalls relationstyp-abhängige Attention nutzen — aber ohne Attractor-Dynamik und ohne explizite dimensionale Decomposition.

---

## Appendix A: Zusammenfassung der Formeln

| Formel | Referenz | Zweck |
|:-------|:---------|:------|
| (Ã_d)ᵢⱼ = (A_d)ᵢⱼ + α·(Π_d(M))ᵢⱼ | §1.5 | Pattern-gewichtete Adjazenz |
| P(cᵢ)_d = (v_d^(2))ᵢ | §1.4 | Fiedler-basierte Position |
| w_d(q) = softmax(qᵀ · ē_d / √16) | §2.3 | Dimensionale Attention |
| a^(t+1) = σ(Σ_d w_d T_d a^(t) + β b) | §3.3 | Spreading Activation |
| Kontraktionsrate = 1/4 | §3.4 | Konvergenzbeweis |
| score = Σ_d w_d FlexKAN_d(s₀,...,s₃,s_bil,f) | §4.3 | Dimension-Aware Scoring |
| M ≈ U_k Σ_k V_kᵀ | §5.2 | Pattern-Dimensionen via SVD |
| IG(c,d) = H(next|c) - H(next|c,d) | §6.4 | Informationsgewinn |
| L' ≈ 2.45 - 1.10 + 0.20 = 1.55 | §7.3 | Erwarteter neuer Loss |

## Appendix B: Komplexitätsübersicht

| Operation | Komplexität | Brain19-Werte | Geschätzte Zeit |
|:----------|:------------|:--------------|:----------------|
| k Fiedler-Vektoren (Lanczos) | O(k·|R|·iter) | O(8·87K·100) | ~50ms |
| Pattern-Matrix M konstruieren | O(|P|·avg_size²) | O(5.9M·9) | ~500ms |
| Truncated SVD von M | O(N·nnz·k) | O(29K·5M·8) | ~10min |
| Dimensionale Spreading Activation | O(iter·k·|R|/k) = O(iter·|R|) | O(14·87K) | ~10ms |
| Dimension-Aware predict_refined | O(K·D_PROJ·INPUT_DIM + KAN) | O(640 + 560) | ~1μs |
| Volle Traversierung (D=5 Hops) | O(D·δ̄·C_predict) | O(5·3·1200) | ~20μs |

## Appendix C: Parameteränderungen

| Komponente | Aktuell | Neu | Delta |
|:-----------|:--------|:----|:------|
| FlexKAN pro Concept | 280 doubles | 560 doubles | +280 |
| CM_FLAT_SIZE | 1900 | 2200 | +300 |
| Gesamt-Speicher (29K CMs) | 451 MB | 522 MB | +71 MB |
| Positions-Cache (P) | 0 | 29K × 8 × 8B = 1.8 MB | +1.8 MB |
| Dimensionale Adjacency | 0 | 8 sparse matrices, ~87K entries each | ~5.6 MB |

---

*Erstellt: 2026-02-14 20:10 UTC*  
*Quellen: concept_model.hpp, focus_cursor.cpp, curiosity_engine.cpp, graph-density-analysis.md, KAN_MINILLM_LANGUAGE_ENGINE.md, meta-relations-plan.md, graph-clustering-plan.md*
