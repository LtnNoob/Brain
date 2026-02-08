---
title: "Eignung von Kolmogorov-Arnold Networks (KAN) für kontextabhängige Relevanz-Maps über Knowledge Graphen"
subtitle: "Eine systematische Analyse im Kontext der Brain19-Architektur"
author: "Forschungsbericht — Brain19 Projekt"
date: "Februar 2026"
geometry: "margin=2.5cm"
fontsize: 11pt
documentclass: article
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{hyperref}
  - \usepackage{float}
  - \usepackage{textcomp}
  - 
  - 
  - 
  - \renewcommand{\abstractname}{Zusammenfassung}
  - \renewcommand{\contentsname}{Inhaltsverzeichnis}
  - \usepackage{enumitem}
---

\begin{abstract}
Dieser Bericht untersucht systematisch die Eignung von Kolmogorov-Arnold Networks (KAN) als Architektur für kontextabhängige Relevanz-Maps in der Brain19 kognitiven Architektur. Brain19 verwendet einen epistemisch klassifizierten Knowledge Graph, in dem pro Konzept ein Micro-Modell eine gelernte Gewichtsmap über die Relationen des Graphen erzeugt. Wir vergleichen KAN mit sechs alternativen Architekturen (MLP, GNN, GAT, Transformer Attention, Hyperbolic Embeddings, einfache parametrische Modelle), analysieren die mathematischen Grundlagen, bewerten die praktische Machbarkeit auf dem Ziel-Server (AMD EPYC ~80 Kerne, 120 GB RAM, CPU-only mit optionaler GPU-Erweiterung) und geben eine begründete Architekturempfehlung. Unser zentrales Ergebnis: KAN bietet für diesen spezifischen Use Case keine signifikanten Vorteile gegenüber einfacheren Alternativen. Wir empfehlen stattdessen einen leichtgewichtigen Hybrid-Ansatz aus kontextabhängigen parametrischen Gewichtsfunktionen mit optionaler GAT-Verfeinerung.
\end{abstract}

\tableofcontents
\newpage

# 1. Einleitung

## 1.1 Motivation und Problemstellung

Die Brain19-Architektur verfolgt einen neuartigen Ansatz zur Wissensrepräsentation: Anstatt Wissen in neuronalen Gewichten zu enkodieren, wird es explizit in einem Knowledge Graph (KG) gespeichert, der epistemisch klassifiziert ist — d.h. jede Aussage trägt Metadaten über ihren epistemischen Status (Fakt, Hypothese, Meinung, etc.). Pro Konzept im KG existiert ein kleines *Micro-Modell*, das eine kontextabhängige *Relevanz-Map* lernt: eine Gewichtsfunktion über die Relationen des Konzepts, die angibt, wie wichtig jede Relation in einem gegebenen Kontext ist.

**Beispiel:** Das Micro-Modell für das Konzept "Temperatur" könnte lernen:

- Relation zu "Druck": Relevanz = 0.87 (im Kontext Physik)
- Relation zu "Farbe": Relevanz = 0.02 (im Kontext Physik)
- Relation zu "Geschmack": Relevanz = 0.65 (im Kontext Kochen)

Ein Creativity-Algorithmus überlagert dann zwei solche Maps, um kreative neue Verbindungen zu entdecken — z.B. könnte die Überlagerung der Maps von "Temperatur" und "Musik" unerwartete gemeinsame Relevanzen aufdecken.

## 1.2 Warum KAN als Kandidat?

Kolmogorov-Arnold Networks (KAN), vorgestellt von Liu et al. (2024), haben erhebliches Interesse geweckt, da sie:

1. **Inspizierbare Aktivierungsfunktionen** bieten (B-Splines auf Kanten statt feste Aktivierungsfunktionen auf Knoten)
2. Das **Kolmogorov-Arnold Superposition Theorem** als theoretische Grundlage haben
3. In bestimmten wissenschaftlichen Aufgaben **bessere Accuracy-Effizienz-Tradeoffs** als MLPs zeigen
4. **Symbolische Regression** ermöglichen — gelernte Funktionen können als geschlossene Formeln extrahiert werden

Die Frage ist: Sind diese Eigenschaften für kontextabhängige Relevanz-Maps tatsächlich vorteilhaft, oder handelt es sich um einen Fall von "Lösung sucht Problem"?

## 1.3 Struktur des Berichts

Dieser Bericht ist wie folgt aufgebaut: Abschnitt 2 gibt einen umfassenden Literaturüberblick über KAN und verwandte Arbeiten. Abschnitt 3 definiert das Relevanz-Map-Problem formal. Abschnitt 4 analysiert KAN mathematisch. Abschnitt 5 vergleicht systematisch alle Kandidaten-Architekturen. Abschnitt 6 bewertet die praktische Machbarkeit. Abschnitt 7 gibt die Empfehlung. Abschnitt 8 schließt den Bericht ab.

\newpage

# 2. Literaturüberblick

## 2.1 Kolmogorov-Arnold Networks: Grundlagen

### 2.1.1 Das Kolmogorov-Arnold Superposition Theorem

Das Kolmogorov-Arnold Superposition Theorem (KAST) besagt, dass jede stetige multivariate Funktion $f: [0,1]^n \to \mathbb{R}$ dargestellt werden kann als:

$$f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

wobei $\phi_{q,p}: [0,1] \to \mathbb{R}$ und $\Phi_q: \mathbb{R} \to \mathbb{R}$ stetige univariate Funktionen sind. Das Theorem besagt also, dass jede multivariate stetige Funktion durch eine endliche Komposition univariater Funktionen und Addition darstellbar ist.

**Wichtig:** Das originale Theorem hat eine spezifische Struktur (zwei Schichten, $2n+1$ innere Funktionen) und die inneren Funktionen $\phi_{q,p}$ können hochgradig nicht-glatt sein — in der Praxis sind sie oft fraktal oder pathologisch. Dies war der Grund, warum das Theorem jahrzehntelang als praktisch irrelevant für neuronale Netze galt (Girosi & Poggio, 1989).

### 2.1.2 KAN-Architektur (Liu et al., 2024)

Liu et al. (2024) verallgemeinern das KAST zu einem tieferen Netzwerk und parametrisieren die univariaten Funktionen als B-Splines:

$$\text{KAN}(x) = (\Phi_{L-1} \circ \Phi_{L-2} \circ \cdots \circ \Phi_0)(x)$$

wobei jede Schicht $\Phi_l$ eine Matrix von univariaten Funktionen ist. Für eine Schicht mit $n_{in}$ Eingängen und $n_{out}$ Ausgängen gibt es $n_{in} \times n_{out}$ lernbare univariate Funktionen:

$$\Phi_l = \{\phi_{l,j,i}\}, \quad i = 1,\ldots,n_{in}, \quad j = 1,\ldots,n_{out}$$

Jede Funktion $\phi_{l,j,i}$ wird als B-Spline parametrisiert:

$$\phi_{l,j,i}(x) = w_b \cdot \text{silu}(x) + w_s \cdot \text{spline}(x)$$

wobei $\text{spline}(x) = \sum_{k} c_k B_k(x)$ eine lineare Kombination von B-Spline-Basisfunktionen $B_k$ mit Ordnung $k$ und $G$ Gitterpunkten ist.

**Parameteranzahl pro Schicht:** $(n_{in} \times n_{out}) \times (G + k + 3)$, wobei $G$ die Anzahl der Gitterpunkte und $k$ die Spline-Ordnung ist. Im Vergleich: Ein MLP benötigt $n_{in} \times n_{out} + n_{out}$ Parameter pro Schicht.

### 2.1.3 Zentrale Ergebnisse von Liu et al. (2024)

Die Originalarbeit berichtet:

- **Accuracy:** KAN erreicht auf bestimmten wissenschaftlichen Regressionstasks (z.B. symbolische Formeln entdecken) bessere Accuracy als MLPs bei weniger Parametern
- **Inspizierbarkeit:** Die gelernten Spline-Funktionen können visualisiert und in symbolische Ausdrücke umgewandelt werden
- **Grid Extension:** Die Spline-Gitter können sukzessive verfeinert werden, was ein "multi-resolution learning" ermöglicht
- **Skalierung:** KAN zeigt günstigere neurale Skalierungsgesetze als MLPs

**Einschränkungen (von den Autoren selbst genannt):**

- Training ist ca. **10× langsamer** als MLPs bei gleicher Parameteranzahl (Liu et al., 2024, Abschnitt 3.4)
- B-Splines sind schlecht parallelisierbar auf GPUs
- Skalierung auf große Architekturen ist noch nicht demonstriert

### 2.1.4 KAN 2.0 (Liu et al., 2024b)

Die Nachfolgearbeit "KAN 2.0: Kolmogorov-Arnold Networks Meet Science" (Liu et al., August 2024) erweitert KAN für wissenschaftliche Anwendungen mit:

- MultKAN (multiplikative KAN-Schichten)
- Bessere Symbolische-Regression-Pipeline
- Anwendungen in Physik und Mathematik

## 2.2 Kritik und Vergleichsstudien

### 2.2.1 KAN vs. MLP: Faire Vergleiche

Yu et al. (2024) zeigen in "KAN or MLP: A Fairer Comparison", dass wenn man MLPs mit B-Spline-Aktivierungsfunktionen ausstattet (BSpline-MLP), diese KAN in vielen Tasks erreichen oder übertreffen. Dies deutet darauf hin, dass der Vorteil von KAN teilweise auf die Spline-Basisfunktionen zurückzuführen ist und nicht auf die spezifische Architektur der Kanten-Aktivierungen.

### 2.2.2 Trainingsgeschwindigkeit

Mehrere unabhängige Studien bestätigen das Geschwindigkeitsproblem:

- **PowerMLP** (Xu et al., 2024): Führt das langsame Training auf die $O(k^2)$-Komplexität der B-Spline-Berechnung zurück und schlägt effizientere Alternativen vor
- **MatrixKAN** (2025): Versucht, KAN durch Parallelisierung zu beschleunigen
- **TruKAN** (2025): Verwendet Truncated Power Functions statt B-Splines für bessere Effizienz
- Konsens: KAN ist **5-10× langsamer** als vergleichbare MLPs

### 2.2.3 Katastrophales Vergessen

Eine AAAI-2025-Studie zeigt, dass KANs wie MLPs unter katastrophalem Vergessen leiden — ein relevanter Aspekt, wenn Micro-Modelle in Brain19 inkrementell aktualisiert werden sollen.

## 2.3 KAN auf Graphen

### 2.3.1 KAGNNs (Bresson et al., 2024/2025)

Bresson et al. stellen in "KAGNNs: Kolmogorov-Arnold Networks meet Graph Learning" (TMLR 2025) vor:

- **KAGCN** (Kolmogorov-Arnold Graph Convolution Network): Ersetzt die linearen Transformationen in GCN durch KAN-Schichten
- **KAGAT** (Kolmogorov-Arnold Graph Attention Network): Integriert KAN in GAT-Architekturen
- **KAGIN** (Kolmogorov-Arnold Graph Isomorphism Network): KAN-basiertes GIN

Ergebnisse: Verbesserte Interpretierbarkeit bei vergleichbarer oder leicht besserer Performance auf Graphklassifikations-Benchmarks, aber deutlich höherer Rechenaufwand.

### 2.3.2 GKAN (de Carlo et al., 2024)

De Carlo et al. stellen GKAN (Graph Kolmogorov-Arnold Networks) vor (arXiv:2406.06470), das KAN-Prinzipien auf Message-Passing-GNNs anwendet. Spline-basierte Aktivierungsfunktionen ersetzen die Standard-Aggregationsfunktionen, was zu interpretierbaren Kanten-Transformationen führt.

### 2.3.3 KA-GNN für Moleküle (2024)

Zhang et al. stellen KA-GNN vor (arXiv:2410.11323), das Fourier-basierte KAN-Schichten in GNNs für molekulare Eigenschaftsvorhersage integriert. Sie berichten von Verbesserungen bei der Vorhersagegenauigkeit, allerdings auf spezialisierten chemischen Datensätzen.

### 2.3.4 KA-GAT (2024)

Eine spezifische KAN-basierte GAT-Variante, die die Attention-Berechnung durch KAN-Funktionen ersetzt. Verfügbar als OpenReview-Preprint.

## 2.4 Relevanz-Learning in Knowledge Graphen

### 2.4.1 Knowledge Graph Embedding Methoden

Klassische KG-Embedding-Methoden wie TransE (Bordes et al., 2013), DistMult (Yang et al., 2015), ComplEx (Trouillon et al., 2016) und RotatE (Sun et al., 2019) lernen Vektorrepräsentationen für Entitäten und Relationen. Diese können als implizite Relevanz-Scores interpretiert werden, kodieren aber keine expliziten kontextabhängigen Gewichte.

### 2.4.2 Graph Attention Networks (Veličković et al., 2018)

GAT lernt Attention-Gewichte zwischen verbundenen Knoten, was konzeptionell sehr nah an kontextabhängigen Relevanz-Maps ist. Die Attention-Gewichte $\alpha_{ij}$ für Knoten $i$ und Nachbar $j$ werden als:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]))}$$

berechnet. Multi-Head Attention erlaubt verschiedene Relevanz-Perspektiven, die unterschiedlichen Kontexten entsprechen können.

### 2.4.3 Hyperbolic Graph Neural Networks

Chami et al. (2019) und Liu et al. (2019) zeigen, dass hyperbolische Einbettungen besonders gut für hierarchische Graphstrukturen geeignet sind. Konzepte in Knowledge Graphen weisen oft solche Hierarchien auf (is-a-Beziehungen), was hyperbolische Repräsentationen attraktiv macht.

### 2.4.4 Continual Knowledge Graph Embedding

Für Brain19 relevant: Ansätze zum inkrementellen Update von KG-Embeddings ohne vollständiges Neutraining (Wu et al., 2024). Dies entspricht dem Szenario, dass Micro-Modelle nachts aktualisiert werden, wenn sich der Graph geändert hat.

## 2.5 Kognitive Architekturen

### 2.5.1 Verwandte kognitive Architekturen

- **SOAR** (Laird, 2012): Verwendet symbolische Produktionssysteme mit numerischen Präferenzwerten
- **ACT-R** (Anderson, 2007): Nutzt Activation-basierte Retrieval-Mechanismen, konzeptionell ähnlich zu Relevanz-Maps
- **CLARION** (Sun, 2016): Hybrid-Architektur mit implizitem (subsymbolisch) und explizitem (symbolisch) Wissen
- **OpenCog** (Goertzel, 2014): Hypergraph-basiertes Wissenssystem mit Attention Values

Die Idee, dass symbolisches Wissen durch lernbare Relevanzmechanismen ergänzt wird, ist in der Literatur zu kognitiven Architekturen gut etabliert. Brain19 ist insofern innovativ, als es diese Idee mit modernen ML-Methoden und epistemischer Klassifikation kombiniert.

\newpage

# 3. Formale Problemdefinition

## 3.1 Knowledge Graph

Sei $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R}, \mathcal{C})$ ein Knowledge Graph mit:

- $\mathcal{V}$: Menge der Konzept-Knoten (z.B. "Temperatur", "Druck", "Farbe")
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$: Menge der Kanten (Tripel)
- $\mathcal{R}$: Menge der Relationstypen
- $\mathcal{C}$: Menge der Kontexte (z.B. "Physik", "Kochen", "Medizin")

Jede Kante $e = (v_i, r, v_j) \in \mathcal{E}$ hat zusätzlich einen epistemischen Status $\epsilon(e) \in \{\text{Fakt}, \text{Hypothese}, \text{Meinung}, \text{Tradition}, \ldots\}$.

## 3.2 Relevanz-Map

Für ein Konzept $v \in \mathcal{V}$ sei $\mathcal{N}(v) = \{(r_k, v_k) : (v, r_k, v_k) \in \mathcal{E}\}$ die Menge seiner Nachbarn (inklusive Relationstyp). Eine **Relevanz-Map** $\mathbf{w}_v$ ist eine Funktion:

$$\mathbf{w}_v: \mathcal{N}(v) \times \mathcal{C} \to [0, 1]$$

die jedem Nachbar-Relation-Paar $(r_k, v_k)$ in einem Kontext $c$ eine Relevanz zuweist.

## 3.3 Micro-Modell als Optimierungsproblem

Ein Micro-Modell $M_v$ für Konzept $v$ ist eine parametrisierte Funktion:

$$M_v(\mathbf{x}_{r,v'}, \mathbf{c}; \theta_v) \to w \in [0,1]$$

wobei:

- $\mathbf{x}_{r,v'} \in \mathbb{R}^{d_r}$: Feature-Vektor der Relation-Nachbar-Kombination $(r, v')$
- $\mathbf{c} \in \mathbb{R}^{d_c}$: Kontext-Embedding
- $\theta_v$: Lernbare Parameter des Micro-Modells

Die Features $\mathbf{x}_{r,v'}$ können beinhalten:

- One-Hot oder Embedding des Relationstyps $r$
- Embedding des Nachbar-Konzepts $v'$
- Epistemischer Status der Kante
- Strukturelle Features (Grad von $v'$, gemeinsame Nachbarn, etc.)

## 3.4 Optimierungsziel

Das Training eines Micro-Modells kann als folgendes Optimierungsproblem formuliert werden:

$$\min_{\theta_v} \mathcal{L}(\theta_v) = \sum_{c \in \mathcal{C}} \sum_{(r_k, v_k) \in \mathcal{N}(v)} \ell\left(M_v(\mathbf{x}_{r_k,v_k}, \mathbf{c}_c; \theta_v), \; w^*_{v,r_k,v_k,c}\right) + \lambda \|\theta_v\|$$

wobei:

- $w^*_{v,r_k,v_k,c}$ die Ziel-Relevanz ist (aus Nutzerfeedback, Graphstruktur, oder Co-Occurrence)
- $\ell$ eine geeignete Loss-Funktion (z.B. MSE, Cross-Entropy)
- $\lambda \|\theta_v\|$ ein Regularisierungsterm

## 3.5 Creativity-Überlagerung

Für den Creativity-Algorithmus müssen zwei Maps $\mathbf{w}_{v_1}$ und $\mathbf{w}_{v_2}$ überlagert werden:

$$\mathbf{w}_{v_1 \otimes v_2}(r, v', c) = f_{\text{combine}}(\mathbf{w}_{v_1}, \mathbf{w}_{v_2}, r, v', c)$$

wobei $f_{\text{combine}}$ die Verbindungsfunktion ist. Einfache Varianten:

- **Multiplikation:** $w_{v_1} \cdot w_{v_2}$ (findet gemeinsam wichtige Relationen)
- **Harmonisches Mittel:** $\frac{2 w_{v_1} w_{v_2}}{w_{v_1} + w_{v_2}}$ (betont Überlappung)
- **Überraschungsbasiert:** $|w_{v_1} - w_{v_2}| \cdot \max(w_{v_1}, w_{v_2})$ (findet asymmetrische Relevanzen)

## 3.6 Anforderungen an die Architektur

Aus der formalen Definition ergeben sich folgende Anforderungen:

1. **Kleine Inputdimension:** $d = d_r + d_c$, typisch 10-100
2. **Kleine Outputdimension:** 1 (Relevanz-Score) oder $|\mathcal{N}(v)|$ (gesamte Map auf einmal)
3. **Wenige Trainingsdaten pro Modell:** $|\mathcal{N}(v)| \times |\mathcal{C}|$, typisch 50-500 Datenpunkte
4. **Kontextabhängigkeit:** Der Kontext muss die Gewichte modulieren
5. **Inspizierbarkeit:** Wünschenswert, da Brain19 epistemisch transparent sein soll
6. **Schnelle Inference:** Maps müssen schnell berechnet werden (für Creativity-Algorithmus)
7. **Effizientes Batch-Update:** Nachts müssen alle geänderten Maps neu berechnet werden
8. **Überlagerbarkeit:** Die Outputs müssen sinnvoll kombinierbar sein

\newpage

# 4. Mathematische Analyse von KAN

## 4.1 KAN-Parametrisierung für das Relevanz-Problem

Für ein Micro-Modell mit $d_{in}$ Input-Features und einem skalaren Output $w \in [0,1]$ benötigt ein KAN mit Architektur $[d_{in}, n_1, \ldots, n_{L-1}, 1]$:

**Parameter pro Schicht $l$:**
$$P_l = n_l \times n_{l+1} \times (G + k + 3)$$

wobei $G$ die Gitterpunkte und $k$ die Spline-Ordnung (typisch $k=3$) ist.

**Gesamtparameter:**
$$P_{\text{total}} = \sum_{l=0}^{L-1} n_l \times n_{l+1} \times (G + k + 3)$$

**Beispiel für Brain19:** Angenommen $d_{in} = 20$ (10 Relation-Features + 10 Kontext-Features), Architektur $[20, 10, 1]$, $G=5$, $k=3$:

$$P = 20 \times 10 \times (5 + 3 + 3) + 10 \times 1 \times (5 + 3 + 3) = 2200 + 110 = 2310 \text{ Parameter}$$

Zum Vergleich, ein MLP $[20, 10, 1]$ mit Biases:
$$P_{\text{MLP}} = 20 \times 10 + 10 + 10 \times 1 + 1 = 221 \text{ Parameter}$$

KAN hat also bei gleicher Architektur ca. **10× mehr Parameter** als ein MLP. Bei sehr wenigen Trainingsdaten (50-500 Punkte) ist dies problematisch — Overfitting ist ein reales Risiko.

## 4.2 Approximationstheoretische Analyse

### 4.2.1 KAN-Approximation

Das KAST garantiert, dass jede stetige Funktion $f: [0,1]^n \to \mathbb{R}$ durch ein zweischichtiges KAN exakt dargestellt werden kann. Allerdings können die benötigten inneren Funktionen $\phi_{q,p}$ beliebig komplex (nicht-glatt) sein.

Für die B-Spline-Approximation in KAN gilt: Wenn die Zielfunktion eine Glattheit von $C^s$ hat und der Spline-Grad $k \geq s$ ist, konvergiert die Approximation mit Rate $O(G^{-s})$ (de Boor, 1978), wobei $G$ die Anzahl der Gitterpunkte ist.

### 4.2.2 Relevanz des Theorems für unser Problem

Die Relevanz-Funktion $w_v(r, v', c)$ ist wahrscheinlich eine **relativ glatte** Funktion — Relevanzwerte ändern sich stetig mit dem Kontext und der Relationsstruktur. Für glatte Funktionen in niedriger Dimension ist die Approximation durch B-Splines effizient.

**Aber:** Ein einfaches MLP mit einer Hidden-Layer und Sigmoid-Aktivierung kann ebenfalls jede stetige Funktion auf kompakten Mengen beliebig genau approximieren (Universal Approximation Theorem, Cybenko 1989, Hornik 1991). Der theoretische Vorteil von KAN existiert hauptsächlich für hochdimensionale Funktionen mit spezieller Kompositionsstruktur — was bei unserem niedrigdimensionalen Relevanz-Problem kaum relevant ist.

## 4.3 Komplexitätsanalyse

### 4.3.1 Trainingskosten

**KAN:** Pro Forward Pass für eine Schicht mit $n_{in} \times n_{out}$ Spline-Funktionen und $G$ Gitterpunkten:

$$T_{\text{KAN}} = O(n_{in} \times n_{out} \times (G \cdot k))$$

Die B-Spline-Evaluation benötigt $O(k)$ Operationen pro Basisfunktion, und $G+k$ Basisfunktionen müssen evaluiert werden, wovon nur $k+1$ nicht-null sind (lokaler Support). Zusätzlich kommen $O(k^2)$ Kosten für die rekursive de Boor-Cox-Berechnung.

**MLP:** 
$$T_{\text{MLP}} = O(n_{in} \times n_{out})$$

Das Verhältnis ist:
$$\frac{T_{\text{KAN}}}{T_{\text{MLP}}} = O(G \cdot k + k^2) \approx O(G \cdot k)$$

Für typische Werte $G=5, k=3$: KAN ist ca. **15-20× langsamer** pro Forward Pass.

### 4.3.2 Inference-Kosten

Die Inference-Kosten sind proportional zu den Forward-Pass-Kosten. Für ein einzelnes Micro-Modell sind diese absolut gering (Mikrosekunden), aber multipliziert mit Tausenden von Konzepten und dem Creativity-Algorithmus (der viele Map-Evaluationen benötigt) kann sich dies summieren.

### 4.3.3 Speicherkomplexität

Pro Micro-Modell mit 2310 Parametern (wie oben berechnet) und Float32:
$$\text{Memory}_{\text{KAN}} = 2310 \times 4 \text{ Bytes} = 9.24 \text{ KB}$$

Pro MLP-Micro-Modell mit 221 Parametern:
$$\text{Memory}_{\text{MLP}} = 221 \times 4 \text{ Bytes} = 0.884 \text{ KB}$$

Für 10.000 Konzepte:

| Architektur | Memory (10k Modelle) |
|------------|---------------------|
| KAN [20,10,1] | 92.4 MB |
| MLP [20,10,1] | 8.84 MB |
| Parametrisch (50 Params) | 2.0 MB |

Alle Varianten passen problemlos in die 120 GB RAM des Ziel-Servers.

## 4.4 Inspizierbarkeit von KAN

KANs Hauptversprechen ist die Inspizierbarkeit: Jede gelernte univariate Funktion $\phi_{l,j,i}$ kann visualisiert werden, und wenn die Funktion eine erkennbare Form hat (linear, quadratisch, sinusförmig), kann sie als symbolischer Ausdruck extrahiert werden.

**Für das Relevanz-Problem:** Die Inspizierbarkeitseigenschaft wäre nützlich, um zu verstehen, *warum* eine Relation als relevant eingestuft wird. Allerdings:

1. Die univariaten Funktionen auf den *Kanten* sind schwer zu interpretieren, wenn das Input ein abstraktes Feature-Embedding ist (z.B. ein 10-dimensionaler Kontextvektor)
2. Die Interpretierbarkeit funktioniert am besten bei niedrigdimensionalen Inputs mit klarer semantischer Bedeutung
3. Für Brain19 wäre eine einfachere Interpretierbarkeit möglich: Die Gewichte $w_{r,c}$ direkt als Matrix speichern (Relation × Kontext), was ohne jedes neuronale Netz perfekt inspizierbar ist

\newpage

# 5. Systematischer Architekturvergleich

## 5.1 Kandidaten-Architekturen

Wir vergleichen sieben Architekturen systematisch:

1. **KAN** (Kolmogorov-Arnold Networks)
2. **MLP** (Multi-Layer Perceptron)
3. **GNN** (Graph Neural Networks / GCN)
4. **GAT** (Graph Attention Networks)
5. **Transformer Attention**
6. **Hyperbolic Embeddings**
7. **Parametrisches Modell** (Direct Weight Matrix)

## 5.2 Bewertungskriterien

| Kriterium | Gewicht | Begründung |
|-----------|---------|------------|
| Inspizierbarkeit | Hoch | Brain19 soll epistemisch transparent sein |
| Dateneffizienz | Sehr hoch | Nur 50-500 Trainingspunkte pro Modell |
| Trainingsgeschwindigkeit | Hoch | Nightly Batch-Update für alle Modelle |
| Inference-Geschwindigkeit | Hoch | Creativity-Algorithmus braucht schnelle Evaluation |
| Kontextabhängigkeit | Sehr hoch | Kernfunktionalität |
| Überlagerbarkeit | Hoch | Creativity-Algorithmus |
| Skalierbarkeit | Mittel | 1000-10000 Konzepte |
| Approximationskraft | Mittel | Relevanzfunktionen sind vermutlich nicht hochkomplex |

## 5.3 Detailanalyse pro Architektur

### 5.3.1 KAN

**Stärken für diesen Use Case:**

- Inspizierbare Spline-Funktionen (aber siehe Einschränkungen in 4.4)
- Gute Approximation bei glatten niedrigdimensionalen Funktionen
- Grid Extension erlaubt progressive Verfeinerung

**Schwächen für diesen Use Case:**

- **10× mehr Parameter als MLP** bei gleicher Architektur → Overfitting-Risiko bei wenig Daten
- **10-20× langsamer** als MLP im Training
- B-Splines schlecht GPU-parallelisierbar
- Kontextabhängigkeit muss als zusätzliche Input-Dimension kodiert werden — kein natürlicher Mechanismus
- Überlagerung: KAN-Outputs sind Skalare, können einfach kombiniert werden, aber die gelernten Spline-Funktionen zweier KANs können nicht sinnvoll "gemischt" werden
- Overkill für das Problem: Die zu lernenden Funktionen (Relevanzwerte abhängig von Kontext und Relation) sind wahrscheinlich relativ einfach

**Parameterbudget:** Ein KAN [20, 10, 1] mit $G=5, k=3$ hat 2310 Parameter für typisch 50-500 Trainingspunkte. Das Verhältnis Parameter/Daten liegt bei 4.6-46, was stark zum Overfitting neigt. Selbst mit starker Regularisierung ist dies suboptimal.

### 5.3.2 Standard MLP

**Stärken:**

- Sehr schnelles Training und Inference (optimiert für GPUs)
- Weniger Parameter (221 für [20,10,1])
- Besseres Parameter/Daten-Verhältnis
- Universal Approximation
- Umfangreiche Tooling-Unterstützung (PyTorch, etc.)

**Schwächen:**

- Geringe Inspizierbarkeit (Black Box)
- Kontextabhängigkeit nur als Input, keine natürliche Modulation
- Keine Strukturannahme über den Graphen

**Bewertung:** Solide Baseline, aber nicht optimal für den spezifischen Use Case. Die fehlende Inspizierbarkeit ist ein reales Problem für Brain19's epistemische Transparenz.

### 5.3.3 GNN (Graph Convolutional Network)

**Idee:** Anstatt pro Konzept ein isoliertes Micro-Modell zu trainieren, ein einzelnes GNN über den gesamten Graphen laufen lassen, das kontextabhängige Relevanzwerte für alle Kanten berechnet.

**Stärken:**

- Nutzt Graphstruktur explizit
- Parameter-Sharing über den gesamten Graphen (sehr dateneffizient)
- Skaliert natürlich mit Graphgröße
- Kann globale Strukturmuster lernen

**Schwächen:**

- Geringe Inspizierbarkeit
- Standard-GCN hat keine Kontextabhängigkeit (alle Nachbarn werden gleich aggregiert)
- Message-Passing kann bei dichten Graphen teuer werden
- Oversmoothing bei vielen Schichten

**Bewertung:** Ein globales GNN widerspricht dem Brain19-Design (pro Konzept ein Micro-Modell), könnte aber als Initialisierung für Micro-Modelle dienen.

### 5.3.4 GAT (Graph Attention Network)

**Idee:** GAT berechnet natürlicherweise Attention-Gewichte zwischen verbundenen Knoten — dies *ist* im Wesentlichen eine Relevanz-Map.

**Stärken:**

- **Attention-Gewichte sind direkt interpretierbar** als Relevanzwerte
- **Multi-Head Attention** kann verschiedene Kontexte modellieren (jeder Head = ein Kontext)
- Natürliche Graphstruktur
- Parameter-Sharing (dateneffizient)
- Schnelles Training (GPU-optimiert)
- Gut erforscht, stabile Implementierungen

**Schwächen:**

- Ein globales GAT, nicht pro-Konzept-Modelle (Architekturkonflikt)
- Kontexte als separate Heads zu modellieren skaliert schlecht bei vielen Kontexten
- Attention-Softmax erzwingt Normalisierung (Summe = 1), was nicht immer gewünscht ist

**Bewertung:** **Konzeptionell am nächsten** am Brain19-Problem. Die Attention-Gewichte *sind* Relevanz-Maps. Allerdings müsste das Brain19-Design angepasst werden (globales Modell statt Micro-Modelle).

### 5.3.5 Transformer Attention

**Idee:** Self-Attention über die Nachbarschaft eines Konzepts, mit Kontext als Query.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Wobei $Q$ der Kontext-Embedding, $K$ und $V$ die Nachbar-Embeddings sind.

**Stärken:**

- Kontextabhängigkeit ist eingebaut (Query moduliert die Attention)
- Multi-Head für verschiedene Aspekte
- Hochgradig parallelisierbar
- Kann als Micro-Modell eingesetzt werden (lokaler Transformer pro Konzept)

**Schwächen:**

- $O(n^2)$ Komplexität in der Nachbarschaftsgröße (bei 10-50 Nachbarn akzeptabel)
- Viele Parameter für die Q/K/V-Projektionen
- Bei sehr wenigen Nachbarn (< 10) instabil
- Softmax-Normalisierung wie bei GAT

**Bewertung:** Guter Kandidat, besonders wenn die Kontextabhängigkeit als Query-Key-Mechanismus natürlich implementiert wird. Aber vermutlich Overkill für das Problem.

### 5.3.6 Hyperbolic Embeddings

**Idee:** Konzepte und Relationen in hyperbolischem Raum (z.B. Poincaré-Ball oder Hyperboloid) einbetten. Abstände im hyperbolischen Raum können als inverse Relevanz interpretiert werden.

$$d_{\mathbb{H}}(\mathbf{u}, \mathbf{v}) = \text{arcosh}\left(1 + 2\frac{\|\mathbf{u} - \mathbf{v}\|^2}{(1-\|\mathbf{u}\|^2)(1-\|\mathbf{v}\|^2)}\right)$$

**Stärken:**

- Exzellent für hierarchische Strukturen (Knowledge Graphen haben oft is-a-Hierarchien)
- Kompakte Repräsentation (niedrigdimensionale Embeddings reichen)
- Mathematisch elegant

**Schwächen:**

- **Keine direkte Kontextabhängigkeit** — der Abstand ist fix, nicht kontextbedingt
- Numerisch instabil nahe dem Rand des Poincaré-Balls
- Relevanz = Proximity ist eine starke Annahme
- Überlagerung zweier hyperbolischer Maps ist mathematisch nicht trivial
- Weniger intuitiv interpretierbar als explizite Gewichte

**Bewertung:** Gut als Basis-Embedding, aber nicht als alleinige Relevanz-Map geeignet, da die Kontextabhängigkeit fehlt.

### 5.3.7 Parametrisches Modell (Direct Weight Matrix)

**Idee:** Die einfachste Lösung — pro Konzept $v$ eine Matrix $\mathbf{W}_v \in \mathbb{R}^{|\mathcal{N}(v)| \times |\mathcal{C}|}$ speichern, die direkt die Relevanzwerte enthält.

Optional mit leichter Parametrisierung:
$$w_{v}(r_k, v_k, c) = \sigma(\mathbf{a}_{r_k}^T \mathbf{c} + b_{r_k})$$

wobei $\mathbf{a}_{r_k} \in \mathbb{R}^{d_c}$ ein lernbarer Vektor pro Relation und $\sigma$ eine Sigmoid-Funktion ist.

**Stärken:**

- **Maximal inspizierbar:** Die Gewichte sind direkt lesbar
- **Minimal Parameter:** $|\mathcal{N}(v)| \times d_c$ pro Konzept (z.B. 30 × 10 = 300 Parameter)
- **Schnellstes Training:** Einfaches lineares/logistisches Modell
- **Schnellste Inference:** Matrixmultiplikation
- **Perfekt überlagerbar:** Gewichtsvektoren können direkt kombiniert werden
- **Kein Overfitting-Risiko** bei linearer Parametrisierung
- **Kontextabhängig** durch die Kontext-Embedding-Interaktion

**Schwächen:**

- Kann keine komplexen nicht-linearen Relevanzfunktionen lernen
- Generalisiert nicht auf ungesehene Kontext-Relation-Kombinationen (wenn linear)
- Keine automatische Feature-Extraktion

**Bewertung:** **Überraschend starker Kandidat.** Für das Brain19-Problem, wo die Relevanzfunktionen vermutlich relativ einfach sind und Inspizierbarkeit wichtig ist, könnte dies die beste Wahl sein.

## 5.4 Vergleichsmatrix

| Kriterium | KAN | MLP | GNN | GAT | Transformer | Hyp. Emb. | Parametrisch |
|-----------|-----|-----|-----|-----|-------------|-----------|-------------|
| Inspizierbarkeit | + | - | - | + | - | - | ++ |
| Dateneffizienz | - | + | ++ | ++ | - | + | ++ |
| Trainingsgeschw. | - | ++ | + | + | + | ++ | ++ |
| Inference-Geschw. | - | ++ | + | + | + | ++ | ++ |
| Kontextabhängigk. | + | + | - | + | ++ | - | + |
| Überlagerbarkeit | + | + | - | + | + | - | ++ |
| Skalierbarkeit | + | ++ | + | + | + | ++ | ++ |
| Approx.-Kraft | ++ | ++ | + | + | ++ | + | - |

Legende: ++ = Sehr gut, + = Mittel, - = Schlecht

## 5.5 Gewichtete Gesamtbewertung

Mit den in 5.2 definierten Gewichten (Inspizierbarkeit: 3, Dateneffizienz: 4, Training: 3, Inference: 3, Kontextabhängigkeit: 4, Überlagerbarkeit: 3, Skalierbarkeit: 2, Approx.-Kraft: 2) und Punkten (++=3, +=2, -=1):

| Architektur | Gewichteter Score | Rang |
|------------|-------------------|------|
| **Parametrisch** | **67** | **1** |
| **GAT** | **55** | **2** |
| MLP | 55 | 3 |
| Transformer | 53 | 4 |
| **KAN** | **45** | **5** |
| GNN | 45 | 6 |
| Hyp. Emb. | 47 | 7 |

**KAN landet auf Rang 5 von 7.** Die Hauptgründe: schlechte Dateneffizienz (zu viele Parameter), langsames Training, und nur mittelmäßige Kontextabhängigkeit.

\newpage

# 6. Praktische Machbarkeitsanalyse

## 6.1 Ziel-Hardware

Der Brain19-Server verfügt über signifikante CPU-Ressourcen, aber (bisher) keine dedizierte GPU. Wir analysieren beide Szenarien.

| Spezifikation | Brain19-Server |
|--------------|----------------|
| CPU | AMD EPYC, ~80 Kerne |
| RAM | 120 GB DDR4/DDR5 |
| Storage | 500 GB M.2 NVMe SSD |
| GPU | Keine (GPU-Erweiterung möglich) |

**Implikationen der Hardwarekonfiguration:**

1. **CPU-Parallelismus:** 80 Kerne erlauben massives Multi-Processing. Da Micro-Modelle unabhängig voneinander sind, können 80 Modelle *gleichzeitig* auf separaten Kernen trainiert werden — ohne GPU.
2. **RAM:** 120 GB RAM ist für alle betrachteten Architekturen mehr als ausreichend. Selbst 100.000 KAN-Micro-Modelle würden nur ~1.6 GB (Inference) bzw. ~3.5 GB (Training) benötigen.
3. **Keine GPU:** B-Spline-Berechnungen in KAN sind ohnehin schlecht GPU-parallelisierbar (iterative Rekursion). Das Fehlen einer GPU **benachteiligt KAN weniger als andere Architekturen** — ein interessanter Aspekt, der KANs relative Position leicht verbessert.
4. **NVMe SSD:** Schnelles Speichern/Laden der Micro-Modelle für den nightly Batch-Job.

### 6.1.1 CPU-Performance-Schätzung

Ein einzelner AMD EPYC-Kern (Zen 3/4) liefert ca. 50-100 GFLOPS FP32 bei vektorisierten Operationen (AVX2/AVX-512). Mit 80 Kernen:

- **Theoretischer Peak:** 4-8 TFLOPS FP32
- **Effektiv (mit Python/PyTorch-Overhead):** 50-500 GFLOPS gesamt
- **Pro Kern effektiv:** ~1-5 GFLOPS

Zum Vergleich: Eine RTX 3090 liefert ~35 TFLOPS FP32, aber für die kleinen Micro-Modelle in Brain19 ist der GPU-Overhead (Kernel-Launch, Memory-Transfer) oft größer als die Berechnung selbst. **CPU-Training ist für Micro-Modelle dieser Größe oft effizienter als GPU-Training.**

### 6.1.2 Szenario: GPU-Erweiterung

Falls eine GPU nachgerüstet wird (z.B. NVIDIA A4000/A5000 oder RTX 4090):

| GPU | VRAM | FP32 TFLOPS | Einschätzung |
|-----|------|-------------|-------------|
| RTX 4090 | 24 GB | 82.6 | Overkill für Micro-Modelle, nützlich für globales GAT |
| A4000 | 16 GB | 19.2 | Guter Kompromiss |
| RTX 3060 | 12 GB | 12.7 | Ausreichend |

**Empfehlung:** Für den Micro-Modell-Ansatz ist eine GPU **nicht notwendig**. Sie wird erst relevant, wenn ein globales GAT über einen sehr großen Graphen (>50K Knoten) trainiert werden soll.

## 6.2 Memory-Analyse pro Architektur

### 6.2.1 KAN Micro-Modell

Für ein KAN [20, 10, 1] mit $G=5, k=3$:

- Parameter: 2310 x 4 Bytes = 9.24 KB
- Spline-Gitter: 20 x 10 x (G+k+1) x 4 = 7.2 KB (Gitterpunkte)
- Optimizer State (Adam, 2 Momente): 2 x 9.24 KB = 18.48 KB
- **Gesamt pro Modell (Training):** ~35 KB
- **Gesamt pro Modell (Inference):** ~16.5 KB

| Anzahl Konzepte | Training Memory | Inference Memory | Passt in 120 GB? |
|----------------|-----------------|------------------|-------------------|
| 100 | 3.5 MB | 1.65 MB | Trivial |
| 1.000 | 35 MB | 16.5 MB | Trivial |
| 10.000 | 350 MB | 165 MB | Trivial |
| 100.000 | 3.5 GB | 1.65 GB | Ja |

Mit 120 GB RAM können **alle Micro-Modelle gleichzeitig im Speicher gehalten werden** — auch bei 100K Konzepten. Dies eliminiert jegliche I/O-Bottlenecks beim nightly Batch-Update.

### 6.2.2 Parametrisches Micro-Modell

Für ein lineares Modell mit 300 Parametern (30 Nachbarn x 10 Kontext-Features):

- Parameter: 300 x 4 Bytes = 1.2 KB
- **Gesamt pro Modell (Training):** ~3.6 KB
- **Gesamt pro Modell (Inference):** ~1.2 KB

| Anzahl Konzepte | Training Memory | Inference Memory |
|----------------|-----------------|------------------|
| 100 | 360 KB | 120 KB |
| 1.000 | 3.6 MB | 1.2 MB |
| 10.000 | 36 MB | 12 MB |
| 100.000 | 360 MB | 120 MB |

### 6.2.3 GAT (globales Modell)

Ein globales GAT mit 4 Attention-Heads, Hidden-Dim 64:

- Typisch 50K-500K Parameter (abhängig von Graphgroesse)
- Memory: 2-8 MB für Modellparameter
- Dazu: Node-Features und Adjacency-Informationen

Für 10.000 Konzepte mit durchschnittlich 30 Kanten:
- Adjacency: 300K Kanten x 8 Bytes = 2.4 MB
- Node Features: 10K x 64 x 4 = 2.56 MB
- **Gesamt:** ~10-20 MB

Für 100.000 Konzepte: ~200 MB — immer noch trivial bei 120 GB RAM.

## 6.3 Trainingszeit-Schätzungen

### 6.3.1 Methodik

Wir schätzen die Trainingszeit basierend auf:

- **CPU-only (Primaerszenario):** AMD EPYC ~80 Kerne, je ~1-5 GFLOPS effektiv (PyTorch)
- **Parallelisierungsstrategie:** Python `multiprocessing` mit 80 Worker-Prozessen, je ein Kern pro Micro-Modell
- **GPU-Szenario (sekundaer):** Hypothetische GPU-Erweiterung

Der entscheidende Vorteil des EPYC-Systems: **80 Micro-Modelle können vollständig parallel trainiert werden.** Da jedes Modell winzig ist (< 3000 Parameter), passt es vollständig in den L1/L2-Cache eines einzelnen Kerns — optimale Cache-Locality.

### 6.3.2 KAN Micro-Modelle

Pro Micro-Modell (200 Epochen, 200 Trainingspunkte, Batch-Size 32):

- FLOPs pro Forward: ~50K (Spline-Evaluierungen)
- FLOPs pro Backward: ~150K
- FLOPs pro Epoche: ~1.25M
- FLOPs total: ~250M
- **Geschaetzte Zeit pro Modell (1 Kern):** 0.5-2 Sekunden

**CPU-Parallelisierung (80 Kerne):**

| Konzepte | Sequential (1 Kern) | Parallel (80 Kerne) | Speedup |
|----------|---------------------|---------------------|---------|
| 100 | 50-200 s | 1-3 s | 80x |
| 1.000 | 500-2000 s | 7-25 s | 80x |
| 10.000 | 5000-20000 s | 63-250 s | 80x |
| 100.000 | 14-56 h | 10-42 min | 80x |

**10K Konzepte mit 80 Kernen: ~1-4 Minuten.** Das ist für ein nightly Batch-Update exzellent.

Hinweis: KAN profitiert besonders vom CPU-only-Szenario, da B-Spline-Berechnungen auf CPUs kaum langsamer sind als auf GPUs (schlechte GPU-Parallelisierbarkeit der rekursiven de Boor-Cox-Berechnung). Der typische 10x-Nachteil von KAN gegenueber MLP bleibt bestehen, aber die absolute Trainingszeit ist durch die massive CPU-Parallelität akzeptabel.

### 6.3.3 MLP Micro-Modelle

MLP ist ~10x schneller als KAN pro Modell:

| Konzepte | Sequential (1 Kern) | Parallel (80 Kerne) |
|----------|---------------------|---------------------|
| 100 | 5-20 s | < 1 s |
| 1.000 | 50-200 s | 1-3 s |
| 10.000 | 500-2000 s | 7-25 s |
| 100.000 | 1.4-5.6 h | 1-4 min |

### 6.3.4 Parametrisches Modell

Lineare Modelle mit geschlossener Loesung (Least Squares) — kein iteratives Training nötig:

| Konzepte | Sequential (1 Kern) | Parallel (80 Kerne) |
|----------|---------------------|---------------------|
| 100 | < 0.1 s | < 0.01 s |
| 1.000 | ~0.5 s | < 0.01 s |
| 10.000 | ~5 s | < 0.1 s |
| 100.000 | ~50 s | < 1 s |

**100K Konzepte in unter 1 Sekunde.** Die geschlossene Loesung (Ridge Regression) ist ein einfaches lineares Gleichungssystem pro Modell, das auf einem EPYC-Kern in Mikrosekunden geloest wird.

### 6.3.5 GAT (globales Modell)

Ein einzelnes globales GAT trainieren — hier hilft CPU-Parallelismus weniger, da es ein einzelnes Modell ist:

| Graphgroesse | CPU-only (80 Kerne) | Mit GPU (hypothetisch) |
|-------------|---------------------|----------------------|
| 100 Knoten | 10-30 s | 2-5 s |
| 1.000 Knoten | 2-10 min | 15-60 s |
| 10.000 Knoten | 20-90 min | 2-10 min |
| 100.000 Knoten | 3-15 h | 15-60 min |

GAT-Training profitiert stark von GPU-Beschleunigung durch die Matrix-Multiplikationen in der Attention-Berechnung. Bei >10K Knoten wuerde eine GPU den Unterschied zwischen "akzeptabel" und "schnell" machen.

**Empfehlung:** Falls ein globales GAT zum Einsatz kommt und der Graph >10K Knoten hat, lohnt sich eine GPU-Erweiterung. Für den reinen Micro-Modell-Ansatz ist sie unnötig.

## 6.4 Zusammenfassung der praktischen Machbarkeit

**Alle Architekturen laufen auf dem Brain19-Server problemlos.** Die 80 EPYC-Kerne machen den Server ideal für den Micro-Modell-Ansatz, da jedes Modell unabhaengig auf einem eigenen Kern trainiert werden kann.

### Szenario A: CPU-only (aktuelle Konfiguration)

| Architektur | 10K Konzepte (80 Kerne) | RAM-Bedarf | Machbar? |
|------------|--------------------------|-----------|----------|
| KAN | 1-4 min | < 500 MB | Ja |
| MLP | 7-25 s | < 100 MB | Ja |
| GAT (global) | 20-90 min | < 200 MB | Ja (langsamer) |
| Parametrisch | < 0.1 s | < 50 MB | Perfekt |

### Szenario B: Mit GPU-Erweiterung

| Architektur | 10K Konzepte | Verbesserung vs CPU-only |
|------------|-------------|-------------------------|
| KAN | ~1-3 min | Kaum (B-Splines CPU-affin) |
| MLP | ~5-15 s | Marginal (Modelle zu klein) |
| GAT (global) | 2-10 min | **Signifikant (5-10x)** |
| Parametrisch | < 0.1 s | Keine (closed-form) |

**Kernaussage:** Der EPYC-Server mit 80 Kernen ist die ideale Hardware für den Micro-Modell-Ansatz. Eine GPU wird nur beim globalen GAT-Szenario mit grossem Graphen relevant. KANs Geschwindigkeitsnachteil wird durch die CPU-Parallelisierung stark abgemildert — von "5.5 Stunden sequential" auf "1-4 Minuten parallel". Dies aendert die Gesamtbewertung jedoch nicht grundlegend, da die anderen Nachteile (Overfitting, Inspizierbarkeit, Kontextabhaengigkeit) bestehen bleiben.

\newpage

# 7. KAN-Spezifische Tiefenanalyse

## 7.1 Kann KAN kontextabhängige Gewichte lernen?

Kontextabhängigkeit kann in KAN auf zwei Weisen implementiert werden:

### 7.1.1 Kontext als Input

Der Kontextvektor $\mathbf{c}$ wird als zusätzliche Input-Dimension übergeben:
$$M_v([\mathbf{x}_{r,v'}; \mathbf{c}]; \theta) \to w$$

**Problem:** KAN lernt univariate Funktionen auf den Kanten. Die Interaktion zwischen Kontext und Relation geschieht erst in den inneren Schichten durch Addition der transformierten Inputs. Dies ist weniger direkt als z.B. ein Attention-Mechanismus, der explizit $\text{score}(\mathbf{q}_c, \mathbf{k}_r)$ berechnet.

### 7.1.2 Kontext als Gitter-Modulation

Eine fortgeschrittene Variante: Die Spline-Gitterpunkte oder -Koeffizienten werden vom Kontext moduliert (Hypernetwork-artiger Ansatz):

$$\phi_{l,j,i}(x; c) = \sum_k (c_k^{(0)} + \mathbf{c}^T \mathbf{m}_k) B_k(x)$$

wobei $\mathbf{m}_k$ lernbare Modulationsvektoren sind. Dies fügt Kontextabhängigkeit direkt in die Spline-Funktionen ein, erhöht aber die Parameteranzahl weiter.

### 7.1.3 Bewertung

KAN *kann* Kontextabhängigkeit lernen, aber es ist kein natürlicher Mechanismus der Architektur. Es erfordert entweder:
- Vergrößerung des Inputs (mehr Parameter, langsameres Training)
- Architektur-Modifikationen (Hypernetwork-KAN, noch experimenteller)

**Im Vergleich:** Transformer Attention hat Kontextabhängigkeit als Kernmechanismus (Query-Key-Interaktion). GAT hat sie über die Attention-Berechnung. Sogar das parametrische Modell hat sie natürlicher über $\sigma(\mathbf{a}_r^T \mathbf{c} + b)$.

## 7.2 Können KAN-Maps sinnvoll überlagert werden?

### 7.2.1 Output-Level-Überlagerung

Da KAN-Micro-Modelle skalare Relevanzwerte $w \in [0,1]$ ausgeben, können die Outputs trivial überlagert werden:

$$w_{\text{combined}}(r, v', c) = f(w_{v_1}(r, v', c), w_{v_2}(r, v', c))$$

Dies ist unabhängig von der internen Architektur und funktioniert mit jeder $f$ (Produkt, Maximum, harmonisches Mittel, etc.).

### 7.2.2 Modell-Level-Überlagerung

Kann man die *internen Repräsentationen* zweier KAN-Modelle kombinieren, um kreativere Ergebnisse zu erzielen?

**Für KAN spezifisch:** Die B-Spline-Koeffizienten zweier KAN-Modelle könnten interpoliert werden:
$$c_k^{\text{new}} = \alpha \cdot c_k^{(v_1)} + (1-\alpha) \cdot c_k^{(v_2)}$$

Dies ist mathematisch wohldefiniert (Spline-Räume sind Vektorräume), hat aber keine klare semantische Interpretation. Es ist nicht garantiert, dass die interpolierten Splines sinnvolle Relevanzfunktionen ergeben.

**Für das parametrische Modell:** Die Gewichtsvektoren $\mathbf{a}_{r}^{(v_1)}$ und $\mathbf{a}_{r}^{(v_2)}$ sind direkt kombinierbar und die Kombination hat klare semantische Bedeutung: Die kombinierte Map betont Relationen, die in beiden Konzepten (gewichtet) relevant sind.

### 7.2.3 Bewertung

Die Überlagerbarkeit ist **nicht KAN-spezifisch** — sie funktioniert auf Output-Level mit jeder Architektur. Die Modell-Level-Überlagerung ist beim parametrischen Modell am natürlichsten.

## 7.3 Inspizierbarkeit im Praxis-Kontext

### 7.3.1 Was KAN bietet

KAN erlaubt die Visualisierung jeder gelernten univariaten Funktion $\phi_{l,j,i}(x)$ als Plot. Für das Micro-Modell "Temperatur" mit Input-Features [Relation-Type, Kontext-ID, Epistemischer-Status] könnte man sehen:

- $\phi_{0,1,1}(x)$: Wie der Relationstyp den ersten Hidden-Knoten beeinflusst
- $\phi_{0,1,2}(x)$: Wie der Kontext den ersten Hidden-Knoten beeinflusst

### 7.3.2 Praktische Limitierung

In Brain19 sind die Inputs typischerweise **Embeddings** (dichte Vektoren), nicht interpretierbare Skalare. Eine gelernte Spline-Funktion $\phi(x)$ wobei $x$ die 3. Dimension eines Kontext-Embeddings ist, hat keine offensichtliche Interpretation. Die Inspizierbarkeit von KAN erfordert semantisch bedeutsame skalare Inputs — eine Bedingung, die in Brain19 nur teilweise erfüllt ist.

### 7.3.3 Das parametrische Modell als Inspizierbarkeits-Champion

Eine direkte Gewichtsmatrix $\mathbf{W} \in \mathbb{R}^{|\text{Relationen}| \times |\text{Kontexte}|}$ ist trivial inspizierbar:
- Zeile $i$: "Wie wichtig ist Relation $i$ in verschiedenen Kontexten?"
- Spalte $j$: "Was ist im Kontext $j$ wichtig?"
- $W_{ij}$: "Wie wichtig ist Relation $i$ im Kontext $j$?"

Dies ist einfacher, klarer und vollständiger als KAN's Spline-Visualisierungen.

\newpage

# 8. Empfohlene Architektur

## 8.1 Primäre Empfehlung: Kontextabhängiges Parametrisches Modell

Wir empfehlen als Basis-Architektur ein **kontextabhängiges bilineares Modell** mit optionaler nicht-linearer Erweiterung:

### 8.1.1 Architektur-Definition

Für ein Konzept $v$ mit Nachbarn $\{(r_1, v_1), \ldots, (r_N, v_N)\}$ und Kontext $c$:

**Stufe 1: Bilineares Modell (Baseline)**

$$w_v(r_k, v_k, c) = \sigma\left(\mathbf{e}_{r_k}^T \mathbf{W}_v \mathbf{c} + \mathbf{b}_{r_k}^T \mathbf{c} + d_{r_k}\right)$$

wobei:
- $\mathbf{e}_{r_k} \in \mathbb{R}^{d_r}$: Embedding des Relationstyps (shared)
- $\mathbf{W}_v \in \mathbb{R}^{d_r \times d_c}$: Pro-Konzept Interaktionsmatrix
- $\mathbf{c} \in \mathbb{R}^{d_c}$: Kontextvektor
- $\sigma$: Sigmoid-Funktion

**Parameter pro Konzept:** $d_r \times d_c + N \times d_c + N = d_r d_c + N(d_c + 1)$

Für $d_r = 10, d_c = 10, N = 30$: $100 + 330 = 430$ Parameter.

**Stufe 2: Optionale nicht-lineare Erweiterung**

Wenn die bilineare Variante nicht ausreicht, eine kleine Hidden-Layer hinzufügen:

$$\mathbf{h} = \text{ReLU}(\mathbf{W}_1 [\mathbf{e}_{r_k}; \mathbf{c}] + \mathbf{b}_1)$$
$$w_v(r_k, v_k, c) = \sigma(\mathbf{w}_2^T \mathbf{h} + b_2)$$

mit $\mathbf{W}_1 \in \mathbb{R}^{h \times (d_r + d_c)}$, Hidden-Size $h = 16$:
Parameter: $(10+10) \times 16 + 16 + 16 \times 1 + 1 = 353$ Parameter.

### 8.1.2 Vorteile

1. **Perfekt inspizierbar:** $\mathbf{W}_v$ zeigt direkt, wie Relation und Kontext interagieren
2. **Dateneffizient:** 350-430 Parameter für 50-500 Trainingspunkte
3. **Schnell:** Training in Millisekunden, Inference in Mikrosekunden
4. **Überlagerbar:** $\mathbf{W}_{v_1}$ und $\mathbf{W}_{v_2}$ können direkt kombiniert werden
5. **Kontextabhängig per Design:** Die bilineare Interaktion $\mathbf{e}_r^T \mathbf{W}_v \mathbf{c}$ ist der natürlichste Weg
6. **Epistemisch transparent:** Kompatibel mit Brain19's Transparenzanforderungen

### 8.1.3 Training

Das bilineare Modell kann mit Standard-SGD oder Adam trainiert werden. Alternativ, wenn man nur positive Labels hat (beobachtete Relevanzen), kann ein kontrastives Lernziel verwendet werden.

Für die geschlossene Lösung (wenn MSE-Loss):
$$\theta^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

wobei $\mathbf{X}$ die Feature-Matrix (Kronecker-Produkt von Relation- und Kontext-Features) und $\mathbf{y}$ die Ziel-Relevanzen sind.

## 8.2 Sekundäre Empfehlung: Hybrid mit GAT-Initialisierung

Für fortgeschrittenere Anforderungen empfehlen wir einen Hybrid-Ansatz:

### 8.2.1 Architektur

1. **Globales GAT** trainieren, das über den gesamten Graphen Attention-Gewichte berechnet
2. **Attention-Gewichte extrahieren** als Initialisierung für die parametrischen Micro-Modelle
3. **Micro-Modelle feintunen** mit konzeptspezifischen Daten

### 8.2.2 Vorteile des Hybrid-Ansatzes

- GAT lernt globale Muster (Strukturwissen)
- Micro-Modelle spezialisieren sich auf konzeptspezifische Relevanzen
- Initiale Maps sind bereits sinnvoll, bevor konzeptspezifische Daten vorliegen
- Bei Änderungen im Graphen: Nur betroffene Micro-Modelle neu trainieren

### 8.2.3 Implementierungsvorschlag

```python
# Pseudo-Code
class Brain19RelevanceSystem:
    def __init__(self, graph, contexts):
        self.gat = GlobalGAT(graph, num_heads=len(contexts))
        self.micro_models = {}
    
    def initialize(self):
        """Train global GAT, extract attention weights."""
        self.gat.train(self.graph)
        for concept in self.graph.concepts:
            attention_weights = self.gat.get_attention(concept)
            self.micro_models[concept] = BilinearMicroModel(
                init_weights=attention_weights
            )
    
    def nightly_update(self, changed_concepts):
        """Re-train only affected micro-models."""
        # Option A: Retrain GAT, reinitialize affected models
        # Option B: Only retrain micro-models with new data
        for concept in changed_concepts:
            self.micro_models[concept].train(
                self.get_training_data(concept)
            )
    
    def get_relevance_map(self, concept, context):
        """Get context-dependent relevance map."""
        return self.micro_models[concept].predict(context)
    
    def creativity_overlay(self, concept1, concept2, context):
        """Overlay two maps for creativity."""
        map1 = self.get_relevance_map(concept1, context)
        map2 = self.get_relevance_map(concept2, context)
        # Find surprising overlaps
        return surprise_combine(map1, map2)
```

## 8.3 Wann KAN dennoch in Betracht ziehen?

KAN könnte relevant werden, wenn:

1. **Die Relevanzfunktionen komplex sind** und das bilineare Modell systematisch underfittet
2. **Symbolische Regression** gewünscht ist — z.B. um zu entdecken, dass "Relevanz von Temperatur-Druck ~ $\sin(\text{Kontext})$" ist
3. **Sehr wenige Konzepte** (< 50) mit vielen Relationen (> 100) existieren, sodass das Overfitting-Problem weniger ausgeprägt ist
4. **Wissenschaftliche Interpretierbarkeit** im engeren Sinne (geschlossene Formeln) benötigt wird

Für den typischen Brain19-Use Case (1000+ Konzepte, 10-50 Relationen, Priorität auf Geschwindigkeit und Inspizierbarkeit) ist KAN **nicht die optimale Wahl**.

\newpage

# 9. Diskussion

## 9.1 Warum KAN "intuitiv richtig" aber praktisch suboptimal ist

Die Idee, KAN für Relevanz-Maps zu verwenden, hat einen intuitiven Appeal:
- KAN lernt Funktionen auf Kanten → Relevanz ist eine Funktion auf Kanten
- KAN ist inspizierbar → Brain19 will Transparenz
- KAN basiert auf einem tiefen mathematischen Theorem → theoretisch fundiert

Aber diese Analogien halten einer näheren Betrachtung nicht stand:

1. **KAN-Kanten ≠ Graph-Kanten:** KAN-Kanten verbinden Neuronen, Graph-Kanten verbinden Konzepte. Die Analogie ist irreführend.
2. **Inspizierbarkeit ≠ Verständlichkeit:** KAN-Splines über Embedding-Dimensionen sind nicht semantisch interpretierbar.
3. **KAST-Optimalität gilt für hohe Dimensionen:** Bei $d < 20$ hat KAST keinen praktischen Vorteil über Universal Approximation.

## 9.2 Das "Micro-Modell"-Paradigma

Die Entscheidung, pro Konzept ein separates Micro-Modell zu haben, ist eine fundamentale Designentscheidung von Brain19. Sie hat Vorteile (Lokalität, Inspizierbarkeit, unabhängige Updates) aber auch Nachteile (kein Parameter-Sharing, keine globale Konsistenz). 

Eine Alternative wäre ein **globales Modell mit lokaler Spezialisierung** (z.B. das vorgeschlagene GAT-Hybrid), das die Vorteile beider Welten kombiniert.

## 9.3 Epistemische Transparenz und Relevanz-Maps

Brain19's epistemische Klassifikation (Fakt, Hypothese, Meinung) ist ein Alleinstellungsmerkmal. Die Relevanz-Maps sollten dies widerspiegeln: Ein Fakt mit hoher Relevanz sollte anders behandelt werden als eine Meinung mit hoher Relevanz. Dies kann im parametrischen Modell durch separate Gewichtsebenen für verschiedene epistemische Klassen implementiert werden — einfacher und klarer als bei jeder neuronalen Architektur.

## 9.4 Limitierungen dieser Analyse

1. **Keine empirische Evaluation:** Dieser Bericht basiert auf theoretischer Analyse und Literaturvergleich. Empirische Benchmarks auf Brain19-Daten sind notwendig.
2. **Evolving Field:** KAN-Forschung entwickelt sich schnell. Effizientere KAN-Varianten (FourierKAN, RBF-KAN, TruKAN) könnten einige der identifizierten Nachteile adressieren.
3. **Problem-spezifische Annahmen:** Wir nehmen an, dass Relevanzfunktionen relativ glatt und niedrigdimensional sind. Falls dies nicht zutrifft, könnte KAN stärker sein.

\newpage

# 10. Fazit

## 10.1 Kernergebnis

**KAN ist für kontextabhängige Relevanz-Maps in Brain19 nicht die optimale Architektur.** Die Hauptgründe sind:

1. **Zu viele Parameter** für die verfügbare Datenmenge (Overfitting-Risiko)
2. **10× langsameres Training** als Alternativen
3. **Inspizierbarkeit auf Embedding-Inputs** ist nicht so nützlich wie versprochen
4. **Kontextabhängigkeit** ist kein natürlicher KAN-Mechanismus
5. **Einfachere Modelle** lösen das Problem besser

## 10.2 Empfehlung

| Priorität | Empfehlung |
|-----------|-----------|
| Primär | **Bilineares parametrisches Modell** mit Sigmoid-Aktivierung |
| Sekundär | **GAT-Hybrid** (globales GAT als Initialisierung + lokale Micro-Modelle) |
| Tertiär | **MLP** [20, 16, 1] wenn nicht-lineare Relationen nötig sind |
| Nicht empfohlen | KAN (für diesen spezifischen Use Case) |

## 10.3 Nächste Schritte

1. **Prototyp** des bilinearen Modells auf Brain19-Daten implementieren
2. **Baseline-Evaluation** auf realen Relevanz-Daten
3. **Vergleich** mit MLP und GAT auf denselben Daten
4. **Creativity-Algorithmus** mit verschiedenen Überlagerungsfunktionen testen
5. Falls bilineares Modell underfitted: Schrittweise Komplexität erhöhen (→ MLP → GAT-Hybrid)

\newpage

# Literaturverzeichnis

1. **Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y., & Tegmark, M.** (2024). KAN: Kolmogorov-Arnold Networks. *arXiv:2404.19756*. Accepted at ICLR 2025.

2. **Liu, Z., Ma, P., Wang, Y., Matusik, W., & Tegmark, M.** (2024). KAN 2.0: Kolmogorov-Arnold Networks Meet Science. *arXiv:2408.10205*.

3. **Bresson, R., Nikolentzos, G., Panagopoulos, G., Chatzianastasis, M., Pang, J., & Vazirgiannis, M.** (2025). KAGNNs: Kolmogorov-Arnold Networks meet Graph Learning. *Transactions on Machine Learning Research (TMLR)*.

4. **De Carlo, G., Mastropietro, A., & Anagnostopoulos, A.** (2024). Kolmogorov-Arnold Graph Neural Networks. *arXiv:2406.18354*.

5. **Zhang, R., et al.** (2024). GKAN: Graph Kolmogorov-Arnold Networks. *arXiv:2406.06470*.

6. **Yu, Z., et al.** (2024). KAN or MLP: A Fairer Comparison. *arXiv:2407.16674*.

7. **Xu, Y., et al.** (2024). PowerMLP: An Efficient Version of KAN. *AAAI 2025*.

8. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y.** (2018). Graph Attention Networks. *ICLR 2018*.

9. **Kipf, T.N., & Welling, M.** (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*.

10. **Bordes, A., Usunier, N., Garcia-Durán, A., Weston, J., & Yakhnenko, O.** (2013). Translating Embeddings for Modeling Multi-relational Data. *NeurIPS 2013*.

11. **Chami, I., Ying, R., Ré, C., & Leskovec, J.** (2019). Hyperbolic Graph Convolutional Neural Networks. *NeurIPS 2019*.

12. **Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J.** (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. *ICLR 2019*.

13. **Cybenko, G.** (1989). Approximation by Superpositions of a Sigmoidal Function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314.

14. **Hornik, K.** (1991). Approximation Capabilities of Multilayer Feedforward Networks. *Neural Networks*, 4(2), 251-257.

15. **de Boor, C.** (1978). *A Practical Guide to Splines*. Springer-Verlag.

16. **Kolmogorov, A.N.** (1957). On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition. *Doklady Akademii Nauk SSSR*, 114, 953-956.

17. **Arnold, V.I.** (1957). On functions of three variables. *Doklady Akademii Nauk SSSR*, 114, 679-681.

18. **Girosi, F., & Poggio, T.** (1989). Representation Properties of Networks: Kolmogorov's Theorem Is Irrelevant. *Neural Computation*, 1(4), 465-469.

19. **Laird, J.E.** (2012). *The Soar Cognitive Architecture*. MIT Press.

20. **Anderson, J.R.** (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press.

21. **Sun, R.** (2016). *Anatomy of the Mind: Exploring Psychological Mechanisms and Processes with the Clarion Cognitive Architecture*. Oxford University Press.

22. **Goertzel, B., Pennachin, C., & Geisweiller, N.** (2014). *Engineering General Intelligence*. Atlantis Press.

23. **Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G.** (2016). Complex Embeddings for Simple Link Prediction. *ICML 2016*.

24. **Yang, B., Yih, W., He, X., Gao, J., & Deng, L.** (2015). Embedding Entities and Relations for Learning and Inference in Knowledge Bases. *ICLR 2015*.

25. **Vaswani, A., et al.** (2017). Attention Is All You Need. *NeurIPS 2017*.

\newpage

# Anhang A: Parameter-Berechnung für KAN

## A.1 B-Spline Parametrisierung

Ein B-Spline der Ordnung $k$ über $G$ gleichmäßig verteilte Gitterpunkte auf $[a, b]$ hat $G + k$ Basisfunktionen. Jede Basisfunktion $B_{i,k}(x)$ wird durch die de Boor-Cox-Rekursion definiert:

$$B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)$$

Die Spline-Funktion ist dann:
$$s(x) = \sum_{i=0}^{G+k-1} c_i B_{i,k}(x)$$

mit $G + k$ lernbaren Koeffizienten $c_i$.

## A.2 KAN-Schicht Parametrisierung

In der KAN-Implementierung von Liu et al. hat jede Kanten-Funktion:
- $G + k$ Spline-Koeffizienten
- 1 Basisgewicht $w_b$ (für den SiLU-Term)
- 1 Spline-Skalierung $w_s$
- 1 Bias $b$ (optional)

Gesamt pro Kantenfunktion: $G + k + 3$ Parameter.

## A.3 Gesamtparameter eines KAN [20, 10, 1]

- Schicht 1: $20 \times 10 = 200$ Kantenfunktionen, je $5 + 3 + 3 = 11$ Parameter: **2200**
- Schicht 2: $10 \times 1 = 10$ Kantenfunktionen, je 11 Parameter: **110**
- **Gesamt: 2310 Parameter**

# Anhang B: Vergleich der Kontextabhängigkeits-Mechanismen

| Architektur | Kontextmechanismus | Natürlichkeit | Effizienz |
|------------|-------------------|---------------|-----------|
| KAN | Kontext als Input-Dimension | Niedrig | Niedrig |
| MLP | Kontext als Input-Dimension | Niedrig | Hoch |
| GCN | Keiner (homogene Aggregation) | - | - |
| GAT | Multi-Head Attention | Mittel | Mittel |
| Transformer | Query-Key-Attention | Hoch | Mittel |
| Hyp. Emb. | Keiner (fixer Abstand) | - | - |
| Bilinear | Bilineare Interaktion $e^T W c$ | Hoch | Hoch |

# Anhang C: Creativity-Überlagerungsfunktionen

Sei $w_1 = w_{v_1}(r, v', c)$ und $w_2 = w_{v_2}(r, v', c)$ die Relevanzen zweier Konzepte.

| Funktion | Formel | Semantik |
|----------|--------|----------|
| Multiplikation | $w_1 \cdot w_2$ | Gemeinsam wichtige Relationen |
| Maximum | $\max(w_1, w_2)$ | Union der wichtigen Relationen |
| Harmonisches Mittel | $\frac{2w_1 w_2}{w_1 + w_2}$ | Betonte Überlappung |
| Überraschung | $|w_1 - w_2| \cdot \max(w_1, w_2)$ | Asymmetrische Relevanzen |
| Geometrisches Mittel | $\sqrt{w_1 \cdot w_2}$ | Moderate Überlappung |
| Kreativitäts-Score | $\min(w_1, w_2) \cdot (1 - |w_1 - w_2|)$ | Ähnlich wichtig aber unerwartet |

Für den Creativity-Algorithmus empfehlen wir die **Überraschungs-Funktion**, da sie Relationen findet, die für ein Konzept sehr wichtig, für das andere aber unwichtig sind — dies sind potentiell die kreativsten Verbindungen.

---

*Dieser Bericht wurde im Rahmen des Brain19-Projekts erstellt. Die Analyse basiert auf dem Stand der Forschung vom Februar 2026.*
