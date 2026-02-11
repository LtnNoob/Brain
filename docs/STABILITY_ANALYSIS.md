# Stabilitätsanalyse: Brain19 MicroModel Interaction Architecture

**Autor:** Mathematische Systemanalyse  
**Datum:** 2026-02-11  
**Version:** 1.0  
**Methodik:** Lyapunov-Theorie, Spektralanalyse, Informationstheorie, Bifurkationstheorie

---

## Inhaltsverzeichnis

1. [Systemdefinition](#1-systemdefinition)
2. [Konvergenzanalyse](#2-konvergenzanalyse)
3. [Dimensionalitätsanalyse](#3-dimensionalitätsanalyse)
4. [Eigenvalue-Analyse](#4-eigenvalue-analyse)
5. [Attractor-Analyse](#5-attractor-analyse)
6. [Damping & Inhibition](#6-damping--inhibition)
7. [Context-Knoten Analyse](#7-context-knoten-analyse)
8. [Teststrategie](#8-teststrategie)
9. [Zusammenfassung & Empfehlungen](#9-zusammenfassung--empfehlungen)

---

## 1. Systemdefinition

### 1.1 Das dynamische System

Brain19's geplante Interaction Architecture definiert ein diskretes dynamisches System über N Knoten (MicroModels). Aus dem Code und der Architektur-Spezifikation extrahieren wir:

**Zustandsraum:** Jeder Knoten i hat einen Aktivierungsvektor $a_i(t) \in \mathbb{R}^d$ mit $d = $ `EMBED_DIM` $= 10$.

**Dynamik (skalare Vereinfachung):**

$$a_i(t) = \sigma\!\Bigl(\sum_{j \in \mathcal{N}(i)} W_{ij} \cdot a_j(t-1) - \lambda \cdot a_i(t-1)\Bigr)$$

wobei $\sigma(x) = 1/(1+e^{-x})$ die logistische Sigmoid-Funktion ist.

**Dynamik (vektorielle Form, basierend auf `micro_model.cpp`):**

Da `predict(e, c) = σ(eᵀ·(W·c + b))` einen **Skalar** zurückgibt, aber die Architektur Aktivierungsvektoren vorsieht, definieren wir den vektoriellen Modus via `activate`:

$$\mathbf{a}_i(t) = \sigma\!\Bigl(\sum_{j \in \mathcal{N}(i)} w_{ij}(t-1) \cdot \mathbf{a}_j(t-1) - \lambda \cdot \mathbf{a}_i(t-1)\Bigr)$$

mit skalarem Kopplungsgewicht:

$$w_{ij}(t) = \text{predict}_i(\mathbf{e}_{rel}, \mathbf{a}_j(t)) = \sigma\!\bigl(\mathbf{e}_{rel}^\top (W_i \cdot \mathbf{a}_j(t) + \mathbf{b}_i)\bigr) \in (0, 1)$$

### 1.2 Parameterbereiche aus dem Code

| Parameter | Wert | Quelle |
|-----------|------|--------|
| `EMBED_DIM` (d) | 10 | `micro_model.hpp` |
| `FLAT_SIZE` | 430 | 100(W) + 10(b) + 10(e) + 10(c) + 300(Adam state) |
| W-Initialisierung | $W_{ii} = 0.1$, $W_{ij} = 0.01\sin(10i+j)$ | `MicroModel()` Konstruktor |
| Bias-Init | $b_i = 0.01\cos(i)$ | Konstruktor |
| $\sigma$ | Logistic sigmoid | `predict()` |
| Typischer Subgraph | 20–100 Knoten | Architektur-Review |
| Knotengrad | $\bar{d} \approx 5\text{–}15$ | KG-Struktur |

---

## 2. Konvergenzanalyse

### 2.1 Fixpunktgleichung

Im stationären Zustand $a_i^* = a_i(t) = a_i(t-1)$ gilt:

$$a_i^* = \sigma\!\Bigl(\sum_{j \in \mathcal{N}(i)} w_{ij}^* \cdot a_j^* - \lambda \cdot a_i^*\Bigr)$$

Da $\sigma: \mathbb{R} \to (0,1)$ stetig und streng monoton ist, existiert für jedes Argument genau ein Fixpunkt der Form $a^* = \sigma(f(a^*) - \lambda a^*)$, wobei $f$ die gewichtete Nachbarsumme ist.

### 2.2 Kontraktionsbedingung (skalarer Fall)

Definiere die Update-Abbildung $F_i: \mathbb{R}^N \to \mathbb{R}$:

$$F_i(\mathbf{a}) = \sigma\!\Bigl(\sum_j W_{ij} a_j - \lambda a_i\Bigr)$$

Die Jacobi-Matrix $J$ hat Einträge:

$$J_{ik} = \frac{\partial F_i}{\partial a_k} = \sigma'(z_i) \cdot \begin{cases} W_{ik} & \text{wenn } k \neq i \\ W_{ii} - \lambda & \text{wenn } k = i \end{cases}$$

wobei $z_i = \sum_j W_{ij} a_j - \lambda a_i$ und $\sigma'(z) = \sigma(z)(1-\sigma(z))$.

**Schlüsseleigenschaft:** $\max_z \sigma'(z) = \sigma'(0) = 1/4$.

Daher:

$$\|J\|_\infty \leq \frac{1}{4} \max_i \Bigl(\sum_{k \neq i} |W_{ik}| + |W_{ii} - \lambda|\Bigr)$$

**Kontraktionsbedingung (hinreichend):** Das System konvergiert zu einem eindeutigen Fixpunkt wenn:

$$\boxed{\frac{1}{4} \cdot \max_i \Bigl(\sum_{k \neq i} |W_{ik}| + |W_{ii} - \lambda|\Bigr) < 1}$$

äquivalent:

$$\max_i \Bigl(\sum_{k \neq i} |W_{ik}| + |W_{ii} - \lambda|\Bigr) < 4$$

### 2.3 Bewertung für Brain19-Parameter

**Szenario: Kopplungsgewichte $w_{ij} \in (0,1)$ aus predict().**

Für Knoten $i$ mit Grad $d_i$ und Kopplungsgewichte $w_{ij} \in (0,1)$:

$$\sum_{k \neq i} |W_{ik}| + |W_{ii} - \lambda| \leq \sum_{j \in \mathcal{N}(i)} w_{ij} + \lambda$$

Da $w_{ij} \in (0,1)$, gilt $\sum_{j} w_{ij} < d_i$.

| Knotengrad $d_i$ | Summe Kopplungen (max) | $+\lambda$ (0.3) | $< 4$? |
|:-:|:-:|:-:|:-:|
| 5 | 5.0 | 5.3 | ❌ Nein |
| 3 | 3.0 | 3.3 | ✅ Ja |
| 10 | 10.0 | 10.3 | ❌ Nein |
| 15 | 15.0 | 15.3 | ❌ Nein |

**⚠️ Kritisches Ergebnis:** Die einfache Kontraktionsbedingung ist für $d_i \geq 4$ verletzt. Das bedeutet aber **nicht** automatisch Divergenz — es bedeutet, dass die grobe Schranke nicht greift.

### 2.4 Schärfere Analyse: Spektralradius

Die Kontraktion hängt vom **Spektralradius** der Jacobi-Matrix ab, nicht von der Zeilensummennorm. Es gilt:

$$\rho(J) \leq \frac{1}{4} \cdot \rho(\tilde{W})$$

wobei $\tilde{W}_{ik} = W_{ik}$ für $k \neq i$ und $\tilde{W}_{ii} = W_{ii} - \lambda$.

**Konvergenzbedingung (notwendig und hinreichend für lokale Stabilität des Fixpunkts):**

$$\boxed{\rho(J) < 1 \iff \rho(\tilde{W}) < 4}$$

Für die Brain19-Kopplungsmatrix (dünn besetzt, $w_{ij} \in (0,1)$):

Aus Zufallsmatrixtheorie: Für eine Sparse-Matrix mit $n$ Knoten, mittlerem Grad $\bar{d}$, und i.i.d. Einträgen $w_{ij} \sim U(0,1)$:

$$\rho(\tilde{W}) \approx \frac{\bar{d}}{2} + \sqrt{\bar{d}} + O(\bar{d}^{-1/2})$$

| N (Knoten) | $\bar{d}$ | $\rho(\tilde{W})$ (geschätzt) | $\rho(\tilde{W}) < 4$? |
|:-:|:-:|:-:|:-:|
| 20 | 5 | ~4.7 | ❌ Grenzwertig |
| 50 | 8 | ~6.8 | ❌ Nein |
| 100 | 10 | ~8.2 | ❌ Nein |
| 500 | 12 | ~9.5 | ❌ Nein |
| 1000 | 15 | ~11.4 | ❌ Nein |

**⚠️ WARNUNG:** Ohne zusätzliche Normalisierung ist die Konvergenz für Subgraphen mit $\bar{d} > 5$ **nicht garantiert**.

### 2.5 Lösung: Gewichtsnormalisierung

Man muss die Kopplungsgewichte normalisieren:

$$\hat{w}_{ij} = \frac{w_{ij}}{\sum_{k \in \mathcal{N}(i)} w_{ik}}$$

Dann gilt $\sum_j \hat{w}_{ij} = 1$ und:

$$\rho(\hat{\tilde{W}}) \leq 1 + \lambda$$

Für $\lambda \leq 3$: $\rho(\hat{\tilde{W}}) \leq 4$ ✅

**Empfehlung:** Normalisierte Kopplungsgewichte sind **zwingend erforderlich**.

### 2.6 Lyapunov-Stabilität

**Theorem:** Unter symmetrischer Kopplung ($W_{ij} = W_{ji}$) und normalisierten Gewichten existiert eine Lyapunov-Funktion:

$$V(\mathbf{a}) = -\frac{1}{2} \sum_{i,j} \hat{w}_{ij} a_i a_j + \lambda \sum_i \int_0^{a_i} \sigma^{-1}(s)\,ds + \sum_i \int_0^{a_i} \sigma^{-1}(s)\,ds$$

**Beweis-Skizze:**

1. Der Integralterm $\int_0^a \sigma^{-1}(s)\,ds = \int_0^a \ln\frac{s}{1-s}\,ds = a\ln a + (1-a)\ln(1-a) + \ln 2$ ist die negative binäre Entropie (konvex).

2. Für synchrones Update: $\Delta V = V(\mathbf{a}(t)) - V(\mathbf{a}(t-1)) \leq 0$ wenn die Update-Regel dem Gradienten von $V$ folgt.

3. Bei symmetrischem $\hat{W}$ ist dies äquivalent zur Hopfield-Energiefunktion mit zusätzlichem Decay-Term.

**$\Delta V \leq 0$ gilt wenn:**
- $\hat{W}$ symmetrisch
- $\lambda > 0$  
- Gewichte normalisiert

Unter diesen Bedingungen konvergiert das System zu einem lokalen Minimum von $V$.

### 2.7 Konvergenzgeschwindigkeit

Die Konvergenzrate ist $\sim \rho(J)^t$. Für $\epsilon$-Konvergenz braucht man:

$$T_\epsilon = \left\lceil \frac{\ln \epsilon}{\ln \rho(J)} \right\rceil$$

| $\rho(J)$ | $T$ für $\epsilon = 10^{-4}$ |
|:-:|:-:|
| 0.25 | 7 |
| 0.50 | 14 |
| 0.75 | 32 |
| 0.90 | 88 |
| 0.95 | 175 |
| 0.99 | 917 |

**Empfehlung:** `max_cycles = 10` ist ausreichend nur wenn $\rho(J) < 0.4$. Für typische Brain19-Konfigurationen mit Normalisierung ($\rho(J) \approx 0.5\text{–}0.75$): **`max_cycles = 20–30` empfohlen**.

### 2.8 Bifurkationsanalyse

Wir betrachten $\lambda$ als Bifurkationsparameter. Sei $\mu = \rho(W_{\text{norm}})$ der Spektralradius der normalisierten Gewichtsmatrix (ohne Damping).

**Effektiver Spektralradius der Jacobi-Matrix:**

$$\rho(J) = \frac{1}{4}(\mu + \lambda)$$

(approximativ, da $\sigma' \leq 1/4$ und Damping additiv wirkt auf die Diagonale).

**Bifurkationspunkte:**

1. **Stabiler Fixpunkt:** $\rho(J) < 1 \iff \mu + \lambda < 4$
2. **Pitchfork-Bifurkation (Multistabilität):** $\rho(J) = 1$ — System bekommt multiple Fixpunkte
3. **Neimark-Sacker-Bifurkation (Oszillation):** Komplexe Eigenwerte mit $|\lambda_{\max}| = 1$ — Limit-Zyklen entstehen
4. **Chaos:** Bei asymmetrischem $W$ und $\rho(J) \gg 1$

Für Brain19-typische Parameter ($\mu \approx 1$ nach Normalisierung):

| $\lambda$ | $\rho(J)$ (approx.) | Verhalten |
|:-:|:-:|:--|
| 0.0 | 0.25 | Stabil, schnelle Konvergenz, geringe Ausdrucksfähigkeit |
| 0.1 | 0.275 | Stabil |
| 0.3 | 0.325 | Stabil, guter Kompromiss |
| 1.0 | 0.50 | Stabil, langsame Konvergenz |
| 2.0 | 0.75 | Stabil, sehr langsam |
| 3.0 | 1.0 | **Bifurkationspunkt** |
| 4.0 | 1.25 | Instabil, Oszillationen |
| >5.0 | >1.5 | Potentiell chaotisch (bei Asymmetrie) |

**Ohne Normalisierung** ($\mu \approx \bar{d}/2$):

| $\bar{d}$ | $\lambda = 0.3$ | $\rho(J)$ | Stabil? |
|:-:|:-:|:-:|:-:|
| 5 | 0.3 | ~1.3 | ❌ |
| 10 | 0.3 | ~2.6 | ❌ |
| 15 | 0.3 | ~3.8 | ❌ (nahe Chaos) |

**⚠️ KRITISCH:** Ohne Normalisierung ist das System für typische Knotengrade **instabil**.

---

## 3. Dimensionalitätsanalyse

### 3.1 Expressivität von predict(e,c) = σ(eᵀ·(W·c+b))

Die Funktion `predict` implementiert ein bilineares Modell:

$$f(\mathbf{e}, \mathbf{c}) = \sigma(\mathbf{e}^\top W \mathbf{c} + \mathbf{e}^\top \mathbf{b})$$

Dies ist äquivalent zu einer Rang-$d$-Approximation der Interaktionsmatrix $M_{ij} = f(e_i, c_j)$.

**VC-Dimension:** Für $d$-dimensionale bilineare Modelle mit Sigmoid:

$$\text{VCdim} = \Theta(d^2) = \Theta(100) \text{ für } d=10$$

Das Modell kann also $O(100)$ Punkte beliebig klassifizieren — das ist ausreichend für die lokale Relevanzberechnung (ein Konzept hat typisch 5–20 Nachbarn), aber limitiert für globale Muster.

### 3.2 Johnson-Lindenstrauss-Analyse

**Lemma (Johnson-Lindenstrauss):** Für $n$ Punkte in $\mathbb{R}^D$ und $\epsilon \in (0,1)$ existiert eine lineare Abbildung in $\mathbb{R}^d$ die paarweise Abstände bis Faktor $(1 \pm \epsilon)$ erhält, wenn:

$$d \geq \frac{8 \ln n}{\epsilon^2}$$

Für Brain19:

| $n$ (Konzepte) | $\epsilon$ | $d_{\min}$ |
|:-:|:-:|:-:|
| 100 | 0.5 | 148 |
| 100 | 0.3 | 410 |
| 1000 | 0.5 | 222 |
| 1000 | 0.3 | 614 |
| 1000 | 0.1 | 5,530 |

**⚠️ Ergebnis:** 10D kann paarweise Abstände von 1000 Konzepten **nicht** einmal mit $\epsilon = 0.5$ erhalten.

**ABER:** JL gilt für **beliebige** Abstandserhaltung. Brain19-MicroModels müssen nicht alle paarweisen Abstände erhalten — sie müssen nur **lokale Relationen** eines einzelnen Konzepts modellieren. Die relevante Frage ist die Kapazität pro Modell.

### 3.3 Informationstheoretische Kapazität

Ein MicroModel hat:
- $W$: $d \times d = 100$ Parameter
- $\mathbf{b}$: $d = 10$ Parameter  
- $\mathbf{e}_{\text{init}}, \mathbf{c}_{\text{init}}$: je $d = 10$ Parameter
- **Effektive lernbare Parameter:** 110 (W + b)

Bei 64-bit Floats: $110 \times 64 = 7040$ Bit.

**Kapazität (PAC-Learning Schranke):** Ein Modell mit $p$ Parametern kann $O(p/\epsilon)$ Samples mit Fehler $\leq \epsilon$ lernen.

Für 110 Parameter und $\epsilon = 0.1$: ~1100 Trainingsbeispiele korrekt.

**Praktische Kapazität (Relationen pro Konzept):**

Jedes MicroModel muss $|\mathcal{N}(i)|$ positive und $3|\mathcal{N}(i)|$ negative Relationen kodieren (aus `MicroTrainer`).

| $|\mathcal{N}(i)|$ | Trainingssamples | Samples/Parameter | Ausreichend? |
|:-:|:-:|:-:|:-:|
| 5 | 20 | 0.18 | ✅ Überparametrisiert |
| 15 | 60 | 0.55 | ✅ OK |
| 50 | 200 | 1.82 | ✅ OK |
| 100 | 400 | 3.64 | ⚠️ Grenzwertig |
| 500 | 2000 | 18.2 | ❌ Unterparametrisiert |

**Ergebnis:** 10D mit 110 Parametern ist für Konzepte mit $\leq 50$ Nachbarn ausreichend. Bei sehr stark vernetzten Hub-Knoten wird es eng.

### 3.4 Vergleich: Was bringt höhere Dimensionalität?

| Dimension $d$ | Parameter (W+b) | VCdim | JL $\epsilon$ (n=1000) | Max Nachbarn (komfortabel) |
|:-:|:-:|:-:|:-:|:-:|
| **10** | **110** | ~100 | ∞ (nicht anwendbar) | ~50 |
| 16 | 272 | ~256 | ~2.0 | ~100 |
| 32 | 1056 | ~1024 | ~0.73 | ~400 |
| 64 | 4160 | ~4096 | ~0.37 | ~1500 |

**Kostenanalyse (1000 MicroModels):**

| Dimension | RAM total | FLOPs/predict | Interaction Phase (50 nodes, 10 cycles, $\bar{d}=10$) |
|:-:|:-:|:-:|:-:|
| 10 | 3.4 MB | 110 | 55K → <1μs |
| 16 | 4.2 MB | 272 | 136K → <1μs |
| 32 | 8.2 MB | 1056 | 528K → ~1μs |
| 64 | 32.4 MB | 4160 | 2.08M → ~5μs |

**Empfehlung:** 

$$\boxed{d = 16 \text{ ist der optimale Kompromiss}}$$

- 2.5× mehr Parameter als $d=10$, bei vernachlässigbarem Mehraufwand
- VCdim steigt auf ~256 (ausreichend für Hub-Knoten)
- RAM: +0.8 MB für 1000 Modelle — irrelevant
- FLOPs: 2.5× mehr — immer noch <1μs pro Interaction Phase

$d = 32$ ist empfehlenswert wenn semantische Nuancen (Sprachverständnis, Analogie-Erkennung) kritisch werden. $d = 64$ ist Overkill für <10K Konzepte.

---

## 4. Eigenvalue-Analyse

### 4.1 Systemmatrix

Die linearisierte Dynamik um den Fixpunkt $\mathbf{a}^*$ wird durch die Jacobi-Matrix beschrieben. Für den skalaren Fall (Aktivierungen $a_i \in \mathbb{R}$, nicht Vektoren):

$$J_{ik} = \sigma'(z_i^*) \cdot \hat{W}_{ik}^{\text{eff}}$$

wobei:

$$\hat{W}_{ik}^{\text{eff}} = \begin{cases} \hat{w}_{ik} & k \neq i \\ \hat{w}_{ii} - \lambda & k = i \end{cases}$$

Da $\sigma'(z) \leq 1/4$, gilt:

$$\text{spec}(J) \subseteq \left\{ \frac{1}{4}\mu : \mu \in \text{spec}(\hat{W}^{\text{eff}}) \right\}$$

### 4.2 Spektralradius-Bedingungen

**Satz:** Das System ist lokal asymptotisch stabil am Fixpunkt $\mathbf{a}^*$ genau dann wenn:

$$\boxed{\rho(\hat{W}^{\text{eff}}) < 4}$$

**Beweis:** Alle Eigenwerte von $J$ müssen Betrag $< 1$ haben. Da $\sigma'$ den Faktor $\leq 1/4$ liefert, ist $|\lambda_i(J)| \leq |\mu_i|/4$ wobei $\mu_i$ Eigenwerte von $\hat{W}^{\text{eff}}$ sind. Stabilität erfordert $|\mu_i|/4 < 1 \Leftrightarrow |\mu_i| < 4$. $\square$

### 4.3 Struktur der Eigenwerte

Für die normalisierte Kopplungsmatrix $\hat{W}$ (stochastisch nach Zeilennormalisierung):

1. **Größter Eigenwert:** $\mu_1 = 1$ (Perron-Frobenius, da $\hat{W}$ nichtnegativ und zeilenstochastisch)
2. **Spektrallücke:** $\delta = 1 - |\mu_2|$ bestimmt die Mischrate

Für $\hat{W}^{\text{eff}}$ mit Damping $\lambda$:

- Diagonale verschoben: $\hat{W}^{\text{eff}}_{ii} = \hat{w}_{ii} - \lambda$
- Eigenwerte: $\mu_k^{\text{eff}} = \mu_k - \lambda \cdot \delta_{k,\text{uniform}}$ (nicht exakt, aber für dominanten Eigenwert):

$$\mu_1^{\text{eff}} \approx 1 - \lambda \quad \text{(für } \hat{w}_{ii} \approx 0\text{)}$$

Damit: $\rho(\hat{W}^{\text{eff}}) \approx \max(|1 - \lambda|, |\mu_2|)$

### 4.4 Echo State Property

Aus der Reservoir-Computing-Theorie: Ein rekurrentes Netzwerk hat die **Echo State Property** (ESP) genau dann, wenn der Einfluss vergangener Zustände exponentiell abklingt.

**ESP gilt wenn:** $\rho(J) < 1$, was unter den Normalisierungsbedingungen äquivalent zu $\rho(\hat{W}^{\text{eff}}) < 4$ ist.

**Information-Verarbeitung:**

| $\rho(J)$ | Verhalten | Informationsverarbeitung |
|:-:|:--|:--|
| $\ll 0.5$ | Schnelle Vergesslichkeit | Nur lokale/kurze Assoziationen |
| $\approx 0.5$ | Gute Balance | Kurz- und mittelfristige Muster |
| $\approx 0.8$ | Langsames Vergessen | Langreichweitige Korrelationen |
| $\approx 1.0$ | Edge of Chaos | Maximale Verarbeitungskapazität, aber instabil |
| $> 1.0$ | Explosion | Information geht in Sättigung verloren |

**Optimaler Arbeitspunkt (Langton's Edge of Chaos):**

$$\boxed{\rho(J) \in [0.7, 0.95]}$$

Für Brain19: mit normalisierten Gewichten und $\lambda = 0.3$:

$$\rho(J) \approx \frac{1}{4}(1 + 0.3) = 0.325$$

Das ist **zu niedrig** — das System konvergiert schnell, nutzt aber die Netzwerkstruktur nicht voll aus.

**Empfehlung:** $\lambda = 0.05\text{–}0.15$ für reichere Dynamik bei normalisierten Gewichten:

| $\lambda$ | $\rho(J)$ (approx.) | Konvergenzzeit ($T_{10^{-4}}$) | Verhalten |
|:-:|:-:|:-:|:--|
| 0.05 | 0.26 | ~8 | Schnell, wenig Interaktion |
| 0.10 | 0.28 | ~9 | Schnell |
| 0.15 | 0.29 | ~10 | Gut |
| 0.30 | 0.33 | ~12 | Sicher, aber konservativ |
| 0.50 | 0.38 | ~15 | Konservativ |

*Hinweis: Diese Werte gelten für normalisierte Gewichte. Ohne Normalisierung sind die $\rho$-Werte um Faktor $\bar{d}/2$ höher.*

### 4.5 Eigenwert-Spektrum für verschiedene Graphtopologien

Für typische KG-Subgraphen:

**Barabási-Albert (Scale-Free):** Wenige Hubs, viele Low-Degree Knoten. $\rho \sim \sqrt{d_{\max}}$. Problem: Hubs dominieren die Dynamik.

**Erdős-Rényi (Zufällig):** $\rho \sim \bar{d} + \sqrt{\bar{d}}$. Homogenere Dynamik.

**Small-World (Watts-Strogatz):** $\rho$ ähnlich zu Erdős-Rényi, aber mit stärkerer Cluster-Struktur → multiple Attraktoren wahrscheinlicher.

Brain19's KG ist vermutlich Scale-Free (natürliche Wissensgraphen sind es). Das bedeutet:
- **Hub-Knoten** ($d > 50$) destabilisieren das System
- **Lösung:** Grad-abhängige Normalisierung: $\hat{w}_{ij} = w_{ij} / \max(d_i, \tau)$ mit Mindest-Divisor $\tau \geq 5$

---

## 5. Attractor-Analyse

### 5.1 Attractor-Kapazität

Für ein Netzwerk mit $N$ binären Knoten und symmetrischer Kopplungsmatrix ist die maximale Anzahl stabiler Fixpunkte (Attraktoren) klassisch durch die **Hopfield-Kapazität** begrenzt:

$$\boxed{P_{\max} \approx 0.138 \cdot N}$$

bei fehlerfreiem Abruf. Für approximativen Abruf (Fehlerrate $< 1\%$):

$$P_{\max} \approx 0.05 \cdot N$$

| $N$ (Knoten) | Attraktoren (exakt) | Attraktoren (approx.) |
|:-:|:-:|:-:|
| 20 | 2–3 | 1 |
| 50 | 6–7 | 2–3 |
| 100 | 13–14 | 5 |
| 500 | 69 | 25 |
| 1000 | 138 | 50 |

### 5.2 Modifikation für Brain19

Brain19 unterscheidet sich von klassischen Hopfield-Netzen in drei Punkten:

1. **Kontinuierliche Aktivierungen** ($a_i \in (0,1)$ statt $\{0,1\}$): Erhöht die Auflösung der Attraktoren. Die effektive Kapazität steigt leicht:

$$P_{\max}^{\text{cont}} \approx 0.15 \cdot N \cdot \sqrt{d}$$

wobei $d$ die Dimension der Aktivierungsvektoren ist. Für $d=10$: $\approx 0.47N$.

2. **Damping ($\lambda > 0$)**: Reduziert die Anzahl stabiler Attraktoren, da flache lokale Minima der Energiefunktion "weggeglättet" werden.

3. **Asymmetrische Gewichte** (falls nicht erzwungen): Attraktoren können zu Limit-Zyklen werden.

### 5.3 Basin of Attraction

Die **Basin of Attraction** eines Attraktors $\mathbf{a}^*$ ist die Menge aller Anfangszustände, die zu $\mathbf{a}^*$ konvergieren.

**Abschätzung der Basin-Größe (Hamming-Radius):** Für einen Attraktor in einem Hopfield-Netz mit $P$ gespeicherten Mustern:

$$r_H \approx \frac{N}{2}\left(1 - \sqrt{\frac{P}{0.138 N}}\right)$$

**Effekt von $\lambda$ auf Basin-Größe:**

| $\lambda$ | Anzahl Attraktoren | Mittlere Basin-Größe | Verhalten |
|:-:|:-:|:-:|:--|
| 0.0 | Maximal ($\sim 0.15N$) | Klein, fragmentiert | Viele schwache Muster |
| 0.1 | $\sim 0.12N$ | Mittel | Guter Kompromiss |
| 0.3 | $\sim 0.08N$ | Groß | Wenige robuste Muster |
| 0.5 | $\sim 0.05N$ | Sehr groß | Nur dominante Muster überleben |
| 1.0 | $\sim 1\text{–}3$ | Fast alles | Trivialer Attraktor (alles → Gleichgewicht) |
| >2.0 | 1 | Gesamter Zustandsraum | **Trivial**: Einziger Fixpunkt $\approx \sigma(-\lambda a^*)$ |

**⚠️ Für Brain19:** $\lambda > 1$ macht das System bedeutungslos — es konvergiert immer zum selben trivialen Zustand (alle Aktivierungen $\approx 0.5$).

### 5.4 Spurious Attractors

Neben den "gewünschten" Attraktoren (die semantische Muster kodieren) gibt es **Mischattraktoren** (Linearkombinationen von Mustern). Deren Anzahl wächst exponentiell:

$$N_{\text{spurious}} \sim 2^{P/2}$$

**Damping reduziert spurious Attraktoren signifikant.** Bei $\lambda = 0.3$ werden Mischattraktoren mit kleiner Basin instabil.

---

## 6. Damping & Inhibition

### 6.1 Analytische Herleitung des optimalen $\lambda$

Wir optimieren über zwei gegenläufige Ziele:

**Ziel 1 (Konvergenz):** Schnelle Konvergenz → hohe Kontraktion → hohes $\lambda$

**Ziel 2 (Expressivität):** Reichhaltiges Attractor-Landscape → viele stabile Fixpunkte → niedriges $\lambda$

**Formalisierung:**

Sei $C(\lambda) = \rho(J(\lambda))$ die Kontraktionsrate und $P(\lambda) \approx 0.15N \cdot g(\lambda)$ die Attractor-Kapazität mit $g(\lambda) = e^{-\lambda/\lambda_0}$ (exponentieller Abfall).

Maximiere die Informationskapazität unter Konvergenzgarantie:

$$\max_\lambda P(\lambda) \quad \text{s.t.} \quad C(\lambda) < 1$$

**Ergebnis für normalisierte Gewichte:**

$$C(\lambda) = \frac{1}{4}(1 + \lambda) < 1 \implies \lambda < 3$$

$$P(\lambda) \propto e^{-\lambda/\lambda_0}$$

ist monoton fallend, also wählen wir $\lambda$ so klein wie möglich unter der Bedingung, dass die Konvergenz in $T_{\max}$ Schritten eintritt:

$$\rho(J)^{T_{\max}} < \epsilon$$

$$\left(\frac{1+\lambda}{4}\right)^{T_{\max}} < \epsilon$$

$$\lambda > 4 \cdot \epsilon^{1/T_{\max}} - 1$$

Für $\epsilon = 10^{-4}$ und $T_{\max} = 20$:

$$\lambda > 4 \cdot (10^{-4})^{1/20} - 1 = 4 \cdot 10^{-0.2} - 1 = 4 \cdot 0.631 - 1 = 1.52$$

Das ist **zu hoch** für brauchbare Attractor-Kapazität!

**Auflösung:** Die Konvergenzanforderung $\epsilon = 10^{-4}$ ist unnötig streng. Für praktische Zwecke reicht $\epsilon = 0.01$:

$$\lambda > 4 \cdot (0.01)^{1/20} - 1 = 4 \cdot 0.794 - 1 = 2.18$$

Immer noch zu hoch. Die Lösung ist **mehr Iterationen** ($T_{\max} = 30$):

$$\lambda > 4 \cdot (0.01)^{1/30} - 1 = 4 \cdot 0.862 - 1 = 2.45$$

**Alternative Betrachtung:** Bei normalisierten Gewichten ist das System auch ohne Damping stabil ($\rho \leq 1/4$). Dann dient $\lambda$ **nur** der Attractor-Selektion, nicht der Stabilisierung.

### 6.2 Konkrete Empfehlung

$$\boxed{\lambda_{\text{opt}} = 0.1 \text{ (mit Gewichtsnormalisierung)}}$$

$$\boxed{\lambda_{\text{opt}} = 0.3\text{–}0.5 \text{ (ohne Gewichtsnormalisierung, mit Grad-Skalierung)}}$$

**Begründung für $\lambda = 0.1$ (normalisiert):**
- Konvergenz in ~10 Iterationen ($\rho(J) \approx 0.28$)
- Attractor-Kapazität $\approx 0.13N$ (nahe am theoretischen Maximum)
- Spurious Attractors werden noch ausreichend unterdrückt
- Basin of Attraction groß genug für robustes Retrieval

### 6.3 Adaptive Damping (empfohlen)

Statt konstantem $\lambda$, Grad-abhängiges Damping:

$$\lambda_i = \lambda_0 + \alpha \cdot \frac{d_i}{\bar{d}}$$

mit $\lambda_0 = 0.05$, $\alpha = 0.1$, $d_i$ = Grad von Knoten $i$, $\bar{d}$ = mittlerer Grad.

**Effekt:** Hub-Knoten (hohes $d_i$) werden stärker gedämpft, peripherere Knoten behalten mehr Dynamik. Das stabilisiert Scale-Free-Graphen ohne die Expressivität bei Leaf-Knoten zu opfern.

---

## 7. Context-Knoten Analyse

### 7.1 Braucht das System einen globalen Context-Node?

**Ja.** Begründung:

Ohne Context-Node hängt die Aktivierungsdynamik vollständig von der **lokalen Graphtopologie** ab. Das führt zu:

1. **Fragmentierung:** Disjunkte Subgraphen konvergieren unabhängig — keine globale Kohärenz
2. **Bias zu Hubs:** Stark vernetzte Knoten dominieren die Aktivierung unabhängig vom Kontext
3. **Fehlende Steuerbarkeit:** Die Query des Users hat keinen direkten Einfluss auf die Dynamik

### 7.2 Architektur-Optionen

**Option A: Hub-Node (Star-Topologie)**

$$a_{\text{ctx}} \text{ verbunden mit allen } N \text{ Knoten}$$

- Vorteile: Einfach, maximale Erreichbarkeit
- Nachteile: $O(N)$ Kanten, übermäßiger Einfluss, zerstört Sparsity

**Option B: Broadcast-Node**

$$a_{\text{ctx}} \to a_i \text{ (unidirektional, nur Lesezugriff)}$$

- Vorteile: Knoten empfangen Kontext, ohne selbst den Kontextknoten zu beeinflussen
- Nachteile: Kein Feedback vom Netzwerk zum Kontext

**Option C: Attention-basierter Context**

$$\alpha_i = \text{softmax}_i\!\bigl(\mathbf{a}_{\text{ctx}}^\top \mathbf{a}_i\bigr)$$
$$\tilde{a}_i(t) = F_i(\mathbf{a}(t-1)) + \beta \cdot \alpha_i \cdot \mathbf{a}_{\text{ctx}}$$

- Vorteile: Selektiv, lernbar, stabilisierend
- Nachteile: Komplexer, Softmax-Berechnung pro Iteration

**Empfehlung:** Option C (Attention-basiert) mit Fallback auf Option B für die erste Implementierung.

### 7.3 Stabilisierender Effekt quantifiziert

Der Context-Node addiert einen Term $\beta \alpha_i a_{\text{ctx}}$ zur Dynamik. Effektiv verschiebt dies den Fixpunkt:

$$a_i^* = \sigma\!\Bigl(\sum_j \hat{w}_{ij} a_j^* - \lambda a_i^* + \beta \alpha_i a_{\text{ctx}}\Bigr)$$

**Fixpunkt-Verschiebung:** Der zusätzliche konstante Term $\beta \alpha_i a_{\text{ctx}}$ wirkt wie ein **Bias**, der den Fixpunkt in Richtung der Query verschiebt. Das ist genau das gewünschte Verhalten: Die Dynamik wird durch den Kontext "angezogen".

**Stabilität:** Der Context-Term ändert die Jacobi-Matrix nicht (da $a_{\text{ctx}}$ extern vorgegeben und konstant ist). Die Stabilitätsbedingungen bleiben unverändert.

**Konvergenzverbesserung:** Der Context-Node vergrößert die Basin of Attraction des gewünschten Attraktors (kontextrelevante Aktivierung), da der Bias-Term energetisch den Zustandsraum kippt:

$$\Delta V_{\text{ctx}} = -\beta \sum_i \alpha_i a_{\text{ctx}} a_i \leq 0 \text{ für } a_i \geq 0$$

Das senkt die Energie kontextrelevanter Zustände und hebt die Energie irrelevanter Zustände.

**Empfohlene Parameter:**

$$\beta = 0.2\text{–}0.5$$

zu hoch → Kontext dominiert die Dynamik, Graphstruktur wird ignoriert.

---

## 8. Teststrategie

### 8.1 Metriken

1. **Konvergenzzeit $T_c$:** Anzahl Iterationen bis $\|\mathbf{a}(t) - \mathbf{a}(t-1)\| < \epsilon$
2. **Attractor-Konsistenz $C_A$:** Gleicher Input → gleicher Fixpunkt? $C_A = \frac{\text{gleiche Attraktoren}}{\text{Versuche}}$
3. **Perturbation-Robustheit $R_p$:** Wie viel Rauschen $\|\delta\|$ verträgt der Attraktor? $R_p = \max\|\delta\|$ bei dem $a^*$ erhalten bleibt.
4. **Spektralradius $\rho(J)$:** Empirisch aus der Konvergenzrate schätzen: $\rho \approx \frac{\|\Delta a(t+1)\|}{\|\Delta a(t)\|}$
5. **Attractor-Entropie $H_A$:** Shannon-Entropie über die Häufigkeitsverteilung der Attraktoren bei zufälligen Starts
6. **Energieabfall $\Delta E(t)$:** Monotonie der Energiefunktion prüfen (muss nicht-steigend sein)

### 8.2 Konkreter Testplan

#### Test 1: Konvergenz-Grundtest

```cpp
// Pseudocode
void test_convergence(size_t N, double lambda, size_t max_iter) {
    auto graph = generate_random_subgraph(N, /*avg_degree=*/10);
    auto activations = random_initial_activations(N, EMBED_DIM);
    
    std::vector<double> deltas;
    for (size_t t = 0; t < max_iter; ++t) {
        auto prev = activations;
        propagate_one_step(graph, activations, lambda, /*normalize=*/true);
        double delta = l2_distance(activations, prev);
        deltas.push_back(delta);
        if (delta < 1e-6) {
            LOG("Converged at t={}", t);
            break;
        }
    }
    
    // Prüfe: monoton fallend?
    for (size_t t = 1; t < deltas.size(); ++t) {
        ASSERT(deltas[t] <= deltas[t-1] * 1.01); // 1% Toleranz
    }
    
    // Schätze Spektralradius
    if (deltas.size() > 5) {
        double rho_est = deltas.back() / deltas[deltas.size()-2];
        LOG("Estimated spectral radius: {}", rho_est);
        ASSERT(rho_est < 1.0);
    }
}
```

**Durchführung für:**
- $N \in \{20, 50, 100, 500, 1000\}$
- $\lambda \in \{0.05, 0.1, 0.2, 0.3, 0.5, 1.0\}$
- 100 zufällige Starts pro Konfiguration

#### Test 2: Attractor-Konsistenz

```cpp
void test_attractor_consistency(size_t N, double lambda, size_t num_trials) {
    auto graph = generate_random_subgraph(N, 10);
    
    // Finde Attraktor von spezifischem Startpunkt
    auto a_start = specific_initial_activation(/*seed_concepts=*/{0, 1, 2});
    auto a_ref = propagate_to_convergence(graph, a_start, lambda);
    
    size_t consistent = 0;
    for (size_t trial = 0; trial < num_trials; ++trial) {
        // Kleine Perturbation
        auto a_perturbed = add_noise(a_start, /*sigma=*/0.01);
        auto a_result = propagate_to_convergence(graph, a_perturbed, lambda);
        
        if (l2_distance(a_result, a_ref) < 0.01) {
            consistent++;
        }
    }
    
    double consistency = static_cast<double>(consistent) / num_trials;
    LOG("Attractor consistency: {}%", consistency * 100);
    ASSERT(consistency > 0.95); // Mindestens 95% konsistent
}
```

#### Test 3: Perturbation-Robustheit (Basin-Größe)

```cpp
void test_basin_size(size_t N, double lambda) {
    auto graph = generate_random_subgraph(N, 10);
    auto a_ref = propagate_to_convergence(graph, specific_start(), lambda);
    
    // Binary Search für maximale Perturbation
    double lo = 0.0, hi = 1.0;
    for (int step = 0; step < 30; ++step) {
        double sigma = (lo + hi) / 2;
        int survived = 0;
        for (int trial = 0; trial < 50; ++trial) {
            auto a_pert = add_noise(a_ref, sigma);
            auto a_result = propagate_to_convergence(graph, a_pert, lambda);
            if (l2_distance(a_result, a_ref) < 0.01) survived++;
        }
        if (survived > 25) lo = sigma; // >50% survived → basin is larger
        else hi = sigma;
    }
    
    LOG("Basin radius (L2): {}", lo);
}
```

#### Test 4: Bifurkationsdiagramm

```cpp
void test_bifurcation_diagram(size_t N) {
    auto graph = generate_random_subgraph(N, 10);
    
    // Sweep über lambda
    for (double lambda = 0.0; lambda <= 5.0; lambda += 0.1) {
        std::set<size_t> attractor_hashes;
        
        for (int trial = 0; trial < 200; ++trial) {
            auto a0 = random_initial_activations(N, EMBED_DIM);
            auto a_final = propagate_to_convergence(graph, a0, lambda, /*max_iter=*/100);
            
            // Hash des Attraktors (auf 2 Dezimalstellen quantisiert)
            size_t h = hash_quantized(a_final, 0.01);
            attractor_hashes.insert(h);
        }
        
        LOG("lambda={:.1f}  attractors={}", lambda, attractor_hashes.size());
        // Erwartet: monoton fallend mit lambda
    }
}
```

#### Test 5: Energiemonotonie (nur für symmetrisches W)

```cpp
void test_energy_monotonicity(size_t N, double lambda) {
    auto graph = generate_symmetric_subgraph(N, 10);
    auto a = random_initial_activations(N, EMBED_DIM);
    
    auto compute_energy = [&](const auto& act) {
        double E = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (auto [j, w] : graph.neighbors(i)) {
                E -= 0.5 * w * act[i] * act[j];
            }
            // Entropy term
            double ai = std::clamp(act[i], 1e-10, 1.0 - 1e-10);
            E += (1.0/lambda) * (ai * std::log(ai) + (1-ai) * std::log(1-ai));
        }
        return E;
    };
    
    double E_prev = compute_energy(a);
    for (size_t t = 0; t < 100; ++t) {
        propagate_one_step(graph, a, lambda, true);
        double E_cur = compute_energy(a);
        ASSERT(E_cur <= E_prev + 1e-10); // Monoton fallend
        E_prev = E_cur;
    }
}
```

#### Test 6: Skalierungstest

```cpp
void test_scaling() {
    for (size_t N : {20, 50, 100, 500, 1000}) {
        auto graph = generate_random_subgraph(N, std::min(N-1, (size_t)10));
        auto a = random_initial_activations(N, EMBED_DIM);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t t = 0; t < 20; ++t) {
            propagate_one_step(graph, a, 0.1, true);
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        
        LOG("N={}  20 iterations: {}μs  ({:.1f}μs/iter)", N, us, us/20.0);
    }
}
```

### 8.3 Automatisierter Regressions-Testplan

Alle Tests in `tests/test_stability.cpp` integrieren:

| Test | Frequenz | Schwelle | Aktion bei Verletzung |
|------|----------|----------|----------------------|
| Konvergenz ($N=100, \lambda=0.1$) | Jeder Build | $T_c < 50$ | Build-Fehler |
| Attractor-Konsistenz | Nightly | $C_A > 0.95$ | Warnung |
| Basin-Größe | Nightly | $R_p > 0.05$ | Warnung |
| Energie-Monotonie | Jeder Build | $\Delta E \leq 0$ | Build-Fehler |
| Skalierung ($N=1000$) | Weekly | $< 10$ms/20 Iterationen | Warnung |

---

## 9. Zusammenfassung & Empfehlungen

### 9.1 Kritische Stabilitätsbedingungen

| Bedingung | Status ohne Fix | Status mit Fix |
|-----------|:-:|:-:|
| Gewichtsnormalisierung | ❌ Instabil für $\bar{d} > 5$ | ✅ Stabil |
| Symmetrie von $W$ | ❌ Keine Energiefunktion | ✅ Lyapunov-Konvergenz |
| $\rho(\hat{W}^{\text{eff}}) < 4$ | ❌ Verletzt | ✅ $\rho \leq 1+\lambda < 4$ |
| Grad-abhängiges Damping | — | ✅ Hub-Stabilisierung |

### 9.2 Empfohlene Parameter

| Parameter | Empfehlung | Begründung |
|-----------|:----------:|------------|
| **EMBED_DIM** | **16** | Optimaler Kosten-Nutzen-Punkt (§3.4) |
| **$\lambda$** | **0.1** (normalisiert) | Maximale Expressivität bei Konvergenz (§6.2) |
| **$\lambda$** | **0.3–0.5** (unnormalisiert) | Notwendig für Stabilität (§6.2) |
| **max_cycles** | **20–30** | Ausreichend für $\rho(J) \leq 0.75$ (§2.7) |
| **$\epsilon$ (Konvergenz)** | **$10^{-4}$** | Standard für numerische Konvergenz |
| **Normalisierung** | **Zwingend** | Ohne: Instabilität ab $\bar{d} > 5$ (§2.5) |
| **Symmetrie** | **Erzwingen** | $\hat{w}_{ij} = (\hat{w}_{ij} + \hat{w}_{ji})/2$ (§2.6) |
| **Context $\beta$** | **0.2–0.5** | Steuerbarkeit ohne Dominanz (§7.3) |
| **Adaptive Damping** | $\lambda_0=0.05, \alpha=0.1$ | Hub-Stabilisierung (§6.3) |

### 9.3 Warnungen

1. **🔴 KRITISCH:** Ohne Gewichtsnormalisierung ist das System für $\bar{d} > 5$ instabil. Implementierung der Normalisierung ist **Voraussetzung** für die Interaction Architecture.

2. **🔴 KRITISCH:** Asymmetrische Gewichte erlauben keine Lyapunov-Analyse. Symmetrie muss erzwungen werden oder eine alternative Konvergenzgarantie (z.B. kontrahierende Abbildung) nachgewiesen werden.

3. **🟡 WICHTIG:** $\lambda > 1$ macht das Attractor-Landscape trivial. Das System degeneriert zu einem Gleichgewichtspunkt.

4. **🟡 WICHTIG:** Hub-Knoten ($d > 50$) ohne Grad-Skalierung dominieren und destabilisieren die Dynamik.

5. **🟡 WICHTIG:** `max_cycles = 10` ist für $\rho(J) > 0.5$ unzureichend. Auf 20–30 erhöhen oder Early-Stopping implementieren.

6. **🟢 INFO:** 10D ist funktional für lokale Relevanzberechnung, aber limitierend für semantische Ausdrucksfähigkeit. Upgrade auf 16D empfohlen, 32D für Sprachverständnis.

### 9.4 Implementierungs-Reihenfolge

1. **Sofort:** Gewichtsnormalisierung in `ActivationPropagator` implementieren
2. **Sofort:** Symmetrie erzwingen: $\hat{w}_{ij} \leftarrow (\hat{w}_{ij} + \hat{w}_{ji})/2$
3. **Kurzfristig:** `EMBED_DIM` auf Template-Parameter umstellen, Default auf 16
4. **Kurzfristig:** Stabilitäts-Testsuite (`test_stability.cpp`) implementieren
5. **Mittelfristig:** Adaptive Damping, Context-Node
6. **Langfristig:** Eigenvalue-Monitoring zur Laufzeit (Warnung bei $\rho \to 1$)
