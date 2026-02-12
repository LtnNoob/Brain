# Brain19 Domain Auto-LLM: Theoretische Analyse

## Vorbemerkung

Dieses Dokument ist eine systematische theoretische Analyse, bevor eine einzige Zeile Code geschrieben wird. Brain19 ist eine sorgfältig konstruierte kognitive Architektur mit harten epistemischen Invarianten. Jede Erweiterung, die diese Invarianten auch nur subtil unterminiert, ist schlimmer als keine Erweiterung.

---

## 1. Philosophische Fundierung: Was ist ein "Domain-spezifisches Mini-LLM"?

### 1.1 Das Grundproblem

Brain19 hat eine klare epistemische Ontologie: Jedes Wissensstück trägt einen expliziten Wahrheitsstatus (FACT, THEORY, HYPOTHESIS, SPECULATION) mit quantifiziertem Vertrauen. Das ist das Fundament. Es ist elegant, weil es **epistemische Demut** erzwingt — nichts wird stillschweigend als wahr angenommen.

Ein LLM hingegen ist das genaue Gegenteil: Es ist ein stochastischer Papagei (Bender et al., 2021), der statistische Muster über Trainingskorpora interpoliert. Es hat **kein Konzept von Wahrheit**. Es kann nicht zwischen Fakt und Halluzination unterscheiden. Es hat keine epistemische Integrität.

**Die zentrale Spannung:**
Wie integriert man ein System ohne Wahrheitsbegriff in eine Architektur, deren Existenzberechtigung der Wahrheitsbegriff ist?

### 1.2 Die Auflösung: Kreativität vs. Epistemik

Die Antwort liegt in einer sauberen funktionalen Trennung:

- **Epistemisches System** (LTM + Epistemic Core): *Was ist wahr?* Autoritativ. Einzige Quelle für Wahrheits-Claims.
- **LLM-Schicht** (Understanding Layer): *Was könnte zusammenhängen? Was übersehen wir?* Kreativ. Generiert Kandidaten, behauptet nichts.

Das ist analog zum menschlichen Gehirn:
- **System 1** (schnell, assoziativ, kreativ) ≈ LLM-Schicht
- **System 2** (langsam, logisch, evaluierend) ≈ Epistemisches System

Kahneman (2011) beschreibt genau diese Arbeitsteilung. System 1 generiert Intuitions-Kandidaten, System 2 evaluiert sie kritisch. **Brain19 braucht die gleiche klare Trennung.**

### 1.3 Was "Domain-spezifisch" wirklich bedeutet

Felix' Vision `auto:motor:getriebe:kann_fahren → spezialisiertes Mini-LLM` muss präzise verstanden werden. Es geht **nicht** darum, dass das LLM "weiß", dass ein Motor zum Auto gehört — das weiß der Knowledge Graph. Es geht darum, dass das LLM:

1. **Domain-spezifische Assoziationsmuster** generieren kann
2. **Relevante Analogien** innerhalb der Domäne erkennt
3. **Kreative Hypothesen** vorschlägt, die domänenspezifisch sinnvoll sind
4. **Konzeptuelle Lücken** im Knowledge Graph identifiziert

Ein "Automotive-LLM" soll nicht wissen, dass ein Getriebe Zahnräder hat. Es soll erkennen können: "Wenn der Knowledge Graph `Motor → ENABLES → Getriebe` und `Getriebe → ENABLES → Rad` hat, dann *könnte* es ein Konzept `Antriebsstrang` geben, das diese Kette abstrahiert." — und das als HYPOTHESIS vorschlagen.

---

## 2. Architektur-Implikationen

### 2.1 Die aktuelle Architektur (Ist-Zustand)

```
Input → STM → CognitiveDynamics (Spreading Activation, Salience)
                     ↓
              UnderstandingLayer
                ├── MiniLLM[] (generisch)
                └── Proposals (ALLE HYPOTHESIS)
                     ↓
              BrainController
                     ↓
              EpistemicCore → Akzeptanz/Ablehnung
                     ↓
              LTM (mit epistemischer Bewertung)
```

**Was gut ist:**
- Saubere Trennung: LLMs → Proposals → Epistemische Bewertung
- HYPOTHESIS-Invariante ist compile-time erzwungen
- READ-ONLY LTM-Zugriff für LLMs
- UnderstandingLayer ist explizit "Vorschlagsgenerator, nicht Entscheider"

**Was fehlt:**
- LLMs sind generisch — sie haben keinen Domain-Kontext
- Keine Priorisierung: Jedes LLM sieht jede Query gleich
- Kein Mechanismus für domain-spezifische kreative Inferenz

### 2.2 Der naive Ansatz (und warum er gefährlich ist)

Der offensichtliche Ansatz: "Packe Domain-Wissen in den System-Prompt des LLMs."

**Problem 1: Epistemische Laundering**
Wenn wir dem LLM sagen "Du bist Experte für Automotive, hier ist was du weißt: Motor hat Zylinder, Getriebe hat Zahnräder...", dann haben wir **epistemisch bewertetes Wissen aus der LTM** in den **statistischen Kontext eines LLMs** überführt. Das LLM vermischt dieses Wissen mit seinem Trainingskorpus und gibt etwas zurück, das **epistemisch unklar** ist: Ist das Ergebnis aus dem KG-Kontext abgeleitet? Oder aus dem Trainingskorpus halluziniert?

Das ist **epistemisches Laundering** — wie Geldwäsche, aber für Wissens-Provenienz. Brain19's ganzer Wert liegt darin, dass die Herkunft und Verlässlichkeit jedes Wissensstücks nachvollziehbar ist. Ein LLM zerstört diese Nachvollziehbarkeit.

**Problem 2: Scheinbare Spezialisierung**
Ein 3B-Parameter-Modell mit einem Automotive-System-Prompt ist kein "Automotive-Experte". Es ist dasselbe generische Modell mit Prompt-Conditioning. Die "Spezialisierung" ist eine Illusion — das Modell hat keine tiefere Repräsentation von Automotive-Mechanik, nur eine Tendenz, automotive-klingende Wörter zu generieren.

**Problem 3: Skalierungsproblem**
"Ein LLM pro Domain" klingt elegant, ist aber impraktikabel. Bei 50 Domains brauchen wir 50 LLM-Instanzen in GPU-Speicher? Oder laden wir sie on-demand (dann ist die "Spezialisierung" nur ein System-Prompt-Swap)?

### 2.3 Der fundierte Ansatz: LLMs als kreative Indexer

Statt LLMs als "Domain-Experten" zu missbrauchen, sollten wir sie als das einsetzen, was sie tatsächlich gut können: **assoziative Muster erkennen und sprachlich ausdrücken**.

Die richtige Rolle eines Domain-LLMs in Brain19:

```
Domain-LLM ≠ "Experte der weiß"
Domain-LLM = "Kreativer Denker der fragt"
```

Konkret:
- **INPUT:** Strukturierte Information aus dem KG (Relations, Concept-Labels, Hierarchien)
- **AUFGABE:** Nicht "Was weißt du über Motoren?" sondern "Gegeben diese Struktur, welche Fragen fehlen? Welche Muster erkennst du? Welche Analogien zu anderen Domänen?"
- **OUTPUT:** Strukturierte Proposals die der KG-Struktur folgen (nicht Freitext)

### 2.4 Revidierte Architektur (Soll-Zustand)

```
LTM (Knowledge Graph)
  ↓
DomainClusteringEngine
  ↓  (erkennt zusammengehörige Konzept-Blöcke)
DomainCluster[]
  ↓
  ╔══════════════════════════════════════════════════╗
  ║  DOMAIN CREATIVE ENGINE (pro qualifiziertem     ║
  ║  Cluster)                                       ║
  ║                                                  ║
  ║  Input: KG-Struktur (NUR Topologie + Labels)    ║
  ║  Tool: LLM                                       ║
  ║  Output: Strukturierte Proposals                 ║
  ║                                                  ║
  ║  INVARIANT:                                      ║
  ║  - Kein Zugriff auf epistemische Bewertungen     ║
  ║  - Kein Zugriff auf Trust-Werte                  ║
  ║  - Output ist IMMER HYPOTHESIS                   ║
  ║  - LLM sieht NIE "dies ist FACT"                ║
  ╚══════════════════════════════════════════════════╝
  ↓
Proposal[]
  ↓
EpistemicCore (evaluiert, akzeptiert/verwirft)
  ↓
LTM (wenn akzeptiert, mit eigener epistemischer Bewertung)
```

**Kritische Design-Entscheidung:** Das Domain-LLM sieht **KEINE epistemischen Bewertungen**. Es sieht nur die Graphstruktur (Knoten, Kanten, Labels). Warum?

Wenn das LLM weiß, dass "Motor ist FACT mit Trust 0.95", wird es seine Outputs daran ausrichten — es wird weniger kreativ, weil es "Fakten" nicht in Frage stellt. Aber genau das Infragestellen ist der Wert des kreativen Systems! Ein kreatives System, das nur bestätigt, ist wertlos.

---

## 3. Risiken für epistemische Integrität

### 3.1 Risk: Epistemische Kontamination

**Szenario:** LLM generiert Proposal "Motor braucht Öl für Schmierung". Das klingt vernünftig und wird vom Epistemic Core als plausible HYPOTHESIS akzeptiert. Aber die Information stammt aus dem LLM-Trainingskorpus, nicht aus einer verifizierten Quelle. Brain19 hat jetzt eine HYPOTHESIS, die wie Wissen aussieht, aber statistisches Rauschen sein könnte.

**Mitigation:** Strenge Provenienz-Markierung. Jeder Proposal, der von einem LLM stammt, muss **permanent** als `source: llm-generated` markiert sein. Diese Markierung darf NIEMALS entfernt werden, auch nicht bei Promotion zu THEORY oder FACT. Der Epistemic Core muss LLM-generierte Proposals **strenger** bewerten (höhere Evidenz-Anforderungen für Promotion).

### 3.2 Risk: Kreativitäts-Monokultur

**Szenario:** Alle Domain-LLMs nutzen dasselbe Base-Model (z.B. Llama 3.2). Damit haben alle dieselben Biases, dieselben blinden Flecken, dieselbe Art zu "denken". Brain19's Kreativität wird durch die Trainingskorpus-Distribution des einen Modells begrenzt.

**Mitigation:** 
- Verschiedene Modelle für verschiedene kreative Aufgaben (ein Modell für Analogien, eins für Widersprüche, eins für Lücken)
- Strukturierte Prompts statt Freitext (reduziert Modell-Bias)
- Periodische Model-Rotation

### 3.3 Risk: Domain-Siloing

**Szenario:** Domain-LLMs werden so spezialisiert, dass cross-domain Insights verloren gehen. Das Automotive-LLM sieht nie Biologie-Konzepte und entdeckt daher nicht die Analogie "Blutkreislauf ≈ Kühlsystem".

**Mitigation:** Expliziter Cross-Domain-Mechanismus. Die kreativsten Insights entstehen an Domänengrenzen, nicht innerhalb von Domänen. Design muss Cross-Domain-Queries priorisieren.

### 3.4 Risk: Computational Overhead ohne Mehrwert

**Szenario:** 20 Domain-LLMs laufen, verbrauchen GPU-Zyklen, generieren Proposals — aber 95% werden vom Epistemic Core verworfen. Der Overhead steht in keinem Verhältnis zum Nutzen.

**Mitigation:** 
- Spezialisierungs-Schwelle (nicht jeder Cluster bekommt ein LLM)
- Quality-Tracking mit Eviction
- Lazy Evaluation: LLM wird nur ausgelöst, wenn CognitiveDynamics einen relevanten Fokus hat
- **Wichtig:** "Kein LLM" muss immer ein gültiger Zustand sein

### 3.5 Risk: Zirkuläre Verstärkung

**Szenario:** LLM generiert Hypothesis H. H wird in KG aufgenommen. Beim nächsten Durchlauf sieht das LLM den KG mit H drin und generiert darauf aufbauende Hypotheses. Über mehrere Zyklen entsteht ein ganzer Ast im KG, der auf einer einzelnen LLM-Halluzination basiert.

**Mitigation:** 
- LLM-generierte Concepts sollten dem LLM im nächsten Zyklus NICHT als Input dienen (oder nur mit expliziter Markierung "dies ist LLM-generiert")
- Maximale Kettentiefe für LLM-basierte Inferenz
- Periodische "Realitätschecks" durch externe Quellen

---

## 4. Saubere Trennung: LLM-Kreativität vs. Epistemisches Wissen

### 4.1 Architektur-Prinzipien

**Prinzip 1: LLMs sind Frage-Generatoren, keine Antwort-Generatoren.**
Das LLM sagt nicht "Motor braucht Öl". Es sagt "Im Konzept-Cluster [Motor, Getriebe, Rad] fehlt möglicherweise ein Konzept für Wartung/Schmierung — INVESTIGATE?"

**Prinzip 2: Epistemische Blindheit des LLMs.**
Das LLM sieht die Graph-Topologie, aber NICHT die epistemischen Bewertungen. Es weiß nicht, was "sicher" ist und was nicht. Das verhindert Bestätigungs-Bias.

**Prinzip 3: Provenienz ist unverlierbar.**
Jedes LLM-generierte Proposal trägt permanent seinen Ursprung. Auch nach Promotion. Auch nach Jahren.

**Prinzip 4: Das epistemische System hat Veto-Recht.**
Kein LLM-Output gelangt ohne epistemische Bewertung in den KG. Niemals. Nicht "meistens". Niemals.

**Prinzip 5: Kreativität ist bounded.**
LLMs haben einen Kreativitäts-Budget pro Zyklus. Nicht unbegrenztes Brainstorming, sondern gezielte, fokussierte kreative Inferenz, gesteuert durch CognitiveDynamics' Salience.

### 4.2 Proposal-Taxonomie

Nicht alle LLM-Proposals sind gleich. Differenzierte Typen mit unterschiedlichen epistemischen Anforderungen:

| Proposal-Typ | Beschreibung | Epistemische Hürde |
|---|---|---|
| **GapProposal** | "Hier fehlt ein Konzept/Relation" | Niedrig (Frage, keine Behauptung) |
| **AnalogyProposal** | "Struktur A ähnelt Struktur B" | Mittel (struktureller Vergleich) |
| **HypothesisProposal** | "Konzept X könnte Eigenschaft Y haben" | Hoch (substantielle Behauptung) |
| **ContradictionProposal** | "A und B scheinen inkonsistent" | Mittel (erfordert Prüfung) |
| **RefinementProposal** | "Konzept X könnte präziser formuliert werden" | Niedrig (Meta-Level) |

**Wichtig:** `GapProposal` und `RefinementProposal` existieren in der aktuellen Understanding Layer noch nicht. Sie sind aber die wertvollsten Beiträge eines kreativen Systems — weil sie keine Wahrheits-Claims machen, sondern auf mögliche Verbesserungen des KG hinweisen.

### 4.3 Das "Creative Session" Modell

Statt permanenter Domain-LLMs: **Zeitlich begrenzte kreative Sessions.**

```
1. CognitiveDynamics identifiziert Fokus-Bereich (Salience)
2. DomainClusteringEngine identifiziert relevante Domain
3. → Creative Session wird gestartet:
   a. KG-Subgraph wird extrahiert (NUR Topologie + Labels)
   b. LLM erhält Subgraph + spezifische kreative Aufgabe
   c. LLM generiert N Proposals (bounded)
   d. Session endet
4. Proposals → EpistemicCore → Bewertung
5. LLM-Ressourcen werden freigegeben
```

Vorteile:
- Kein permanenter Ressourcenverbrauch
- Klare Lifecycle-Grenzen
- Natürlicher Schutz gegen Endlos-Loops
- Sessions sind auditierbar

---

## 5. Domain Clustering: Theoretische Überlegungen

### 5.1 Was ist eine "Domain" im Kontext von Brain19?

Eine Domain ist NICHT einfach ein Graph-Cluster. Eine Domain ist ein **semantisch kohärenter Wissensbereich**, in dem:
1. Konzepte untereinander stärker verbunden sind als nach außen (Community-Eigenschaft)
2. Es eine **hierarchische Ordnung** gibt (Generalisierung → Spezialisierung)
3. **Domain-spezifische Inferenzregeln** gelten (nicht alle Relationstypen sind in jeder Domain gleich relevant)
4. Ein **Vokabular** existiert (wiederkehrende Label-Muster)

### 5.2 Wann lohnt sich Spezialisierung?

Nicht jeder Cluster braucht ein eigenes kreatives System. Kriterien:

- **Größe:** Mindestens N Konzepte (zu klein = zu wenig Kontext für sinnvolle Kreativität)
- **Strukturelle Tiefe:** Mindestens M Hierarchie-Ebenen (flache Cluster haben wenig Potential für kreative Inferenz)
- **Aktivitäts-Level:** Der Bereich wird aktiv genutzt (keine Kreativität für toten Wissensbestand)
- **Epistemische Heterogenität:** Mix aus FACT und HYPOTHESIS ist interessanter als nur FACTs (kreatives System arbeitet am Rand des Wissens, nicht im gesicherten Zentrum)
- **Lückendichte:** Cluster mit vielen fehlenden erwartbaren Relationen profitieren mehr von kreativer Exploration

### 5.3 Hierarchische vs. Flache Domains

Felix' Beispiel `auto:motor:getriebe:kann_fahren` impliziert eine **hierarchische** Sicht. Das ist wichtig: Die Hierarchie definiert die kreative Granularität.

```
auto (Top-Level Domain)
├── motor (Sub-Domain)
│   ├── verbrennung
│   └── elektro
├── getriebe (Sub-Domain)
│   ├── automatik
│   └── manuell
└── fahrwerk (Sub-Domain)
    ├── rad
    └── bremse
```

Kreative Sessions auf verschiedenen Ebenen haben verschiedene Qualitäten:
- **Top-Level (auto):** Abstrakte Insights, Cross-Sub-Domain Analogien
- **Sub-Domain (motor):** Spezifischere Hypothesen, Lücken-Erkennung
- **Leaf-Level:** Zu spezifisch für kreative LLM-Inferenz (besser durch domänen-spezifische Logik)

---

## 6. Offene Fragen (vor Implementation zu klären)

1. **Soll das LLM die Graph-Topologie als Text sehen oder als strukturierten Input?**
   - Text: Natürlicher für LLMs, aber lossy
   - Strukturiert (z.B. JSON/triples): Präziser, aber unnatürlich für LLMs
   - Hybrid: Strukturierte Darstellung in natürlichsprachlichem Format

2. **Wie verhindert man, dass das LLM seinen Trainingskorpus als "Domain-Wissen" einspeist?**
   - Prompt-Engineering reicht nicht (LLMs ignorieren Instruktionen)
   - Strukturierte Outputs erzwingen (nur Graphoperationen als Output)
   - Post-Processing: Proposals müssen auf KG-Konzepte referenzieren

3. **Wie interagiert das System mit der KAN-Schicht?**
   - KAN (Kolmogorov-Arnold Networks) in Brain19 sind für Funktionsapproximation
   - Potentielle Synergie: KAN erkennt numerische Muster, LLM erkennt konzeptuelle Muster
   - Braucht saubere Integration

4. **Welche Modellgrößen eignen sich für strukturiertes kreatives Reasoning?**
   - Kleine Modelle (3B) halluzinieren mehr → mehr "Kreativität" aber weniger Qualität
   - Größere Modelle (70B) sind präziser → weniger Überraschungen aber höhere Qualität
   - Vielleicht verschiedene Modellgrößen für verschiedene Proposal-Typen

5. **Wie misst man den Wert eines kreativen Systems?**
   - Acceptance Rate der Proposals ist ein Proxy, aber nicht das volle Bild
   - Ein System das 100% akzeptierte Proposals generiert ist NICHT kreativ — es bestätigt nur Offensichtliches
   - Optimal: Moderate Acceptance Rate mit gelegentlichen hochwertigem Non-Obvious-Insights
   - Braucht eigene Metrik: "Novelty × Epistemic Value"

---

## 7. Empfehlung: Nächste Schritte

Nicht implementieren, sondern:

1. **KG-Analyse:** Verstehe die tatsächliche Struktur und Größe des aktuellen Brain19-KG. Abstrakte Clustering-Algorithmen sind sinnlos ohne Verständnis der Daten.

2. **Prompt-Experimente:** Teste, welche Art von Prompts tatsächlich nützliche kreative Proposals generieren. Manuelle Tests, keine Architektur.

3. **Epistemische Provenienz definieren:** Design das Provenienz-Tracking-System BEVOR die LLM-Integration gebaut wird. Das ist der Kern der epistemischen Sicherheit.

4. **Creative Session Protocol:** Formalisiere den Ablauf einer kreativen Session als Zustandsmaschine. Definiere alle Invarianten.

5. **Dann erst:** Implementation, schrittweise, mit Tests auf epistemische Invarianten bei jedem Schritt.

---

*"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise." — Dijkstra*

Brain19's Abstraktion ist die epistemische Schicht. Jede Erweiterung muss diese Abstraktion respektieren und stärken, nicht verwässern.
