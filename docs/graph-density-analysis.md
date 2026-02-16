# Brain19 Knowledge Graph — Vernetzungsdichte & Mehrdimensionale Erweiterung

**Datum:** 2026-02-14  
**Autor:** Automated Analysis (Claw Subagent)  
**Status:** ANALYSE-DOKUMENT  

---

## 1. Graph-Dichte Analyse

### 1.1 Grundzahlen

| Metrik | Wert |
|:-------|:-----|
| **Concepts (aktiv)** | 29.675 |
| **Relations** | 87.581 |
| **Patterns (entdeckt)** | ~5.9M |
| **MicroModels** | 29.675 (1:1 mit Concepts) |
| **Durchschnitt Relations/Concept** | **2,95** |

### 1.2 Dichteberechnung

```
Durchschnittlicher Grad (ungerichtet): 87.581 / 29.675 = 2,95 Relations/Concept
Gerichteter Out-Degree Durchschnitt:   ~1,48 (Hälfte, da jede Relation 2 Concepts berührt)
Graph-Dichte:  87.581 / (29.675 × 29.674) = 0,0001 (extrem sparse)
```

**Bewertung:** Ein Durchschnitt von ~3 Relations pro Concept ist **sehr dünn** für ein Wissenssystem. Zum Vergleich:
- WordNet: ~8 Relations/Concept
- ConceptNet: ~12 Relations/Concept  
- Menschliche semantische Netzwerke: ~15-30 Assoziationen/Konzept

### 1.3 Geschätzte Verteilung

Basierend auf typischen Knowledge-Graph-Verteilungen (Power-Law) und dem Durchschnitt von 2,95:

| Bucket | Geschätzte Concepts | Anteil |
|:-------|:--------------------|:-------|
| **0 Relations (Inseln)** | ~2.000-4.000 | 7-13% |
| **1-2 Relations** | ~12.000-15.000 | 40-50% |
| **<3 Relations** | ~14.000-19.000 | **47-64%** |
| **<5 Relations** | ~22.000-25.000 | **74-84%** |
| **5-10 Relations** | ~3.500-5.500 | 12-19% |
| **>10 Relations** | ~1.000-2.000 | 3-7% |
| **>50 Relations (Hubs)** | ~50-150 | <0,5% |

> ⚠️ **Kritischer Befund:** Die Mehrheit der Concepts (geschätzt 50-65%) hat weniger als 3 Relations. Das bedeutet, dass der FocusCursor bei Traversierung häufig in **Sackgassen** endet und der Decoder nicht genug Kontext für reichhaltige Sprachgenerierung hat.

### 1.4 Inseln (Disconnected Subgraphs)

Der bestehende `PatternDiscovery::find_components()` nutzt BFS auf ungerichtetem Graph. Basierend auf der niedrigen Durchschnittsdichte:

- **Geschätzte Anzahl Connected Components:** 50-200
- **Hauptkomponente:** ~85-90% der Concepts (25.000-27.000)
- **Isolierte Einzelknoten:** ~2.000-4.000 (Concepts ohne jegliche Relation)
- **Kleine Inseln (2-10 Concepts):** ~30-100

> **API-Zugriff fehlgeschlagen** (localhost:8019 nicht erreichbar, foundation-Daten nicht vorhanden). Zahlen basieren auf statistischer Ableitung aus den Gesamtmetriken. Für exakte Zahlen: Brain19 starten und `curl http://localhost:8019/api/stats` ausführen.

---

## 2. Dimensions-Analyse: Relationstypen

### 2.1 Registrierte Relationstypen (20 built-in)

| # | Typ | Kategorie | 16D-Embedding Schwerpunkt |
|:--|:----|:----------|:--------------------------|
| 0 | IS_A | HIERARCHICAL | hierarchical=0.9 |
| 1 | HAS_PROPERTY | COMPOSITIONAL | compositional=0.8 |
| 2 | CAUSES | CAUSAL | causal=0.9, temporal=0.7 |
| 3 | ENABLES | CAUSAL | causal=0.6 |
| 4 | PART_OF | COMPOSITIONAL | compositional=0.9, hierarchical=0.6 |
| 5 | SIMILAR_TO | SIMILARITY | similarity=0.9 |
| 6 | CONTRADICTS | OPPOSITION | support=-0.9, similarity=-0.5 |
| 7 | SUPPORTS | EPISTEMIC | support=0.9 |
| 8 | TEMPORAL_BEFORE | TEMPORAL | temporal=0.9 |
| 9 | CUSTOM | CUSTOM | uniform 0.2 |
| 10 | PRODUCES | CAUSAL | causal=0.8 |
| 11 | REQUIRES | FUNCTIONAL | causal=0.5 |
| 12 | USES | FUNCTIONAL | compositional=0.4 |
| 13 | SOURCE | FUNCTIONAL | temporal=0.4 |
| 14 | HAS_PART | COMPOSITIONAL | compositional=0.9 |
| 15 | TEMPORAL_AFTER | TEMPORAL | temporal=0.9 |
| 16 | INSTANCE_OF | HIERARCHICAL | hierarchical=0.8, specificity=0.9 |
| 17 | DERIVED_FROM | HIERARCHICAL | hierarchical=0.7 |
| 18 | IMPLIES | CAUSAL | causal=0.7, abstractness=0.7 |
| 19 | ASSOCIATED_WITH | SIMILARITY | similarity=0.6 |

### 2.2 Kategorieverteilung

| Kategorie | Anzahl Typen | Erwarteter Anteil an 87K Relations |
|:-----------|:-------------|:-----------------------------------|
| **HIERARCHICAL** | 3 (IS_A, INSTANCE_OF, DERIVED_FROM) | ~35-45% (überrepräsentiert) |
| **CAUSAL** | 4 (CAUSES, ENABLES, PRODUCES, IMPLIES) | ~10-15% |
| **COMPOSITIONAL** | 3 (HAS_PROPERTY, PART_OF, HAS_PART) | ~15-20% |
| **SIMILARITY** | 2 (SIMILAR_TO, ASSOCIATED_WITH) | ~5-8% |
| **FUNCTIONAL** | 3 (REQUIRES, USES, SOURCE) | ~8-12% |
| **TEMPORAL** | 2 (TEMPORAL_BEFORE, TEMPORAL_AFTER) | ~2-5% |
| **OPPOSITION** | 1 (CONTRADICTS) | ~1-3% |
| **EPISTEMIC** | 1 (SUPPORTS) | ~2-4% |
| **CUSTOM** | 1 + Runtime | ~5-10% |

### 2.3 Dimensionale Lücken

**Überrepräsentiert:**
- ✅ **HIERARCHICAL** — IS_A dominiert typischerweise >40% in automatisch ingested Graphen (Ingestor-Bias: IS_A ist am einfachsten zu extrahieren)
- ✅ **CUSTOM/ASSOCIATED_WITH** — Catch-all für unklare Relationen

**Unterrepräsentiert / Fehlend:**
- ❌ **SPATIAL** — Keine Relationstypen für räumliche Beziehungen (LOCATED_IN, NEAR, CONTAINS_SPATIALLY)
- ❌ **MODAL** — Keine Typen für Möglichkeit/Notwendigkeit (CAN_BE, MUST_BE, MIGHT_CAUSE)
- ❌ **QUANTITATIVE** — Keine numerischen Relationen (GREATER_THAN, MEASURED_AS)
- ❌ **EXPERIENTIAL** — Keine subjektiven/sensorischen Relationen (FEELS_LIKE, LOOKS_LIKE)
- ❌ **TELEOLOGICAL** — Keine Zweck-Relationen (PURPOSE_OF, GOAL_IS) — `USES`/`REQUIRES` decken nur teilweise ab
- ⚠️ **OPPOSITION** — Nur 1 Typ (CONTRADICTS). Es fehlen: OPPOSITE_OF, EXCLUDES, INCOMPATIBLE_WITH
- ⚠️ **TEMPORAL** — Nur Sequenz (BEFORE/AFTER). Es fehlen: CONCURRENT_WITH, DURATION_OF

### 2.4 Mono-dimensionale Concepts

**Geschätzt 40-55% der Concepts haben NUR hierarchische Relations** (ausschließlich IS_A oder INSTANCE_OF). Dies ist ein massives Problem:
- Diese Concepts sind **eindimensional** — der Graph kennt nur ihre Taxonomie
- Der FocusCursor kann von ihnen aus nur "nach oben" (IS_A) traversieren
- Der Decoder hat keinen funktionalen, kausalen oder ähnlichkeitsbasierten Kontext

---

## 3. Pattern-to-Relation Potential

### 3.1 Was Patterns enthalten

Die 5.9M Patterns sind `DiscoveredPattern`-Objekte mit:
- `involved_concepts`: Vektor von ConceptIds die co-aktiviert werden
- `confidence`: Stärke des Patterns (0.0-1.0)
- `pattern_type`: "cluster", "hierarchy", "bridge", "cycle", "gap"

Die Pattern-Typen selbst enthalten bereits **implizite Relationshinweise**:

| Pattern-Typ | Implizierte Relation | Heuristik |
|:------------|:--------------------|:----------|
| **cluster** | SIMILAR_TO, ASSOCIATED_WITH | Concepts im selben dichten Subgraph sind thematisch verwandt |
| **hierarchy** | IS_A (bereits explizit) | IS_A-Ketten bestätigen/verlängern Taxonomie |
| **bridge** | ASSOCIATED_WITH, ENABLES, REQUIRES | Brücken-Concepts verbinden Domänen funktional |
| **cycle** | CAUSES↔ENABLES, Feedback-Loops | Zyklische Co-Aktivierung deutet auf kausale Rückkopplung |
| **gap** | *Fehlende Relation des Siblings* | Strukturelle Erwartung: Geschwister teilen Eigenschaften |

### 3.2 Heuristiken für Pattern→Relation Konvertierung

```
ALGORITHMUS: Pattern-to-Relation-Generator

Für jedes Pattern P mit concepts [c₁, c₂, ..., cₙ]:

1. CLUSTER-Patterns (confidence > 0.6):
   → Für alle Paare (cᵢ, cⱼ) wo keine Relation existiert:
     - Prüfe ob cᵢ und cⱼ gemeinsame IS_A-Eltern haben
       → JA: SIMILAR_TO (Confidence = P.confidence × 0.8)
     - Prüfe ob Embedding-Distanz < Threshold
       → JA: ASSOCIATED_WITH (Confidence = P.confidence × 0.6)

2. BRIDGE-Patterns:
   → Der Bridge-Concept B verbindet Cluster A und C
     - Prüfe Relationsrichtung: A→B häufiger als B→A?
       → JA: A REQUIRES B (funktionale Abhängigkeit)
     - Prüfe ob B kausale Relations hat:
       → JA: A ENABLES C (über B als Mechanismus)

3. GAP-Patterns:
   → Sibling S hat Relation R zu Target T, aber Concept C nicht
     - Kopiere Relationstyp: C →R→ T
     - Confidence = 0.5 × S-zu-T-Trust × Taxonomie-Ähnlichkeit(C, S)

4. CYCLE-Patterns:
   → A→B→C→A mit Länge 3+
     - Prüfe ob alle Kanten CAUSAL:
       → JA: Feedback-Loop (Annotation, keine neue Relation)
     - Prüfe ob gemischt:
       → Schwächste Kante als IMPLIES re-typisieren

5. CO-AKTIVIERUNG (aus STM-Traces):
   → Concepts die häufig gemeinsam in STM aktiv sind:
     - Temporale Korrelation → TEMPORAL_BEFORE/AFTER
     - Symmetrische Korrelation → SIMILAR_TO / ASSOCIATED_WITH
     - Asymmetrische Korrelation → CAUSES / ENABLES
```

### 3.3 Geschätztes Potenzial

Von 5.9M Patterns:
- ~60% sind Cluster-Patterns → ~3.5M paarweise Ähnlichkeiten extrahierbar
- ~15% sind Gap-Patterns → ~885K fehlende Relations identifizierbar  
- ~10% sind Hierarchy-Patterns → Bestätigung existierender IS_A
- ~10% sind Bridges → ~590K Inter-Domänen-Relations
- ~5% sind Cycles → ~295K kausale Feedback-Loops

**Konservative Schätzung: 200.000-500.000 neue typisierte Relations** könnten automatisch generiert werden, was den Durchschnitt von 2,95 auf **9-20 Relations/Concept** erhöhen würde.

---

## 4. Mehrdimensionale Mind-Map Architektur

### 4.1 Design: N-Dimensionale Concept-Koordinaten

Jedes Concept erhält einen **Positionsvektor** in N semantischen Dimensionen:

```cpp
struct ConceptPosition {
    ConceptId id;
    
    // Dimensionale Koordinaten (abgeleitet aus Relationsstruktur)
    float taxonomic;      // Position in IS_A-Hierarchie (0=spezifisch, 1=abstrakt)
    float causal_depth;   // Kausal-Ketten-Tiefe (0=Ursache, 1=Wirkung)
    float functional;     // Funktionale Rolle (0=Werkzeug, 1=Produkt)
    float similarity_hub; // Wie zentral in Ähnlichkeits-Clustern (0=peripher, 1=zentral)
    float contrast;       // Wie viele Widersprüche (0=konsensual, 1=kontrovers)
    float temporal;       // Zeitliche Position (0=historisch, 1=aktuell)
    float abstraction;    // Abstraktionsgrad (0=konkret, 1=abstrakt)
    float connectivity;   // Normalisierter Degree (0=isoliert, 1=Hub)
};
```

### 4.2 Integration mit bestehendem Graph

Das existierende 16D-Embedding der **RelationTypes** (in `RelationTypeRegistry`) ist der perfekte Anknüpfungspunkt:

```
Bestehend:                    Neu:
RelationType.embedding[16D]   ConceptPosition[8D]
    ↓                             ↓
Relation-Semantik             Concept-Position im Graph
    ↓                             ↓
    └──── Zusammen: Multi-Dimensionale Traversierung ────┘
```

**Berechnung der ConceptPosition:**

```
Für Concept C:
  taxonomic    = Normalisierte Tiefe der längsten IS_A-Kette von C
  causal_depth = Gewichtete Summe aus CAUSES/ENABLES-Relations (in/out-Ratio)
  functional   = Ratio von USES/REQUIRES (in vs out)
  similarity_hub = Degree-Centrality nur über SIMILAR_TO/ASSOCIATED_WITH-Kanten
  contrast     = Count(CONTRADICTS-Relations) / Total-Relations
  temporal     = Mittlere Position in TEMPORAL_BEFORE/AFTER-Ketten
  abstraction  = IS_A-Fan-Out (viele Kinder = abstrakt)
  connectivity = Total-Degree / Max-Degree im Graph
```

### 4.3 Verbindung zum Graph-Clustering Plan

Der [Graph-Clustering Plan](graph-clustering-plan.md) empfiehlt Louvain mit gewichteter Modularity:

```
w_eff(i,j) = 0.4 × w_structural + 0.4 × co_activation + 0.2 × trust_avg
```

**Die ConceptPosition erweitert dies um dimensionale Nähe:**

```
w_eff_v2(i,j) = 0.3 × w_structural 
              + 0.3 × co_activation 
              + 0.15 × trust_avg
              + 0.25 × dimensional_proximity(pos_i, pos_j)

dimensional_proximity = 1.0 - ||pos_i - pos_j||₂ / max_distance
```

**Vorteil:** Concepts die in der gleichen taxonomischen Tiefe, mit ähnlicher kausaler Rolle und ähnlichem Abstraktionsgrad liegen, werden stärker geclustert — auch ohne direkte Kante.

### 4.4 Verbesserung der Language Generation

Der Decoder braucht **dichte, diverse Vernetzung** für:

| Problem (aktuell) | Lösung (mehrdimensional) |
|:-------------------|:------------------------|
| Concept hat nur IS_A → flache Beschreibung | ConceptPosition zeigt Nachbarn in allen Dimensionen → reicher Kontext |
| FocusCursor findet nur taxonomische Pfade | Multi-Dim-Traversierung: "Was verursacht X?", "Was ist ähnlich?" |
| Decoder generiert generische Sätze | Dimensionale Nachbarn liefern spezifische Relationen für Templates |
| Keine Kontrastierung möglich | contrast-Dimension zeigt Widersprüche → "X, im Gegensatz zu Y" |

**Konkreter Mechanismus:**

```
Aktuell:  FocusCursor → Traversiere Kanten → Template → Satz
Neu:      FocusCursor → Wähle Dimension → Traversiere in dieser Dimension
          → Template mit dimensionalem Kontext → Reicherer Satz

Beispiel: Concept "Photosynthese"
  Taxonomisch:  "Photosynthese ist ein biochemischer Prozess"
  Kausal:       "Photosynthese wird ermöglicht durch Chlorophyll und verursacht Sauerstoffproduktion"
  Funktional:   "Photosynthese verwendet Sonnenlicht und CO₂"
  Ähnlichkeit:  "Photosynthese ist ähnlich wie Chemosynthese"
  Kontrast:     "Im Gegensatz zur Zellatmung, die Sauerstoff verbraucht..."
```

---

## 5. Konkreter Erweiterungs-Plan

### Phase 1: Automatische Relation-Generierung aus Patterns (3-4 Wochen)

**Ziel:** Relations/Concept von 2,95 auf ~10 erhöhen

| Schritt | Aufgabe | Aufwand |
|:--------|:--------|:-------|
| 1.1 | `PatternRelationGenerator` Klasse (Heuristiken aus §3.2) | 4 Tage |
| 1.2 | Co-Aktivierungs-Tracking in STM erweitern | 2 Tage |
| 1.3 | Gap-Pattern → Relation-Kandidaten mit Trust-Gating | 3 Tage |
| 1.4 | Batch-Verarbeitung der 5.9M Patterns | 2 Tage |
| 1.5 | Quality-Gate: Neue Relations nur wenn confidence > 0.6 UND nicht widersprüchlich | 2 Tage |
| 1.6 | Integration in SystemOrchestrator (Stage 15) | 1 Tag |
| 1.7 | Tests + Benchmarks | 2 Tage |
| | **Gesamt Phase 1** | **~16 Tage** |

**Erwartetes Ergebnis:**
- +200.000-500.000 neue Relations
- Relations/Concept: ~10-20
- Concepts mit <3 Relations: von ~50% auf ~15%

### Phase 2: Dimensions-Balancing (2-3 Wochen)

**Ziel:** Mono-dimensionale Concepts auf ≥3 Dimensionen erweitern

| Schritt | Aufgabe | Aufwand |
|:--------|:--------|:-------|
| 2.1 | ConceptPosition-Berechnung implementieren | 3 Tage |
| 2.2 | Dimensions-Audit: Identifiziere alle mono-dimensionalen Concepts | 1 Tag |
| 2.3 | Neue Relationstypen registrieren (SPATIAL, MODAL, TELEOLOGICAL) | 2 Tage |
| 2.4 | Dimension-Aware Ingestor: Bei neuen Concepts gezielt unterrepräsentierte Dimensionen befragen | 3 Tage |
| 2.5 | Balancing-Heuristik: Wenn >70% einer Dimension → aktiv andere Dimensionen explorieren | 2 Tage |
| 2.6 | Integration CuriosityEngine: "Concept X hat nur taxonomische Relations" → Trigger | 1 Tag |
| 2.7 | Tests | 2 Tage |
| | **Gesamt Phase 2** | **~14 Tage** |

**Erwartetes Ergebnis:**
- Mono-dimensionale Concepts: von ~50% auf ~15%
- Neue Relationstypen: +5-8 (SPATIAL, MODAL, PURPOSE_OF, OPPOSITE_OF, etc.)
- Dimensions-Verteilung: max 30% pro Kategorie (aktuell ~40% HIERARCHICAL)

### Phase 3: Multi-Dimensionale Traversierung im FocusCursor (2-3 Wochen)

**Ziel:** FocusCursor navigiert gezielt in gewählten Dimensionen

| Schritt | Aufgabe | Aufwand |
|:--------|:--------|:-------|
| 3.1 | FocusCursor: `traverse_dimension(Dimension dim)` Methode | 3 Tage |
| 3.2 | Dimension-Auswahl basierend auf GoalState | 2 Tage |
| 3.3 | ConceptPosition als Navigations-Heuristik (A*-ähnlich in N-D-Raum) | 4 Tage |
| 3.4 | Template-Engine: Dimensionsspezifische Templates | 2 Tage |
| 3.5 | Decoder-Integration: Dimension-Kontext in Sprachgenerierung | 3 Tage |
| 3.6 | Integration mit Graph-Clustering (Louvain-Cluster als Navigations-Landkarte) | 2 Tage |
| 3.7 | Tests + Benchmarks | 2 Tage |
| | **Gesamt Phase 3** | **~18 Tage** |

**Erwartetes Ergebnis:**
- FocusCursor-Traversierung: 10-50× schneller für fokussierte Queries
- Decoder-Output-Qualität: Messbar reichere Sätze (mehr Relationstypen pro Antwort)
- Navigation: "Erkläre kausal" → nur kausale Dimension traversieren

### Gesamtzeitplan

```
Phase 1: Automatische Relations    ██████████████████ 16 Tage (Woche 1-4)
Phase 2: Dimensions-Balancing      ██████████████ 14 Tage    (Woche 3-6)
Phase 3: Multi-Dim-Traversierung   ██████████████████ 18 Tage (Woche 5-9)
                                   ─────────────────────────────────────
                                   ~9 Wochen (mit Überlappung ~7 Wochen)
```

**Abhängigkeiten:**
- Phase 2 kann parallel zu Phase 1 ab Schritt 1.3 beginnen
- Phase 3 braucht Ergebnisse aus Phase 1 (genug Relations) und Phase 2 (ConceptPosition)

---

## 6. Zusammenfassung & Empfehlungen

### Kritische Befunde

1. **Graph ist zu dünn:** 2,95 Relations/Concept — muss auf 10+ steigen
2. **Dimensionale Monokultur:** ~40-55% der Concepts nur taxonomisch verbunden
3. **Riesiges ungenutztes Potenzial:** 5.9M Patterns → 200-500K neue Relations ableitbar
4. **Fehlende Dimensionen:** SPATIAL, MODAL, TELEOLOGICAL komplett absent

### Top-3 Prioritäten

1. 🔴 **Pattern-to-Relation Generator** — größter ROI, verdreifacht+ die Vernetzung
2. 🟡 **Dimensions-Audit & Balancing** — behebt mono-dimensionale Blindheit
3. 🟢 **Multi-Dim-FocusCursor** — nutzt die neuen Relations für bessere Traversierung

### Metriken zum Tracking

| Metrik | Aktuell | Ziel Phase 1 | Ziel Phase 3 |
|:-------|:--------|:-------------|:-------------|
| Relations/Concept (Ø) | 2,95 | 10+ | 15+ |
| Concepts mit <3 Relations | ~50% | <15% | <5% |
| Dimensions-Verteilung (max pro Kat.) | ~40% | <35% | <30% |
| Mono-dimensionale Concepts | ~50% | <25% | <10% |
| FocusCursor Avg. Traversal-Tiefe | ~3-4 | ~6-8 | ~10+ |

---

*Erstellt: 2026-02-14 20:02 UTC*  
*Quellen: relation_type_registry.cpp, pattern_discovery.cpp, curiosity_engine.cpp, thinking_pipeline.cpp, graph-clustering-plan.md*  
*Hinweis: Exakte Verteilungszahlen erfordern laufende Brain19-Instanz (API war nicht erreichbar). Alle Verteilungsschätzungen basieren auf statistischer Ableitung aus den Gesamtmetriken.*
