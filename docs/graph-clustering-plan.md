# Brain19 Graph-Clustering Plan — 3-Agenten-Panel

**Datum:** 2026-02-14  
**Status:** ARCHITEKTUR-ENTSCHEIDUNG  
**Kontext:** 29K Concepts, 87K Relations, 5.9M Patterns — flacher Graph → hierarchisches Clustering

---

## Ausgangslage

Der Knowledge Graph ist aktuell **flach**: alle 29K Concepts auf einer Ebene, keine Gruppierung. `PatternDiscovery::find_clusters()` nutzt Connected Components (BFS) — das findet Zusammenhangskomponenten, keine semantischen Cluster. Die 5.9M Patterns aus Pattern Discovery enthalten implizit Co-Aktivierungsdaten, die als Ähnlichkeitsmetrik nutzbar sind.

**Ziel:** Dynamische, hierarchische Cluster wie eine Mindmap — natürliche Themengruppen (Biologie, Physik, Chemie...) die untereinander verlinkt sind und sich mit neuem Wissen reorganisieren.

---

## Agent 1 — Praktiker (Software-Engineer)

### Algorithmus-Wahl: Louvain

| Algorithmus | Komplexität | Hierarchisch | C++ Aufwand | Eignung |
|:------------|:------------|:-------------|:------------|:--------|
| **Louvain** | O(n·log n) | ✅ Dendrogramm | Mittel (~400 LOC) | ⭐⭐⭐⭐⭐ |
| Label Propagation | O(n·k) | ❌ Flat | Gering (~150 LOC) | ⭐⭐⭐ |
| Spectral Clustering | O(n³) | ❌ Flat | Hoch (Eigenwerte) | ⭐ (29K zu groß) |
| Infomap | O(n·log n) | ✅ | Hoch (ext. Dep) | ⭐⭐⭐⭐ |

**Empfehlung: Louvain-Algorithmus.** Gründe:
1. O(n·log n) — bei 29K Nodes ~0.5s auf i5-6600K
2. Produziert natürlich ein Dendrogramm (Multi-Level)
3. Kein externer Dependency nötig — 400 LOC in C++
4. Gut erprobt für Graphen dieser Größe (optimiert für 10K–10M Nodes)

### Performance auf i5-6600K (4 Kerne)

```
Louvain auf 29K Nodes, 87K Edges:
  Phase 1 (Local Moving):  ~200ms (single-threaded genügt)
  Phase 2 (Aggregation):   ~50ms
  3-4 Iterationen:         ~800ms total
  
Parallelisierung: NICHT nötig bei dieser Größe.
Re-Clustering (inkrementell): ~100ms (nur betroffene Nachbarschaft)
```

**Memory:** ~3MB zusätzlich (Cluster-Zuordnung + Dendrogramm)

### Implementierungsplan

#### Neue Dateien

```
backend/clustering/
├── graph_clustering.hpp          # Hauptklasse
├── graph_clustering.cpp          # Louvain + Hierarchie
├── cluster_info.hpp              # Datenstrukturen
├── cluster_cache.hpp             # Cache für schnellen Lookup
├── cluster_cache.cpp
└── test_graph_clustering.cpp     # Tests
```

#### Datenstrukturen

```cpp
// cluster_info.hpp
struct ClusterInfo {
    uint32_t cluster_id;
    uint32_t parent_cluster_id;      // Hierarchie-Ebene darüber
    std::string auto_label;           // "Biologie", "Thermodynamik" etc.
    std::vector<ConceptId> members;
    double modularity_score;          // Qualität des Clusters
    size_t level;                     // 0=Blatt, 1=Mittel, 2=Top
};

struct ClusterHierarchy {
    std::vector<std::vector<ClusterInfo>> levels;  // levels[0]=feinste Ebene
    double total_modularity;
    std::unordered_map<ConceptId, uint32_t> concept_to_cluster;  // level 0
};
```

#### Integration in bestehenden Code

| Komponente | Änderung | Dateien |
|:-----------|:---------|:--------|
| **SystemOrchestrator** | Stage 14.5: ClusterManager nach PatternDiscovery | `system_orchestrator.hpp/.cpp` (+30 LOC) |
| **PatternDiscovery** | `find_clusters()` delegiert an GraphClustering | `pattern_discovery.cpp` (+20 LOC) |
| **FocusCursor** | Cluster-aware Traversierung: bevorzuge Intra-Cluster-Kanten | `focus_cursor.cpp` (+50 LOC) |
| **LongTermMemory** | `get_cluster(ConceptId)` Accessor | `long_term_memory.hpp/.cpp` (+15 LOC) |
| **Periodic Maintenance** | Re-Clustering alle 30min (mit Checkpoint) | `system_orchestrator.cpp` (+10 LOC) |
| **PersistentLTM** | Cluster-Zuordnung serialisieren | `persistent_ltm.cpp` (+40 LOC) |
| **ChatInterface** | "In welchem Bereich liegt X?" beantworten | `chat_interface.cpp` (+20 LOC) |

#### LOC-Schätzung

| Datei | LOC | Komplexität |
|:------|:----|:------------|
| graph_clustering.hpp | 60 | Niedrig |
| graph_clustering.cpp | 350 | Hoch (Louvain-Kern) |
| cluster_info.hpp | 40 | Niedrig |
| cluster_cache.hpp/.cpp | 80 | Mittel |
| test_graph_clustering.cpp | 120 | Mittel |
| Integration (diverse Dateien) | 185 | Mittel |
| **Gesamt** | **~835** | |

#### Louvain-Kernalgorithmus (Pseudocode)

```cpp
ClusterHierarchy GraphClustering::cluster(const LongTermMemory& ltm) {
    // Phase 1: Initialisierung — jeder Node = eigener Cluster
    auto nodes = ltm.get_active_concepts();
    std::unordered_map<ConceptId, uint32_t> assignment;
    for (auto id : nodes) assignment[id] = id;
    
    bool improved = true;
    while (improved) {
        improved = false;
        // Phase 2: Local Moving — jeder Node zum Nachbar-Cluster mit max ΔQ
        for (auto node : nodes) {
            auto best = find_best_community(node, assignment, ltm);
            if (best.delta_q > 1e-6) {
                assignment[node] = best.cluster_id;
                improved = true;
            }
        }
        // Phase 3: Aggregation — Cluster → Super-Nodes
        if (improved) {
            levels_.push_back(current_level);
            aggregate(assignment);
        }
    }
    return build_hierarchy(levels_);
}
```

### Auto-Labeling der Cluster

```cpp
std::string auto_label_cluster(const std::vector<ConceptId>& members, 
                                const LongTermMemory& ltm) {
    // Strategie: Häufigstes IS_A-Target als Label
    std::unordered_map<ConceptId, int> parent_counts;
    for (auto id : members) {
        for (auto& rel : ltm.get_outgoing_relations(id)) {
            if (rel.type == RelationType::IS_A)
                parent_counts[rel.target]++;
        }
    }
    // Häufigstes Parent = Cluster-Label
    auto best = std::max_element(parent_counts.begin(), parent_counts.end(),
        [](auto& a, auto& b) { return a.second < b.second; });
    if (best != parent_counts.end()) {
        auto concept = ltm.retrieve_concept(best->first);
        if (concept) return concept->label;
    }
    return "Cluster_" + std::to_string(members[0]);
}
```

---

## Agent 2 — Wissenschaftler (KI/Graph-Theory)

### Optimaler Algorithmus für Brain19's Graph-Struktur

Brain19's KG ist ein **gerichteter, gewichteter, heterogener Multi-Relational Graph** mit:
- 29K Nodes, 87K Edges → mittlerer Grad ~6 (sparse)
- 20+ Relationstypen mit unterschiedlicher Semantik
- Epistemische Gewichte (Trust 0.05–1.0)
- Zeitliche Dimension (Wissen wird hinzugefügt, selten gelöscht)

**Empfehlung: Hierarchisches Community Detection mit gewichteter Modularity**

#### Warum hierarchisch statt flat?

| Aspekt | Flat Clustering | Hierarchisch | Overlapping |
|:-------|:----------------|:-------------|:------------|
| Mindmap-Analogie | ❌ Keine Zoom-Ebenen | ✅ Natürliche Ebenen | ⚠️ Komplex |
| FocusCursor-Integration | Einmalig auswerten | Cluster-Zoom bei Traversierung | Mehrdeutig |
| Kognitive Plausibilität | Unrealistisch | ✅ Cortex-analog | ✅ Aber teurer |
| Implementierung | Einfach | Mittel (Louvain liefert es gratis) | Aufwändig |

**Hierarchisch ist der klare Gewinner** — Louvain liefert es als Nebenprodukt.

#### Co-Aktivierung aus Patterns als Ähnlichkeitsmetrik

Die 5.9M Patterns enthalten Co-Aktivierungsinformation: Welche Concepts werden häufig zusammen aktiviert?

**Formalisierung:**

```
Sei P = {p₁, ..., p_5.9M} die Pattern-Menge.
Für Concepts i, j definiere:

  co_activation(i, j) = |{p ∈ P : i ∈ p ∧ j ∈ p}| / |{p ∈ P : i ∈ p ∨ j ∈ p}|
  
  (Jaccard-Koeffizient der Pattern-Zugehörigkeit)
```

Diese Co-Aktivierung kann als **zusätzliche Kantengewichtung** in Louvain einfließen:

```
w_eff(i,j) = α · w_structural(i,j) + β · co_activation(i,j) + γ · trust(i,j)

Empfohlene Gewichte: α=0.4, β=0.4, γ=0.2
```

**Vorteil:** Cluster basieren nicht nur auf expliziten Relationen, sondern auch auf gelerntem Nutzungsverhalten. Concepts die oft zusammen gedacht werden, landen im selben Cluster — auch wenn keine direkte Relation existiert.

#### Mathematische Fundierung: Modularity

Louvain maximiert die **Modularity Q**:

```
Q = (1/2m) · Σᵢⱼ [Aᵢⱼ - (kᵢ·kⱼ)/(2m)] · δ(cᵢ, cⱼ)

Aᵢⱼ = gewichtete Adjazenzmatrix (w_eff)
kᵢ   = Σⱼ Aᵢⱼ (gewichteter Grad)
m    = (1/2) · Σᵢⱼ Aᵢⱼ (totales Kantengewicht)
cᵢ   = Cluster-Zuordnung von Node i
δ    = Kronecker-Delta
```

**Erwartete Modularity für Brain19:** Q ∈ [0.4, 0.7] (typisch für Wissens-Graphen mit klarer Domänenstruktur).

Zum Vergleich:
- Q < 0.3: keine signifikante Cluster-Struktur
- Q ∈ [0.3, 0.7]: gute Community-Struktur
- Q > 0.7: sehr starke Separation (möglicherweise disconnected)

#### Information-Theoretischer Ansatz (Optional, Phase 2)

Alternativ zu Modularity: **Map Equation** (Infomap-Algorithmus). Minimiert die Description Length eines Random Walks auf dem Graph:

```
L(M) = q_↻ · H(Q) + Σᵢ pᵢ_↻ · H(Pⁱ)

q_↻  = Rate der Inter-Cluster-Übergänge
H(Q) = Entropie der Cluster-Übergänge
pᵢ_↻ = Rate der Intra-Cluster-Übergänge in Cluster i
H(Pⁱ) = Entropie der Bewegung in Cluster i
```

**Vorteil:** Besser für hierarchisches Clustering als Modularity (keine Resolution-Limit-Problem).  
**Nachteil:** Komplexere Implementierung. → **Für Phase 2 vormerken.**

#### Query-Performance-Verbesserung durch Clustering

**Theorem:** Cluster-aware FocusCursor reduziert den Suchraum von O(n) auf O(k + c), wobei k = Cluster-Größe und c = Anzahl Cluster-Übergänge.

**Begründung:**
1. FocusCursor traversiert aktuell den gesamten Graphen (29K Nodes potentiell)
2. Mit Cluster-Awareness: Erst Intra-Cluster exploration (typisch 100-500 Nodes), dann gezielter Inter-Cluster-Sprung
3. **Speedup-Faktor:** ~10-50× für fokussierte Queries ("Was ist Photosynthese?" bleibt im Bio-Cluster)
4. Cross-Domain-Queries ("Verbindung Physik-Biologie") nutzen Bridge-Concepts als Cluster-Übergänge

**Formell:**
```
T_ohne_cluster = O(d · n)        mit d=Traversierungstiefe, n=29K
T_mit_cluster  = O(d · k + c)   mit k≈500 (avg Cluster), c≈5 (Cluster-Hops)
Speedup ≈ n/k = 29000/500 = 58×  (für Intra-Cluster-Queries)
```

---

## Agent 3 — Visionär (Produkt/Architektur)

### Fundamentale Fähigkeitsänderung

Clustering transformiert Brain19 von einem **flachen Assoziationsnetz** zu einem **strukturierten Denksystem**:

| Vorher (flach) | Nachher (clustered) |
|:----------------|:--------------------|
| "Photosynthese ist verwandt mit 47 Concepts" | "Photosynthese gehört zum Bio-Cluster, verknüpft mit Chemie-Cluster über Energieumwandlung" |
| FocusCursor wandert ziellos | FocusCursor navigiert gezielt zwischen Wissensbereichen |
| Pattern Discovery findet Co-Occurrence | Pattern Discovery findet **Domänen-Interaktionen** |
| Frage "Ordne ein" → unmöglich | Frage "Ordne ein" → "Gehört zur Biologie, Unterkategorie Pflanzenphysiologie" |

### Analogie zum menschlichen Gehirn

#### Cortical Columns
Das menschliche Gehirn organisiert Wissen in **kortikalen Säulen** (~150K Neuronen pro Minicolumn). Jede Säule verarbeitet ein Feature. Brain19's Cluster sind das direkte Analogon:

```
Cortex                          Brain19
━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━
Temporal Lobe → Semantik        Bio-Cluster → Biologie
Frontal Lobe → Planung          Meta-Cluster → Logik/Kausalität
Parietal Lobe → Spatial         Physik-Cluster → Raum/Kräfte
Long-Range Connections          Bridge-Concepts zwischen Clustern
```

#### Semantic Memory Organization (Collins & Quillian, 1969)
Das klassische Modell semantischer Netzwerke postuliert **hierarchische Speicherung** — Eigenschaften werden auf der abstraktesten anwendbaren Ebene gespeichert:

```
Tier → hat Haut, kann sich bewegen
  └─ Vogel → kann fliegen, hat Federn
       └─ Kanarienvogel → ist gelb, kann singen
```

Genau diese Struktur entsteht durch hierarchisches Clustering + IS_A-Relationen.

### Emergentes Clustering statt expliziter Berechnung

**Kernfrage:** Kann Clustering aus dem Training emergieren statt explizit berechnet zu werden?

**Antwort: Ja, teilweise — und Brain19 hat den Mechanismus bereits.**

Die MicroModels lernen Relevanz-Patterns. Wenn "Photosynthese" und "Chlorophyll" ähnliche Relevanz-Vektoren produzieren, sind sie implizit geclustert — im **Embedding-Space** der MicroModels.

```
Phase 1 (jetzt):  Explizites Louvain auf Graph-Topologie
Phase 2 (später): Emergente Cluster aus MicroModel-Embedding-Ähnlichkeit
Phase 3 (Vision): Fusion — Topologische Cluster + Embedding-Cluster → Robuste Cluster
```

**Emergente Cluster-Vorteile:**
- Brauchen keine explizite Berechnung
- Adaptieren sich automatisch mit MicroModel-Training
- Fangen **funktionale** Ähnlichkeit (nicht nur strukturelle)

**Emergente Cluster-Nachteile:**
- Nicht direkt interpretierbar (Embedding-Space ist opak)
- Nicht stabil (ändern sich mit Training)
- → Daher: Explizites Clustering als **Backbone**, emergente Cluster als **Signal**

### Interaktion mit Meta-Relations

Clustering interagiert synergistisch mit dem Meta-Relations-Plan:

| Meta-Relation | Cluster-Interaktion |
|:--------------|:--------------------|
| **IMPLIES** | Transitiv innerhalb eines Clusters → stärkt Cluster-Kohäsion |
| **CONTRADICTS** | Zwischen Clustern → markiert Cluster-Grenzen; innerhalb → Cluster-Split-Signal |
| **REQUIRES** | Dependency zwischen Clustern → gerichtete Cluster-Relationen |
| **EXCLUDES** | Gegenseitiger Ausschluss → harter Cluster-Separator |

**Cluster-Level Meta-Relations (neu):**
```
Bio-Cluster  ─REQUIRES─→  Chemie-Cluster  ("Biologie braucht Chemie")
Physik-Cluster ─IMPLIES─→ Mathe-Cluster   ("Physik impliziert Mathematik")
Kreationismus-Cluster ─CONTRADICTS─→ Evolution-Cluster
```

Dies ermöglicht **Reasoning auf Cluster-Ebene** — eine fundamental neue Fähigkeit:
- "Wenn du Physik verstehst, verstehst du auch Teile der Mathematik" (IMPLIES)
- "Biologie ohne Chemie-Grundlagen ist unvollständig" (REQUIRES)

### Langfrist-Vision: Selbst-Reorganisierende Cluster

```
Phase 1: Statisches Clustering (Louvain alle 30min)
     │
Phase 2: Inkrementelles Clustering (bei jedem neuen Concept)
     │    └─ Nur betroffener Cluster wird re-evaluiert (~100ms)
     │
Phase 3: Attention-basierte Cluster
     │    └─ Cluster-Zugehörigkeit abhängig vom aktuellen Kontext
     │       "Wasser" gehört zum Chemie-Cluster bei Chemie-Fragen,
     │       zum Bio-Cluster bei Bio-Fragen
     │
Phase 4: Meta-kognitive Cluster
          └─ System erkennt, dass es einen neuen Wissensbereich 
             gelernt hat und erstellt proaktiv einen Cluster
             → "Ich habe jetzt 50 Concepts über Quantenmechanik —
                das ist ein eigener Bereich"
```

**Phase 4 ist die wirkliche Vision:** Brain19 organisiert sein eigenes Wissen so, wie ein Mensch beim Lernen mentale Kategorien bildet. Nicht top-down vorgegeben, sondern bottom-up emergent.

### Visualisierung: Mindmap-Frontend

```
┌──────────────────────────────────────────┐
│           Brain19 Knowledge Map          │
│                                          │
│    ┌─────────┐     ┌──────────┐         │
│    │ Biologie ├─────┤ Chemie   │         │
│    │  (1.2K)  │     │  (890)   │         │
│    └────┬────┘     └────┬─────┘         │
│         │               │                │
│    ┌────┴────┐     ┌────┴─────┐         │
│    │ Pflanzen│     │ Organisch│         │
│    │  (340)  │     │  (210)   │         │
│    └─────────┘     └──────────┘         │
│                                          │
│    ┌─────────┐     ┌──────────┐         │
│    │ Physik  ├─────┤  Mathe   │         │
│    │  (2.1K) │     │  (1.8K)  │         │
│    └─────────┘     └──────────┘         │
└──────────────────────────────────────────┘
```

---

## Synthese: Konsensus-Empfehlung

### Alle drei Agenten stimmen überein:

1. **Louvain-Algorithmus** ist die richtige Wahl (performant, hierarchisch, einfach zu implementieren)
2. **Co-Aktivierung aus Patterns** als zusätzliches Kantengewicht nutzen
3. **Hierarchisches Clustering** mit 3 Ebenen (Fein → Mittel → Grob)
4. **Inkrementelles Re-Clustering** statt Neuberechnung bei kleinen Änderungen

### Empfohlene Gewichtung der Kanten

```
w_eff(i,j) = 0.4 · w_structural + 0.4 · co_activation + 0.2 · trust_avg
```

### Konkreter Zeitplan

| Phase | Aufgabe | Dauer | Abhängigkeit |
|:------|:--------|:------|:-------------|
| **Phase 1** | Louvain-Implementierung + ClusterInfo | 3 Tage | — |
| **Phase 2** | Integration LTM + FocusCursor | 2 Tage | Phase 1 |
| **Phase 3** | Co-Aktivierung aus Patterns als Gewicht | 2 Tage | Phase 1 |
| **Phase 4** | Auto-Labeling + Persistenz | 1.5 Tage | Phase 2 |
| **Phase 5** | Cluster-Level Meta-Relations | 2 Tage | Phase 2 + Meta-Relations-Plan |
| **Phase 6** | Inkrementelles Re-Clustering | 1.5 Tage | Phase 2 |
| **Phase 7** | Frontend-Visualisierung (Mindmap) | 3 Tage | Phase 4 |
| **Phase 8** | Tests + Benchmarks | 1.5 Tage | Alles |
| | **Gesamt** | **~16.5 Tage** | |

### Priorität

**Phase 1-4 sind essentiell** (~8.5 Tage) — damit hat Brain19 funktionierendes hierarchisches Clustering.

Phase 5-8 sind wertvolle Erweiterungen, aber nicht blockierend.

### Risiken

| Risiko | Wahrscheinlichkeit | Mitigation |
|:-------|:-------------------|:-----------|
| Modularity zu niedrig (kein klares Clustering) | 20% | Co-Aktivierung-Gewicht erhöhen |
| Cluster zu granular (tausende Mini-Cluster) | 30% | Min-Cluster-Size = 10, Max-Levels = 4 |
| Performance-Regression durch Cluster-Lookup | 10% | O(1) HashMap-Lookup, gecacht |
| Auto-Labels unbrauchbar | 40% | Fallback auf häufigstes Label im Cluster |
| Inkrementelles Update korrumpiert Hierarchie | 20% | Periodischer Full-Recompute als Safety-Net |

### Architektur-Entscheidung

```
✅ EMPFOHLEN:  Louvain + Co-Aktivierung + 3-Level-Hierarchie
❌ ABGELEHNT:  Spectral Clustering (O(n³), zu teuer)
❌ ABGELEHNT:  Flat Clustering (keine Hierarchie, kein Mindmap)
⏳ VORGEMERKT: Infomap für Phase 2 (bessere Hierarchie, Resolution-Limit-frei)
⏳ VORGEMERKT: Emergente Cluster aus MicroModel-Embeddings (Phase 3)
⏳ VORGEMERKT: Kontext-abhängige Cluster-Zugehörigkeit (Phase 4)
```

---

*Erstellt durch 3-Agenten-Panel, 2026-02-14*  
*Felix Hirschpek, Brain19 Project*
