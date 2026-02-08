# Cognitive Dynamics Module

## Überblick

**Cognitive Dynamics** ist das Aufmerksamkeits- und Fokus-Management-System von Brain19. Es implementiert mechanistische, deterministische Prozesse zur Steuerung der kognitiven Ressourcen-Allokation.

## Architektur-Prinzipien

### ✅ ERLAUBT (Was Cognitive Dynamics MACHT)

1. **Spreading Activation**
   - Trust-gewichtete Aktivierungs-Propagation
   - Depth-limited (konfigurierbar)
   - Cycle Detection
   - Bounded values [0.0, 1.0]

2. **Salience Computation**
   - Wichtigkeits-Ranking basierend auf:
     - Activation Level
     - Trust (aus LTM, READ-ONLY)
     - Connectivity (Anzahl Relationen)
   - Sortierung nach kombiniertem Score

3. **Focus Management**
   - Arbeitsgedächtnis-Simulation
   - Kapazitätslimit (konfigurierbar)
   - Time-based Decay
   - Access-time Tracking

4. **Thought Path Ranking**
   - Findet Inferenz-Pfade im Knowledge-Graph
   - Priorisierung nach:
     - Path Length (kürzere Pfade bevorzugt)
     - Accumulated Trust (höhere Trust-Produkte bevorzugt)
     - Relation Weights
   - Max Depth limit

### ❌ VERBOTEN (Was Cognitive Dynamics NICHT macht)

1. **Keine Wissens-Erzeugung**
   - Erstellt KEINE neuen Konzepte
   - Erstellt KEINE neuen Relationen
   - Generiert KEINE Hypothesen

2. **Keine Epistemischen Entscheidungen**
   - Ändert NICHT Trust-Werte
   - Ändert NICHT EpistemicType
   - Ändert NICHT EpistemicStatus
   - Promoted NICHT SPECULATION zu FACT

3. **Keine Autonomie**
   - Trifft KEINE Entscheidungen über Wichtigkeit
   - Wählt NICHT automatisch Konzepte aus
   - Führt KEINE Aktionen aus
   - Nur mechanische Berechnungen

4. **READ-ONLY Zugriff**
   - LTM: Nur lesen, nie schreiben
   - Trust: Nur als Gewichtungsfaktor verwenden
   - Epistemic Metadata: Nur konsumieren, nie ändern

## Komponenten

### 1. Spreading Activation

```cpp
SpreadingStats spread_activation(
    ConceptId seed,
    double initial_activation,
    ContextId context,
    const LongTermMemory& ltm,      // READ-ONLY!
    ShortTermMemory& stm            // WRITE: Nur Aktivierungen
);
```

**Algorithmus:**
1. Starte mit Seed-Konzept und initialer Aktivierung
2. Hole ausgehende Relationen aus LTM (read-only)
3. Für jede Relation:
   - Lese Trust des Ziel-Konzepts (read-only)
   - Berechne propagierte Aktivierung:
     ```
     propagated = current_activation * relation_weight * trust * decay_factor
     ```
   - Schreibe Aktivierung in STM (bounded [0.0, 1.0])
4. Rekursiv bis max_depth erreicht
5. Cycle Detection verhindert Endlosschleifen

**Konfiguration:**
```cpp
SpreadingConfig config;
config.max_depth = 3;           // Maximale Rekursionstiefe
config.min_activation = 0.1;    // Schwellwert für Propagation
config.decay_per_level = 0.8;   // Decay-Faktor pro Ebene
```

### 2. Salience Computation

```cpp
std::vector<SalienceScore> compute_salience_batch(
    const std::vector<ConceptId>& concept_ids,
    ContextId context,
    const LongTermMemory& ltm,      // READ-ONLY!
    const ShortTermMemory& stm,     // READ-ONLY!
    uint64_t current_tick
);
```

**Formel:**
```
salience = activation * trust * connectivity_factor * recency_bonus
```

**Komponenten:**
- `activation`: Aus STM (0.0-1.0)
- `trust`: Aus LTM (READ-ONLY, 0.0-1.0)
- `connectivity_factor`: Normalisierte Anzahl Relationen
- `recency_bonus`: Zeitbasierter Boost (optional)

**Sortierung:**
- Höchste Salience zuerst
- Deterministische Reihenfolge bei Gleichstand (ID-basiert)

### 3. Focus Management

```cpp
void update_focus(
    ConceptId concept_id,
    double focus_score,
    uint64_t current_tick
);

void decay_focus(
    uint64_t current_tick,
    double decay_rate
);
```

**Focus Entry:**
```cpp
struct FocusEntry {
    ConceptId concept_id;
    double score;               // [0.0, 1.0]
    uint64_t last_access_tick;
};
```

**Kapazitätslimit:**
- Wenn Limit überschritten: Niedrigste Focus-Scores werden entfernt
- LRU (Least Recently Used) bei Gleichstand

**Decay:**
```
new_score = old_score * exp(-decay_rate * time_delta)
```

### 4. Thought Path Ranking

```cpp
std::vector<ThoughtPath> find_paths(
    ConceptId start,
    ConceptId goal,
    const LongTermMemory& ltm,      // READ-ONLY!
    size_t max_paths,
    size_t max_depth
);
```

**Path Scoring:**
```
path_score = (∏ trust_values) * (∏ relation_weights) / path_length
```

**Algorithmus:**
- Breadth-First Search mit Depth-Limit
- Cycle Detection
- Top-K Pfade nach Score

## Verwendung

### Demo

```bash
cd backend
make -f Makefile.cognitive
./demo_cognitive_dynamics
```

**Output:**
```
╔══════════════════════════════════════════════════════╗
║  Brain19 - Cognitive Dynamics Integration Demo       ║
╚══════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1: Spreading Activation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Spreading activation from 'Cat' (initial activation = 1.0)...

Spreading Statistics:
  Concepts activated: 8
  Max depth reached: 2
  Total activation added: 4.013

Activation levels:
  Cat:          1.000
  Mammal:       0.745
  Fur:          0.706
  ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2: Salience Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Salience scores (sorted by importance):

  Cat              0.944
  Mammal           0.753
  Fur              0.663
  ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3: Focus Management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current focus set:
  Fur (focus score: 1.000)
  Mammal (focus score: 0.900)
  Cat (focus score: 0.800)

After decay:
  Fur (focus score: 0.810)
  Mammal (focus score: 0.729)
  Cat (focus score: 0.648)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4: Thought Path Ranking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Top thought paths:
  Path 1 (score: 0.676): Cat → Fur
  Path 2 (score: 0.667): Cat → Whiskers
  Path 3 (score: 0.654): Cat → Predator
  ...

═══════════════════════════════════════════════════════
  ALL EPISTEMIC INVARIANTS PRESERVED ✓
═══════════════════════════════════════════════════════
```

### Tests

```bash
./test_cognitive_dynamics
```

**Test Suite (8 Tests):**

1. **Epistemic Invariants Preservation**
   - Trust-Werte unverändert
   - EpistemicType unverändert
   - EpistemicStatus unverändert

2. **Spreading Determinism**
   - Gleicher Input → gleicher Output
   - Reproduzierbar

3. **Bounded Activations**
   - Alle Werte ∈ [0.0, 1.0]

4. **Cycle Detection**
   - Keine Endlosschleifen bei zyklischen Graphen

5. **Focus Decay**
   - Time-based Decay funktioniert

6. **Salience Ranking**
   - Trust beeinflusst Ranking korrekt
   - Sortierung deterministisch

7. **Thought Path Finding**
   - Pfade werden gefunden
   - Ranking nach Score korrekt

8. **INVALIDATED Concepts**
   - INVALIDATED concepts propagieren nicht

**Erwartung:** Alle Tests bestehen ✅

## Integration

### In eigenem Code verwenden

```cpp
#include "cognitive/cognitive_dynamics.hpp"
#include "memory/brain_controller.hpp"
#include "ltm/long_term_memory.hpp"

using namespace brain19;

// Setup
BrainController brain;
brain.initialize();

LongTermMemory ltm;
CognitiveDynamics cognitive;

// Wissen aufbauen
auto cat = ltm.store_concept("Cat", "Feline",
    EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

auto mammal = ltm.store_concept("Mammal", "Warm-blooded",
    EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));

ltm.add_relation(cat, mammal, RelationType::IS_A, 0.9);

// Spreading Activation
ContextId ctx = brain.create_context();
ShortTermMemory* stm = brain.get_stm_mutable();

auto stats = cognitive.spread_activation(cat, 1.0, ctx, ltm, *stm);

std::cout << "Activated: " << stats.concepts_activated << " concepts\n";
std::cout << "Max depth: " << stats.max_depth_reached << "\n";

// Salience berechnen
std::vector<ConceptId> concepts = {cat, mammal};
auto scores = cognitive.compute_salience_batch(concepts, ctx, ltm, *stm, 0);

for (const auto& score : scores) {
    std::cout << "Concept " << score.concept_id
              << " salience: " << score.salience << "\n";
}

// Cleanup
brain.destroy_context(ctx);
brain.shutdown();
```

## Konfiguration

### SpreadingConfig

```cpp
SpreadingConfig config;
config.max_depth = 3;               // Maximale Rekursionstiefe (1-10)
config.min_activation = 0.1;        // Schwellwert für Propagation (0.0-1.0)
config.decay_per_level = 0.8;       // Decay pro Ebene (0.0-1.0)
config.enable_bidirectional = false; // Bidirektionale Propagation
```

### SalienceConfig

```cpp
SalienceConfig config;
config.activation_weight = 0.4;     // Gewicht für Activation (0.0-1.0)
config.trust_weight = 0.3;          // Gewicht für Trust (0.0-1.0)
config.connectivity_weight = 0.2;   // Gewicht für Connectivity (0.0-1.0)
config.recency_weight = 0.1;        // Gewicht für Recency (0.0-1.0)
config.normalize_output = true;     // Output auf [0.0, 1.0] normalisieren
```

### FocusConfig

```cpp
FocusConfig config;
config.max_capacity = 7;            // Miller's Law: 7 ± 2
config.decay_rate = 0.1;            // Decay pro Tick
config.min_score_threshold = 0.1;   // Minimum Focus Score
```

## Architektur-Garantien

### Epistemische Invarianten

**KRITISCH:** Cognitive Dynamics **DARF NIEMALS** epistemische Metadaten ändern.

**Verifikation:**
```cpp
// BEFORE spreading
auto before = ltm.retrieve_concept(cat);
double trust_before = before->epistemic.trust;

// Spreading Activation
cognitive.spread_activation(cat, 1.0, ctx, ltm, *stm);

// AFTER spreading
auto after = ltm.retrieve_concept(cat);
double trust_after = after->epistemic.trust;

assert(trust_before == trust_after);  // MUST be equal
assert(before->epistemic.type == after->epistemic.type);
assert(before->epistemic.status == after->epistemic.status);
```

### Determinismus

**Garantie:** Gleicher Input → gleicher Output

**Verifikation:**
```cpp
// Run 1
auto stats1 = cognitive.spread_activation(seed, 1.0, ctx1, ltm, stm1);

// Run 2 (identischer Graph)
auto stats2 = cognitive.spread_activation(seed, 1.0, ctx2, ltm, stm2);

assert(stats1.concepts_activated == stats2.concepts_activated);
assert(stats1.max_depth_reached == stats2.max_depth_reached);
```

### Bounded Values

**Garantie:** Alle Werte ∈ [0.0, 1.0]

**Enforcement:**
```cpp
double clamp_activation(double value) {
    return std::max(0.0, std::min(1.0, value));
}

double clamp_salience(double value) {
    return std::max(0.0, std::min(1.0, value));
}
```

## Dateien

```
backend/cognitive/
├── cognitive_dynamics.hpp      # Hauptklasse + API
├── cognitive_dynamics.cpp      # Implementation
├── cognitive_config.hpp        # Konfigurationsstrukturen
└── README.md                   # Diese Datei
```

## Performance

**Typische Werte (8-Konzept-Graph):**
- Spreading Activation: < 1ms
- Salience Computation: < 0.1ms
- Focus Management: < 0.05ms
- Path Finding (max_depth=3): < 2ms

**Komplexität:**
- Spreading: O(N * D) wobei N = Konzepte, D = max_depth
- Salience: O(N log N) wegen Sortierung
- Focus: O(K) wobei K = Kapazität
- Path Finding: O(B^D) wobei B = Branching Factor, D = max_depth

## Erweiterungen

### Geplant (Understanding Layer)

- **Abstraction Formation:** Muster in aktivierten Konzepten erkennen
- **Analogy Detection:** Strukturelle Ähnlichkeiten finden
- **Contradiction Detection:** Inkonsistenzen identifizieren
- **Evidence Accumulation:** Vertrauens-Updates vorschlagen (nicht ausführen!)

**Wichtig:** Auch diese Erweiterungen bleiben **read-only** bezüglich LTM!

## Lizenz

Teil des Brain19 Cognitive Architecture Projekts.

---

**Status:** ✅ Implementiert, getestet, dokumentiert
**Datum:** 21. Januar 2026
**Version:** 1.0
