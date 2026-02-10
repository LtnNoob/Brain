# Code Audit: Phase 7 — KAN-LLM Hybrid Integration

> **Auditor:** Senior C++ Code Review (automated)
> **Datum:** 2026-02-10
> **Scope:** `backend/hybrid/*`, `tests/test_kan_llm_hybrid.cpp`
> **Kontext:** KAN Module (Phase 2), Epistemic System, Understanding Proposals, ROADMAP Phase 7

---

## Gesamtscore: 6.4 / 10

| Kriterium | Score | Kommentar |
|-----------|-------|-----------|
| Korrektheit | 6 | Funktioniert für Happy-Path, mehrere Edge Cases unbehandelt |
| Architektur | 8 | Saubere Trennung, passt zur Brain19-Philosophie |
| KAN-Integration | 5 | Nur synthetische Daten, kein echtes Hypothesis-Learning |
| Epistemic Correctness | 7 | MSE-Trust Mapping brauchbar, aber Trust-Inflation moeglich |
| LLM-Integration | 4 | Keyword-Matching viel zu simpel, hohe Verlustrate |
| Scalability | 6 | O(n^2) Cross-Domain, sonst akzeptabel |
| Code Quality | 8 | Sauberes C++20, gute RAII, = delete konsequent |

---

## Component: HypothesisTranslator

**Score: 5/10**

### CRITICAL: Keyword-Matching ist fundamental unzureichend

**Finding:** `detect_pattern()` nutzt einfaches `string::find()` auf Keywords. Das ist ein Proof-of-Concept, kein Produktionscode.

**Verlustrate-Analyse -- welche Hypothesen gehen verloren:**
- Hypothesen ohne englische Signalwoerter (z.B. "Druck steigt mit Temperatur")
- Implizite Beziehungen: "A causes B which modulates C" -> nur LINEAR erkannt, obwohl moeglicherweise POLYNOMIAL
- Negierte Muster: "NOT exponential" wird als EXPONENTIAL erkannt
- Multivariate: "X depends on Y and Z" -> immer input_dim=1 (hardcoded)
- Komposit-Muster: "starts linear then becomes exponential" -> nur eines erkannt

```cpp
// BUG: "not exponential" matched "exponential"
if (contains_any(text, {"exponential", ...}))
```

**Severity:** CRITICAL -- Die Uebersetzungsschicht ist der Flaschenhals des gesamten Systems.

### HIGH: Synthetische Daten haben keinen Bezug zur Hypothese

Die generierten Trainingsdaten sind **immer identisch** fuer einen Pattern-Typ. `y = 0.7x + 0.1` fuer JEDE lineare Hypothese, unabhaengig vom Inhalt. Das KAN validiert damit nur, ob es eine kanonische Funktion lernen kann -- nicht ob die Hypothese stimmt.

```cpp
// PROBLEM: Hypothese "Temperatur steigt mit Hoehe" bekommt exakt
// die gleichen Trainingsdaten wie "Preis steigt mit Nachfrage"
double y = 0.7 * x + 0.1;  // Immer gleich
```

### MEDIUM: Division by zero bei n=1

```cpp
double step = (max - min) / static_cast<double>(n - 1);
// n=1 -> division by zero
```

### LOW: M_PI nicht portabel

`M_PI` ist POSIX, nicht C++ Standard. Besser: `std::numbers::pi` (C++20).

**Fix-Vorschlaege:**
1. Kurzfristig: Negation-Check vor Keyword-Match ("not " + keyword)
2. Mittelfristig: Parameter-Extraktion aus Hypothesentext (Regex fuer Zahlen, Variablennamen)
3. Langfristig: Mini-LLM fuer Hypothese->Formalismus Uebersetzung (wie in KAN_LLM_HYBRID_THEORY.md beschrieben)
4. input_dim aus Hypothese ableiten statt hardcoded 1
5. Guard n >= 2 in allen Generatoren

---

## Component: EpistemicBridge

**Score: 7/10**

### HIGH: MSE->Trust Mapping ist nicht wissenschaftlich fundiert

Die Schwellenwerte 0.01 und 0.1 sind **willkuerlich**. MSE ist skalenabhaengig -- eine MSE von 0.005 auf [0,1]-normalisierten Daten ist etwas voellig anderes als MSE 0.005 auf [0,1000]-Daten. Da die Trainingsdaten immer auf [0,1] normalisiert sind, ist das aktuell **konsistent**, aber fragil.

**Ist das wissenschaftlich sinnvoll?** Bedingt. MSE als Fit-Metrik ist Standard. Die Abbildung auf epistemische Kategorien ist eine Design-Entscheidung, keine wissenschaftliche Aussage. Die lineare Interpolation innerhalb der Baender ist vernuenftig. Problematisch ist:

- **Trust-Inflation durch Bonuses:** convergence_bonus(0.1) + interpretability_bonus(0.05) kann Trust von 0.9 auf 1.05 treiben (wird auf 1.0 geclampt). Ein THEORY-Kandidat mit MSE=0.001 bekommt Trust **1.0** -- das ist epistemisch fragwuerdig fuer KAN-validierte synthetische Daten.
- **Kein Penalty fuer Overfitting:** Wenn KAN die synthetischen Daten perfekt fittet (was trivial ist bei kanonischen Funktionen), bekommt die Hypothese maximalen Trust.

### MEDIUM: check_interpretability() zu simpel

Prueft nur 1-Layer, 1->1 KAN und nur Linearitaet (Varianz der Koeffizienten-Differenzen). Quadratische, exponentielle etc. Formen werden nicht erkannt, obwohl der Bonus "interpretable form" heisst.

### LOW: to_string(type) im build_explanation nutzt freie Funktion

Koennte mit EpistemicType-Ueberladung kollidieren -- es gibt to_string() sowohl in epistemic_metadata.hpp als auch hypothesis_translator.hpp.

**Fix-Vorschlaege:**
1. Trust-Cap fuer synthetisch validierte Hypothesen (z.B. max 0.8)
2. Overfitting-Detection: Train/Test-Split der synthetischen Daten
3. MSE-Schwellenwerte konfigurierbar machen (bereits Config) + domain-spezifisch setzen
4. check_interpretability() auf alle Pattern-Typen erweitern

---

## Component: KanValidator

**Score: 7/10**

### MEDIUM: Config-Override verliert Translator-Suggestions

```cpp
train_config.max_iterations = config_.max_epochs;       // Ueberschreibt Translator-Vorschlag
train_config.convergence_threshold = config_.convergence_threshold;
```

Der HypothesisTranslator gibt pattern-spezifische Configs vor (z.B. 2000 Iterationen fuer PERIODIC), aber KanValidator ueberschreibt diese mit seinen eigenen Defaults (1000). Das untergraebt die pattern-spezifische Optimierung.

### LOW: validated Definition inkonsistent

```cpp
bool validated = assessment.converged &&
                 assessment.metadata.type != EpistemicType::SPECULATION;
```

Ein HYPOTHESIS mit Trust 0.4 gilt als "validated", was semantisch fragwuerdig ist -- es ist eher "nicht widerlegt".

**Fix-Vorschlaege:**
1. Nur ueberschreiben wenn Config explizit gesetzt (z.B. std::optional<size_t> max_epochs)
2. validated -> not_rejected oder threshold-basiert

---

## Component: DomainManager

**Score: 6/10**

### HIGH: Heuristiken sind willkuerlich und fragil

classify_relations() nutzt hartcodierte Regeln wie:
```cpp
if (causes_count > 0 && has_property_count > 0 && causes_count >= social_count)
    return DomainType::PHYSICAL;
```

**Probleme:**
- "Stress causes burnout" + "Burnout has_property severity" -> PHYSICAL (falsch, ist SOCIAL)
- Ein einziger TEMPORAL_BEFORE Relation reicht fuer TEMPORAL, selbst wenn 10 CAUSES-Relationen existieren
- Reihenfolge der if-Kette ist entscheidend und nicht dokumentiert

### MEDIUM: Cross-Domain Insights O(n^2)

find_cross_domain_insights() iteriert ueber alle Domain-Paare x alle Concepts x alle Relations. Bei vielen Domains und Concepts wird das teuer. Aktuell 5 Domains = max 10 Paare, also noch okay.

### MEDIUM: bridges-Vektor hat Duplikate

```cpp
bridges.push_back(ca);           // Kann mehrfach gepusht werden
bridges.push_back(rel.target);   // wenn ca multiple Relations zu set_b hat
```

### LOW: Novelty-Score hardcoded fuer nur 2 Paare

Nur PHYSICAL<->SOCIAL (0.8) und BIOLOGICAL<->ABSTRACT (0.7) haben spezielle Scores. Alle anderen: 0.5.

**Fix-Vorschlaege:**
1. Domain-Heuristik: Voting-basiert statt if-Kette (hoechster Count gewinnt, mit Tie-Breaking)
2. std::unordered_set<ConceptId> fuer bridges statt Vektor
3. Novelty-Matrix statt hardcoded Pairs
4. Langfristig: ML-basierte Domain-Klassifikation aus Embedding-Space

---

## Component: RefinementLoop

**Score: 6/10**

### CRITICAL: Konvergiert das wirklich?

**Nein, nicht zuverlaessig.** Das Problem:

1. Der refiner-Callback bekommt Textfeedback ("MSE=0.3, bitte verfeinern") -- aber die Trainingsdaten sind **immer kanonisch** (y=0.7x+0.1 fuer LINEAR). Egal wie der LLM die Hypothese reformuliert, solange das Pattern gleich bleibt, sind die Trainingsdaten identisch -> **MSE aendert sich nicht**.

2. Wenn der Refiner das Pattern aendert (z.B. von LINEAR zu POLYNOMIAL), aendert sich die Topologie und die Daten -- aber das hat nichts mit "Verfeinerung" zu tun, sondern ist ein **komplett neues** Training.

3. **Terminierung ist garantiert** durch max_iterations und improvement_threshold. Aber "Terminierung" != "Konvergenz".

### HIGH: Doppel-Validation Bug

```cpp
ValidationResult last_validation = validator_.validate(current_hypothesis); // VERSCHWENDET
for (...) {
    auto validation = validator_.validate(current_hypothesis); // Ueberschreibt sofort
    ...
    last_validation = std::move(validation);
```

Erste Validation vor der Schleife ist pure CPU-Verschwendung.

### MEDIUM: Improvement-Check bei identischen Refinements

Bei identischem Refiner (wie im Test) stoppt Loop in Iteration 2 wegen Stall -- korrekt, aber zeigt dass der Loop mit echtem LLM untested ist.

**Fix-Vorschlaege:**
1. **Doppel-Validation entfernen:** last_validation vor der Schleife loeschen
2. Refinement muss Trainingsdaten beeinflussen koennen (z.B. Parameter-Extraktion aus refinierter Hypothese)
3. Residuum sollte strukturiert sein (nicht nur String), damit Refiner gezielt reagieren kann

---

## Component: Tests

**Score: 7/10**

### MEDIUM: Schwache Assertions

```cpp
ASSERT(insights.size() >= 0);  // IMMER wahr (size_t ist unsigned)
```

Test 8 (cross_domain_query) testet effektiv nur "kein Crash", nicht Korrektheit.

### MEDIUM: Kein Test fuer Edge Cases

Fehlende Tests:
- n=1 Trainingsdaten (Division by zero)
- Leere Hypothese
- Sehr lange Hypothesentexte
- UTF-8/Sonderzeichen in Hypothesen
- range_min == range_max
- Trust-Boundary-Tests (exakt auf Schwellenwert)

### LOW: Kein Negativ-Test fuer Trust-Inflation

Kein Test prueft, ob Trust > 0.95 fuer synthetische Daten verhindert wird.

**Fix-Vorschlaege:**
1. ASSERT(insights.size() >= 0) -> ASSERT(insights.size() >= 1) oder entfernen
2. Edge-Case-Tests ergaenzen
3. Property-based Testing: Fuer jeden Pattern muss Trust < 0.85 gelten (synthetisch)

---

## Architektur-Empfehlungen

### 1. Synthetische Daten sind das Kernproblem

Das gesamte System validiert aktuell nur: "Kann ein KAN eine kanonische mathematische Funktion lernen?" -- Die Antwort ist **immer ja** (bei genug Iterationen). Das sagt **nichts** ueber die Hypothese aus.

**Empfehlung:** Phase 7.1 sollte echte Daten aus dem LTM extrahieren. Hypothese "A causes B" -> Historische A/B-Werte aus STM/LTM als Trainingsdaten.

### 2. Multivariate Support fehlt komplett

Alles ist input_dim=1, output_dim=1. Reale Hypothesen sind fast immer multivariat.

### 3. Der Refinement-Loop braucht strukturiertes Feedback

String-basiertes Feedback ("MSE=0.3, bitte verfeinern") kann kein LLM sinnvoll nutzen. Besser: Strukturiertes Residuum (Pattern mismatch regions, suggested parameter adjustments).

### 4. Gute Basis fuer Iteration

Trotz aller Kritik: Die **Architektur** ist solide. HypothesisTranslator -> KanValidator -> EpistemicBridge -> RefinementLoop ist die richtige Dekomposition. Die Abstraktionen sind sauber. Die Config-Objekte erlauben schrittweise Verbesserung ohne Refactoring. Das ist ein guter Prototyp, der jetzt mit echten Daten und besserem NLU gefuellt werden muss.

---

## Zusammenfassung der Findings

| Severity | Count | Highlights |
|----------|-------|------------|
| CRITICAL | 2 | Keyword-Matching verliert Hypothesen; Refinement konvergiert nicht sinnvoll |
| HIGH | 4 | Synthetische Daten beweisen nichts; Domain-Heuristik fragil; Trust-Inflation; Doppel-Validation |
| MEDIUM | 7 | Div/0, Config-Override, O(n^2), schwache Tests, Duplikate, Interpretability zu simpel |
| LOW | 4 | M_PI, Novelty hardcoded, Provenance-Kopien, to_string Kollision |

---

*Audit erstellt 2026-02-10. Naechster Review empfohlen nach Fix der CRITICAL Issues.*
