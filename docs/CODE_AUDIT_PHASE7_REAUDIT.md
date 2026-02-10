# Code Re-Audit: Phase 7 — KAN-LLM Hybrid (Post-Fix)

> **Auditor:** Senior C++ Code Review
> **Datum:** 2026-02-10
> **Scope:** `backend/hybrid/*`, `tests/test_kan_llm_hybrid.cpp`
> **Vorheriger Score:** 6.4/10
> **Kontext:** Re-Audit nach Fixes für C1, C2, H1, H2

---

## Neuer Gesamtscore: 7.6 / 10

| Kriterium | Vorher | Nachher | Kommentar |
|-----------|--------|---------|-----------|
| Korrektheit | 6 | 7.5 | Alle CRITICAL Bugs adressiert, ein paar Lücken bleiben |
| Architektur | 8 | 8 | Unverändert gut |
| KAN-Integration | 5 | 6.5 | Hypothesis-spezifische Daten deutlich besser |
| Epistemic Correctness | 7 | 8 | Trust-Inflation gebändigt |
| LLM-Integration | 4 | 6 | NLP-lite großer Sprung, aber noch nicht produktionsreif |
| Scalability | 6 | 6 | Unverändert |
| Code Quality | 8 | 8 | Sauber geblieben, gute Guard-Clauses |

---

## Fix-Bewertungen

### C1: Keyword-Matching → NLP-lite Parser — **VERIFIED** ✅

**Was gefixt wurde:**
- `detect_pattern_detailed()` ersetzt das primitive `contains_any()`-Matching durch gewichtetes Score-Akkumulations-System über alle Klauseln
- Negation-Detection via `is_negated()` (30-Char Lookback für Negationswörter)
- Sentence Decomposition via `split_sentences()` (Satzzeichen + Konjunktionen)
- Quantifier Handling via `detect_quantifier_modifier()` mit gestuften Werten
- Confidence Scoring mit konfigurierbarem Threshold (0.3)
- Multi-Variable Detection via `count_variables()` (Single-Letter + Phrasen-Heuristik)

**Negation-Detection Bewertung:**
- ✅ "not exponential" → korrekt negiert (Lookback findet "not ")
- ✅ "never increases" → "never " in Negator-Liste
- ✅ "does not correlate" → "not " gefunden
- ⚠️ **Schwäche:** Lookback von 30 Chars ist fragil. Bei längeren Einschüben zwischen Negator und Keyword könnte es versagen.
- ⚠️ **Schwäche:** Doppelte Negation ("not non-linear") wird nicht erkannt → würde fälschlich als negiert gelten

**Sentence Decomposition:**
- ✅ Splittet auf `.;!?` und auf `, but `, `, however `, ` whereas `
- ⚠️ Splittet NICHT auf `, where ` — relevant für den Testfall unten
- ⚠️ Splittet NICHT auf ` and ` zwischen unabhängigen Klauseln

**Quantifier Handling — Modifier-Werte:**
- `rarely/seldom` → 0.3 ✅ sinnvoll
- `sometimes/possibly` → 0.5 ✅ sinnvoll
- `often/usually` → 0.8 ✅ sinnvoll
- `always/certainly` → 1.0 ✅ sinnvoll
- Default → 0.9 ✅ gut gewählt (leicht konservativ)

**Confidence Threshold 0.3:**
- ✅ Sinnvoll. Verhindert dass "slightly related" als Pattern durchgeht.
- ⚠️ Ein einzelnes schwaches Keyword ("rate") mit Default-Quantifier (0.9) ergibt 0.3 × 0.9 = 0.27 → gerade noch NOT_QUANTIFIABLE. Knapp aber OK.

**Multi-Variable Detection:**
- ⚠️ **Fragil.** Nutzt Regex `\b([a-z])\b` für Einzelbuchstaben und filtert nur `a` und `i`. Matcht auch "s", "t" etc. als False Positives.
- ⚠️ `input_dim` wird auf `min(detected_variables, 1)` gecapped → Multi-Variable Detection wird effektiv ignoriert.

**TESTFALL: "The relationship between dopamine levels and motivation follows an inverted-U curve, where both low and high concentrations reduce performance"**

Analyse des Codes:
1. `split_sentences()`: Kein Split (kein `.;!?`, kein `, but `, `, however `, ` whereas `) → bleibt ein einziger Satz
2. Keyword-Matching auf den vollen Text:
   - Kein starkes Pattern-Keyword matcht! ("inverted-U", "curve" sind nicht in den Keyword-Listen)
   - "low" → matcht "lower" nicht, kein Match
   - "reduces" → nicht in Keywords ("decreases" wäre drin, aber nicht "reduces")
   - "performance" → kein Match
3. **Ergebnis: `NOT_QUANTIFIABLE`** mit confidence ≈ 0

**Problem:** Das ist eine klare POLYNOMIAL (inverted-U = Parabel) Hypothese, aber ohne "quadratic", "polynomial", "squared", "parabolic" Keywords wird sie nicht erkannt. Semantische Muster fehlen.

**Empfehlung:** `{"inverted-u", 0.85}, {"u-shaped", 0.85}, {"bell curve", 0.8}, {"diminishing returns", 0.7}` zu den POLYNOMIAL-Keywords hinzufügen.

---

### C2: RefinementLoop konvergiert nicht → Residuum-basierte Iteration — **VERIFIED** ✅

**Doppel-Validation Bug:**
- ✅ **Vollständig gefixt.** Die überflüssige `validator_.validate()` vor der Schleife ist entfernt. Kommentar dokumentiert den Fix. `std::optional<ValidationResult>` statt Pre-Loop-Validation.

**Konvergenz-Analyse:**
- ✅ MSE-Delta wird jetzt zwischen Iterationen verglichen (`prev_mse - current_mse`)
- ✅ Stall-Detection: `abs(improvement) < improvement_threshold` stoppt bei Stagnation
- ✅ Terminierung ist **garantiert** durch:
  1. `max_iterations` Hard-Limit
  2. `mse_threshold` Erfolgs-Abbruch
  3. `improvement_threshold` Stall-Abbruch

**Residuum-Feedback:**
- ✅ `build_residual_feedback()` gibt jetzt strukturierte Info: MSE, Konvergenz-Status, Trust, und qualitative Einordnung (>0.5 = "completely different", >0.1 = "parameter adjustments", <0.1 = "fine-tuning")
- ⚠️ **Immer noch String-basiert.** Ein strukturiertes Feedback-Objekt wäre robuster.
- ⚠️ **Kernproblem bleibt:** Wenn der Refiner das Pattern nicht ändert, bleiben die Trainingsdaten identisch → MSE ändert sich nicht → Stall nach Iteration 2. Korrekt behandelt, aber Loop ist für identische Refiner effektiv ein Single-Shot-Validator.

**Neue Beobachtung:** `provenance_chain` hält alle trainierten KAN-Modelle im RAM via `shared_ptr`. Bei 5 Iterationen OK, bei vielen problematisch.

---

### H1: Synthetische Daten → Hypothesis-specific Data — **PARTIALLY_FIXED** ⚠️

**Was gefixt wurde:**
- ✅ `NumericHints` Extraktion via Regex für Zahlen, Slopes, Ranges, Scales
- ✅ `DataQuality` Enum: `SYNTHETIC_CANONICAL` vs `SYNTHETIC_SPECIFIC` vs `EXTRACTED`
- ✅ Hint-basierte Range-Nutzung in `translate()`
- ✅ Verschiedene Parameter pro Hypothese wenn Hints vorhanden
- ✅ Linear-Generator nutzt 3 verschiedene Parameter-Sets statt nur `y = 0.7x + 0.1`

**Warum PARTIALLY_FIXED:**
- ⚠️ **Linear-Interleaving Bug (NEW-1):** Der Linear-Generator interleaved 3 Parameter-Sets per `i % 3`. Punkt 0 hat Slope 0.7, Punkt 1 hat Slope 1.5, Punkt 2 hat Slope 0.3. Das erzeugt **inkohärente** Trainingsdaten die kein KAN sinnvoll lernen kann — drei Funktionen übereinander statt einer realistischen Beziehung.
- ⚠️ **Ohne Hints bleiben Polynomial/Exponential/Periodic/Threshold/Conditional kanonisch.** Nur Linear hat (fehlerhaften) Diversity-Mechanismus.

**NumericHints Extraction:**
- ✅ Regex `[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?` — robust für übliche Zahlenformate
- ⚠️ Matcht auch irrelevante Zahlen (Jahreszahlen, Referenz-Nummern)
- ⚠️ Range-Pattern matcht nur erste Occurrence

**DataQuality Propagation:**
- ✅ Durchgängig: `translate()` → `KanTrainingProblem` → `validate()` → `assess()` → `EpistemicAssessment`

---

### H2: Trust-Inflation → Hard Cap 0.6 — **VERIFIED** ✅

**0.6 Cap Durchgängigkeit:**
- ✅ `compute_trust()` wendet `synthetic_trust_cap = 0.6` an für alles was nicht `EXTRACTED` ist
- ✅ Zusätzlich `synthetic_multiplier = 0.6` auf den base_trust
- ✅ `build_explanation()` zeigt "[TRUST CAPPED at 0.6]" → Transparenz

**Novelty Penalty -0.15:**
- ✅ Wird angewendet wenn `iterations_used < trivial_convergence_epochs (10)`
- ⚠️ Harter Sprung statt Gradient. Akzeptabel für v1.

**Minimum 50 Datenpunkte:**
- ✅ Enforced: `num_data_points < 50` → `trust = min(trust, 0.5)`
- ⚠️ **Bypass bei Default:** `num_data_points` Default ist 0, und Check ist `num_data_points > 0 && ...` → 0 umgeht den Check. KanValidator setzt es korrekt, aber andere Caller könnten es vergessen.

**Rechnungsbeispiel (Worst Case, SYNTHETIC_CANONICAL, MSE=0.001, 100 data points):**
- base_trust = 0.88, +fast_bonus = 0.98, ×0.6 = 0.588, cap(0.6) = 0.588
- **Final: 0.588** ✅ Unter 0.6, Cap funktioniert

---

## Neue Bugs Eingeführt

### NEW-1: Linear Data Interleaving erzeugt inkohärente Trainingsdaten (MEDIUM)

```cpp
// generate_linear_data() ohne Hints:
auto& [s, b] = params[i % params.size()]; // 3 verschiedene Slopes pro Datenpunkt
```

Aufeinanderfolgende Punkte haben verschiedene Slopes → kein KAN kann das als "eine" Funktion lernen.

**Fix:** Pro Aufruf EINEN zufälligen Parameter-Set wählen:
```cpp
auto& [s, b] = params[std::hash<size_t>{}(n) % params.size()];
```

### NEW-2: count_variables() False Positives (LOW)

Single-Letter Regex matcht zu aggressiv. Nur `a` und `i` gefiltert, aber "s", "t" etc. sind ebenfalls häufig.

### NEW-3: num_data_points=0 Default bypass (LOW)

`assess()` Default `num_data_points=0` umgeht den MinPoints-Check.

---

## Verbleibende Issues aus dem Original-Audit

| Issue | Status |
|-------|--------|
| Config-Override in KanValidator | **NICHT GEFIXT** |
| Domain-Heuristik fragil | **NICHT GEFIXT** |
| Cross-Domain O(n²) | **NICHT GEFIXT** (akzeptabel) |
| bridges-Vektor Duplikate | **NICHT GEFIXT** |
| check_interpretability() zu simpel | **NICHT GEFIXT** |
| Novelty-Score hardcoded | **NICHT GEFIXT** |
| `ASSERT(insights.size() >= 0)` | **GEFIXT** ✅ |
| M_PI | **GEFIXT** ✅ |
| Division by zero n=1 | **GEFIXT** ✅ |

---

## Test Coverage Analyse

| Test | Deckt Fix | Bewertung |
|------|-----------|-----------|
| test_negation_detection (13) | C1 Negation | ✅ Gut |
| test_confidence_scoring (14) | C1 Confidence | ✅ |
| test_conditional_detection (15) | C1 Conditional | ✅ |
| test_quantifier_modifier (16) | C1 Quantifier | ✅ |
| test_numeric_hint_extraction (17) | H1 NumericHints | ✅ |
| test_data_quality_tracking (18) | H1 DataQuality | ✅ |
| test_trust_inflation_cap (19) | H2 Cap | ✅ |
| test_trivial_convergence_penalty (20) | H2 Novelty | ✅ |
| test_min_data_points_trust (21) | H2 MinPoints | ✅ |
| test_division_by_zero_guard (22) | Edge Case | ✅ |

**Fehlende Tests:**
- ❌ Semantische Muster (inverted-U, bell curve)
- ❌ Doppel-Negation
- ❌ `num_data_points=0` Bypass
- ❌ Linear-Interleaving Kohärenz
- ❌ Refinement mit pattern-änderndem Refiner

---

## Zusammenfassung

| Fix | Bewertung | Kommentar |
|-----|-----------|-----------|
| C1: NLP-lite Parser | **VERIFIED** | Großer Fortschritt. Semantic Gap bleibt. |
| C2: RefinementLoop | **VERIFIED** | Doppel-Validation weg, Terminierung garantiert. |
| H1: Hypothesis-specific Data | **PARTIALLY_FIXED** | Hints gut, Linear-Interleaving neuer Bug. |
| H2: Trust-Inflation Cap | **VERIFIED** | 0.6 Cap durchgängig enforced. |

**Score-Verbesserung: 6.4 → 7.6 (+1.2)**

Die beiden CRITICAL Bugs sind substantiell adressiert. Die Trust-Inflation ist gebändigt. Das System ist von einem fragilen PoC zu einem brauchbaren Prototyp geworden.

---

## Empfehlungen für nächste Iteration

1. **LINEAR-Interleaving fixen** (NEW-1) — trivial, aber Korrektheitsproblem
2. **Semantische Keywords erweitern** — "inverted-U", "bell curve", "diminishing returns"
3. **num_data_points Default absichern** — 0 als "unknown" behandeln und cappen
4. **Config-Override in KanValidator** — `std::optional` für Overrides
5. **Langfristig:** Echte Daten aus LTM extrahieren (DataQuality::EXTRACTED)

---

*Re-Audit erstellt 2026-02-10. Nächster Review empfohlen nach Fix von NEW-1 und semantischer Keyword-Erweiterung.*
