# Code Audit: Phase 7 — KAN-LLM Hybrid (Final, Iteration 3)

> **Auditor:** Senior C++ Code Review
> **Datum:** 2026-02-10
> **Scope:** `backend/hybrid/*`, `tests/test_kan_llm_hybrid.cpp`
> **Score-Verlauf:** 6.4 → 7.6 → **8.4 / 10**

---

## Gesamtscore: 8.4 / 10

| Kriterium | Iter 1 | Iter 2 | Iter 3 | Kommentar |
|-----------|--------|--------|--------|-----------|
| Korrektheit | 6 | 7.5 | **8.5** | Alle CRITICALs gefixt, Interleaving gefixt |
| Architektur | 8 | 8 | **8** | Unverändert solide |
| KAN-Integration | 5 | 6.5 | **7.5** | Kohärente Daten, Hints funktionieren |
| Epistemic Correctness | 7 | 8 | **9** | Trust-Cap durchgängig, num_data_points=0 gefixt |
| LLM-Integration | 4 | 6 | **7** | Semantic keywords erweitert (inverted-U etc.) |
| Scalability | 6 | 6 | **6** | Unverändert |
| Code Quality | 8 | 8 | **8.5** | Variable-Filter erweitert, Guards sauber |
| Test Coverage | 7 | 7.5 | **7.5** | Keine neuen Tests für die Fixes |

---

## Issue-Tracking: Alle vorherigen Findings

### Aus Audit 1 (6.4/10)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| C1 | Keyword-Matching fundamental unzureichend | CRITICAL | **FIXED** ✅ NLP-lite Parser mit Confidence |
| C2 | RefinementLoop konvergiert nicht sinnvoll | CRITICAL | **FIXED** ✅ Doppel-Validation weg, Stall-Detection |
| H1 | Synthetische Daten haben keinen Bezug zur Hypothese | HIGH | **FIXED** ✅ NumericHints + DataQuality |
| H2 | Trust-Inflation möglich | HIGH | **FIXED** ✅ Hard Cap 0.6 |
| H3 | Domain-Heuristik fragil (if-Kette) | HIGH | **NOT FIXED** — akzeptabel für v1 |
| H4 | Doppel-Validation im RefinementLoop | HIGH | **FIXED** ✅ |
| M1 | Division by zero bei n=1 | MEDIUM | **FIXED** ✅ |
| M2 | Config-Override in KanValidator | MEDIUM | **NOT FIXED** — still overrides translator suggestions |
| M3 | Cross-Domain O(n²) | MEDIUM | **NOT FIXED** — akzeptabel bei 5 Domains |
| M4 | bridges-Vektor Duplikate | MEDIUM | **NOT FIXED** |
| M5 | check_interpretability() zu simpel | MEDIUM | **NOT FIXED** |
| M6 | Schwache Test-Assertions | MEDIUM | **FIXED** ✅ `insights.size() >= 0` entfernt |
| M7 | Keine Edge-Case Tests | MEDIUM | **PARTIALLY FIXED** — div/0 Test da, Rest fehlt |
| L1 | M_PI nicht portabel | LOW | **FIXED** ✅ → `std::numbers::pi` |
| L2 | Novelty-Score hardcoded | LOW | **NOT FIXED** |
| L3 | to_string Kollision | LOW | **NOT FIXED** |

### Aus Audit 2 (7.6/10) — Neue Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| NEW-1 | Linear-Interleaving erzeugt inkohärente Daten | MEDIUM | **FIXED** ✅ |
| NEW-2 | count_variables() False Positives | LOW | **FIXED** ✅ |
| NEW-3 | num_data_points=0 Default bypass | LOW | **FIXED** ✅ |

---

## Detailprüfung der Iteration-3-Fixes

### NEW-1 FIX: Linear-Interleaving → Kohärente Datensätze ✅

**Code:**
```cpp
// FIX NEW-1: Each subset uses ONE consistent slope/bias pair
size_t points_per_set = n / params.size();
size_t remainder = n % params.size();
size_t global_idx = 0;
for (size_t p = 0; p < params.size(); ++p) {
    auto& [s, b] = params[p];
    size_t count = points_per_set + (p < remainder ? 1 : 0);
    for (size_t j = 0; j < count; ++j) { ... }
}
```

**Bewertung:** Korrekt gefixt. Die 100 Punkte werden jetzt in 3 **kontiguierliche** Blöcke aufgeteilt (34+33+33), jeder mit konsistentem Slope. Das KAN sieht jetzt 3 separate lineare Segmente — das ist eine sinnvolle piecewise-linear Funktion, die ein KAN lernen kann.

**Verbleibendes Concern (MINOR):** Die 3 Blöcke haben Diskontinuitäten an den Übergangspunkten (z.B. bei x≈0.33: y springt von 0.7*0.33+0.1=0.331 auf 1.5*0.34-0.2=0.31). Das ist lernbar, aber nicht "linear" im klassischen Sinn. Für den Zweck (kanonische Daten ohne Hints) ist das akzeptabel — es testet ob das KAN *überhaupt* eine Funktion lernen kann.

### NEW-2 FIX: Variable False Positives → Erweiterte Filter ✅

**Code:**
```cpp
if (v != 'a' && v != 'i' && v != 's' && v != 't' && v != 'o') {
    vars.insert(v);
}
```

**Bewertung:** Guter pragmatischer Fix. Die häufigsten Non-Variable-Buchstaben (a, i, s, t, o) werden jetzt gefiltert. Verbleibende False Positives möglich bei "e" (häufig in Englisch), aber da `input_dim` ohnehin auf 1 gecappt wird, ist der Impact minimal.

### NEW-3 FIX: num_data_points=0 bypass → Treat as insufficient ✅

**Code:**
```cpp
// FIX NEW-3: num_data_points=0 means "unknown" → treat as insufficient data
if (num_data_points < config_.min_data_points_for_high_trust) {
    trust = std::min(trust, 0.5);
}
```

**Bewertung:** Korrekt. Die Bedingung `num_data_points > 0 && num_data_points < 50` wurde zu `num_data_points < 50` vereinfacht. Da 0 < 50, wird `num_data_points=0` jetzt korrekt als insufficient behandelt und Trust auf 0.5 gecappt.

### Semantische Keywords (inverted-U etc.) ✅

**Code:**
```cpp
{"inverted-u", 0.85}, {"inverted u", 0.85}, {"u-shaped", 0.85}, {"u shaped", 0.85},
{"bell curve", 0.8}, {"bell-curve", 0.8}, {"diminishing returns", 0.7},
{"peaks at", 0.7}, {"optimal at", 0.7}
```

**Bewertung:** Alle empfohlenen semantischen Keywords wurden zu POLYNOMIAL hinzugefügt. Der Testfall "inverted-U curve" aus dem Re-Audit wird jetzt korrekt als POLYNOMIAL erkannt. Weights sind sinnvoll gewählt (0.7-0.85).

### Trust-Cap 0.6 Durchgängigkeitsprüfung ✅

Geprüft in `compute_trust()`:
1. `synthetic_multiplier = 0.6` angewendet auf base_trust ✅
2. `synthetic_trust_cap = 0.6` als Hard Cap ✅
3. `num_data_points < 50` → Trust ≤ 0.5 ✅
4. `trivial_convergence_penalty = 0.15` bei < 10 Epochs ✅
5. `build_explanation()` zeigt "[TRUST CAPPED at 0.6]" ✅

**Worst-Case-Rechnung (SYNTHETIC_CANONICAL, MSE=0.001, 100 data points, 200 iterations):**
- base_trust = 0.9 - 0.2*(0.001/0.01) = 0.88
- +convergence_bonus (200/1000=0.2 < 0.3): 0.88 + 0.1 = 0.98
- ×synthetic_multiplier: 0.98 × 0.6 = 0.588
- cap(0.6): 0.588
- **Final: 0.588** ✅ Unter 0.6

---

## Neue Issues in Iteration 3

### NEW-4: Keine Tests für die Iteration-3-Fixes (LOW)

Die drei Fixes (Interleaving, Variable-Filter, num_data_points=0) haben **keine dedizierten Tests**. Die bestehenden Tests decken sie implizit ab (test_trust_inflation_cap nutzt 100 data points, test_division_by_zero_guard testet n=1), aber es fehlen:
- ❌ Test: Linear-Daten mit n=100 prüfen dass keine Slope-Sprünge innerhalb eines Blocks
- ❌ Test: `num_data_points=0` → Trust ≤ 0.5
- ❌ Test: "inverted-U curve" → POLYNOMIAL

### NEW-5: Linear-Daten Diskontinuität an Block-Grenzen (LOW)

Die 3-Block-Strategie erzeugt Sprünge an x≈0.33 und x≈0.66. Ein KAN könnte das als piecewise-linear lernen, aber das ist semantisch nicht "linear". Für kanonische Daten akzeptabel.

### NEW-6: KanValidator Config-Override weiterhin aktiv (MEDIUM, inherited)

```cpp
train_config.max_iterations = config_.max_epochs;       // Überschreibt Translator-Vorschlag
train_config.convergence_threshold = config_.convergence_threshold;
```

PERIODIC bekommt vom Translator 2000 Iterationen vorgeschlagen, KanValidator überschreibt auf 1000. Das kann zu schlechteren Ergebnissen für komplexe Patterns führen.

---

## Verbleibende Issues (Zusammenfassung)

| Severity | Count | Issues |
|----------|-------|--------|
| CRITICAL | 0 | — |
| HIGH | 1 | Domain-Heuristik fragil (akzeptabel für v1) |
| MEDIUM | 4 | Config-Override, O(n²) Cross-Domain, bridges Duplikate, check_interpretability simpel |
| LOW | 5 | Novelty hardcoded, to_string Kollision, keine Tests für Iter-3 Fixes, Linear-Diskontinuität, variable "e" false positive |

---

## Gesamtbewertung: Ist Phase 7 production-ready?

**Ja, für den definierten Scope als Prototyp / v1.**

Phase 7 hat sich von einem fragilen PoC (6.4) über einen brauchbaren Prototyp (7.6) zu einer **soliden v1-Implementierung (8.4)** entwickelt:

✅ **Alle CRITICAL Issues sind behoben**
✅ **Trust-Inflation ist gebändigt** (Hard Cap 0.6 durchgängig)
✅ **Trainingsdaten sind kohärent** (Interleaving-Bug gefixt)
✅ **Pattern-Erkennung deutlich robuster** (Negation, Confidence, semantische Muster)
✅ **Edge Cases abgesichert** (div/0, n=0, num_data_points=0)
✅ **Architektur ist sauber** und erlaubt iterative Verbesserung

**Was für 9.0+ fehlt:**
1. Config-Override in KanValidator → `std::optional` für Overrides
2. Echte Daten aus LTM (DataQuality::EXTRACTED Pfad)
3. Multivariate Support (input_dim > 1)
4. Dedizierte Tests für alle Fixes
5. Domain-Heuristik: Voting-basiert statt if-Kette

**Was für 10.0 fehlt:**
- Mini-LLM für Hypothesis→Formalismus (statt Keywords)
- Train/Test-Split für Overfitting-Detection
- Strukturiertes Feedback-Objekt statt String im RefinementLoop
- Property-based Testing

---

*Final Audit erstellt 2026-02-10. Phase 7 ist bereit für Integration in den Main-Branch.*
