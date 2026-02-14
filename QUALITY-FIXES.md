# Brain19 Quality Fixes Plan

Implement ALL fixes below in order. Build and test after all changes.

## Fix 1: Convergence Threshold (concept_model.hpp or concept_trainer.cpp)
- Change `convergence_threshold` from `1e-6` to `1e-4`
- Change `max_epochs` from `100` to `500`

## Fix 2: Training Loop Merge (concept_trainer.cpp)
In `train_all()`, merge the two separate loops into one. Per concept: first bilinear train, then immediately train_refined for its outgoing relations. Move `concept_store` and `RECALL_HASH` before the loop.

## Fix 3: Quality Gate in predict_edge (concept_pattern_engine.cpp)
Add a quality gate: only use scores from converged models with enough samples. Track training state (converged, sample_count, final_loss) in ConceptModel. In `predict_edge()`: if model not converged or samples < 10, return 0.0. Discount scores based on training quality.

## Fix 4: Hypothesis Threshold (concept_pattern_engine.cpp)  
Raise missing-link threshold from 0.5 to 0.7. Add bidirectional consistency check: if A→B strong, B→A should be at least 0.3. Add shared-neighbor check: without common neighbors, require score > 0.85.

## Fix 5: Patience-based Early Stopping (concept_model.cpp)
Replace strict threshold convergence with patience-based: track best_loss, if no improvement for 10 epochs → converged. Also mark as converged if final_loss < 0.01.

## Fix 6: Skip Concepts Without Relations (concept_trainer.cpp)
In train_all(), count positive samples. If num_positives == 0, skip training for that concept.

## Fix 7: Query Parsing (chat_interface.cpp)
1. Expand STOP_WORDS with German pronouns: mir, mich, dir, dich, ihm, ihn, uns, euch, ihnen, mein, dein, sein, unser, etc.
2. Add INTENT_VERBS set: erklaer, erklaere, beschreib, beschreibe, zeig, zeige, definier, definiere, vergleich, vergleiche, finde, such, suche, explain, describe, show, tell, define, compare, find, search, list
3. Filter intent verbs from keywords (treat them like stopwords)
4. Weight keywords by length: `weight = min(2.0, 0.5 + len/8.0)` and multiply into concept scores
5. For multi-word concept labels: if full phrase not in query but only partial word match with <50% coverage → multiply score by 0.2

## Build & Test
```
cd backend && find . -name '*.o' -delete && make -f Makefile brain19 -j4 && cp brain19 /home/hirschpekf/brain19/brain19
```
Then test with: kill old brain19, start new one, wait for training, query "Erkläre mir Photosynthese" via API.
