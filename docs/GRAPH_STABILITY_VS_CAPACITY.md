# Brain19 -- Graph Stability vs. Capacity Analysis

Generated: 2026-02-11T18:28:46.803590 UTC

------------------------------------------------------------------------

## Core Question

Is a stable reasoning graph with 50--100 nodes sufficient for generating
useful answers?

------------------------------------------------------------------------

## 1. Stability vs. Decoder Drift

In traditional autoregressive language models, error accumulation occurs
due to free token generation without structural anchoring.

In Brain19:

-   The decoder is conditioned on a dynamic reasoning graph.
-   Token generation follows graph structure.
-   Error accumulation shifts from **sequential drift** to **graph
    stability**.

Thus, the primary bottleneck becomes:

> How stable and deep can the reasoning graph remain under Lyapunov
> constraints?

------------------------------------------------------------------------

## 2. What 50--100 Nodes Can Realistically Support

### Simple Explanations (e.g., physical processes)

-   5--15 nodes
-   1--2 causal chains
-   3--10 relations

100 nodes are more than sufficient.

### Medium Complexity (e.g., economics, medicine)

-   20--40 nodes
-   Multiple interacting mechanisms
-   3--6 abstraction layers

Still well within capacity.

### High Complexity (e.g., global systems analysis)

-   60--150 nodes
-   Feedback loops
-   Multi-domain interactions

Upper bound approached but still manageable with pruning.

------------------------------------------------------------------------

## 3. True Limiting Factor

The constraint is not token count but:

-   Graph expansion stability
-   Inhibition control
-   Weight normalization
-   Convergence enforcement

Current safeguards:

  -----------------------------------------------------------------------
  Bottleneck                           Implemented Control Mechanism
  ------------------------------------ ----------------------------------
  Graph explosion                      Relevance threshold + max 100
                                       nodes

  Contradictory relations              Inhibition + Energy function

  Semantic misweighting                Weight normalization

  Expansion instability                Damping (λ=0.1), convergence ε,
                                       max 20--30 cycles
  -----------------------------------------------------------------------

Stability condition: - Symmetric weights - Spectral radius \< 4 -
Damping applied

------------------------------------------------------------------------

## 4. Capacity Assessment

A 50--100 node stable graph is sufficient for:

-   Structured explanations
-   Mechanistic reasoning
-   Multi-step causal analysis
-   Most applied knowledge tasks

It is insufficient for:

-   Full world modeling
-   Extremely deep recursive reasoning
-   Large-scale theoretical synthesis

These would require hierarchical or modular subgraphs.

------------------------------------------------------------------------

## 5. Final Assessment

The limitation shifts from:

> "How many tokens can the decoder sustain?"

to:

> "How deep and coherent can the reasoning graph remain?"

Given the implemented stability constraints, the architecture is likely
sufficient for the majority of practical reasoning tasks.

Scalability beyond that requires hierarchical graph composition rather
than increasing raw node count.
