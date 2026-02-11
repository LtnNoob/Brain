#!/usr/bin/env python3
"""Generate complete Brain19 technical documentation as HTML."""

import os
import re
import glob

REPO = "/home/hirschpekf/brain19/backend"

# Module definitions: (id, name, files_glob, responsibility, authority)
MODULES = [
    ("main", "main.cpp", ["main.cpp"], "Application entry point — CLI parsing, mode selection", "Top-level"),
    ("common", "common/types", ["common/types.hpp"], "Core type aliases (ConceptId, ContextId, RelationId)", "Foundation"),
    ("epistemic", "epistemic/epistemic_metadata", ["epistemic/epistemic_metadata.hpp"], "Epistemic type system — EpistemicType, EpistemicStatus, EpistemicMetadata with compile-time enforcement", "Core invariant"),
    ("ltm", "ltm/ — Long-Term Memory", ["ltm/*.hpp", "ltm/*.cpp"], "Persistent knowledge storage — ConceptInfo, RelationInfo, concept/relation CRUD with epistemic enforcement", "Knowledge authority"),
    ("memory", "memory/ — Short-Term Memory", ["memory/activation_level.hpp", "memory/active_relation.hpp", "memory/stm_entry.hpp", "memory/stm.hpp", "memory/stm.cpp", "memory/brain_controller.hpp", "memory/brain_controller.cpp"], "Activation-based working memory — STM (contexts, decay, export/import), BrainController (minimal orchestration)", "Activation authority"),
    ("micromodel", "micromodel/ — MicroModel System", ["micromodel/*.hpp", "micromodel/*.cpp"], "Per-concept bilinear models (predict, train, serialize) — Registry, EmbeddingManager, MicroTrainer, RelevanceMap, binary Persistence", "Learning authority"),
    ("kan", "kan/ — Kolmogorov-Arnold Network", ["kan/*.hpp", "kan/*.cpp"], "B-spline univariate functions (KANNode), layer grids (KANLayer), multi-layer modules (KANModule) with training — FunctionHypothesis wrapper", "Function approximation"),
    ("adapter", "adapter/kan_adapter", ["adapter/*.hpp", "adapter/*.cpp"], "Clean interface between BrainController and KAN — module lifecycle, train, evaluate", "KAN bridge"),
    ("cognitive", "cognitive/ — Cognitive Dynamics", ["cognitive/*.hpp", "cognitive/*.cpp"], "Spreading Activation (trust-weighted, depth-limited), Salience Computation, Focus Management (Miller's 7±2), Thought Path Ranking (beam search)", "Cognitive processing — READ-ONLY on LTM"),
    ("curiosity", "curiosity/ — Curiosity Engine", ["curiosity/*.hpp", "curiosity/*.cpp"], "Pure signal generator — observes system state, emits CuriosityTriggers (shallow relations, low exploration)", "Signal generation only"),
    ("understanding", "understanding/ — Understanding Layer", ["understanding/*.hpp", "understanding/*.cpp"], "MiniLLM abstract interface (extract_meaning, generate_hypotheses, detect_analogies, detect_contradictions) + StubMiniLLM + OllamaMiniLLM + UnderstandingLayer + MiniLLMFactory — ALL outputs HYPOTHESIS", "Semantic analysis — READ-ONLY on LTM"),
    ("hybrid", "hybrid/ — KAN-LLM Hybrid", ["hybrid/*.hpp", "hybrid/*.cpp"], "HypothesisTranslator (NLP→KAN training problem), EpistemicBridge (MSE→Trust mapping), KanValidator (end-to-end validation), DomainManager (domain-specific KAN config), RefinementLoop (bidirectional LLM↔KAN dialog)", "Validation pipeline"),
    ("ingestor", "ingestor/ — Ingestion Pipeline", ["ingestor/text_chunker.hpp", "ingestor/entity_extractor.hpp", "ingestor/relation_extractor.hpp", "ingestor/trust_tagger.hpp", "ingestor/proposal_queue.hpp", "ingestor/knowledge_ingestor.hpp", "ingestor/ingestion_pipeline.hpp", "ingestor/*.cpp"], "Complete ingestion: TextChunker → EntityExtractor → RelationExtractor → TrustTagger → ProposalQueue → LTM. Supports JSON/CSV/plaintext. ADDITIVE only.", "Ingestion authority"),
    ("importers", "importers/ — External Importers", ["importers/*.hpp", "importers/*.cpp"], "WikipediaImporter, ScholarImporter, HttpClient — external knowledge sources → KnowledgeProposal", "External data bridge"),
    ("bootstrap", "bootstrap/ — Foundation Bootstrap", ["bootstrap/*.hpp", "bootstrap/*.cpp"], "FoundationConcepts (4-tier seeding: ontology→categories→relations→science, ~233 concepts), BootstrapInterface (guided onboarding), ContextAccumulator (domain coverage tracking)", "Bootstrap authority"),
    ("llm", "llm/ — LLM Integration", ["llm/*.hpp", "llm/*.cpp"], "OllamaClient (HTTP REST to Ollama), ChatInterface (LLM-powered verbalization of LTM knowledge, epistemic metadata in prompts) — LLM is a TOOL, not an agent", "Verbalization tool"),
    ("persistent", "persistent/ — Persistence Layer", ["persistent/*.hpp", "persistent/*.cpp"], "WAL (Write-Ahead Log), PersistentLTM (disk storage), CheckpointManager/Restore, STMSnapshot, StringPool — binary formats with checksums", "Persistence authority"),
    ("concurrent", "concurrent/ — Thread Safety", ["concurrent/*.hpp"], "SharedLTM, SharedSTM, SharedRegistry, SharedEmbeddings — reader-writer lock wrappers. DeadlockDetector, LockHierarchy — debug tooling", "Concurrency safety"),
    ("streams", "streams/ — Thinking Streams", ["streams/*.hpp", "streams/*.cpp"], "ThinkStream (autonomous thinking thread with work queue), StreamOrchestrator (lifecycle, health monitoring), StreamScheduler (category-based: Perception/Reasoning/Memory/Creative with budgets), StreamMonitor (latency histograms, alerts), MPMCQueue (lock-free Vyukov queue)", "Parallel thinking"),
    ("core", "core/ — System Orchestrator", ["core/system_orchestrator.hpp", "core/system_orchestrator.cpp", "core/brain19_app.hpp", "core/brain19_app.cpp", "core/thinking_pipeline.hpp", "core/thinking_pipeline.cpp"], "SystemOrchestrator (15-subsystem initialization, ask/ingest/checkpoint, periodic maintenance), Brain19App (REPL + CLI dispatch), ThinkingPipeline (multi-step thinking cycles with spreading→salience→curiosity→understanding→KAN validation)", "System authority"),
    ("evolution", "evolution/ — Knowledge Evolution", ["evolution/*.hpp", "evolution/*.cpp"], "PatternDiscovery (repeated patterns in KG), EpistemicPromotion (HYPOTHESIS→THEORY promotion based on evidence), ConceptProposer (new concept suggestions from patterns)", "Knowledge evolution"),
]

def count_lines(filepath):
    try:
        with open(filepath) as f:
            return sum(1 for _ in f)
    except:
        return 0

def get_classes(filepath):
    classes = []
    try:
        with open(filepath) as f:
            content = f.read()
        # Find class/struct declarations
        for m in re.finditer(r'(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?(?:\s*\{)?', content):
            name = m.group(1)
            if name not in ('Config', 'Stats') and not name.startswith('_'):
                classes.append(name)
    except:
        pass
    return list(dict.fromkeys(classes))  # dedupe preserving order

def get_functions(filepath):
    funcs = []
    try:
        with open(filepath) as f:
            content = f.read()
        # Public method declarations in headers
        for m in re.finditer(r'(?:virtual\s+)?(?:static\s+)?(?:[\w:<>]+(?:\s*[*&])?)\s+(\w+)\s*\(', content):
            name = m.group(1)
            if name not in ('if', 'for', 'while', 'switch', 'return', 'throw', 'delete', 'operator') and not name.startswith('_'):
                funcs.append(name)
    except:
        pass
    return list(dict.fromkeys(funcs))[:30]  # dedupe, limit

def get_files_for_module(globs):
    files = []
    for g in globs:
        files.extend(sorted(glob.glob(os.path.join(REPO, g))))
    return files

# Start HTML generation
html = []
html.append('''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain19 — Complete Technical Documentation</title>
<style>
  :root { --bg:#0d1117; --surface:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
          --accent:#58a6ff; --green:#3fb950; --yellow:#d29922; --red:#f85149; --code-bg:#1c2128; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--text); line-height:1.6; padding:2rem; max-width:1200px; margin:0 auto; }
  h1 { font-size:2rem; border-bottom:1px solid var(--border); padding-bottom:.5rem; margin-bottom:1.5rem; }
  h2 { color:var(--accent); font-size:1.4rem; margin:2rem 0 .8rem; border-bottom:1px solid var(--border); padding-bottom:.3rem; }
  h3 { color:var(--green); font-size:1.1rem; margin:1.2rem 0 .4rem; }
  h4 { color:var(--yellow); font-size:.95rem; margin:.8rem 0 .2rem; }
  p,li { margin-bottom:.4rem; }
  ul { padding-left:1.5rem; }
  code { background:var(--code-bg); padding:.1rem .3rem; border-radius:3px; font-size:.85em; color:var(--accent); }
  pre { background:var(--code-bg); border:1px solid var(--border); border-radius:6px; padding:.8rem; overflow-x:auto; margin:.4rem 0 .8rem; }
  pre code { background:none; padding:0; color:var(--text); }
  section { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1.2rem; margin-bottom:1.2rem; }
  table { width:100%; border-collapse:collapse; margin:.4rem 0 .8rem; font-size:.9em; }
  th,td { text-align:left; padding:.3rem .6rem; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; font-size:.8em; text-transform:uppercase; }
  .tag { display:inline-block; padding:.05rem .4rem; border-radius:10px; font-size:.75em; font-weight:600; margin-right:.2rem; }
  .tag-ro { background:#3fb95033; color:var(--green); }
  .tag-rw { background:#d2992233; color:var(--yellow); }
  .tag-core { background:#58a6ff33; color:var(--accent); }
  .tag-ext { background:#f8514933; color:var(--red); }
  .weak { background:#f8514915; border-left:3px solid var(--red); padding:.5rem .8rem; margin:.3rem 0; border-radius:0 4px 4px 0; font-size:.9em; }
  .note { background:#58a6ff15; border-left:3px solid var(--accent); padding:.5rem .8rem; margin:.3rem 0; border-radius:0 4px 4px 0; font-size:.9em; }
  .invariant { background:#3fb95015; border-left:3px solid var(--green); padding:.5rem .8rem; margin:.3rem 0; border-radius:0 4px 4px 0; font-size:.9em; }
  nav { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:.8rem 1.2rem; margin-bottom:1.2rem; }
  nav a { color:var(--accent); text-decoration:none; margin-right:.6rem; font-size:.9em; }
  nav a:hover { text-decoration:underline; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; }
  @media (max-width:768px) { .grid { grid-template-columns:1fr; } }
  .stat { text-align:center; padding:.5rem; background:var(--code-bg); border-radius:4px; }
  .stat-num { font-size:1.5rem; font-weight:700; color:var(--accent); }
  .stat-label { font-size:.75rem; color:var(--muted); text-transform:uppercase; }
  details { margin:.3rem 0; }
  summary { cursor:pointer; color:var(--accent); font-weight:600; }
  summary:hover { text-decoration:underline; }
</style>
</head>
<body>
<h1>🧠 Brain19 — Complete Technical Documentation</h1>
<p>Auto-generated module-by-module documentation. Generated: 2026-02-11</p>
''')

# Stats
total_files = 0
total_lines = 0
all_classes = []
for mid, mname, mglobs, mresp, mauth in MODULES:
    files = get_files_for_module(mglobs)
    total_files += len(files)
    for f in files:
        total_lines += count_lines(f)
        all_classes.extend(get_classes(f))

html.append(f'''
<div class="grid" style="grid-template-columns:repeat(4,1fr); margin-bottom:1.2rem;">
  <div class="stat"><div class="stat-num">{len(MODULES)}</div><div class="stat-label">Modules</div></div>
  <div class="stat"><div class="stat-num">{total_files}</div><div class="stat-label">Source Files</div></div>
  <div class="stat"><div class="stat-num">{total_lines:,}</div><div class="stat-label">Lines of Code</div></div>
  <div class="stat"><div class="stat-num">{len(set(all_classes))}</div><div class="stat-label">Classes/Structs</div></div>
</div>
''')

# Navigation
html.append('<nav><strong>Modules:</strong> ')
for mid, mname, *_ in MODULES:
    html.append(f'<a href="#{mid}">{mname.split("/")[0] if "/" in mname else mid}</a>')
html.append('</nav>')

# Architecture overview
html.append('''
<section>
<h2>Architecture Overview</h2>
<pre><code>┌─────────────────────────────────────────────────────────────────┐
│                    Brain19App (REPL + CLI)                       │
│                         main.cpp                                │
├─────────────────────────────────────────────────────────────────┤
│                   SystemOrchestrator                            │
│    (15 subsystems, lifecycle, ask/ingest/checkpoint)            │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ LTM      │ STM      │ Cognitive│ Under-   │ Hybrid              │
│ (know-   │ (activa- │ Dynamics │ standing │ (KAN Validator,     │
│  ledge)  │  tion)   │ (spread, │ Layer    │  Epistemic Bridge,  │
│          │          │  focus)  │ (MiniLLM)│  Refinement Loop)   │
├──────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ MicroModel Registry │ KAN Modules        │ Ingestion Pipeline   │
│ (per-concept learn) │ (B-spline approx)  │ (text→concepts)      │
├──────────┴──────────┴──────────┴──────────┴─────────────────────┤
│ Persistence (WAL, Checkpoints)  │  Streams (parallel thinking)  │
│ Concurrent (SharedLTM/STM/...)  │  Bootstrap (foundation seed)  │
└─────────────────────────────────┴───────────────────────────────┘</code></pre>

<h3>Key Architectural Invariants</h3>
<div class="invariant"><strong>Epistemic Enforcement:</strong> No default-constructible EpistemicMetadata. Every concept MUST have explicit type + trust. Knowledge is NEVER deleted, only INVALIDATED.</div>
<div class="invariant"><strong>READ-ONLY Boundaries:</strong> CognitiveDynamics, UnderstandingLayer, MiniLLMs — all READ-ONLY on LTM. Only SystemOrchestrator + IngestionPipeline write to LTM.</div>
<div class="invariant"><strong>HYPOTHESIS Enforcement:</strong> All MiniLLM outputs are HYPOTHESIS. No automatic FACT promotion. EpistemicBridge maps KAN training results to trust scores.</div>
<div class="invariant"><strong>Bounded Values:</strong> Activation ∈ [0,1], Trust ∈ [0,1], Salience ∈ [0,1], Weight ∈ [0,1]. All clamped.</div>
</section>
''')

# Per-module sections
for mid, mname, mglobs, mresp, mauth in MODULES:
    files = get_files_for_module(mglobs)
    line_count = sum(count_lines(f) for f in files)
    classes = []
    functions = []
    for f in files:
        classes.extend(get_classes(f))
        functions.extend(get_functions(f))
    classes = list(dict.fromkeys(classes))
    functions = list(dict.fromkeys(functions))[:25]
    
    file_list = [os.path.relpath(f, REPO) for f in files]
    
    html.append(f'''
<section id="{mid}">
<h2>Module: {mname}</h2>
<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Files</td><td>{len(files)}</td></tr>
  <tr><td>Lines</td><td>{line_count:,}</td></tr>
  <tr><td>Responsibility</td><td>{mresp}</td></tr>
  <tr><td>Authority</td><td><strong>{mauth}</strong></td></tr>
  <tr><td>Classes/Structs</td><td>{len(classes)}</td></tr>
</table>

<h3>Files</h3>
<ul>
{"".join(f"<li><code>{fn}</code></li>" for fn in file_list)}
</ul>
''')
    
    if classes:
        html.append('<h3>Classes & Structs</h3><ul>')
        for c in classes:
            html.append(f'<li><code>{c}</code></li>')
        html.append('</ul>')
    
    if functions:
        html.append('<h3>Key Functions</h3>')
        html.append('<details><summary>Show functions</summary><ul>')
        for fn in functions:
            html.append(f'<li><code>{fn}()</code></li>')
        html.append('</ul></details>')
    
    html.append('</section>')

# Dependency graph
html.append('''
<section>
<h2>Module Dependency Graph</h2>
<pre><code>main.cpp
  └─ core/brain19_app
       └─ core/system_orchestrator
            ├─ ltm/long_term_memory ← epistemic/epistemic_metadata ← common/types
            ├─ memory/stm + brain_controller
            ├─ micromodel/ (registry, embeddings, trainer, relevance_map, persistence)
            ├─ kan/ (node → layer → module) ← adapter/kan_adapter
            ├─ cognitive/cognitive_dynamics (reads LTM + writes STM)
            ├─ curiosity/curiosity_engine (reads STM)
            ├─ understanding/ (mini_llm → ollama_mini_llm → understanding_layer)
            ├─ hybrid/ (hypothesis_translator → kan_validator → epistemic_bridge → refinement_loop, domain_manager)
            ├─ ingestor/ (text_chunker → entity_extractor → relation_extractor → trust_tagger → proposal_queue → ingestion_pipeline)
            ├─ importers/ (wikipedia, scholar → http_client)
            ├─ bootstrap/ (foundation_concepts, bootstrap_interface, context_accumulator)
            ├─ llm/ (ollama_client → chat_interface)
            ├─ persistent/ (wal, persistent_ltm, checkpoint_manager, checkpoint_restore, stm_snapshot)
            ├─ concurrent/ (shared_ltm, shared_stm, shared_registry, shared_embeddings)
            ├─ streams/ (think_stream → stream_orchestrator → stream_scheduler → stream_monitor)
            └─ evolution/ (pattern_discovery, epistemic_promotion, concept_proposal)
</code></pre>
</section>

<section>
<h2>Data Flow: Ask a Question</h2>
<pre><code>User: "What is gravity?"
  │
  ├─ SystemOrchestrator::ask(question)
  │    ├─ ThinkingPipeline::run_cycle(seed_concepts)
  │    │    ├─ CognitiveDynamics::spread_activation(seeds)     → STM activated
  │    │    ├─ CognitiveDynamics::get_top_k_salient()          → salient concepts
  │    │    ├─ CognitiveDynamics::find_best_paths()            → thought paths
  │    │    ├─ CuriosityEngine::observe_and_generate_triggers() → curiosity signals
  │    │    ├─ UnderstandingLayer::perform_understanding_cycle() → proposals (HYPOTHESIS)
  │    │    └─ KanValidator::validate(hypothesis)               → ValidationResult + Trust
  │    │
  │    └─ ChatInterface::ask_with_context(question, ltm, salient, paths)
  │         ├─ Build epistemic context (concept labels + trust + relations)
  │         ├─ OllamaClient::chat(messages)                    → LLM verbalization
  │         └─ ChatResponse { answer, epistemic_note, referenced_concepts }
  │
  └─ Display answer with epistemic metadata
</code></pre>
</section>

<section>
<h2>Data Flow: Ingest Knowledge</h2>
<pre><code>Input: "Einstein developed the theory of general relativity in 1915."
  │
  ├─ IngestionPipeline::ingest_text(text, auto_approve)
  │    ├─ TextChunker::chunk_text()              → TextChunk[]
  │    ├─ EntityExtractor::extract_from_chunks()  → ExtractedEntity[] ("Einstein", "general relativity")
  │    ├─ RelationExtractor::extract_relations()  → ExtractedRelation[] (Einstein → relativity: CAUSES)
  │    ├─ TrustTagger::suggest_from_text()        → TrustAssignment (THEORY, trust=0.90)
  │    ├─ ProposalQueue::add_proposal()           → IngestProposal (PENDING)
  │    ├─ [auto_approve or human review]
  │    ├─ LTM::store_concept()                    → ConceptId (with EpistemicMetadata)
  │    ├─ LTM::add_relation()                     → RelationId
  │    └─ MicroModelRegistry::ensure_models_for()  → new MicroModels created
  │
  └─ IngestionResult { concepts_stored, relations_stored }
</code></pre>
</section>

<section>
<h2>Key Design Decisions</h2>
<ul>
<li><strong>No default EpistemicMetadata:</strong> <code>EpistemicMetadata() = delete</code> — forces explicit epistemic decisions at every knowledge creation point</li>
<li><strong>Knowledge never deleted:</strong> INVALIDATED status with trust &lt; 0.2 preserves epistemic history</li>
<li><strong>MicroModel per concept:</strong> Each concept has its own 430-parameter bilinear model (W·c+b, 10D) for personalized relevance scoring</li>
<li><strong>KAN for function learning:</strong> B-spline basis functions (Cox-de Boor recursion) enable smooth univariate approximation without fixed activation functions</li>
<li><strong>Bidirectional LLM↔KAN refinement:</strong> Hypotheses from LLM → KAN training → residual feedback → refined hypothesis</li>
<li><strong>Epistemic Bridge:</strong> Maps KAN training metrics (MSE, convergence) to trust scores: MSE&lt;0.01→THEORY, MSE&lt;0.1→HYPOTHESIS, else→SPECULATION</li>
<li><strong>Parallel thinking streams:</strong> Category-based (Perception/Reasoning/Memory/Creative) with priority scheduling and anti-starvation</li>
<li><strong>Lock-free work queues:</strong> Vyukov bounded MPMC queue for stream task distribution</li>
<li><strong>WAL + Checkpoints:</strong> Write-ahead log for durability, periodic checkpoints for fast recovery</li>
<li><strong>Foundation seeding:</strong> 4-tier bootstrap (~233 concepts, ~144 relations) provides ontological scaffold for new knowledge</li>
</ul>
</section>

<section>
<h2>Known Weak Points & Gaps</h2>
<div class="weak"><strong>⚠️ MiniLLM→LTM validation loop not wired:</strong> Proposals from UnderstandingLayer go through KanValidator but the SystemOrchestrator doesn't yet write validated results back to LTM (~50 lines missing)</div>
<div class="weak"><strong>⚠️ MicroModels learn topology not semantics:</strong> Training is one-shot from graph structure. No incremental learning from query feedback.</div>
<div class="weak"><strong>⚠️ InteractionLayer doesn't exist:</strong> MicroModels don't communicate with each other. Only passive RelevanceMap aggregation, no Hopfield-like activation propagation between models.</div>
<div class="weak"><strong>⚠️ EMBED_DIM = 10 too small:</strong> Stability analysis recommends 16D minimum. Current 10D limits representational capacity.</div>
<div class="weak"><strong>⚠️ No weight normalization:</strong> System unstable at node degree &gt;5 without normalization of inter-model weights.</div>
<div class="weak"><strong>⚠️ No SIGINT handler in main():</strong> Ctrl+C in command mode skips shutdown() and final checkpoint.</div>
<div class="weak"><strong>⚠️ Ollama dependency:</strong> ChatInterface and OllamaMiniLLM require external Ollama server. No fallback for offline operation beyond StubMiniLLM.</div>
<div class="weak"><strong>⚠️ No symmetry enforcement:</strong> W_ij inter-model weights (future InteractionLayer) must be symmetric for convergence guarantee.</div>
</section>
''')

html.append('</body></html>')

# Write
output = os.path.join("/home/hirschpekf/brain19/docs/technical/index.html")
with open(output, 'w') as f:
    f.write('\n'.join(html))

print(f"Generated: {output}")
print(f"Modules: {len(MODULES)}, Files: {total_files}, Lines: {total_lines:,}, Classes: {len(set(all_classes))}")
