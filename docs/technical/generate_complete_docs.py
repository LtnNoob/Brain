#!/usr/bin/env python3
"""Generate complete Brain19 technical documentation as self-contained HTML."""

import datetime

# ──────────────────────────────────────────────────────────────────────────────
# CSS + JS template
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --yellow: #d29922; --red: #f85149;
  --code-bg: #1c2128;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: 'SF Mono','Fira Code','Cascadia Code',monospace;
  background: var(--bg); color: var(--text);
  line-height: 1.65; padding: 2rem; max-width: 1300px; margin: 0 auto;
}
h1 { font-size:1.8rem; border-bottom:1px solid var(--border); padding-bottom:.5rem; margin-bottom:1.2rem; color:var(--accent); }
h2 { color:var(--accent); font-size:1.3rem; margin:1.5rem 0 .6rem; border-bottom:1px solid var(--border); padding-bottom:.2rem; }
h3 { color:var(--green); font-size:1.05rem; margin:1rem 0 .3rem; }
h4 { color:var(--yellow); font-size:.9rem; margin:.6rem 0 .2rem; }
p, li { margin-bottom:.35rem; font-size:.88rem; }
ul { padding-left:1.3rem; }
code { background:var(--code-bg); padding:.1rem .25rem; border-radius:3px; font-size:.82em; color:var(--accent); }
pre { background:var(--code-bg); border:1px solid var(--border); border-radius:6px; padding:.6rem; overflow-x:auto; margin:.3rem 0 .6rem; font-size:.8em; }
pre code { background:none; padding:0; color:var(--text); }
section { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1rem; margin-bottom:1rem; }
table { width:100%; border-collapse:collapse; margin:.3rem 0 .6rem; font-size:.82em; }
th, td { text-align:left; padding:.25rem .5rem; border-bottom:1px solid var(--border); }
th { color:var(--muted); font-weight:600; font-size:.75em; text-transform:uppercase; }
.tag { display:inline-block; padding:.05rem .35rem; border-radius:10px; font-size:.7em; font-weight:600; margin-right:.15rem; }
.tag-ro  { background:#3fb95033; color:var(--green); }
.tag-rw  { background:#d2992233; color:var(--yellow); }
.tag-core{ background:#58a6ff33; color:var(--accent); }
.tag-ext { background:#f8514933; color:var(--red); }
.weak { background:#f8514915; border-left:3px solid var(--red); padding:.4rem .7rem; margin:.25rem 0; border-radius:0 4px 4px 0; font-size:.82em; }
.invariant { background:#3fb95015; border-left:3px solid var(--green); padding:.4rem .7rem; margin:.25rem 0; border-radius:0 4px 4px 0; font-size:.82em; }
.note { background:#58a6ff15; border-left:3px solid var(--accent); padding:.4rem .7rem; margin:.25rem 0; border-radius:0 4px 4px 0; font-size:.82em; }
nav { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:.6rem 1rem; margin-bottom:1rem; }
nav a { color:var(--accent); text-decoration:none; margin-right:.5rem; font-size:.82em; }
nav a:hover { text-decoration:underline; }
details { margin:.2rem 0; }
summary { cursor:pointer; color:var(--accent); font-weight:600; font-size:.88em; }
summary:hover { text-decoration:underline; }
.grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.5rem; margin-bottom:1rem; }
.stat { text-align:center; padding:.4rem; background:var(--code-bg); border-radius:4px; }
.stat-num { font-size:1.4rem; font-weight:700; color:var(--accent); }
.stat-label { font-size:.7rem; color:var(--muted); text-transform:uppercase; }
@media (max-width:768px) { .grid { grid-template-columns:1fr 1fr; } body { padding:.5rem; } }
#controls { position:fixed; top:1rem; right:1rem; z-index:100; display:flex; gap:.3rem; }
#controls button { background:var(--surface); border:1px solid var(--border); color:var(--accent); padding:.3rem .6rem; border-radius:4px; cursor:pointer; font-size:.75rem; font-family:inherit; }
#controls button:hover { background:var(--code-bg); }
"""

JS = """
function expandAll(){document.querySelectorAll('details').forEach(d=>d.open=true);}
function collapseAll(){document.querySelectorAll('details').forEach(d=>d.open=false);}
document.querySelectorAll('nav a[href^="#"]').forEach(a=>{
  a.addEventListener('click',e=>{e.preventDefault();
    document.querySelector(a.getAttribute('href')).scrollIntoView({behavior:'smooth'});
  });
});
"""

# ──────────────────────────────────────────────────────────────────────────────
# Module data: 20 modules in dependency order
# ──────────────────────────────────────────────────────────────────────────────

def e(s):
    """HTML escape"""
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def tag(label, cls):
    return f'<span class="tag {cls}">{label}</span>'

def member_table(rows):
    """rows: list of (name, type, desc)"""
    h = '<table><tr><th>Name</th><th>Type</th><th>Description</th></tr>\n'
    for name, typ, desc in rows:
        h += f'<tr><td><code>{e(name)}</code></td><td><code>{e(typ)}</code></td><td>{e(desc)}</td></tr>\n'
    h += '</table>\n'
    return h

def method_list(methods):
    """methods: list of (signature, description)"""
    h = '<details><summary>Methods</summary><ul>\n'
    for sig, desc in methods:
        h += f'<li><code>{e(sig)}</code> &mdash; {e(desc)}</li>\n'
    h += '</ul></details>\n'
    return h

def invariant_box(text):
    return f'<div class="invariant"><strong>Invariant:</strong> {e(text)}</div>\n'

def weak_box(text):
    return f'<div class="weak"><strong>Weak Point:</strong> {e(text)}</div>\n'

def note_box(text):
    return f'<div class="note">{text}</div>\n'

# ──────────────────────────────────────────────────────────────────────────────
# Build HTML content
# ──────────────────────────────────────────────────────────────────────────────

modules_nav = [
    ("common","common"),("epistemic","epistemic"),("memory","memory"),("ltm","ltm"),
    ("micromodel","micromodel"),("kan","kan"),("adapter","adapter"),
    ("cognitive","cognitive"),("curiosity","curiosity"),("understanding","understanding"),
    ("hybrid","hybrid"),("ingestor","ingestor"),("importers","importers"),
    ("bootstrap","bootstrap"),("llm","llm"),("persistent","persistent"),
    ("concurrent","concurrent"),("streams","streams"),("core","core"),("evolution","evolution"),
]

def build_html():
    parts = []
    p = parts.append

    p(f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain19 -- Complete Technical Documentation</title>
<style>{CSS}</style>
</head>
<body>
<div id="controls">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
</div>

<h1>Brain19 -- Complete Technical Documentation</h1>
<p>Auto-generated module-by-module documentation covering all classes, structs, enums, and functions.</p>
<p style="color:var(--muted);font-size:.78em;">Generated: {datetime.date.today().isoformat()} | 20 modules | ~26,000 LOC | C++20</p>

<div class="grid">
  <div class="stat"><div class="stat-num">20</div><div class="stat-label">Modules</div></div>
  <div class="stat"><div class="stat-num">~136</div><div class="stat-label">Source Files</div></div>
  <div class="stat"><div class="stat-num">~26,000</div><div class="stat-label">Lines of Code</div></div>
  <div class="stat"><div class="stat-num">~230</div><div class="stat-label">Classes/Structs</div></div>
</div>

<nav><strong>Modules:</strong> ''')
    for mid, label in modules_nav:
        p(f'<a href="#{mid}">{label}</a> ')
    p('</nav>\n')

    # Architecture overview
    p('''<section>
<h2>Architecture Overview</h2>
<pre><code>+-------------------------------------------------------------------+
|                    Brain19App (REPL + CLI)                         |
|                         main.cpp                                  |
+-------------------------------------------------------------------+
|                   SystemOrchestrator                              |
|    (28 subsystems, lifecycle, ask/ingest/checkpoint)              |
+----------+----------+----------+----------+-----------------------+
| LTM      | STM      | Cognitive| Under-   | Hybrid                |
| (know-   | (activa- | Dynamics | standing | (KanValidator,        |
|  ledge)  |  tion)   | (spread, | Layer    |  EpistemicBridge,     |
|          |          |  focus)  | (MiniLLM)|  RefinementLoop)      |
+----------+----------+----------+----------+-----------------------+
| MicroModel Registry | KAN Modules        | Ingestion Pipeline    |
| (per-concept learn) | (B-spline approx)  | (text-&gt;concepts)      |
+---------------------+--------------------+-----------------------+
| Persistence (WAL, Checkpoints)  |  Streams (parallel thinking)   |
| Concurrent (SharedLTM/STM/...)  |  Bootstrap (foundation seed)   |
+---------------------+----------+--------------------------------+
| Evolution (PatternDiscovery, EpistemicPromotion, ConceptProposer) |
+-------------------------------------------------------------------+</code></pre>

<h3>Key Architectural Invariants</h3>
<div class="invariant"><strong>Epistemic Enforcement:</strong> No default-constructible EpistemicMetadata. Every concept MUST have explicit type + trust. Knowledge is NEVER deleted, only INVALIDATED.</div>
<div class="invariant"><strong>READ-ONLY Boundaries:</strong> CognitiveDynamics, UnderstandingLayer, MiniLLMs -- all READ-ONLY on LTM. Only SystemOrchestrator + IngestionPipeline write to LTM.</div>
<div class="invariant"><strong>HYPOTHESIS Enforcement:</strong> All MiniLLM outputs are HYPOTHESIS. No automatic FACT promotion. THEORY-to-FACT requires human review.</div>
<div class="invariant"><strong>Bounded Values:</strong> Activation in [0,1], Trust in [0,1], Salience in [0,1], Weight in [0,1]. All clamped.</div>
<div class="invariant"><strong>Lock Hierarchy:</strong> LTM (1) &gt; STM (2) &gt; Registry (3) &gt; Embeddings (4). Never acquire lower-numbered lock while holding higher-numbered lock.</div>
<div class="invariant"><strong>Tools Not Agents:</strong> All subsystems are mechanical delegates. No subsystem makes autonomous epistemic decisions.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 1: common
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="common">
<h2>Module: common</h2>
<p><strong>Files:</strong> <code>common/types.hpp</code></p>
<p><strong>Purpose:</strong> Core type aliases shared across all modules.</p>
<p><strong>Dependencies:</strong> None (foundation module)</p>

<h3>Type Aliases</h3>
''' + member_table([
    ("ConceptId", "uint64_t", "Unique identifier for concepts"),
    ("ContextId", "uint64_t", "Unique identifier for STM contexts"),
    ("RelationId", "uint64_t", "Unique identifier for relations"),
]) + '''
<div class="invariant"><strong>Invariant:</strong> All IDs are non-zero when valid. Zero indicates invalid/not-found.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 2: epistemic
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="epistemic">
<h2>Module: epistemic</h2>
<p><strong>Files:</strong> <code>epistemic/epistemic_metadata.hpp</code></p>
<p><strong>Purpose:</strong> Epistemic type system enforcing knowledge certainty classification at compile time.</p>
<p><strong>Dependencies:</strong> None (standard library only)</p>

<h3>Enum: EpistemicType</h3>
<p>Categorization of knowledge certainty. Every knowledge item MUST have an explicit type.</p>
''' + member_table([
    ("FACT", "enum", "Verified, reproducible, high-certainty knowledge"),
    ("DEFINITION", "enum", "Definitional/tautological knowledge"),
    ("THEORY", "enum", "Well-supported, falsifiable claims"),
    ("HYPOTHESIS", "enum", "Testable but not yet strongly supported"),
    ("INFERENCE", "enum", "Derived from other knowledge"),
    ("SPECULATION", "enum", "Low-certainty, exploratory knowledge"),
]) + '''

<h3>Enum: EpistemicStatus</h3>
<p>Lifecycle state of knowledge.</p>
''' + member_table([
    ("ACTIVE", "enum", "Currently valid and usable"),
    ("CONTEXTUAL", "enum", "Valid only in specific contexts"),
    ("SUPERSEDED", "enum", "Replaced by better knowledge (not wrong)"),
    ("INVALIDATED", "enum", "Known to be incorrect (never deleted, only marked)"),
]) + '''

<h3>Struct: EpistemicMetadata ''' + tag("CORE","tag-core") + '''</h3>
<p><strong>Purpose:</strong> MANDATORY metadata for ALL knowledge items. Enforces epistemic explicitness at compile time.</p>

<h4>Member Variables</h4>
''' + member_table([
    ("type", "EpistemicType", "Type of knowledge certainty"),
    ("status", "EpistemicStatus", "Current lifecycle state"),
    ("trust", "double", "Trust level in [0.0, 1.0]"),
]) + '''

<h4>Constructors &amp; Methods</h4>
<details><summary>Show all methods</summary>
<ul>
<li><code>EpistemicMetadata() = delete</code> -- No default constructor. Enforces explicit epistemic decisions at compile time.</li>
<li><code>explicit EpistemicMetadata(EpistemicType t, EpistemicStatus s, double trust_value)</code> -- Full constructor. Validates trust in [0.0, 1.0], throws <code>std::out_of_range</code> if invalid. Debug assertion warns if INVALIDATED has trust &gt;= 0.2.</li>
<li><code>operator=(const EpistemicMetadata&amp;) = default</code> -- Copy assignment (for container ops).</li>
<li><code>operator=(EpistemicMetadata&amp;&amp;) = default</code> -- Move assignment.</li>
<li><code>EpistemicMetadata(const EpistemicMetadata&amp;) = default</code> -- Copy constructor.</li>
<li><code>EpistemicMetadata(EpistemicMetadata&amp;&amp;) = default</code> -- Move constructor.</li>
<li><code>bool is_valid() const</code> -- Returns true if trust in [0.0, 1.0]. O(1).</li>
<li><code>bool is_active() const</code> -- Returns (status == ACTIVE).</li>
<li><code>bool is_invalidated() const</code> -- Returns (status == INVALIDATED).</li>
<li><code>bool is_superseded() const</code> -- Returns (status == SUPERSEDED).</li>
<li><code>bool is_contextual() const</code> -- Returns (status == CONTEXTUAL).</li>
</ul>
</details>

<h4>Helper Functions</h4>
<details><summary>Show helpers</summary>
<ul>
<li><code>EpistemicMetadata create_fact_metadata(double trust)</code> -- Convenience: FACT + ACTIVE.</li>
<li><code>EpistemicMetadata create_hypothesis_metadata(double trust)</code> -- Convenience: HYPOTHESIS + ACTIVE.</li>
<li><code>EpistemicMetadata create_invalidated_metadata(EpistemicType original_type, double trust = 0.05)</code> -- Preserves original type, sets INVALIDATED status with very low trust.</li>
<li><code>std::string epistemic_type_to_string(EpistemicType)</code> -- Returns "FACT", "DEFINITION", etc.</li>
<li><code>std::string epistemic_status_to_string(EpistemicStatus)</code> -- Returns "ACTIVE", "INVALIDATED", etc.</li>
</ul>
</details>

<div class="invariant"><strong>Invariant:</strong> Every knowledge item MUST have EpistemicMetadata. Construction without metadata is IMPOSSIBLE (default constructor deleted). Trust MUST be in [0.0, 1.0]. Knowledge is NEVER deleted, only INVALIDATED.</div>
<div class="weak"><strong>Weak Point:</strong> No thread safety mechanism. No persistence (no serialization). Assert in constructor disabled in release builds (NDEBUG).</div>
<div class="note"><strong>Thread Safety:</strong> Not thread-safe (no synchronization primitives).</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 3: memory (STM)
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="memory">
<h2>Module: memory (Short-Term Memory)</h2>
<p><strong>Files:</strong> <code>memory/activation_level.hpp</code>, <code>memory/active_relation.hpp</code>, <code>memory/stm_entry.hpp</code>, <code>memory/stm.hpp</code>, <code>memory/stm.cpp</code>, <code>memory/brain_controller.hpp</code>, <code>memory/brain_controller.cpp</code></p>
<p><strong>Purpose:</strong> Activation-based working memory. STM stores ONLY activation states, never knowledge content.</p>
<p><strong>Dependencies:</strong> common/types.hpp, epistemic/epistemic_metadata.hpp</p>

<h3>Enum: ActivationLevel</h3>
<p>Read-only classification: LOW (&lt;0.3), MEDIUM (0.3-0.7), HIGH (&gt;=0.7).</p>

<h3>Enum: ActivationClass</h3>
<p>CORE_KNOWLEDGE (decays slower), CONTEXTUAL (decays faster).</p>

<h3>Enum: RelationType</h3>
<p>IS_A, HAS_PROPERTY, CAUSES, ENABLES, PART_OF, SIMILAR_TO, CONTRADICTS, SUPPORTS, TEMPORAL_BEFORE, CUSTOM.</p>

<h3>Struct: ActiveRelation</h3>
''' + member_table([
    ("source", "ConceptId", "Source concept ID"),
    ("target", "ConceptId", "Target concept ID"),
    ("type", "RelationType", "Type of relation"),
    ("activation", "double", "Activation level in [0.0, 1.0]"),
    ("last_used", "time_point", "Timestamp of last access"),
]) + '''

<h3>Struct: STMEntry</h3>
''' + member_table([
    ("concept_id", "ConceptId", "ID of the concept"),
    ("activation", "double", "Activation level in [0.0, 1.0]"),
    ("classification", "ActivationClass", "CORE_KNOWLEDGE or CONTEXTUAL"),
    ("last_used", "time_point", "Timestamp of last access"),
]) + '''

<h3>Class: ShortTermMemory ''' + tag("CORE","tag-core") + '''</h3>
<p>Purely mechanical activation layer. Never stores knowledge, only activation states. Never evaluates correctness.</p>

<h4>Member Variables (Private)</h4>
''' + member_table([
    ("contexts_", "unordered_map<ContextId, Context>", "All managed contexts"),
    ("next_context_id_", "ContextId", "Generator for new context IDs (starts 1)"),
    ("core_decay_rate_", "double", "Exponential decay for core knowledge (default 0.05)"),
    ("contextual_decay_rate_", "double", "Exponential decay for contextual (default 0.15)"),
    ("relation_decay_rate_", "double", "Exponential decay for relations (default 0.25)"),
    ("relation_inactive_threshold_", "double", "Below this: relation inactive (default 0.1)"),
    ("relation_removal_threshold_", "double", "Below this: relation removed (default 0.01)"),
    ("concept_removal_threshold_", "double", "Below this: concept removed (default 0.01)"),
]) + '''

<details><summary>All Methods (25)</summary>
<ul>
<li><code>ContextId create_context()</code> -- Creates new context with unique ID. O(1).</li>
<li><code>void destroy_context(ContextId)</code> -- Removes context and all data. O(n).</li>
<li><code>void clear_context(ContextId)</code> -- Clears contents, context remains. O(n).</li>
<li><code>void activate_concept(ContextId, ConceptId, double, ActivationClass)</code> -- Sets/updates concept activation, clamps [0,1]. O(1).</li>
<li><code>void activate_relation(ContextId, ConceptId src, ConceptId tgt, RelationType, double)</code> -- Sets/updates relation activation. O(1).</li>
<li><code>void boost_concept(ContextId, ConceptId, double delta)</code> -- Adds delta, clamps [0,1]. O(1).</li>
<li><code>void boost_relation(ContextId, ConceptId src, ConceptId tgt, double delta)</code> -- Adds delta. O(1).</li>
<li><code>double get_concept_activation(ContextId, ConceptId) const</code> -- Returns activation or 0.0. O(1).</li>
<li><code>double get_relation_activation(ContextId, ConceptId, ConceptId) const</code> -- Returns activation or 0.0. O(1).</li>
<li><code>ActivationLevel get_concept_level(ContextId, ConceptId) const</code> -- LOW/MEDIUM/HIGH. O(1).</li>
<li><code>vector&lt;ConceptId&gt; get_active_concepts(ContextId, double threshold) const</code> -- All above threshold. O(n).</li>
<li><code>vector&lt;ActiveRelation&gt; get_active_relations(ContextId, double threshold) const</code> -- All above threshold. O(n).</li>
<li><code>void decay_all(ContextId, double time_delta_seconds)</code> -- Exponential decay: factor = exp(-rate * dt). Removes below threshold. O(n).</li>
<li><code>void set_core_decay_rate(double)</code> / <code>set_contextual_decay_rate(double)</code> / <code>set_relation_decay_rate(double)</code></li>
<li><code>void set_relation_inactive_threshold(double)</code> / <code>set_relation_removal_threshold(double)</code> / <code>set_concept_removal_threshold(double)</code></li>
<li><code>STMSnapshotData export_state() const</code> -- Exports all contexts to snapshot. O(n).</li>
<li><code>void import_state(const STMSnapshotData&amp;)</code> -- Imports from snapshot. O(n).</li>
</ul>
</details>

<h3>Class: BrainController</h3>
<p>Minimal orchestration layer. DOES NOT learn, reason, evaluate, or decide importance. All major decisions delegated to callers.</p>
''' + member_table([
    ("stm_", "unique_ptr<ShortTermMemory>", "Owned STM instance"),
    ("initialized_", "bool", "Initialization flag"),
    ("thinking_states_", "unordered_map<ContextId, ThinkingState>", "Per-context thinking state"),
]) + '''
<details><summary>Methods</summary>
<ul>
<li><code>bool initialize()</code> -- Creates STM. Idempotent.</li>
<li><code>void shutdown()</code> -- Clears thinking states, resets STM.</li>
<li><code>ContextId create_context()</code> -- Creates context in STM.</li>
<li><code>void begin_thinking(ContextId)</code> / <code>end_thinking(ContextId)</code> -- Lifecycle tracking.</li>
<li><code>void activate_concept_in_context(...)</code> -- Delegates to STM.</li>
<li><code>void decay_context(ContextId, double)</code> -- Delegates to STM.</li>
<li><code>const ShortTermMemory* get_stm() const</code> / <code>ShortTermMemory* get_stm_mutable()</code></li>
</ul>
</details>

<div class="invariant"><strong>Invariant:</strong> STM never stores knowledge, only activation states. STM never evaluates correctness, trust, or importance. All activation decisions are caller-decided. Decay only on explicit call.</div>
<div class="weak"><strong>Weak Point:</strong> No thread safety. Relation hash could theoretically collide (very unlikely with hash_combine). No persistence except snapshot export/import.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 4: ltm
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="ltm">
<h2>Module: ltm (Long-Term Memory)</h2>
<p><strong>Files:</strong> <code>ltm/long_term_memory.hpp</code>, <code>ltm/long_term_memory.cpp</code>, <code>ltm/relation.hpp</code></p>
<p><strong>Purpose:</strong> Persistent knowledge storage. Enforces epistemic integrity -- knowledge is NEVER deleted, only INVALIDATED.</p>

<h3>Struct: ConceptInfo ''' + tag("CORE","tag-core") + '''</h3>
''' + member_table([
    ("id", "ConceptId", "Unique concept identifier"),
    ("label", "std::string", "Human-readable concept name"),
    ("definition", "std::string", "Formal definition"),
    ("epistemic", "EpistemicMetadata", "REQUIRED epistemic metadata (no default)"),
]) + '''
<p><code>ConceptInfo() = delete</code> -- No default constructor. Enforces epistemic explicitness at compile time.</p>

<h3>Struct: RelationInfo</h3>
''' + member_table([
    ("id", "RelationId", "Unique identifier"),
    ("source", "ConceptId", "Source concept (from)"),
    ("target", "ConceptId", "Target concept (to)"),
    ("type", "RelationType", "Type of relation"),
    ("weight", "double", "Spreading activation strength [0.0, 1.0]"),
]) + '''
<p>Relations are DIRECTED. Relations are NOT epistemic entities (no trust/type).</p>

<h3>Class: LongTermMemory ''' + tag("CORE","tag-core") + '''</h3>
''' + member_table([
    ("concepts_", "unordered_map<ConceptId, ConceptInfo>", "Persistent concept storage"),
    ("next_concept_id_", "ConceptId", "ID generator (starts 1)"),
    ("relations_", "unordered_map<RelationId, RelationInfo>", "Persistent relation storage"),
    ("outgoing_relations_", "unordered_map<ConceptId, vector<RelationId>>", "Outgoing edge index"),
    ("incoming_relations_", "unordered_map<ConceptId, vector<RelationId>>", "Incoming edge index"),
    ("next_relation_id_", "RelationId", "Relation ID generator (starts 1)"),
    ("total_relations_", "size_t", "Cached count of all relations"),
]) + '''

<details><summary>All Methods (19)</summary>
<ul>
<li><code>ConceptId store_concept(const string&amp; label, const string&amp; definition, EpistemicMetadata epistemic)</code> -- Stores with REQUIRED epistemic. O(1).</li>
<li><code>optional&lt;ConceptInfo&gt; retrieve_concept(ConceptId) const</code> -- Returns concept or nullopt. O(1).</li>
<li><code>bool exists(ConceptId) const</code> -- O(1).</li>
<li><code>bool update_epistemic_metadata(ConceptId, EpistemicMetadata)</code> -- Atomic: creates new then move-assigns. O(1).</li>
<li><code>bool invalidate_concept(ConceptId, double trust = 0.05)</code> -- NEVER deletes. Original type preserved. Status set to INVALIDATED. O(1).</li>
<li><code>vector&lt;ConceptId&gt; get_concepts_by_type(EpistemicType) const</code> -- O(n).</li>
<li><code>vector&lt;ConceptId&gt; get_concepts_by_status(EpistemicStatus) const</code> -- O(n).</li>
<li><code>vector&lt;ConceptId&gt; get_active_concepts() const</code> -- All ACTIVE. O(n).</li>
<li><code>RelationId add_relation(ConceptId src, ConceptId tgt, RelationType, double weight = 1.0)</code> -- Returns 0 if either concept missing. O(1).</li>
<li><code>optional&lt;RelationInfo&gt; get_relation(RelationId) const</code> -- O(1).</li>
<li><code>vector&lt;RelationInfo&gt; get_outgoing_relations(ConceptId) const</code> / <code>get_incoming_relations(ConceptId) const</code> -- O(m).</li>
<li><code>vector&lt;RelationInfo&gt; get_relations_between(ConceptId, ConceptId) const</code> -- O(m).</li>
<li><code>bool remove_relation(RelationId)</code> -- Removes and updates indices. O(m).</li>
<li><code>size_t get_relation_count(ConceptId) const</code> -- in + out count. O(1).</li>
<li><code>vector&lt;ConceptId&gt; get_all_concept_ids() const</code> -- O(n).</li>
</ul>
</details>

<div class="invariant"><strong>Invariant:</strong> Every concept has valid EpistemicMetadata. Every relation has valid source/target. INVALIDATED concepts remain in storage forever. Outgoing/incoming indices always consistent.</div>
<div class="weak"><strong>Weak Point:</strong> No thread safety. No persistence mechanism (see persistent module). Removing relations is O(n) in concept's relation count. No batch operations or transaction support.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 5: micromodel
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="micromodel">
<h2>Module: micromodel</h2>
<p><strong>Files:</strong> <code>micromodel/micro_model.hpp/.cpp</code>, <code>micro_model_registry.hpp/.cpp</code>, <code>embedding_manager.hpp/.cpp</code>, <code>micro_trainer.hpp/.cpp</code>, <code>relevance_map.hpp/.cpp</code>, <code>persistence.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Bilinear micro-model system for personalized relevance prediction. Math: v = W*c + b (10D), z = e^T*v (scalar), w = sigma(z) in (0,1). Adam optimizer.</p>

<h3>Constants</h3>
<p><code>EMBED_DIM = 10</code>, <code>FLAT_SIZE = 430</code> (100 W + 10 b + 10 e_init + 10 c_init + 300 training state).</p>
<p>Type aliases: <code>Vec10 = array&lt;double,10&gt;</code>, <code>Mat10x10 = array&lt;double,100&gt;</code> (row-major).</p>

<h3>Struct: MicroTrainingConfig</h3>
''' + member_table([
    ("learning_rate", "double", "Adam learning rate (default 0.01)"),
    ("max_epochs", "size_t", "Maximum training epochs (default 100)"),
    ("convergence_threshold", "double", "Loss improvement threshold (default 1e-6)"),
    ("adam_beta1", "double", "First moment decay (default 0.9)"),
    ("adam_beta2", "double", "Second moment decay (default 0.999)"),
    ("adam_epsilon", "double", "Numerical stability (default 1e-8)"),
]) + '''

<h3>Class: MicroModel ''' + tag("CORE","tag-core") + '''</h3>
''' + member_table([
    ("W_", "Mat10x10", "Weight matrix (100 doubles, near-identity init)"),
    ("b_", "Vec10", "Bias vector (10 doubles)"),
    ("e_init_", "Vec10", "Default relation embedding"),
    ("c_init_", "Vec10", "Default context embedding"),
    ("state_", "TrainingState", "Adam optimizer state (300 doubles)"),
]) + '''
<details><summary>Methods</summary>
<ul>
<li><code>double predict(const Vec10&amp; e, const Vec10&amp; c)</code> -- Forward pass: v = W*c + b, z = dot(e,v), return sigmoid(z).</li>
<li><code>double train_step(const Vec10&amp; e, const Vec10&amp; c, double target, const Config&amp;)</code> -- Single Adam step. Returns MSE loss.</li>
<li><code>MicroTrainingResult train(const vector&lt;TrainingSample&gt;&amp;, const Config&amp;)</code> -- Multi-epoch batch training with convergence detection.</li>
<li><code>void to_flat(array&lt;double,430&gt;&amp; out)</code> -- Serialize to 430-double array.</li>
<li><code>void from_flat(const array&lt;double,430&gt;&amp; in)</code> -- Deserialize from 430-double array.</li>
</ul>
</details>

<h3>Class: MicroModelRegistry</h3>
<p>1:1 mapping from ConceptIds to MicroModels.</p>
<details><summary>Methods</summary>
<ul>
<li><code>bool create_model(ConceptId)</code> -- Create new default MicroModel.</li>
<li><code>MicroModel* get_model(ConceptId)</code> -- Non-owning pointer, nullptr if missing.</li>
<li><code>bool has_model(ConceptId) const</code> / <code>bool remove_model(ConceptId)</code></li>
<li><code>size_t ensure_models_for(const LongTermMemory&amp;)</code> -- Bulk-create for all LTM concepts.</li>
</ul>
</details>

<h3>Class: EmbeddingManager</h3>
<p>Fixed 10 relation type embeddings (heuristic 10D vectors). Dynamic context embeddings from deterministic hash.</p>
<p>Embedding dimensions: 0=hierarchical, 1=causal, 2=compositional, 3=similarity, 4=temporal, 5=support/opposition, 6=specificity, 7=directionality, 8=abstractness, 9=strength.</p>

<h3>Class: MicroTrainer</h3>
<p>Generates training data from KG structure. Positives from relations (incoming discounted 0.8x). Negatives from non-connected concepts (3:1 ratio, target 0.05).</p>

<h3>Class: RelevanceMap</h3>
<p>Evaluates micro-model over all KG nodes. Supports overlay ops: ADDITION, MAX, WEIGHTED_AVERAGE for combining perspectives.</p>

<h3>Persistence (brain19::persistence namespace)</h3>
<p>Binary "BM19" format. Header (32B): magic + version + model_count + context_count. Per model (3448B): concept_id + 430 doubles. XOR checksum footer.</p>
<details><summary>Functions</summary>
<ul>
<li><code>bool save(filepath, registry, embeddings)</code> -- Serialize to file.</li>
<li><code>bool load(filepath, registry, embeddings)</code> -- Deserialize with checksum verification.</li>
<li><code>bool validate(filepath)</code> -- Verify integrity without loading.</li>
</ul>
</details>

<div class="invariant"><strong>Invariant:</strong> One model per concept ID. FLAT_SIZE=430 doubles per model fixed. All arrays properly sized. Weights remain finite. Timestep monotonically increasing.</div>
<div class="weak"><strong>Weak Point:</strong> No NaN/Inf checking during training. No thread safety. XOR checksum only detects bit flips, not block insertion/deletion. Context name max 1024 bytes hardcoded. No file locking.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 6: kan
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="kan">
<h2>Module: kan (Kolmogorov-Arnold Networks)</h2>
<p><strong>Files:</strong> <code>kan/kan_node.hpp/.cpp</code>, <code>kan_layer.hpp/.cpp</code>, <code>kan_module.hpp/.cpp</code>, <code>function_hypothesis.hpp</code></p>
<p><strong>Purpose:</strong> B-spline-based function approximation. KANNode (univariate) -> KANLayer (n_in x n_out grid) -> KANModule (multi-layer chain).</p>

<h3>Class: KANNode</h3>
<p>Univariate learnable function using cubic B-spline basis. Input clamped to [0,1]. Cox-de Boor recursive evaluation.</p>
''' + member_table([
    ("num_knots_", "size_t", "Number of knots (minimum 4)"),
    ("knots_", "vector<double>", "Uniform knot vector, size = num_knots_ + 4"),
    ("coefficients_", "vector<double>", "Spline coefficients, size = num_knots_"),
]) + '''

<h3>Class: KANLayer</h3>
<p>n_in x n_out grid of KANNodes. output_j = sum_i phi_{i,j}(input_i). Row-major indexing.</p>

<h3>Class: KANModule</h3>
<p>Multi-layer KAN chain. Supports arbitrary topology [input_dim, hidden1, ..., output_dim]. Training via gradient descent with finite-difference backprop.</p>
''' + member_table([
    ("input_dim_", "size_t", "Input dimension"),
    ("output_dim_", "size_t", "Output dimension"),
    ("layer_dims_", "vector<size_t>", "Full topology"),
    ("layers_", "vector<unique_ptr<KANLayer>>", "Chain of layers"),
]) + '''

<h3>Struct: FunctionHypothesis</h3>
<p>Pure data wrapper: metadata about a learned function. NO logic.</p>
''' + member_table([
    ("input_dim", "size_t", "Input dimension"),
    ("output_dim", "size_t", "Output dimension"),
    ("module", "shared_ptr<KANModule>", "Trained KAN module"),
    ("training_iterations", "size_t", "Iterations performed"),
    ("training_error", "double", "Final MSE loss"),
    ("created_at", "time_point", "Creation timestamp"),
]) + '''

<div class="invariant"><strong>Invariant:</strong> knots_.size() = num_knots_ + 4. coefficients_.size() = num_knots_. B-spline partition of unity. layers_.size() = layer_dims_.size() - 1.</div>
<div class="weak"><strong>Weak Point:</strong> Gradient computation uses finite differences (1e-6 epsilon). No batch normalization or regularization. No learning rate scheduling. Cox-de Boor 1e-10 threshold may be insufficient.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 7: adapter
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="adapter">
<h2>Module: adapter</h2>
<p><strong>Files:</strong> <code>adapter/kan_adapter.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Clean interface between BrainController and KAN system. Explicit delegation, NO decision logic. Module lifecycle with uint64_t IDs.</p>

<h3>Class: KANAdapter</h3>
''' + member_table([
    ("modules_", "unordered_map<uint64_t, KANModuleEntry>", "Registry of created modules"),
    ("next_module_id_", "uint64_t", "Auto-incrementing ID counter (starts 1)"),
]) + '''
<details><summary>Methods</summary>
<ul>
<li><code>uint64_t create_kan_module(size_t input_dim, size_t output_dim, size_t num_knots = 10)</code></li>
<li><code>uint64_t create_kan_module_multilayer(const vector&lt;size_t&gt;&amp; layer_dims, size_t num_knots = 10)</code></li>
<li><code>unique_ptr&lt;FunctionHypothesis&gt; train_kan_module(uint64_t id, const vector&lt;DataPoint&gt;&amp;, const KanTrainingConfig&amp;)</code></li>
<li><code>vector&lt;double&gt; evaluate_kan_module(uint64_t id, const vector&lt;double&gt;&amp; inputs) const</code></li>
<li><code>void destroy_kan_module(uint64_t)</code> / <code>bool has_module(uint64_t) const</code></li>
</ul>
</details>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 8: cognitive
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="cognitive">
<h2>Module: cognitive</h2>
<p><strong>Files:</strong> <code>cognitive/cognitive_config.hpp</code>, <code>cognitive_dynamics.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Spreading activation, salience computation, focus management, thought path ranking. READ-ONLY on LTM, writes only to STM.</p>

<h3>Config Structs</h3>
<p><strong>ActivationSpreaderConfig:</strong> max_depth=3, damping=0.8, threshold=0.01, trust_weighted=true, relation_weighted=true.</p>
<p><strong>FocusManagerConfig:</strong> max_focus_size=7 (Miller's number), decay_rate=0.1, attention_boost=0.3.</p>
<p><strong>SalienceComputerConfig:</strong> activation_weight=0.4, trust_weight=0.3, connectivity_weight=0.2, recency_weight=0.1, query_boost_factor=0.5.</p>
<p><strong>ThoughtPathConfig:</strong> max_paths=10, max_depth=5, depth_penalty=0.1. Weights: salience=0.5, trust=0.3, coherence=0.2.</p>

<h3>Data Structs</h3>
<p><code>ActivationEntry</code>, <code>FocusEntry</code>, <code>SalienceScore</code> (with factor breakdown), <code>ThoughtPathNode</code>, <code>ThoughtPath</code>, <code>SpreadingStats</code>.</p>

<h3>Class: CognitiveDynamics ''' + tag("READ-ONLY","tag-ro") + '''</h3>
''' + member_table([
    ("config_", "CognitiveDynamicsConfig", "Configuration parameters"),
    ("stats_", "Stats", "Thread-safe statistics (atomic counters)"),
    ("focus_sets_", "unordered_map<ContextId, vector<FocusEntry>>", "Per-context focus management"),
    ("current_tick_", "uint64_t", "Logical time counter for recency"),
]) + '''

<details><summary>Spreading Activation</summary>
<ul>
<li><code>SpreadingStats spread_activation(ConceptId source, double initial, ContextId, const LTM&amp;, STM&amp;)</code> -- Trust-weighted, depth-limited, cycle-safe recursive spreading. Visited set prevents re-traversal. Propagated = activation * weight * trust * damping^(depth+1).</li>
<li><code>SpreadingStats spread_activation_multi(const vector&lt;ConceptId&gt;&amp;, ...)</code> -- Multi-source spreading with shared visited set.</li>
</ul>
</details>

<details><summary>Salience Computation</summary>
<ul>
<li><code>SalienceScore compute_salience(ConceptId, ContextId, LTM, STM, tick)</code> -- Multi-factor: activation, trust, connectivity, recency. Weighted sum, clamped [0, max_salience].</li>
<li><code>vector&lt;SalienceScore&gt; compute_salience_batch(...)</code> -- Batch with max_connectivity normalization.</li>
<li><code>vector&lt;SalienceScore&gt; compute_query_salience(...)</code> -- With query boost for direct/adjacent matches.</li>
</ul>
</details>

<details><summary>Focus Management</summary>
<ul>
<li><code>void init_focus(ContextId)</code> / <code>void focus_on(ContextId, ConceptId, double boost)</code></li>
<li><code>void decay_focus(ContextId)</code> -- score *= (1.0 - decay_rate). Prunes below threshold.</li>
<li><code>void clear_focus(ContextId)</code> / <code>vector&lt;FocusEntry&gt; get_focus_set(ContextId)</code></li>
</ul>
</details>

<details><summary>Thought Path Ranking</summary>
<ul>
<li><code>vector&lt;ThoughtPath&gt; find_best_paths(ConceptId source, ContextId, LTM, STM)</code> -- Beam search, max_paths width.</li>
<li><code>vector&lt;ThoughtPath&gt; find_paths_to(ConceptId source, ConceptId target, ...)</code> -- Target-directed beam search.</li>
<li><code>double score_path(const ThoughtPath&amp;, ContextId, LTM, STM)</code> -- path_score = (w_sal*avg_sal + w_trust*avg_trust + w_coher) * depth_penalty^length.</li>
</ul>
</details>

<div class="invariant"><strong>Invariant:</strong> All activation in [min, max]. All salience in [0, max_salience]. focus_sets size &lt;= max_focus_size. Depth &lt;= max_depth. Visited set prevents cycles.</div>
<div class="weak"><strong>Weak Point:</strong> focus_sets_ grows forever (no context cleanup). recency_contrib O(max_focus_size) per concept. expand_paths() copies entire paths. Coherence hardcoded to 1.0.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 9: curiosity
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="curiosity">
<h2>Module: curiosity</h2>
<p><strong>Files:</strong> <code>curiosity/curiosity_trigger.hpp</code>, <code>curiosity_engine.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Pure signal generator. Observes system state, emits CuriosityTriggers. NO actions, NO learning.</p>

<h3>Enum: TriggerType</h3>
<p>SHALLOW_RELATIONS, MISSING_DEPTH, LOW_EXPLORATION, RECURRENT_WITHOUT_FUNCTION, UNKNOWN.</p>

<h3>Struct: CuriosityTrigger</h3>
''' + member_table([
    ("type", "TriggerType", "Classification of trigger"),
    ("context_id", "ContextId", "Context where detected"),
    ("related_concept_ids", "vector<ConceptId>", "Concepts involved"),
    ("description", "std::string", "Human-readable observation"),
]) + '''

<h3>Class: CuriosityEngine</h3>
<p>Stateless pattern detection: shallow_relation_ratio (default 0.3), low_exploration_min_concepts (default 5).</p>
<details><summary>Methods</summary>
<ul>
<li><code>vector&lt;CuriosityTrigger&gt; observe_and_generate_triggers(const vector&lt;SystemObservation&gt;&amp;)</code> -- Detects SHALLOW_RELATIONS and LOW_EXPLORATION patterns.</li>
</ul>
</details>
<div class="weak"><strong>Weak Point:</strong> Stateless (no temporal pattern detection). RECURRENT_WITHOUT_FUNCTION never generated. Hardcoded thresholds.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 10: understanding
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="understanding">
<h2>Module: understanding</h2>
<p><strong>Files:</strong> <code>understanding/understanding_proposals.hpp</code>, <code>mini_llm.hpp/.cpp</code>, <code>ollama_mini_llm.hpp/.cpp</code>, <code>mini_llm_factory.hpp</code>, <code>understanding_layer.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Semantic analysis layer using Mini-LLMs. All outputs are HYPOTHESIS. READ-ONLY LTM access.</p>

<h3>Proposal Structs</h3>
<p><strong>MeaningProposal:</strong> Semantic interpretation. epistemic_type ALWAYS HYPOTHESIS (immutable).</p>
<p><strong>HypothesisProposal:</strong> Pattern-based hypothesis. SuggestedEpistemic enforces HYPOTHESIS type.</p>
<p><strong>AnalogyProposal:</strong> Structural similarity detection (not semantic).</p>
<p><strong>ContradictionProposal:</strong> Inconsistency marker. Understanding marks; Epistemic Core resolves.</p>

<h3>Abstract Class: MiniLLM ''' + tag("READ-ONLY","tag-ro") + '''</h3>
<p>CAN: interpret texts, detect patterns, generate proposals. CANNOT: modify KG, set trust, make epistemic decisions, promote FACTs.</p>
<details><summary>Pure Virtual Methods</summary>
<ul>
<li><code>string get_model_id() const</code></li>
<li><code>vector&lt;MeaningProposal&gt; extract_meaning(active_concepts, ltm, stm, context) const</code></li>
<li><code>vector&lt;HypothesisProposal&gt; generate_hypotheses(evidence, ltm, stm, context) const</code></li>
<li><code>vector&lt;AnalogyProposal&gt; detect_analogies(set_a, set_b, ltm, stm, context) const</code></li>
<li><code>vector&lt;ContradictionProposal&gt; detect_contradictions(active, ltm, stm, context) const</code></li>
</ul>
</details>

<h3>Class: StubMiniLLM</h3>
<p>Placeholder for testing. Conservative confidence (0.3-0.5). CRITICAL: throws if proposal not HYPOTHESIS.</p>

<h3>Class: OllamaMiniLLM ''' + tag("EXTERNAL","tag-ext") + '''</h3>
<p>Real semantic analysis via Ollama API. Confidence estimated from hedging/certainty language markers.</p>

<h3>Class: UnderstandingLayer</h3>
<p>Aggregates proposals from registered Mini-LLMs. Filters by confidence thresholds. Rate-limited (max_proposals_per_cycle).</p>

<div class="invariant"><strong>Invariant:</strong> All proposals HYPOTHESIS-typed. All LTM access read-only. No state modification. Deterministic aggregation.</div>
<div class="weak"><strong>Weak Point:</strong> Simple string-based parsing for Ollama responses (brittle). No retry logic. proposal_counter_ not thread-safe.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 11: hybrid
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="hybrid">
<h2>Module: hybrid</h2>
<p><strong>Files:</strong> <code>hybrid/kan_validator.hpp/.cpp</code>, <code>epistemic_bridge.hpp/.cpp</code>, <code>hypothesis_translator.hpp/.cpp</code>, <code>domain_manager.hpp/.cpp</code>, <code>refinement_loop.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> KAN-LLM hybrid validation pipeline. Translates LLM hypotheses to KAN-testable form, validates via function approximation, maps results to epistemic trust scores.</p>

<h3>Key Classes</h3>
<p><strong>KanValidator:</strong> Validates hypothesis proposals using KAN function approximation.</p>
<p><strong>EpistemicBridge:</strong> Maps KAN training results to epistemic trust scores.</p>
<p><strong>HypothesisTranslator:</strong> Converts LLM hypotheses to KAN-testable DataPoint format.</p>
<p><strong>DomainManager:</strong> Manages domain-specific KAN modules and routing.</p>
<p><strong>RefinementLoop:</strong> Iterative hypothesis refinement cycle.</p>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 12: ingestor
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="ingestor">
<h2>Module: ingestor</h2>
<p><strong>Files:</strong> <code>ingestor/text_chunker.hpp/.cpp</code>, <code>entity_extractor.hpp/.cpp</code>, <code>relation_extractor.hpp/.cpp</code>, <code>trust_tagger.hpp/.cpp</code>, <code>proposal_queue.hpp/.cpp</code>, <code>knowledge_ingestor.hpp/.cpp</code>, <code>ingestion_pipeline.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Complete knowledge ingestion pipeline: Input -> Chunking -> Entity/Relation Extraction -> Trust Tagging -> ProposalQueue -> Human Review -> LTM.</p>

<h3>Pipeline Architecture</h3>
<pre><code>Input (JSON/CSV/Text)
  |
KnowledgeIngestor (structured) / TextChunker (plain text)
  |
EntityExtractor + RelationExtractor
  |
TrustTagger (assigns trust suggestions)
  |
ProposalQueue (staging area)
  |
[Human Review] (approve/reject/modify)
  |
LTM.store_concept() + LTM.add_relation()</code></pre>

<h3>Class: TextChunker</h3>
<p>Sentence-based chunking with overlap. Config: sentences_per_chunk=3, overlap=1, max_chunk_chars=2000.</p>

<h3>Class: EntityExtractor</h3>
<p>4 extraction strategies: capitalized phrases, quoted terms, definition patterns ("X is a ..."), frequent domain terms. 50+ English stopwords filtered.</p>

<h3>Class: RelationExtractor</h3>
<p>12 regex patterns covering 9 relation types. Entity-pair specific extraction with 22 keywords. Confidence boosted 1.3x for known entity matches.</p>

<h3>Class: TrustTagger</h3>
<p>Maps trust categories to epistemic system. Heuristic text confidence: +0.2 certainty language, +0.15 citations, -0.25 hedging.</p>
<p>Trust ranges: FACTS [0.95,0.99], DEFINITIONS [0.90,0.99], THEORIES [0.85,0.95], HYPOTHESES [0.50,0.80], SPECULATION [0.10,0.40].</p>

<h3>Class: ProposalQueue</h3>
<p>Staging area. Status lifecycle: PENDING -> APPROVED/REJECTED/MODIFIED/EXPIRED. Linear search O(n). Auto-approve and batch operations supported.</p>

<h3>Class: KnowledgeIngestor</h3>
<p>Hand-written JSON/CSV parsing (no external library). Scope-aware JSON extraction. CSV with quoted field support.</p>

<h3>Class: IngestionPipeline ''' + tag("CORE","tag-core") + '''</h3>
<p>Connects all components. Pipeline NEVER writes to LTM without ProposalQueue. All trust assignments are SUGGESTIONS. Pipeline is ADDITIVE ONLY.</p>

<div class="invariant"><strong>Invariant:</strong> Nothing enters LTM without going through ProposalQueue. All trust assignments are suggestions until review. Existing LTM data NEVER modified. New concepts get new IDs. Pipeline is additive only.</div>
<div class="weak"><strong>Weak Point:</strong> English-only NLP. Heuristic patterns. No negation handling. Confidence values arbitrary. JSON parser fragile.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 13: importers
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="importers">
<h2>Module: importers</h2>
<p><strong>Files:</strong> <code>importers/http_client.hpp/.cpp</code>, <code>wikipedia_importer.hpp/.cpp</code>, <code>scholar_importer.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> External data importers. WikipediaImporter fetches and cleans Wikipedia articles. ScholarImporter fetches Google Scholar data.</p>

<h3>Class: HttpClient ''' + tag("EXTERNAL","tag-ext") + '''</h3>
<p>CURL-based HTTP client. GET/POST methods. Timeout: 30 seconds default.</p>

<h3>Class: WikipediaImporter</h3>
<p>Fetches Wikipedia REST API. Cleans HTML: strips tags, decodes entities. Extracts plain text content.</p>

<h3>Class: ScholarImporter</h3>
<p>Fetches Google Scholar search results. Parses title, authors, year, abstract. Creates THEORY-trust proposals.</p>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 14: bootstrap
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="bootstrap">
<h2>Module: bootstrap</h2>
<p><strong>Files:</strong> <code>bootstrap/foundation_concepts.hpp/.cpp</code>, <code>context_accumulator.hpp/.cpp</code>, <code>bootstrap_interface.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Solves bootstrap "chicken-and-egg" problem by pre-seeding LTM with 500+ high-trust foundation concepts.</p>

<h3>Class: FoundationConcepts (Static)</h3>
<p>4 tiers of seeding:</p>
<ul>
<li><strong>Tier 1 (48 concepts):</strong> Ontology -- Entity, Object, Action, Property, Relation, Event, Knowledge, Definition, Fact. All DEFINITION trust 0.97-0.99.</li>
<li><strong>Tier 2 (100 concepts):</strong> Categories -- Organism, Person, Place, Country, Science, Physics, Computer, Algorithm. All trust &gt;= 0.95.</li>
<li><strong>Tier 3 (145 relations):</strong> IS_A, PART_OF, HAS_PROPERTY, CAUSES, ENABLES, SUPPORTS, SIMILAR_TO, TEMPORAL_BEFORE chains.</li>
<li><strong>Tier 4 (150 concepts):</strong> Science -- Atom, Molecule, Cell, DNA, Integer, Function, Turing_Machine. All trust &gt;= 0.95.</li>
</ul>

<h3>Class: ContextAccumulator</h3>
<p>Tracks domain coverage across 10 domains. Finds knowledge gaps (coverage &lt; 0.3). Keyword-based domain classification.</p>

<h3>Class: BootstrapInterface</h3>
<p>Orchestrates: seed foundation -> extract entities -> propose to human -> accumulate feedback.</p>

<div class="invariant"><strong>Invariant:</strong> All foundation concepts ACTIVE status, trust &gt;= 0.95. Relations reference only concepts in same batch. Thread-local label map prevents race conditions.</div>
<div class="weak"><strong>Weak Point:</strong> 500+ hardcoded definitions. No mechanism to update after seeding. Simple heuristic entity extraction.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 15: llm
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="llm">
<h2>Module: llm</h2>
<p><strong>Files:</strong> <code>llm/ollama_client.hpp/.cpp</code>, <code>chat_interface.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> LLM integration via Ollama REST API. ChatInterface is a TOOL (read-only), not an AGENT.</p>

<h3>Class: OllamaClient ''' + tag("EXTERNAL","tag-ext") + '''</h3>
''' + member_table([
    ("config_", "OllamaConfig", "host, model, temperature, num_predict"),
    ("initialized_", "bool", "Flag for initialize() call"),
]) + '''
<p>CURL-based HTTP client. call_once global init. 5-minute timeout. JSON parsing via nlohmann::json.</p>

<h3>Class: ChatInterface</h3>
<p>LLM-powered verbalization. CRITICAL: LLM is read-only tool. Receives concepts+epistemic metadata, returns natural language preserving epistemic integrity.</p>
<details><summary>Methods</summary>
<ul>
<li><code>ChatResponse ask(const string&amp; question, const LTM&amp;)</code> -- Keyword-match relevant concepts. Build system prompt + epistemic context. Fallback mode without LLM.</li>
<li><code>ChatResponse ask_with_context(question, ltm, salient_concepts, thought_paths)</code> -- With pre-computed thinking results.</li>
<li><code>string explain_concept(ConceptId, const LTM&amp;)</code> -- Formatted concept explanation.</li>
<li><code>string compare(ConceptId, ConceptId, const LTM&amp;)</code> -- Side-by-side comparison.</li>
</ul>
</details>

<div class="invariant"><strong>Invariant:</strong> LLM never modifies LTM. Epistemic metadata always included in prompts. Fallback mode available without Ollama.</div>
<div class="weak"><strong>Weak Point:</strong> Hardcoded German system prompt. Naive keyword matching O(n). No prompt caching. No retry logic.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 16: persistent
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="persistent">
<h2>Module: persistent</h2>
<p><strong>Files:</strong> <code>persistent/persistent_records.hpp</code>, <code>persistent_store.hpp</code>, <code>string_pool.hpp</code>, <code>wal.hpp/.cpp</code>, <code>persistent_ltm.hpp/.cpp</code>, <code>stm_snapshot.hpp/.cpp</code>, <code>checkpoint_manager.hpp/.cpp</code>, <code>checkpoint_restore.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> mmap-backed persistence with WAL (Write-Ahead Log) for crash recovery.</p>

<h3>Record Structs</h3>
<p><strong>PersistentConceptRecord</strong> (128 bytes, 2 cache lines): concept_id, label/def offsets into StringPool, epistemic_type/status/trust, timestamps.</p>
<p><strong>PersistentRelationRecord</strong> (64 bytes, 1 cache line): relation_id, source, target, type, weight.</p>

<h3>Class Template: PersistentStore&lt;RecordT&gt;</h3>
<p>Generic mmap-backed growable record store. File lifecycle, growth via mremap, header management.</p>
''' + member_table([
    ("filepath_", "string", "Path to .dat file"),
    ("fd_", "int", "File descriptor"),
    ("mapped_", "void*", "mmap pointer"),
    ("mapped_size_", "size_t", "Current mmap size"),
]) + '''

<h3>Class: StringPool</h3>
<p>Append-only mmap-backed string storage. 4GB limit (uint32_t offsets). No deduplication. No compaction.</p>

<h3>WAL System</h3>
<p><strong>WALOpType:</strong> STORE_CONCEPT(1), ADD_RELATION(2), REMOVE_RELATION(3), INVALIDATE_CONCEPT(4), UPDATE_METADATA(5).</p>
<p><strong>WALEntryHeader</strong> (32 bytes): magic "WL19", sequence_number, operation, payload_size, CRC32 checksum.</p>
<p><strong>CRC32:</strong> FIPS polynomial 0xEDB88320. Lazy-initialized 256-entry lookup table.</p>

<h3>Class: WALWriter</h3>
<p>Append-only with atomic writes via writev(2). fsync after every write. Sequence recovery on open.</p>

<h3>Class: WALRecovery</h3>
<p>Replays WAL entries after crash. IDEMPOTENT: skips entries already applied. Stops on first corrupt entry.</p>

<h3>Class: PersistentLTM ''' + tag("CORE","tag-core") + '''</h3>
<p>mmap-backed LTM. Same API as LongTermMemory but with crash recovery.</p>
<p>Write pattern: WAL log BEFORE mmap write. In-memory indices rebuilt on load.</p>

<h3>Checkpoint System</h3>
<p><strong>CheckpointManager:</strong> Creates tagged checkpoints. Saves STM snapshots + store sync.</p>
<p><strong>CheckpointRestore:</strong> Restores STM from checkpoint. Validates data integrity.</p>

<div class="invariant"><strong>Invariant:</strong> WAL entries logged before mmap write. Recovery is idempotent. next_id never decreases. File always has valid header. Records contiguous after header.</div>
<div class="weak"><strong>Weak Point:</strong> No concurrent access (single-threaded). WAL grows unbounded until checkpoint. 4GB StringPool limit. No record-level locking. CRC32 is weak vs cryptographic hash.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 17: concurrent
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="concurrent">
<h2>Module: concurrent</h2>
<p><strong>Files:</strong> <code>concurrent/shared_ltm.hpp</code>, <code>shared_stm.hpp</code>, <code>shared_registry.hpp</code>, <code>shared_embeddings.hpp</code>, <code>lock_hierarchy.hpp</code>, <code>deadlock_detector.hpp</code></p>
<p><strong>Purpose:</strong> Thread-safe wrappers for all shared subsystems. Lock hierarchy enforcement.</p>

<h3>Lock Hierarchy</h3>
<p>LTM (1) &gt; STM (2) &gt; Registry (3) &gt; Embeddings (4). Never acquire lower-numbered lock while holding higher-numbered. Debug builds throw on violation.</p>

<h3>Class: SharedLTM ''' + tag("READ-WRITE","tag-rw") + '''</h3>
<p>shared_mutex wrapper. Multiple readers, single writer. All write ops use unique_lock, reads use shared_lock.</p>

<h3>Class: SharedSTM ''' + tag("READ-WRITE","tag-rw") + '''</h3>
<p>Per-context fine-grained locking. Global shared_mutex + per-context shared_mutex (stored as unique_ptr to prevent rehash invalidation). ContextMutexRef RAII pattern holds both locks.</p>

<h3>Class: SharedRegistry ''' + tag("READ-WRITE","tag-rw") + '''</h3>
<p>Registry-level + per-model locking. ModelGuard RAII pattern: holds registry shared_lock + per-model exclusive lock.</p>

<h3>Class: SharedEmbeddings ''' + tag("READ-WRITE","tag-rw") + '''</h3>
<p>shared_mutex with fast-path/slow-path optimization for lazy context creation. Callback-based mutable access.</p>

<h3>Class: HierarchicalMutex</h3>
<p>RAII wrapper enforcing lock hierarchy. check_hierarchy() in debug builds, no-op in release.</p>

<h3>Class: DeadlockDetector (DEBUG only)</h3>
<p>Singleton. Logs lock events (max 10000). DFS cycle detection in wait-for graph. No-op stub in release.</p>

<div class="invariant"><strong>Invariant:</strong> Lock ordering: LTM(1) before STM(2) before Registry(3) before Embeddings(4). All SharedLTM/STM/Registry/Embeddings operations hold appropriate locks. Per-context mutexes stored as unique_ptr (rehash-safe).</div>
<div class="weak"><strong>Weak Point:</strong> No timeout on lock acquisition. DeadlockDetector drops oldest events on overflow. No automatic deadlock resolution.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 18: streams
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="streams">
<h2>Module: streams</h2>
<p><strong>Files:</strong> <code>streams/stream_config.hpp</code>, <code>lock_free_queue.hpp</code>, <code>think_stream.hpp/.cpp</code>, <code>stream_orchestrator.hpp/.cpp</code>, <code>stream_scheduler.hpp/.cpp</code>, <code>stream_monitor.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Multi-stream parallel thinking. Lock-free queues, autonomous thinking threads, monitoring.</p>

<h3>StreamConfig</h3>
''' + member_table([
    ("max_streams", "uint32_t", "0 = auto-detect via hardware_concurrency()"),
    ("backoff_strategy", "BackoffStrategy", "SpinYieldSleep (3-tier idle backoff)"),
    ("tick_interval", "milliseconds", "10ms between autonomous ticks"),
    ("monitor_interval", "milliseconds", "1000ms health-check interval"),
    ("shutdown_timeout", "seconds", "5s graceful shutdown timeout"),
    ("stall_threshold", "milliseconds", "5000ms stall detection threshold"),
    ("subsystem_flags", "Subsystem", "Bitfield: Spreading|Salience|Curiosity|Understanding"),
]) + '''

<h3>Class Template: MPMCQueue&lt;T&gt;</h3>
<p>Vyukov bounded MPMC lock-free queue. Power-of-2 capacity. ABA-safe via sequence counters. Cache-line aligned head/tail (alignas(64)).</p>
<p>Memory ordering: cell sequences use acquire/release; head/tail use relaxed.</p>

<h3>Class Template: SPSCQueue&lt;T&gt;</h3>
<p>Single-Producer Single-Consumer lock-free queue. Simpler than MPMC (no CAS). One slot reserved for empty detection.</p>

<h3>Enum: StreamState</h3>
<p>Created -> Starting -> Running -> Paused -> Stopping -> Stopped -> Error.</p>

<h3>Class: ThinkStream</h3>
<p>Single thinking thread. Autonomous tick loop with configurable backoff. Runs spreading, salience, curiosity, understanding subsystems per tick.</p>

<h3>Class: StreamOrchestrator</h3>
<p>Creates, manages, and destroys ThinkStreams. Distributes work via lock-free queues.</p>

<h3>Class: StreamScheduler</h3>
<p>Schedules work items to streams. Balances load across available streams.</p>

<h3>Class: StreamMonitor</h3>
<p>Health monitoring thread. Detects stalls (no progress for stall_threshold). Generates alerts via callback.</p>

<div class="invariant"><strong>Invariant:</strong> Queue capacity always power of 2. Sequence counters prevent ABA. head/tail never go backward. Streams transition through state machine only in valid order.</div>
<div class="weak"><strong>Weak Point:</strong> Lock-free queues cannot block (only try_push/try_pop). Approximate size only. Spins on CAS failures under high contention.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 19: core
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="core">
<h2>Module: core</h2>
<p><strong>Files:</strong> <code>core/thinking_pipeline.hpp/.cpp</code>, <code>system_orchestrator.hpp/.cpp</code>, <code>brain19_app.hpp/.cpp</code>, <code>main.cpp</code></p>
<p><strong>Purpose:</strong> Top-level orchestration. ThinkingPipeline (10-step cognitive cycle), SystemOrchestrator (28 subsystems), Brain19App (REPL).</p>

<h3>Class: ThinkingPipeline</h3>
<p>10-step cognitive cycle:</p>
<ol>
<li>Activate seed concepts in STM (CORE_KNOWLEDGE class)</li>
<li>Spreading activation across LTM</li>
<li>Compute salience scores, initialize focus</li>
<li>Generate and combine relevance maps (WEIGHTED_AVERAGE)</li>
<li>Find best thought paths (max 20)</li>
<li>Run curiosity engine</li>
<li>Run understanding layer (if enabled)</li>
<li>Run KAN-LLM validation (if enabled)</li>
<li>Complete</li>
<li>Return result with timing</li>
</ol>

<h3>Class: SystemOrchestrator ''' + tag("CORE","tag-core") + '''</h3>
<p>Central orchestrator. 28+ subsystems. 15-stage initialization with cleanup on failure.</p>
''' + member_table([
    ("config_", "Config", "data_dir, persistence, WAL, streams, Ollama settings"),
    ("running_", "atomic<bool>", "System running state"),
    ("init_stage_", "int", "Initialization stage (0-15)"),
    ("subsystem_mtx_", "recursive_mutex", "Orchestrator-level synchronization"),
    ("active_context_", "ContextId", "Active context for interactive use"),
    ("periodic_thread_", "std::thread", "Background task thread"),
]) + '''
<p>15 initialization stages in order: LTM, WAL+Persistence, BrainController, Embeddings+Registry+Trainer, CognitiveDynamics, CuriosityEngine, KANAdapter, UnderstandingLayer, KAN-LLM Hybrid, IngestionPipeline, ChatInterface, Shared Wrappers, Streams, Evolution, Foundation Seeding.</p>

<details><summary>Key Methods</summary>
<ul>
<li><code>bool initialize()</code> -- 15-stage init with exception handling and cleanup_from_stage().</li>
<li><code>void shutdown()</code> -- Reverse order: stop periodic thread, final checkpoint, stop streams, destroy context, shutdown brain, reset all subsystems.</li>
<li><code>ChatResponse ask(const string&amp; question)</code> -- Find seeds via LTM label search, run thinking cycle, build context, call chat. Locked with subsystem_mtx_.</li>
<li><code>IngestionResult ingest_text(const string&amp;, bool auto_approve)</code> -- Ingest via pipeline, ensure MicroModels. Locked.</li>
<li><code>void create_checkpoint(const string&amp; tag)</code> -- Export STM, save via CheckpointManager, rotate old checkpoints.</li>
<li><code>ThinkingResult run_thinking_cycle(const vector&lt;ConceptId&gt;&amp; seeds)</code> -- Execute pipeline + evolution.</li>
</ul>
</details>

<h3>Class: Brain19App</h3>
<p>REPL interface. Commands: ask, ingest, import, status, streams, checkpoint, restore, concepts, explain, think, help, quit.</p>

<h3>main.cpp</h3>
<p>CLI: --data-dir, --ollama-host, --ollama-model, --no-persistence, --no-foundation, --max-streams, --no-monitor. Interactive REPL or single command mode.</p>

<div class="invariant"><strong>Invariant:</strong> Init stages strictly ordered. Streams use SharedLTM etc, not direct references. Periodic thread only accesses evolution subsystems. Auto-checkpoint every 30 minutes. Auto-maintenance every 5 minutes.</div>
<div class="weak"><strong>Weak Point:</strong> 15-stage init is single failure point. recursive_mutex may hide deadlocks. WAL logging in evolution may fail silently. No timeout on checkpoint ops.</div>
</section>
''')

    # ═══════════════════════════════════════════════════════════════════════════
    # MODULE 20: evolution
    # ═══════════════════════════════════════════════════════════════════════════
    p('''<section id="evolution">
<h2>Module: evolution</h2>
<p><strong>Files:</strong> <code>evolution/pattern_discovery.hpp/.cpp</code>, <code>epistemic_promotion.hpp/.cpp</code>, <code>concept_proposal.hpp/.cpp</code></p>
<p><strong>Purpose:</strong> Knowledge graph evolution: pattern discovery, epistemic promotion/demotion, concept proposal generation.</p>

<h3>Class: PatternDiscovery</h3>
<p>Discovers structural patterns: clusters (BFS connected components), hierarchies (IS_A chains), bridges (cross-component nodes), cycles (DFS), gaps (missing sibling relations).</p>

<h3>Class: EpistemicPromotion ''' + tag("CORE","tag-core") + '''</h3>
<p>Manages epistemic lifecycle: SPECULATION -> HYPOTHESIS -> THEORY -> FACT.</p>

<h4>Promotion Rules</h4>
<table>
<tr><th>From</th><th>To</th><th>Requirements</th><th>Human Review</th></tr>
<tr><td>SPECULATION</td><td>HYPOTHESIS</td><td>3+ supports, validation &gt; 0.3, no contradictions</td><td>No</td></tr>
<tr><td>HYPOTHESIS</td><td>THEORY</td><td>5+ theory+ supports, validation &gt; 0.6, 2+ independent sources, no contradictions</td><td>No</td></tr>
<tr><td>THEORY</td><td>FACT</td><td>validation &gt; 0.7, 5+ supports, no contradictions</td><td><strong>YES</strong></td></tr>
<tr><td>Any</td><td>Lower</td><td>Contradictions detected</td><td>No</td></tr>
</table>

<p>Demotion chain on contradiction: FACT->THEORY(0.6), THEORY->HYPOTHESIS(0.4), HYPOTHESIS->SPECULATION(0.2).</p>

<h3>Class: ConceptProposer</h3>
<p>Generates proposals from: curiosity triggers, relevance anomalies, cross-concept analogies.</p>
''' + member_table([
    ("initial_type", "EpistemicType", "Enforced SPECULATION or HYPOTHESIS only"),
    ("initial_trust", "double", "CAPPED at 0.5"),
]) + '''
<p>Quality scoring: evidence_count * 0.15 + initial_trust + (HYPOTHESIS ? +0.2 : 0) + verified_evidence * 0.1.</p>

<div class="invariant"><strong>Invariant:</strong> THEORY-to-FACT requires human review (NEVER automatic). System-generated concepts ALWAYS start as SPECULATION or HYPOTHESIS. Initial trust CAPPED at 0.5. Demotions automatic on contradictions.</div>
<div class="weak"><strong>Weak Point:</strong> No cycle deduplication in pattern discovery. Gap detection only IS_A hierarchies. Validation score ignores relation type quality. No timestamping of promotions. Label collisions possible with generic naming.</div>
</section>
''')

    # Close
    p(f'''
<section>
<h2>Document Info</h2>
<p>This documentation covers all 20 modules of the Brain19 C++20 backend in dependency order. Generated {datetime.date.today().isoformat()}.</p>
<p>Key design principles: "Tools not Agents", epistemic integrity (knowledge never deleted), compile-time enforcement of metadata, bounded activation values, lock hierarchy for thread safety.</p>
</section>

<script>{JS}</script>
</body>
</html>''')

    return ''.join(parts)


if __name__ == '__main__':
    html = build_html()
    with open('/home/hirschpekf/brain19/docs/technical/index.html', 'w') as f:
        f.write(html)
    print(f"Written {len(html)} bytes to index.html")
