#!/usr/bin/env python3
"""Generate Brain19 technical documentation HTML with all Mermaid diagrams.

Reads:
  - ARCHITECTURE_OVERVIEW.md
  - ARCHITECTURE_DIAGRAMS.md
  - CLASS_DIAGRAMS.md
  - SEQUENCE_DIAGRAMS.md
  - STATE_DIAGRAMS.md

Outputs:
  - index.html (single-page documentation with inline Mermaid rendering)
"""

import html
import os
import re
import sys

DOCS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

SOURCES = [
    ("overview", "Architecture Overview", "ARCHITECTURE_OVERVIEW.md"),
    ("arch", "Architecture Diagrams", "ARCHITECTURE_DIAGRAMS.md"),
    ("class", "Class Diagrams", "CLASS_DIAGRAMS.md"),
    ("seq", "Sequence Diagrams", "SEQUENCE_DIAGRAMS.md"),
    ("state", "State Diagrams", "STATE_DIAGRAMS.md"),
]

# ── Markdown parser ──────────────────────────────────────────────────

def slugify(prefix, text):
    s = re.sub(r"[^a-zA-Z0-9\s-]", "", text.lower())
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"-+", "-", s)
    return f"{prefix}-{s}"


def parse_md(md_text, prefix):
    """Parse markdown into a list of section dicts."""
    sections = []
    lines = md_text.split("\n")
    i = 0
    cur = None

    while i < len(lines):
        line = lines[i]

        # ## Heading
        if line.startswith("## "):
            if cur:
                sections.append(cur)
            heading = line[3:].strip()
            cur = {"heading": heading, "id": slugify(prefix, heading),
                   "prefix": prefix, "parts": []}
            i += 1
            continue

        if cur is None:
            i += 1
            continue

        # ### Sub-heading
        if line.startswith("### "):
            cur["parts"].append(("h3", line[4:].strip()))
            i += 1
            continue

        # ```mermaid block
        if line.strip() == "```mermaid":
            buf = []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                buf.append(lines[i])
                i += 1
            cur["parts"].append(("mermaid", "\n".join(buf)))
            i += 1
            continue

        # ``` code block
        m = re.match(r"^```(\w*)$", line.strip())
        if m:
            lang = m.group(1)
            buf = []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                buf.append(lines[i])
                i += 1
            cur["parts"].append(("code", "\n".join(buf), lang))
            i += 1
            continue

        # Table
        if line.strip().startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            cur["parts"].append(("table", tbl))
            continue

        # Skip blanks, HRs, TOC links
        stripped = line.strip()
        if (not stripped or stripped == "---"
                or re.match(r"^\d+\.\s+\[", stripped)
                or stripped.startswith("> Updated:")):
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            cur["parts"].append(("quote", stripped[2:]))
            i += 1
            continue

        # List item
        if stripped.startswith("- "):
            cur["parts"].append(("li", stripped[2:]))
            i += 1
            continue

        # Numbered list
        nm = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if nm:
            cur["parts"].append(("li", nm.group(2)))
            i += 1
            continue

        # Plain text
        cur["parts"].append(("p", stripped))
        i += 1

    if cur:
        sections.append(cur)
    return sections


# ── HTML renderers ───────────────────────────────────────────────────

def md_inline(text):
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+)`", r'<code>\1</code>', t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def render_table(rows):
    if len(rows) < 2:
        return ""
    hdrs = [c.strip() for c in rows[0].split("|")[1:-1]]
    out = ["<table><thead><tr>"]
    for h in hdrs:
        out.append(f"<th>{md_inline(h)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows[2:]:
        cells = [c.strip() for c in row.split("|")[1:-1]]
        out.append("<tr>")
        for c in cells:
            out.append(f"<td>{md_inline(c)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render_section(sec):
    o = [f'<h2 id="{sec["id"]}">{html.escape(sec["heading"])}</h2>']
    in_list = False
    for part in sec["parts"]:
        kind = part[0]
        if kind != "li" and in_list:
            o.append("</ul>")
            in_list = False
        if kind == "mermaid":
            escaped = html.escape(part[1])
            o.append('<div class="diagram-box"><pre class="mermaid">')
            o.append(escaped)
            o.append("</pre></div>")
        elif kind == "code":
            o.append(f'<pre class="code-block"><code>{html.escape(part[1])}</code></pre>')
        elif kind == "table":
            o.append(render_table(part[1]))
        elif kind == "h3":
            o.append(f"<h3>{md_inline(part[1])}</h3>")
        elif kind == "quote":
            o.append(f'<blockquote>{md_inline(part[1])}</blockquote>')
        elif kind == "li":
            if not in_list:
                o.append("<ul>")
                in_list = True
            o.append(f"<li>{md_inline(part[1])}</li>")
        elif kind == "p":
            o.append(f"<p>{md_inline(part[1])}</p>")
    if in_list:
        o.append("</ul>")
    return "\n".join(o)


# ── Navigation builder ───────────────────────────────────────────────

def build_nav(all_sections):
    cats = {}
    for sec in all_sections:
        p = sec["prefix"]
        cats.setdefault(p, []).append(sec)

    labels = {
        "overview": "Architecture Overview",
        "arch": "Architecture Diagrams",
        "class": "Class Diagrams",
        "seq": "Sequence Diagrams",
        "state": "State Diagrams",
    }

    o = []
    for key in ["overview", "arch", "class", "seq", "state"]:
        secs = cats.get(key, [])
        label = labels.get(key, key)
        o.append(f'<div class="nav-cat open">')
        o.append(f'<div class="nav-hdr" onclick="this.parentElement.classList.toggle(\'open\')">')
        o.append(f'<span class="arr">&#9654;</span>{html.escape(label)}'
                 f'<span class="cnt">{len(secs)}</span></div>')
        o.append('<div class="nav-links">')
        for s in secs:
            short = s["heading"]
            if len(short) > 48:
                short = short[:45] + "..."
            o.append(f'<a href="#{s["id"]}" title="{html.escape(s["heading"])}">'
                     f'{html.escape(short)}</a>')
        o.append("</div></div>")
    return "\n".join(o)


# ── Page template ────────────────────────────────────────────────────

PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain19 — Technical Documentation</title>
<style>
:root {{
  --bg0:#0d1117; --bg1:#161b22; --bg2:#21262d; --bg3:#1c2128;
  --bdr:#30363d; --tx:#e6edf3; --tx2:#8b949e; --link:#58a6ff;
  --blue:#2563eb; --green:#3fb950; --yellow:#d29922; --red:#f85149;
  --purple:#bc8cff; --sidebar:280px;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
html{{scroll-behavior:smooth;scroll-padding-top:24px}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif;
  background:var(--bg0);color:var(--tx);line-height:1.6;display:flex;min-height:100vh}}

/* Sidebar */
#side{{position:fixed;top:0;left:0;width:var(--sidebar);height:100vh;
  background:var(--bg1);border-right:1px solid var(--bdr);overflow-y:auto;z-index:100}}
#side-hdr{{padding:20px 16px;border-bottom:1px solid var(--bdr);background:var(--bg2);
  position:sticky;top:0;z-index:1}}
#side-hdr h1{{font-size:18px;color:var(--tx)}}
#side-hdr p{{font-size:11px;color:var(--tx2);margin-top:4px}}
.nav-cat{{border-bottom:1px solid var(--bdr)}}
.nav-hdr{{padding:10px 16px;font-size:13px;font-weight:600;color:var(--tx2);
  cursor:pointer;user-select:none;display:flex;align-items:center;gap:6px}}
.nav-hdr:hover{{color:var(--tx)}}
.arr{{font-size:10px;transition:transform .2s;display:inline-block}}
.nav-cat.open .arr{{transform:rotate(90deg)}}
.cnt{{margin-left:auto;background:var(--bg2);padding:1px 7px;border-radius:10px;
  font-size:11px;color:var(--tx2)}}
.nav-links{{display:none;padding-bottom:6px}}
.nav-cat.open .nav-links{{display:block}}
.nav-links a{{display:block;padding:3px 16px 3px 30px;font-size:12px;color:var(--tx2);
  text-decoration:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.nav-links a:hover{{color:var(--link);background:var(--bg2)}}
.nav-links a.active{{color:var(--link);border-left:2px solid var(--blue);padding-left:28px}}

/* Main */
#main{{margin-left:var(--sidebar);flex:1;max-width:1100px;padding:40px 48px 80px}}
h2{{font-size:22px;margin:48px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--bdr)}}
h2:first-of-type{{margin-top:0}}
h3{{font-size:17px;margin:24px 0 10px;color:var(--tx)}}
p{{margin:8px 0}}
ul{{padding-left:24px;margin:6px 0}}
li{{margin:4px 0}}
blockquote{{border-left:3px solid var(--blue);padding:8px 16px;margin:12px 0;
  color:var(--tx2);background:var(--bg1);border-radius:0 6px 6px 0}}
a{{color:var(--link);text-decoration:none}}
a:hover{{text-decoration:underline}}
code{{font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;
  font-size:13px;background:var(--bg2);padding:2px 6px;border-radius:4px;color:var(--purple)}}
strong{{font-weight:600}}
table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}}
th{{background:var(--bg2);padding:8px 12px;text-align:left;border:1px solid var(--bdr);font-weight:600}}
td{{padding:7px 12px;border:1px solid var(--bdr)}}
tr:nth-child(even) td{{background:var(--bg1)}}
.code-block{{background:var(--bg1);border:1px solid var(--bdr);border-radius:6px;
  padding:16px;margin:14px 0;overflow-x:auto;font-family:'SFMono-Regular',Consolas,monospace;
  font-size:13px;line-height:1.5;white-space:pre;color:var(--tx)}}
.code-block code{{background:none;padding:0;color:inherit}}
.diagram-box{{margin:18px 0;padding:20px;background:var(--bg1);border:1px solid var(--bdr);
  border-radius:8px;overflow-x:auto}}
.diagram-box .mermaid{{text-align:center}}
.diagram-box svg{{max-width:100%;height:auto}}
.cat-banner{{background:linear-gradient(135deg,var(--bg2),var(--bg1));
  border:1px solid var(--bdr);border-radius:8px;padding:24px;margin:48px 0 24px}}
.cat-banner h2{{margin:0;padding:0;border:none;font-size:26px}}
.cat-banner .sub{{color:var(--tx2);margin:6px 0 0;font-size:14px}}
#loading{{position:fixed;top:0;left:var(--sidebar);right:0;background:var(--blue);
  color:#fff;text-align:center;padding:8px;font-size:13px;z-index:200;transition:opacity .5s}}
#loading.done{{opacity:0;pointer-events:none}}
#top-btn{{position:fixed;bottom:24px;right:24px;width:40px;height:40px;border-radius:50%;
  background:var(--blue);color:#fff;border:none;cursor:pointer;font-size:18px;
  display:none;align-items:center;justify-content:center;z-index:100;
  box-shadow:0 2px 8px rgba(0,0,0,.3)}}
#top-btn:hover{{background:#3b82f6}}
@media(max-width:900px){{#side{{display:none}}#main{{margin-left:0;padding:20px}}}}
</style>
</head>
<body>

<div id="loading">Rendering {diagram_count} diagrams&hellip;</div>

<nav id="side">
  <div id="side-hdr">
    <h1>Brain19</h1>
    <p>C++20 Cognitive Architecture<br>Technical Documentation</p>
  </div>
  {nav}
</nav>

<main id="main">
  <h1 style="font-size:30px;margin-bottom:6px">Brain19 &mdash; Technical Documentation</h1>
  <p style="color:var(--tx2);margin-bottom:32px">
    {diagram_count} Mermaid diagrams &bull; 5 source documents &bull;
    Generated from actual code in <code>backend/</code>, <code>api/</code>, <code>frontend/</code>
    &bull; Updated 2026-02-12
    &bull; <code>http://172.16.16.104:8080/brain19/</code>
  </p>

  {content}

  <hr style="border:none;border-top:1px solid var(--bdr);margin:48px 0">
  <p style="color:var(--tx2);text-align:center;padding:16px 0">
    Brain19 Technical Documentation &bull; Felix Hirschpek, 2026
  </p>
</main>

<button id="top-btn" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">&uarr;</button>

<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({{
  startOnLoad:true,
  theme:'dark',
  securityLevel:'loose',
  flowchart:{{htmlLabels:true,curve:'basis'}},
  sequence:{{mirrorActors:false,wrap:true}},
  themeVariables:{{
    darkMode:true,
    background:'#161b22',
    primaryColor:'#2563eb',
    primaryTextColor:'#e6edf3',
    primaryBorderColor:'#30363d',
    lineColor:'#8b949e',
    secondaryColor:'#21262d',
    tertiaryColor:'#161b22',
    noteBkgColor:'#21262d',
    noteTextColor:'#e6edf3',
    noteBorderColor:'#30363d',
    actorBkg:'#2563eb',
    actorTextColor:'#ffffff',
    actorBorder:'#1d4ed8',
    signalColor:'#e6edf3',
    signalTextColor:'#e6edf3',
    labelBoxBkgColor:'#21262d',
    labelBoxBorderColor:'#30363d',
    labelTextColor:'#e6edf3',
    loopTextColor:'#8b949e',
    activationBorderColor:'#58a6ff',
    activationBkgColor:'#21262d',
    sequenceNumberColor:'#e6edf3'
  }}
}});
// Hide loading after render
var _t=setInterval(function(){{
  if(document.querySelectorAll('.mermaid svg').length>0){{
    clearInterval(_t);
    setTimeout(function(){{document.getElementById('loading').classList.add('done')}},800);
  }}
}},500);
setTimeout(function(){{document.getElementById('loading').classList.add('done')}},8000);
// Scroll spy
window.addEventListener('scroll',function(){{
  var hs=document.querySelectorAll('h2[id]'),cur='',y=window.scrollY+80;
  hs.forEach(function(h){{if(h.offsetTop<=y)cur=h.id}});
  document.querySelectorAll('.nav-links a').forEach(function(a){{
    a.classList.toggle('active',a.getAttribute('href')==='#'+cur);
  }});
  document.getElementById('top-btn').style.display=window.scrollY>400?'flex':'none';
}});
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────

def main():
    all_sections = []
    for prefix, label, filename in SOURCES:
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        md = open(path, encoding="utf-8").read()
        secs = parse_md(md, prefix)
        all_sections.extend(secs)

    # Count mermaid diagrams
    diagram_count = sum(
        1 for s in all_sections for p in s["parts"] if p[0] == "mermaid"
    )

    # Build nav
    nav_html = build_nav(all_sections)

    # Build content with category banners
    banners = {
        "overview": ("Architecture Overview",
                     "Complete system architecture reference for the Brain19 C++20 Cognitive Architecture."),
        "arch": ("Architecture Diagrams",
                 f"17 Mermaid component, flow, and data-flow diagrams generated from source code."),
        "class": ("Class Diagrams",
                  "11 UML class diagrams for all core module hierarchies."),
        "seq": ("Sequence Diagrams",
                "9 extended sequence diagrams for key workflows."),
        "state": ("State Diagrams",
                  "9 state machine diagrams + appendix for all stateful components."),
    }

    content_parts = []
    seen_prefix = set()
    for sec in all_sections:
        p = sec["prefix"]
        if p not in seen_prefix:
            seen_prefix.add(p)
            if p in banners:
                title, desc = banners[p]
                content_parts.append(
                    f'<div class="cat-banner"><h2>{html.escape(title)}</h2>'
                    f'<p class="sub">{html.escape(desc)}</p></div>'
                )
        content_parts.append(render_section(sec))

    content_html = "\n".join(content_parts)

    # Assemble page
    page_html = PAGE.format(
        nav=nav_html,
        content=content_html,
        diagram_count=diagram_count,
    )

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(page_html)

    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"Generated: {OUTPUT}")
    print(f"  Sections: {len(all_sections)}")
    print(f"  Mermaid diagrams: {diagram_count}")
    print(f"  File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
