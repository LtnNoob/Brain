# Brain19 - Interaktives Chat-Interface

## Übersicht

Das Chat-Interface ermöglicht **natürliche Kommunikation** mit Ihrem Brain19-Wissenssystem.

**WICHTIG:** Dies ist KEIN autonomer LLM-Agent!
- Nur Verbalisierung von vorhandenem LTM-Wissen
- KEINE autonomen Entscheidungen
- KEINE versteckten Modifikationen
- Epistemic metadata wird IMMER angezeigt

---

## Kompilieren und Ausführen

```bash
cd backend
make -f Makefile.chat
./demo_chat
```

---

## Verwendung

### Interaktive Befehle

```
Brain19> summary
# Zeigt Wissensübersicht mit epistemischer Verteilung

Brain19> list facts
# Listet alle FAKTEN auf

Brain19> list theories
# Listet alle THEORIEN auf

Brain19> list hypo
# Listet alle HYPOTHESEN auf

Brain19> list spec
# Listet alle SPEKULATIONEN auf

Brain19> explain 1
# Erklärt Konzept mit ID 1 (mit epistemischen Metadaten)

Brain19> compare 1 3
# Vergleicht zwei Konzepte epistemisch

Brain19> quit
# Beendet Chat
```

### Fragen stellen

Stellen Sie einfach Fragen:

```
Brain19> Katze
Basierend auf meinem Wissen:

**Katze** (FACT, sehr hohe Gewissheit):
Ein kleines fleischfressendes Säugetier (Felis catus), 
seit Jahrtausenden domestiziert

Referenzierte IDs: 1
```

```
Brain19> Multiversum
Basierend auf meinem Wissen:

**Multiversum** (SPECULATION, niedrige Gewissheit):
Spekulative Hypothese über die Existenz multipler paralleler Universen

⚠️ Diese Antwort enthält SPEKULATION oder HYPOTHESEN.
```

---

## Beispiel-Sitzung

```bash
$ ./demo_chat

╔══════════════════════════════════════════════════════════╗
║          Brain19 - Interaktiver Chat                    ║
║     Lokales Wissens-System mit epistemischer Awareness   ║
╚══════════════════════════════════════════════════════════╝

Initialisiere Brain19...
✓ BrainController
✓ LongTermMemory
✓ ChatInterface

Lade Demo-Wissen...
✓ 7 Konzepte geladen
  - 2 FAKTEN
  - 3 THEORIEN (1 invalidiert)
  - 1 HYPOTHESE
  - 1 SPEKULATION

Brain19> summary

╔═══════════════════════════════════════════╗
║  Brain19 - Wissensübersicht               ║
╚═══════════════════════════════════════════╝

Gesamt: 7 aktive Konzepte

Nach Typ:
  • Fakten:       2
  • Theorien:     3
  • Hypothesen:   1
  • Spekulationen: 1

⚠️ 1 invalidierte(s) Konzept(e)

Brain19> explain 1

╔═══════════════════════════════════════════╗
║  Katze
╚═══════════════════════════════════════════╝

Epistemischer Typ: FACT
Status: ACTIVE
Trust: 98%

Definition:
Ein kleines fleischfressendes Säugetier (Felis catus), 
seit Jahrtausenden domestiziert

Epistemische Erklärung:
Dies ist ein verifiziertes FAKTUM mit hoher Gewissheit.
Es basiert auf reproduzierbarer Evidenz.

Brain19> quit

Auf Wiedersehen!
```

---

## Demo-Wissen

Das System wird mit folgendem Wissen initialisiert:

### FAKTEN (Trust ≥ 0.95)
1. **Katze** - Domestiziertes Säugetier
2. **Erde** - Dritter Planet von der Sonne

### THEORIEN (wissenschaftlich belegt)
3. **Evolution** (Trust: 0.95) - Natürliche Selektion
4. **Quantenmechanik** (Trust: 0.95) - Atomphysik
7. **Phlogiston-Theorie** (Trust: 0.05) - ⚠️ INVALIDIERT

### HYPOTHESEN (unter Untersuchung)
5. **Dunkle Materie** (Trust: 0.70) - Hypothetische Materie

### SPEKULATIONEN (niedrige Gewissheit)
6. **Multiversum** (Trust: 0.30) - Parallele Universen

---

## Wie es funktioniert

### 1. Keyword-Matching
```cpp
// Einfache Relevanz-Prüfung
if (lower_question.find(lower_label) != std::string::npos) {
    relevant.push_back(info);
}
```

### 2. Epistemische Verbalisierung
```cpp
// Epistemic metadata wird IMMER angezeigt
answer << "**" << info.label << "** ";
answer << "(" << to_string(info.epistemic.type);

if (info.epistemic.trust >= 0.9) {
    answer << ", sehr hohe Gewissheit";
} // ...
```

### 3. Warnung bei Unsicherheit
```cpp
if (info.epistemic.type == EpistemicType::SPECULATION ||
    info.epistemic.type == EpistemicType::HYPOTHESIS) {
    response.contains_speculation = true;
}
```

---

## Architektur-Compliance

✅ **"Tools not Agents"**
- Nur Verbalisierung, keine Entscheidungen
- Read-only Zugriff auf LTM
- Keine versteckten Aktionen

✅ **Epistemic Rigor**
- Metadata wird IMMER angezeigt
- Unsicherheit wird kommuniziert
- Invalidiertes Wissen wird markiert

✅ **Transparency**
- Referenzierte Konzept-IDs werden angezeigt
- Keine Black-Box-Operationen
- Vollständig nachvollziehbar

---

## Limitierungen (by Design)

**KEIN echtes NLP:**
- Einfaches Keyword-Matching
- Keine semantische Analyse
- Keine Kontext-Auflösung

**KEINE Autonomie:**
- Keine LTM-Modifikationen
- Keine automatischen Schlussfolgerungen
- Keine versteckten Inferenzen

**Diese "Limitierungen" sind ABSICHT:**
- Transparenz > Intelligenz
- Kontrolle > Automatisierung
- Vorhersagbarkeit > "Smarte" Features

---

## Nächste Schritte (Optional)

### Option 1: Echtes LLM (llama.cpp)
```cpp
// Integration mit lokalem Llama-Modell
llama_context* ctx = llama_init_from_file("model.gguf");
// ...
```

**Aufwand:** 1-2 Wochen
**Benefit:** Natürlichere Antworten
**Trade-off:** Höhere Komplexität

### Option 2: Erweiterte Keyword-Suche
```cpp
// TF-IDF, BM25, etc.
double relevance = compute_tf_idf(question, definition);
```

**Aufwand:** 2-3 Tage
**Benefit:** Bessere Relevanz
**Trade-off:** Immer noch kein echtes NLP

### Option 3: So lassen
```
EMPFOHLEN für jetzt
- System ist funktional
- Einfach zu verstehen
- Wartbar
- Transparent
```

---

## Häufige Fragen

**Q: Kann ich eigenes Wissen hinzufügen?**  
A: Ja, aber nur via Code (kein automatisches Lernen).
   Editiere `setup_demo_knowledge()` in `demo_chat.cpp`.

**Q: Warum kein echtes LLM?**  
A: Fokus auf Transparenz und Kontrolle.
   Echtes LLM ist optional (siehe "Nächste Schritte").

**Q: Kann das System lernen?**  
A: Nein. By design. Nur explizites Wissen via LTM-API.

**Q: Warum so simpel?**  
A: "Tools not agents" - Simplicity über Intelligenz.

---

## Status

✅ **FUNKTIONSFÄHIG**
- Kompiliert ohne Fehler
- Alle Befehle funktionieren
- Epistemic metadata wird korrekt angezeigt
- Demo-Wissen geladen

✅ **ARCHITEKTUR-KONFORM**
- Keine autonomen Entscheidungen
- Read-only LTM-Zugriff
- Vollständige Transparenz

⚠️ **LIMITIERUNGEN BEKANNT**
- Einfaches Keyword-Matching
- Keine semantische Analyse
- (Das ist OK - by design)

---

**Viel Spaß beim Chatten mit Brain19! 🧠💬**
