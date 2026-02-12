# Brain19 - Chat mit Ollama LLM 🧠💬

## Echtes LLM-Interface für natürliche Kommunikation!

Brain19 nutzt jetzt **Ollama** für intelligente Verbalisierung von epistemischem Wissen.

---

## ⚡ Schnellstart

### 1. Abhängigkeiten installieren

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y libcurl4-openssl-dev nlohmann-json3-dev

# Oder nutze make:
sudo make -f Makefile.ollama install-deps
```

### 2. Ollama installieren & starten

```bash
# Ollama installieren
curl -fsSL https://ollama.com/install.sh | sh

# Ollama starten (in separatem Terminal)
ollama serve

# Model herunterladen (empfohlen: llama3.2:1b)
ollama pull llama3.2:1b
```

### 3. Brain19 kompilieren & starten

```bash
cd backend
make -f Makefile.ollama
./demo_chat
```

---

## 🎯 Verfügbare Modelle

### Klein & Schnell (empfohlen für Start)
```bash
ollama pull llama3.2:1b      # 2GB RAM, schnell
ollama pull phi3:mini         # 2GB RAM, sehr effizient
```

### Mittelgroß (bessere Qualität)
```bash
ollama pull mistral:7b        # 5GB RAM, gut
ollama pull llama3.1:8b       # 8GB RAM, sehr gut
```

### Groß (beste Qualität, braucht viel RAM)
```bash
ollama pull llama3.1:70b      # 40GB RAM, exzellent
```

**Tipp:** Start mit `llama3.2:1b` - klein, schnell, gut genug!

---

## 💬 Beispiel-Session

```
╔══════════════════════════════════════════════════════════╗
║          Brain19 - Interaktiver Chat                    ║
║     Lokales Wissens-System mit epistemischer Awareness   ║
╚══════════════════════════════════════════════════════════╝

Initialisiere Brain19...
✓ BrainController
✓ LongTermMemory
✓ Ollama LLM initialized: llama3.2:1b
✓ ChatInterface (LLM aktiv)

Lade Demo-Wissen...
✓ 7 Konzepte geladen

Brain19> Was weißt du über Katzen?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nach meinem Wissen ist eine Katze (FACT, 98% Vertrauen) ein kleines 
fleischfressendes Säugetier (Felis catus), das seit Jahrtausenden 
domestiziert wurde. Katzen gehören zu den beliebtesten Haustieren 
weltweit und zeichnen sich durch ihre Unabhängigkeit und 
Jagdfähigkeiten aus.

Referenzierte IDs: 1
[LLM: 1234ms]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Brain19> Erkläre mir das Multiversum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Dies ist SPEKULATION mit niedriger Gewissheit (30% Vertrauen).

Das Multiversum ist eine spekulative Hypothese, die besagt, dass 
es möglicherweise multiple parallele Universen gibt. Diese Idee 
ist nicht wissenschaftlich belegt und sollte mit großer Vorsicht 
behandelt werden. Es handelt sich um eine theoretische Überlegung 
in der Kosmologie, die derzeit nicht testbar ist.

⚠️ Diese Antwort enthält SPEKULATION oder HYPOTHESEN.

Referenzierte IDs: 6
[LLM: 1456ms]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Brain19> Vergleiche Evolution und Multiversum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evolution (THEORY, 95% Vertrauen) ist eine gut-belegte 
wissenschaftliche Theorie mit umfangreicher empirischer Evidenz. 
Sie erklärt biologische Veränderungen durch natürliche Selektion 
und ist falsifizierbar.

Im Gegensatz dazu ist das Multiversum (SPECULATION, 30% Vertrauen) 
eine spekulative Hypothese ohne empirische Belege. Während Evolution 
durch Fossilien, genetische Daten und Beobachtungen gestützt wird, 
bleibt das Multiversum eine theoretische Idee ohne Testmöglichkeit.

Die epistemische Differenz ist enorm: Evolution ist etablierte 
Wissenschaft, Multiversum ist Spekulation.

Referenzierte IDs: 3, 6
[LLM: 2103ms]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🏗️ Wie es funktioniert

### 1. Epistemic Prompt Engineering

Das LLM erhält einen **System Prompt** mit strengen epistemischen Regeln:

```
You are Brain19's verbalization interface.

CRITICAL RULES:
1. You ONLY verbalize knowledge from LTM
2. You ALWAYS include epistemic metadata
3. You NEVER invent information
4. You MUST distinguish facts from speculation

RESPONSE FORMAT:
"According to my knowledge, cats (FACT, 98% certainty)..."

WARNINGS:
- SPECULATION → "⚠️ This is speculative"
- INVALIDATED → "⚠️ Known to be incorrect"
- Trust < 50% → "⚠️ Low certainty"
```

### 2. Epistemic Context Injection

Für jede Frage wird relevantes LTM-Wissen in den Prompt injiziert:

```
AVAILABLE KNOWLEDGE FROM LTM:

--- Concept: Katze ---
Type: FACT
Status: ACTIVE
Trust: 98%
Definition: Ein kleines fleischfressendes Säugetier...

Frage: Was weißt du über Katzen?
```

### 3. LLM Verbalisierung

Das LLM verarbeitet den epistemischen Kontext und generiert eine natürliche Antwort unter Beachtung der Metadaten.

---

## 🎛️ Konfiguration

### Model wechseln

Editiere `demo_chat.cpp`:

```cpp
OllamaConfig ollama_config;
ollama_config.model = "mistral:7b";  // Anderes Modell
ollama_config.temperature = 0.5;     // Weniger kreativ
ollama_config.num_predict = 1024;    // Längere Antworten
```

### Fallback-Modus

Wenn Ollama nicht verfügbar ist, fällt das System auf einfaches Keyword-Matching zurück:

```
⚠ ChatInterface (Fallback-Modus, kein LLM)
  Tipp: Starte Ollama mit 'ollama serve'
```

---

## ✅ Features

### Epistemic Awareness ✅
- LLM kommuniziert IMMER epistemic metadata
- FACT vs SPECULATION wird klar unterschieden
- Trust-Level werden verbalisiert

### Natural Language ✅
- Versteht komplexe Fragen
- Generiert flüssige Antworten
- Kontextuelles Verständnis

### Transparency ✅
- Referenzierte Konzept-IDs werden angezeigt
- LLM-Antwortzeit wird gemessen
- Fallback bei LLM-Ausfall

### Safety ✅
- LLM hat read-only LTM-Zugriff
- Keine autonomen Entscheidungen
- Keine LTM-Modifikationen möglich

---

## 🚨 Wichtige Limitierungen

### LLM ist ein TOOL, kein Agent!

**Was das LLM KANN:**
- ✅ LTM-Wissen verbalisieren
- ✅ Epistemic metadata kommunizieren
- ✅ Natürliche Antworten generieren
- ✅ Fragen interpretieren

**Was das LLM NICHT KANN:**
- ❌ LTM modifizieren
- ❌ Eigene Entscheidungen treffen
- ❌ Wissen erfinden (Prompt verbietet es!)
- ❌ Autonome Aktionen ausführen

### Prompt Engineering Limits

Das System ist so gut wie:
1. Der System Prompt (epistemic rules)
2. Die Qualität des LLMs
3. Der injizierte LTM-Kontext

**Ein schlechtes LLM kann trotz guter Prompts Fehler machen!**

---

## 🔧 Troubleshooting

### "Ollama not available"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

### "Model not found"

```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.2:1b
```

### Kompilierungsfehler

```bash
# Install missing dependencies
sudo apt install -y libcurl4-openssl-dev nlohmann-json3-dev

# Or use:
sudo make -f Makefile.ollama install-deps
```

### LLM antwortet nicht epistemisch

Das kann bei kleinen Modellen passieren. Versuchen Sie:
1. Größeres Modell (`mistral:7b` oder `llama3.1:8b`)
2. Niedrigere Temperature (0.3-0.5 für präzisere Antworten)

---

## 📊 Performance

### Typische Antwortzeiten

| Model | RAM | Latenz | Qualität |
|-------|-----|--------|----------|
| llama3.2:1b | 2GB | ~1-2s | Gut |
| phi3:mini | 2GB | ~1-2s | Gut |
| mistral:7b | 5GB | ~2-4s | Sehr gut |
| llama3.1:8b | 8GB | ~3-5s | Exzellent |

**Hardware:** AMD Ryzen 5 / Intel i5, 16GB RAM

---

## 🎊 Status

✅ **FUNKTIONSFÄHIG**
- Ollama-Integration komplett
- Epistemic Prompting implementiert
- Fallback-Modus funktioniert
- LLM-Timing wird gemessen

✅ **ARCHITEKTUR-KONFORM**
- LLM ist TOOL, kein Agent
- Read-only LTM-Zugriff
- Keine autonomen Entscheidungen
- Epistemic metadata enforced

⚠️ **PROMPT ENGINEERING ONGOING**
- System Prompt kann optimiert werden
- Kleinere Modelle sind manchmal inconsistent
- Größere Modelle empfohlen für Production

---

## 🚀 Nächste Schritte (Optional)

### 1. Prompt-Optimierung
- A/B-Testing verschiedener System Prompts
- Few-shot Examples einbauen
- Kontext-Fenster optimieren

### 2. Streaming Support
- Realtime Antworten (Wort für Wort)
- Bessere UX bei langsamen Modellen

### 3. Multi-Turn Conversations
- Konversations-History pflegen
- Follow-up Fragen unterstützen

---

## 📚 Ressourcen

- [Ollama Docs](https://ollama.com)
- [Ollama Models](https://ollama.com/library)
- [Llama 3.2 Info](https://ollama.com/library/llama3.2)
- [Brain19 Architecture](../README.md)

---

**Brain19 spricht jetzt natürliche Sprache - mit epistemischer Integrität! 🎉**
