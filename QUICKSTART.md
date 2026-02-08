# Brain19 mit Ollama LLM - Quickstart 🚀

## Installation (3 Befehle)

```bash
# 1. Entpacken
tar -xzf brain19_ollama_final.tar.gz
cd brain19_complete_project

# 2. Automatische Installation
./install.sh

# 3. Starten
cd backend
./demo_chat
```

**Das war's!** 🎉

---

## Was install.sh macht

1. ✅ Installiert libcurl & nlohmann-json
2. ✅ Installiert Ollama
3. ✅ Kompiliert Brain19
4. ✅ Lädt llama3.2:3b Model herunter
5. ✅ Verifiziert Installation

**Zeit:** ~5-10 Minuten (abhängig von Download-Geschwindigkeit)

---

## Manuelle Installation

Falls Sie lieber manuell installieren:

### 1. Abhängigkeiten

```bash
sudo apt update
sudo apt install -y libcurl4-openssl-dev nlohmann-json3-dev
```

### 2. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # In Hintergrund starten
ollama pull llama3.2:3b
```

### 3. Brain19 kompilieren

```bash
cd brain19_complete_project/backend
make -f Makefile.ollama
```

### 4. Starten

```bash
./demo_chat
```

---

## Erste Schritte

```
Brain19> summary
# Zeigt Wissensübersicht

Brain19> Was weißt du über Katzen?
# LLM verarbeitet Frage und antwortet mit epistemischen Metadaten

Brain19> Vergleiche Evolution und Multiversum
# LLM vergleicht epistemisch

Brain19> quit
# Beenden
```

---

## 🆕 Cognitive Dynamics Demo

Cognitive Dynamics demonstriert Aufmerksamkeits-Management und Reasoning:

```bash
cd backend
make -f Makefile.cognitive
./demo_cognitive_dynamics
```

**Demonstriert:**
- **Spreading Activation:** Aktivierung breitet sich durch Knowledge-Graph aus
- **Salience Computation:** Ranking nach Wichtigkeit (activation × trust × connectivity)
- **Focus Management:** Arbeitsgedächtnis-Simulation mit Kapazitätslimit
- **Thought Path Finding:** Beste Inferenz-Pfade finden

**Output:**
```
Spreading Statistics:
  Concepts activated: 8
  Max depth reached: 2

Salience scores:
  Cat       0.944
  Mammal    0.753
  Fur       0.663

Top thought paths:
  Cat → Fur
  Cat → Mammal → Animal

ALL EPISTEMIC INVARIANTS PRESERVED ✓
```

**Tests ausführen:**
```bash
./test_cognitive_dynamics
# Erwartung: Alle 8 Tests bestehen ✅
```

---

## Verfügbare Befehle

| Befehl | Funktion |
|--------|----------|
| `summary` | Wissensübersicht |
| `list facts` | Alle Fakten |
| `list theories` | Alle Theorien |
| `explain <id>` | Konzept erklären |
| `compare <id1> <id2>` | Vergleichen |
| **Freie Fragen** | **Natürliche Sprache!** |

---

## Model wechseln

Kleineres Model (schneller):
```bash
ollama pull phi3:mini
```

Besseres Model (langsamer, präziser):
```bash
ollama pull mistral:7b
```

In `demo_chat.cpp` ändern:
```cpp
ollama_config.model = "mistral:7b";
```

Neu kompilieren:
```bash
make -f Makefile.ollama
```

---

## Troubleshooting

### "Ollama not available"
```bash
# In separatem Terminal:
ollama serve
```

### Model nicht gefunden
```bash
ollama list  # Verfügbare Models
ollama pull llama3.2:3b  # Model herunterladen
```

### Kompilierungsfehler
```bash
# Dependencies neu installieren
sudo apt install -y libcurl4-openssl-dev nlohmann-json3-dev
```

---

## Weiterführende Docs

- `OLLAMA_README.md` - Vollständige Dokumentation
- `SYSTEM_OVERVIEW.md` - Brain19 Architektur
- `backend/llm/README.md` - LLM Interface Details

---

## Was ist neu?

### ✅ Echtes LLM
- Natürliche Sprachverarbeitung
- Kontextuelles Verständnis
- Flüssige Antworten

### ✅ Epistemic Prompting
- LLM MUSS Metadaten kommunizieren
- FACT vs SPECULATION unterscheiden
- Trust-Level verbalisieren

### ✅ Lokale Ausführung
- Keine Cloud
- Volle Privatsphäre
- Offline-fähig (nach Model-Download)

### ✅ Fallback-Modus
- System funktioniert auch ohne Ollama
- Automatischer Fallback auf Keyword-Matching

---

**Brain19 spricht jetzt natürliche Sprache! 🧠💬✨**

**Viel Erfolg!**
