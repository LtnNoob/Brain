# Brain19 — Quickstart

## Installation (3 Befehle)

```bash
# 1. Entpacken
tar -xzf brain19_complete.tar.gz
cd brain19_complete_project

# 2. Automatische Installation
./install.sh

# 3. Starten
cd backend
./demo_chat
```

**Das war's!**

---

## Was install.sh macht

1. Installiert libcurl & nlohmann-json
2. Kompiliert Brain19
3. Verifiziert Installation

**Zeit:** ~5-10 Minuten (abhängig von Download-Geschwindigkeit)

---

## Manuelle Installation

Falls Sie lieber manuell installieren:

### 1. Abhängigkeiten

```bash
sudo apt update
sudo apt install -y libcurl4-openssl-dev nlohmann-json3-dev
```

### 2. Brain19 kompilieren

```bash
cd brain19_complete_project/backend
make
```

### 3. Starten

```bash
./demo_chat
```

---

## Erste Schritte

```
Brain19> summary
# Zeigt Wissensübersicht

Brain19> Was weisst du ueber Katzen?
# Verarbeitet Frage und antwortet mit epistemischen Metadaten

Brain19> Vergleiche Evolution und Multiversum
# Vergleicht epistemisch

Brain19> quit
# Beenden
```

---

## Cognitive Dynamics Demo

Cognitive Dynamics demonstriert Aufmerksamkeits-Management und Reasoning:

```bash
cd backend
make -f Makefile.cognitive
./demo_cognitive_dynamics
```

**Demonstriert:**
- **Spreading Activation:** Aktivierung breitet sich durch Knowledge-Graph aus
- **Salience Computation:** Ranking nach Wichtigkeit (activation x trust x connectivity)
- **Focus Management:** Arbeitsgedaechtnis-Simulation mit Kapazitaetslimit
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
  Cat -> Fur
  Cat -> Mammal -> Animal

ALL EPISTEMIC INVARIANTS PRESERVED
```

**Tests ausfuehren:**
```bash
./test_cognitive_dynamics
# Erwartung: Alle 8 Tests bestehen
```

---

## Verfuegbare Befehle

| Befehl | Funktion |
|--------|----------|
| `summary` | Wissensuebersicht |
| `list facts` | Alle Fakten |
| `list theories` | Alle Theorien |
| `explain <id>` | Konzept erklaeren |
| `compare <id1> <id2>` | Vergleichen |
| **Freie Fragen** | **Natuerliche Sprache!** |

---

## Troubleshooting

### Kompilierungsfehler
```bash
# Dependencies neu installieren
sudo apt install -y libcurl4-openssl-dev nlohmann-json3-dev
```

---

## Weiterfuehrende Docs

- `SYSTEM_OVERVIEW.md` - Brain19 Architektur

---

## Was ist neu?

### Knowledge-Only Mode
- Eigenstaendige Wissensverarbeitung
- Epistemische Metadaten
- Template-basierte Textgenerierung

### Epistemic System
- FACT vs SPECULATION unterscheiden
- Trust-Level verbalisieren

### Lokale Ausfuehrung
- Keine Cloud
- Volle Privatsphaere
- Komplett offline-faehig

### Fallback-Modus
- Automatischer Fallback auf Keyword-Matching

---

**Viel Erfolg!**
