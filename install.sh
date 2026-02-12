#!/bin/bash
# Brain19 - Ollama LLM Installation Script

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Brain19 - Ollama LLM Integration Setup             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}Bitte NICHT als root ausführen!${NC}"
    echo "Nutzen Sie: ./install.sh"
    exit 1
fi

echo "📦 Schritt 1/5: System-Abhängigkeiten"
echo "────────────────────────────────────"

# Check package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt-get"
elif command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
else
    echo -e "${RED}Error: apt/apt-get nicht gefunden${NC}"
    exit 1
fi

echo "Installiere libcurl und nlohmann-json..."
if ! pkg-config --exists libcurl; then
    echo "  Installing libcurl4-openssl-dev..."
    sudo $PKG_MANAGER update
    sudo $PKG_MANAGER install -y libcurl4-openssl-dev
fi

if [ ! -f "/usr/include/nlohmann/json.hpp" ]; then
    echo "  Installing nlohmann-json3-dev..."
    sudo $PKG_MANAGER install -y nlohmann-json3-dev
fi

echo -e "${GREEN}✓${NC} System-Abhängigkeiten installiert"
echo ""

echo "🦙 Schritt 2/5: Ollama Installation"
echo "────────────────────────────────────"

if ! command -v ollama &> /dev/null; then
    echo "Ollama nicht gefunden. Installiere..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓${NC} Ollama installiert"
else
    echo -e "${GREEN}✓${NC} Ollama bereits installiert"
fi

# Check Ollama version
OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
echo "  Version: $OLLAMA_VERSION"
echo ""

echo "🔧 Schritt 3/5: Brain19 kompilieren"
echo "────────────────────────────────────"

cd backend
make -f Makefile.ollama clean
make -f Makefile.ollama

if [ -f "./demo_chat" ]; then
    echo -e "${GREEN}✓${NC} Brain19 erfolgreich kompiliert"
else
    echo -e "${RED}✗${NC} Kompilierung fehlgeschlagen"
    exit 1
fi
echo ""

echo "📥 Schritt 4/5: Ollama Model herunterladen"
echo "────────────────────────────────────────────"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Ollama läuft nicht. Starte Ollama..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${RED}✗${NC} Konnte Ollama nicht starten"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Ollama gestartet (PID: $OLLAMA_PID)"
else
    echo -e "${GREEN}✓${NC} Ollama läuft bereits"
fi

# Check if model exists
if ollama list | grep -q "llama3.2:1b"; then
    echo -e "${GREEN}✓${NC} Model llama3.2:1b bereits vorhanden"
else
    echo "Lade Model herunter (kann einige Minuten dauern)..."
    ollama pull llama3.2:1b
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Model llama3.2:1b heruntergeladen"
    else
        echo -e "${RED}✗${NC} Model-Download fehlgeschlagen"
        exit 1
    fi
fi
echo ""

echo "✅ Schritt 5/5: Verifikation"
echo "───────────────────────────"

echo "Testing Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Ollama läuft und ist erreichbar"
else
    echo -e "${RED}✗${NC} Ollama nicht erreichbar"
    exit 1
fi

echo "Verfügbare Models:"
ollama list | grep -E "NAME|llama"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           Installation erfolgreich! 🎉               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Start Brain19 Chat:"
echo "  cd backend"
echo "  ./demo_chat"
echo ""
echo "Ollama Management:"
echo "  Starten:  ollama serve"
echo "  Stoppen:  pkill ollama"
echo "  Models:   ollama list"
echo ""
echo "Weitere Infos: siehe OLLAMA_README.md"
echo ""
