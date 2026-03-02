#!/bin/bash
# ============================================================
# Agent-Translate — Cài đặt toàn bộ project
# ============================================================
# Usage:  ./install.sh
# ============================================================

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║     🌐 Agent-Translate — Installer        ║"
echo "  ╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Check Python ──
echo -e "${BOLD}[1/5] 🐍 Checking Python...${NC}"
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo -e "  ${GREEN}✅ $PY_VERSION${NC}"
else
    echo -e "  ${RED}❌ Python3 not found! Please install Python 3.10+${NC}"
    echo "     brew install python3"
    exit 1
fi

# ── 2. Check Node.js ──
echo -e "${BOLD}[2/5] 📦 Checking Node.js...${NC}"
if command -v node &>/dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    echo -e "  ${GREEN}✅ Node.js $NODE_VERSION${NC}"
else
    echo -e "  ${RED}❌ Node.js not found! Please install Node.js 18+${NC}"
    echo "     brew install node"
    exit 1
fi

if command -v npm &>/dev/null; then
    NPM_VERSION=$(npm --version 2>&1)
    echo -e "  ${GREEN}✅ npm $NPM_VERSION${NC}"
else
    echo -e "  ${RED}❌ npm not found!${NC}"
    exit 1
fi

# ── 3. Install Python dependencies ──
echo ""
echo -e "${BOLD}[3/5] 🐍 Installing Python dependencies...${NC}"
echo -e "  ${YELLOW}→ pip install -r requirements.txt${NC}"
pip3 install -r requirements.txt
echo -e "  ${GREEN}✅ Python dependencies installed!${NC}"

# ── 4. Install Frontend dependencies ──
echo ""
echo -e "${BOLD}[4/5] ⚛️  Installing Frontend dependencies...${NC}"
echo -e "  ${YELLOW}→ npm install (in frontend/)${NC}"
cd "$DIR/frontend"
npm install
cd "$DIR"
echo -e "  ${GREEN}✅ Frontend dependencies installed!${NC}"

# ── 5. Create required directories ──
echo ""
echo -e "${BOLD}[5/5] 📁 Setting up directories...${NC}"
mkdir -p models training_runs datasets subtitles corrections logs
echo -e "  ${GREEN}✅ Directories created!${NC}"

# ── Done ──
echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Installation complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}To start the app:${NC}"
echo -e "    ${CYAN}./start.sh${NC}"
echo ""
echo -e "  ${BOLD}To train models:${NC}"
echo -e "    ${CYAN}PYTHONUNBUFFERED=1 python3 scripts/train_direct_models.py --lang ja-vi --epochs 3 --batch-size 16 --samples 5000${NC}"
echo ""
