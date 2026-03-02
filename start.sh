#!/bin/bash
# ============================================================
# Agent-Translate — Start Backend + Frontend
# ============================================================
# Usage:  ./start.sh
# Stop:   Ctrl+C (kills both backend and frontend)
# ============================================================

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║       🌐 Agent-Translate Server           ║"
echo "  ║───────────────────────────────────────────║"
echo "  ║  Backend:   http://localhost:8000          ║"
echo "  ║  Frontend:  http://localhost:3001          ║"
echo "  ╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# Kill any existing processes on our ports
echo -e "${YELLOW}🔄 Cleaning up old processes...${NC}"
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :3001 | xargs kill -9 2>/dev/null || true
sleep 1

# Trap Ctrl+C to kill all background processes
cleanup() {
    echo ""
    echo -e "${RED}🛑 Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ All processes stopped.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start Backend ──
echo -e "${GREEN}🚀 Starting Backend (FastAPI on port 8000)...${NC}"
PYTHONUNBUFFERED=1 python3 scripts/start_server.py > >(sed "s/^/[Backend]  /") 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo -e "${YELLOW}⏳ Waiting for backend...${NC}"
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend ready!${NC}"
        break
    fi
    sleep 1
done

# ── Start Frontend ──
echo -e "${GREEN}🚀 Starting Frontend (Vite on port 3001)...${NC}"
cd "$DIR/frontend"
npm run dev > >(sed "s/^/[Frontend] /") 2>&1 &
FRONTEND_PID=$!
cd "$DIR"

echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ All services running!${NC}"
echo -e "${GREEN}  🌐 Open: ${CYAN}http://localhost:3001${NC}"
echo -e "${GREEN}  ⛔ Press Ctrl+C to stop${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
