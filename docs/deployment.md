# Deployment Guide (Docker & Manual)

## Option 1: Docker Compose (Recommended)
```bash
docker-compose up --build
```
Access points:
- Frontend: http://localhost:5173
- Backend API Docs: http://localhost:8000/docs

## Option 2: Local Python + Vite Development
```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/build_index.py --sample 500
uvicorn app.main:app --port 8000 --reload

# Frontend
cd frontend
npm install
npm run dev
```
