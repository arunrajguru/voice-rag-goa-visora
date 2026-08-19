import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data

def test_config_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert data["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"

def test_text_rag_endpoint():
    with TestClient(app) as client:
        response = client.post("/api/rag", json={"query": "What is MSMARCO-XI?"})
        assert response.status_code == 200
        data = response.json()
        assert "transcript" in data
        assert "answer" in data
        assert "timings" in data
