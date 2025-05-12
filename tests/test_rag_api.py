import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.rag import AnswerResponse

@pytest.fixture(scope="module")
def client():
    """A TestClient for our FastAPI app."""
    return TestClient(app)

@pytest.fixture(autouse=True)
def stub_get_response(monkeypatch):
    """
    Replace the internal _get_response() with a stub that
    always returns the same AnswerResponse.
    """
    monkeypatch.setattr(
        "app.api.rag._get_response",
        lambda question: AnswerResponse(
            answer="stubbed answer",
            sources=["stub_source.txt"]
        )
    )

def test_ask_success(client):
    """When we POST a valid question, we get our stubbed answer back."""
    resp = client.post("/api/ask", json={"question": "Hello?"})
    assert resp.status_code == 200
    assert resp.json() == {
        "answer": "stubbed answer",
        "sources": ["stub_source.txt"]
    }

def test_ask_empty_question(client):
    """Whitespace-only questions return HTTP 400."""
    resp = client.post("/api/ask", json={"question": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Question cannot be empty"

@pytest.mark.parametrize("payload", [
    {},              # missing "question"
    {"foo": "bar"},  # wrong field
])
def test_ask_bad_payload(client, payload):
    """Malformed JSON bodies return HTTP 422."""
    resp = client.post("/api/ask", json=payload)
    assert resp.status_code == 422
