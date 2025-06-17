from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Verifica se o endpoint raiz está operacional e retorna JSON esperado."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data.get("status") in {"running", "ok"}
    assert isinstance(data.get("features"), list) 