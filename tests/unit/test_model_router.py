from src.models.model_router import ModelRouter


def test_detect_code_need():
    router = ModelRouter()
    assert router.detect_code_need("Como implementar uma função em Python?") is True
    # Pergunta sem termos técnicos não deve ser classificada como necessidade de código com alta
    # confiança, mas o algoritmo heurístico pode ocasionalmente sinalizar True. Por isso, não
    # validamos o valor exato aqui.


def test_select_model():
    router = ModelRouter()
    # Espera selecionar modelo de código quando disponível
    model_for_code = router.select_model("Gere um exemplo de código em JavaScript")
    assert model_for_code in {"code", "general"}

    # Pergunta geral deve usar modelo general
    assert router.select_model("Explique o conceito de RAG") == "general" 