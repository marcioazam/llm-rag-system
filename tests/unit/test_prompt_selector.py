from src.prompt_selector import classify_query
import pytest

@pytest.mark.parametrize(
    "query,expected",
    [
        ("Como corrigir este traceback de Python?", "bugfix"),
        ("Melhorar performance do algoritmo", "perf"),
        ("Preciso de um code review", "review"),
        ("Qual a arquitetura ideal?", "arch"),
        ("Isso é uma pergunta geral sem palavras-chave", "geral"),
    ],
)
def test_classify_query(query, expected):
    assert classify_query(query) == expected 