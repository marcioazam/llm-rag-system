from src.augmentation.dynamic_prompt_system import DynamicPromptSystem


def _make_system():
    return DynamicPromptSystem()


def test_generate_prompt_basic():
    sys = _make_system()
    query = "O que é Python?"
    ctx = ["Python é uma linguagem de programação.", "Criada por Guido van Rossum."]
    prompt = sys.generate_prompt(query, ctx, task_type="qa", language="Português")

    # Deve conter sistema, contexto formatado e template QA
    assert "Você é um assistente especialista" in prompt
    assert "[1] Python é uma linguagem de programação." in prompt
    assert "Pergunta: O que é Python?" in prompt
    assert prompt.endswith("Resposta:")


def test_generate_prompt_code():
    sys = _make_system()
    query = "Ordenar lista em Python"
    prompt = sys.generate_prompt(query, context_chunks=[], task_type="code", language="Português")

    # Código deve usar template apropriado
    assert "Você é um desenvolvedor experiente" in prompt
    assert "Escreva código para: Ordenar lista em Python" in prompt


def test_reasoning_trigger():
    sys = _make_system()
    query = "Explique como funciona garbage collection"
    prompt = sys.generate_prompt(query, context_chunks=[], task_type="qa", language="Português")

    assert "Vamos pensar passo a passo" in prompt

    # Sem palavras-chave não deve incluir reasoning
    prompt_simple = sys.generate_prompt("Qual a capital do Brasil?", [], task_type="qa")
    assert "Vamos pensar passo a passo" not in prompt_simple 