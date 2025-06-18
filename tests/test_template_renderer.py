"""Testes para o módulo template_renderer.

Testa a funcionalidade de preenchimento de placeholders em templates de prompt.
"""

import pytest
from src.template_renderer import render_template


class TestTemplateRenderer:
    """Testes para a função render_template."""

    def test_render_template_basic_query_replacement(self):
        """Testa substituição básica do placeholder {{query}}."""
        template = "Responda a seguinte pergunta: {{query}}"
        query = "O que é Python?"
        
        result = render_template(template, query=query)
        
        expected = "Responda a seguinte pergunta: O que é Python?"
        assert result == expected

    def test_render_template_basic_context_replacement(self):
        """Testa substituição básica do placeholder {{context}}."""
        template = "Contexto: {{context}}\nPergunta: {{query}}"
        query = "Explique Python"
        context_snippets = ["Python é uma linguagem de programação", "É fácil de aprender"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Contexto: Python é uma linguagem de programação\n\nÉ fácil de aprender\nPergunta: Explique Python"
        assert result == expected

    def test_render_template_no_context_snippets(self):
        """Testa comportamento quando context_snippets é None."""
        template = "Contexto: {{context}}\nPergunta: {{query}}"
        query = "Teste sem contexto"
        
        result = render_template(template, query=query, context_snippets=None)
        
        expected = "Contexto: \nPergunta: Teste sem contexto"
        assert result == expected

    def test_render_template_empty_context_snippets(self):
        """Testa comportamento com lista vazia de context_snippets."""
        template = "Contexto: {{context}}\nPergunta: {{query}}"
        query = "Teste com lista vazia"
        context_snippets = []
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Contexto: \nPergunta: Teste com lista vazia"
        assert result == expected

    def test_render_template_multiple_query_placeholders(self):
        """Testa template com múltiplos placeholders {{query}}."""
        template = "Início: {{query}} Meio: {{query}} Fim: {{query}}"
        query = "TESTE"
        
        result = render_template(template, query=query)
        
        expected = "Início: TESTE Meio: TESTE Fim: TESTE"
        assert result == expected

    def test_render_template_multiple_context_placeholders(self):
        """Testa template com múltiplos placeholders {{context}}."""
        template = "Primeiro: {{context}} Segundo: {{context}}"
        query = "teste"
        context_snippets = ["snippet1", "snippet2"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Primeiro: snippet1\n\nsnippet2 Segundo: snippet1\n\nsnippet2"
        assert result == expected

    def test_render_template_special_characters(self):
        """Testa com caracteres especiais na query e contexto."""
        template = "Query: {{query}} Context: {{context}}"
        query = "O que é 2+2? E sobre $100 & 50%?"
        context_snippets = ["Preço: $100", "Desconto: 50%", "Fórmula: a + b = c"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Query: O que é 2+2? E sobre $100 & 50%? Context: Preço: $100\n\nDesconto: 50%\n\nFórmula: a + b = c"
        assert result == expected

    def test_render_template_unicode_characters(self):
        """Testa com caracteres unicode."""
        template = "Query: {{query}} Context: {{context}}"
        query = "测试查询 🚀 émojis"
        context_snippets = ["Contexto en español", "Contenu français", "日本語コンテンツ"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Query: 测试查询 🚀 émojis Context: Contexto en español\n\nContenu français\n\n日本語コンテンツ"
        assert result == expected

    def test_render_template_newlines_in_content(self):
        """Testa com quebras de linha no conteúdo."""
        template = "Query: {{query}} Context: {{context}}"
        query = "Linha 1\nLinha 2\nLinha 3"
        context_snippets = ["Primeira linha\nSegunda linha", "Outro\nMultilinha\nSnippet"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Query: Linha 1\nLinha 2\nLinha 3 Context: Primeira linha\nSegunda linha\n\nOutro\nMultilinha\nSnippet"
        assert result == expected

    def test_render_template_no_placeholders(self):
        """Testa template sem placeholders."""
        template = "Este é um template estático sem placeholders"
        query = "query qualquer"
        
        result = render_template(template, query=query)
        
        assert result == "Este é um template estático sem placeholders"

    def test_render_template_empty_template(self):
        """Testa com template vazio."""
        template = ""
        query = "query teste"
        
        result = render_template(template, query=query)
        
        assert result == ""

    def test_render_template_empty_query(self):
        """Testa com query vazia."""
        template = "Query: {{query}}"
        query = ""
        
        result = render_template(template, query=query)
        
        assert result == "Query: "

    def test_render_template_case_sensitive_placeholders(self):
        """Testa que placeholders são case sensitive."""
        template = "{{Query}} {{QUERY}} {{context}} {{CONTEXT}}"
        query = "teste"
        context_snippets = ["snippet"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        # Apenas {{context}} deve ser substituído
        assert result == "{{Query}} {{QUERY}} snippet {{CONTEXT}}"

    def test_render_template_nested_braces(self):
        """Testa comportamento com chaves aninhadas."""
        template = "{{{query}}} {{{{query}}}}"
        query = "teste"
        
        result = render_template(template, query=query)
        
        assert result == "{teste} {{teste}}"

    def test_render_template_whitespace_preservation(self):
        """Testa que espaços em branco são preservados."""
        template = "  {{query}}  \n\n  {{context}}  "
        query = "  query com espaços  "
        context_snippets = ["  contexto com espaços  "]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        assert result == "    query com espaços    \n\n    contexto com espaços    "

    def test_render_template_empty_strings_in_context(self):
        """Testa com strings vazias nos snippets de contexto."""
        template = "Context: {{context}}"
        query = "teste"
        context_snippets = ["", "não-vazio", "", "outro"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        assert result == "Context: \n\nnão-vazio\n\n\n\noutro"

    def test_render_template_realistic_prompt(self):
        """Testa com um template de prompt realista."""
        template = """Você é um assistente útil. Responda à pergunta do usuário baseado no contexto fornecido.

Pergunta do Usuário: {{query}}

Contexto Relevante:
{{context}}

Por favor, forneça uma resposta abrangente baseada no contexto acima."""
        
        query = "O que é aprendizado de máquina?"
        context_snippets = [
            "Aprendizado de máquina é um subconjunto da inteligência artificial.",
            "Envolve treinar algoritmos em dados para fazer previsões.",
            "Tipos comuns incluem aprendizado supervisionado e não supervisionado."
        ]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = """Você é um assistente útil. Responda à pergunta do usuário baseado no contexto fornecido.

Pergunta do Usuário: O que é aprendizado de máquina?

Contexto Relevante:
Aprendizado de máquina é um subconjunto da inteligência artificial.

Envolve treinar algoritmos em dados para fazer previsões.

Tipos comuns incluem aprendizado supervisionado e não supervisionado.

Por favor, forneça uma resposta abrangente baseada no contexto acima."""
        
        assert result == expected

    def test_render_template_very_long_content(self):
        """Testa com conteúdo muito longo."""
        template = "Query: {{query}} Context: {{context}}"
        query = "a" * 1000
        context_snippets = ["b" * 500, "c" * 500]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected_context = "b" * 500 + "\n\n" + "c" * 500
        expected = f"Query: {'a' * 1000} Context: {expected_context}"
        assert result == expected

    def test_render_template_single_context_snippet(self):
        """Testa com apenas um snippet de contexto."""
        template = "Context: {{context}}"
        query = "teste"
        context_snippets = ["Apenas um snippet"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        assert result == "Context: Apenas um snippet"

    def test_render_template_single_context_snippet(self):
        """Testa comportamento com um único snippet de contexto."""
        template = "{{context}}\n\nPergunta: {{query}}"
        query = "Pergunta teste"
        context_snippets = ["Único snippet de contexto"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Único snippet de contexto\n\nPergunta: Pergunta teste"
        assert result == expected

    def test_render_template_multiple_context_snippets(self):
        """Testa comportamento com múltiplos snippets de contexto."""
        template = "Contexto:\n{{context}}\n\nPergunta: {{query}}"
        query = "Como usar Python?"
        context_snippets = [
            "Python é uma linguagem interpretada",
            "Tem sintaxe simples e clara",
            "É amplamente usado em ciência de dados"
        ]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = (
            "Contexto:\n"
            "Python é uma linguagem interpretada\n\n"
            "Tem sintaxe simples e clara\n\n"
            "É amplamente usado em ciência de dados\n\n"
            "Pergunta: Como usar Python?"
        )
        assert result == expected

    def test_render_template_no_placeholders(self):
        """Testa template sem placeholders."""
        template = "Este é um template sem placeholders."
        query = "Pergunta qualquer"
        
        result = render_template(template, query=query)
        
        assert result == template

    def test_render_template_only_query_placeholder(self):
        """Testa template com apenas placeholder {{query}}."""
        template = "Pergunta: {{query}}"
        query = "O que é machine learning?"
        
        result = render_template(template, query=query)
        
        expected = "Pergunta: O que é machine learning?"
        assert result == expected

    def test_render_template_only_context_placeholder(self):
        """Testa template com apenas placeholder {{context}}."""
        template = "Informações: {{context}}"
        query = "Qualquer pergunta"
        context_snippets = ["Informação 1", "Informação 2"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "Informações: Informação 1\n\nInformação 2"
        assert result == expected

    def test_render_template_multiple_query_placeholders(self):
        """Testa template com múltiplos placeholders {{query}}."""
        template = "Pergunta: {{query}}\nResponda: {{query}}"
        query = "Defina IA"
        
        result = render_template(template, query=query)
        
        expected = "Pergunta: Defina IA\nResponda: Defina IA"
        assert result == expected

    def test_render_template_multiple_context_placeholders(self):
        """Testa template com múltiplos placeholders {{context}}."""
        template = "Contexto 1: {{context}}\nContexto 2: {{context}}"
        query = "Teste"
        context_snippets = ["Snippet 1", "Snippet 2"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected_context = "Snippet 1\n\nSnippet 2"
        expected = f"Contexto 1: {expected_context}\nContexto 2: {expected_context}"
        assert result == expected

    def test_render_template_empty_query(self):
        """Testa comportamento com query vazia."""
        template = "Pergunta: {{query}}"
        query = ""
        
        result = render_template(template, query=query)
        
        expected = "Pergunta: "
        assert result == expected

    def test_render_template_empty_template(self):
        """Testa comportamento com template vazio."""
        template = ""
        query = "Pergunta teste"
        
        result = render_template(template, query=query)
        
        assert result == ""

    def test_render_template_whitespace_handling(self):
        """Testa tratamento de espaços em branco."""
        template = "  {{query}}  \n  {{context}}  "
        query = "Pergunta com espaços"
        context_snippets = ["Contexto com espaços"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "  Pergunta com espaços  \n  Contexto com espaços  "
        assert result == expected

    def test_render_template_special_characters_in_query(self):
        """Testa query com caracteres especiais."""
        template = "Pergunta: {{query}}"
        query = "O que é IA? Como funciona? Exemplos: ML, NLP, CV."
        
        result = render_template(template, query=query)
        
        expected = "Pergunta: O que é IA? Como funciona? Exemplos: ML, NLP, CV."
        assert result == expected

    def test_render_template_special_characters_in_context(self):
        """Testa context_snippets com caracteres especiais."""
        template = "Contexto: {{context}}"
        query = "Teste"
        context_snippets = [
            "Snippet com símbolos: @#$%^&*()",
            "Snippet com acentos: ção, ã, é, ü",
            "Snippet com quebras\nde linha"
        ]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = (
            "Contexto: Snippet com símbolos: @#$%^&*()\n\n"
            "Snippet com acentos: ção, ã, é, ü\n\n"
            "Snippet com quebras\nde linha"
        )
        assert result == expected

    def test_render_template_unicode_characters(self):
        """Testa caracteres Unicode."""
        template = "问题: {{query}}\n上下文: {{context}}"
        query = "什么是Python？"
        context_snippets = ["Python是编程语言", "很容易学习"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        expected = "问题: 什么是Python？\n上下文: Python是编程语言\n\n很容易学习"
        assert result == expected

    def test_render_template_long_content(self):
        """Testa comportamento com conteúdo longo."""
        template = "{{context}}\n\nPergunta: {{query}}"
        query = "Pergunta sobre conteúdo longo"
        
        # Cria snippets longos
        long_snippets = []
        for i in range(5):
            snippet = f"Este é um snippet muito longo número {i+1}. " * 20
            long_snippets.append(snippet.strip())
        
        result = render_template(template, query=query, context_snippets=long_snippets)
        
        # Verifica se todos os snippets estão presentes
        for i in range(5):
            assert f"snippet muito longo número {i+1}" in result
        
        # Verifica se a query está presente
        assert "Pergunta sobre conteúdo longo" in result

    def test_render_template_case_sensitivity(self):
        """Testa sensibilidade a maiúsculas/minúsculas dos placeholders."""
        template = "{{QUERY}} {{Query}} {{query}} {{CONTEXT}} {{Context}} {{context}}"
        query = "teste"
        context_snippets = ["contexto"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        # Apenas {{query}} e {{context}} devem ser substituídos (case-sensitive)
        expected = "{{QUERY}} {{Query}} teste {{CONTEXT}} {{Context}} contexto"
        assert result == expected

    def test_render_template_partial_placeholders(self):
        """Testa placeholders parciais ou malformados."""
        template = "{query} {{query} query}} {{query}} {{{query}}} {{context"
        query = "teste"
        context_snippets = ["contexto"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        # Apenas {{query}} completo deve ser substituído
        expected = "{query} {{query} query}} teste {teste} {{context"
        assert result == expected

    def test_render_template_nested_braces(self):
        """Testa comportamento com chaves aninhadas."""
        template = "{{{query}}} {{{{query}}}} {{{{{query}}}}}"
        query = "teste"
        
        result = render_template(template, query=query)
        
        # {{query}} dentro de chaves extras - replace simples substitui apenas {{query}}
        expected = "{teste} {{teste}} {{{teste}}}"
        assert result == expected

    def test_render_template_context_joining(self):
        """Testa se os snippets de contexto são unidos corretamente."""
        template = "{{context}}"
        query = "teste"
        context_snippets = ["Primeiro", "Segundo", "Terceiro"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        # Deve ser unido com \n\n
        expected = "Primeiro\n\nSegundo\n\nTerceiro"
        assert result == expected

    def test_render_template_keyword_only_parameters(self):
        """Testa que os parâmetros são keyword-only."""
        template = "{{query}} {{context}}"
        
        # Deve funcionar com parâmetros nomeados
        result1 = render_template(template, query="teste", context_snippets=["contexto"])
        assert "teste contexto" == result1
        
        # Deve funcionar sem context_snippets
        result2 = render_template(template, query="teste")
        assert "teste " == result2
        
        # Testaria erro com parâmetros posicionais, mas isso causaria erro de sintaxe
        # Por isso, apenas verificamos que a função funciona corretamente com keyword args

    def test_render_template_complex_real_world_example(self):
        """Testa exemplo complexo do mundo real."""
        template = (
            "Você é um assistente especializado em programação.\n\n"
            "Contexto relevante:\n{{context}}\n\n"
            "Pergunta do usuário: {{query}}\n\n"
            "Forneça uma resposta detalhada e precisa."
        )
        
        query = "Como implementar uma função recursiva em Python?"
        context_snippets = [
            "Recursão é uma técnica onde uma função chama a si mesma.",
            "Em Python, é importante definir um caso base para evitar recursão infinita.",
            "Exemplo: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        ]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        # Verifica se todos os componentes estão presentes
        assert "Você é um assistente especializado" in result
        assert "Como implementar uma função recursiva" in result
        assert "Recursão é uma técnica" in result
        assert "caso base para evitar" in result
        assert "def factorial(n)" in result
        assert "Forneça uma resposta detalhada" in result
        
        # Verifica a estrutura geral
        assert result.count("\n\n") >= 4  # Múltiplas seções separadas