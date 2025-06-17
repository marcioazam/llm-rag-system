"""Testes para o módulo ContextInjector.

Testa a funcionalidade de injeção de contexto, filtragem por relevância,
limitação de tokens e extração de snippets chave.
"""

import pytest
from src.augmentation.context_injector import ContextInjector


class TestContextInjector:
    """Testes para a classe ContextInjector."""

    def setup_method(self):
        """Configuração para cada teste."""
        self.injector = ContextInjector(
            relevance_threshold=0.7,
            max_tokens=1000,
            max_symbols=5,
            max_relations=5
        )
        
        # Dados de teste
        self.sample_docs = [
            {
                "id": "doc1",
                "content": "Python é uma linguagem de programação. É fácil de aprender. Muito popular para ciência de dados.",
                "score": 0.9,
                "metadata": {
                    "source": "python_guide.txt",
                    "symbols": [{"name": "Python"}, {"name": "linguagem"}],
                    "relations": [{"target": "programação"}, {"target": "ciência"}]
                }
            },
            {
                "id": "doc2",
                "content": "JavaScript é usado para desenvolvimento web. É uma linguagem dinâmica. Roda no navegador.",
                "score": 0.8,
                "metadata": {
                    "source": "js_basics.txt",
                    "symbols": [{"name": "JavaScript"}, {"name": "web"}],
                    "relations": [{"target": "desenvolvimento"}, {"target": "navegador"}]
                }
            },
            {
                "id": "doc3",
                "content": "Java é uma linguagem compilada. Usada em aplicações empresariais. Tem tipagem estática.",
                "score": 0.6,  # Abaixo do threshold
                "metadata": {
                    "source": "java_intro.txt",
                    "symbols": [{"name": "Java"}, {"name": "compilada"}],
                    "relations": [{"target": "empresariais"}, {"target": "tipagem"}]
                }
            }
        ]

    def test_inject_context_basic(self):
        """Testa injeção básica de contexto."""
        query = "linguagem programação"
        result = self.injector.inject_context(query, self.sample_docs)
        
        # Deve retornar snippets dos docs com score >= 0.7
        assert len(result) == 2
        assert "python_guide.txt" in result[0]
        assert "js_basics.txt" in result[1]
        assert "Python é uma linguagem" in result[0]
        assert "JavaScript é usado" in result[1]

    def test_inject_context_with_symbols_and_relations(self):
        """Testa injeção de contexto incluindo símbolos e relações."""
        query = "Python"
        result = self.injector.inject_context(query, self.sample_docs[:1])
        
        assert len(result) == 1
        snippet = result[0]
        assert "Símbolos: Python, linguagem" in snippet
        assert "Relações: programação, ciência" in snippet
        assert "Fonte: python_guide.txt" in snippet

    def test_relevance_threshold_filtering(self):
        """Testa filtragem por threshold de relevância."""
        query = "test"
        result = self.injector.inject_context(query, self.sample_docs)
        
        # Doc3 tem score 0.6, abaixo do threshold 0.7
        assert len(result) == 2
        sources = [snippet for snippet in result if "java_intro.txt" in snippet]
        assert len(sources) == 0

    def test_max_tokens_limit(self):
        """Testa limitação por número máximo de tokens."""
        # Injector com limite muito baixo
        small_injector = ContextInjector(max_tokens=10)
        query = "linguagem"
        result = small_injector.inject_context(query, self.sample_docs)
        
        # Deve parar quando atingir o limite de tokens
        assert len(result) <= 2
        total_tokens = sum(len(snippet.split()) for snippet in result)
        assert total_tokens <= 15  # Margem para metadados

    def test_max_symbols_limit(self):
        """Testa limitação do número de símbolos exibidos."""
        # Doc com muitos símbolos
        doc_with_many_symbols = {
            "id": "doc_symbols",
            "content": "Teste com muitos símbolos.",
            "score": 0.9,
            "metadata": {
                "source": "symbols.txt",
                "symbols": [{"name": f"Symbol{i}"} for i in range(10)]
            }
        }
        
        query = "símbolos"
        result = self.injector.inject_context(query, [doc_with_many_symbols])
        
        # Deve limitar a 5 símbolos (max_symbols)
        snippet = result[0]
        symbol_count = snippet.count("Symbol")
        assert symbol_count == 5

    def test_max_relations_limit(self):
        """Testa limitação do número de relações exibidas."""
        # Doc com muitas relações
        doc_with_many_relations = {
            "id": "doc_relations",
            "content": "Teste com muitas relações.",
            "score": 0.9,
            "metadata": {
                "source": "relations.txt",
                "relations": [{"target": f"Target{i}"} for i in range(10)]
            }
        }
        
        query = "relações"
        result = self.injector.inject_context(query, [doc_with_many_relations])
        
        # Deve limitar a 5 relações (max_relations)
        snippet = result[0]
        target_count = snippet.count("Target")
        assert target_count == 5

    def test_empty_docs_list(self):
        """Testa comportamento com lista vazia de documentos."""
        query = "test"
        result = self.injector.inject_context(query, [])
        
        assert result == []

    def test_docs_without_score(self):
        """Testa documentos sem campo score."""
        docs_no_score = [
            {
                "id": "doc_no_score",
                "content": "Documento sem score.",
                "metadata": {"source": "no_score.txt"}
            }
        ]
        
        query = "documento"
        result = self.injector.inject_context(query, docs_no_score)
        
        # Deve usar score padrão de 1 e incluir o documento
        assert len(result) == 1
        assert "no_score.txt" in result[0]

    def test_docs_without_metadata(self):
        """Testa documentos sem metadados."""
        docs_no_metadata = [
            {
                "id": "doc_no_meta",
                "content": "Documento sem metadados.",
                "score": 0.8
            }
        ]
        
        query = "documento"
        result = self.injector.inject_context(query, docs_no_metadata)
        
        # Deve usar fonte padrão
        assert len(result) == 1
        assert "Fonte: Desconhecido" in result[0]

    def test_extract_key_snippets(self):
        """Testa extração de snippets chave."""
        query = "linguagem programação"
        snippets = self.injector._extract_key_snippets(query, self.sample_docs[:1])
        
        assert len(snippets) == 1
        snippet = snippets[0]
        assert snippet["source"] == "python_guide.txt"
        assert "Python é uma linguagem" in snippet["content"]
        assert snippet["relevance"] == 0.9

    def test_split_sentences(self):
        """Testa divisão de texto em sentenças."""
        text = "Primeira sentença. Segunda sentença! Terceira sentença?"
        sentences = ContextInjector._split_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "Primeira sentença."
        assert sentences[1] == "Segunda sentença!"
        assert sentences[2] == "Terceira sentença?"

    def test_query_term_overlap_scoring(self):
        """Testa pontuação baseada em sobreposição de termos."""
        # Doc com alta sobreposição de termos
        high_overlap_doc = {
            "id": "high_overlap",
            "content": "Python linguagem programação fácil. Java também linguagem programação. C++ linguagem baixo nível.",
            "score": 0.8,
            "metadata": {"source": "languages.txt"}
        }
        
        query = "Python linguagem programação"
        snippets = self.injector._extract_key_snippets(query, [high_overlap_doc])
        
        # Primeira sentença deve ter maior pontuação por ter mais termos da query
        snippet_content = snippets[0]["content"]
        assert "Python linguagem programação fácil" in snippet_content

    def test_position_scoring(self):
        """Testa pontuação baseada na posição da sentença."""
        # Doc onde a primeira sentença tem menos overlap mas deve ser favorecida pela posição
        position_doc = {
            "id": "position_test",
            "content": "Introdução ao tópico. Esta sentença tem linguagem programação Python. Conclusão final.",
            "score": 0.8,
            "metadata": {"source": "position.txt"}
        }
        
        query = "linguagem programação"
        snippets = self.injector._extract_key_snippets(query, [position_doc])
        
        # Deve incluir a sentença com maior overlap, mesmo não sendo a primeira
        snippet_content = snippets[0]["content"]
        assert "linguagem programação Python" in snippet_content

    def test_custom_thresholds(self):
        """Testa configuração personalizada de thresholds."""
        custom_injector = ContextInjector(
            relevance_threshold=0.5,
            max_tokens=500,
            max_symbols=3,
            max_relations=3
        )
        
        query = "linguagem"
        result = custom_injector.inject_context(query, self.sample_docs)
        
        # Com threshold 0.5, deve incluir todos os 3 documentos
        assert len(result) == 3
        assert any("java_intro.txt" in snippet for snippet in result)

    def test_sorting_by_relevance(self):
        """Testa ordenação por relevância (score)."""
        query = "linguagem"
        result = self.injector.inject_context(query, self.sample_docs)
        
        # Deve estar ordenado por score (doc1: 0.9, doc2: 0.8)
        assert "python_guide.txt" in result[0]  # Score mais alto primeiro
        assert "js_basics.txt" in result[1]