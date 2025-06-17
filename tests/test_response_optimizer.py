"""Testes para o módulo ResponseOptimizer.

Testa a funcionalidade de adição de citações numeradas às respostas.
"""

import pytest
from src.generation.response_optimizer import ResponseOptimizer


class TestResponseOptimizer:
    """Testes para a classe ResponseOptimizer."""

    def setup_method(self):
        """Configuração para cada teste."""
        self.optimizer = ResponseOptimizer()
        
        # Dados de teste
        self.sample_sources = [
            {
                "id": "source1",
                "metadata": {
                    "source": "python_guide.txt"
                },
                "content": "Python é uma linguagem de programação."
            },
            {
                "id": "source2",
                "metadata": {
                    "source": "js_basics.txt"
                },
                "content": "JavaScript é usado para desenvolvimento web."
            },
            {
                "id": "source3",
                "metadata": {
                    "source": "java_intro.txt"
                },
                "content": "Java é uma linguagem compilada."
            }
        ]

    def test_add_citations_basic(self):
        """Testa adição básica de citações."""
        answer = "Python é uma excelente linguagem para iniciantes."
        result = self.optimizer.add_citations(answer, self.sample_sources)
        
        expected_citations = (
            "[1] python_guide.txt\n"
            "[2] js_basics.txt\n"
            "[3] java_intro.txt"
        )
        
        assert result == f"{answer}\n\n{expected_citations}"

    def test_add_citations_empty_sources(self):
        """Testa comportamento com lista vazia de fontes."""
        answer = "Esta é uma resposta sem fontes."
        result = self.optimizer.add_citations(answer, [])
        
        assert result == answer

    def test_add_citations_none_sources(self):
        """Testa comportamento com fontes None."""
        answer = "Esta é uma resposta sem fontes."
        result = self.optimizer.add_citations(answer, None)
        
        assert result == answer

    def test_add_citations_single_source(self):
        """Testa adição de citação com uma única fonte."""
        answer = "Python é fácil de aprender."
        single_source = [self.sample_sources[0]]
        result = self.optimizer.add_citations(answer, single_source)
        
        expected = f"{answer}\n\n[1] python_guide.txt"
        assert result == expected

    def test_add_citations_missing_metadata(self):
        """Testa comportamento com fontes sem metadados."""
        sources_no_metadata = [
            {
                "id": "source1",
                "content": "Conteúdo sem metadados."
            }
        ]
        
        answer = "Resposta baseada em fonte sem metadados."
        result = self.optimizer.add_citations(answer, sources_no_metadata)
        
        expected = f"{answer}\n\n[1] Fonte 1"
        assert result == expected

    def test_add_citations_missing_source_field(self):
        """Testa comportamento com metadados sem campo 'source'."""
        sources_no_source_field = [
            {
                "id": "source1",
                "metadata": {
                    "title": "Documento sem campo source"
                },
                "content": "Conteúdo do documento."
            }
        ]
        
        answer = "Resposta baseada em fonte sem campo source."
        result = self.optimizer.add_citations(answer, sources_no_source_field)
        
        expected = f"{answer}\n\n[1] Fonte 1"
        assert result == expected

    def test_add_citations_empty_metadata(self):
        """Testa comportamento com metadados vazios."""
        sources_empty_metadata = [
            {
                "id": "source1",
                "metadata": {},
                "content": "Conteúdo com metadados vazios."
            }
        ]
        
        answer = "Resposta baseada em fonte com metadados vazios."
        result = self.optimizer.add_citations(answer, sources_empty_metadata)
        
        expected = f"{answer}\n\n[1] Fonte 1"
        assert result == expected

    def test_add_citations_multiple_sources_numbering(self):
        """Testa numeração correta com múltiplas fontes."""
        answer = "Linguagens de programação são diversas."
        result = self.optimizer.add_citations(answer, self.sample_sources)
        
        # Verifica se a numeração está correta
        assert "[1] python_guide.txt" in result
        assert "[2] js_basics.txt" in result
        assert "[3] java_intro.txt" in result
        
        # Verifica a ordem das citações
        lines = result.split('\n')
        citation_lines = [line for line in lines if line.startswith('[')]
        assert len(citation_lines) == 3
        assert citation_lines[0] == "[1] python_guide.txt"
        assert citation_lines[1] == "[2] js_basics.txt"
        assert citation_lines[2] == "[3] java_intro.txt"

    def test_add_citations_preserves_answer_content(self):
        """Testa se o conteúdo original da resposta é preservado."""
        answer = "Esta é uma resposta complexa com\nmúltiplas linhas\ne caracteres especiais: !@#$%^&*()"
        result = self.optimizer.add_citations(answer, self.sample_sources[:1])
        
        # Verifica se a resposta original está intacta
        assert result.startswith(answer)
        assert "múltiplas linhas" in result
        assert "!@#$%^&*()" in result

    def test_add_citations_large_number_of_sources(self):
        """Testa comportamento com grande número de fontes."""
        # Cria 10 fontes
        many_sources = []
        for i in range(10):
            source = {
                "id": f"source{i+1}",
                "metadata": {
                    "source": f"document_{i+1}.txt"
                },
                "content": f"Conteúdo do documento {i+1}."
            }
            many_sources.append(source)
        
        answer = "Resposta baseada em muitas fontes."
        result = self.optimizer.add_citations(answer, many_sources)
        
        # Verifica se todas as 10 citações estão presentes
        for i in range(1, 11):
            assert f"[{i}] document_{i}.txt" in result

    def test_add_citations_special_characters_in_source_names(self):
        """Testa comportamento com caracteres especiais nos nomes das fontes."""
        special_sources = [
            {
                "id": "source1",
                "metadata": {
                    "source": "arquivo com espaços.txt"
                }
            },
            {
                "id": "source2",
                "metadata": {
                    "source": "arquivo-com-hífens.txt"
                }
            },
            {
                "id": "source3",
                "metadata": {
                    "source": "arquivo_com_underscores.txt"
                }
            },
            {
                "id": "source4",
                "metadata": {
                    "source": "arquivo.com.pontos.txt"
                }
            }
        ]
        
        answer = "Resposta com fontes de nomes especiais."
        result = self.optimizer.add_citations(answer, special_sources)
        
        assert "[1] arquivo com espaços.txt" in result
        assert "[2] arquivo-com-hífens.txt" in result
        assert "[3] arquivo_com_underscores.txt" in result
        assert "[4] arquivo.com.pontos.txt" in result

    def test_add_citations_unicode_characters(self):
        """Testa comportamento com caracteres Unicode."""
        unicode_sources = [
            {
                "id": "source1",
                "metadata": {
                    "source": "arquivo_português_ção.txt"
                }
            },
            {
                "id": "source2",
                "metadata": {
                    "source": "文档.txt"  # Caracteres chineses
                }
            },
            {
                "id": "source3",
                "metadata": {
                    "source": "файл.txt"  # Caracteres cirílicos
                }
            }
        ]
        
        answer = "Resposta com fontes Unicode."
        result = self.optimizer.add_citations(answer, unicode_sources)
        
        assert "[1] arquivo_português_ção.txt" in result
        assert "[2] 文档.txt" in result
        assert "[3] файл.txt" in result

    def test_add_citations_empty_answer(self):
        """Testa comportamento com resposta vazia."""
        answer = ""
        result = self.optimizer.add_citations(answer, self.sample_sources[:1])
        
        expected = "\n\n[1] python_guide.txt"
        assert result == expected

    def test_add_citations_whitespace_answer(self):
        """Testa comportamento com resposta contendo apenas espaços."""
        answer = "   \n\t  "
        result = self.optimizer.add_citations(answer, self.sample_sources[:1])
        
        expected = f"{answer}\n\n[1] python_guide.txt"
        assert result == expected

    def test_add_citations_format_consistency(self):
        """Testa consistência do formato das citações."""
        answer = "Teste de formato."
        result = self.optimizer.add_citations(answer, self.sample_sources)
        
        # Verifica se há exatamente duas quebras de linha antes das citações
        parts = result.split(answer)
        assert len(parts) == 2
        citations_part = parts[1]
        assert citations_part.startswith("\n\n")
        
        # Verifica se cada citação está em uma linha separada
        citation_lines = citations_part.strip().split('\n')
        assert len(citation_lines) == 3
        
        # Verifica formato de cada citação
        for i, line in enumerate(citation_lines, 1):
            assert line.startswith(f"[{i}] ")

    def test_add_citations_source_extraction_edge_cases(self):
        """Testa casos extremos na extração do nome da fonte."""
        edge_case_sources = [
            {
                "id": "source1",
                "metadata": {
                    "source": None  # Source é None
                }
            },
            {
                "id": "source2",
                "metadata": {
                    "source": ""  # Source é string vazia
                }
            },
            {
                "id": "source3",
                "metadata": {
                    "source": "   "  # Source é apenas espaços
                }
            }
        ]
        
        answer = "Teste de casos extremos."
        result = self.optimizer.add_citations(answer, edge_case_sources)
        
        # O comportamento real: None e strings vazias são usadas como estão
        assert "[1] None" in result
        assert "[2] " in result  # String vazia
        assert "[3]    " in result  # Espaços são preservados