from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re

try:
    from tree_sitter import Parser, Node
    from tree_sitter_languages import get_language
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Parser = None
    Node = None

from .base_analyzer import BaseStaticAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class CodeSymbol:
    """Representação de um símbolo de código com contexto enriquecido."""
    name: str
    type: str  # function, class, method, variable, import
    line: int
    column: int = 0
    end_line: int = 0
    signature: str = ""
    docstring: str = ""
    complexity: int = 0
    dependencies: List[str] = field(default_factory=list)
    context: str = ""  # Contexto ao redor do símbolo
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeAnalysisResult:
    """Resultado completo da análise de código."""
    symbols: List[CodeSymbol]
    imports: List[Dict[str, str]]
    complexity_metrics: Dict[str, Any]
    code_quality_issues: List[Dict[str, Any]]
    language_specific_info: Dict[str, Any]
    rag_optimized_chunks: List[Dict[str, Any]]  # Chunks otimizados para RAG


class EnhancedTreeSitterAnalyzer(BaseStaticAnalyzer):
    """
    Analisador Tree-sitter aprimorado para integração com RAG.
    
    Melhorias:
    1. Análise mais profunda de símbolos
    2. Extração de contexto semântico
    3. Métricas de complexidade
    4. Chunks otimizados para RAG
    5. Análise de qualidade de código
    """

    def __init__(self, language: str):
        if not TREE_SITTER_AVAILABLE:
            raise RuntimeError("Tree-sitter não está disponível. Instale com: pip install tree-sitter tree-sitter-languages")
        
        self.language = language.lower()
        
        try:
            # Mapear nomes de linguagem para tree-sitter
            lang_mapping = {
                "python": "python",
                "javascript": "javascript", 
                "typescript": "typescript",
                "java": "java",
                "csharp": "c_sharp",
                "c#": "c_sharp",
                "go": "go",
                "ruby": "ruby",
                "rust": "rust",
                "cpp": "cpp",
                "c++": "cpp",
                "c": "c"
            }
            
            ts_lang = lang_mapping.get(self.language, self.language)
            lang_obj = get_language(ts_lang)
            
        except Exception as exc:
            raise RuntimeError(f"Gramática tree-sitter para {self.language} não encontrada: {exc}") from exc
        
        self.parser = Parser()
        self.parser.set_language(lang_obj)
        
        # Queries específicas por linguagem
        self.queries = self._load_language_queries()

    def _load_language_queries(self) -> Dict[str, str]:
        """Carrega queries tree-sitter específicas por linguagem."""
        
        queries = {}
        
        if self.language == "python":
            queries = {
                "functions": """
                    (function_definition 
                        name: (identifier) @func_name
                        parameters: (parameters) @params
                        body: (block) @body
                    ) @function
                """,
                "classes": """
                    (class_definition
                        name: (identifier) @class_name
                        superclasses: (argument_list)? @superclasses
                        body: (block) @body
                    ) @class
                """,
                "imports": """
                    (import_statement
                        name: (dotted_name) @module
                    ) @import
                    (import_from_statement
                        module_name: (dotted_name)? @module
                        name: (dotted_name) @name
                    ) @import_from
                """,
                "docstrings": """
                    (expression_statement
                        (string) @docstring
                    )
                """
            }
            
        elif self.language in ["javascript", "typescript"]:
            queries = {
                "functions": """
                    (function_declaration
                        name: (identifier) @func_name
                        parameters: (formal_parameters) @params
                        body: (statement_block) @body
                    ) @function
                    
                    (arrow_function
                        parameters: (formal_parameters) @params
                        body: (_) @body
                    ) @arrow_function
                """,
                "classes": """
                    (class_declaration
                        name: (identifier) @class_name
                        superclass: (class_heritage)? @superclass
                        body: (class_body) @body
                    ) @class
                """,
                "imports": """
                    (import_statement
                        source: (string) @module
                    ) @import
                """,
                "exports": """
                    (export_statement) @export
                """
            }
            
        elif self.language == "java":
            queries = {
                "methods": """
                    (method_declaration
                        name: (identifier) @method_name
                        parameters: (formal_parameters) @params
                        body: (block) @body
                    ) @method
                """,
                "classes": """
                    (class_declaration
                        name: (identifier) @class_name
                        superclass: (superclass)? @superclass
                        body: (class_body) @body
                    ) @class
                """,
                "imports": """
                    (import_declaration
                        (scoped_identifier) @import_path
                    ) @import
                """
            }
            
        elif self.language == "go":
            queries = {
                "functions": """
                    (function_declaration
                        name: (identifier) @func_name
                        parameters: (parameter_list) @params
                        body: (block) @body
                    ) @function
                """,
                "types": """
                    (type_declaration
                        (type_spec
                            name: (type_identifier) @type_name
                        )
                    ) @type_decl
                """,
                "imports": """
                    (import_spec
                        path: (interpreted_string_literal) @import_path
                    ) @import
                """
            }
            
        # Adicionar query universal para comentários
        queries["comments"] = """
            (comment) @comment
        """
        
        return queries

    def analyze_comprehensive(self, code: str, file_path: Optional[str] = None) -> CodeAnalysisResult:
        """
        Realiza análise completa do código para integração com RAG.
        
        Args:
            code: Código fonte para analisar
            file_path: Caminho do arquivo (opcional)
            
        Returns:
            CodeAnalysisResult com análise completa
        """
        logger.info(f"🔍 Iniciando análise tree-sitter para {self.language}")
        
        # Parse do código
        tree = self.parser.parse(code.encode())
        code_bytes = code.encode()
        
        # Análises específicas
        symbols = self._extract_enhanced_symbols(tree, code_bytes, code)
        imports = self._extract_imports(tree, code_bytes)
        complexity_metrics = self._calculate_complexity_metrics(tree, symbols)
        quality_issues = self._detect_quality_issues(tree, code_bytes, code)
        language_info = self._extract_language_specific_info(tree, code_bytes)
        rag_chunks = self._create_rag_optimized_chunks(symbols, code, file_path)
        
        result = CodeAnalysisResult(
            symbols=symbols,
            imports=imports,
            complexity_metrics=complexity_metrics,
            code_quality_issues=quality_issues,
            language_specific_info=language_info,
            rag_optimized_chunks=rag_chunks
        )
        
        logger.info(f"✅ Análise concluída: {len(symbols)} símbolos, {len(rag_chunks)} chunks RAG")
        return result

    def _extract_enhanced_symbols(self, tree, code_bytes: bytes, code: str) -> List[CodeSymbol]:
        """Extrai símbolos com informações enriquecidas."""
        symbols = []
        code_lines = code.split('\n')
        
        for query_name, query_text in self.queries.items():
            if query_name == "comments":
                continue
                
            try:
                query = self.parser.language.query(query_text)
                captures = query.captures(tree.root_node)
                
                for node, capture_name in captures:
                    symbol = self._create_enhanced_symbol(
                        node, capture_name, code_bytes, code_lines, query_name
                    )
                    if symbol:
                        symbols.append(symbol)
                        
            except Exception as e:
                logger.warning(f"Erro na query {query_name}: {e}")
                
        return symbols

    def _create_enhanced_symbol(
        self, 
        node, 
        capture_name: str, 
        code_bytes: bytes, 
        code_lines: List[str],
        query_type: str
    ) -> Optional[CodeSymbol]:
        """Cria símbolo enriquecido com contexto."""
        
        try:
            # Informações básicas
            name = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            column = node.start_point[1]
            
            # Determinar tipo do símbolo
            symbol_type = self._determine_symbol_type(capture_name, query_type, node)
            
            # Extrair assinatura
            signature = self._extract_signature(node, code_bytes)
            
            # Extrair docstring/comentários
            docstring = self._extract_docstring(node, code_bytes, code_lines, line)
            
            # Calcular complexidade
            complexity = self._calculate_node_complexity(node)
            
            # Extrair contexto
            context = self._extract_context(code_lines, line, end_line)
            
            # Extrair dependências
            dependencies = self._extract_dependencies(node, code_bytes)
            
            # Metadados específicos
            metadata = {
                "byte_range": (node.start_byte, node.end_byte),
                "line_range": (line, end_line),
                "node_type": node.type,
                "has_docstring": bool(docstring),
                "is_public": not name.startswith("_") if name else True,
                "estimated_loc": end_line - line + 1
            }
            
            return CodeSymbol(
                name=name,
                type=symbol_type,
                line=line,
                column=column,
                end_line=end_line,
                signature=signature,
                docstring=docstring,
                complexity=complexity,
                dependencies=dependencies,
                context=context,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Erro ao criar símbolo: {e}")
            return None

    def _determine_symbol_type(self, capture_name: str, query_type: str, node) -> str:
        """Determina o tipo do símbolo baseado no contexto."""
        
        # Mapear baseado no capture name
        type_mapping = {
            "func_name": "function",
            "method_name": "method", 
            "class_name": "class",
            "type_name": "type",
            "import_path": "import",
            "module": "import"
        }
        
        if capture_name in type_mapping:
            return type_mapping[capture_name]
            
        # Mapear baseado no query type
        if "function" in query_type:
            return "function"
        elif "class" in query_type:
            return "class"
        elif "import" in query_type:
            return "import"
        elif "method" in query_type:
            return "method"
            
        # Fallback baseado no tipo do nó
        node_type = node.type if node else ""
        if "function" in node_type:
            return "function"
        elif "class" in node_type:
            return "class"
        elif "method" in node_type:
            return "method"
            
        return "symbol"

    def _extract_signature(self, node, code_bytes: bytes) -> str:
        """Extrai assinatura completa de função/método."""
        try:
            # Para funções, pegar da declaração até o final dos parâmetros
            if node.type in ["function_declaration", "method_declaration", "function_definition"]:
                # Encontrar nó de parâmetros
                for child in node.children:
                    if "parameter" in child.type or child.type == "formal_parameters":
                        end_byte = child.end_byte
                        signature = code_bytes[node.start_byte:end_byte].decode("utf-8", errors="ignore")
                        return signature.strip()
                        
            # Fallback: primeira linha do nó
            first_line_end = code_bytes.find(b'\n', node.start_byte)
            if first_line_end == -1:
                first_line_end = node.end_byte
                
            return code_bytes[node.start_byte:first_line_end].decode("utf-8", errors="ignore").strip()
            
        except Exception:
            return ""

    def _extract_docstring(self, node, code_bytes: bytes, code_lines: List[str], line: int) -> str:
        """Extrai docstring ou comentários associados."""
        docstring = ""
        
        try:
            # Para Python, procurar string literal no início da função/classe
            if self.language == "python":
                for child in node.children:
                    if child.type == "block":
                        for stmt in child.children:
                            if stmt.type == "expression_statement":
                                for expr in stmt.children:
                                    if expr.type == "string":
                                        docstring = code_bytes[expr.start_byte:expr.end_byte].decode("utf-8", errors="ignore")
                                        docstring = docstring.strip('"""\'\'\'')
                                        break
                                break
                        break
                        
            # Para outras linguagens, procurar comentários antes da declaração
            else:
                for i in range(max(0, line - 3), line):
                    if i < len(code_lines):
                        line_content = code_lines[i].strip()
                        if line_content.startswith(("//", "#", "/*", "*")):
                            docstring += line_content + "\n"
                            
        except Exception as e:
            logger.debug(f"Erro ao extrair docstring: {e}")
            
        return docstring.strip()

    def _calculate_node_complexity(self, node) -> int:
        """Calcula complexidade ciclomática do nó."""
        complexity = 1  # Base complexity
        
        # Contar estruturas de controle que aumentam complexidade
        complexity_keywords = {
            "if_statement", "elif_clause", "else_clause",
            "for_statement", "while_statement", "do_statement",
            "switch_statement", "case_clause", "default_clause",
            "try_statement", "catch_clause", "except_clause",
            "conditional_expression", "ternary_expression"
        }
        
        def count_complexity(node):
            count = 0
            if node.type in complexity_keywords:
                count += 1
            for child in node.children:
                count += count_complexity(child)
            return count
            
        complexity += count_complexity(node)
        return complexity

    def _extract_context(self, code_lines: List[str], start_line: int, end_line: int) -> str:
        """Extrai contexto ao redor do símbolo."""
        # Pegar algumas linhas antes e depois
        context_start = max(0, start_line - 3)
        context_end = min(len(code_lines), end_line + 2)
        
        context_lines = code_lines[context_start:context_end]
        return "\n".join(context_lines)

    def _extract_dependencies(self, node, code_bytes: bytes) -> List[str]:
        """Extrai dependências/chamadas dentro do nó."""
        dependencies = []
        
        def find_calls(node):
            if node.type in ["call", "call_expression", "function_call"]:
                # Extrair nome da função chamada
                for child in node.children:
                    if child.type in ["identifier", "attribute", "member_expression"]:
                        call_name = code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                        dependencies.append(call_name)
                        break
                        
            for child in node.children:
                find_calls(child)
                
        find_calls(node)
        return list(set(dependencies))  # Remove duplicatas

    def _extract_imports(self, tree, code_bytes: bytes) -> List[Dict[str, str]]:
        """Extrai informações de imports."""
        imports = []
        
        if "imports" in self.queries:
            try:
                query = self.parser.language.query(self.queries["imports"])
                captures = query.captures(tree.root_node)
                
                for node, capture_name in captures:
                    import_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                    
                    imports.append({
                        "text": import_text,
                        "type": capture_name,
                        "line": node.start_point[0] + 1,
                        "module": self._extract_module_name(import_text)
                    })
                    
            except Exception as e:
                logger.warning(f"Erro ao extrair imports: {e}")
                
        return imports

    def _extract_module_name(self, import_text: str) -> str:
        """Extrai nome do módulo de uma declaração de import."""
        # Implementação simples com regex
        patterns = [
            r'import\s+([^\s;]+)',
            r'from\s+([^\s]+)\s+import',
            r'require\([\'"]([^\'"]+)[\'\"]\)',
            r'#include\s*[<"]([^>"]+)[>"]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, import_text)
            if match:
                return match.group(1)
                
        return import_text.strip()

    def _calculate_complexity_metrics(self, tree, symbols: List[CodeSymbol]) -> Dict[str, Any]:
        """Calcula métricas de complexidade do código."""
        
        total_complexity = sum(symbol.complexity for symbol in symbols)
        function_count = len([s for s in symbols if s.type in ["function", "method"]])
        class_count = len([s for s in symbols if s.type == "class"])
        
        # Calcular linhas de código
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count
            
        total_nodes = count_nodes(tree.root_node)
        
        return {
            "total_complexity": total_complexity,
            "average_complexity": total_complexity / max(1, function_count),
            "function_count": function_count,
            "class_count": class_count,
            "total_symbols": len(symbols),
            "estimated_nodes": total_nodes,
            "complexity_density": total_complexity / max(1, total_nodes)
        }

    def _detect_quality_issues(self, tree, code_bytes: bytes, code: str) -> List[Dict[str, Any]]:
        """Detecta problemas básicos de qualidade de código."""
        issues = []
        code_lines = code.split('\n')
        
        # Detectar linhas muito longas
        for i, line in enumerate(code_lines):
            if len(line) > 120:
                issues.append({
                    "type": "line_too_long",
                    "line": i + 1,
                    "message": f"Linha muito longa ({len(line)} caracteres)",
                    "severity": "warning"
                })
                
        return issues

    def _extract_language_specific_info(self, tree, code_bytes: bytes) -> Dict[str, Any]:
        """Extrai informações específicas da linguagem."""
        info = {"language": self.language}
        
        if self.language == "python":
            # Detectar versão do Python baseada em sintaxe
            code_str = code_bytes.decode("utf-8", errors="ignore")
            if "async def" in code_str or "await " in code_str:
                info["python_features"] = ["async_await"]
            if "f'" in code_str or 'f"' in code_str:
                info["python_features"] = info.get("python_features", []) + ["f_strings"]
                
        elif self.language in ["javascript", "typescript"]:
            code_str = code_bytes.decode("utf-8", errors="ignore") 
            if "=>" in code_str:
                info["js_features"] = ["arrow_functions"]
            if "async " in code_str:
                info["js_features"] = info.get("js_features", []) + ["async_await"]
                
        return info

    def _create_rag_optimized_chunks(
        self, 
        symbols: List[CodeSymbol], 
        code: str,
        file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Cria chunks otimizados para sistemas RAG."""
        chunks = []
        
        # Chunk por símbolo principal (funções e classes)
        for symbol in symbols:
            if symbol.type in ["function", "method", "class"]:
                chunk_content = []
                
                # Cabeçalho do chunk
                chunk_content.append(f"# {symbol.type.title()}: {symbol.name}")
                
                if symbol.signature:
                    chunk_content.append(f"Signature: {symbol.signature}")
                    
                if symbol.docstring:
                    chunk_content.append(f"Documentation: {symbol.docstring}")
                    
                # Contexto do código
                chunk_content.append("\nCode:")
                chunk_content.append(symbol.context)
                
                # Informações adicionais
                metadata_info = []
                metadata_info.append(f"Complexity: {symbol.complexity}")
                metadata_info.append(f"Line: {symbol.line}-{symbol.end_line}")
                
                if symbol.dependencies:
                    metadata_info.append(f"Dependencies: {', '.join(symbol.dependencies[:5])}")
                    
                chunk_content.append(f"\nMetadata: {' | '.join(metadata_info)}")
                
                chunks.append({
                    "content": "\n".join(chunk_content),
                    "type": "code_symbol",
                    "symbol_name": symbol.name,
                    "symbol_type": symbol.type,
                    "file_path": file_path,
                    "line_range": (symbol.line, symbol.end_line),
                    "metadata": {
                        "complexity": symbol.complexity,
                        "has_docstring": bool(symbol.docstring),
                        "dependencies_count": len(symbol.dependencies),
                        "estimated_loc": symbol.end_line - symbol.line + 1,
                        "language": self.language
                    }
                })
                
        return chunks

    # Implementação da interface base
    def extract_symbols(self, code: str) -> List[Dict[str, Any]]:
        """Implementação da interface base (versão simplificada)."""
        result = self.analyze_comprehensive(code)
        return [
            {
                "name": symbol.name,
                "type": symbol.type,
                "line": symbol.line,
                "signature": symbol.signature,
                "complexity": symbol.complexity
            }
            for symbol in result.symbols
        ]

    def extract_relations(self, code: str) -> List[Dict[str, Any]]:
        """Implementação da interface base (versão simplificada)."""
        result = self.analyze_comprehensive(code)
        relations = []
        
        for import_info in result.imports:
            relations.append({
                "source": "file",
                "target": import_info["module"],
                "relation_type": "imports",
                "line": import_info["line"]
            })
            
        return relations    # Implementao da interface base

