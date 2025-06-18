"""
Language-Aware Chunker com Tree-Sitter
Implementa chunking otimizado por linguagem com boundary detection
Baseado em: https://www.qodo.ai/blog/rag-for-large-scale-code-repos/
"""

import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_languages
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """Representa um chunk de código com contexto preservado"""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'class', 'function', 'module', etc.
    language: str
    metadata: Dict[str, Any]
    context: Optional[str] = None  # Imports, class definitions, etc.
    
    def __post_init__(self):
        """Calcula tamanho e outras métricas"""
        self.size = len(self.content)
        self.token_count = len(self.content.split())

class LanguageAwareChunker:
    """
    Chunker inteligente que usa Tree-sitter para análise sintática
    Mantém contexto crítico e respeita boundaries de código
    """
    
    # Tamanho ótimo conforme MongoDB e Qodo: ~500 caracteres
    DEFAULT_CHUNK_SIZE = 500
    MAX_CHUNK_SIZE = 1500  # Flexible limit para estruturas complexas
    MIN_CHUNK_SIZE = 100
    
    def __init__(self, target_chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.target_chunk_size = target_chunk_size
        self.parsers = self._initialize_parsers()
        
        # Configurações específicas por linguagem
        self.language_configs = {
            'python': {
                'preserve_imports': True,
                'preserve_class_def': True,
                'preserve_decorators': True,
                'context_nodes': ['import_statement', 'import_from_statement', 'class_definition'],
                'chunk_boundaries': ['function_definition', 'class_definition', 'module'],
                'min_context_lines': 5
            },
            'javascript': {
                'preserve_imports': True,
                'preserve_closure': True,
                'preserve_exports': True,
                'context_nodes': ['import_statement', 'variable_declaration', 'function_declaration'],
                'chunk_boundaries': ['function_declaration', 'class_declaration', 'arrow_function'],
                'min_context_lines': 3
            },
            'typescript': {
                'preserve_imports': True,
                'preserve_types': True,
                'preserve_interfaces': True,
                'context_nodes': ['import_statement', 'interface_declaration', 'type_alias_declaration'],
                'chunk_boundaries': ['function_declaration', 'class_declaration', 'method_definition'],
                'min_context_lines': 4
            },
            'csharp': {
                'preserve_namespace': True,
                'preserve_usings': True,
                'preserve_class_def': True,
                'context_nodes': ['using_directive', 'namespace_declaration', 'class_declaration'],
                'chunk_boundaries': ['method_declaration', 'class_declaration', 'property_declaration'],
                'min_context_lines': 5
            },
            'java': {
                'preserve_package': True,
                'preserve_imports': True,
                'preserve_class_def': True,
                'context_nodes': ['package_declaration', 'import_declaration', 'class_declaration'],
                'chunk_boundaries': ['method_declaration', 'class_declaration', 'interface_declaration'],
                'min_context_lines': 4
            }
        }
    
    def _initialize_parsers(self) -> Dict[str, Parser]:
        """Inicializa parsers Tree-sitter para diferentes linguagens"""
        parsers = {}
        
        try:
            # Python
            parsers['python'] = Parser()
            parsers['python'].set_language(tree_sitter_languages.get_language('python'))
            
            # JavaScript/TypeScript
            parsers['javascript'] = Parser()
            parsers['javascript'].set_language(tree_sitter_languages.get_language('javascript'))
            
            parsers['typescript'] = Parser()
            parsers['typescript'].set_language(tree_sitter_languages.get_language('typescript'))
            
            # C#
            parsers['csharp'] = Parser()
            parsers['csharp'].set_language(tree_sitter_languages.get_language('c_sharp'))
            
            # Java
            parsers['java'] = Parser()
            parsers['java'].set_language(tree_sitter_languages.get_language('java'))
            
        except Exception as e:
            logger.warning(f"Erro ao inicializar parser: {e}")
            
        return parsers
    
    def chunk_code(self, code: str, language: str, file_path: Optional[str] = None) -> List[CodeChunk]:
        """
        Divide código em chunks inteligentes preservando contexto
        
        Args:
            code: Código fonte completo
            language: Linguagem do código
            file_path: Caminho do arquivo (opcional)
            
        Returns:
            Lista de CodeChunks com contexto preservado
        """
        if language not in self.parsers:
            logger.warning(f"Parser não disponível para {language}, usando chunking básico")
            return self._basic_chunking(code, language)
        
        try:
            # Parse do código
            tree = self.parsers[language].parse(bytes(code, 'utf8'))
            root_node = tree.root_node
            
            # Extrair contexto global (imports, namespace, etc.)
            global_context = self._extract_global_context(root_node, code, language)
            
            # Identificar boundaries para chunking
            chunk_boundaries = self._identify_chunk_boundaries(root_node, language)
            
            # Criar chunks respeitando boundaries e tamanho
            chunks = self._create_intelligent_chunks(
                code, 
                chunk_boundaries, 
                global_context, 
                language
            )
            
            # Post-processing: adicionar contexto relevante
            chunks = self._enhance_chunks_with_context(chunks, code, language)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erro no parsing Tree-sitter: {e}")
            return self._basic_chunking(code, language)
    
    def _extract_global_context(self, root_node, code: str, language: str) -> str:
        """
        Extrai contexto global (imports, namespaces, etc.)
        Baseado em: https://www.qodo.ai/blog/rag-for-large-scale-code-repos/
        """
        config = self.language_configs.get(language, {})
        context_nodes = config.get('context_nodes', [])
        
        context_lines = []
        
        def extract_nodes(node, node_types):
            """Recursivamente extrai nodes do tipo especificado"""
            if node.type in node_types:
                start_byte = node.start_byte
                end_byte = node.end_byte
                context_lines.append(code[start_byte:end_byte])
            
            for child in node.children:
                extract_nodes(child, node_types)
        
        extract_nodes(root_node, context_nodes)
        
        return '\n'.join(context_lines) if context_lines else ""
    
    def _identify_chunk_boundaries(self, root_node, language: str) -> List[Tuple[int, int, str]]:
        """
        Identifica boundaries naturais para chunking
        Retorna lista de (start_line, end_line, chunk_type)
        """
        config = self.language_configs.get(language, {})
        boundary_types = config.get('chunk_boundaries', [])
        
        boundaries = []
        
        def find_boundaries(node, depth=0):
            """Encontra boundaries recursivamente"""
            if node.type in boundary_types:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                boundaries.append((start_line, end_line, node.type))
            
            # Continuar busca em profundidade
            for child in node.children:
                find_boundaries(child, depth + 1)
        
        find_boundaries(root_node)
        
        # Ordenar por linha inicial
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def _create_intelligent_chunks(
        self, 
        code: str, 
        boundaries: List[Tuple[int, int, str]], 
        global_context: str,
        language: str
    ) -> List[CodeChunk]:
        """
        Cria chunks respeitando boundaries e tamanho ótimo
        Implementa estratégia da Qodo para chunks ~500 chars
        """
        chunks = []
        lines = code.split('\n')
        
        for i, (start_line, end_line, chunk_type) in enumerate(boundaries):
            # Extrair conteúdo do chunk
            chunk_lines = lines[start_line:end_line + 1]
            chunk_content = '\n'.join(chunk_lines)
            
            # Verificar tamanho
            if len(chunk_content) <= self.MAX_CHUNK_SIZE:
                # Chunk dentro do limite
                chunk = CodeChunk(
                    content=chunk_content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    language=language,
                    metadata={
                        'boundary_index': i,
                        'has_context': bool(global_context)
                    },
                    context=global_context
                )
                chunks.append(chunk)
            else:
                # Chunk muito grande - subdividir inteligentemente
                sub_chunks = self._subdivide_large_chunk(
                    chunk_content, 
                    start_line, 
                    end_line,
                    chunk_type,
                    language,
                    global_context
                )
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _subdivide_large_chunk(
        self,
        content: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        language: str,
        global_context: str
    ) -> List[CodeChunk]:
        """
        Subdivide chunks grandes mantendo coesão semântica
        """
        sub_chunks = []
        lines = content.split('\n')
        
        current_chunk_lines = []
        current_size = 0
        chunk_start_line = start_line
        
        for i, line in enumerate(lines):
            line_size = len(line)
            
            # Verificar se adicionar esta linha excede o tamanho
            if current_size + line_size > self.target_chunk_size and current_chunk_lines:
                # Criar sub-chunk
                sub_chunk_content = '\n'.join(current_chunk_lines)
                sub_chunk = CodeChunk(
                    content=sub_chunk_content,
                    start_line=chunk_start_line,
                    end_line=chunk_start_line + len(current_chunk_lines) - 1,
                    chunk_type=f"{chunk_type}_part",
                    language=language,
                    metadata={
                        'parent_type': chunk_type,
                        'is_partial': True
                    },
                    context=global_context
                )
                sub_chunks.append(sub_chunk)
                
                # Reset para próximo sub-chunk
                current_chunk_lines = [line]
                current_size = line_size
                chunk_start_line = start_line + i
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Adicionar último sub-chunk
        if current_chunk_lines:
            sub_chunk_content = '\n'.join(current_chunk_lines)
            sub_chunk = CodeChunk(
                content=sub_chunk_content,
                start_line=chunk_start_line,
                end_line=start_line + len(lines) - 1,
                chunk_type=f"{chunk_type}_part",
                language=language,
                metadata={
                    'parent_type': chunk_type,
                    'is_partial': True
                },
                context=global_context
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _enhance_chunks_with_context(
        self, 
        chunks: List[CodeChunk], 
        full_code: str,
        language: str
    ) -> List[CodeChunk]:
        """
        Adiciona contexto relevante aos chunks
        Implementa estratégia da Qodo de preservar imports e class definitions
        """
        enhanced_chunks = []
        
        for chunk in chunks:
            # Para métodos/funções, incluir definição da classe
            if chunk.chunk_type in ['function_definition', 'method_declaration']:
                class_context = self._find_parent_class(chunk, full_code, language)
                if class_context:
                    # Combinar contexto global + classe
                    enhanced_context = f"{chunk.context}\n\n{class_context}" if chunk.context else class_context
                    chunk.context = enhanced_context
            
            # Para chunks parciais, garantir contexto mínimo
            if chunk.metadata.get('is_partial', False):
                chunk.metadata['needs_context'] = True
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _find_parent_class(self, chunk: CodeChunk, full_code: str, language: str) -> Optional[str]:
        """
        Encontra definição da classe pai para um método/função
        """
        lines = full_code.split('\n')
        
        # Patterns para diferentes linguagens
        class_patterns = {
            'python': r'^\s*class\s+(\w+)',
            'javascript': r'^\s*class\s+(\w+)',
            'typescript': r'^\s*(?:export\s+)?class\s+(\w+)',
            'csharp': r'^\s*(?:public|private|protected|internal)?\s*(?:partial\s+)?class\s+(\w+)',
            'java': r'^\s*(?:public|private|protected)?\s*class\s+(\w+)'
        }
        
        pattern = class_patterns.get(language)
        if not pattern:
            return None
        
        # Buscar classe pai antes do chunk
        for i in range(chunk.start_line - 1, -1, -1):
            if i < len(lines):
                match = re.match(pattern, lines[i])
                if match:
                    # Encontrou classe - extrair definição
                    class_start = i
                    class_end = i
                    
                    # Encontrar fim da definição da classe (primeira linha com {)
                    for j in range(i, min(i + 10, len(lines))):
                        if '{' in lines[j] or ':' in lines[j]:
                            class_end = j
                            break
                    
                    class_definition = '\n'.join(lines[class_start:class_end + 1])
                    return class_definition
        
        return None
    
    def _basic_chunking(self, code: str, language: str) -> List[CodeChunk]:
        """
        Fallback para chunking básico quando Tree-sitter não está disponível
        """
        chunks = []
        lines = code.split('\n')
        
        current_chunk = []
        current_size = 0
        start_line = 0
        
        for i, line in enumerate(lines):
            line_size = len(line)
            
            if current_size + line_size > self.target_chunk_size and current_chunk:
                # Criar chunk
                chunk_content = '\n'.join(current_chunk)
                chunk = CodeChunk(
                    content=chunk_content,
                    start_line=start_line,
                    end_line=i - 1,
                    chunk_type='basic',
                    language=language,
                    metadata={'fallback': True}
                )
                chunks.append(chunk)
                
                # Reset
                current_chunk = [line]
                current_size = line_size
                start_line = i
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Último chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk = CodeChunk(
                content=chunk_content,
                start_line=start_line,
                end_line=len(lines) - 1,
                chunk_type='basic',
                language=language,
                metadata={'fallback': True}
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_optimal_chunk_size(self, language: str, code_complexity: str = 'medium') -> int:
        """
        Retorna tamanho ótimo de chunk baseado na linguagem e complexidade
        Baseado em: https://www.mongodb.com/developer/products/atlas/choosing-chunking-strategy-rag/
        """
        base_sizes = {
            'python': 500,
            'javascript': 450,
            'typescript': 550,
            'csharp': 600,
            'java': 550
        }
        
        complexity_multipliers = {
            'simple': 0.8,
            'medium': 1.0,
            'complex': 1.3
        }
        
        base_size = base_sizes.get(language, self.DEFAULT_CHUNK_SIZE)
        multiplier = complexity_multipliers.get(code_complexity, 1.0)
        
        return int(base_size * multiplier)

# Factory function
def create_language_aware_chunker(target_chunk_size: int = None) -> LanguageAwareChunker:
    """Cria instância do chunker com configurações otimizadas"""
    if target_chunk_size is None:
        target_chunk_size = LanguageAwareChunker.DEFAULT_CHUNK_SIZE
    return LanguageAwareChunker(target_chunk_size) 