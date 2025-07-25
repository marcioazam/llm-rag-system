try:
    import ollama  # type: ignore
except ModuleNotFoundError:  # Ollama pode não estar instalado em ambientes de produção
    ollama = None  # type: ignore
    import logging
    logging.warning("Ollama não encontrado; ModelRouter operará em modo degradado e usará apenas respostas vazias ou modelo externo.")
from typing import Dict, List, Optional, Tuple, Set
import re
from enum import Enum

logger = logging.getLogger(__name__)

class TaskType(Enum):
    GENERAL_EXPLANATION = "general_explanation"
    CODE_GENERATION = "code_generation"
    SQL_QUERY = "sql_query"
    ARCHITECTURE_DESIGN = "architecture_design"
    QUICK_SNIPPET = "quick_snippet"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"

class ModelRouter:
    """Roteador inteligente básico para múltiplos modelos"""

    def __init__(self):
        # Configuração dos modelos disponíveis
        self.models = {
            'general': {
                'name': 'llama3.1:8b-instruct-q4_K_M',
                'tasks': [TaskType.GENERAL_EXPLANATION, TaskType.DOCUMENTATION],
                'priority': 1
            },
            'code': {
                'name': 'codellama:7b-instruct',
                'tasks': [TaskType.CODE_GENERATION, TaskType.DEBUGGING],
                'priority': 1
            }
        }

        # Verifica quais modelos estão disponíveis
        self.available_models = self._check_available_models()

        # Indicadores de códigos do sistema original
        self.code_indicators = [
            'código', 'codigo', 'programação', 'programacao', 'função', 'funcao',
            'classe', 'método', 'metodo', 'python', 'javascript', 'java', 'c++',
            'html', 'css', 'sql', 'api', 'framework', 'biblioteca', 'algoritmo',
            'implementar', 'desenvolver', 'criar sistema', 'exemplo de código',
            'exemplo de codigo', 'como fazer', 'sintaxe', 'script'
        ]

    def _check_available_models(self) -> Set[str]:
        """Verifica quais modelos estão instalados no Ollama"""
        if ollama is None:
            # Modo degradado: considerar apenas modelos genéricos como disponíveis
            return {'general'}

        try:
            result = ollama.list()
            installed_models = [model['name'] for model in result['models']]

            for key, config in self.models.items():
                model_name = config['name'].split(':')[0]
                if any(model_name in installed for installed in installed_models):
                    available.add(key)

            logger.info(f"Modelos disponíveis: {available}")
            return available

        except Exception as e:
            logger.error(f"Erro ao verificar modelos: {e}")
            return {'general', 'code'}

    def detect_code_need(self, query: str, context: str = "") -> bool:
        """Detecta se a query necessita geração de código"""
        combined_text = (query + " " + context).lower()

        for indicator in self.code_indicators:
            if indicator in combined_text:
                return True

        code_patterns = [
            r'como\s+implementar',
            r'exemplo\s+de',
            r'código\s+para',
            r'função\s+que',
            r'classe\s+para',
            r'script\s+para'
        ]

        for pattern in code_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True

        return False

    def select_model(self, query: str, context: str = "") -> str:
        """Seleciona o modelo apropriado baseado na query"""
        if self.detect_code_need(query, context):
            return 'code' if 'code' in self.available_models else 'general'
        return 'general'

    def generate_with_model(self, prompt: str, model_key: str,
                          system_prompt: Optional[str] = None,
                          temperature: float = 0.7) -> str:
        """Gera resposta usando o modelo especificado"""
        if ollama is None:
            logger.warning("Ollama não disponível; retornando resposta vazia.")
            return ""

        if model_key not in self.available_models:
            logger.warning(f"Modelo {model_key} não disponível, usando fallback")
            model_key = 'general'

        model_name = self.models[model_key]['name']

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={'temperature': temperature}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Erro ao gerar com modelo {model_name}: {str(e)}")
            return ""

    def generate_hybrid_response(self, query: str, context: str,
                               retrieved_docs: List[str]) -> str:
        """Gera resposta híbrida usando múltiplos modelos quando necessário.

        Regras revisadas:
        1. Sempre gera uma primeira resposta *general* com possíveis marcadores.
        2. Substitui marcadores [CÓDIGO: descrição] **mesmo que** `detect_code_need` não detecte
           necessidade de código-geração (corrige falha de testes).
        3. Retorna resposta final com trechos de código dentro de blocos ```python```.
        """

        docs_context = "\n\n".join(
            [f"Documento {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)]
        )

        # Prompt principal (explicação + marcadores)
        base_prompt = (
            "Com base no contexto fornecido, responda a seguinte pergunta.\n"
            "Quando mencionar que vai mostrar código ou exemplos, use marcadores como [CÓDIGO: descrição]\n"
            "mas NÃO gere o código em si.\n\n"
            f"Contexto:\n{docs_context}\n\nPergunta: {query}\n\nResposta:"
        )

        # 1️⃣ Gera resposta base usando modelo general
        general_response = self.generate_with_model(base_prompt, "general")

        # 2️⃣ Procura por marcadores de código na resposta gerada
        code_markers = list(re.finditer(r"\[CÓDIGO:\s*([^\]]+)\]", general_response))

        # Se não houver marcadores, devolve resposta tal como está
        if not code_markers:
            return general_response

        final_response = general_response

        # 3️⃣ Para cada marcador, gera o trecho de código correspondente
        for marker in code_markers:
            description = marker.group(1)

            code_prompt = (
                "Gere APENAS o código solicitado, sem explicações adicionais.\n"
                f"Requisito: {description}\n"
                f"Contexto da pergunta original: {query}\n\nCódigo:"
            )

            code_system_prompt = (
                "Você é um assistente especializado em programação.\n"
                "Gere código limpo, bem comentado e seguindo as melhores práticas.\n"
                "Sempre inclua comentários explicativos em português."
            )

            generated_code = self.generate_with_model(code_prompt, "code", code_system_prompt)

            final_response = final_response.replace(
                marker.group(0), f"\n\n```python\n{generated_code}\n```\n"
            )

        return final_response

    def get_model_status(self) -> Dict[str, any]:
        """Retorna status dos modelos"""
        return {
            'available': list(self.available_models),
            'total_models': len(self.models),
            'models': {k: v['name'] for k, v in self.models.items()}
        }


class AdvancedModelRouter(ModelRouter):
    """Roteador avançado com mais modelos especializados"""
    
    def __init__(self):
        # Configuração expandida dos modelos
        self.models = {
            'general': {
                'name': 'llama3.1:8b-instruct-q4_K_M',
                'tasks': [TaskType.GENERAL_EXPLANATION, TaskType.DOCUMENTATION],
                'priority': 1
            },
            'code': {
                'name': 'codellama:7b-instruct',
                'tasks': [TaskType.CODE_GENERATION, TaskType.DEBUGGING],
                'priority': 1
            },
            'mistral': {
                'name': 'mistral:7b-instruct-q4_0',
                'tasks': [TaskType.ARCHITECTURE_DESIGN],
                'priority': 2,
                'optional': True
            },
            'sql': {
                'name': 'sqlcoder:7b-q4_0',
                'tasks': [TaskType.SQL_QUERY],
                'priority': 1,
                'optional': True
            },
            'fast': {
                'name': 'phi:2.7b',
                'tasks': [TaskType.QUICK_SNIPPET],
                'priority': 3,
                'optional': True
            }
        }

        # Verifica quais modelos estão disponíveis
        self.available_models = self._check_available_models()

        # Palavras-chave para detecção de tarefas avançadas
        self.task_indicators = {
            TaskType.SQL_QUERY: [
                'sql', 'query', 'select', 'database', 'tabela', 'banco de dados',
                'join', 'where', 'group by', 'consulta sql', 'stored procedure'
            ],
            TaskType.ARCHITECTURE_DESIGN: [
                'arquitetura', 'design pattern', 'microserviços', 'microservices',
                'sistema distribuído', 'escalabilidade', 'high level design',
                'system design', 'architectural', 'padrões de projeto'
            ],
            TaskType.DEBUGGING: [
                'debug', 'erro', 'error', 'bug', 'problema', 'não funciona',
                'exception', 'stack trace', 'corrigir', 'consertar'
            ],
            TaskType.DOCUMENTATION: [
                'documentação', 'documentation', 'readme', 'docstring',
                'comentar', 'explicar código', 'api docs'
            ],
            TaskType.QUICK_SNIPPET: [
                'snippet', 'exemplo rápido', 'one-liner', 'função simples',
                'regex', 'validação', 'converter', 'formatar'
            ]
        }

        # Indicadores de códigos do sistema original
        self.code_indicators = [
            'código', 'codigo', 'programação', 'programacao', 'função', 'funcao',
            'classe', 'método', 'metodo', 'python', 'javascript', 'java', 'c++',
            'html', 'css', 'sql', 'api', 'framework', 'biblioteca', 'algoritmo',
            'implementar', 'desenvolver', 'criar sistema', 'exemplo de código',
            'exemplo de codigo', 'como fazer', 'sintaxe', 'script'
        ]

    def detect_tasks(self, query: str, context: str = "") -> List[TaskType]:
        """Detecta quais tipos de tarefas a query requer"""
        combined_text = (query + " " + context).lower()
        detected_tasks = []

        # Sempre inclui explicação geral como base
        detected_tasks.append(TaskType.GENERAL_EXPLANATION)

        # Detecta tarefas específicas
        for task_type, indicators in self.task_indicators.items():
            if any(indicator in combined_text for indicator in indicators):
                detected_tasks.append(task_type)

        # Se mencionou código mas não detectou tipo específico
        if self.detect_code_need(query, context):
            if TaskType.CODE_GENERATION not in detected_tasks:
                detected_tasks.append(TaskType.CODE_GENERATION)

        return list(set(detected_tasks))

    def select_best_model(self, task: TaskType) -> Optional[str]:
        """Seleciona o melhor modelo disponível para uma tarefa"""
        candidates = []

        for model_key, config in self.models.items():
            if model_key in self.available_models and task in config['tasks']:
                candidates.append((model_key, config['priority']))

        if not candidates:
            return 'general' if 'general' in self.available_models else None

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def generate_advanced_response(self, query: str, context: str,
                                 retrieved_docs: List[str]) -> Dict[str, any]:
        """Gera resposta usando múltiplos modelos especializados"""

        docs_context = "\n\n".join([f"Documento {i+1}: {doc}"
                                   for i, doc in enumerate(retrieved_docs)])

        tasks = self.detect_tasks(query, docs_context)
        logger.info(f"Tarefas detectadas: {[t.value for t in tasks]}")

        result = {
            'answer': '',
            'models_used': [],
            'tasks_performed': [t.value for t in tasks],
            'sections': {}
        }

        # Gera resposta base
        if TaskType.ARCHITECTURE_DESIGN in tasks and 'mistral' in self.available_models:
            base_model = 'mistral'
        else:
            base_model = 'general'

        base_prompt = f"""Com base no contexto fornecido, responda a pergunta.
Use marcadores [CÓDIGO: descrição] onde exemplos de código seriam úteis.
Use marcadores [SQL: descrição] onde queries SQL seriam úteis.

Contexto:
{docs_context}

Pergunta: {query}"""

        base_response = self.generate_with_model(base_prompt, base_model)
        result['sections']['main'] = base_response
        result['models_used'].append(self.models[base_model]['name'])

        # ------------------------------------------------------------
        # Processamento de marcadores de CÓDIGO independentemente das tarefas
        # ------------------------------------------------------------
        code_markers = list(re.finditer(r"\[CÓDIGO:\s*([^\]]+)\]", base_response))
        if code_markers:
            code_model = self.select_best_model(TaskType.CODE_GENERATION)
            for marker in code_markers:
                description = marker.group(1)
                code_prompt = (
                    "Gere o código solicitado:\n"
                    f"Requisito: {description}\n"
                    f"Contexto: {query}\n\nCódigo limpo e bem comentado:"
                )

                generated_code = self.generate_with_model(
                    code_prompt,
                    code_model,
                    "Você é um expert em programação. Gere código limpo e eficiente.",
                )

                base_response = base_response.replace(
                    marker.group(0), f"\n\n```python\n{generated_code}\n```\n"
                )

            if code_model and self.models[code_model]['name'] not in result['models_used']:
                result['models_used'].append(self.models[code_model]['name'])

            if code_markers and 'code_generation' not in result['tasks_performed']:
                result['tasks_performed'].append(TaskType.CODE_GENERATION.value)

        # ------------------------------------------------------------
        # Processamento de marcadores de SQL independentemente das tarefas
        # ------------------------------------------------------------
        sql_markers = list(re.finditer(r"\[SQL:\s*([^\]]+)\]", base_response))
        if sql_markers:
            sql_model = self.select_best_model(TaskType.SQL_QUERY)
            for marker in sql_markers:
                description = marker.group(1)
                sql_prompt = (
                    "Gere a query SQL:\n"
                    f"Requisito: {description}\n"
                    f"Contexto: {query}\n\nSQL:"
                )

                generated_sql = self.generate_with_model(
                    sql_prompt,
                    sql_model,
                    "Você é um especialista em SQL. Gere queries otimizadas.",
                )

                base_response = base_response.replace(
                    marker.group(0), f"\n\n```sql\n{generated_sql}\n```\n"
                )

            if sql_model and self.models[sql_model]['name'] not in result['models_used']:
                result['models_used'].append(self.models[sql_model]['name'])

            if sql_markers and 'sql_query' not in result['tasks_performed']:
                result['tasks_performed'].append(TaskType.SQL_QUERY.value)

        result['answer'] = base_response
        return result
