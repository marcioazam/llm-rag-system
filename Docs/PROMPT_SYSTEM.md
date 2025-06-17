# Prompt System Overview (v2024-06-15)

Este documento explica como funciona o subsistema de templates (prompts) integrado ao RAGPipeline.

## 1. Diretórios & Arquivos

| Caminho | Propósito |
|---------|-----------|
| `src/prompts/*.md` | Conteúdo Markdown de cada template ativo |
| `src/prompts/registry.json` | Metadados de templates (id, escopo, estimativa de tokens) |
| `src/prompt_selector.py` | Classificação heurística e seleção de template |
| `src/template_renderer.py` | Preenchimento de placeholders (`{{query}}`, `{{context}}`) |
| `src/ab_test.py` | Controle de variante A/B (`with_prompt` x `no_prompt`) |
| `scripts/prompt_usage_report.py` | Script para gerar relatório de uso via Prometheus |

## 2. Fluxo em Tempo de Execução
1. `RAGPipeline.query()` recebe a pergunta do usuário.
2. `decide_variant()` escolhe a variante:
   - 50 % com template (`with_prompt`)
   - 50 % comportamento legacy (`no_prompt`)
   - Pode ser forçada via ENV `RAG_AB_TEST`.
3. Se for `with_prompt`, `prompt_selector.select_prompt()` devolve `(id, template_text)`.
4. `template_renderer.render_template()` injeta `query` e `context` no template.
5. O template renderizado é **prependado** ao prompt dinâmico (ContextInjector + DynamicPromptSystem).
6. Métricas são atualizadas e a resposta JSON inclui:
   ```json
   {
     "prompt_id": "plan_and_solve",
     "prompt_variant": "with_prompt"
   }
   ```

## 3. Placeholders Disponíveis
| Placeholder | Descrição |
|-------------|-----------|
| `{{query}}` | Pergunta original do usuário |
| `{{context}}` | Bloco de contexto recuperado/selecionado |

> Se o template não possuir placeholder, ele será usado como está.

## 4. Variáveis de Ambiente
| Nome | Exemplo | Efeito |
|------|---------|--------|
| `RAG_WITH_PROMPT_RATIO` | `0.3` | Percentual (0-1) de queries que usam template |
| `RAG_AB_TEST` | `with` ou `no` | Força variante específica (ignora ratio) |

## 5. Métricas Prometheus
| Métrica | Labels | Significado |
|---------|--------|-------------|
| `rag_prompt_usage_total` | `prompt_id` | Contagem de queries por template |
| `rag_prompt_variant_total` | `variant` | Distribuição `with_prompt`/`no_prompt` |
| (existentes) `rag_queries_total`, `rag_query_latency_seconds`, `rag_errors_total` |

## 6. Relatórios Rápidos
Execute:
```bash
python scripts/prompt_usage_report.py --url http://localhost:8001/metrics
```
Saída exemplo:
```
Uso por Prompt ID:
-------------------
plan_and_solve               182
quick_fix_bug                 41

Distribuição A/B (with_prompt vs no_prompt):
-------------------------------------------
with_prompt          123  (50.8%)
no_prompt            119  (49.2%)
```

## 7. Curadoria Contínua
1. Time de IA revisa métricas semanalmente (script acima ou dashboards).  
2. Templates com baixa utilidade (<5 % de aprovação) são movidos para pasta `src/prompts/archive/`.  
3. Novas propostas devem incluir:
   - arquivo `.md`,
   - registro em `src/prompts/registry.json`,
   - estimativa de tokens.
4. Versionamento: `id@vYYYY.MM.DD` no campo `version` do registro.

## 8. Checklist para Adicionar Novo Template
- [ ] Criar arquivo Markdown em `src/prompts/`
- [ ] Atualizar `src/prompts/registry.json`
- [ ] Garantir placeholders padrão se necessário
- [ ] Rodar `python -m pytest` (deve continuar passando)
- [ ] Deploy em ambiente de staging antes de produção

---
Última atualização: 2024-06-15 