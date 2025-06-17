# Code Review Checklist Prompt

## Objetivo
Auxiliar na revisão de Pull Requests fornecendo uma lista de verificação objetiva e comentários de melhoria.

## Instruções
Siga os passos abaixo quando receber um diff ou lista de arquivos alterados:
1. **Contexto**
   - Leia a descrição do PR, ticket/jira e objetivos.
2. **Checklist Geral**
   - [ ] Legibilidade de código (nomes claros, funções curtas)
   - [ ] Complexidade (evita duplicação, reduz nested loops/ifs)
   - [ ] Padrões de projeto aplicados corretamente
   - [ ] Tratamento de erros e logging
   - [ ] Cobertura de testes atualizada (novos testes ou ajustados)
   - [ ] Documentação/Comentários atualizados
   - [ ] Segurança (SQL Injection, XSS, validação de input)
   - [ ] Performance (uso de cache, N+1 queries)
3. **Comentários Automatizados**
   Gere comentários específicos por arquivo e linha, se possível:
   ```text
   src/module.py:42 ❌ Função com 35 linhas — sugerir extrair em funções menores.
   src/api/routes.py:88 ⚠️ Falta tratamento de exceção em chamada externa.
   ```
4. **Resumo Final**
   - Indique se o PR pode ser aprovado, precisa de ajustes menores ou re-trabalho.
   - Destaque pontos críticos a serem corrigidos antes do merge.

## Placeholders
- {{diff}} – diff ou commit range.
- {{file_paths}} – lista de arquivos alterados.
- {{language}} – linguagem principal do repositório.

---
Use este template quando o usuário pedir "revise este PR" ou similar. 