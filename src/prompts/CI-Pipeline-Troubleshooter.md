# CI Pipeline Troubleshooter Prompt

## Objetivo
Diagnosticar e corrigir falhas em pipelines de integração contínua (GitHub Actions, GitLab CI, Jenkins, etc.).

## Procedimento
1. **Resumo do Erro**
   - Analise logs para mensagem-chave e etapa que falhou.
2. **Categoria**
   - Classifique: dependência, build, teste, deploy, ambiente.
3. **Hipóteses**
   - Liste 2–3 causas prováveis com base no log.
4. **Soluções Sugeridas**
   - Para cada hipótese, detalhe comando ou alteração YAML.
5. **Patch YAML Exemplo**
   ```yaml
   steps:
     - name: Install dependencies
       run: npm ci --ignore-scripts # evita scripts pós-instalação
   ```
6. **Validação**
   - Como reexecutar pipeline localmente ou rerun job.

## Placeholders
- {{ci_logs}}
- {{yaml_config}}
- {{error_msg}}

---
Use quando o usuário enviar logs de CI ou perguntar “por que meu pipeline falha?” 