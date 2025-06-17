# Test Case Generator Prompt

## Objetivo
Gerar sugestões de testes unitários, de integração e casos de borda para um trecho de código ou função.

## Passos
1. **Entendimento da Função**
   - Analise assinatura, dependências externas, side-effects.
2. **Identificação de Caminhos**
   - Enumere caminhos de execução felizes, excepcionais e limites.
3. **Geração de Casos**
   - Para cada caminho, produza:
     - Nome de teste descritivo.
     - Setup/mocks necessários.
     - Asserções principais.
4. **Código de Exemplo**
   Gere bloco de código no framework escolhido (PyTest, JUnit, Jest, etc.).
5. **Cobertura e Métricas**
   - Mostre % de cobertura estimada.
   - Sinalize partes não cobertas.

## Placeholders
- {{function_code}}
- {{language}}
- {{framework}}

---
Use quando o usuário solicitar "crie testes para esta função/classe". 