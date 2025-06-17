# Quick-Fix-Bug Prompt

## Contexto
Use este roteiro ultra-enxuto quando o usuário reportar um erro específico (ex.: stack-trace, teste falhando, Exception) e precisar de uma correção rápida.

## Passos
1. **Diagnóstico Rápido**
   - Reproduzir ou analisar stack-trace/erro
   - Identificar o arquivo ⟶ função ⟶ linha crítica
2. **Hipótese de Causa**
   - Descrever em 1-2 frases o que provavelmente causou o erro
3. **Patch Minimalista**
   - Mostrar o menor diff possível para resolver o problema
   - Incluir verificação de tipo/nulos ou boundary check se aplicável
4. **Teste de Verificação**
   - Adicionar/atualizar teste unitário que falhava ⟶ deve passar
5. **Riscos & Próximos Passos**
   - Citar efeitos colaterais possívei­s e sugerir revisão futura se necessário

## Exemplo de Saída
```diff
--- a/src/module.py
+++ b/src/module.py
@@ def divide(a, b):
-    return a / b
+    if b == 0:
+        raise ValueError("division by zero")
+    return a / b
```

**Teste adicionado**
```python
def test_divide_by_zero():
    import pytest
    with pytest.raises(ValueError):
        divide(1, 0)
```

---
Use este template apenas se o problema for um bug localizado; para refatorações maiores use `Plan-and-Solve`. 