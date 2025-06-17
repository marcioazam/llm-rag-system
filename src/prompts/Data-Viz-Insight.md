# Data-Viz-Insight Prompt

## Quando usar
Quando o usuário precisar gerar um gráfico/visualização e interpretar os insights do dataset rapidamente.

## Estrutura
1. **Entendimento dos Dados**
   - Identificar variáveis, tipos e objetivo da visualização
2. **Escolha do Gráfico**
   - Tabela de referência rápida (ex.: séries temporais ⟶ line chart, distribuição ⟶ histogram, categorias ⟶ bar chart, correlação ⟶ scatter)
3. **Implementação em Código**
   - Biblioteca recomendada (Matplotlib/Seaborn/Plotly)
   - Código mínimo para gerar o gráfico
4. **Interpretação**
   - 2-3 bullets com os principais insights que o gráfico deve revelar
5. **Próximos Passos**
   - Sugestões de visualizações adicionais ou análise estatística complementar

## Exemplo de Saída
```python
import seaborn as sns
import matplotlib.pyplot as plt

data = df.groupby('month')['sales'].sum().reset_index()

sns.lineplot(data=data, x='month', y='sales', marker='o')
plt.title('Vendas Mensais 2024')
plt.ylabel('Total R$')
plt.show()
```

**Insights**
- Picos de venda em março e novembro (Black Friday)
- Queda acentuada em janeiro (pós-férias)
- Tendência geral de crescimento linear de ~5% ao mês

---
Use em conjunto com `Performance-Optimization-Protocol` se o gráfico revelar gargalos de performance. 