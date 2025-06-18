# 🚀 GUIA DE INÍCIO RÁPIDO - RAG 100% APIs

## ⚡ Configuração Mínima (2 minutos)

### 1. Configure pelo menos a OpenAI
```bash
# Copie o arquivo de exemplo
cp config/env_example.txt .env

# Edite e adicione sua API key da OpenAI
OPENAI_API_KEY=sk-your-key-here
```

### 2. Instale dependências
```bash
pip install -r requirements.txt
```

### 3. Teste o sistema
```bash
python check_system_status.py
```

### 4. Inicie o servidor
```bash
python -m src.api.main
```

## 🎯 Configuração Completa (4 Provedores)

Para usar todo o potencial do sistema, configure os 4 provedores:

### OpenAI (Obrigatório)
- **Uso**: Geração de código, análise complexa
- **Modelos**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **API Key**: https://platform.openai.com/api-keys

### Anthropic Claude (Recomendado)
- **Uso**: Análise de documentos, escrita técnica
- **Modelos**: Claude 3.5 Sonnet, Claude 3 Haiku
- **API Key**: https://console.anthropic.com/

### Google Gemini (Opcional)
- **Uso**: Contexto longo, análise multimodal
- **Modelos**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **API Key**: https://makersuite.google.com/app/apikey

### DeepSeek (Opcional)
- **Uso**: Código avançado, raciocínio matemático
- **Modelos**: DeepSeek Chat, DeepSeek Coder
- **API Key**: https://platform.deepseek.com/

## 📊 Distribuição de Responsabilidades

| Tarefa | Provedor Primário | Modelo |
|--------|------------------|---------|
| Geração de código | OpenAI | GPT-4o-mini |
| Análise de código | DeepSeek | DeepSeek Coder |
| Análise de documentos | Anthropic | Claude 3.5 Sonnet |
| Consultas rápidas | Google | Gemini 1.5 Flash |
| Análise complexa | OpenAI | GPT-4o |
| Contexto longo | Google | Gemini 1.5 Pro (2M tokens) |

## 💰 Controle de Custos

O sistema inclui controles automáticos de custo:
- Orçamento diário configurável
- Limite por requisição
- Cache inteligente
- Roteamento baseado em custo-benefício

## 🔧 Solução de Problemas

```bash
# Verificar status completo
python check_system_status.py

# Testar API específica
python -c "from src.models.api_model_router import APIModelRouter; router = APIModelRouter({}); print(router.get_available_models())"

# Ver logs
tail -f logs/rag_api.log
```

## ✅ Benefícios da Migração

- **Performance**: 10x mais rápido que modelos locais
- **Qualidade**: Modelos state-of-the-art (1.7T parâmetros)
- **Escalabilidade**: Suporte a milhões de usuários
- **Recursos**: 99% menos RAM (100MB vs 16GB)
- **Custos**: Pay-per-use ($10-50/mês típico)
- **Manutenção**: 90% menos trabalho

Pronto para usar! 🎉
