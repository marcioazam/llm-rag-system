
# Correção para model_router.py
# Substitua a linha que causa erro por:

# Ao invés de:
# model_name = model["name"]  # Isso causa o erro

# Use:
model_name = model.get("name", "unknown")  # Método seguro

# Ou, se quiser ser mais robusto:
model_name = model.get("name") or model.get("name") or model.get("model") or "unknown"

# Exemplo de função corrigida:
def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("models", []):
                # Método seguro para acessar o nome
                name = model.get("name") or model.get("name") or model.get("model") or "unknown"
                models.append({
                    "name": name,
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", "")
                })
            return models
        return []
    except Exception as e:
        logger.error(f"Erro ao obter modelos: {e}")
        return []
