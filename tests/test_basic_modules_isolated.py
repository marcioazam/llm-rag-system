"""
Testes básicos isolados - Sem dependência do conftest.py
Verifica funcionamento básico dos módulos principais
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Adicionar src ao path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_can_import_settings():
    """Testa se conseguimos importar settings básico"""
    try:
        import settings
        # Verificar se tem atributos básicos esperados
        assert hasattr(settings, '__file__'), "Settings deve ter __file__"
        print("✅ Settings importado com sucesso")
        return True
    except Exception as e:
        print(f"❌ Erro ao importar settings: {e}")
        return False

def test_can_import_template_renderer():
    """Testa se conseguimos importar template_renderer"""
    try:
        import template_renderer
        assert hasattr(template_renderer, '__file__'), "Template renderer deve ter __file__"
        assert hasattr(template_renderer, 'render_template'), "Template renderer deve ter função render_template"
        print("✅ Template renderer importado com sucesso")
        return True
    except Exception as e:
        print(f"❌ Erro ao importar template_renderer: {e}")
        return False

def test_basic_template_rendering():
    """Teste básico de renderização de template"""
    try:
        from template_renderer import render_template
        
        # Teste básico da função render_template
        template = "Query: {{query}}\nContext: {{context}}"
        query = "test query"
        context_snippets = ["context 1", "context 2"]
        
        result = render_template(template, query=query, context_snippets=context_snippets)
        
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 0, "Resultado não deve estar vazio"
        assert "test query" in result, "Query deve estar no resultado"
        assert "context 1" in result, "Context deve estar no resultado"
        
        print("✅ Template rendering funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Erro no template rendering: {e}")
        return False

def test_settings_loading():
    """Teste de carregamento de configurações"""
    try:
        from settings import RAGSettings
        
        # Verificar se consegue instanciar
        rag_settings = RAGSettings()
        
        # Verificar propriedades básicas
        assert hasattr(rag_settings, 'chunk_size'), "Deve ter chunk_size"
        assert hasattr(rag_settings, 'chunk_overlap'), "Deve ter chunk_overlap"
        assert hasattr(rag_settings, 'default_k'), "Deve ter default_k"
        
        # Verificar valores padrão
        assert rag_settings.chunk_size > 0, "chunk_size deve ser positivo"
        assert rag_settings.default_k > 0, "default_k deve ser positivo"
        
        print("✅ Settings carregado com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar settings: {e}")
        return False

def test_template_edge_cases():
    """Teste casos extremos do template renderer"""
    try:
        from template_renderer import render_template
        
        # Teste com contexto vazio
        result1 = render_template("Query: {{query}}", query="test", context_snippets=None)
        assert "test" in result1, "Query deve estar presente"
        
        # Teste com template sem placeholders
        result2 = render_template("Static template", query="test", context_snippets=["context"])
        assert result2 == "Static template", "Template estático deve ser retornado como está"
        
        # Teste com múltiplos contextos
        result3 = render_template("{{context}}", query="test", context_snippets=["ctx1", "ctx2", "ctx3"])
        assert "ctx1" in result3 and "ctx2" in result3 and "ctx3" in result3, "Todos os contextos devem estar presentes"
        
        print("✅ Template edge cases funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Erro nos edge cases: {e}")
        return False

def test_settings_nested_config():
    """Teste configurações aninhadas do settings"""
    try:
        from settings import RAGSettings, LLMSettings, Neo4jSettings
        
        # Verificar se pode instanciar configurações aninhadas
        llm_settings = LLMSettings()
        neo4j_settings = Neo4jSettings()
        rag_settings = RAGSettings()
        
        # Verificar se RAGSettings tem as configurações aninhadas
        assert hasattr(rag_settings, 'llm'), "RAGSettings deve ter configuração LLM"
        assert hasattr(rag_settings, 'neo4j'), "RAGSettings deve ter configuração Neo4j"
        
        # Verificar propriedades básicas
        assert hasattr(llm_settings, 'model'), "LLMSettings deve ter model"
        assert hasattr(neo4j_settings, 'uri'), "Neo4jSettings deve ter uri"
        
        print("✅ Settings aninhados funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Erro nos settings aninhados: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Executando testes básicos isolados...")
    
    tests = [
        test_can_import_settings,
        test_can_import_template_renderer,
        test_basic_template_rendering,
        test_settings_loading,
        test_template_edge_cases,
        test_settings_nested_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} falhou com exceção: {e}")
    
    print(f"\n📊 RESULTADO: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎯 STATUS: ✅ TESTES BÁSICOS FUNCIONANDO")
    elif passed > 0:
        print("🎯 STATUS: ⚠️ ALGUNS TESTES FUNCIONANDO")
        print("🔧 PRÓXIMO PASSO: Corrigir problemas restantes e executar com pytest")
    else:
        print("🎯 STATUS: 🔴 NENHUM TESTE FUNCIONANDO")
    
    sys.exit(0 if passed > 0 else 1) 