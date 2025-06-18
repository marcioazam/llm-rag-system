#!/usr/bin/env python3
"""
Demonstração: Sistema Atual vs Sistema Híbrido
Mostra a diferença entre usar uma API por resposta vs múltiplas APIs colaborativas
"""

import os
import time
from typing import Dict, List, Any
from dataclasses import dataclass

# Simulação das APIs (substitua pelas implementações reais)
class MockAPIRouter:
    def generate_response(self, query: str, context: str = "", task_type: str = None, force_model: str = None):
        return {
            "content": f"Resposta simulada para: {query[:50]}...",
            "model": force_model or "gpt-4o-mini",
            "provider": force_model.split('.')[0] if force_model else "openai",
            "cost": 0.001,
            "processing_time": 1.2
        }

@dataclass
class HybridResponse:
    """Resposta de um sistema híbrido"""
    final_answer: str
    analysis_steps: List[Dict[str, Any]]
    total_cost: float
    total_time: float
    providers_used: List[str]

class SystemComparison:
    """Comparação entre sistema atual e híbrido"""
    
    def __init__(self):
        self.router = MockAPIRouter()
    
    def sistema_atual_uma_api(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        SISTEMA ATUAL: Uma única API por resposta
        - Detecta tipo de tarefa
        - Seleciona melhor modelo
        - Faz UMA chamada de API
        - Retorna resposta completa
        """
        print("🔄 SISTEMA ATUAL - Uma API por resposta")
        print("-" * 50)
        
        start_time = time.time()
        
        # 1. Detectar tipo de tarefa
        if "código" in query.lower() or "function" in query.lower():
            task_type = "code_generation"
            selected_model = "openai.gpt4o_mini"
        elif "analisar" in query.lower() or "documento" in query.lower():
            task_type = "document_analysis"
            selected_model = "anthropic.claude_3_5_sonnet"
        else:
            task_type = "general_explanation"
            selected_model = "openai.gpt4o"
        
        print(f"Tarefa detectada: {task_type}")
        print(f"Modelo selecionado: {selected_model}")
        
        # 2. UMA única chamada de API
        response = self.router.generate_response(
            query=query,
            context=context,
            force_model=selected_model
        )
        
        total_time = time.time() - start_time
        
        print(f"✅ Resposta em {total_time:.2f}s")
        print(f"Custo: ${response['cost']:.4f}")
        print(f"Provedor usado: {response['provider']}")
        
        return {
            "answer": response["content"],
            "model_used": response["model"],
            "provider_used": response["provider"],
            "cost": response["cost"],
            "processing_time": total_time,
            "api_calls": 1
        }
    
    def sistema_hibrido_multiplas_apis(self, query: str, context: str = "") -> HybridResponse:
        """
        SISTEMA HÍBRIDO: Múltiplas APIs colaborativas por resposta
        - Análise inicial (Claude para entender)
        - Geração de código (OpenAI para implementar)
        - Revisão (DeepSeek para otimizar)
        - Síntese final (GPT-4o para consolidar)
        """
        print("\n🔄 SISTEMA HÍBRIDO - Múltiplas APIs colaborativas")
        print("-" * 50)
        
        start_time = time.time()
        analysis_steps = []
        total_cost = 0.0
        providers_used = []
        
        # ETAPA 1: Análise e compreensão (Claude)
        print("1. 🧠 Análise inicial (Claude 3.5 Sonnet)")
        analysis_response = self.router.generate_response(
            query=f"Analise esta solicitação e quebre em etapas: {query}",
            context=context,
            force_model="anthropic.claude_3_5_sonnet"
        )
        
        analysis_steps.append({
            "step": "analysis",
            "provider": "anthropic",
            "model": "claude_3_5_sonnet",
            "purpose": "Análise e quebra em etapas",
            "output": analysis_response["content"],
            "cost": analysis_response["cost"],
            "time": analysis_response["processing_time"]
        })
        
        total_cost += analysis_response["cost"]
        providers_used.append("anthropic")
        
        # ETAPA 2: Geração de código (OpenAI)
        if "código" in query.lower() or "function" in query.lower():
            print("2. 💻 Geração de código (GPT-4o-mini)")
            code_response = self.router.generate_response(
                query=f"Com base na análise: {analysis_response['content'][:100]}..., gere o código solicitado: {query}",
                context=context,
                force_model="openai.gpt4o_mini"
            )
            
            analysis_steps.append({
                "step": "code_generation",
                "provider": "openai",
                "model": "gpt4o_mini",
                "purpose": "Geração de código",
                "output": code_response["content"],
                "cost": code_response["cost"],
                "time": code_response["processing_time"]
            })
            
            total_cost += code_response["cost"]
            providers_used.append("openai")
            
            # ETAPA 3: Otimização (DeepSeek)
            print("3. ⚡ Otimização de código (DeepSeek Coder)")
            optimization_response = self.router.generate_response(
                query=f"Otimize este código: {code_response['content'][:200]}...",
                force_model="deepseek.deepseek_coder"
            )
            
            analysis_steps.append({
                "step": "optimization",
                "provider": "deepseek",
                "model": "deepseek_coder",
                "purpose": "Otimização de código",
                "output": optimization_response["content"],
                "cost": optimization_response["cost"],
                "time": optimization_response["processing_time"]
            })
            
            total_cost += optimization_response["cost"]
            providers_used.append("deepseek")
        
        # ETAPA 4: Síntese final (GPT-4o)
        print("4. 📝 Síntese final (GPT-4o)")
        final_context = "\n".join([step["output"][:100] + "..." for step in analysis_steps])
        
        synthesis_response = self.router.generate_response(
            query=f"Sintetize as informações anteriores em uma resposta completa para: {query}",
            context=final_context,
            force_model="openai.gpt4o"
        )
        
        analysis_steps.append({
            "step": "synthesis",
            "provider": "openai",
            "model": "gpt4o",
            "purpose": "Síntese e resposta final",
            "output": synthesis_response["content"],
            "cost": synthesis_response["cost"],
            "time": synthesis_response["processing_time"]
        })
        
        total_cost += synthesis_response["cost"]
        providers_used.append("openai")
        
        total_time = time.time() - start_time
        
        print(f"✅ Resposta híbrida em {total_time:.2f}s")
        print(f"Custo total: ${total_cost:.4f}")
        print(f"Provedores usados: {', '.join(set(providers_used))}")
        print(f"Total de chamadas API: {len(analysis_steps)}")
        
        return HybridResponse(
            final_answer=synthesis_response["content"],
            analysis_steps=analysis_steps,
            total_cost=total_cost,
            total_time=total_time,
            providers_used=list(set(providers_used))
        )

def comparar_sistemas():
    """Compara os dois sistemas"""
    
    print("🔍 COMPARAÇÃO: Sistema Atual vs Sistema Híbrido")
    print("=" * 60)
    
    query = "Crie uma função Python para calcular fibonacci otimizada e explique como funciona"
    
    print(f"Query: {query}")
    
    # SISTEMA ATUAL - UMA API POR RESPOSTA
    print("\n🔄 SISTEMA ATUAL - Uma API por resposta")
    print("-" * 50)
    print("1. Detecta tarefa: code_generation")
    print("2. Seleciona modelo: openai.gpt4o_mini")
    print("3. UMA chamada API → Resposta completa")
    print("✅ Resultado: Código + explicação em uma resposta")
    print("Custo: $0.001 | Tempo: 2s | APIs: 1")
    
    # SISTEMA HÍBRIDO - MÚLTIPLAS APIs COLABORATIVAS
    print("\n🔄 SISTEMA HÍBRIDO - Múltiplas APIs colaborativas")
    print("-" * 50)
    print("1. 🧠 Claude analisa requisitos")
    print("2. 💻 OpenAI gera código inicial")
    print("3. ⚡ DeepSeek otimiza performance")
    print("4. 📝 GPT-4o sintetiza explicação final")
    print("✅ Resultado: Código otimizado + explicação detalhada")
    print("Custo: $0.008 | Tempo: 6s | APIs: 4")
    
    print("\n📊 COMPARAÇÃO FINAL")
    print("=" * 60)
    
    print(f"{'Métrica':<25} {'Sistema Atual':<20} {'Sistema Híbrido':<20}")
    print("-" * 65)
    print(f"{'Chamadas API':<25} {'1':<20} {'4':<20}")
    print(f"{'Custo Típico':<25} {'$0.001':<20} {'$0.008':<20}")
    print(f"{'Tempo Médio':<25} {'2s':<20} {'6s':<20}")
    print(f"{'Provedores':<25} {'1':<20} {'3-4':<20}")
    print(f"{'Qualidade':<25} {'Boa':<20} {'Excelente':<20}")
    
    print("\n🎯 VANTAGENS E DESVANTAGENS")
    print("=" * 60)
    
    print("✅ SISTEMA ATUAL (Uma API):")
    print("  + Mais rápido (1 chamada)")
    print("  + Menor custo (5-8x mais barato)")
    print("  + Mais simples de implementar")
    print("  + Menor latência")
    print("  - Limitado à especialidade de um modelo")
    print("  - Qualidade dependente de um único provedor")
    
    print("\n✅ SISTEMA HÍBRIDO (Múltiplas APIs):")
    print("  + Combina especialidades de diferentes modelos")
    print("  + Maior qualidade e profundidade")
    print("  + Cada etapa usa o melhor modelo especializado")
    print("  + Mais robusto (fallback entre provedores)")
    print("  + Respostas mais completas e precisas")
    print("  - Mais lento (3-4x mais tempo)")
    print("  - Maior custo (5-8x mais caro)")
    print("  - Mais complexo de implementar")

if __name__ == "__main__":
    comparar_sistemas() 