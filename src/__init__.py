# -*- coding: utf-8 -*-

import importlib, sys

# ------------------------------------------------------------------
# Aliases de módulos para compatibilidade (importações sem prefixo 'src.')
# ------------------------------------------------------------------

try:
    # chunking
    sys.modules.setdefault('chunking', importlib.import_module('src.chunking'))
    sys.modules.setdefault('chunking.semantic_chunker', importlib.import_module('src.chunking.semantic_chunker'))
    sys.modules.setdefault('chunking.recursive_chunker', importlib.import_module('src.chunking.recursive_chunker'))
    # retrieval
    sys.modules.setdefault('retrieval', importlib.import_module('src.retrieval'))
    sys.modules.setdefault('retrieval.retriever', importlib.import_module('src.retrieval.retriever'))
    # pipeline
    sys.modules.setdefault('rag_pipeline', importlib.import_module('src.rag_pipeline'))
    # prompt_selector
    sys.modules.setdefault('prompt_selector', importlib.import_module('src.prompt_selector'))
    
    # Importar e exportar prompt_selector para compatibilidade
    from .prompt_selector import select_prompt, classify_query
    prompt_selector = importlib.import_module('src.prompt_selector')
    
except (ModuleNotFoundError, ImportError):
    pass
