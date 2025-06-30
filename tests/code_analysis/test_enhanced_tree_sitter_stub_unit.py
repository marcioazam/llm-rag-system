import importlib, sys, types
import pytest

# Remover módulos para garantir stub livre
for modname in ["tree_sitter", "tree_sitter_languages"]:
    sys.modules.pop(modname, None)

analyzer_mod = importlib.import_module("src.code_analysis.enhanced_tree_sitter_analyzer")


def test_tree_sitter_availability_flag():
    assert isinstance(analyzer_mod.TREE_SITTER_AVAILABLE, bool)


def test_analyzer_init_behaviour():
    if analyzer_mod.TREE_SITTER_AVAILABLE:
        inst = analyzer_mod.EnhancedTreeSitterAnalyzer("python")
        assert inst.language == "python"
    else:
        with pytest.raises(RuntimeError):
            analyzer_mod.EnhancedTreeSitterAnalyzer("python") 