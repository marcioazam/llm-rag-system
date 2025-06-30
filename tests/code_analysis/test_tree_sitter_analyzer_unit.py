import sys, types, importlib

import pytest

# Stub tree_sitter and tree_sitter_languages before import
parser_mod = types.ModuleType("tree_sitter")
class _DummyParser:
    def set_language(self, lang):
        pass
    def parse(self, code):
        class _Node:
            start_byte = 0
            end_byte = 0
            start_point = (0, 0)
            def __init__(self):
                self.root_node = self
        return _Node()
parser_mod.Parser = _DummyParser
sys.modules["tree_sitter"] = parser_mod

tsl_mod = types.ModuleType("tree_sitter_languages")

def _get_language(lang):
    class _Lang:
        def query(self, q):
            class _Q:
                def captures(self, node):
                    return []  # no symbols
            return _Q()
    return _Lang()

tsl_mod.get_language = _get_language
sys.modules["tree_sitter_languages"] = tsl_mod

# Now import analyzer
tsa_mod = importlib.import_module("src.code_analysis.tree_sitter_analyzer")
TreeSitterAnalyzer = tsa_mod.TreeSitterAnalyzer  # type: ignore


def test_extract_symbols_empty():
    analyzer = TreeSitterAnalyzer("javascript")
    symbols = analyzer.extract_symbols("function foo() {}")
    assert symbols == [] 