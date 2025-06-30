import types, importlib, sys

# Import original module fresh
sys.modules.pop("src.code_analysis.tree_sitter_analyzer", None)

tsa = importlib.import_module("src.code_analysis.tree_sitter_analyzer")
TreeSitterAnalyzer = tsa.TreeSitterAnalyzer  # type: ignore


class _Node:
    def __init__(self, start, end):
        self.start_byte = start
        self.end_byte = end
        self.start_point = (0, start)


class _DummyQuery:
    def __init__(self, nodes):
        self._nodes = nodes
    def captures(self, _root):  # noqa: D401
        return [(n, None) for n in self._nodes]


class _DummyLang:
    def __init__(self, nodes):
        self._nodes = nodes
    def query(self, _src):  # noqa: D401
        return _DummyQuery(self._nodes)


class _DummyParser:
    def __init__(self, nodes):
        self.language = _DummyLang(nodes)
    def parse(self, _bytes):  # noqa: D401
        return types.SimpleNamespace(root_node=None)
    def set_language(self, _lang):
        pass


def _make_analyzer(lang: str, code: str, nodes):
    an = TreeSitterAnalyzer(lang)
    an.parser = _DummyParser(nodes)  # type: ignore
    return an.extract_relations(code)


def test_js_import_relation():
    code = 'import "fs";'
    # Node spans the quoted string including quotes
    nodes = [_Node(7, 11)]  # positions of "\"fs\""
    rels = _make_analyzer("javascript", code, nodes)
    assert rels and rels[0]["target"] == "fs"


def test_go_import_relation():
    code = 'import "fmt"'
    nodes = [_Node(7, 12)]
    rels = _make_analyzer("go", code, nodes)
    assert rels and rels[0]["target"] == "fmt" 