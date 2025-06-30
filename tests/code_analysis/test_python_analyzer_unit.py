from src.code_analysis.python_analyzer import PythonAnalyzer

PY_CODE = (
    "\"\"\"Módulo de exemplo\"\"\"\n"
    "import os\n"
    "from sys import path as sys_path\n\n"
    "class Foo:\n"
    "    \"\"\"Classe Foo\"\"\"\n"
    "    def method(self):\n"
    "        \"\"\"Método\"\"\"\n"
    "        pass\n\n"
    "def bar():\n"
    "    \"\"\"Função bar\"\"\"\n"
    "    return 42\n"
)

AN = PythonAnalyzer()

def test_extract_symbols():
    symbols = AN.extract_symbols(PY_CODE)
    names = {s["name"] for s in symbols}
    assert {"Foo", "bar"}.issubset(names)


def test_extract_relations():
    rels = AN.extract_relations(PY_CODE)
    targets = {r["target"] for r in rels}
    assert "os" in targets and "sys.path" in targets


def test_extract_docstrings():
    docs = AN.extract_docstrings(PY_CODE)
    assert any(d["type"] == "module" for d in docs)
    assert any(d["name"] == "Foo" for d in docs) 