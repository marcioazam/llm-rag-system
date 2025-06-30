import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
STABLE_LIST = ROOT / "tests" / "stable_tests.txt"

if not STABLE_LIST.exists():
    print("Arquivo tests/stable_tests.txt não encontrado", file=sys.stderr)
    sys.exit(1)

with STABLE_LIST.open(encoding="utf-8") as fp:
    tests = [line.strip() for line in fp if line.strip() and not line.startswith("#")]

if not tests:
    print("Nenhum teste estável definido.", file=sys.stderr)
    sys.exit(1)

# Executa pytest com a lista de testes estáveis
command = [sys.executable, "-m", "pytest", *tests]
sys.exit(subprocess.call(command)) 