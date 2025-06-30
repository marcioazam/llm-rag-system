from __future__ import annotations

"""Utility for classifying user queries and selecting the most appropriate prompt template.

This is **step 5 & 6** of the prompt-integration roadmap.
The selector applies lightweight heuristics so it can run quickly without
external ML models. It can be replaced by embedding similarity later.
"""

from pathlib import Path
import json
import re
from typing import Dict, Tuple, Any

# ---------------------------------------------------------------------------
# Locate registry.json: prefer src/prompts/, fallback to project-root/prompts/
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent
_CANDIDATES = [
    _BASE_DIR / "prompts" / "registry.json",         # src/prompts/
    _BASE_DIR.parent / "prompts" / "registry.json",  # ../prompts/ (legacy)
]

_REGISTRY_PATH: Path | None = None
for _p in _CANDIDATES:
    if _p.exists():
        _REGISTRY_PATH = _p
        break

if _REGISTRY_PATH is None:
    raise FileNotFoundError(
        "Prompt registry not found in expected locations: " + ", ".join(str(p) for p in _CANDIDATES)
    )

with _REGISTRY_PATH.open(encoding="utf-8") as fp:
    _PROMPT_REGISTRY: Dict[str, Dict[str, Any]] = {item["id"]: item for item in json.load(fp)}

# ---------------------------------------------------------------------------
# Heuristic keyword mapping (can be fine-tuned through experimentation)
# ---------------------------------------------------------------------------
_KEYWORD_MAP = {
    "bugfix": [
        r"traceback|exception|error|stack ?trace|undefined|not found|null pointer",
    ],
    "review": [
        r"pull request|code review|revisar (?:este )?pr|revis[aã]o de c[óo]digo|review",
    ],
    "perf": [
        r"lento|performance|lat[êe]ncia|slow|optimi[sz]e|throughput|cpu|memory|high latency|latency issues",
    ],
    "arch": [
        r"arquitetura|architecture|adr|design decision|decisão arquitetural",
    ],
    # testgen deve vir ANTES de test para capturar "gerar teste" corretamente
    "testgen": [
        r"gerar test|criar test|generate test|test case|cobertura de test|melhorar cobertura",
    ],
    "test": [
        r"(?<!gerar )(?<!criar )(?<!generate )teste|unit test|coverage|mock|jest|pytest|cypress",
    ],
    "data_viz": [
        r"gr[áa]fico|plot|visualiza[çc][ãa]o|chart|histogram|scatter|line plot|bar chart",
    ],
    "ci": [
        r"ci|pipeline|github actions|gitlab ci|jenkins|build falhou|build failed",
    ],
}

_DEFAULT_TYPE = "geral"

_TYPE_TO_PROMPTS = {
    "bugfix": ["quick_fix_bug", "plan_and_solve"],
    "review": ["code_review_checklist"],
    "perf": ["performance_optimization_protocol", "plan_and_solve"],
    "arch": ["architecture_decision_records"],
    "test": ["testing_standards"],
    "testgen": ["test_case_generator"],
    "data_viz": ["data_viz_insight"],
    "ci": ["ci_pipeline_troubleshooter"],
    "geral": ["plan_and_solve"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query(query: str) -> str:
    """Return a coarse task type for the query.

    The function uses simple regex matching for speed. The first matching
    category is returned; if none match, 'geral' is returned.
    """
    lowered = query.lower()
    for qtype, patterns in _KEYWORD_MAP.items():
        for pat in patterns:
            if re.search(pat, lowered):
                return qtype
    return _DEFAULT_TYPE


def select_prompt(query: str, depth: str = "quick") -> Tuple[str, str]:
    """Select the most suitable prompt template.

    Parameters
    ----------
    query: str
        The user query.
    depth: str, optional
        'quick' or 'deep'. Affects whether we allow heavy templates.

    Returns
    -------
    Tuple[id, template_text]
        id: template identifier from the registry
        template_text: raw markdown content of the template
    """
    qtype = classify_query(query)
    candidate_ids = _TYPE_TO_PROMPTS.get(qtype, _TYPE_TO_PROMPTS[_DEFAULT_TYPE])

    # Filter by depth if needed (skip deep_dive templates when depth == 'quick')
    filtered = []
    for pid in candidate_ids:
        scope = _PROMPT_REGISTRY[pid]["scope"]
        if depth == "quick" and scope in {"deep_dive", "full_solution"}:
            continue
        filtered.append(pid)

    if not filtered:
        filtered = candidate_ids  # fallback to at least one

    prompt_id = filtered[0]
    prompt_meta = _PROMPT_REGISTRY[prompt_id]

    prompt_path = _REGISTRY_PATH.parent / prompt_meta["file"]
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file '{prompt_path}' not found for id '{prompt_id}'.")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    return prompt_id, prompt_text


# Convenience: expose registry for external inspection
REGISTRY = _PROMPT_REGISTRY 