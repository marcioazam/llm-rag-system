import argparse
import re
import requests
from collections import defaultdict
from typing import Dict

METRICS_ENDPOINT = "http://localhost:8001/metrics"

PROMPT_USAGE_RE = re.compile(r"rag_prompt_usage_total\{prompt_id=\"(?P<pid>[^\"]+)\"} (?P<count>\d+)")
VARIANT_USAGE_RE = re.compile(r"rag_prompt_variant_total\{variant=\"(?P<variant>[^\"]+)\"} (?P<count>\d+)")


def fetch_metrics(url: str) -> str:
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.text


def parse_usage(metrics_text: str) -> Dict[str, int]:
    prompt_counts = defaultdict(int)
    for line in metrics_text.splitlines():
        m = PROMPT_USAGE_RE.match(line)
        if m:
            prompt_counts[m.group("pid")] += int(float(m.group("count")))
    return dict(prompt_counts)


def parse_variants(metrics_text: str) -> Dict[str, int]:
    variant_counts = defaultdict(int)
    for line in metrics_text.splitlines():
        m = VARIANT_USAGE_RE.match(line)
        if m:
            variant_counts[m.group("variant")] += int(float(m.group("count")))
    return dict(variant_counts)


def main():
    parser = argparse.ArgumentParser(description="Gera relatório simples de uso de prompts a partir das métricas Prometheus expostas pelo RAGPipeline.")
    parser.add_argument("--url", default=METRICS_ENDPOINT, help="Endpoint HTTP de métricas (default: %(default)s)")
    args = parser.parse_args()

    metrics = fetch_metrics(args.url)

    prompts = parse_usage(metrics)
    variants = parse_variants(metrics)

    if not prompts and not variants:
        print("Nenhuma métrica de uso de prompt encontrada. Certifique-se de que o servidor Prometheus esteja ativo e o pipeline tenha sido acionado.")
        return

    if prompts:
        print("\nUso por Prompt ID:\n-------------------")
        for pid, count in sorted(prompts.items(), key=lambda x: x[1], reverse=True):
            print(f"{pid:25s}  {count:>6d}")

    if variants:
        print("\nDistribuição A/B (with_prompt vs no_prompt):\n-------------------------------------------")
        total = sum(variants.values())
        for var, count in variants.items():
            pct = (count / total) * 100 if total else 0
            print(f"{var:12s}  {count:>6d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main() 