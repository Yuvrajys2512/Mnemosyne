"""
Run the Mnemosyne eval suite.

Usage:
    python -m mnemosyne.eval                       # Groq provider (needs GROQ_API_KEY)
    python -m mnemosyne.eval --provider ollama     # Ollama fallback
    python -m mnemosyne.eval --dimension recall_accuracy
    python -m mnemosyne.eval --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from mnemosyne.eval.runner import run_eval
from mnemosyne.eval.scenarios import (
    ALL_SCENARIOS,
    CONSOLIDATION_SCENARIOS,
    EDGE_CASE_SCENARIOS,
    RECALL_SCENARIOS,
    RETENTION_SCENARIOS,
)
from mnemosyne.providers import GroqProvider, OllamaProvider

DIMENSION_MAP = {
    "recall_accuracy": RECALL_SCENARIOS,
    "consolidation": CONSOLIDATION_SCENARIOS,
    "retention": RETENTION_SCENARIOS,
    "edge_cases": EDGE_CASE_SCENARIOS,
}


def _build_provider(args):
    if args.provider == "ollama":
        return OllamaProvider(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
        )
    api_key = args.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY not set. "
            "Export it or use --provider ollama.",
            file=sys.stderr,
        )
        sys.exit(1)
    return GroqProvider(api_key=api_key, model=args.groq_model)


def _print_report(report, fmt: str) -> None:
    if fmt == "json":
        data = {
            "pass_rate": report.pass_rate,
            "mean_score": report.mean_score,
            "passed": report.passed,
            "failed": report.failed,
            "total": report.total,
            "results": [
                {
                    "scenario": r.scenario_name,
                    "dimension": r.dimension,
                    "score": r.score,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        print(json.dumps(data, indent=2))
        return

    print("\n" + "=" * 60)
    print(f"  MNEMOSYNE EVAL — {report.passed}/{report.total} passed  "
          f"({report.pass_rate:.0%})")
    print("=" * 60)

    for dim, results in report.by_dimension().items():
        passed = sum(1 for r in results if r.passed)
        print(f"\n  {dim}  [{passed}/{len(results)}]")
        for r in results:
            icon = "✓" if r.passed else "✗"
            err = f"  ERR: {r.error}" if r.error else ""
            print(f"    {icon} {r.scenario_name:<45} {r.score:.2f}{err}")

    print("\n" + "=" * 60)
    print(f"  Overall pass rate : {report.pass_rate:.1%}")
    print(f"  Mean score        : {report.mean_score:.3f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mnemosyne eval suite")
    parser.add_argument("--provider", choices=["groq", "ollama"], default="groq")
    parser.add_argument("--groq-api-key", default=None)
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--dimension", choices=list(DIMENSION_MAP.keys()), default=None)
    parser.add_argument("--storage-path", default=None)
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    args = parser.parse_args()

    if args.list:
        for s in ALL_SCENARIOS:
            print(f"{s.dimension:<20}  {s.name}")
        return

    scenarios = DIMENSION_MAP[args.dimension] if args.dimension else ALL_SCENARIOS
    provider = _build_provider(args)

    with tempfile.TemporaryDirectory(prefix="mnemosyne_eval_", ignore_cleanup_errors=True) as tmpdir:
        storage = args.storage_path or tmpdir
        report = run_eval(
            scenarios=scenarios,
            provider=provider,
            storage_path=storage,
            pass_threshold=args.pass_threshold,
        )

    _print_report(report, fmt=args.format)
    sys.exit(0 if report.pass_rate >= args.pass_threshold else 1)


if __name__ == "__main__":
    main()
