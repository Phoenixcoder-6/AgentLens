"""
app/main.py — AgentLens Pipeline Runner
========================================
CLI entry point for running the reference pipeline.

Usage:
    conda activate agentlens
    python app/main.py
    python app/main.py --topic "The history of the Apollo space program"

Day 4: Runs the pipeline and prints results to the terminal.
       No capture or storage yet — that comes in Days 5-8.
"""

from __future__ import annotations

import argparse
import sys
import os
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline import run_pipeline
from capture.session import CaptureSession
from schema.models import StepStatus


DEFAULT_TOPIC = "The history of the Apollo space program"


def _separator(title: str = "", width: int = 70) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return f"{'─' * pad} {title} {'─' * pad}"
    return "─" * width


def _wrap(text: str, width: int = 70, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def print_results(state: dict, topic: str) -> None:
    """Pretty-print the pipeline results to the terminal."""
    print()
    print(_separator("AGENTLENS PIPELINE RUN"))
    print(f"  Topic: {topic}")
    print(_separator())
    print()

    # ── Researcher output ──
    print(_separator("RESEARCHER OUTPUT"))
    print()
    for line in state["research_findings"].splitlines():
        print(f"  {line}")
    print()
    print(f"  Sources cited : {state['source_count']}")
    print(f"  Entities found: {state['entity_count']}")

    # ── Writer output ──
    print()
    print(_separator("WRITER OUTPUT"))
    print()
    for line in state["written_report"].splitlines():
        print(f"  {line}")

    # ── Verifier output ──
    print()
    print(_separator("VERIFIER OUTPUT"))
    print()
    status = "APPROVED" if state["verified"] else "NEEDS REVISION"
    print(f"  Status : {status}")
    print(f"  Result : {state['verification_result']}")
    if state["revision_notes"]:
        print()
        print("  Revision notes:")
        for line in state["revision_notes"].splitlines():
            print(f"    {line}")

    print()
    print(_separator())
    print(f"  Pipeline complete. Verified: {state['verified']}")
    print(_separator())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AgentLens reference pipeline (Researcher → Writer → Verifier)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help=f'Research topic to run the pipeline on (default: "{DEFAULT_TOPIC}")',
    )
    args = parser.parse_args()

    print(f"\nStarting pipeline for topic: '{args.topic}'")
    print("Running agents: Researcher → Writer → Verifier ...\n")

    try:
        CaptureSession.start_trace(workflow="research_report_pipeline")
        state = run_pipeline(topic=args.topic)
        CaptureSession.end_trace()
        print_results(state, args.topic)
    except EnvironmentError as e:
        CaptureSession.end_trace(status=StepStatus.ERROR, error=str(e))
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        CaptureSession.end_trace(status=StepStatus.ERROR, error=str(e))
        print(f"[ERROR] Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
