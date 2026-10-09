"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"\W+", "-", str(topic).lower()).strip("-")  # "/" "." and spaces are non-word: no path escape
    return slug[:60].strip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f'Research topic: "{topic.strip()}"\n\n'
        "Run the full workflow from your instructions: plan, delegate to researcher subagents in parallel, merge the "
        "notes into sources.json, write the report body, run the finalizer and the validator until it prints OK, "
        "and spot-check a few citations. Stop when the report is final."
    )


def _field(message, name):
    """Messages are LangChain objects or plain dicts: read a field from either."""
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Lead messages only: subagent tokens are not included, so this undercounts the real cost.
    """
    calls = Counter()
    tokens_in = tokens_out = 0
    for message in messages or []:
        for call in _field(message, "tool_calls") or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                calls[name] += 1
        usage = _field(message, "usage_metadata") or {}
        tokens_in += int(usage.get("input_tokens", 0) or 0)
        tokens_out += int(usage.get("output_tokens", 0) or 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": calls.get("task", 0),
        "tool_calls": dict(calls),
        "tokens": {"input": tokens_in, "output": tokens_out},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    A failed run must never leave an empty or half-written report behind: everything is validated BEFORE any write.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = files.get(REPORT_PATH)
    raw_sources = files.get(SOURCES_PATH)
    if not report or not report.strip():
        raise RuntimeError(f"the agent did not produce a report ({REPORT_PATH} is missing or empty)")
    if not raw_sources:
        raise RuntimeError(f"the agent did not produce {SOURCES_PATH}")
    try:
        sources = json.loads(raw_sources.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list):
        raise RuntimeError(f"{SOURCES_PATH} must be a JSON list")

    families = sorted({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")})
    meta = {"topic": topic.strip(), **summarize(messages, elapsed, model_name),
            "n_sources": len(sources), "source_families": families}

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report)
    return report_path


def _model_name():
    return os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or "unknown"


def _warn_if_invalid(report_path):
    """Quick host-side check with YOUR validator (no network, no cost). Only a warning: the run itself succeeded."""
    try:
        from check_citations import check
        sources_path = report_path.with_name(report_path.name[: -len(".md")] + ".sources.json")
        problems = check(report_path.read_text(encoding="utf-8"), json.loads(sources_path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not run check_citations here: {exc}", file=sys.stderr)
        return
    if problems:
        print("WARNING: your validator reports problems in the saved report (fix the PROMPTS and run again):",
              file=sys.stderr)
        for problem in problems[:10]:
            print(f"  - {problem}", file=sys.stderr)


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    model_name = _model_name()
    start = time.monotonic()
    print(f"Researching: {topic!r} (model {model_name}) ... this can take several minutes")
    try:
        with open_sandbox() as backend:                 # always stopped and removed, even on errors
            made = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if getattr(made, "exit_code", 0) not in (0, None):
                raise RuntimeError(f"cannot create the work folders in the sandbox: {made.output}")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
            report_path = save_outputs(backend, topic, result.get("messages", []),
                                       time.monotonic() - start, model_name)
    except Exception as exc:  # noqa: BLE001 - sandbox cleanup already ran; report ONE clear failure and write nothing
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Saved: {report_path}")
    _warn_if_invalid(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))