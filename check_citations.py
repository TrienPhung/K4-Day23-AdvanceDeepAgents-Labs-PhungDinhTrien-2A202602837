"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

# [3]  [1, 2]  [1-3]  [2-3]; but not the Markdown link [3](url)
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
# fenced code blocks and inline code spans: citations inside them are ignored
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"(?m)^[ \t]*(?:[-*][ \t]+)?\[(\d+)\](.*)$")
_URL = re.compile(r"https?://[^\s<>\"]+")


def _expand(group):
    """'1, 2' -> [1, 2];  '1-3' -> [1, 2, 3]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _cited_numbers(body):
    """Numbers cited in the body, ignoring code spans/blocks and Markdown links."""
    cited = set()
    for i, segment in enumerate(_CODE.split(body)):
        if i % 2:  # odd indexes are code
            continue
        for match in _GROUP.finditer(segment):
            cited.update(_expand(match.group(1)))
    return cited


def _norm_url(url):
    return url.rstrip(".,;:")


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []

    # --- sources.json -------------------------------------------------------
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    by_n = {}
    seen_urls = {}
    for idx, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"sources[{idx}] is not an object")
            continue
        n = entry.get("n")
        url = entry.get("url")
        label = f"source [{n}]" if isinstance(n, int) and not isinstance(n, bool) else f"sources[{idx}]"

        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"{label}: n must be an integer (got {n!r})")
        elif n in by_n:
            problems.append(f"{label}: number n={n} appears more than once in sources.json")
        else:
            by_n[n] = entry

        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"{label}: url must start with http:// or https:// (got {url!r})")
        elif url in seen_urls:
            problems.append(f"{label}: duplicate url {url} (also {seen_urls[url]})")
        else:
            seen_urls[url] = label

    # --- split body / references -------------------------------------------
    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        problems.append("missing '## References' heading")
        body, ref_text = report_text, ""
    else:
        body = report_text[: headings[-1].start()]
        ref_text = report_text[headings[-1].end():]

    # --- citations in body vs sources --------------------------------------
    cited = _cited_numbers(body)
    for n in sorted(cited - set(by_n)):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(set(by_n) - cited):
        problems.append(f"source [{n}] never cited")

    # --- reference lines ----------------------------------------------------
    if headings:
        ref_lines = {}
        for match in _REF_LINE.finditer(ref_text):
            n = int(match.group(1))
            ref_lines.setdefault(n, []).append(match.group(2))

        for n in sorted(by_n):
            if n not in ref_lines:
                problems.append(f"source [{n}] has no line in ## References")
        for n, lines in sorted(ref_lines.items()):
            if n not in by_n:
                problems.append(f"## References has [{n}] which is not a source")
            if len(lines) > 1:
                problems.append(f"## References has number [{n}] {len(lines)} times")
            for rest in lines[:1]:
                urls = _URL.findall(rest)
                if len(urls) != 1:
                    problems.append(
                        f"reference [{n}] must contain exactly one URL (found {len(urls)})"
                        + ("; one line must not bundle several sources" if len(urls) > 1 else "")
                    )
                elif n in by_n:
                    expected = by_n[n].get("url")
                    if isinstance(expected, str) and _norm_url(urls[0]) != _norm_url(expected):
                        problems.append(
                            f"reference [{n}] url {urls[0]} != sources.json url {expected}"
                        )
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))