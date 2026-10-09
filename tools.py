"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET  # arXiv answers with Atom XML

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()  # so `python tools.py` can read EXA_API_KEY from .env (the key stays on the host)

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRY_STATUSES = {429, 500, 502, 503, 504}
USER_AGENT = "deep-research-lab/1.0"
SUMMARY_CHARS = 600
PAGE_CHARS = 12000


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def _parse_retry_after(value):
    """Retry-After header (seconds) -> float, or None when absent / not a number (HTTP-date is ignored)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _retry_info(exc):
    """(is_retryable, retry_after) for an exception raised inside fn()."""
    if isinstance(exc, RetryableError):
        return True, exc.retry_after
    if isinstance(exc, httpx.TransportError):  # timeouts, connection resets...
        return True, None
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in RETRY_STATUSES:
        return True, _parse_retry_after(exc.response.headers.get("Retry-After"))
    return False, None


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError (or a transient HTTP/network error), wait and call it again.

    * server gave Retry-After -> wait that long (capped at `cap`);
    * otherwise exponential backoff base * 2**attempt, capped at `cap`, with random jitter;
    * last attempt failed -> re-raise WITHOUT sleeping;
    * any other exception (e.g. a programming error, HTTP 400/401) is NOT retried.
    """
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified right below
            retryable, retry_after = _retry_info(exc)
            if not retryable or attempt == attempts - 1:
                raise
            if retry_after is not None:
                delay = min(float(retry_after), cap)
            else:
                ceiling = min(cap, base * 2 ** attempt)
                delay = ceiling / 2 + random.uniform(0, ceiling / 2)  # "equal jitter": never above cap
            time.sleep(delay)


# ---- small shared helpers ----
def _redact(text):
    """Hide the Exa key (it travels in the URL, so it can leak into exception messages)."""
    text = str(text)
    key = os.environ.get("EXA_API_KEY", "").strip()
    if key:
        text = text.replace(key, "***")
    return re.sub(r"(exaApiKey=)[^&\s'\"]+", r"\1***", text)


def _error(exc):
    return _redact(f"ERROR: {type(exc).__name__}: {exc}")[:500]


def _clean(text):
    return " ".join(str(text or "").split())


def _clip(text, limit, marker=""):
    return text if len(text) <= limit else text[:limit] + marker


def _to_int(value, default, lo, hi):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _request(method, url, *, params=None, json_body=None, headers=None, timeout=30):
    """One HTTP call. 429/5xx -> HTTPStatusError and network errors -> TransportError: with_retry retries those."""
    merged = {"User-Agent": USER_AGENT, **(headers or {})}
    resp = httpx.request(method, url, params=params, json=json_body, headers=merged,
                         timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp


# ---- TODO 2: arXiv ----
ARXIV_MIN_GAP = 3.0  # seconds between two arXiv calls (their API etiquette)
_ATOM = "{http://www.w3.org/2005/Atom}"
_arxiv_lock = threading.Lock()  # subagents run in parallel threads: serialise arXiv calls
_arxiv_last = None


def _arxiv_get(params):
    global _arxiv_last
    with _arxiv_lock:
        if _arxiv_last is not None:
            wait = ARXIV_MIN_GAP - (time.monotonic() - _arxiv_last)
            if wait > 0:
                time.sleep(wait)
        try:
            return _request("GET", ARXIV_URL, params=params)
        finally:
            _arxiv_last = time.monotonic()


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        terms = [t for t in re.findall(r"[^\W_]+", str(query)) if t not in {"AND", "OR", "NOT", "ANDNOT"}][:12]
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{t}" for t in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": _to_int(max_results, 10, 1, 30),
        }
        resp = with_retry(lambda: _arxiv_get(params), attempts=6, base=2.0, cap=60.0)  # arXiv 429s are common
        records = []
        for entry in ET.fromstring(resp.content).findall(f"{_ATOM}entry"):
            raw_id = entry.findtext(f"{_ATOM}id") or ""
            if "/abs/" not in raw_id:  # arXiv reports errors as a fake <entry> without /abs/
                continue
            arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1].strip())
            records.append({
                "id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "published": (entry.findtext(f"{_ATOM}published") or "")[:10],
                "title": _clean(entry.findtext(f"{_ATOM}title")),
                "summary": _clip(_clean(entry.findtext(f"{_ATOM}summary")), SUMMARY_CHARS),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001 - tools never raise
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_records(items, prefer_ai_summary=False):
    """Map Hugging Face items ({"paper": {...}, ...}) to compact records; skip items without paper.id."""
    if not isinstance(items, list):
        raise ValueError("unexpected response from Hugging Face (not a list)")
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        paper = item["paper"] if isinstance(item.get("paper"), dict) else item
        paper_id = paper.get("id")
        if not paper_id:
            continue
        summary = (paper.get("ai_summary") if prefer_ai_summary else None) or paper.get("summary") or item.get("summary")
        records.append({
            "id": str(paper_id),
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clip(_clean(summary), SUMMARY_CHARS),
            "upvotes": paper.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "",
            "stars": paper.get("githubStars") or 0,
        })
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": _to_int(limit, 30, 1, 100)}
        if str(date).strip():
            params["date"] = str(date).strip()
        resp = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        records = _hf_records(resp.json())
        words = str(keyword).lower().split()
        if words:
            records = [r for r in records if all(w in f"{r['title']} {r['summary']}".lower() for w in words)]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not str(query).strip():
            return "NO RESULTS"
        params = {"q": str(query).strip(), "limit": _to_int(limit, 10, 1, 50)}
        resp = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params))
        records = _hf_records(resp.json(), prefer_ai_summary=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
# Exa's free tier answers HTTP 200 with a "rate limited" message instead of 429. We look for it in result._meta and in
# short answers. If your own curl test (GUIDE 1.4) shows a different flag, adjust _RATE_LIMIT_RE / _exa_rate_limited.
_RATE_LIMIT_RE = re.compile(r"rate[\s_-]?limit|throttl|too many requests|\b429\b|exaapikey", re.IGNORECASE)


def _exa_rate_limited(result, text):
    meta = result.get("_meta")
    if meta and _RATE_LIMIT_RE.search(json.dumps(meta)):
        return True
    return len(text) < 1000 and bool(_RATE_LIMIT_RE.search(text))  # a short message, not a real page about rate limits


def _mcp_payload(resp):
    """The HTTP body is JSON or server-sent events ("data: {...}"): return the JSON-RPC message."""
    body = resp.text
    payload = None
    if body.lstrip().startswith("{"):
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
    if payload is None:
        for line in body.splitlines():
            if line.startswith("data:"):
                try:
                    obj = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                    payload = obj
                    break
    if not isinstance(payload, dict):
        raise ValueError("Exa returned an unreadable response")
    return payload


def _exa_call(tool_name, arguments):
    """One MCP tools/call over HTTP. Returns the text content. Retries rate limits (also the HTTP-200 kind)."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    key = os.environ.get("EXA_API_KEY", "").strip()
    params = {"exaApiKey": key} if key else None  # the key goes in the URL -> redact it from any error text

    def attempt():
        payload = _mcp_payload(_request("POST", EXA_URL, params=params, json_body=body, headers=headers, timeout=60))
        if "error" in payload:
            err = payload["error"]
            message = err.get("message", "") if isinstance(err, dict) else str(err)
            if _RATE_LIMIT_RE.search(message):
                raise RetryableError("Exa rate limit")
            raise ValueError(f"Exa error: {_clip(message, 300)}")
        result = payload.get("result") or {}
        parts = result.get("content") or []
        text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text").strip()
        if _exa_rate_limited(result, text):
            raise RetryableError("Exa rate limit")
        if result.get("isError"):
            raise ValueError(f"Exa tool error: {_clip(text, 300)}")
        return text

    return with_retry(attempt, attempts=7, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not str(query).strip():
            return "NO RESULTS"
        arguments = {
            "query": str(query).strip(),
            "objective": str(objective).strip() or f"Find the most relevant pages for: {str(query).strip()}",
            "numResults": _to_int(num_results, 5, 1, 10),
        }
        text = _exa_call("web_search_exa", arguments)
        return _clip(text, PAGE_CHARS, "\n...[truncated]") if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        url = str(url).strip()
        if not url.startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        text = _exa_call("web_fetch_exa", {"urls": [url]})
        return _clip(text, PAGE_CHARS, "\n...[truncated]") if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")