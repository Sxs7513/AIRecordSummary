from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Protocol, cast
from urllib.parse import quote, urlencode

import httpx

from l2_core.rag.adjudication.contracts import GroundedResearchFinding, GroundedSource

RATE_LIMIT_STATUS_CODE = 429
RATE_LIMIT_MAX_ATTEMPTS = 3
RATE_LIMIT_RETRY_SECONDS = 10.0

logger = logging.getLogger("rag")

CHROME_AIO_TAB_MARKER = "ars_aio_bridge=1"
CHROME_AIO_STABLE_POLLS = 3

_CHROME_NAVIGATE_SCRIPT = """
on run argv
    set targetUrl to item 1 of argv
    set tabMarker to item 2 of argv
    if application "Google Chrome" is not running then error "Google Chrome is not running"
    tell application "Google Chrome"
        if (count of windows) is 0 then error "Google Chrome has no open window"
        set targetTab to missing value
        repeat with chromeWindow in windows
            repeat with chromeTab in tabs of chromeWindow
                if (URL of chromeTab contains tabMarker) then
                    set targetTab to chromeTab
                    exit repeat
                end if
            end repeat
            if targetTab is not missing value then exit repeat
        end repeat
        if targetTab is missing value then
            set originalActiveTab to active tab index of window 1
            tell window 1 to set targetTab to make new tab at end of tabs with properties {URL:targetUrl}
            set active tab index of window 1 to originalActiveTab
        else
            set URL of targetTab to targetUrl
        end if
    end tell
    return "ok"
end run
"""

_CHROME_SNAPSHOT_SCRIPT = """
on run argv
    set tabMarker to item 1 of argv
    set javascriptSource to item 2 of argv
    if application "Google Chrome" is not running then error "Google Chrome is not running"
    tell application "Google Chrome"
        repeat with chromeWindow in windows
            repeat with chromeTab in tabs of chromeWindow
                if (URL of chromeTab contains tabMarker) then
                    return execute chromeTab javascript javascriptSource
                end if
            end repeat
        end repeat
    end tell
    error "AI Overview bridge tab is missing"
end run
"""

_AI_OVERVIEW_JAVASCRIPT = r"""
(() => {
  const containers = Array.from(document.querySelectorAll('[data-streaming-container]'));
  const candidates = containers
    .map((node) => ({ node, text: (node.innerText || '').trim() }))
    .filter((item) => item.text.length > 0)
    .sort((left, right) => right.text.length - left.text.length);
  if (candidates.length === 0) {
    return JSON.stringify({ text: '', sources: [], url: location.href, title: document.title });
  }
  const root = candidates[0].node;
  const seen = new Set();
  const sources = [];
  for (const anchor of root.querySelectorAll('a[href]')) {
    const url = new URL(anchor.href, location.href).href;
    if (!url.startsWith('http') || seen.has(url)) continue;
    seen.add(url);
    sources.push({ title: (anchor.innerText || anchor.getAttribute('aria-label') || '').trim(), url });
    if (sources.length >= 8) break;
  }
  return JSON.stringify({ text: candidates[0].text, sources, url: location.href, title: document.title });
})()
"""


class GroundedSearchClient(Protocol):
    async def search(self, proposal_id: str, query: str) -> GroundedResearchFinding: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChromeAiOverviewSnapshot:
    text: str
    sources: tuple[GroundedSource, ...]
    url: str = ""
    title: str = ""


class ChromeAutomation(Protocol):
    async def navigate(self, url: str) -> None: ...

    async def snapshot(self) -> ChromeAiOverviewSnapshot: ...


class AppleScriptChromeAutomation:
    """Control one dedicated background tab in an already-running macOS Chrome."""

    async def navigate(self, url: str) -> None:
        await self._run_script(_CHROME_NAVIGATE_SCRIPT, (url, CHROME_AIO_TAB_MARKER))

    async def snapshot(self) -> ChromeAiOverviewSnapshot:
        raw = await self._run_script(
            _CHROME_SNAPSHOT_SCRIPT,
            (CHROME_AIO_TAB_MARKER, _AI_OVERVIEW_JAVASCRIPT),
        )
        try:
            payload = cast(object, json.loads(raw))
        except ValueError as error:
            raise RuntimeError("Chrome returned an invalid AI Overview snapshot") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Chrome returned a non-object AI Overview snapshot")
        values = cast(dict[object, object], payload)
        sources_value = values.get("sources")
        sources: list[GroundedSource] = []
        if isinstance(sources_value, list):
            for source_value in cast(list[object], sources_value):
                if not isinstance(source_value, dict):
                    continue
                source = cast(dict[object, object], source_value)
                url = source.get("url")
                title = source.get("title")
                if isinstance(url, str) and url:
                    sources.append(GroundedSource(title=title if isinstance(title, str) else "", url=url))
        text = values.get("text")
        url = values.get("url")
        title = values.get("title")
        return ChromeAiOverviewSnapshot(
            text=text.strip() if isinstance(text, str) else "",
            sources=tuple(sources[:8]),
            url=url if isinstance(url, str) else "",
            title=title if isinstance(title, str) else "",
        )

    @staticmethod
    async def _run_script(script: str, arguments: Sequence[str]) -> str:
        if sys.platform != "darwin":
            raise RuntimeError("Chrome AI Overview bridge currently requires macOS")
        process = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            if "JavaScript from Apple Events" in detail or "execute javascript" in detail.lower():
                raise RuntimeError(
                    "Enable Chrome View > Developer > Allow JavaScript from Apple Events for AI Overview search"
                )
            raise RuntimeError(f"Chrome AI Overview bridge failed: {detail or f'exit {process.returncode}'}")
        return stdout.decode("utf-8", errors="replace").strip()


class ChromeAiOverviewSearchClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 45.0,
        poll_interval_seconds: float = 1.0,
        automation: ChromeAutomation | None = None,
        poll_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("Chrome AI Overview timing values must be positive")
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._automation = automation or AppleScriptChromeAutomation()
        self._poll_sleep = poll_sleep
        self._lock = asyncio.Lock()

    async def search(self, proposal_id: str, query: str) -> GroundedResearchFinding:
        normalized_query = " ".join(query.split())[:240]
        if not normalized_query:
            raise ValueError("AI Overview query cannot be empty")
        search_url = "https://www.google.com/search?" + urlencode(
            {"q": normalized_query, "hl": "en", "gl": "us", "ars_aio_bridge": "1"}
        )
        async with self._lock:
            await self._automation.navigate(search_url)
            deadline = monotonic() + self._timeout_seconds
            last_text = ""
            stable_polls = 0
            latest = ChromeAiOverviewSnapshot(text="", sources=())
            while monotonic() < deadline:
                latest = await self._automation.snapshot()
                if latest.text:
                    stable_polls = stable_polls + 1 if latest.text == last_text else 1
                    last_text = latest.text
                    if stable_polls >= CHROME_AIO_STABLE_POLLS:
                        return GroundedResearchFinding(
                            proposal_id=proposal_id,
                            query=normalized_query,
                            summary=latest.text,
                            sources=list(latest.sources),
                        )
                else:
                    stable_polls = 0
                    last_text = ""
                await self._poll_sleep(self._poll_interval_seconds)
        logger.info(
            "Evidence 裁决 Agent Chrome AI Overview 等待超时 query=%s page_title=%s page_url=%s",
            normalized_query,
            latest.title,
            latest.url,
        )
        return GroundedResearchFinding(
            proposal_id=proposal_id,
            query=normalized_query,
            summary="Google Search did not return an AI Overview for this query.",
            sources=[],
        )

    async def close(self) -> None:
        return None


class GeminiGroundedSearchClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._retry_sleep = retry_sleep
        self._client: httpx.AsyncClient | None = None

    async def search(self, proposal_id: str, query: str) -> GroundedResearchFinding:
        normalized_query = " ".join(query.split())[:240]
        if not normalized_query:
            raise ValueError("grounded search query cannot be empty")
        request_payload: dict[str, object] = {
            "contents": [{"parts": [{"text": normalized_query}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 600},
        }
        response = await self._post_with_rate_limit_retry(request_payload)
        response.raise_for_status()
        payload = _as_dict(cast(object, response.json()))
        candidates_value = payload.get("candidates")
        candidates = cast(list[object], candidates_value) if isinstance(candidates_value, list) else []
        candidate = _as_dict(candidates[0]) if candidates else {}
        content = _as_dict(candidate.get("content"))
        parts_value = content.get("parts")
        parts = cast(list[object], parts_value) if isinstance(parts_value, list) else []
        summary_parts: list[str] = []
        for part_value in parts:
            part = _as_dict(part_value)
            text_value = part.get("text")
            if isinstance(text_value, str):
                summary_parts.append(text_value)
        summary = "\n".join(summary_parts).strip()
        metadata = _as_dict(candidate.get("groundingMetadata"))
        chunks_value = metadata.get("groundingChunks")
        chunks = cast(list[object], chunks_value) if isinstance(chunks_value, list) else []
        sources: list[GroundedSource] = []
        seen: set[str] = set()
        for chunk_value in chunks:
            web = _as_dict(_as_dict(chunk_value).get("web"))
            url_value = web.get("uri")
            if not isinstance(url_value, str):
                continue
            if url_value in seen:
                continue
            seen.add(url_value)
            title_value = web.get("title")
            sources.append(GroundedSource(title=title_value if isinstance(title_value, str) else "", url=url_value))
            if len(sources) >= 8:
                break
        return GroundedResearchFinding(
            proposal_id=proposal_id,
            query=normalized_query,
            summary=summary,
            sources=sources,
        )

    async def _post_with_rate_limit_retry(self, request_payload: dict[str, object]) -> httpx.Response:
        for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
            response = await self._get_client().post(
                f"{self._base_url}/models/{quote(self._model, safe='')}:generateContent",
                headers={"x-goog-api-key": self._api_key, "content-type": "application/json"},
                json=request_payload,
            )
            if response.status_code != RATE_LIMIT_STATUS_CODE or attempt == RATE_LIMIT_MAX_ATTEMPTS:
                return response
            logger.info(
                "Evidence 裁决 Agent Web Search 被限流，等待重试 model=%s attempt=%d/%d retry_in_seconds=%g",
                self._model,
                attempt,
                RATE_LIMIT_MAX_ATTEMPTS,
                RATE_LIMIT_RETRY_SECONDS,
            )
            await self._retry_sleep(RATE_LIMIT_RETRY_SECONDS)
        raise AssertionError("rate-limit retry loop must return a response")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_seconds),
                transport=self._transport,
            )
        return self._client


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    mapping = cast(dict[object, object], value)
    return {str(key): item for key, item in mapping.items()}
