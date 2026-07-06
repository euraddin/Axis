"""Web fetch and web search tools for Axis coding sessions."""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

import httpx

from axis_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolExecutor,
)
from axis_agent.types import JSONValue

DEFAULT_FETCH_TIMEOUT_SECONDS = 30.0
DEFAULT_FETCH_MAX_BYTES = 1_000_000  # 1 MB
DEFAULT_FETCH_MAX_CONTENT_LENGTH = 50_000  # characters shown to the model
DEFAULT_SEARCH_TIMEOUT_SECONDS = 15.0
DEFAULT_SEARCH_MAX_RESULTS = 10

_USER_AGENT = "Axis/0.1 (coding-agent; +https://github.com/axis)"

# Hostnames that resolve to these networks will be rejected.
_SSRF_BLOCKED_NETWORKS = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("224.0.0.0/4"),  # multicast
    ipaddress.IPv4Network("240.0.0.0/4"),  # reserved
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fe80::/10"),
    ipaddress.IPv6Network("fc00::/7"),
)


class WebToolError(ValueError):
    """A web tool received an invalid or unsafe request."""


# ---------------------------------------------------------------------------
# HTML-to-text conversion
# ---------------------------------------------------------------------------


class _HTMLToText(HTMLParser):
    """Extract readable text from HTML, stripping scripts, styles, and metadata."""

    def __init__(self) -> None:
        super().__init__()
        self._output: list[str] = []
        self._skip_depth = 0
        self._skip_tags: frozenset[str] = frozenset(
            {"script", "style", "noscript", "iframe", "svg", "canvas", "head"}
        )
        self._block_tags: frozenset[str] = frozenset(
            {
                "p",
                "div",
                "article",
                "section",
                "nav",
                "header",
                "footer",
                "main",
                "aside",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "li",
                "br",
                "hr",
                "tr",
                "blockquote",
                "pre",
                "table",
                "ul",
                "ol",
                "dl",
                "figure",
                "figcaption",
                "form",
                "fieldset",
            }
        )
        self._last_char = "\n"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        attr_dict = dict(attrs)

        if tag == "a" and "href" in attr_dict:
            href = attr_dict["href"]
            if href and not href.startswith("#"):
                self._append(f" [{href}] ")

        if tag in self._block_tags:
            self._append("\n")

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self._append(f"\n{'#' * level} ")

        if tag == "img":
            alt = attr_dict.get("alt", "")
            src = attr_dict.get("src", "")
            if alt or src:
                self._append(f"\n[Image: {alt or src}]\n")

        if tag == "br":
            self._append("\n")

        if tag in {"li"}:
            self._append("\n- ")

        if tag in {"hr"}:
            self._append("\n---\n")

        if tag in {"pre", "code"}:
            self._append("\n```\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._block_tags:
            self._append("\n")
        if tag in {"pre", "code"}:
            self._append("\n```\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._append(text + " ")

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        entities: dict[str, str] = {
            "amp": "&",
            "lt": "<",
            "gt": ">",
            "quot": '"',
            "apos": "'",
            "nbsp": " ",
        }
        self._append(entities.get(name, f"&{name};"))

    def _append(self, text: str) -> None:
        if not text:
            return
        self._last_char = text[-1]
        self._output.append(text)

    def get_text(self) -> str:
        text = "".join(self._output)
        # Collapse 3+ newlines to 2, and strip leading/trailing whitespace.
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n +\n", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    """Convert HTML content to readable plain text."""
    parser = _HTMLToText()
    try:
        parser.feed(html)
    except Exception:
        # Fall back to basic tag stripping if the structured parser fails.
        return _basic_html_strip(html)
    return parser.get_text()


def _basic_html_strip(html: str) -> str:
    """Strip HTML tags and return bare text."""
    # Remove script and style sections
    for tag in ("script", "style", "noscript", "head"):
        html = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&apos;", "'").replace("&#x27;", "'")
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------


def _resolve_hostname(hostname: str) -> str:
    """Resolve a hostname and return the first IP address string."""
    try:
        infos = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise WebToolError(f"Cannot resolve hostname: {hostname}") from exc

    for info in infos:
        addr = str(info[4][0]) if info[4][0] else ""
        if addr:
            return addr
    raise WebToolError(f"No addresses found for hostname: {hostname}")


def _is_private_ip(addr: str) -> bool:
    """Return True when *addr* belongs to a blocked internal network."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return any(ip in network for network in _SSRF_BLOCKED_NETWORKS)


def validate_url(url: str) -> urllib.parse.ParseResult:
    """Parse and validate a URL, raising a WebToolError for unsafe targets."""
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise WebToolError("URL must include a scheme (http or https) and a hostname")
    if parsed.scheme not in {"http", "https"}:
        raise WebToolError(f"Unsupported URL scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if hostname is None:
        raise WebToolError("URL has no extractable hostname")

    resolved_ip = _resolve_hostname(hostname)
    if _is_private_ip(resolved_ip):
        raise WebToolError(
            f"URL resolves to a private/internal IP address ({resolved_ip}) and is blocked"
        )

    return parsed


# ---------------------------------------------------------------------------
# web_fetch tool
# ---------------------------------------------------------------------------


def _summarize_as_markdown(parsed_url: urllib.parse.ParseResult, text: str) -> str:
    """Wrap fetched text in a structured markdown summary block."""
    url = urllib.parse.urlunparse(parsed_url)
    lines = [
        f"# Fetched: {url}",
        "",
        f"**Source:** {parsed_url.netloc}",
        f"**Scheme:** {parsed_url.scheme}",
        "",
        "---",
        "",
        text,
    ]
    return "\n".join(lines)


async def _fetch_url(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_FETCH_MAX_BYTES,
    max_content_length: int = DEFAULT_FETCH_MAX_CONTENT_LENGTH,
) -> AgentToolResult:
    """Fetch a URL and return its text content."""
    try:
        parsed_url = validate_url(url)
    except WebToolError as exc:
        return AgentToolResult(
            tool_call_id="",
            name="web_fetch",
            ok=False,
            content=str(exc),
            error=str(exc),
            data={"url": url},
        )

    owned_client: httpx.AsyncClient | None = None

    if client is None:
        client = owned_client = httpx.AsyncClient()

    try:
        try:
            response = await client.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
                timeout=timeout,
            )
        except httpx.TimeoutException:
            return AgentToolResult(
                tool_call_id="",
                name="web_fetch",
                ok=False,
                content=f"Request timed out after {timeout} seconds.",
                error=f"Timeout: {url}",
                data={"url": url, "timed_out": True},
            )
        except httpx.HTTPError as exc:
            return AgentToolResult(
                tool_call_id="",
                name="web_fetch",
                ok=False,
                content=f"Failed to fetch URL: {exc}",
                error=str(exc),
                data={"url": url, "error_type": type(exc).__name__},
            )

        content_type = response.headers.get("content-type", "").lower()
        if response.status_code >= 400:
            return AgentToolResult(
                tool_call_id="",
                name="web_fetch",
                ok=False,
                content=(
                    f"HTTP {response.status_code} from {parsed_url.netloc}.\n\n"
                    f"Response body:\n{response.text[:max_content_length]}"
                ),
                error=f"HTTP {response.status_code}",
                data={"url": url, "status_code": response.status_code},
            )

        # Read response bytes up to the limit.
        raw = response.content[:max_bytes]
        truncated_bytes = len(response.content) > max_bytes

        # Decode.
        charset = "utf-8"
        if "charset=" in content_type:
            try:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            except (ValueError, IndexError):
                charset = "utf-8"

        try:
            decoded = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            decoded = raw.decode("utf-8", errors="replace")

        # Convert HTML to readable text, otherwise use raw text.
        is_html = "text/html" in content_type or (
            not content_type and decoded.strip().startswith("<")
        )
        text = html_to_text(decoded) if is_html else decoded.strip()

        # Truncate text content.
        truncated_text = len(text) > max_content_length
        if truncated_text:
            text = text[:max_content_length] + "\n\n[Content truncated]"

        output = _summarize_as_markdown(parsed_url, text)

        suffix_parts: list[str] = []
        if truncated_bytes:
            suffix_parts.append(
                f"Response body was truncated to {max_bytes / (1024 * 1024):.1f}MB."
            )
        if truncated_text:
            suffix_parts.append(
                f"Text content was truncated to {max_content_length:,} characters."
            )
        if suffix_parts:
            output += "\n\n" + " ".join(suffix_parts)

        return AgentToolResult(
            tool_call_id="",
            name="web_fetch",
            ok=True,
            content=output,
            data={
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "content_length": len(response.content),
                "truncated_bytes": truncated_bytes,
                "truncated_text": truncated_text,
            },
        )
    finally:
        if owned_client is not None:
            await owned_client.aclose()


@dataclass(frozen=True, slots=True)
class WebFetchToolDefinition:
    """Definition for the ``web_fetch`` tool."""

    name: str = "web_fetch"
    description: str = (
        "Fetch a URL and return its text content as markdown. Supports HTTP and HTTPS. "
        "HTML pages are converted to readable text; non-HTML content is returned as-is. "
        f"Content is limited to {DEFAULT_FETCH_MAX_CONTENT_LENGTH:,} characters. "
        "Private/internal IP addresses are blocked for security."
    )
    prompt_snippet: str = "Fetch web page content and convert to readable text"
    prompt_guidelines: tuple[str, ...] = (
        "Use web_fetch to read documentation, API references, and articles instead of "
        "guessing.",
        "When web_fetch returns an error, diagnose it and try a different URL or approach.",
        "Web_fetch content may be truncated; use bash with curl for binary or very large "
        "resources.",
    )
    input_schema: Mapping[str, JSONValue] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch. Must use http or https scheme.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    })
    executor: ToolExecutor | None = field(default=None, compare=False, hash=False)
    requires_approval: bool = False

    def with_executor(self, *, client: httpx.AsyncClient) -> WebFetchToolDefinition:
        """Bind a shared HTTP client to the executor."""

        async def execute(
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
        ) -> AgentToolResult:
            del signal
            url = _str_arg(arguments, "url")
            return await _fetch_url(url, client=client)

        return self.__class__(executor=execute)

    def to_agent_tool(self) -> AgentTool:
        if self.executor is None:
            raise RuntimeError("web_fetch tool has no bound executor")
        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=self.executor,
            requires_approval=self.requires_approval,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_web_fetch_tool(
    *,
    client: httpx.AsyncClient | None = None,
) -> AgentTool:
    """Create the ``web_fetch`` tool with a shared HTTP client."""
    return WebFetchToolDefinition().with_executor(
        client=client or httpx.AsyncClient()
    ).to_agent_tool()


# ---------------------------------------------------------------------------
# web_search tool
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SearchResult:
    """One search result from DuckDuckGo HTML output."""

    title: str
    url: str
    snippet: str


def _search_duckduckgo_html(query: str, *, timeout: float) -> list[_SearchResult]:
    """Search DuckDuckGo via its HTML interface and parse results."""
    import urllib.request

    params = urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(
        f"https://html.duckduckgo.com/html/?{params}",
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise WebToolError(f"Search request failed: {exc}") from exc

    return _parse_duckduckgo_html(html)


_DDG_RESULT_RE = re.compile(
    r'<a[^>]*rel="nofollow"[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


def _parse_duckduckgo_html(html: str) -> list[_SearchResult]:
    """Extract search results from DuckDuckGo's HTML response."""
    results: list[_SearchResult] = []

    # DuckDuckGo HTML displays results as links with class "result__a"
    link_matches = list(_DDG_RESULT_RE.finditer(html))
    snippet_matches = list(_DDG_SNIPPET_RE.finditer(html))

    for i, link_match in enumerate(link_matches):
        url = _clean_ddg_url(link_match.group(1))
        title = _strip_tags(link_match.group(2)).strip()
        if not title or not url:
            continue

        snippet = ""
        if i < len(snippet_matches):
            snippet = _strip_tags(snippet_matches[i].group(1)).strip()

        results.append(_SearchResult(title=title, url=url, snippet=snippet))

    return results


def _clean_ddg_url(raw: str) -> str:
    """Extract the real target URL from DuckDuckGo's redirect wrapper."""
    # DuckDuckGo wraps URLs in a redirect like //duckduckgo.com/l/?uddg=...
    parsed = urllib.parse.urlparse(raw)
    if parsed.path == "/l/" or "uddg=" in raw:
        qs = urllib.parse.parse_qs(parsed.query)
        uddg = qs.get("uddg", [])
        if uddg and uddg[0]:
            return uddg[0]
    # If it's a protocol-relative URL, add https
    if raw.startswith("//"):
        return f"https:{raw}"
    return raw


def _strip_tags(html: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", html).replace("&amp;", "&").replace("&lt;", "<")


def _format_search_results(
    query: str,
    results: list[_SearchResult],
    *,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
) -> str:
    """Format search results as readable markdown."""
    lines = [f"# Search Results: {query}", ""]
    if not results:
        lines.append("No results found.")
        return "\n".join(lines)

    shown = results[:max_results]
    for i, result in enumerate(shown, 1):
        title = result.title or "Untitled"
        lines.append(f"## {i}. {title}")
        lines.append(f"**URL:** {result.url}")
        if result.snippet:
            lines.append(f"**Snippet:** {result.snippet}")
        lines.append("")

    if len(results) > max_results:
        lines.append(
            f"\n*Showing {max_results} of {len(results)} results. "
            f"Refine your query for more specific results.*"
        )

    return "\n".join(lines)


async def _execute_web_search(
    arguments: Mapping[str, JSONValue],
    signal: ToolCancellationToken | None = None,
) -> AgentToolResult:
    del signal
    query = _str_arg(arguments, "query")
    if not query.strip():
        return AgentToolResult(
            tool_call_id="",
            name="web_search",
            ok=False,
            content="Search query cannot be empty.",
            error="Empty query",
        )

    try:
        results = await _run_in_thread(
            _search_duckduckgo_html, query, timeout=DEFAULT_SEARCH_TIMEOUT_SECONDS
        )
    except WebToolError as exc:
        return AgentToolResult(
            tool_call_id="",
            name="web_search",
            ok=False,
            content=f"Web search failed: {exc}",
            error=str(exc),
            data={"query": query},
        )

    max_results = _optional_int_arg(arguments, "max_results")
    if max_results is None or max_results <= 0:
        max_results = DEFAULT_SEARCH_MAX_RESULTS

    output = _format_search_results(query, results, max_results=max_results)
    return AgentToolResult(
        tool_call_id="",
        name="web_search",
        ok=True,
        content=output,
        data={
            "query": query,
            "total_results": len(results),
            "shown_results": min(len(results), max_results),
        },
    )


def create_web_search_tool_definition() -> WebSearchToolDefinition:
    """Create the ``web_search`` tool definition."""
    return WebSearchToolDefinition()


@dataclass(frozen=True, slots=True)
class WebSearchToolDefinition:
    """Definition for the ``web_search`` tool."""

    name: str = "web_search"
    description: str = (
        "Search the web and return a list of results with titles, URLs, and snippets. "
        f"Returns up to {DEFAULT_SEARCH_MAX_RESULTS} results by default. "
        "Use this to find documentation, solutions, and current information before "
        "fetching specific pages with web_fetch."
    )
    prompt_snippet: str = "Search the web for documentation and information"
    prompt_guidelines: tuple[str, ...] = (
        "Search the web before guessing about APIs, configuration, or error messages.",
        "Use web_search to find relevant documentation, then use web_fetch to read "
        "the specific pages.",
        "Craft search queries with specific keywords, version numbers, and error "
        "messages for better results.",
    )
    input_schema: Mapping[str, JSONValue] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific with keywords and versions.",
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum results to return (default {DEFAULT_SEARCH_MAX_RESULTS}).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    })
    requires_approval: bool = False

    def to_agent_tool(self) -> AgentTool:
        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=_execute_web_search,
            requires_approval=self.requires_approval,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


def create_web_search_tool() -> AgentTool:
    """Create the ``web_search`` tool."""
    return WebSearchToolDefinition().to_agent_tool()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_in_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous function in a thread pool to avoid blocking."""
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise WebToolError(f"{name} must be a string")
    return value


def _optional_int_arg(arguments: Mapping[str, JSONValue], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WebToolError(f"{name} must be an integer")
    return value


def create_web_tools(
    *,
    http_client: httpx.AsyncClient | None = None,
) -> list[AgentTool]:
    """Create the web_fetch and web_search tools.

    The caller is responsible for closing *http_client* when the session ends.
    """
    client = http_client or httpx.AsyncClient()
    return [
        create_web_fetch_tool(client=client),
        create_web_search_tool(),
    ]
