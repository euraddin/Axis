"""Tests for Axis web_fetch and web_search tools."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from axis_coding.tools import create_coding_tools
from axis_coding.web_tools import (
    DEFAULT_FETCH_MAX_CONTENT_LENGTH,
    WebToolError,
    _basic_html_strip,
    _format_search_results,
    _is_private_ip,
    _parse_duckduckgo_html,
    _resolve_hostname,
    _SearchResult,
    _strip_tags,
    create_web_fetch_tool,
    create_web_search_tool,
    create_web_tools,
    html_to_text,
    validate_url,
)

# ---------------------------------------------------------------------------
# URL validation & SSRF protection
# ---------------------------------------------------------------------------


class TestValidateURL:
    def test_rejects_missing_scheme(self) -> None:
        with pytest.raises(WebToolError, match="scheme"):
            validate_url("example.com")

    def test_rejects_unsupported_scheme(self) -> None:
        with pytest.raises(WebToolError, match="Unsupported URL scheme"):
            validate_url("ftp://example.com/file")

    def test_rejects_no_hostname(self) -> None:
        with pytest.raises(WebToolError, match="scheme"):
            validate_url("invalid-url")

    def test_accepts_http_url(self) -> None:
        parsed = validate_url("http://example.com/page")
        assert parsed.scheme == "http"
        assert parsed.netloc == "example.com"

    def test_accepts_https_url_with_path(self) -> None:
        parsed = validate_url("https://docs.python.org/3/library/urllib.parse.html")
        assert parsed.scheme == "https"
        assert parsed.netloc == "docs.python.org"


class TestPrivateIPDetection:
    def test_loopback_is_private(self) -> None:
        assert _is_private_ip("127.0.0.1") is True
        assert _is_private_ip("::1") is True

    def test_private_ranges_are_private(self) -> None:
        assert _is_private_ip("10.0.0.1") is True
        assert _is_private_ip("172.16.0.1") is True
        assert _is_private_ip("192.168.1.1") is True
        assert _is_private_ip("169.254.1.1") is True

    def test_public_ips_are_not_private(self) -> None:
        assert _is_private_ip("8.8.8.8") is False
        assert _is_private_ip("1.1.1.1") is False
        assert _is_private_ip("93.184.216.34") is False  # example.com

    def test_zero_ip_is_private(self) -> None:
        assert _is_private_ip("0.0.0.0") is True

    def test_multicast_is_private(self) -> None:
        assert _is_private_ip("224.0.0.1") is True

    def test_link_local_ipv6_is_private(self) -> None:
        assert _is_private_ip("fe80::1") is True

    def test_unique_local_ipv6_is_private(self) -> None:
        assert _is_private_ip("fc00::1") is True
        assert _is_private_ip("fd00::1") is True

    def test_invalid_ip_is_not_private(self) -> None:
        assert _is_private_ip("not-an-ip") is False


class TestResolveHostname:
    def test_resolves_real_hostname(self) -> None:
        # This should resolve to a public IP.
        addr = _resolve_hostname("example.com")
        assert addr is not None
        assert not _is_private_ip(addr)

    def test_localhost_is_resolved_and_blocked(self) -> None:
        addr = _resolve_hostname("localhost")
        assert _is_private_ip(addr) is True

    def test_nonexistent_hostname_raises(self) -> None:
        with pytest.raises(WebToolError, match="Cannot resolve hostname"):
            _resolve_hostname("thishostnamedefinitelydoesnotexist.invalid")


# ---------------------------------------------------------------------------
# HTML-to-text conversion
# ---------------------------------------------------------------------------


class TestHTMLToText:
    def test_strips_basic_tags(self) -> None:
        html = "<p>Hello <b>world</b></p>"
        text = html_to_text(html)
        assert "Hello" in text
        assert "world" in text
        assert "<p>" not in text
        assert "<b>" not in text

    def test_removes_script_tags_and_content(self) -> None:
        html = "<html><body><p>Visible</p><script>alert('xss')</script></body></html>"
        text = html_to_text(html)
        assert "Visible" in text
        assert "alert" not in text
        assert "xss" not in text

    def test_removes_style_tags_and_content(self) -> None:
        html = (
            "<html><head><style>body { color: red; }"
            "</style></head><body><p>Text</p></body></html>"
        )
        text = html_to_text(html)
        assert "Text" in text
        assert "color" not in text

    def test_extracts_link_hrefs(self) -> None:
        html = '<a href="https://example.com">Click here</a>'
        text = html_to_text(html)
        assert "Click here" in text
        assert "https://example.com" in text

    def test_skips_fragment_links(self) -> None:
        html = '<a href="#section">Jump</a>'
        text = html_to_text(html)
        assert "Jump" in text
        assert "#section" not in text

    def test_handles_headings(self) -> None:
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        text = html_to_text(html)
        assert "# Title" in text
        assert "## Subtitle" in text

    def test_handles_list_items(self) -> None:
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        text = html_to_text(html)
        assert "- Item 1" in text
        assert "- Item 2" in text

    def test_handles_images_with_alt_text(self) -> None:
        html = '<img src="photo.jpg" alt="A nice photo">'
        text = html_to_text(html)
        assert "Image" in text
        assert "A nice photo" in text

    def test_handles_empty_input(self) -> None:
        assert html_to_text("") == ""

    def test_handles_complex_page(self) -> None:
        html = """<!DOCTYPE html>
<html>
<head><title>Test</title><style>div{}</style></head>
<body>
  <header>Site header</header>
  <main>
    <h1>Main Title</h1>
    <p>First paragraph with a <a href="https://docs.example.com">link</a>.</p>
    <p>Second paragraph.</p>
    <ul><li>Item A</li><li>Item B</li></ul>
  </main>
  <footer>Copyright 2024</footer>
  <script>console.log('hidden')</script>
</body>
</html>"""
        text = html_to_text(html)
        assert "# Main Title" in text
        assert "First paragraph" in text
        assert "https://docs.example.com" in text
        assert "Second paragraph" in text
        assert "Item A" in text
        assert "Item B" in text
        assert "Site header" in text
        assert "Copyright" in text
        assert "console.log" not in text
        assert "hidden" not in text

    def test_collapses_excessive_whitespace(self) -> None:
        html = "<p>Hello</p>\n\n\n\n<p>World</p>"
        text = html_to_text(html)
        # Should not have 4+ consecutive newlines.
        assert "\n\n\n\n" not in text

    def test_decodes_html_entities(self) -> None:
        html = "<p>&amp; &lt; &gt; &quot;</p>"
        text = html_to_text(html)
        assert "&" in text
        assert "<" in text or "&lt;" not in text


class TestBasicHTMLStrip:
    def test_strips_all_tags(self) -> None:
        html = "<html><body><p>Text</p></body></html>"
        text = _basic_html_strip(html)
        assert "Text" in text
        assert "<" not in text

    def test_removes_script_content(self) -> None:
        html = "<p>Hello</p><script>secret</script>"
        text = _basic_html_strip(html)
        assert "Hello" in text
        assert "secret" not in text


# ---------------------------------------------------------------------------
# web_fetch tool (HTTP mocking)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Simulates an httpx.Response for testing."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._text = text
        self.content = content or text.encode("utf-8")
        self.headers = httpx.Headers(headers or {})

    @property
    def text(self) -> str:
        return self._text


class TestWebFetchTool:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return AsyncMock(spec=httpx.AsyncClient)

    def test_fetches_html_page(self, mock_client: AsyncMock) -> None:
        html = "<html><body><h1>API Docs</h1><p>Documentation content.</p></body></html>"
        mock_client.get.return_value = _FakeResponse(
            status_code=200,
            content=html.encode("utf-8"),
            headers={"content-type": "text/html; charset=utf-8"},
        )

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://httpbin.org/bytes/32"})

            assert result.ok is True
            assert "API Docs" in result.content
            assert "Documentation content" in result.content
            assert "https://httpbin.org/bytes/32" in result.content
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert call_args[0][0] == "https://httpbin.org/bytes/32"

        asyncio.run(run())

    def test_fetches_plain_text(self, mock_client: AsyncMock) -> None:
        text = "Plain text response\nLine 2"
        mock_client.get.return_value = _FakeResponse(
            status_code=200,
            content=text.encode("utf-8"),
            headers={"content-type": "text/plain"},
        )

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://example.com/data.txt"})
            assert result.ok is True
            assert "Plain text response" in result.content

        asyncio.run(run())

    def test_handles_http_error(self, mock_client: AsyncMock) -> None:
        mock_client.get.return_value = _FakeResponse(
            status_code=404,
            text="Not Found",
            headers={"content-type": "text/plain"},
        )

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://example.com/missing"})
            assert result.ok is False
            assert "404" in result.content
            assert "Not Found" in result.content

        asyncio.run(run())

    def test_handles_timeout(self, mock_client: AsyncMock) -> None:
        mock_client.get.side_effect = httpx.TimeoutException("timed out")

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://example.com/slow"})
            assert result.ok is False
            assert "timed out" in result.content.lower()

        asyncio.run(run())

    def test_handles_network_error(self, mock_client: AsyncMock) -> None:
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://example.com/dead"})
            assert result.ok is False
            assert "connection refused" in result.content.lower()

        asyncio.run(run())

    def test_rejects_private_ip_url(self, mock_client: AsyncMock) -> None:
        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "http://127.0.0.1/admin"})
            assert result.ok is False
            assert "private" in result.content.lower()
            assert "internal" in result.content.lower()
            mock_client.get.assert_not_called()

        asyncio.run(run())

    def test_rejects_ftp_scheme(self, mock_client: AsyncMock) -> None:
        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "ftp://files.example.com/readme"})
            assert result.ok is False
            assert "scheme" in result.content.lower()

        asyncio.run(run())

    def test_truncates_large_content(self, mock_client: AsyncMock) -> None:
        large = "A" * (DEFAULT_FETCH_MAX_CONTENT_LENGTH + 500)
        mock_client.get.return_value = _FakeResponse(
            status_code=200,
            content=large.encode("utf-8"),
            headers={"content-type": "text/plain"},
        )

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://example.com/large"})
            assert result.ok is True
            assert len(result.content) < len(large)
            assert "truncated" in result.content.lower()

        asyncio.run(run())

    def test_follows_redirects(self, mock_client: AsyncMock) -> None:
        mock_client.get.return_value = _FakeResponse(
            status_code=200,
            content=b"Final destination",
            headers={"content-type": "text/plain"},
        )

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://example.com/redirect"})
            assert result.ok is True
            call_kwargs = mock_client.get.call_args
            assert call_kwargs is not None
            assert call_kwargs[1]["follow_redirects"] is True

        asyncio.run(run())

    def test_includes_metadata_in_data(self, mock_client: AsyncMock) -> None:
        mock_client.get.return_value = _FakeResponse(
            status_code=200,
            content=b"Hello",
            headers={"content-type": "text/html"},
        )

        async def run() -> None:
            tool = create_web_fetch_tool(client=mock_client)
            result = await tool.execute({"url": "https://example.com"})
            assert result.data is not None
            assert result.data["url"] == "https://example.com"
            assert result.data["status_code"] == 200
            assert "content_type" in result.data

        asyncio.run(run())


# ---------------------------------------------------------------------------
# web_search tool
# ---------------------------------------------------------------------------


_DDG_HTML_PAGE = """
<!DOCTYPE html>
<html>
<body>
  <div class="results">
    <div class="result">
      <a rel="nofollow" class="result__a"
         href="https://docs.python.org/3/">Python 3 Documentation</a>
      <a class="result__snippet">Official Python 3.x documentation</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://realpython.com/">Real Python Tutorials</a>
      <a class="result__snippet">Learn Python with tutorials</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://pypi.org/">PyPI</a>
      <a class="result__snippet">The Python Package Index</a>
    </div>
  </div>
</body>
</html>
"""


class TestDuckDuckGoParsing:
    def test_parses_results_from_html(self) -> None:
        results = _parse_duckduckgo_html(_DDG_HTML_PAGE)
        assert len(results) == 3
        assert results[0].title == "Python 3 Documentation"
        assert results[0].url == "https://docs.python.org/3/"
        assert "Official Python 3" in results[0].snippet

    def test_empty_page_returns_empty(self) -> None:
        results = _parse_duckduckgo_html("<html><body></body></html>")
        assert results == []

    def test_page_without_results(self) -> None:
        html = "<html><body><p>No results found.</p></body></html>"
        results = _parse_duckduckgo_html(html)
        assert results == []


class TestStripTags:
    def test_removes_html_tags(self) -> None:
        assert _strip_tags("<b>bold</b>") == "bold"

    def test_decodes_ampersand(self) -> None:
        assert _strip_tags("A &amp; B") == "A & B"

    def test_handles_nested_tags(self) -> None:
        assert _strip_tags("<div><span>text</span></div>") == "text"


class TestFormatSearchResults:
    def test_formats_results_as_markdown(self) -> None:
        results = [
            _SearchResult(title="Result 1", url="https://a.com", snippet="First result"),
            _SearchResult(title="Result 2", url="https://b.com", snippet="Second result"),
        ]
        output = _format_search_results("test query", results)
        assert "# Search Results: test query" in output
        assert "## 1. Result 1" in output
        assert "**URL:** https://a.com" in output
        assert "**Snippet:** First result" in output
        assert "## 2. Result 2" in output

    def test_respects_max_results(self) -> None:
        results = [
            _SearchResult(title=f"Result {i}", url=f"https://{i}.com", snippet=f"Snippet {i}")
            for i in range(15)
        ]
        output = _format_search_results("query", results, max_results=5)
        assert "## 1." in output
        assert "## 5." in output
        assert "## 6." not in output
        assert "15 results" in output or "Showing 5" in output

    def test_no_results(self) -> None:
        output = _format_search_results("nothing", [])
        assert "No results found" in output


class TestWebSearchTool:
    def test_rejects_empty_query(self) -> None:
        async def run() -> None:
            tool = create_web_search_tool()
            result = await tool.execute({"query": ""})
            assert result.ok is False
            assert "empty" in result.content.lower()

        asyncio.run(run())

    def test_rejects_whitespace_query(self) -> None:
        async def run() -> None:
            tool = create_web_search_tool()
            result = await tool.execute({"query": "   "})
            assert result.ok is False
            assert "empty" in result.content.lower()

        asyncio.run(run())

    def test_handles_search_failure(self) -> None:
        """When the search request itself fails, the tool returns an error result."""
        with patch(
            "axis_coding.web_tools._search_duckduckgo_html",
            side_effect=WebToolError("Search request failed: connection error"),
        ):

            async def run() -> None:
                tool = create_web_search_tool()
                result = await tool.execute({"query": "test query"})
                assert result.ok is False
                assert "connection error" in result.content.lower()

            asyncio.run(run())

    def test_successful_search_via_mock(self) -> None:
        """Simulate a complete search through DuckDuckGo with mocked HTTP."""
        fake_html = _DDG_HTML_PAGE
        with patch(
            "axis_coding.web_tools._search_duckduckgo_html",
            return_value=_parse_duckduckgo_html(fake_html),
        ):

            async def run() -> None:
                tool = create_web_search_tool()
                result = await tool.execute({"query": "python docs"})

                assert result.ok is True
                assert "Python 3 Documentation" in result.content
                assert "docs.python.org" in result.content
                assert "Real Python" in result.content
                assert result.data is not None
                assert result.data["query"] == "python docs"
                assert result.data["total_results"] == 3

            asyncio.run(run())

    def test_respects_max_results(self) -> None:
        """Simulate a search where we only want 2 results."""
        fake_html = _DDG_HTML_PAGE
        with patch(
            "axis_coding.web_tools._search_duckduckgo_html",
            return_value=_parse_duckduckgo_html(fake_html),
        ):

            async def run() -> None:
                tool = create_web_search_tool()
                result = await tool.execute({"query": "python docs", "max_results": 2})

                assert result.ok is True
                assert result.data is not None
                assert result.data["shown_results"] == 2
                assert "## 3." not in result.content

            asyncio.run(run())

    def test_handles_empty_search_results(self) -> None:
        """When DuckDuckGo returns no results."""
        with patch(
            "axis_coding.web_tools._search_duckduckgo_html",
            return_value=[],
        ):

            async def run() -> None:
                tool = create_web_search_tool()
                result = await tool.execute({"query": "xyznonexistent12345"})
                assert result.ok is True
                assert "No results found" in result.content

            asyncio.run(run())


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestWebToolsIntegration:
    def test_create_web_tools_returns_both_tools(self) -> None:
        tools = create_web_tools()
        names = {tool.name for tool in tools}
        assert "web_fetch" in names
        assert "web_search" in names
        assert len(tools) == 2

    def test_create_coding_tools_includes_web_tools_by_default(self) -> None:
        tools = create_coding_tools()
        names = {tool.name for tool in tools}
        assert "read" in names
        assert "write" in names
        assert "edit" in names
        assert "bash" in names
        assert "web_fetch" in names
        assert "web_search" in names

    def test_create_coding_tools_can_exclude_web_tools(self) -> None:
        tools = create_coding_tools(include_web_tools=False)
        names = {tool.name for tool in tools}
        assert "read" in names
        assert "web_fetch" not in names
        assert "web_search" not in names

    def test_web_tools_have_sane_metadata(self) -> None:
        tools = create_web_tools()
        for tool in tools:
            assert tool.name
            assert tool.description
            assert tool.prompt_snippet
            assert tool.input_schema
            assert tool.requires_approval is False
            if tool.name == "web_search":
                assert len(tool.prompt_guidelines) >= 2
            if tool.name == "web_fetch":
                assert len(tool.prompt_guidelines) >= 2

    def test_web_tools_are_executable(self) -> None:
        """All web tools must have a callable executor."""
        tools = create_web_tools()
        for tool in tools:
            assert callable(tool.executor)

    def test_web_fetch_requires_url_argument(self) -> None:
        tools = create_web_tools()
        fetch_tool = next(t for t in tools if t.name == "web_fetch")
        required = fetch_tool.input_schema.get("required", [])
        assert "url" in required

    def test_web_search_requires_query_argument(self) -> None:
        tools = create_web_tools()
        search_tool = next(t for t in tools if t.name == "web_search")
        required = search_tool.input_schema.get("required", [])
        assert "query" in required
