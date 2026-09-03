# ============================================
# TOOLS FOR RESEARCH AGENTS
# ============================================

import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ── Tavily Client ─────────────────────────────────────────────
_api_key = os.getenv("TAVILY_API_KEY", "")
if not _api_key:
    raise EnvironmentError("TAVILY_API_KEY is missing from environment/secrets.")

tavily = TavilyClient(api_key=_api_key)


# ── Skip list for scraping ────────────────────────────────────
SKIP_DOMAINS = [
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "instagram.com", "facebook.com",
    "tiktok.com", "linkedin.com",
    "reddit.com",
]

SKIP_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".zip", ".png", ".jpg", ".jpeg"]


@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily.
    Returns AI summary + page titles + full content + source URLs.
    """
    try:
        results = tavily.search(
            query=query,
            max_results=5,                  # increased from 4
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
        )

        out = []

        # ── Tavily AI summary ─────────────────────────────────
        if results.get("answer"):
            out.append(
                f"## AI SUMMARY\n{results['answer']}\n"
            )

        # ── Individual results ────────────────────────────────
        for i, r in enumerate(results.get("results", []), 1):
            title   = r.get("title", "No Title")
            url     = r.get("url", "")
            score   = r.get("score", 0)

            # Prefer raw_content (full page) over snippet
            content = r.get("raw_content") or r.get("content", "")

            # Trim per-result to avoid token overflow
            content_trimmed = content[:2000].strip()

            out.append(
                f"## Result {i}: {title}\n"
                f"URL: {url}\n"
                f"Relevance Score: {score:.2f}\n"
                f"Content:\n{content_trimmed}\n"
            )

        if not out:
            return "No search results found. Try rephrasing the query."

        return "\n----\n".join(out)

    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return clean text content from a URL.
    Skips PDFs, media files, and non-scrapable platforms.
    """
    url = url.strip()

    # ── Skip non-scrapable domains ────────────────────────────
    if any(domain in url.lower() for domain in SKIP_DOMAINS):
        return f"Skipped (platform not scrapable): {url}"

    # ── Skip non-HTML file types ──────────────────────────────
    if any(url.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return f"Skipped (unsupported file type): {url}"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
        }

        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()

        # ── Check content type ────────────────────────────────
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Skipped {url}: not a readable page (type: {content_type})."

        # ── Parse HTML ────────────────────────────────────────
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove all noise tags
        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "aside", "form", "iframe",
            "noscript", "svg", "button", "meta",
        ]):
            tag.decompose()

        # ── Prefer main content areas ─────────────────────────
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.find(id="main-content")
            or soup.find(class_="post-content")
            or soup.find(class_="article-body")
            or soup.find(class_="content")
            or soup.body
        )

        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # ── Clean blank lines ─────────────────────────────────
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 30]
        clean = "\n".join(lines)

        if not clean:
            return f"No readable content found at: {url}"

        # Return up to 4000 chars to stay within token limits
        return f"Source: {url}\n\n{clean[:4000]}"

    except requests.exceptions.Timeout:
        return f"Could not scrape {url}: Request timed out."

    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "unknown"
        return f"Could not scrape {url}: HTTP error {code}."

    except requests.exceptions.ConnectionError:
        return f"Could not scrape {url}: Connection refused or DNS failed."

    except Exception as e:
        return f"Could not scrape {url}: {str(e)}"