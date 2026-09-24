"""Small, public search tools used by the research agent."""

import html
import json
import os
import re
from urllib.parse import quote

import requests

HEADERS = {"User-Agent": "ResearchAgentMiniProject/1.0 (educational project)"}
TIMEOUT = 20


def _abstract_from_index(index: dict | None) -> str:
    """Restore OpenAlex's inverted abstract representation."""
    if not index:
        return ""
    words = [""] * (max(position for positions in index.values() for position in positions) + 1)
    for word, positions in index.items():
        for position in positions:
            words[position] = word
    return " ".join(words)[:1200]


def paper_search_tool(query: str) -> str:
    """Search academic works in OpenAlex or Crossref and return source links.

    Args:
        query: Keywords to search for in academic works.
    """
    if not query.strip():
        return "No paper query provided."
    params = {"search": query, "per_page": 4}
    if os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    try:
        response = requests.get(
            "https://api.openalex.org/works", params=params,
            headers=HEADERS, timeout=TIMEOUT,
        )
        response.raise_for_status()
        results = [
            {"title": entry.get("display_name", ""), "url": entry.get("id"),
             "year": entry.get("publication_year"),
             "abstract": _abstract_from_index(entry.get("abstract_inverted_index")),
             "provider": "OpenAlex"}
            for entry in response.json().get("results", [])
        ]
        if results:
            return json.dumps(results, ensure_ascii=False)
    except requests.RequestException:
        pass

    # Crossref keeps the tool useful if the anonymous OpenAlex budget is exhausted.
    try:
        response = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": query, "rows": 4},
            headers=HEADERS, timeout=TIMEOUT,
        )
        response.raise_for_status()
        results = []
        for entry in response.json().get("message", {}).get("items", []):
            doi = entry.get("DOI", "")
            if not doi:
                continue
            abstract = html.unescape(re.sub(r"<[^>]+>", "", entry.get("abstract", "")))
            results.append({
                "title": (entry.get("title") or [""])[0],
                "url": "https://doi.org/" + quote(doi, safe="/"),
                "year": (entry.get("published", {}).get("date-parts") or [[None]])[0][0],
                "abstract": abstract[:1200],
                "provider": "Crossref",
            })
        return json.dumps(results, ensure_ascii=False)
    except requests.RequestException as error:
        return f"Paper search is unavailable: {type(error).__name__}."


def wikipedia_search_tool(query: str) -> str:
    """Search English Wikipedia for background articles and introductory extracts.

    Args:
        query: Topic to search for on Wikipedia.
    """
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "generator": "search", "gsrsearch": query,
                    "gsrlimit": 3, "prop": "extracts", "exintro": 1,
                    "explaintext": 1, "format": "json"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {}).values()
        results = []
        for page in pages:
            title = page["title"]
            results.append({
                "title": title,
                "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                "extract": page.get("extract", "")[:1800],
            })
        return json.dumps(results, ensure_ascii=False)
    except requests.RequestException as error:
        return f"Wikipedia search is unavailable: {type(error).__name__}."


def tavily_search_tool(query: str) -> str:
    """Search the current web with Tavily and return source snippets and links.

    Args:
        query: Web search query.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily search is unavailable: TAVILY_API_KEY is not set."
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "search_depth": "basic", "max_results": 4},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        results = [
            {"title": item.get("title"), "url": item.get("url"),
             "content": item.get("content", "")[:900]}
            for item in response.json().get("results", [])
        ]
        return json.dumps(results, ensure_ascii=False)
    except requests.RequestException as error:
        return f"Tavily search is unavailable: {type(error).__name__}."
