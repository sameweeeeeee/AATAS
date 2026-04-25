import logging
import re
from ddgs import DDGS

log = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    # Beef up vague single-word queries that DDG struggles with
    generic_map = {
        "news":          "latest world news today",
        "todays news":   "latest world news today",
        "today's news":  "latest world news today",
        "latest news":   "latest world news today",
        "headlines":     "top news headlines today",
        "weather":       "weather today",
    }
    query = generic_map.get(query.lower().strip(), query)
    
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
            log.info(f"DDG search for '{query}' returned {len(raw)} results")
            if not raw:
                log.warning(f"DDG returned empty results for query: '{query}'")
            results = []
            for r in raw:
                results.append({
                    "title":   r.get("title", ""),
                    "link":    r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
            return results
    except Exception as e:
        log.error(f"Web search failed for '{query}': {type(e).__name__}: {e}")
        return []


def scrape_page(url: str) -> str:
    """
    Fetches the content of a URL and extracts clean text.
    """
    import requests
    from bs4 import BeautifulSoup

    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    except Exception as e:
        log.error(f"Scraping failed for {url}: {e}")
        return ""