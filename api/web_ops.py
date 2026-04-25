import logging
import re
from ddgs import DDGS

log = logging.getLogger(__name__)


def search_web(
    query: str,
    max_results: int = 5,
    db=None,          # optional SQLAlchemy Session — enables cache when provided
) -> list[dict]:
    """
    Search the web via DuckDuckGo and return a list of result dicts.

    When a SQLAlchemy *db* Session is supplied the function:
      1. Checks the persistent SearchCache for a fresh result.
      2. Returns cached results immediately on a hit (no network call).
      3. On a miss, performs the live DDG search and stores the result
         in the cache with an appropriate TTL (short for time-sensitive
         queries like weather/prices, longer for general queries).

    The cache is implemented in db/database.py — see SearchCache model and
    get_cached_search / set_cached_search helpers for TTL configuration.
    """
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

    # ── Cache look-up ─────────────────────────────
    if db is not None:
        try:
            from db.database import get_cached_search, set_cached_search
            cached = get_cached_search(db, query)
            if cached is not None:
                log.info(f"Cache HIT for '{query}' ({len(cached)} results served without network call)")
                return cached
            log.info(f"Cache MISS for '{query}' — performing live search")
        except Exception as exc:
            log.warning(f"Cache look-up failed (continuing with live search): {exc}")

    # ── Live DDG search ───────────────────────────
    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
            log.info(f"DDG search for '{query}' returned {len(raw)} results")
            if not raw:
                log.warning(f"DDG returned empty results for query: '{query}'")
            for r in raw:
                results.append({
                    "title":   r.get("title", ""),
                    "link":    r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        log.error(f"Web search failed for '{query}': {type(e).__name__}: {e}")
        return []

    # ── Store in cache ────────────────────────────
    if db is not None and results:
        try:
            from db.database import set_cached_search
            set_cached_search(db, query, results)
            log.info(f"Cached {len(results)} results for '{query}'")
        except Exception as exc:
            log.warning(f"Failed to write search cache (non-fatal): {exc}")

    return results


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