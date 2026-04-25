import requests
from bs4 import BeautifulSoup
import re
import logging

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def search_web(query: str, max_results: int = 5) -> list[dict]:
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "no_redirect": "1",
        "skip_disambig": "1",
    }
    headers = {"User-Agent": USER_AGENT}
    results = []

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Top answer (Wikipedia-style)
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append({
                "title":   data.get("Heading") or query,
                "link":    data["AbstractURL"],
                "snippet": data["AbstractText"],
            })

        # Related topic results
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(results) >= max_results:
                        break
                    if sub.get("FirstURL") and sub.get("Text"):
                        results.append({
                            "title":   sub["Text"][:80],
                            "link":    sub["FirstURL"],
                            "snippet": sub["Text"],
                        })
            elif topic.get("FirstURL") and topic.get("Text"):
                results.append({
                    "title":   topic["Text"][:80],
                    "link":    topic["FirstURL"],
                    "snippet": topic["Text"],
                })

        # Fallback to HTML scrape if JSON returned nothing
        if not results:
            results = _html_fallback_search(query, max_results, headers)

        return results

    except Exception as e:
        log.error(f"Web search failed: {e}")
        try:
            return _html_fallback_search(query, max_results, headers)
        except Exception as e2:
            log.error(f"HTML fallback also failed: {e2}")
            return []


def _html_fallback_search(query: str, max_results: int, headers: dict) -> list[dict]:
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    found_divs = []
    for tag, cls in [("div", "result__body"), ("div", "links_main"), ("div", "result")]:
        found_divs = soup.find_all(tag, class_=cls)
        if found_divs:
            break

    for div in found_divs:
        if len(results) >= max_results:
            break
        title_tag = div.find("a", class_="result__a") or div.find("h2")
        snippet_tag = div.find("a", class_="result__snippet") or div.find("p")
        if title_tag and title_tag.get("href"):
            link = title_tag["href"]
            if not link.startswith("/"):
                results.append({
                    "title":   title_tag.get_text(strip=True),
                    "link":    link,
                    "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                })

    return results
def scrape_page(url: str) -> str:
    """
    Fetches the content of a URL and extracts clean text.
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=" ")
        
        # Break into lines and remove leading/trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        log.error(f"Scraping failed for {url}: {e}")
        return ""
