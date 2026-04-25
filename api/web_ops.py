import requests
from bs4 import BeautifulSoup
import re
import logging

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Performs a DuckDuckGo search and returns a list of results.
    Each result: {"title": str, "link": str, "snippet": str}
    """
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        
        # DuckDuckGo HTML results are in result__body divs
        for div in soup.find_all("div", class_="result__body"):
            if len(results) >= max_results:
                break
                
            title_tag = div.find("a", class_="result__a")
            snippet_tag = div.find("a", class_="result__snippet")
            
            if title_tag and title_tag.get("href"):
                title = title_tag.get_text().strip()
                link = title_tag.get("href")
                snippet = snippet_tag.get_text().strip() if snippet_tag else ""
                
                # Filter out DDG internal links if any
                if not link.startswith("/"):
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
        
        return results
    except Exception as e:
        log.error(f"Web search failed: {e}")
        return []

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
