import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def debug_search():
    query = "test"
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": USER_AGENT}
    
    print(f"Requesting {url}...")
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response length: {len(response.text)}")
    
    if "result__body" in response.text:
        print("Found result__body in HTML")
    else:
        print("Did NOT find result__body in HTML")
        # print(response.text[:500])

if __name__ == "__main__":
    debug_search()
