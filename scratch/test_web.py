import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.web_ops import search_web, scrape_page
from api.brain import AATASBrain
from db.database import SessionLocal, get_or_create_user

def test_web():
    print("--- Testing Web Search ---")
    query = "latest SpaceX news"
    results = search_web(query)
    
    if not results:
        print("No results found. (Maybe network issue?)")
    else:
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']} ({r['link']})")
            
        print("\n--- Testing Scraping & Summarization ---")
        url = results[0]['link']
        print(f"Scraping: {url}")
        content = scrape_page(url)
        
        if content:
            print(f"Content length: {len(content)}")
            brain = AATASBrain()
            summary = brain.summarise_email(results[0]['title'], content)
            print(f"\nSummary:\n{summary}")
        else:
            print("Scraping failed.")

if __name__ == "__main__":
    test_web()
