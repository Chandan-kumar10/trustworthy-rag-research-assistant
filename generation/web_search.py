import requests
from config import TAVILY_API_KEY


def web_search(query: str, max_results: int = 5):
    """Returns list of {title, url, content, score}. Empty list if no key or failure."""
    if not TAVILY_API_KEY:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
            timeout=30,
        )
        if r.status_code != 200:
            print("Tavily error:", r.status_code, r.text[:200])
            return []
        return r.json().get("results", [])
    except Exception as e:
        print("Tavily failed:", e)
        return []