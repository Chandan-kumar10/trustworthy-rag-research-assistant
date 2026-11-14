import time
import requests
from config import GROQ_API_KEY, GROQ_MODEL

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def ask_llm(system: str, user: str, max_tokens: int = 800) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing. Put it in the .env file.")

    for attempt in range(6):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        if resp.status_code == 429:
            wait = 5 + attempt * 10   # 5s, 15s, 25s ... wait then retry
            print(f"  (rate limit, waiting {wait}s...)")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Groq error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]

    raise RuntimeError("Groq rate limit: still failing after retries.")