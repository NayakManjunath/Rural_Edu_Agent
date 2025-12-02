# agents/curator.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from duckduckgo_search import DDGS
import trafilatura

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CURATED_JSON = DATA_DIR / "curated.json"


def web_search(query: str, num_results: int = 5):
    """
    DuckDuckGo web search, safe for kids, no API key required.
    """
    with DDGS() as ddg:
        results = ddg.text(query, max_results=num_results)
    return results


def extract_clean_text(url: str) -> str:
    """
    Extract article text from a URL using trafilatura.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text or ""
    except Exception as e:
        log.warning(f"Text extraction failed for {url}: {e}")
    return ""


def curate_from_web(query: str):
    """
    Given a query like 'fractions for kids', fetch articles and save.
    """
    search_results = web_search(query)
    curated_items = []

    for i, result in enumerate(search_results):
        url = result.get("href", "")
        clean_text = extract_clean_text(url)

        curated_items.append({
            "id": f"web_{i}",
            "title": result.get("title", "No Title"),
            "url": url,
            "language": "en",
            "transcript": clean_text[:3000],
            "summary": "",
            "size_bytes": len(clean_text)
        })

    save_curated(curated_items)
    return curated_items


def load_curated():
    if CURATED_JSON.exists():
        return json.loads(CURATED_JSON.read_text(encoding="utf-8"))
    return []


def save_curated(resources: List[Dict[str, Any]]):
    CURATED_JSON.write_text(
        json.dumps(resources, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )



# import logging
# from dataclasses import dataclass
# from typing import List, Dict, Any
# import requests
# from pathlib import Path
# import json

# log = logging.getLogger(__name__)

# DATA_DIR = Path("data")
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# CURATED_JSON = DATA_DIR / "curated.json"

# @dataclass
# class Resource:
#     id: str
#     title: str
#     url: str
#     language: str
#     summary: str = ""
#     transcript: str = ""
#     size_bytes: int = 0

# def load_curated() -> List[Dict[str, Any]]:
#     if CURATED_JSON.exists():
#         # READ WITH UTF-8
#         return json.loads(CURATED_JSON.read_text(encoding="utf-8"))
#     return []

# def save_curated(resources: List[Dict[str, Any]]):
#     # WRITE WITH UTF-8
#     CURATED_JSON.write_text(
#         json.dumps(resources, indent=2, ensure_ascii=False),
#         encoding="utf-8"
#     )

# def fetch_simple(url: str) -> str:
#     try:
#         r = requests.get(url, timeout=10)
#         return r.text
#     except Exception as e:
#         log.warning("Fetch failed for %s: %s", url, e)
#         return ""

# def curate_from_list(list_of_urls: List[Dict[str,str]]):
#     items = []
#     for meta in list_of_urls:
#         item = {
#             "id": meta["id"],
#             "title": meta.get("title", ""),
#             "url": meta.get("url", ""),
#             "language": meta.get("language", "en"),
#             "transcript": meta.get("transcript", ""),
#             "size_bytes": len(meta.get("transcript", "")),
#             "summary": meta.get("summary", "")
#         }
#         items.append(item)

#     save_curated(items)
#     return items


# # agents/curator.py
# """
# Curator: fetches and normalizes learning resources. For competition start,
# use a small CSV / JSON list of URLs and sample transcripts.
# """
# import logging
# from dataclasses import dataclass, asdict
# from typing import List, Dict, Any
# import requests  # for simple fetch; more robust scrapers later
# from pathlib import Path
# import json

# log = logging.getLogger(__name__)
# DATA_DIR = Path("data")
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# CURATED_JSON = DATA_DIR / "curated.json"

# @dataclass
# class Resource:
#     id: str
#     title: str
#     url: str
#     language: str  # language code detected or provided
#     summary: str = ""
#     transcript: str = ""
#     size_bytes: int = 0

# def load_curated() -> List[Dict[str,Any]]:
#     if CURATED_JSON.exists():
#         # return json.loads(CURATED_JSON.read_text())
#         return json.loads(CURATED_JSON.read_text(encoding="utf-8"))
#         return []

# def save_curated(resources: List[Dict[str,Any]]):
#     # CURATED_JSON.write_text(json.dumps(resources, indent=2, ensure_ascii=False))
#     CURATED_JSON.write_text(
#     json.dumps(resources, indent=2, ensure_ascii=False),
#     encoding="utf-8"
#     )


# def fetch_simple(url: str) -> str:
#     # Simple HTTP fetch. Replace with robust extractor for HTML / video transcripts.
#     try:
#         r = requests.get(url, timeout=10)
#         return r.text
#     except Exception as e:
#         log.warning("Fetch failed for %s: %s", url, e)
#         return ""

# def curate_from_list(list_of_urls: List[Dict[str,str]]):
#     """
#     list_of_urls: [{'id':..., 'title':..., 'url':..., 'language':...}, ...]
#     This function fetches content and stores minimal metadata.
#     """
#     items = load_curated()
#     for meta in list_of_urls:
#         html = fetch_simple(meta["url"])
#         item = {
#             "id": meta["id"],
#             "title": meta.get("title", ""),
#             "url": meta["url"],
#             "language": meta.get("language", "en"),
#             "transcript": html[:2000],
#             "size_bytes": len(html)
#         }
#         items.append(item)
#     save_curated(items)
#     return items
