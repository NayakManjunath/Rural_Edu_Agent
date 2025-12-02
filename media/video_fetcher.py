# media/video_fetcher.py
from googleapiclient.discovery import build
import os

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

def fetch_educational_videos(query: str, max_results: int = 3):
    """
    Uses YouTube API to fetch safe-for-kids educational videos.
    """
    if not YOUTUBE_API_KEY:
        return []  # no key? return empty

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    request = youtube.search().list(
        part="snippet",
        q=query + " for kids",
        type="video",
        safeSearch="strict",
        maxResults=max_results
    )
    response = request.execute()

    videos = []
    for item in response.get("items", []):
        videos.append({
            "title": item["snippet"]["title"],
            "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
            "video_url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })

    return videos
