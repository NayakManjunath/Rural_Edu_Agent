# media/image_fetcher.py
import wikipedia

def fetch_illustrations(topic: str, max_images: int = 2):
    """
    Fetch images from Wikipedia for a topic (safe for education).
    """
    try:
        page = wikipedia.page(topic)
        images = page.images[:max_images]
        return images
    except:
        return []
