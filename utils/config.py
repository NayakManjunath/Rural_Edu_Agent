# utils/config.py
# Centralized configuration. Keep secrets out of the repo; use environment variables in production.
import os

# Example env vars: set in your JupyterLab kernel or a .env file (not in repo)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TRANSLATION_API = os.getenv("TRANSLATION_API", "")  # e.g., reverie / google / custom
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "local")  # 'local' or 'gemini'
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/sessions.db")

# bandwidth tiers (kbps)
BANDWIDTH_PROFILES = {
    "low": 50,      # very constrained (text-only)
    "medium": 512,  # can handle short audio / low-res video
    "high": 5000    # full video
}
