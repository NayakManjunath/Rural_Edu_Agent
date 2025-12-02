# nlp/summarize.py
"""
Summarizer wrapper. Choose between a local transformers summarizer and LLM-based API.
"""
from typing import Dict
import logging
log = logging.getLogger(__name__)

try:
    from transformers import pipeline
    _local_summarizer = pipeline("summarization")
except Exception:
    _local_summarizer = None

class Summarizer:
    def __init__(self, mode="local"):
        self.mode = mode

    def summarize(self, text: str, max_length=120) -> Dict[str,str]:
        """
        Returns a dict { 'summary': str, 'method': 'local'|'llm_api' }
        """
        if self.mode == "local" and _local_summarizer:
            out = _local_summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return {"summary": out[0]["summary_text"], "method": "local"}
        # fallback: naive extractive summary
        sentences = text.split(".")
        top_k = max(1, min(3, len(sentences)))
        summary = ". ".join(sentences[:top_k]).strip()
        return {"summary": summary, "method": "extractive_fallback"}

# Example:
# s = Summarizer(mode='local')
# s.summarize(long_text)
