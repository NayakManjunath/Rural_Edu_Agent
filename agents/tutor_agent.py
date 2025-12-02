# agents/tutor_agent.py
from storage.session_store import load_session, save_session
from nlp.summarize import Summarizer
from media.video_fetcher import fetch_educational_videos
from media.image_fetcher import fetch_illustrations

class TutorAgent:
    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer

    def answer(self, session_id: str, question: str, docs):
        context = "\n".join([d.get("transcript", "")[:2000] for d in docs])

        combined = f"Context: {context}\n\nQuestion: {question}"
        summary = self.summarizer.summarize(combined, max_length=150)["summary"]

        videos = fetch_educational_videos(question)
        images = fetch_illustrations(question)

        sess = load_session(session_id)
        memory = sess.get("memory", {})
        qa = memory.get("qa_history", [])

        qa.append({"q": question, "a": summary})
        memory["qa_history"] = qa

        save_session(session_id, sess.get("profile", {}), memory)

        return {
            "answer": summary,
            "videos": videos,
            "images": images
        }



# # agents/tutor_agent.py
# """
# Tutor agent: given question + session context, retrieve resources + ask LLM for answer.
# This is a simple orchestrator pattern—plug in real retriever and LLM later.
# """
# from typing import Dict, Any, List
# from storage.session_store import load_session, save_session
# from nlp.summarize import Summarizer
# import logging

# log = logging.getLogger(__name__)

# class TutorAgent:
#     def __init__(self, summarizer: Summarizer):
#         self.summarizer = summarizer

#     def answer_question(self, session_id: str, question: str, retrieved_docs: List[Dict[str,Any]]):
#         """
#         retrieved_docs: list of resources (title, excerpt), up to N items.
#         For now, we do a simple concatenation and summary (as a proxy for LLM answer).
#         """
#         context_text = "\n\n".join([d.get("transcript","")[:2000] for d in retrieved_docs])
#         long_text = f"Context:\n{context_text}\n\nQuestion: {question}"
#         summary = self.summarizer.summarize(long_text, max_length=150)
#         # store in session memory a short Q/A
#         sess = load_session(session_id)
#         memory = sess.get("memory", {})
#         qa_history = memory.get("qa_history", [])
#         qa_history.append({"question": question, "answer": summary["summary"]})
#         memory["qa_history"] = qa_history
#         save_session(session_id, sess.get("profile", {}), memory)
#         return {"answer": summary["summary"], "method": summary["method"]}

# Usage:
# from nlp.summarize import Summarizer
# from agents.tutor_agent import TutorAgent
# tutor = TutorAgent(Summarizer())
# tutor.answer_question("sess1", "What is photosynthesis?", retrieved_docs=[...])
