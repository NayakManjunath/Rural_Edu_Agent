# agents/manager.py

import logging
import re

from agents.curator import load_curated
from agents.tutor_agent import TutorAgent
from nlp.summarize import Summarizer
from agents.math_agent import MathAgent
from storage.session_store import load_session, save_session

log = logging.getLogger(__name__)


class AgentManager:
    _summarizer = None

    def __init__(self):
        # Cache summarizer only once
        if AgentManager._summarizer is None:
            AgentManager._summarizer = Summarizer(mode="local")

        self.summarizer = AgentManager._summarizer
        self.tutor = TutorAgent(self.summarizer)
        self.math_agent = MathAgent()

    # ----------------------------------------------------------------------
    # Simple keyword search
    # ----------------------------------------------------------------------
    def find_relevant(self, query: str, top_k=3):
        resources = load_curated()
        scored = []

        for r in resources:
            score = 0
            text = (r.get("title", "") + " " + r.get("transcript", "")).lower()

            if query.lower() in text:
                score += 10

            scored.append((score, r))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    # ----------------------------------------------------------------------
    # Math question detector
    # ----------------------------------------------------------------------
    def _is_math_question(self, question: str) -> bool:
        q = question.strip().lower()

        math_keywords = [
            "+", "-", "*", "/", "=", "half", "quarter", "percent", "%", "of",
            "sin", "cos", "tan", "degrees", "radians", "derivative", "integral",
            "solve", "limit", "matrix", "determinant", "mean", "median",
            "variance", "^"
        ]

        if any(k in q for k in math_keywords):
            return True

        # regex for expressions:  5*2, 1/2 of 20, sin(90)
        if re.search(r"(\d+\s*[/\+\-\*\^]\s*\d+)|(\d+/\d+\s*of)|\b(sin|cos|tan)\b", q):
            return True

        return False

    # ----------------------------------------------------------------------
    # Main routing method
    # ----------------------------------------------------------------------
    def handle_question(self, session_id: str, question: str):
        q = question.strip()

        # Route to MathAgent if it's math
        if self._is_math_question(q):
            result = self.math_agent.solve(q)

            # Save math result to session memory
            sess = load_session(session_id)
            memory = sess.get("memory", {})
            qa = memory.get("qa_history", [])

            qa.append({"q": q, "a": str(result.get("answer"))})
            memory["qa_history"] = qa

            save_session(session_id, sess.get("profile", {}), memory)

            return {
                "answer": result.get("answer"),
                "videos": [],
                "images": []
            }

        # Otherwise → normal tutor agent
        docs = self.find_relevant(question)
        return self.tutor.answer(session_id, question, docs)




# # agents/manager.py

# from agents.curator import load_curated
# from agents.tutor_agent import TutorAgent
# from nlp.summarize import Summarizer
# from agents.math_agent import MathAgent
# from storage.session_store import load_session, save_session
# import logging
# import re

# log = logging.getLogger(__name__)

# class AgentManager:
#     _summarizer = None

#     def __init__(self):
#         # Cache summarizer (loads only once)
#         if AgentManager._summarizer is None:
#             AgentManager._summarizer = Summarizer(mode="local")

#         self.summarizer = AgentManager._summarizer
#         self.tutor = TutorAgent(self.summarizer)
#         self.math_agent = MathAgent()

#     def find_relevant(self, query: str, top_k=3):
#         """
#         Simple keyword-based search across curated resources.
#         """
#         resources = load_curated()
#         scored = []

#         for r in resources:
#             score = 0
#             text = (r.get("title", "") + " " + r.get("transcript", "")).lower()
#             if query.lower() in text:
#                 score += 10

#             scored.append((score, r))

#         scored.sort(key=lambda x: -x[0])
#         return [r for _, r in scored[:top_k]]

#     def _is_math_question(self, question: str) -> bool:
#         q = question.strip().lower()

#         # quick keywords
#         math_keywords = [
#             "+", "-", "*", "/", "=", "half", "quarter", "percent", "%", "of",
#             "sin", "cos", "tan", "degrees", "radians", "derivative", "integral",
#             "solve", "limit", "matrix", "determinant", "mean", "median", "variance", "^"
#         ]

#         if any(k in q for k in math_keywords):
#             return True

#         # numeric expression detection, e.g., "5*2+1", "1/2 of 20", "sin(90)"
#         if re.search(r"(\d+\s*[/\+\-\*\^]\s*\d+)|(\d+/\d+\s*of)|\b(sin|cos|tan)\b", q):
#             return True

#         return False

#         docs = self.find_relevant(question)
#         return self.tutor.answer(session_id, question, docs)


    # def handle_question(self, session_id: str, question: str):
    #     q = question.lower()

    #     # Math detection keywords
    #     math_keywords = ["+", "-", "*", "/", "=", "half", "quarter", "percent", "%", "of"]

    #     # If it's a math question → use MathAgent
    #     if any(k in q for k in math_keywords):
    #         result = self.math_agent.solve(question)

    #         if result is not None:
    #             # Save to memory
    #             sess = load_session(session_id)
    #             memory = sess.get("memory", {})
    #             qa = memory.get("qa_history", [])

    #             qa.append({"q": question, "a": f"The answer is {result}"})
    #             memory["qa_history"] = qa

    #             save_session(session_id, sess.get("profile", {}), memory)

    #             return {
    #                 "answer": f"The answer is {result}",
    #                 "videos": [],
    #                 "images": []
    #             }

    # Otherwise → do normal tutor processing
    



# # agents/manager.py

# from agents.curator import load_curated
# from agents.tutor_agent import TutorAgent
# from nlp.summarize import Summarizer
# from agents.math_agent import MathAgent
# import logging

# log = logging.getLogger(__name__)

# class AgentManager:
#     _summarizer = None

#     def __init__(self):
#         # Cache summarizer
#         if AgentManager._summarizer is None:
#             AgentManager._summarizer = Summarizer(mode="local")

#         self.summarizer = AgentManager._summarizer
#         self.tutor = TutorAgent(self.summarizer)
#         self.math_agent = MathAgent()

#     def find_relevant(self, query: str, top_k=3):
#         """
#         Very simple keyword based retrieval.
#         """
#         resources = load_curated()
#         scored = []

#         for r in resources:
#             score = 0
#             text = (r.get("title", "") + " " + r.get("transcript", "")).lower()

#             if query.lower() in text:
#                 score += 10

#             scored.append((score, r))

#         scored.sort(key=lambda x: -x[0])
#         return [r for _, r in scored[:top_k]]

#     def handle_question(self, session_id: str, question: str):
#         q = question.lower()

#         # If question contains numbers, fractions, %, or math-like words → use MathAgent
#         math_keywords = ["+", "-", "*", "/", "=", "half", "quarter", "percent", "%", "of"]
#         # if any(k in q for k in math_keywords):
#         #     result = self.math_agent.solve(question)
#         #     if result is not None:
#         #         return {
#         #             "answer": f"The answer is {result}",
#         #             "videos": [],
#         #             "images": []
#         #         }
        
        
#     if any(k in q for k in math_keywords):
#         result = self.math_agent.solve(question)
#     if result is not None:

#         # Save to memory
#         sess = load_session(session_id)
#         memory = sess.get("memory", {})
#         qa = memory.get("qa_history", [])

#         qa.append({"q": question, "a": f"The answer is {result}"})
#         memory["qa_history"] = qa

#         save_session(session_id, sess.get("profile", {}), memory)

#         return {
#             "answer": f"The answer is {result}",
#             "videos": [],
#             "images": []
#         }


#         # Otherwise handle with tutor agent
#         docs = self.find_relevant(question)
#         return self.tutor.answer(session_id, question, docs)




# # agents/manager.py
# from agents.curator import load_curated
# from agents.tutor_agent import TutorAgent
# from nlp.summarize import Summarizer
# from agents.math_agent import MathAgent
# import logging

# log = logging.getLogger(__name__)

# class AgentManager:
#     _summarizer = None

#     # def __init__(self):
#     #     # Cache summarizer for speed
#     #     if not AgentManager._summarizer:
#     #         AgentManager._summarizer = Summarizer(mode="local")

#     #     self.summarizer = AgentManager._summarizer
#     #     self.tutor = TutorAgent(self.summarizer)
  
#     def __init__(self):
#         if not AgentManager._summarizer:
#             AgentManager._summarizer = Summarizer(mode="local")

#     self.summarizer = AgentManager._summarizer
#     self.tutor = TutorAgent(self.summarizer)
#     self.math_agent = MathAgent()


#     def find_relevant(self, query: str, top_k=3):
#         """
#         Very simple keyword based retrieval.
#         You can replace this with embeddings later.
#         """
#         resources = load_curated()
#         scored = []

#         for r in resources:
#             score = 0
#             text = (r.get("title", "") + " " + r.get("transcript", "")).lower()
#             if query.lower() in text:
#                 score += 10

#             scored.append((score, r))

#         scored.sort(key=lambda x: -x[0])
#         return [r for _, r in scored[:top_k]]

#     # def handle_question(self, session_id: str, question: str):
#     #     docs = self.find_relevant(question)
#     #     return self.tutor.answer(session_id, question, docs)
#     def handle_question(self, session_id: str, question: str):
#         q = question.lower()

#     # If question contains numbers, fractions, %, or math-like words → use MathAgent
#     math_keywords = ["+", "-", "*", "/", "=", "half", "quarter", "percent", "%", "of"]
#     if any(k in q for k in math_keywords):
#         result = self.math_agent.solve(question)
#         if result is not None:
#             return {
#                 "answer": f"The answer is {result}",
#                 "videos": [],
#                 "images": []
#             }

#     # Otherwise → do regular tutor processing
#     docs = self.find_relevant(question)
#     return self.tutor.answer(session_id, question, docs)


# # agents/manager.py
# """
# Simple orchestrator that delegates tasks to specialist agents.
# This keeps each agent single-purpose and testable.
# """
# from typing import Dict, Any, List
# from agents.curator import load_curated
# from agents.tutor_agent import TutorAgent
# from nlp.summarize import Summarizer
# import logging

# log = logging.getLogger(__name__)

# class AgentManager:
#     def __init__(self):
#         self.summarizer = Summarizer(mode="local")
#         self.tutor = TutorAgent(self.summarizer)
#         # additional agents (translator, curator) can be added similarly

#     def find_relevant(self, query: str, top_k=3) -> List[Dict[str,Any]]:
#         # simple keyword match over curated title/transcript
#         resources = load_curated()
#         scored = []
#         for r in resources:
#             score = 0
#             text = (r.get("title","") + " " + r.get("transcript","")).lower()
#             if query.lower() in text:
#                 score += 10
#             scored.append((score, r))
#         scored.sort(key=lambda x: -x[0])
#         return [r for _, r in scored[:top_k]]

#     def handle_question(self, session_id: str, question: str):
#         docs = self.find_relevant(question)
#         return self.tutor.answer_question(session_id, question, docs)
