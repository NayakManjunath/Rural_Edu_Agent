# agents/math_agent.py
"""
Full Hybrid MathAgent (Option B)
- Fractions
- Trigonometry
- ODE solver (dy/dx = f(x))
- Algebra solving
- Calculus (derivative, integral, limit)
- Linear algebra
- Statistics
- SymPy automatic expression solving
- LLM fallback
- API limiter
"""

from typing import Optional, Any, Dict
import os
import re
import math
import json
import traceback
import sympy as sp

# Optional API limiter
try:
    from utils.api_limiter import APILimiter
except Exception:
    import time
    class APILimiter:
        def __init__(self, max_calls_per_min=5, max_calls_per_session=50):
            self.max_calls_per_min = max_calls_per_min
            self.max_calls_per_session = max_calls_per_session
            self.calls = []
            self.session_calls = 0

        def allowed(self):
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]
            if len(self.calls) >= self.max_calls_per_min:
                return False, "rate_limit"
            if self.session_calls >= self.max_calls_per_session:
                return False, "session_limit"
            return True, None

        def record_call(self):
            import time
            self.calls.append(time.time())
            self.session_calls += 1

# optional OpenAI LLM
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    import openai
except Exception:
    openai = None


class MathAgent:
    def __init__(self, use_llm_if_available=True):
        self.use_llm = bool(use_llm_if_available and OPENAI_KEY and openai is not None)
        self.api_limiter = APILimiter(max_calls_per_min=5, max_calls_per_session=30)

    # ---------------- LLM wrapper ----------------
    def _llm_solve(self, prompt: str) -> str:
        if not self.use_llm:
            return ""

        allowed, reason = self.api_limiter.allowed()
        if not allowed:
            if reason == "rate_limit":
                return "LLM rate limit reached — switching to offline solver."
            if reason == "session_limit":
                return "LLM session limit reached — switching offline."
            return ""

        try:
            self.api_limiter.record_call()
            openai.api_key = OPENAI_KEY
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"LLM error: {e}"

    # ---------------- main solver ----------------
    def solve(self, question: str) -> Dict[str, Any]:
        q = question.strip()

        try:
            # 1. Fractions
            out = self._handle_fraction_of(q)
            if out is not None:
                return {"answer": out, "method": "rules"}

            # 2. Trigonometry
            out = self._handle_trig(q)
            if out is not None:
                return {"answer": out, "method": "rules"}

            # 3. ODE (dy/dx = f(x))
            out = self._handle_differential_eq(q)
            if out is not None:
                return {"answer": out, "method": "sympy"}

            # 4. Algebra
            out = self._handle_algebra(q)
            if out is not None:
                return {"answer": out, "method": "sympy"}

            # 5. Calculus
            out = self._handle_calculus(q)
            if out is not None:
                return {"answer": out, "method": "sympy"}

            # 6. Linear algebra
            out = self._handle_linear_algebra(q)
            if out is not None:
                return {"answer": out, "method": "sympy"}

            # 7. Statistics
            out = self._handle_statistics(q)
            if out is not None:
                return {"answer": out, "method": "rules"}

            # 8. SymPy fallback
            out = self._handle_sympy_expression(q)
            if out is not None:
                return {"answer": out, "method": "sympy"}

            # 9. LLM fallback
            if self.use_llm:
                llm_out = self._llm_solve(f"Solve: {q}")
                if llm_out:
                    return {"answer": llm_out, "method": "llm"}

            return {"answer": "Sorry, I couldn't solve that.", "method": "none"}

        except Exception as e:
            return {"answer": f"Error: {e}", "method": "error", "trace": traceback.format_exc()}

    # ---------------- handlers ----------------

    def _handle_fraction_of(self, q: str) -> Optional[float]:
        try:
            lower = q.lower()
            if "half of" in lower:
                nums = re.findall(r"\d+\.?\d*", lower)
                if nums:
                    return float(nums[0]) / 2.0

            if "of" in lower and "/" in lower:
                left, right = lower.split("of", 1)
                left = left.strip()
                if "/" in left:
                    n, d = left.split("/", 1)
                    nums = re.findall(r"\d+\.?\d*", right)
                    if nums:
                        return (float(n) / float(d)) * float(nums[0])
            return None
        except:
            return None

    def _handle_trig(self, q: str) -> Optional[float]:
        try:
            m = re.search(
                r"\b(sin|cos|tan)\b\s*\(?\s*([+-]?\d+(\.\d+)?)\s*\)?\s*(deg|degree|degrees|°|radians|rad)?",
                q, flags=re.I
            )
            if not m:
                return None

            func = m.group(1).lower()
            angle = float(m.group(2))
            unit = (m.group(4) or "").lower()

            if "rad" in unit:
                rad = angle
            else:
                rad = math.radians(angle)

            if func == "sin":
                return round(math.sin(rad), 10)
            if func == "cos":
                return round(math.cos(rad), 10)
            if func == "tan":
                try:
                    return round(math.tan(rad), 10)
                except:
                    return "undefined"
        except:
            return None

    def _handle_differential_eq(self, q: str) -> Optional[str]:
        """
    Solve ODE of the form dy/dx = f(x)
    """
    try:
        txt = q.lower().replace("^", "**")

        # match ANY non-alphanumeric slash between dy and dx
        m = re.search(r"dy\s*[^0-9a-zA-Z]\s*dx\s*=\s*(.+)", txt)
        if not m:
            return None

        rhs = m.group(1).strip()

        # --- FIX: Insert multiplication where missing ---
        # Convert 2x -> 2*x, 3x^2 -> 3*x**2 etc
        rhs = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", rhs)
        rhs = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", rhs)

        x = sp.symbols("x")
        expr = sp.sympify(rhs)     # this will now succeed
        integral = sp.integrate(expr, x)

        return f"y = {sp.simplify(integral)} + C"
    except:
        return None


    # def _handle_differential_eq(self, q: str) -> Optional[str]:
    #     """
    #     Solve dy/dx = f(x)
    #     """
    #     try:
    #         txt = q.lower().replace("^", "**")

    #         # match ANY non-alphanumeric symbol between dy and dx
    #         m = re.search(r"dy\s*[^0-9a-zA-Z]\s*dx\s*=\s*(.+)", txt)
    #         if not m:
    #             return None

    #         rhs = m.group(1).strip()

    #         x = sp.symbols("x")
    #         expr = sp.sympify(rhs)
    #         integral = sp.integrate(expr, x)

    #         return f"y = {integral} + C"
    #     except:
    #         return None

    def _handle_algebra(self, q: str) -> Optional[Any]:
        try:
            lower = q.lower()
            if "solve" in lower or "=" in lower:
                expr = lower.replace("solve", "").replace("^", "**")
                if "=" in expr:
                    left, right = expr.split("=", 1)
                    x = sp.symbols("x")
                    eq = sp.Eq(sp.sympify(left), sp.sympify(right))
                    sol = sp.solve(eq, dict=True)
                    return sol
                else:
                    x = sp.symbols("x")
                    sol = sp.solve(sp.sympify(expr), x)
                    return sol
            return None
        except:
            return None

    def _handle_calculus(self, q: str) -> Optional[Any]:
        try:
            txt = q.lower().replace("^", "**")

            m = re.search(r"(derivative|differentiate|d/dx)\s*(of\s*)?(.+)", txt)
            if m:
                x = sp.symbols("x")
                return str(sp.diff(sp.sympify(m.group(3)), x))

            m = re.search(r"(integral|integrate)\s*(of\s*)?(.+)", txt)
            if m:
                x = sp.symbols("x")
                return str(sp.integrate(sp.sympify(m.group(3)), x))

            m = re.search(r"limit of (.+) as x->(.+)", txt)
            if m:
                expr = sp.sympify(m.group(1))
                val = sp.sympify(m.group(2))
                x = sp.symbols("x")
                return str(sp.limit(expr, x, val))

            return None
        except:
            return None

    def _handle_linear_algebra(self, q: str) -> Optional[Any]:
        try:
            m = re.search(r"(\[\[.*\]\])", q)
            if not m:
                return None

            mat = sp.Matrix(json.loads(m.group(1)))
            lower = q.lower()
            if "det" in lower:
                return float(mat.det())
            if "inverse" in lower:
                return mat.inv().tolist()
            if "eigen" in lower:
                return mat.eigenvals()
            return None
        except:
            return None

    def _handle_statistics(self, q: str) -> Optional[Any]:
        try:
            lower = q.lower()
            if any(t in lower for t in ["mean", "median", "mode", "variance", "std", "standard deviation"]):
                nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", q)]
                if not nums:
                    return None
                import statistics as st
                if "mean" in lower:
                    return st.mean(nums)
                if "median" in lower:
                    return st.median(nums)
                if "mode" in lower:
                    return st.mode(nums)
                if "variance" in lower:
                    return st.pvariance(nums)
                if "std" in lower:
                    return st.pstdev(nums)
            return None
        except:
            return None

    def _handle_sympy_expression(self, q: str) -> Optional[Any]:
        try:
            cleaned = q.replace("^", "**")
            expr = sp.sympify(cleaned)
            if expr.is_Number:
                return float(expr)
            return str(sp.simplify(expr))
        except:
            return None



# # agents/math_agent.py

# """
# Unified MathAgent with:
# - Arithmetic
# - Fractions
# - Trigonometry
# - ODE solver (dy/dx = ...)
# - Algebra
# - Calculus
# - Linear Algebra
# - Statistics
# - LLM fallback
# """

# from typing import Optional, Dict, Any
# import sympy as sp
# import math
# import re
# import json
# import traceback
# import os

# # Optional LLM (OpenAI)
# try:
#     from openai import OpenAI
#     OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
#     client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
# except:
#     client = None


# # ==========================================================
# # MATH AGENT
# # ==========================================================

# class MathAgent:
#     def __init__(self, use_llm_if_available: bool = True):
#         self.use_llm = use_llm_if_available and client is not None

#     # ------------------------------------------------------
#     # LLM fallback
#     # ------------------------------------------------------
#     def _llm_solve(self, prompt: str) -> str:
#         if not self.use_llm:
#             return ""

#         try:
#             resp = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=300,
#                 temperature=0.0
#             )
#             return resp.choices[0].message["content"]
#         except Exception as e:
#             return f"LLM error: {e}"


#     # ==========================================================
#     # PUBLIC ENTRY
#     # ==========================================================
#     def solve(self, question: str) -> Dict[str, Any]:
#         q = question.strip()

#         try:
#             # FRACTIONS
#             frac = self._handle_fraction_of(q)
#             if frac is not None:
#                 return {"answer": frac, "method": "rules"}

#             # TRIGONOMETRY
#             trig = self._handle_trig(q)
#             if trig is not None:
#                 return {"answer": trig, "method": "rules"}

#             # ODE dy/dx = ...
#             ode = self._handle_differential_eq(q)
#             if ode is not None:
#                 return {"answer": ode, "method": "sympy"}

#             # BASIC SYMPY EXPRESSION
#             expr = self._handle_sympy_expression(q)
#             if expr is not None:
#                 return {"answer": expr, "method": "sympy"}

#             # ALGEBRAIC EQUATIONS
#             algebra = self._handle_algebra(q)
#             if algebra is not None:
#                 return {"answer": algebra, "method": "sympy"}

#             # CALCULUS (derivative, integral, limit)
#             calc = self._handle_calculus(q)
#             if calc is not None:
#                 return {"answer": calc, "method": "sympy"}

#             # LINEAR ALGEBRA
#             la = self._handle_linear_algebra(q)
#             if la is not None:
#                 return {"answer": la, "method": "sympy"}

#             # STATISTICS
#             stats = self._handle_statistics(q)
#             if stats is not None:
#                 return {"answer": stats, "method": "rules"}

#             # LLM Fallback
#             if self.use_llm:
#                 out = self._llm_solve(f"Solve: {q}")
#                 if out:
#                     return {"answer": out, "method": "llm"}

#             return {"answer": "Sorry, I couldn't solve that.", "method": "none"}

#         except Exception as e:
#             return {
#                 "answer": "Error",
#                 "method": "error",
#                 "trace": traceback.format_exc()
#             }



#     # ==========================================================
#     # HANDLERS
#     # ==========================================================

#     # -------- FRACTIONS --------
#     def _handle_fraction_of(self, q: str) -> Optional[float]:
#         try:
#             text = q.lower()

#             # "half of 20"
#             if "half of" in text:
#                 nums = re.findall(r"[-+]?\d*\.?\d+", text)
#                 if nums:
#                     return float(nums[0]) / 2

#             # "1/2 of 20"
#             if "of" in text and "/" in text:
#                 left, right = text.split("of", 1)
#                 frac = left.strip()
#                 num = float(re.findall(r"[-+]?\d*\.?\d+", right)[0])
#                 n, d = frac.split("/")
#                 return (float(n) / float(d)) * num

#             return None

#         except:
#             return None


#     # -------- TRIG --------
#     def _handle_trig(self, q: str) -> Optional[float]:
#         try:
#             m = re.search(
#                 r"(sin|cos|tan)\s*\(?\s*([+-]?\d+(\.\d+)?)\s*(degrees|deg|°|radians|rad)?",
#                 q.lower()
#             )
#             if not m:
#                 return None

#             func = m.group(1)
#             angle = float(m.group(2))
#             unit = m.group(4)

#             if unit and "rad" in unit:
#                 rad = angle
#             else:
#                 rad = math.radians(angle)

#             if func == "sin":
#                 return round(math.sin(rad), 10)
#             if func == "cos":
#                 return round(math.cos(rad), 10)
#             if func == "tan":
#                 try:
#                     return round(math.tan(rad), 10)
#                 except:
#                     return "undefined"

#             return None

#         except:
#             return None


#     # -------- ODE dy/dx = ... --------
#     # def _handle_differential_eq(self, q: str) -> Optional[str]:
#     #     try:
#     #         txt = q.lower().replace("^", "**")

#     #         # support "/", "⁄", "∕"
#     #         m = re.search(r"dy\s*[/⁄∕]\s*dx\s*=\s*(.+)", txt)
#     #         if not m:
#     #             return None

#     #         rhs = m.group(1).strip()

#     #         x = sp.symbols("x")
#     #         expr = sp.sympify(rhs)
#     #         antider = sp.integrate(expr, x)

#     #         return f"y = {antider} + C"

#     #     except:
#     #         return None

#     def _handle_differential_eq(self, q: str) -> Optional[str]:
#         """
#         Solve simple ODEs of the form 'dy/dx = <expression>'.
#         Matches: dy/dx, dy⁄dx, dy∕dx, dy|dx, etc.
#         """
#         try:
#             txt = q.lower().replace("^", "**").strip()

#             # match ANY non-alphanumeric separator between dy and dx
#             m = re.search(r"dy\s*[^a-z0-9]\s*dx\s*=\s*(.+)", txt)
#             if not m:
#                 return None

#             rhs = m.group(1).strip()

#             x = sp.symbols("x")
#             expr = sp.sympify(rhs)
#             integral = sp.integrate(expr, x)

#             return f"y = {integral} + C"

#         except Exception:
#             return None

#     # -------- BASIC SYMPY --------
#     def _handle_sympy_expression(self, q: str) -> Optional[Any]:
#         try:
#             cleaned = q.replace("^", "**")
#             expr = sp.sympify(cleaned)

#             if expr.is_Number:
#                 return float(expr)
#             return str(sp.simplify(expr))

#         except:
#             return None


#     # -------- ALGEBRA --------
#     def _handle_algebra(self, q: str) -> Optional[Any]:
#         try:
#             if "solve" in q.lower() or "=" in q:
#                 text = q.replace("^", "**").lower().replace("solve", "")

#                 if "=" in text:
#                     left, right = text.split("=", 1)
#                     x = sp.symbols("x")
#                     eq = sp.Eq(sp.sympify(left), sp.sympify(right))
#                     return str(sp.solve(eq, dict=True))

#                 else:
#                     x = sp.symbols("x")
#                     return str(sp.solve(sp.sympify(text), x))

#         except:
#             return None


#     # -------- CALCULUS --------
#     def _handle_calculus(self, q: str) -> Optional[Any]:
#         try:
#             txt = q.lower().replace("^", "**")

#             # derivative
#             m = re.search(r"(derivative|differentiate|d/dx)\s*of\s*(.+)", txt)
#             if m:
#                 expr = sp.sympify(m.group(2))
#                 x = sp.symbols("x")
#                 return str(sp.diff(expr, x))

#             # integral
#             m = re.search(r"(integral|integrate)\s*of\s*(.+)", txt)
#             if m:
#                 expr = sp.sympify(m.group(2))
#                 x = sp.symbols("x")
#                 return str(sp.integrate(expr, x))

#             # limit
#             m = re.search(r"limit\s+of\s+(.+)\s+as\s+x\s*->\s*([0-9\+\-inf]+)", txt)
#             if m:
#                 expr = sp.sympify(m.group(1))
#                 x = sp.symbols("x")
#                 point = sp.sympify(m.group(2))
#                 return str(sp.limit(expr, x, point))

#             return None

#         except:
#             return None


#     # -------- LINEAR ALGEBRA --------
#     def _handle_linear_algebra(self, q: str) -> Optional[Any]:
#         try:
#             lower = q.lower()

#             m = re.search(r"(\[\[.*\]\])", q)
#             if not m:
#                 return None

#             mat = sp.Matrix(json.loads(m.group(1)))

#             if "det" in lower or "determinant" in lower:
#                 return float(mat.det())

#             if "inverse" in lower:
#                 return str(mat.inv().tolist())

#             return None

#         except:
#             return None


#     # -------- STATISTICS --------
#     def _handle_statistics(self, q: str) -> Optional[Any]:
#         try:
#             txt = q.lower()
#             nums = re.findall(r"[-+]?\d*\.?\d+", txt)

#             if not nums:
#                 return None

#             nums = [float(n) for n in nums]

#             import statistics as s

#             if "mean" in txt:
#                 return s.mean(nums)
#             if "median" in txt:
#                 return s.median(nums)
#             if "mode" in txt:
#                 try:
#                     return s.mode(nums)
#                 except:
#                     return None
#             if "variance" in txt:
#                 return s.pvariance(nums)
#             if "std" in txt or "standard deviation" in txt:
#                 return s.pstdev(nums)

#             return None

#         except:
#             return None



# # agents/math_agent.py

# """
# Hybrid MathAgent:
# - Rule-based math
# - SymPy math
# - ODE solving (dy/dx = ...)
# - Calculus, algebra, trig, statistics, linear algebra
# - Optional LLM fallback with API limiter
# """

# from typing import Optional, Dict, Any
# import sympy as sp
# import math
# import re
# import json
# import traceback
# import os

# # Optional LLM (only used if API key exists)
# try:
#     from openai import OpenAI
#     OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
#     client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
# except:
#     client = None


# # ==========================================================
# # MathAgent Class
# # ==========================================================

# class MathAgent:
#     def __init__(self, use_llm_if_available: bool = True):
#         self.use_llm = use_llm_if_available and (client is not None)

#     # ------------------------------------------------------
#     # LLM fallback
#     # ------------------------------------------------------
#     def _llm_solve(self, prompt: str) -> str:
#         if not self.use_llm:
#             return ""

#         try:
#             resp = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=400,
#                 temperature=0.0
#             )
#             return resp.choices[0].message["content"]
#         except Exception as e:
#             return f"LLM error: {e}"

#     # ==========================================================
#     # PUBLIC ENTRY: solve()
#     # ==========================================================
#     def solve(self, question: str) -> Dict[str, Any]:
#         q = question.strip()

#         try:
#             # FRACTIONS
#             frac = self._handle_fraction_of(q)
#             if frac is not None:
#                 return {"answer": frac, "method": "rules"}

#             # TRIG
#             trig = self._handle_trig(q)
#             if trig is not None:
#                 return {"answer": trig, "method": "rules"}

#             # ODE: dy/dx = ...
#             ode = self._handle_differential_eq(q)
#             if ode is not None:
#                 return {"answer": ode, "method": "sympy"}

#             # BASIC SYMPY
#             expr = self._handle_sympy_expression(q)
#             if expr is not None:
#                 return {"answer": expr, "method": "sympy"}

#             # ALGEBRA
#             algebra = self._handle_algebra(q)
#             if algebra is not None:
#                 return {"answer": algebra, "method": "sympy"}

#             # CALCULUS
#             calculus = self._handle_calculus(q)
#             if calculus is not None:
#                 return {"answer": calculus, "method": "sympy"}

#             # LINEAR ALGEBRA
#             la = self._handle_linear_algebra(q)
#             if la is not None:
#                 return {"answer": la, "method": "sympy"}

#             # STATISTICS
#             stats = self._handle_statistics(q)
#             if stats is not None:
#                 return {"answer": stats, "method": "rules"}

#             # LLM fallback
#             if self.use_llm:
#                 out = self._llm_solve(f"Solve this: {q}")
#                 if out:
#                     return {"answer": out, "method": "llm"}

#             return {"answer": "Sorry, I couldn't solve that.", "method": "none"}

#         except Exception as e:
#             return {"answer": "Error", "method": "error", "trace": traceback.format_exc()}


#     # ==========================================================
#     # HANDLERS
#     # ==========================================================

#     # -------- FRACTIONS --------
#     def _handle_fraction_of(self, q: str) -> Optional[float]:
#         try:
#             lower = q.lower()

#             # "half of 20"
#             if "half of" in lower:
#                 nums = re.findall(r"[-+]?\d*\.?\d+", lower)
#                 if nums:
#                     return float(nums[0]) / 2

#             # "1/2 of 20"
#             if "of" in lower and "/" in lower:
#                 left, right = lower.split("of", 1)
#                 frac = left.strip()
#                 num = float(re.findall(r"[-+]?\d*\.?\d+", right)[0])
#                 n, d = frac.split("/")
#                 return (float(n) / float(d)) * num

#             return None
#         except:
#             return None

#     # -------- TRIG --------
#     def _handle_trig(self, q: str) -> Optional[float]:
#         try:
#             m = re.search(
#                 r"(sin|cos|tan)\s*\(?\s*([+-]?\d+(\.\d+)?)\s*(degrees|degree|deg|°|rad|radians)?\s*\)?",
#                 q.lower()
#             )
#             if not m:
#                 return None

#             func = m.group(1)
#             angle = float(m.group(2))
#             unit = m.group(4)

#             # degrees vs radians
#             if unit and ("rad" in unit):
#                 radians = angle
#             else:
#                 radians = math.radians(angle)

#             if func == "sin":
#                 return round(math.sin(radians), 10)
#             if func == "cos":
#                 return round(math.cos(radians), 10)
#             if func == "tan":
#                 try:
#                     return round(math.tan(radians), 10)
#                 except:
#                     return "undefined"

#             return None

#         except:
#             return None

#     # -------- ODE dy/dx = ... --------
#     def _handle_differential_eq(self, q: str) -> Optional[str]:
#         try:
#             txt = q.lower().replace("^", "**")

#             m = re.search(r"dy\s*/\s*dx\s*=\s*(.+)", txt)
#             if not m:
#                 return None

#             rhs = m.group(1).strip()
#             x = sp.symbols("x")
#             expr = sp.sympify(rhs)
#             integral = sp.integrate(expr, x)

#             return f"y = {str(integral)} + C"

#         except:
#             return None


#     # -------- BASIC SYMPY --------
#     def _handle_sympy_expression(self, q: str) -> Optional[Any]:
#         try:
#             cleaned = q.replace("^", "**")
#             expr = sp.sympify(cleaned)
#             if expr.is_Number:
#                 return float(expr)
#             return str(sp.simplify(expr))
#         except:
#             return None

#     # -------- ALGEBRA --------
#     def _handle_algebra(self, q: str) -> Optional[Any]:
#         try:
#             if "solve" in q.lower() or "=" in q:
#                 text = q.replace("^", "**").lower().replace("solve", "")

#                 if "=" in text:
#                     left, right = text.split("=", 1)
#                     x = sp.symbols("x")
#                     eq = sp.Eq(sp.sympify(left), sp.sympify(right))
#                     solutions = sp.solve(eq, dict=True)
#                     return str(solutions)

#                 else:
#                     x = sp.symbols("x")
#                     sol = sp.solve(sp.sympify(text), x)
#                     return str(sol)

#         except:
#             return None

#     # -------- CALCULUS --------
#     def _handle_calculus(self, q: str) -> Optional[Any]:
#         try:
#             txt = q.lower().replace("^", "**")

#             # derivative
#             m = re.search(r"(derivative|differentiate|d/dx)\s*of\s*(.+)", txt)
#             if m:
#                 expr = sp.sympify(m.group(2))
#                 x = sp.symbols("x")
#                 return str(sp.diff(expr, x))

#             # integral
#             m = re.search(r"(integral|integrate)\s*of\s*(.+)", txt)
#             if m:
#                 expr = sp.sympify(m.group(2))
#                 x = sp.symbols("x")
#                 return str(sp.integrate(expr, x))

#             # limit
#             m = re.search(r"limit\s*of\s*(.+)\s*as\s*x\s*->\s*([0-9\+\-inf]+)", txt)
#             if m:
#                 expr = sp.sympify(m.group(1))
#                 point = sp.sympify(m.group(2))
#                 x = sp.symbols("x")
#                 return str(sp.limit(expr, x, point))

#             return None

#         except:
#             return None

#     # -------- LINEAR ALGEBRA --------
#     def _handle_linear_algebra(self, q: str) -> Optional[Any]:
#         try:
#             lower = q.lower()

#             if "det" in lower or "determinant" in lower:
#                 m = re.search(r"(\[\[.*\]\])", q)
#                 if m:
#                     mat = sp.Matrix(json.loads(m.group(1)))
#                     return float(mat.det())

#             if "inverse" in lower:
#                 m = re.search(r"(\[\[.*\]\])", q)
#                 if m:
#                     mat = sp.Matrix(json.loads(m.group(1)))
#                     inv = mat.inv()
#                     return str(inv.tolist())

#             return None

#         except:
#             return None

#     # -------- STATISTICS --------
#     # def _handle_statistics(self, q: str) -> Optional[Any]:
#     #     try:
#     #         lower = q.lower()
#     #         nums = re.findall(r"[-+]?\d*\.?\d+", lower)
#     #         nums = [float(x) for x in nums]

#     #         if not nums:
#     #             return None

#     #         import statistics as stats

#     #         if "mean" in lower:
#     #             return stats.mean(nums)
#     #         if "median" in lower:
#     #             return stats.median(nums)
#     #         if "mode" in lower:
#     #             try: return stats.mode(nums)
#     #             except: return None
#     #         if "variance" in lower:
#     #             return stats.pvariance(nums)
#     #         if "std" in lower or "standard deviation" in lower:
#     #             return stats.pstdev(nums)

#     #         return None

#     #     except:
#     #         return None
#         def _handle_differential_eq(self, q: str) -> Optional[str]:
#             try:
#                 txt = q.lower().replace("^", "**").strip()

#             # match: dy/dx = ...
#             # supports / , ⁄ , ∕ 
#                 m = re.search(r"dy\s*[/⁄∕]\s*dx\s*=\s*(.+)", txt)
#                 if not m:
#                     return None

#                 rhs = m.group(1).strip()

#                 x = sp.symbols("x")
#                 expr = sp.sympify(rhs)
#                 integral = sp.integrate(expr, x)

#                 return f"y = {integral} + C"

#             except:
#                 return None




# # agents/math_agent.py

# import math
# import re
# import sympy as sp
# import traceback
# import json
# import os
# from typing import Optional, Any, Dict

# from utils.api_limiter import APILimiter

# # Optional LLM
# try:
#     import openai
# except:
#     openai = None

# OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


# class MathAgent:
#     def __init__(self, use_llm_if_available=True):
#         self.use_llm = use_llm_if_available and bool(OPENAI_KEY) and openai is not None

#         self.api_limiter = APILimiter(
#             max_calls_per_min=5,
#             max_calls_per_session=20
#         )

#     # ----------------------------------------------------------------------
#     # LLM fallback
#     # ----------------------------------------------------------------------
#     def _llm_solve(self, prompt: str) -> str:
#         if not self.use_llm:
#             return ""

#         allowed, reason = self.api_limiter.allowed()
#         if not allowed:
#             return "LLM limit reached – using offline solver."

#         try:
#             self.api_limiter.record_call()
#             openai.api_key = OPENAI_KEY

#             resp = openai.ChatCompletion.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=400
#             )

#             return resp["choices"][0]["message"]["content"].strip()

#         except Exception:
#             return "LLM error – using offline solver."

#     # ----------------------------------------------------------------------
#     # MAIN SOLVER
#     # ----------------------------------------------------------------------
#     def solve(self, question: str) -> Dict[str, Any]:
#         q = question.strip()
#         lower = q.lower()

#         try:
#             # FRACTION HANDLER
#             frac = self._handle_fraction_of(q)
#             if frac is not None:
#                 return {"answer": frac, "method": "rules"}

#             # TRIGONOMETRY
#             trig = self._handle_trig(q)
#             if trig is not None:
#                 return {"answer": trig, "method": "rules"}

#             # BASIC EXPRESSIONS
#             basic = self._handle_sympy_expression(q)
#             if basic is not None:
#                 return {"answer": basic, "method": "sympy"}

#             # ALGEBRA
#             algebra = self._handle_algebra(q)
#             if algebra is not None:
#                 return {"answer": algebra, "method": "sympy"}

#             # CALCULUS
#             calc = self._handle_calculus(q)
#             if calc is not None:
#                 return {"answer": calc, "method": "sympy"}

#             # LINEAR ALGEBRA
#             lin = self._handle_linear_algebra(q)
#             if lin is not None:
#                 return {"answer": lin, "method": "sympy"}

#             # STATISTICS
#             stats = self._handle_statistics(q)
#             if stats is not None:
#                 return {"answer": stats, "method": "rules"}

#             # ODE handler: dy/dx = ...
#             ode_res = self._handle_differential_eq(q)
#             if ode_res is not None:
#                 return {"answer": ode_res, "method": "sympy"}


#             # LLM LAST
#             if self.use_llm:
#                 out = self._llm_solve(f"Solve and explain: {q}")
#                 return {"answer": out, "method": "llm"}

#             return {"answer": "Sorry, I couldn't solve that.", "method": "none"}

#         except Exception as e:
#             return {"answer": str(e), "method": "error", "trace": traceback.format_exc()}

#     # ----------------------------------------------------------------------
#     # SPECIFIC HANDLERS
#     # ----------------------------------------------------------------------

#     def _handle_fraction_of(self, q: str) -> Optional[float]:
#         try:
#             lower = q.lower()
#             if "half of" in lower:
#                 nums = re.findall(r"[+-]?\d*\.?\d+", lower)
#                 return float(nums[0]) / 2 if nums else None

#             # e.g. "1/2 of 20"
#             if "of" in lower and "/" in lower:
#                 left, right = lower.split("of", 1)
#                 n, d = left.strip().split("/")
#                 num = float(re.findall(r"[+-]?\d*\.?\d+", right)[0])
#                 return (float(n) / float(d)) * num

#         except:
#             return None

#     def _handle_trig(self, q: str) -> Optional[float]:
#         m = re.search(
#             r"(sin|cos|tan)\s*\(?\s*([+-]?\d+(\.\d+)?)\s*\)?\s*(deg|degree|degrees|°|rad|radians|in radians|in degrees)?",
#             q.lower()
#         )
#         if m:
#             func = m.group(1)
#             angle = float(m.group(2))
#             unit = m.group(4)

#             if unit and ("rad" in unit):
#                 rad = angle
#             else:
#                 rad = math.radians(angle)

#             if func == "sin": return round(math.sin(rad), 10)
#             if func == "cos": return round(math.cos(rad), 10)
#             if func == "tan":
#                 try: return round(math.tan(rad), 10)
#                 except: return "undefined"

#         return None

#     def _handle_sympy_expression(self, q: str):
#         try:
#             cleaned = q.replace("^", "**")
#             expr = sp.sympify(cleaned, evaluate=True)
#             if expr.is_Number:
#                 return float(expr)
#             return str(sp.N(expr))
#         except:
#             return None

#     def _handle_algebra(self, q: str):
#         try:
#             txt = q.lower().replace("solve", "").replace("^", "**")
#             if "=" in txt:
#                 left, right = txt.split("=")
#                 x = sp.symbols("x")
#                 eq = sp.Eq(sp.sympify(left), sp.sympify(right))
#                 return sp.solve(eq, dict=True)
#             else:
#                 return sp.solve(sp.sympify(txt))
#         except:
#             return None

#     def _handle_calculus(self, q: str):
#         ql = q.lower()

#         # derivative
#         m = re.search(r"(derivative|differentiate|d/dx)\s+(.+)", ql)
#         if m:
#             expr = m.group(2).replace("^", "**")
#             x = sp.symbols("x")
#             return str(sp.diff(sp.sympify(expr), x))

#         # integral
#         m = re.search(r"(integral|integrate)\s+(.+)", ql)
#         if m:
#             expr = m.group(2).replace("^", "**")
#             x = sp.symbols("x")
#             return str(sp.integrate(sp.sympify(expr), x))

#         # limit
#         m = re.search(r"limit\s+of\s+(.+)\s+as\s+x->\s*([\-0-9inf\+\.]+)", ql)
#         if m:
#             expr = m.group(1).replace("^", "**")
#             x = sp.symbols("x")
#             point = sp.sympify(m.group(2))
#             return str(sp.limit(sp.sympify(expr), x, point))

#         return None

#     def _handle_linear_algebra(self, q: str):
#         try:
#             if "det" in q.lower():
#                 m = re.search(r"(\[\[.*\]\])", q)
#                 mat = sp.Matrix(json.loads(m.group(1)))
#                 return float(mat.det())

#             if "inverse" in q.lower():
#                 m = re.search(r"(\[\[.*\]\])", q)
#                 mat = sp.Matrix(json.loads(m.group(1)))
#                 return mat.inv().tolist()

#         except:
#             return None

#     def _handle_statistics(self, q: str):
#         try:
#             lower = q.lower()
#             if any(k in lower for k in ["mean", "median", "mode", "variance", "std"]):
#                 nums = [float(n) for n in re.findall(r"[+-]?\d*\.?\d+", q)]
#                 import statistics as stats

#                 if "mean" in lower: return stats.mean(nums)
#                 if "median" in lower: return stats.median(nums)
#                 if "mode" in lower:
#                     try: return stats.mode(nums)
#                     except: return None
#                 if "variance" in lower: return stats.pvariance(nums)
#                 if "std" in lower: return stats.pstdev(nums)

#         except:
#             return None

#         def _handle_differential_eq(self, q: str) -> Optional[str]:
#             """
#                 Handle very simple ODEs of the form 'dy/dx = <expr_in_x>'.
#                 Returns a string like 'y = <antiderivative> + C' or None if not matched.
#             """
#         try:
#             txt = q.strip().lower().replace("^", "**")
#             # match patterns like: dy/dx = 2x  OR dy/dx=2*x
#             m = re.search(r"dy\s*/\s*dx\s*=\s*(.+)", txt)
#             if not m:
#                 return None

#             rhs = m.group(1).strip()
#             # Use sympy to integrate RHS w.r.t x
#             x = sp.symbols("x")
#             expr = sp.sympify(rhs)
#             antider = sp.integrate(expr, x)
#             antider_s = str(sp.simplify(antider))
#             # return with C; replace ** with ^ if you prefer caret notation
#             return f"y = {antider_s} + C"

#         except Exception:
#             return None



# # agents/math_agent.py
# # """
# # HYBRID MATH AGENT (FINAL MERGED VERSION)
# # ----------------------------------------

# # This file merges:
# # ✅ All logic from your older MathAgent (fractions, trig, algebra, calculus, linear algebra, statistics)
# # ✅ PLUS API limiter
# # ✅ PLUS safe LLM fallback
# # ✅ PLUS original explain_concept()
# # """

# import math
# import re
# import sympy as sp
# import traceback
# import json
# import os
# from typing import Optional, Any, Dict

# # API limiter utility
# from utils.api_limiter import APILimiter

# # Optional OpenAI client
# try:
#     import openai
# except:
#     openai = None

# OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


# class MathAgent:
#     def __init__(self, use_llm_if_available=True):
#         self.use_llm = use_llm_if_available and bool(OPENAI_KEY) and openai is not None

#         # API rate limiting
#         self.api_limiter = APILimiter(
#             max_calls_per_min=5,
#             max_calls_per_session=20
#         )

#     # ------------------------------------------------------------------------
#     # LLM WRAPPER WITH LIMITER & FALLBACK
#     # ------------------------------------------------------------------------
#     def _llm_solve(self, prompt: str) -> str:
#         """LLM wrapper that NEVER crashes and respects rate limits."""
#         if not self.use_llm:
#             return ""

#         allowed, reason = self.api_limiter.allowed()
#         if not allowed:
#             if reason == "rate_limit":
#                 return "LLM rate limit reached — using offline engine."
#             if reason == "session_limit":
#                 return "LLM session limit reached."
#             return "LLM unavailable — offline mode."

#         try:
#             self.api_limiter.record_call()
#             openai.api_key = OPENAI_KEY

#             resp = openai.ChatCompletion.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=512
#             )

#             return resp["choices"][0]["message"]["content"].strip()

#         except:
#             return "LLM error — switched to offline mode."


#     # agents/math_agent.py 

# def _handle_trig(self, q: str) -> Optional[float]:
#     """
#     Recognize sin/cos/tan queries with explicit units (degrees or radians)
#     Examples handled:
#       - "sin 90"
#       - "sin 90 degrees"
#       - "sin 90 deg"
#       - "sin 90 in radians"
#       - "sin(90) radians"
#       - "sin 1.5708 rad"
#     """
#     q_clean = q.strip().lower()

#     # capture patterns like "sin 90 in radians" or "sin(90) degrees"
#     m = re.search(
#         r"\b(sin|cos|tan)\b\s*\(?\s*([+-]?\d+(\.\d+)?)\s*\)?\s*(?:in\s+)?\s*(deg|degree|degrees|°|rad|radians)?",
#         q_clean, flags=re.I
#     )
#     if m:
#         func = m.group(1).lower()
#         angle = float(m.group(2))
#         unit_token = m.group(4)

#         # If explicit unit says radians -> do NOT convert.
#         if unit_token and ("rad" in unit_token):
#             rad = angle
#         # if explicit unit says degrees or symbol -> convert to radians
#         elif unit_token and ("deg" in unit_token or "degree" in unit_token or "°" in (unit_token or "")):
#             rad = math.radians(angle)
#         else:
#             # No explicit unit: default behavior:
#             # - If user wrote "in radians" earlier, we would have matched it in unit_token.
#             # - For ambiguous queries like "sin 90", prefer degrees (common), but you can change to radians if desired.
#             # We'll use degrees as default to match common user expectation.
#             rad = math.radians(angle)

#         if func == "sin":
#             return round(math.sin(rad), 10)
#         if func == "cos":
#             return round(math.cos(rad), 10)
#         if func == "tan":
#             try:
#                 return round(math.tan(rad), 10)
#             except Exception:
#                 return "undefined"

#     return None


#     # ------------------------------------------------------------------------
#     # PUBLIC ENTRYPOINT — SOLVE
#     # ------------------------------------------------------------------------
#     def solve(self, question: str) -> Dict[str, Any]:
#         q = question.strip()
#         lower = q.lower()

#         try:
#             wants_steps = any(k in lower for k in ["explain", "steps", "derive", "step-by-step"])

#             # ---------------- FRACTIONS ----------------
#             frac = self._handle_fraction_of(q)
#             if frac is not None:
#                 return {"answer": frac, "method": "rules"}

#             # ---------------- TRIG ----------------
#             trig = self._handle_trig(q)
#             if trig is not None:
#                 return {"answer": trig, "method": "rules"}

#             # ---------------- BASIC SYMPY EXPR ----------------
#             sym_basic = self._handle_sympy_expression(q)
#             if sym_basic is not None and not wants_steps:
#                 return {"answer": sym_basic, "method": "sympy"}

#             # ---------------- ALGEBRA ----------------
#             algebra = self._handle_algebra(q)
#             if algebra is not None and not wants_steps:
#                 return {"answer": algebra, "method": "sympy"}

#             # ---------------- CALCULUS ----------------
#             calc = self._handle_calculus(q)
#             if calc is not None:
#                 return {"answer": calc, "method": "sympy"}

#             # ---------------- LINEAR ALGEBRA ----------------
#             lin = self._handle_linear_algebra(q)
#             if lin is not None:
#                 return {"answer": lin, "method": "sympy"}

#             # ---------------- STATISTICS ----------------
#             stats = self._handle_statistics(q)
#             if stats is not None:
#                 return {"answer": stats, "method": "rules"}

#             # ---------------- LLM FALLBACK ----------------
#             if self.use_llm:
#                 llm_ans = self._llm_solve(f"Solve and explain step-by-step: {q}")
#                 return {"answer": llm_ans, "method": "llm"}

#             # ---------------- FINAL SYMPY FALLBACK ----------------
#             if sym_basic is not None:
#                 return {"answer": sym_basic, "method": "sympy"}

#             return {"answer": "Sorry, I couldn't solve that.", "method": "none"}

#         except Exception as e:
#             return {"answer": str(e), "method": "error", "trace": traceback.format_exc()}

#     # ------------------------------------------------------------------------
#     # HANDLERS — original logic preserved
#     # ------------------------------------------------------------------------
#     def _handle_fraction_of(self, q: str) -> Optional[float]:
#         try:
#             lower = q.lower()

#             if "half of" in lower:
#                 nums = re.findall(r"[-+]?\d*\.?\d+", lower)
#                 if nums:
#                     return float(nums[0]) / 2

#             if "of" in lower and "/" in lower:
#                 left, right = lower.split("of", 1)
#                 frac = left.strip()
#                 num = float(re.findall(r"[-+]?\d*\.?\d+", right)[0])
#                 n, d = frac.split("/")
#                 return (float(n) / float(d)) * num

#         except:
#             return None

#     def _handle_trig(self, q: str) -> Optional[float]:
#         m = re.search(
#             r"(sin|cos|tan)\s*\(?\s*([+-]?\d+(\.\d+)?)\s*(deg|degrees|°|rad|radians)?\s*\)?",
#             q, re.I
#         )
#         if m:
#             func = m.group(1).lower()
#             angle = float(m.group(2))
#             unit = m.group(4)

#             rad = angle if (unit and unit.startswith("rad")) else math.radians(angle)

#             if func == "sin": return round(math.sin(rad), 10)
#             if func == "cos": return round(math.cos(rad), 10)
#             if func == "tan":
#                 try:
#                     return round(math.tan(rad), 10)
#                 except:
#                     return "undefined"

#         return None

#     def _handle_sympy_expression(self, q: str) -> Optional[Any]:
#         try:
#             cleaned = q.strip().replace("^", "**")

#             expr = sp.sympify(cleaned, evaluate=True)
#             if expr.is_Number:
#                 return float(expr)
#             return str(sp.N(expr))

#         except:
#             return None

#     def _handle_algebra(self, q: str) -> Optional[Any]:
#         try:
#             if "solve" in q.lower() or "=" in q:
#                 expr = q.lower().replace("solve", "").replace("^", "**")

#                 if "=" in expr:
#                     left, right = expr.split("=")
#                     x = sp.symbols("x")
#                     eq = sp.Eq(sp.sympify(left), sp.sympify(right))
#                     sol = sp.solve(eq, dict=True)
#                     return sol
#                 else:
#                     x = sp.symbols("x")
#                     return sp.solve(sp.sympify(expr), x)

#         except:
#             return None

#     def _handle_calculus(self, q: str) -> Optional[Any]:
#         lower = q.lower()

#         # derivative
#         m = re.search(r"(derivative|differentiate|d/dx)\s+(.+)", lower)
#         if m:
#             expr = m.group(2).replace("^", "**")
#             x = sp.symbols("x")
#             return str(sp.diff(sp.sympify(expr), x))

#         # integral
#         m = re.search(r"(integral|integrate)\s+(.+)", lower)
#         if m:
#             expr = m.group(2).replace("^", "**")
#             x = sp.symbols("x")
#             return str(sp.integrate(sp.sympify(expr), x))

#         # limit
#         m = re.search(r"limit\s+of\s+(.+)\s+as\s+x\s*->\s*([a-z0-9+\-inf\.]+)", lower)
#         if m:
#             expr = m.group(1).replace("^", "**")
#             point = sp.sympify(m.group(2))
#             x = sp.symbols("x")
#             return str(sp.limit(sp.sympify(expr), x, point))

#         return None

#     def _handle_linear_algebra(self, q: str) -> Optional[Any]:
#         try:
#             lower = q.lower()

#             # determinant of [[1,2],[3,4]]
#             if "determinant" in lower or "det" in lower:
#                 m = re.search(r"(\[\[.*\]\])", q)
#                 if m:
#                     mat = sp.Matrix(json.loads(m.group(1)))
#                     return float(mat.det())

#             # inverse
#             if "inverse" in lower:
#                 m = re.search(r"(\[\[.*\]\])", q)
#                 if m:
#                     mat = sp.Matrix(json.loads(m.group(1)))
#                     inv = mat.inv()
#                     return inv.tolist()

#         except:
#             return None

#     def _handle_statistics(self, q: str) -> Optional[Any]:
#         try:
#             lower = q.lower()
#             if any(k in lower for k in ["mean", "median", "mode", "variance", "std"]):
#                 nums = re.findall(r"[+-]?\d*\.?\d+", q)
#                 nums = [float(n) for n in nums]

#                 import statistics as stats

#                 if "mean" in lower: return stats.mean(nums)
#                 if "median" in lower: return stats.median(nums)
#                 if "mode" in lower:
#                     try: return stats.mode(nums)
#                     except: return None
#                 if "variance" in lower: return stats.pvariance(nums)
#                 if "std" in lower: return stats.pstdev(nums)

#         except:
#             return None

#     # ------------------------------------------------------------------------
#     # ORIGINAL CONCEPT EXPLAINER (PRESERVED)
#     # ------------------------------------------------------------------------
#     def explain_concept(self, topic: str) -> str:
#         canned = {
#             "arithmetic": "Arithmetic studies basic operations such as addition, subtraction, multiplication, and division.",
#             "algebra": "Algebra uses symbols and variables to express equations.",
#             "geometry": "Geometry studies shapes, angles, and figures.",
#             "trigonometry": "Trigonometry studies relationships between angles and sides of triangles.",
#             "calculus": "Calculus studies change — derivatives and integrals.",
#             "statistics": "Statistics analyzes data and probability.",
#             "number theory": "Number theory studies properties of integers.",
#             "linear algebra": "Linear algebra studies matrices and vectors."
#         }
#         t = topic.lower()
#         if t in canned and not self.use_llm:
#             return canned[t]

#         if self.use_llm:
#             return self._llm_solve(f"Explain {topic} for a middle school student.")

#         return canned.get(t, "No explanation available.")




# # agents/math_agent.py
# import re

# class MathAgent:
#     def solve(self, question: str):
#         """
#         Very simple math solver using Python eval safely.
#         Supports fractions, %, basic arithmetic, word problems.
#         """

#         # Extract numbers and operators
#         q = question.lower()

#         # Handle word forms like "half of 20"
#         if "half of" in q:
#             num = re.findall(r"\d+", q)
#             if num:
#                 return float(num[0]) / 2

#         # Handle "1/2 of 20"
#         if "of" in q and "/" in q:
#             parts = q.split("of")
#             frac = parts[0].strip()
#             num = float(parts[1].strip())

#             try:
#                 numerator, denominator = frac.split("/")
#                 numerator = float(numerator)
#                 denominator = float(denominator)
#                 return (numerator / denominator) * num
#             except:
#                 pass

#         # Safe eval for basic math (only digits and + - * / .)
#         safe_expr = re.sub(r"[^0-9\+\-\*\/\.]", "", q)

#         if safe_expr:
#             try:
#                 return eval(safe_expr)
#             except:
#                 pass

#         return None
