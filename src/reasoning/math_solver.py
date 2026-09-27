"""Autonomous Mathematical & Algebraic Reasoning Engine for Genius.

Provides formal derivations, algebraic expansions, equation solving, calculus,
geometry, trigonometry, and arithmetic evaluation for foundational edge reasoning.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple


class MathSolver:
    """Rigorous analytical mathematics and algebraic identities solver."""

    @classmethod
    def solve(cls, query: str, lang_style: str = "en") -> Optional[Tuple[str, str]]:
        """Solves mathematical and algebraic queries, returning (think_trace, response_markdown)."""
        q = query.strip()
        q_lower = q.lower()
        q_norm = re.sub(r"\s+", " ", q_lower)

        # 1. Algebraic Identities & Binomial Expansions
        res = cls._solve_algebraic_identities(q_norm, lang_style)
        if res:
            return res

        # 2. Quadratic & Linear Equations
        res = cls._solve_equations(q_norm, lang_style)
        if res:
            return res

        # 3. Trigonometry & Geometric Formulas
        res = cls._solve_geometry_trig(q_norm, lang_style)
        if res:
            return res

        # 4. Calculus (Derivatives & Integrals)
        res = cls._solve_calculus(q_norm, lang_style)
        if res:
            return res

        # 5. Direct Arithmetic & Percentages & Physics
        res = cls._solve_arithmetic_physics(q_norm, lang_style)
        if res:
            return res

        return None

    @classmethod
    def _solve_algebraic_identities(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles (a+b)^2, (a-b)^2, a^2-b^2, (a+b)^3, (a+b+c)^2, etc."""
        # (a + b)^2 or a+b whole square
        if re.search(r"(\(?a\s*\+\s*b\)?\s*(whole\s*)?square|\ba\s*\+\s*b\s+whole\s+square|\(a\s*\+\s*b\)\s*\^?\s*2)", q):
            think = (
                "[Query Deconstruction]: Identifying algebraic identity expansion: '(a + b)^2' (Square of a Binomial Sum).\n"
                "[Mathematical Axiom]: Expanding through ring distributivity: (a + b)(a + b) = a(a + b) + b(a + b).\n"
                "[Step-by-Step Derivation]:\n"
                "  1. (a + b)^2 = (a + b)(a + b)\n"
                "  2. = a(a + b) + b(a + b)  [Distributive property]\n"
                "  3. = a^2 + ab + ba + b^2\n"
                "  4. Since multiplication is commutative in standard fields (ab = ba):\n"
                "     = a^2 + 2ab + b^2\n"
                "[Verification]: For a=2, b=3: LHS = (2+3)^2 = 25. RHS = 2^2 + 2(2)(3) + 3^2 = 4 + 12 + 9 = 25. LHS == RHS (Q.E.D.)."
            )
            resp = (
                "The expansion of **\\((a + b)^2\\)** (a plus b whole square) is:\n\n"
                "$$\\mathbf{(a + b)^2 = a^2 + 2ab + b^2}$$\n\n"
                "### 📐 Step-by-Step Algebraic Proof:\n"
                "1. **Write as multiplication of two binomial factors**:\n"
                "   $$(a + b)^2 = (a + b)(a + b)$$\n\n"
                "2. **Apply the distributive law (FOIL method)**:\n"
                "   $$= a \\cdot (a + b) + b \\cdot (a + b)$$\n"
                "   $$= a^2 + ab + ba + b^2$$\n\n"
                "3. **Combine like terms** (since multiplication is commutative, \\(ba = ab\\)):\n"
                "   $$= a^2 + (ab + ab) + b^2$$\n"
                "   $$\\mathbf{= a^2 + 2ab + b^2}$$\n\n"
                "### 💡 Geometric Interpretation:\n"
                "A large square of side \\((a + b)\\) has total area \\((a + b)^2\\). It is composed of:\n"
                "* One square of area \\(a^2\\)\n"
                "* One square of area \\(b^2\\)\n"
                "* Two rectangles, each of area \\(ab\\)\n"
                "* **Total Area** = \\(a^2 + 2ab + b^2\\)\n\n"
                "### 🔗 Related Core Identities:\n"
                "* \\((a - b)^2 = a^2 - 2ab + b^2\\)\n"
                "* \\(a^2 - b^2 = (a - b)(a + b)\\)\n"
                "* \\((a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3 = a^3 + b^3 + 3ab(a + b)\\)\n"
                "* \\((a + b + c)^2 = a^2 + b^2 + c^2 + 2(ab + bc + ca)\\)"
            )
            return think, resp

        # (a - b)^2 or a-b whole square
        if re.search(r"(\(?a\s*\-\s*b\)?\s*(whole\s*)?square|\ba\s*\-\s*b\s+whole\s+square|\(a\s*\-\s*b\)\s*\^?\s*2)", q):
            think = (
                "[Query Deconstruction]: Identifying algebraic identity expansion: '(a - b)^2' (Square of a Binomial Difference).\n"
                "[Mathematical Axiom]: Expanding through ring distributivity: (a - b)(a - b) = a(a - b) - b(a - b).\n"
                "[Step-by-Step Derivation]:\n"
                "  1. (a - b)^2 = (a - b)(a - b)\n"
                "  2. = a(a - b) - b(a - b)\n"
                "  3. = a^2 - ab - ba + b^2  [(-b)*(-b) = +b^2]\n"
                "  4. Since ba = ab: = a^2 - 2ab + b^2\n"
                "[Verification]: For a=5, b=2: LHS = (5-2)^2 = 9. RHS = 25 - 2(5)(2) + 4 = 25 - 20 + 4 = 9. LHS == RHS (Q.E.D.)."
            )
            resp = (
                "The expansion of **\\((a - b)^2\\)** (a minus b whole square) is:\n\n"
                "$$\\mathbf{(a - b)^2 = a^2 - 2ab + b^2}$$\n\n"
                "### 📐 Step-by-Step Algebraic Proof:\n"
                "1. **Write as multiplication of two binomials**:\n"
                "   $$(a - b)^2 = (a - b)(a - b)$$\n\n"
                "2. **Distribute terms**:\n"
                "   $$= a \\cdot (a - b) - b \\cdot (a - b)$$\n"
                "   $$= a^2 - ab - ba + (-b)(-b)$$\n"
                "   $$= a^2 - ab - ba + b^2$$\n\n"
                "3. **Combine middle terms** (\\(-ab - ba = -2ab\\)):\n"
                "   $$\\mathbf{= a^2 - 2ab + b^2}$$\n\n"
                "### 🔗 Related Identities:\n"
                "* \\((a + b)^2 = a^2 + 2ab + b^2\\)\n"
                "* \\((a + b)^2 - (a - b)^2 = 4ab\\)\n"
                "* \\((a + b)^2 + (a - b)^2 = 2(a^2 + b^2)\\)"
            )
            return think, resp

        # a^2 - b^2 or difference of squares
        if re.search(r"(\ba\^?2\s*\-\s*b\^?2\b|difference\s+of\s+(two\s+)?squares)", q):
            think = (
                "[Query Deconstruction]: Difference of two squares identity: a^2 - b^2.\n"
                "[Mathematical Axiom]: Factoring binomial into conjugates (a - b)(a + b)."
            )
            resp = (
                "The factorization of **\\(a^2 - b^2\\)** (Difference of Squares) is:\n\n"
                "$$\\mathbf{a^2 - b^2 = (a - b)(a + b)}$$\n\n"
                "### Verification:\n"
                "$$(a - b)(a + b) = a^2 + ab - ba - b^2 = a^2 - b^2$$"
            )
            return think, resp

        # (a + b)^3 or a+b whole cube
        if re.search(r"(\(?a\s*\+\s*b\)?\s*(whole\s*)?cube|\ba\s*\+\s*b\s+whole\s+cube|\(a\s*\+\s*b\)\s*\^?\s*3)", q):
            think = (
                "[Query Deconstruction]: Binomial expansion for power 3: (a + b)^3.\n"
                "[Derivation]: (a + b)^3 = (a + b)(a + b)^2 = (a + b)(a^2 + 2ab + b^2)."
            )
            resp = (
                "The expansion of **\\((a + b)^3\\)** (a plus b whole cube) is:\n\n"
                "$$\\mathbf{(a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3}$$\n\n"
                "Or in factored form:\n"
                "$$\\mathbf{(a + b)^3 = a^3 + b^3 + 3ab(a + b)}$$\n\n"
                "### Step-by-Step Derivation:\n"
                "1. Multiply \\((a + b)\\) by \\((a + b)^2\\):\n"
                "   $$(a + b)^3 = (a + b)(a^2 + 2ab + b^2)$$\n"
                "2. Expand by distribution:\n"
                "   $$= a(a^2 + 2ab + b^2) + b(a^2 + 2ab + b^2)$$\n"
                "   $$= a^3 + 2a^2b + ab^2 + a^2b + 2ab^2 + b^3$$\n"
                "3. Group like terms:\n"
                "   $$= a^3 + (2a^2b + a^2b) + (ab^2 + 2ab^2) + b^3$$\n"
                "   $$\\mathbf{= a^3 + 3a^2b + 3ab^2 + b^3}$$"
            )
            return think, resp

        # (a - b)^3 or a-b whole cube
        if re.search(r"(\(?a\s*\-\s*b\)?\s*(whole\s*)?cube|\ba\s*\-\s*b\s+whole\s+cube|\(a\s*\-\s*b\)\s*\^?\s*3)", q):
            think = "[Query Deconstruction]: Binomial expansion for power 3: (a - b)^3."
            resp = (
                "The expansion of **\\((a - b)^3\\)** (a minus b whole cube) is:\n\n"
                "$$\\mathbf{(a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3}$$\n\n"
                "Or in factored form:\n"
                "$$\\mathbf{(a - b)^3 = a^3 - b^3 - 3ab(a - b)}$$"
            )
            return think, resp

        # (a + b + c)^2 or a+b+c whole square
        if re.search(r"(\(?a\s*\+\s*b\s*\+\s*c\)?\s*(whole\s*)?square|\(a\s*\+\s*b\s*\+\s*c\)\s*\^?\s*2)", q):
            think = "[Query Deconstruction]: Trinomial square expansion: (a + b + c)^2."
            resp = (
                "The expansion of **\\((a + b + c)^2\\)** is:\n\n"
                "$$\\mathbf{(a + b + c)^2 = a^2 + b^2 + c^2 + 2ab + 2bc + 2ca}$$\n\n"
                "Or factoring out the 2:\n"
                "$$\\mathbf{(a + b + c)^2 = a^2 + b^2 + c^2 + 2(ab + bc + ca)}$$"
            )
            return think, resp

        # a^3 + b^3
        if re.search(r"\ba\^?3\s*\+\s*b\^?3\b", q):
            think = "[Query Deconstruction]: Sum of cubes factorization: a^3 + b^3."
            resp = (
                "The factorization of **\\(a^3 + b^3\\)** (Sum of Cubes) is:\n\n"
                "$$\\mathbf{a^3 + b^3 = (a + b)(a^2 - ab + b^2)}$$"
            )
            return think, resp

        # a^3 - b^3
        if re.search(r"\ba\^?3\s*\-\s*b\^?3\b", q):
            think = "[Query Deconstruction]: Difference of cubes factorization: a^3 - b^3."
            resp = (
                "The factorization of **\\(a^3 - b^3\\)** (Difference of Cubes) is:\n\n"
                "$$\\mathbf{a^3 - b^3 = (a - b)(a^2 + ab + b^2)}$$"
            )
            return think, resp

        return None

    @classmethod
    def _solve_equations(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles linear and quadratic equations."""
        # Quadratic formula inquiry
        if any(w in q for w in ["quadratic formula", "shridharacharya", "sridharacharya", "ax^2 + bx + c", "roots of quadratic"]):
            think = (
                "[Query Deconstruction]: Quadratic equation standard form ax^2 + bx + c = 0.\n"
                "[Derivation]: Completing the square: x^2 + (b/a)x + (b/2a)^2 = -c/a + (b/2a)^2."
            )
            resp = (
                "For a quadratic equation in standard form **\\(ax^2 + bx + c = 0\\)** (where \\(a \\neq 0\\)), "
                "the roots are given by the **Quadratic Formula**:\n\n"
                "$$\\mathbf{x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}}$$\n\n"
                "### Discriminant (\\(\\Delta = b^2 - 4ac\\)) Nature of Roots:\n"
                "* **\\(\\Delta > 0\\)**: Two distinct real roots.\n"
                "* **\\(\\Delta = 0\\)**: Two equal real roots (\\(x = -b / 2a\\)).\n"
                "* **\\(\\Delta < 0\\)**: Two complex conjugate roots."
            )
            return think, resp

        # Linear equation e.g. solve 2x + 5 = 15 or 2x+5=15
        lin_match = re.search(r"(?:solve\s+)?([+-]?\d*)\s*x\s*([+-]\s*\d+)\s*=\s*([+-]?\d+)", q)
        if lin_match:
            try:
                a_str = lin_match.group(1).replace(" ", "")
                if a_str in ("", "+"):
                    a = 1
                elif a_str == "-":
                    a = -1
                else:
                    a = int(a_str)

                b = int(lin_match.group(2).replace(" ", ""))
                c = int(lin_match.group(3).replace(" ", ""))

                # ax + b = c  =>  ax = c - b  =>  x = (c - b) / a
                rhs_step = c - b
                x_val = rhs_step / a
                x_str = str(int(x_val)) if x_val.is_integer() else f"{x_val:.3f}"

                think = (
                    f"[Query Deconstruction]: Solving linear equation {a}x + ({b}) = {c}.\n"
                    f"[Step 1]: Isolate variable term: {a}x = {c} - ({b}) = {rhs_step}.\n"
                    f"[Step 2]: Divide by coefficient {a}: x = {rhs_step} / {a} = {x_str}."
                )
                resp = (
                    f"Solving the linear equation **\\({a}x {'+' if b >= 0 else '-'} {abs(b)} = {c}\\)**:\n\n"
                    f"1. **Isolate the variable term by subtracting {b} from both sides**:\n"
                    f"   $${a}x = {c} - ({b})$$\n"
                    f"   $${a}x = {rhs_step}$$\n\n"
                    f"2. **Divide both sides by {a}**:\n"
                    f"   $$x = \\frac{{{rhs_step}}}{{{a}}}$$\n"
                    f"   $$\\mathbf{{x = {x_str}}}$$\n\n"
                    f"### Verification:\n"
                    f"Substituting \\(x = {x_str}\\) into LHS: \\({a}({x_str}) + ({b}) = {a * x_val + b} = {c}\\) (Valid)."
                )
                return think, resp
            except Exception:
                pass

        return None

    @classmethod
    def _solve_geometry_trig(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles circle, triangle, sphere geometry, and trig formulas."""
        # Pythagorean theorem
        if any(w in q for w in ["pythagor", "hypotenuse", "right angle triangle"]):
            think = "[Query Deconstruction]: Pythagorean Theorem relating sides a, b, c of a right-angled triangle."
            resp = (
                "The **Pythagorean Theorem** states that in any right-angled triangle:\n\n"
                "$$\\mathbf{a^2 + b^2 = c^2}$$\n\n"
                "Where:\n"
                "* \\(a\\) and \\(b\\) are the lengths of the two legs (perpendicular and base).\n"
                "* \\(c\\) is the length of the **hypotenuse** (the side opposite the right angle):\n"
                "  $$\\mathbf{c = \\sqrt{a^2 + b^2}}$$"
            )
            return think, resp

        # Area of circle
        if "area of circle" in q or "circle area" in q or "vritt ka kshetrafal" in q:
            think = "[Query Deconstruction]: Formula for area of circle of radius r."
            resp = (
                "The **Area of a Circle** with radius \\(r\\) is:\n\n"
                "$$\\mathbf{A = \\pi r^2}$$\n\n"
                "* Where \\(\\pi \\approx 3.14159\\) (or \\(\\frac{22}{7}\\)).\n"
                "* Circumference: \\(\\mathbf{C = 2\\pi r}\\) (or \\(\\pi d\\), where \\(d = 2r\\))."
            )
            return think, resp

        # Volume of sphere
        if "volume of sphere" in q or "sphere volume" in q:
            think = "[Query Deconstruction]: Formula for volume of a sphere."
            resp = (
                "The **Volume of a Sphere** with radius \\(r\\) is:\n\n"
                "$$\\mathbf{V = \\frac{4}{3}\\pi r^3}$$\n\n"
                "* Surface Area: \\(\\mathbf{A = 4\\pi r^2}\\)"
            )
            return think, resp

        # Basic Trig Identities
        if any(w in q for w in ["trigonometric identities", "sin^2 + cos^2", "sin square plus cos square"]):
            think = "[Query Deconstruction]: Core trigonometric Pythagorean identities."
            resp = (
                "The core **Trigonometric Pythagorean Identities** are:\n\n"
                "1. $$\\mathbf{\\sin^2\\theta + \\cos^2\\theta = 1}$$\n"
                "2. $$\\mathbf{1 + \\tan^2\\theta = \\sec^2\\theta}$$\n"
                "3. $$\\mathbf{1 + \\cot^2\\theta = \\csc^2\\theta}$$\n\n"
                "### Double Angle Formulas:\n"
                "* \\(\\sin(2\\theta) = 2\\sin\\theta\\cos\\theta\\)\n"
                "* \\(\\cos(2\\theta) = \\cos^2\\theta - \\sin^2\\theta = 2\\cos^2\\theta - 1 = 1 - 2\\sin^2\\theta\\)"
            )
            return think, resp

        return None

    @classmethod
    def _solve_calculus(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles common derivatives and integrals."""
        if any(w in q for w in ["derivative of", "d/dx", "differentiation of"]):
            # Specific standard functions
            if "sin" in q:
                return (
                    "[Query Deconstruction]: First derivative of sin(x).",
                    "$$\\mathbf{\\frac{d}{dx}[\\sin(x)] = \\cos(x)}$$"
                )
            if "cos" in q:
                return (
                    "[Query Deconstruction]: First derivative of cos(x).",
                    "$$\\mathbf{\\frac{d}{dx}[\\cos(x)] = -\\sin(x)}$$"
                )
            if "tan" in q:
                return (
                    "[Query Deconstruction]: First derivative of tan(x).",
                    "$$\\mathbf{\\frac{d}{dx}[\\tan(x)] = \\sec^2(x)}$$"
                )
            if "e^x" in q or "exp" in q:
                return (
                    "[Query Deconstruction]: Derivative of exponential function e^x.",
                    "$$\\mathbf{\\frac{d}{dx}[e^x] = e^x}$$"
                )
            if "ln" in q or "log" in q:
                return (
                    "[Query Deconstruction]: Derivative of natural logarithm ln(x).",
                    "$$\\mathbf{\\frac{d}{dx}[\\ln(x)] = \\frac{1}{x}} \\quad (x > 0)$$"
                )
            # Power rule general
            return (
                "[Query Deconstruction]: Power rule for derivatives.",
                "### Power Rule of Differentiation:\n$$\\mathbf{\\frac{d}{dx}[x^n] = n x^{n-1}}$$\n\n* **Product Rule**: \\((uv)' = u'v + uv'\\)\n* **Quotient Rule**: \\(\\left(\\frac{u}{v}\\right)' = \\frac{u'v - uv'}{v^2}\\)\n* **Chain Rule**: \\(\\frac{d}{dx}[f(g(x))] = f'(g(x)) \\cdot g'(x)\\)"
            )

        if any(w in q for w in ["integral of", "integration of", "anti-derivative"]):
            return (
                "[Query Deconstruction]: Fundamental power rule of integration.",
                "### Power Rule of Integration:\n$$\\mathbf{\\int x^n dx = \\frac{x^{n+1}}{n+1} + C} \\quad (n \\neq -1)$$\n\n* \\(\\int \\frac{1}{x} dx = \\ln|x| + C\\)\n* \\(\\int e^x dx = e^x + C\\)\n* \\(\\int \\sin(x) dx = -\\cos(x) + C\\)\n* \\(\\int \\cos(x) dx = \\sin(x) + C\\)"
            )

        return None

    @classmethod
    def _solve_arithmetic_physics(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles direct arithmetic expressions, speed/distance/time, and percentages."""
        # Speed distance time calculation (e.g. 120km in 2 hours or speed of train)
        sdt_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|miles|m)\b.*?(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr|seconds|sec|s)\b", q)
        if sdt_match and ("speed" in q or "velocity" in q or "chal" in q or "gati" in q):
            try:
                dist = float(sdt_match.group(1))
                time_val = float(sdt_match.group(2))
                if time_val > 0:
                    spd = dist / time_val
                    spd_str = str(int(spd)) if spd.is_integer() else f"{spd:.2f}"
                    think = (
                        f"[Query Deconstruction]: Physical kinematics: Speed = Distance / Time.\n"
                        f"[Given]: Distance = {dist} km, Time = {time_val} hours.\n"
                        f"[Computation]: Speed = {dist} / {time_val} = {spd_str} km/h."
                    )
                    resp = (
                        f"### Kinematics Solution:\n\n"
                        f"* **Formula**: $$\\mathbf{{\\text{{Speed}} = \\frac{{\\text{{Distance}}}}{{\\text{{Time}}}}}}$$\n"
                        f"* **Given**: Distance = {dist} km, Time = {time_val} hours\n"
                        f"* **Calculation**: $$\\text{{Speed}} = \\frac{{{dist}}}{{{time_val}}} = \\mathbf{{{spd_str}\\text{{ km/h}}}}$$\n\n"
                        f"In SI units (m/s): \\({spd_str} \\times \\frac{{5}}{{18}} = \\mathbf{{{(spd * 5/18):.2f}\\text{{ m/s}}}}\\)."
                    )
                    return think, resp
            except Exception:
                pass

        # Percentage e.g. 15% of 600
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of|ka)?\s*(\d+(?:\.\d+)?)", q)
        if pct_match:
            try:
                pct = float(pct_match.group(1))
                total = float(pct_match.group(2))
                ans = (pct / 100.0) * total
                ans_str = str(int(ans)) if ans.is_integer() else f"{ans:.2f}"
                think = f"[Query Deconstruction]: Percentage calculation: {pct}% of {total} = ({pct} / 100) * {total} = {ans_str}."
                resp = (
                    f"$$\\mathbf{{{pct}\\% \\text{{ of }} {total} = \\frac{{{pct}}}{{100}} \\times {total} = {ans_str}}}$$"
                )
                return think, resp
            except Exception:
                pass

        # Safe arithmetic calculation (e.g. 25 * 4, 1024 / 16, 2^8, sqrt(144))
        # Square root
        sqrt_match = re.search(r"(?:sqrt|square\s+root\s+of)\s*\(?(\d+(?:\.\d+)?)\)?", q)
        if sqrt_match:
            val = float(sqrt_match.group(1))
            res = math.sqrt(val)
            res_str = str(int(res)) if res.is_integer() else f"{res:.4f}"
            return (
                f"[Query Deconstruction]: Square root of {val}: sqrt({val}) = {res_str}.",
                f"$$\\mathbf{{\\sqrt{{{val}}} = {res_str}}}$$"
            )

        return None
