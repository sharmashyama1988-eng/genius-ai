"""Autonomous Mathematical & Algebraic Reasoning Engine for Genius.

Provides formal derivations, algebraic expansions, equation solving, calculus,
geometry, trigonometry, series/progressions, statistics, combinatorics, unit
conversions, and safe AST expression evaluation for foundational edge reasoning.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable, Dict, Optional, Tuple


# Safe AST Mathematical Evaluator Configuration
_SAFE_OPERATORS: Dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log10,
    "ln": math.log,
    "exp": math.exp,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "fact": math.factorial,
    "factorial": math.factorial,
}

_SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast(node: ast.AST) -> Any:
    """Recursively evaluates a safe AST math expression."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    if isinstance(node, ast.Name):
        if node.id.lower() in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id.lower()]
        raise ValueError(f"Unknown mathematical variable: {node.id}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in _SAFE_FUNCTIONS:
                args = [_eval_ast(arg) for arg in node.args]
                return _SAFE_FUNCTIONS[func_name](*args)
        raise ValueError("Unsupported function call")
    raise ValueError(f"Unsupported AST node: {type(node)}")


class MathSolver:
    """Rigorous analytical mathematics, algebra, calculus, and arithmetic solver."""

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

        # 3. Series & Progressions (AP, GP, Special Sums)
        res = cls._solve_series(q_norm, lang_style)
        if res:
            return res

        # 4. Logarithm Laws & Identities
        res = cls._solve_logarithms(q_norm, lang_style)
        if res:
            return res

        # 5. Trigonometry & Geometric Formulas
        res = cls._solve_geometry_trig(q_norm, lang_style)
        if res:
            return res

        # 6. Calculus (Derivatives & Integrals)
        res = cls._solve_calculus(q_norm, lang_style)
        if res:
            return res

        # 7. Statistics & Combinatorics (nPr, nCr, Bayes)
        res = cls._solve_stats_combinatorics(q_norm, lang_style)
        if res:
            return res

        # 8. Unit & Physical Conversions
        res = cls._solve_unit_conversions(q_norm, lang_style)
        if res:
            return res

        # 9. Direct Arithmetic, Physics & AST Evaluator
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
    def _solve_series(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles Arithmetic Progression (AP), Geometric Progression (GP), and series sums."""
        # AP formulas
        if any(w in q for w in ["arithmetic progression", "ap formula", "nth term of ap", "sum of ap"]):
            think = "[Query Deconstruction]: Formulating standard Arithmetic Progression (AP) equations."
            resp = (
                "### 📈 Arithmetic Progression (AP) Core Formulas:\n\n"
                "Let \\(a\\) be the first term and \\(d\\) be the common difference.\n\n"
                "1. **\\(n\\)-th Term (\\(a_n\\) or \\(T_n\\))**:\n"
                "   $$\\mathbf{a_n = a + (n - 1)d}$$\n\n"
                "2. **Sum of First \\(n\\) Terms (\\(S_n\\))**:\n"
                "   $$\\mathbf{S_n = \\frac{n}{2} [2a + (n - 1)d] = \\frac{n}{2} (a + l)}$$\n"
                "   *(where \\(l = a_n\\) is the last term)*.\n\n"
                "3. **Common Difference (\\(d\\))**:\n"
                "   $$d = a_k - a_{k-1}$$"
            )
            return think, resp

        # GP formulas
        if any(w in q for w in ["geometric progression", "gp formula", "nth term of gp", "sum of gp"]):
            think = "[Query Deconstruction]: Formulating standard Geometric Progression (GP) equations."
            resp = (
                "### 📊 Geometric Progression (GP) Core Formulas:\n\n"
                "Let \\(a\\) be the first term and \\(r\\) be the common ratio (\\(r \\neq 1\\)).\n\n"
                "1. **\\(n\\)-th Term (\\(a_n\\) or \\(T_n\\))**:\n"
                "   $$\\mathbf{a_n = a \\cdot r^{n-1}}$$\n\n"
                "2. **Sum of First \\(n\\) Terms (\\(S_n\\))**:\n"
                "   $$\\mathbf{S_n = \\frac{a(r^n - 1)}{r - 1}} \\quad (r > 1) \\quad \\text{or} \\quad \\mathbf{S_n = \\frac{a(1 - r^n)}{1 - r}} \\quad (r < 1)$$\n\n"
                "3. **Sum to Infinity (\\(S_\\infty\\))** (valid only when \\(|r| < 1\\)):\n"
                "   $$\\mathbf{S_\\infty = \\frac{a}{1 - r}}$$"
            )
            return think, resp

        # Sum of first n natural numbers, squares, cubes
        if any(w in q for w in ["sum of first n natural numbers", "sum of natural numbers", "sum of squares of n"]):
            think = "[Query Deconstruction]: Closed-form summation formulas for power series."
            resp = (
                "### 🔢 Standard Series Summation Formulas:\n\n"
                "1. **Sum of first \\(n\\) natural numbers**:\n"
                "   $$\\mathbf{\\sum_{k=1}^n k = 1 + 2 + 3 + \\dots + n = \\frac{n(n + 1)}{2}}$$\n\n"
                "2. **Sum of squares of first \\(n\\) natural numbers**:\n"
                "   $$\\mathbf{\\sum_{k=1}^n k^2 = 1^2 + 2^2 + \\dots + n^2 = \\frac{n(n + 1)(2n + 1)}{6}}$$\n\n"
                "3. **Sum of cubes of first \\(n\\) natural numbers**:\n"
                "   $$\\mathbf{\\sum_{k=1}^n k^3 = 1^3 + 2^3 + \\dots + n^3 = \\left[ \\frac{n(n + 1)}{2} \\right]^2}$$"
            )
            return think, resp

        return None

    @classmethod
    def _solve_logarithms(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles logarithm rules and identities."""
        if any(w in q for w in ["log rules", "logarithm formulas", "laws of log", "log properties"]):
            think = "[Query Deconstruction]: Fundamental algebraic laws and properties of logarithms."
            resp = (
                "### 🪵 Core Laws of Logarithms:\n\n"
                "For any base \\(b > 0, b \\neq 1\\) and positive arguments \\(x, y\\):\n\n"
                "1. **Product Rule**:\n"
                "   $$\\mathbf{\\log_b(xy) = \\log_b(x) + \\log_b(y)}$$\n\n"
                "2. **Quotient Rule**:\n"
                "   $$\\mathbf{\\log_b\\left(\\frac{x}{y}\\right) = \\log_b(x) - \\log_b(y)}$$\n\n"
                "3. **Power Rule**:\n"
                "   $$\\mathbf{\\log_b(x^k) = k \\cdot \\log_b(x)}$$\n\n"
                "4. **Change of Base Formula**:\n"
                "   $$\\mathbf{\\log_b(x) = \\frac{\\log_a(x)}{\\log_a(b)} = \\frac{\\ln(x)}{\\ln(b)}}$$\n\n"
                "5. **Fundamental Identities**:\n"
                "   * \\(\\log_b(1) = 0\\)\n"
                "   * \\(\\log_b(b) = 1\\)\n"
                "   * \\(b^{\\log_b(x)} = x\\)"
            )
            return think, resp
        return None

    @classmethod
    def _solve_geometry_trig(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles geometry, trigonometry, and Pythagorean formulas."""
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

        # Volume of cylinder
        if "volume of cylinder" in q or "cylinder volume" in q:
            think = "[Query Deconstruction]: Formula for cylinder volume V = pi * r^2 * h."
            resp = (
                "The **Volume of a Cylinder** with radius \\(r\\) and height \\(h\\) is:\n\n"
                "$$\\mathbf{V = \\pi r^2 h}$$\n\n"
                "* Curved Surface Area (CSA): \\(\\mathbf{2\\pi r h}\\)\n"
                "* Total Surface Area (TSA): \\(\\mathbf{2\\pi r (r + h)}\\)"
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
                "* \\(\\cos(2\\theta) = \\cos^2\\theta - \\sin^2\\theta = 2\\cos^2\\theta - 1 = 1 - 2\\sin^2\\theta\\)\n"
                "* \\(\\tan(2\\theta) = \\frac{2\\tan\\theta}{1 - \\tan^2\\theta}\\)"
            )
            return think, resp

        return None

    @classmethod
    def _solve_calculus(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles common derivatives and integrals."""
        if any(w in q for w in ["derivative of", "d/dx", "differentiation of"]):
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
            return (
                "[Query Deconstruction]: Power rule for derivatives.",
                "### Power Rule of Differentiation:\n$$\\mathbf{\\frac{d}{dx}[x^n] = n x^{n-1}}$$\n\n"
                "* **Product Rule**: \\((uv)' = u'v + uv'\\)\n"
                "* **Quotient Rule**: \\(\\left(\\frac{u}{v}\\right)' = \\frac{u'v - uv'}{v^2}\\)\n"
                "* **Chain Rule**: \\(\\frac{d}{dx}[f(g(x))] = f'(g(x)) \\cdot g'(x)\\)"
            )

        if any(w in q for w in ["integral of", "integration of", "anti-derivative"]):
            return (
                "[Query Deconstruction]: Fundamental power rule of integration.",
                "### Power Rule of Integration:\n$$\\mathbf{\\int x^n dx = \\frac{x^{n+1}}{n+1} + C} \\quad (n \\neq -1)$$\n\n"
                "* \\(\\int \\frac{1}{x} dx = \\ln|x| + C\\)\n"
                "* \\(\\int e^x dx = e^x + C\\)\n"
                "* \\(\\int \\sin(x) dx = -\\cos(x) + C\\)\n"
                "* \\(\\int \\cos(x) dx = \\sin(x) + C\\)"
            )

        return None

    @classmethod
    def _solve_stats_combinatorics(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles Bayes' theorem, permutations, combinations, and basic statistics."""
        # Bayes theorem
        if "bayes" in q:
            think = "[Query Deconstruction]: Bayes' Theorem for conditional probability."
            resp = (
                "### 🎲 Bayes' Theorem:\n\n"
                "$$\\mathbf{P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}}$$\n\n"
                "Where:\n"
                "* \\(P(A|B)\\): **Posterior Probability** (probability of hypothesis \\(A\\) given evidence \\(B\\)).\n"
                "* \\(P(B|A)\\): **Likelihood** (probability of evidence \\(B\\) given hypothesis \\(A\\)).\n"
                "* \\(P(A)\\): **Prior Probability** of hypothesis \\(A\\).\n"
                "* \\(P(B)\\): **Marginal Probability** of evidence \\(B\\) (\\(\\sum_i P(B|A_i)P(A_i)\\))."
            )
            return think, resp

        # Combinatorics formulas (nPr & nCr)
        if any(w in q for w in ["permutation formula", "combination formula", "npr", "ncr"]):
            think = "[Query Deconstruction]: Combinatorics definitions for Permutations (nPr) and Combinations (nCr)."
            resp = (
                "### 🔢 Permutations & Combinations:\n\n"
                "1. **Permutations (Order Matters)**:\n"
                "   $$\\mathbf{P(n, r) = {}^n P_r = \\frac{n!}{(n - r)!}}$$\n\n"
                "2. **Combinations (Order Does NOT Matter)**:\n"
                "   $$\\mathbf{C(n, r) = {}^n C_r = \\binom{n}{r} = \\frac{n!}{r! (n - r)!}}$$\n\n"
                "### Key Property:\n"
                "* \\({}^n C_r = {}^n C_{n-r}\\)\n"
                "* \\({}^n C_0 = {}^n C_n = 1\\)\n"
                "* \\({}^n P_r = r! \\cdot {}^n C_r\\)"
            )
            return think, resp

        return None

    @classmethod
    def _solve_unit_conversions(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles Celsius <-> Fahrenheit <-> Kelvin temperature conversions."""
        # Temperature: e.g. convert 100 c to f or 37 celsius in fahrenheit
        c_to_f = re.search(r"(\d+(?:\.\d+)?)\s*(?:c|celsius)\s+(?:to|in|mein)\s+(?:f|fahrenheit)", q)
        if c_to_f:
            c = float(c_to_f.group(1))
            f = (c * 9.0 / 5.0) + 32.0
            think = f"[Query Deconstruction]: Temperature conversion: F = (C * 9/5) + 32. C={c} -> F={f:.2f}."
            resp = (
                f"### Temperature Conversion (Celsius to Fahrenheit):\n\n"
                f"* **Formula**: $$\\mathbf{{F = \\left(C \\times \\frac{{9}}{{5}}\\right) + 32}}$$\n"
                f"* **Calculation**: $$\\left({c} \\times 1.8\\right) + 32 = \\mathbf{{{f:.2f}^\\circ \\text{{F}}}}$$"
            )
            return think, resp

        f_to_c = re.search(r"(\d+(?:\.\d+)?)\s*(?:f|fahrenheit)\s+(?:to|in|mein)\s+(?:c|celsius)", q)
        if f_to_c:
            f = float(f_to_c.group(1))
            c = (f - 32.0) * 5.0 / 9.0
            think = f"[Query Deconstruction]: Temperature conversion: C = (F - 32) * 5/9. F={f} -> C={c:.2f}."
            resp = (
                f"### Temperature Conversion (Fahrenheit to Celsius):\n\n"
                f"* **Formula**: $$\\mathbf{{C = (F - 32) \\times \\frac{{5}}{{9}}}}$$\n"
                f"* **Calculation**: $$({f} - 32) \\times \\frac{{5}}{{9}} = \\mathbf{{{c:.2f}^\\circ \\text{{C}}}}$$"
            )
            return think, resp

        return None

    @classmethod
    def _solve_arithmetic_physics(cls, q: str, lang: str) -> Optional[Tuple[str, str]]:
        """Handles kinematics, percentages, square roots, and safe AST expression evaluation."""
        # Speed distance time calculation (e.g. 120km in 2 hours)
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

        # Square root: sqrt(144) or square root of 144
        sqrt_match = re.search(r"(?:sqrt|square\s+root\s+of)\s*\(?(\d+(?:\.\d+)?)\)?", q)
        if sqrt_match:
            try:
                val = float(sqrt_match.group(1))
                res = math.sqrt(val)
                res_str = str(int(res)) if res.is_integer() else f"{res:.4f}"
                return (
                    f"[Query Deconstruction]: Square root of {val}: sqrt({val}) = {res_str}.",
                    f"$$\\mathbf{{\\sqrt{{{val}}} = {res_str}}}$$"
                )
            except Exception:
                pass

        # General Safe AST Evaluation for expressions like "calculate 2^10 + 50 * 3"
        expr_cand = q
        for prefix in ["calculate", "solve", "evaluate", "what is", "compute", "value of"]:
            if expr_cand.startswith(prefix):
                expr_cand = expr_cand[len(prefix):].strip()
        expr_cand = expr_cand.rstrip(" =?").strip()

        # Check if expr_cand looks like a pure math expression
        if re.search(r"^[\d\.\s\+\-\*\/\^\(\)\,\w]+$", expr_cand) and re.search(r"[\+\-\*\/\^]", expr_cand):
            try:
                clean_expr = expr_cand.replace("^", "**").replace("×", "*").replace("÷", "/")
                tree = ast.parse(clean_expr, mode="eval")
                val = _eval_ast(tree)
                val_str = str(int(val)) if isinstance(val, (int, float)) and float(val).is_integer() else f"{val:.4f}"
                think = f"[Query Deconstruction]: Evaluating arithmetic expression: {expr_cand} = {val_str}."
                resp = f"$$\\mathbf{{{expr_cand} = {val_str}}}$$"
                return think, resp
            except Exception:
                pass

        return None
