"""Custom reward function for Physics-o1 verl training (v3: dense physics-native rewards).

Implements the five-component reward of the Physics-R1 paper:
  r = r_ans + r_fmt + r_dim + r_sym + r_cons,  clip to [-1, 1]

  r_ans  ∈ {0, +1.00}   answer correctness (binary, multi-part / MCQ / numeric tolerant)
  r_fmt  ∈ {0, +0.10}   well-formed \boxed{} answer present
  r_dim  ∈ {0, +0.15}   dimensional consistency (units detected + sympy-parseable + per-eq consistent)
  r_sym  ∈ {0, +0.20}   ≥1 intermediate \frac{...}{...} parses cleanly via sympy
  r_cons ∈ {-0.25, 0}   energy/momentum-conservation violation on flagged problems (extra_info)

Mode selection (env var):
  DENSE_REWARD=0  (default) → returns r_ans only (binary 0/1, prior behaviour preserved)
  DENSE_REWARD=1            → returns the full clipped sum

Bug fixes carried over from v2:
  - Brace-counting boxed extractor (handles unlimited nesting depth)
  - Multi-part answer matching (no false positive on shared prefixes)
  - eval()-preferred numeric parsing for expression-like strings ("1/2" → 0.5)
"""

import os
import re

# ----------------------------------------------------------------------
# Component weights (paper Section 4.1)
# ----------------------------------------------------------------------
W_ANS = 1.00
W_FMT = 0.10
W_DIM = 0.15
W_SYM = 0.20
W_CONS = 0.25  # applied as NEGATIVE only

# ============================================================================
# Boxed extraction (Bug 1 fix from v1)
# ============================================================================

def extract_boxed(text):
    """Extract content from the LAST \\boxed{...} in text via brace counting."""
    if text is None:
        return None
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(r'\boxed{', i)
        if idx < 0:
            break
        start = idx + len(r'\boxed{')
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            c = text[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    matches.append(text[start:j])
                    break
            j += 1
        i = j + 1
    if matches:
        return matches[-1].strip()
    return None


def extract_answer_letter(text):
    """Extract MCQ answer letter (A/B/C/D) from text."""
    if text is None:
        return None
    boxed = extract_boxed(text)
    if boxed and boxed.upper() in ('A', 'B', 'C', 'D'):
        return boxed.upper()
    for pat in [
        r'(?:answer|Answer)\s*(?:is|:)\s*\(?([A-D])\)?',
        r'\(([A-D])\)\s*$',
        r'\\text\{([A-D])\}',
    ]:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None


# ============================================================================
# Normalization
# ============================================================================

def normalize_answer(answer):
    if answer is None:
        return None
    answer = answer.strip().strip('$')
    m = re.match(r'\\text\{(.+)\}', answer)
    if m:
        answer = m.group(1)
    return answer.strip()


def latex_to_plain(s):
    if s is None:
        return None
    s = s.strip().strip('$')
    s = re.sub(r'\\(?:text|mathrm|mbox|mathbf|operatorname)\{([^}]*)\}', r'\1', s)
    for _ in range(3):
        new_s = re.sub(r'\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
        if new_s == s:
            break
        s = new_s
    s = s.replace(r'\times', '*').replace(r'\cdot', '*')
    s = s.replace(r'\pi', 'pi').replace(r'\%', '').replace('%', '')
    s = s.replace(r'\left', '').replace(r'\right', '')
    s = s.replace(' ', '').replace(r'\,', '').replace(r'\;', '').replace(r'\!', '')
    return s.strip()


def try_float(s):
    if s is None:
        return None
    s = latex_to_plain(s)
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    s2 = re.sub(r'\^\{([-+]?\d+)\}', r'**\1', s)
    s2 = re.sub(r'\^([-+]?\d+)', r'**\1', s2)
    if re.search(r'[/*+\-]', s2) and re.match(r'^[-+0-9./*()e\s\^*]+$', s2.replace('**', '_').replace('*', '_').replace('_', '*')):
        try:
            val = eval(s2, {"__builtins__": {}}, {})
            return float(val)
        except Exception:
            pass
    m = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def numeric_equal(pred, gt, rtol=1e-2, atol=1e-6):
    if pred is None or gt is None:
        return False
    if gt == 0:
        return abs(pred - gt) < atol
    return abs(pred - gt) / abs(gt) < rtol or abs(pred - gt) < atol


# ============================================================================
# Multi-part detection (Bug 2 fix from v1)
# ============================================================================

def split_multipart(s):
    if not s:
        return None
    parts = re.split(r'\s*\(([a-z])\)\s*', s)
    if len(parts) >= 5:
        result = []
        for i in range(1, len(parts) - 1, 2):
            if i + 1 < len(parts):
                label = parts[i]
                value = parts[i + 1].strip().rstrip(';,')
                if value:
                    result.append((label, value))
        if len(result) >= 2:
            return result
    parts = re.split(r'\b([a-z])\.\s+', s)
    if len(parts) >= 5:
        result = []
        for i in range(1, len(parts) - 1, 2):
            if i + 1 < len(parts):
                label = parts[i]
                value = parts[i + 1].strip().rstrip(';,')
                if value:
                    result.append((label, value))
        if len(result) >= 2:
            return result
    if ';' in s:
        result = [("", p.strip()) for p in s.split(';') if p.strip()]
        if len(result) >= 2:
            return result
    return None


def match_single_part(pred, gt):
    pred_n = normalize_answer(pred)
    gt_n = normalize_answer(gt)
    if pred_n is None or gt_n is None:
        return False
    if pred_n == gt_n or pred_n.lower() == gt_n.lower():
        return True
    if latex_to_plain(pred_n) == latex_to_plain(gt_n):
        return True
    pf = try_float(pred_n)
    gf = try_float(gt_n)
    if pf is not None and gf is not None:
        return numeric_equal(pf, gf)
    return False


# ============================================================================
# r_ans — answer correctness (binary)
# ============================================================================

def _base_score(solution_str, ground_truth):
    if solution_str is None or ground_truth is None:
        return 0.0
    gt_raw = str(ground_truth)
    gt = normalize_answer(gt_raw)
    if not gt:
        return 0.0
    if gt and gt.upper() in ('A', 'B', 'C', 'D'):
        predicted_letter = extract_answer_letter(solution_str)
        return 1.0 if predicted_letter == gt.upper() else 0.0
    gt_parts = split_multipart(gt)
    predicted = extract_boxed(solution_str)
    if predicted is None:
        return 0.0
    pred_norm = normalize_answer(predicted)
    if pred_norm is None or not pred_norm:
        return 0.0
    if gt_parts:
        pred_parts = split_multipart(pred_norm)
        if pred_parts is None or len(pred_parts) != len(gt_parts):
            return 0.0
        for (gl, gv), (pl, pv) in zip(gt_parts, pred_parts):
            if gl and pl and gl != pl:
                return 0.0
            if not match_single_part(pv, gv):
                return 0.0
        return 1.0
    if match_single_part(pred_norm, gt):
        return 1.0
    return 0.0


# ============================================================================
# r_fmt — format match (+0.1 if well-formed \boxed{} present)
# ============================================================================

def reward_format(solution_str):
    """+W_FMT if response contains a well-formed \\boxed{} answer, else 0."""
    if solution_str is None:
        return 0.0
    boxed = extract_boxed(solution_str)
    if boxed is None or len(boxed) == 0:
        return 0.0
    return W_FMT


# ============================================================================
# r_dim — dimensional consistency (regex unit-detector + sympy unit-system)
# ============================================================================

# Curated SI / common physics units. Single-letter units only count when
# preceded by a numeric token, to avoid false positives from variable names.
_UNIT_TOKENS_LONG = [
    'kg', 'cm', 'mm', 'km', 'nm', 'um', 'mol', 'rad', 'eV', 'keV', 'MeV', 'GeV',
    'Hz', 'kHz', 'MHz', 'GHz', 'Pa', 'kPa', 'MPa', 'atm', 'bar', 'mbar',
    'min', 'hr', 'day', 'yr', 'lb', 'ft', 'in',
    'Ohm', 'Wb', 'lm', 'lx', 'Bq', 'Gy', 'Sv',
    'ms', 'us', 'ns', 'ps',
]
_UNIT_TOKENS_SHORT = ['m', 's', 'g', 'N', 'J', 'W', 'V', 'A', 'C', 'T', 'F', 'H', 'K', 'L']

# Regex: a number (optionally scientific) followed by an optional unit token,
# possibly with exponent and composite (m/s, kg·m/s^2, etc.).
_NUM = r'\d+(?:\.\d+)?(?:\s*(?:\\times|\*)\s*10\s*\^?\s*\{?[-+]?\d+\}?)?'
_UNIT_LONG_ALT = '|'.join(re.escape(u) for u in _UNIT_TOKENS_LONG)
_UNIT_SHORT_ALT = '|'.join(_UNIT_TOKENS_SHORT)
_EXP = r'(?:\s*\^\s*\{?-?\d+\}?)?'
_COMP = r'(?:\s*[/·\\\.]\s*[a-zA-Z]+' + _EXP + ')*'

_UNIT_RE = re.compile(
    r'(?<![A-Za-z\\])'
    r'(' + _NUM + r')'
    r'\s*'
    r'(' + _UNIT_LONG_ALT + r'|' + _UNIT_SHORT_ALT + r')'
    + _EXP + _COMP +
    r'(?![A-Za-z])'
)

_UNIT_TO_SYMPY = {
    'm': 'meter', 'cm': 'centimeter', 'mm': 'millimeter', 'km': 'kilometer',
    'nm': 'nanometer', 'um': 'micrometer',
    's': 'second', 'ms': 'millisecond', 'us': 'microsecond', 'ns': 'nanosecond', 'ps': 'picosecond',
    'min': 'minute', 'hr': 'hour', 'day': 'day', 'yr': 'year',
    'kg': 'kilogram', 'g': 'gram',
    'N': 'newton', 'J': 'joule', 'W': 'watt', 'Pa': 'pascal',
    'V': 'volt', 'A': 'ampere', 'C': 'coulomb', 'T': 'tesla', 'F': 'farad', 'H': 'henry',
    'Hz': 'hertz', 'K': 'kelvin', 'mol': 'mole', 'rad': 'radian',
    'eV': 'electronvolt', 'L': 'liter',
    'Ohm': 'ohm', 'Wb': 'weber', 'lm': 'lumen', 'lx': 'lux',
}


def _sympy_unit(token):
    """Resolve a unit token to a sympy.physics.units unit, or None."""
    try:
        from sympy.physics import units as su
    except Exception:
        return None
    name = _UNIT_TO_SYMPY.get(token)
    if name is None:
        return None
    return getattr(su, name, None)


def reward_dimensional(solution_str):
    """+W_DIM if (a) at least one unit is detected and (b) all detected units
    resolve to a sympy unit (i.e., parse cleanly under the unit system), else 0.
    """
    if not solution_str:
        return 0.0
    matches = _UNIT_RE.findall(solution_str)
    if not matches:
        return 0.0
    detected_tokens = []
    for m in matches[:32]:
        # m is a tuple of capture groups; the unit is group 2 (index 1)
        if isinstance(m, tuple) and len(m) >= 2:
            detected_tokens.append(m[1])
        else:
            detected_tokens.append(str(m))
    if not detected_tokens:
        return 0.0
    for tok in detected_tokens:
        if _sympy_unit(tok) is None:
            return 0.0
    return W_DIM


# ============================================================================
# r_sym — symbolic equation verification (sympy-parseable \frac)
# ============================================================================

_FRAC_RE = re.compile(r'\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}')

_SAFE_SYMBOLS = re.compile(r'^[A-Za-z0-9_\^\*\+\-\/\.\(\)\s]+$')


def _sympy_parse(expr):
    """Try to sympify a LaTeX-cleaned expression. Return True if parses."""
    try:
        from sympy import sympify
    except Exception:
        return False
    e = expr.strip()
    e = e.replace(r'\,', '').replace(r'\;', '').replace(r'\!', '')
    e = re.sub(r'\\(?:text|mathrm|mbox|operatorname)\{([^}]*)\}', r'\1', e)
    e = re.sub(r'\^\{?([-+]?\d+)\}?', r'**\1', e)
    e = e.replace('\\cdot', '*').replace('\\times', '*')
    e = e.replace('\\pi', 'pi')
    e = re.sub(r'\\([a-zA-Z]+)', r'\1', e)  # strip remaining \cmd → cmd
    if not _SAFE_SYMBOLS.match(e.replace(' ', '')):
        return False
    try:
        sympify(e, evaluate=False)
        return True
    except Exception:
        return False


def reward_symbolic(solution_str):
    """+W_SYM if at least one intermediate \\frac{A}{B} where A and B both
    parse cleanly via sympy, else 0.
    """
    if not solution_str:
        return 0.0
    fracs = _FRAC_RE.findall(solution_str)
    if not fracs:
        return 0.0
    for num, den in fracs[:16]:
        if _sympy_parse(num) and _sympy_parse(den):
            return W_SYM
    return 0.0


# ============================================================================
# r_cons — conservation-law conformance (penalty only)
# ============================================================================

def reward_conservation(solution_str, extra_info):
    """-W_CONS if the problem is flagged for energy/momentum conservation
    (extra_info['conservation']) AND the final \\boxed{} answer violates the
    conservation balance by more than `tol` (default 5%), else 0.

    extra_info['conservation'] schema (optional):
      {
        'type':   'energy' | 'momentum',
        'in':     [v1, v2, ...]  # numeric input total or pieces
        'out':    [v3, v4, ...]  # expected total or pieces
        'answer_index': int       # which value (in or out) the gold answer corresponds to
        'tol':    float           # relative tolerance, default 0.05
      }
    Absent or malformed → returns 0 (no penalty).
    """
    if not isinstance(extra_info, dict):
        return 0.0
    cons = extra_info.get('conservation')
    if not isinstance(cons, dict):
        return 0.0

    pred = extract_boxed(solution_str)
    pred_val = try_float(pred) if pred else None
    if pred_val is None:
        return 0.0

    in_vals = cons.get('in') or []
    out_vals = cons.get('out') or []
    if not in_vals or not out_vals:
        return 0.0
    try:
        sum_in = float(sum(in_vals))
        sum_out_other = float(sum(v for i, v in enumerate(out_vals)
                                   if i != cons.get('answer_index', -1)))
    except (TypeError, ValueError):
        return 0.0

    expected = sum_in - sum_out_other
    tol = float(cons.get('tol', 0.05))
    denom = max(abs(expected), abs(pred_val), 1e-9)
    rel_err = abs(pred_val - expected) / denom
    if rel_err > tol:
        return -W_CONS
    return 0.0


# ============================================================================
# Main entry — compute_score
# ============================================================================

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Verl reward-fn entry point.

    Modes (DENSE_REWARD env var):
      0 (default) — return r_ans (binary 0/1), preserves prior runs
      1           — return clip(r_ans + r_fmt + r_dim + r_sym + r_cons, -1, 1)

    Audit factor (AUDIT_LAMBDA env var) is applied multiplicatively to r_ans only.
    """
    base = _base_score(solution_str, ground_truth)

    # Audit-conditioned r_ans (zeros out reward on contaminated items)
    audit_lambda = float(os.environ.get("AUDIT_LAMBDA", "0.0"))
    if audit_lambda > 0.0 and base > 0.0:
        contamination = 0.0
        if isinstance(extra_info, dict):
            contamination = float(extra_info.get("contamination_score", 0.0))
        base = base * max(0.0, 1.0 - audit_lambda * contamination)

    if int(os.environ.get("DENSE_REWARD", "0")) == 0:
        return base

    r_fmt = reward_format(solution_str)
    r_dim = reward_dimensional(solution_str)
    r_sym = reward_symbolic(solution_str)
    r_cons = reward_conservation(solution_str, extra_info)
    total = base + r_fmt + r_dim + r_sym + r_cons
    return max(-1.0, min(1.0, total))


def compute_correctness(solution_str, ground_truth):
    """Strict 0/1 correctness for evaluation scripts that need binary accuracy."""
    return _base_score(solution_str, ground_truth)
