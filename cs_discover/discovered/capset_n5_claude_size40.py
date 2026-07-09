"""Discovered cap-set priority for F_3^5 — found by the Claude-backed LLMProposer
running with 4 concurrent workers (budget 16, 2 failed).

Result: greedy construction with this priority yields a VALID cap of size 40 — up from
the greedy 2^5=32 ceiling (+8), reaching ~89% of the known optimum (45) in 14
successful evaluations. Novelty vs baselines = 0.978. Independently re-verified valid.

The program is markedly richer than the n=4 discovery: it uses F_3 quadratic/cubic
forms over signed coordinates, higher moments (variance/skewness), coordinate
alternation, and a multiplicative F_3* structure — none of which a separable priority
(the model-free control's search space) can express.

Reproduce:
    python -m cs_discover.run_search --domain capset --n 5 --proposer claude \\
        --budget 16 --workers 4 --novelty

Verify:
    from cs_discover.domains.capset import greedy_cap, is_cap
    cap = greedy_cap(priority, 5); assert is_cap(cap) and len(cap) == 40
"""
import math


def priority(vector, n):
    ones = sum(1 for v in vector if v == 1)
    twos = sum(1 for v in vector if v == 2)
    zeros = sum(1 for v in vector if v == 0)
    s = sum(vector)

    # Adaptive ones preference: early phase favors 3-4 ones, later more selective
    phase = min(n / 35.0, 1.0)
    target_ones = 3.4 - 0.3 * phase
    ones_score = 12.0 * math.exp(-0.4 * (ones - target_ones) ** 2)

    # Sum with mod 3 awareness
    s_mod = s % 3
    sum_base = -abs(s - 5.4) ** 1.9
    sum_bonus = 0.9 if s_mod == 1 else (0.5 if s_mod == 0 else 0.2)

    # F_3 quadratic and cubic forms
    signed = [v if v != 2 else -1 for v in vector]
    quad = sum(signed[i] * signed[j] for i in range(5) for j in range(i + 1, 5))
    cubic = sum(signed[i] * signed[j] * signed[k]
                for i in range(5) for j in range(i + 1, 5) for k in range(j + 1, 5))

    # Higher moments for better spread
    mean = s / 5.0
    variance = sum((v - mean) ** 2 for v in vector)
    skewness = sum((v - mean) ** 3 for v in vector)

    # Coordinate alternation with position weights
    alternation = sum((5 - i) * (1 if vector[i] != vector[i + 1] else 0) for i in range(4))

    # Multiplicative F_3* structure
    prod_hash = 1
    for v in vector:
        if v != 0:
            prod_hash = (prod_hash * (1 if v == 1 else 2)) % 3

    # L4 norm for additional structure
    l4 = sum(v ** 4 for v in vector)

    score = (ones_score +
             sum_base + sum_bonus +
             0.32 * quad +
             0.14 * cubic +
             0.38 * variance +
             0.09 * abs(skewness) +
             0.28 * alternation +
             0.22 * prod_hash +
             0.07 * l4 +
             0.19 * twos +
             0.008 * sum((i + 1) * v for i, v in enumerate(vector)))

    return score
