"""Discovered cap-set priority for F_3^4 — found by the Claude-backed LLMProposer.

Result: greedy construction with this priority yields a VALID cap of size 20 — the
proven maximum for n=4 — found in 10 successful evaluations (budget 12, 2 failed).
This breaks the 2^n=16 ceiling that the model-free MutationProposer provably cannot
exceed (any separable priority saturates at 2^n). The win comes from the NON-separable
terms below (min digit count, variance, pairwise modular interactions).

Reproduce:
    python -m cs_discover.run_search --domain capset --n 4 --proposer claude --budget 12

Verify this specific program:
    from cs_discover.domains.capset import greedy_cap, is_cap
    cap = greedy_cap(priority, 4); assert is_cap(cap) and len(cap) == 20
"""


def priority(vector, n):
    # Count each digit
    c0 = sum(1 for v in vector if v == 0)
    c1 = sum(1 for v in vector if v == 1)
    c2 = sum(1 for v in vector if v == 2)

    # STRONGLY prefer vectors containing all three digits {0,1,2}
    # These are geometrically more diverse and less likely to complete lines
    min_count = min(c0, c1, c2)

    # Variance (prefer non-uniform distributions)
    s = sum(vector)
    variance = sum((v - s / 4.0) ** 2 for v in vector)

    # Prefer moderate sums (4-6 range)
    sum_score = -abs(s - 5)

    # Non-linear position weight for tie-breaking
    pos_score = sum(vector[i] * ((i + 1) ** 1.5) for i in range(4))

    # Pairwise modular interactions
    mod_interact = sum(((vector[i] + vector[j]) % 3) * (i + j + 1)
                       for i in range(4)
                       for j in range(i + 1, 4))

    # Adaptive: later in the search, emphasize diversity more
    diversity_weight = 20 + n * 0.3

    return (min_count * diversity_weight + variance * 12 + sum_score * 4
            + pos_score * 0.03 + mod_interact * 0.15)
