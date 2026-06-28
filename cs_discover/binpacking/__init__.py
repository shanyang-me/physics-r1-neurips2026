"""Online bin-packing discovery domain (the CS-Discover MVP task family).

The discovery artifact is a `priority(item, remaining, capacity)` function returning a
score per open bin. The verifier (verifier.py) performs the actual placement and
enforces feasibility, so the candidate program can influence *choice* but never
produce an infeasible packing.
"""
