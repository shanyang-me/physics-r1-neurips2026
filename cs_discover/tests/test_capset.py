"""Cap-set verifier + constructor tests. Run: python cs_discover/tests/test_capset.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cs_discover.domains.capset import (
    all_vectors, is_cap, greedy_cap, lexicographic_priority,
)


def test_line_is_rejected():
    # (0,0), (1,1), (2,2) sum to (0,0) mod 3 -> a line -> NOT a cap.
    assert not is_cap([(0, 0), (1, 1), (2, 2)])


def test_non_line_triple_is_cap():
    # (0,0), (1,0), (0,1): sum (1,1) != 0 -> a valid cap.
    assert is_cap([(0, 0), (1, 0), (0, 1)])


def test_duplicates_rejected():
    assert not is_cap([(0, 0), (0, 0)])


def test_greedy_produces_valid_cap():
    for n in (2, 3, 4):
        cap = greedy_cap(lexicographic_priority, n)
        assert is_cap(cap), f"greedy cap invalid at n={n}"


def test_known_small_optima_not_exceeded():
    # Known maximum cap sizes: n=2 -> 4, n=3 -> 9.
    KNOWN_MAX = {2: 4, 3: 9}
    for n, mx in KNOWN_MAX.items():
        cap = greedy_cap(lexicographic_priority, n)
        assert len(cap) <= mx, f"n={n}: size {len(cap)} exceeds known max {mx}"


def test_vector_space_size():
    assert len(all_vectors(3)) == 27
    assert len(all_vectors(4)) == 81


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} cap-set tests passed.")


if __name__ == "__main__":
    _run_all()
