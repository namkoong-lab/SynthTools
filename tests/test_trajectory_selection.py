"""Unit tests for deterministic random task selection used by the HPC
trajectory jobs (50 independent nodes, one seed each)."""

from trajectory_generation.loader import select_random_ids


IDS = [f"task_{i:04d}" for i in range(200)]


def test_deterministic_for_same_seed():
    a = select_random_ids(IDS, seed=7)
    b = select_random_ids(IDS, seed=7)
    assert a == b


def test_is_a_permutation_when_uncapped():
    out = select_random_ids(IDS, seed=3)
    assert sorted(out) == sorted(IDS)
    assert len(out) == len(IDS)


def test_actually_shuffles():
    out = select_random_ids(IDS, seed=3)
    assert out != IDS  # vanishingly unlikely to be identity for 200 items


def test_distinct_seeds_give_distinct_orders():
    a = select_random_ids(IDS, seed=1)
    b = select_random_ids(IDS, seed=2)
    assert a != b


def test_sample_caps_length():
    out = select_random_ids(IDS, seed=5, sample=10)
    assert len(out) == 10
    # the capped draw is the prefix of the full permutation
    assert out == select_random_ids(IDS, seed=5)[:10]


def test_sample_larger_than_input_returns_full_permutation():
    out = select_random_ids(IDS, seed=5, sample=10_000)
    assert sorted(out) == sorted(IDS)


def test_sample_zero_or_negative_returns_empty():
    assert select_random_ids(IDS, seed=5, sample=0) == []
    assert select_random_ids(IDS, seed=5, sample=-4) == []


def test_overlap_between_two_seeds_is_partial():
    """Two seeds drawing the same sample size overlap on some tasks (so a
    task can get multiple rollouts) but are not identical sets."""
    a = set(select_random_ids(IDS, seed=1, sample=100))
    b = set(select_random_ids(IDS, seed=2, sample=100))
    assert a & b           # some overlap -> multiplicity
    assert a != b          # not the same set -> coverage spreads
