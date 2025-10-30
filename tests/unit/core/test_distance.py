import numpy as np

from evolib.core.distance import genotype_distance, normalized_genotype_distance
from evolib.core.genotype import (
    BinaryGenotype,
    HybridGenotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)


def test_binary_distance_and_normalization():
    a = BinaryGenotype.random(20, p=0.0)
    b = BinaryGenotype.random(20, p=1.0)
    raw = genotype_distance(a, b)
    norm = normalized_genotype_distance(a, b)
    assert raw == 20
    assert np.isclose(norm, 1.0)


def test_real_distance_and_normalization_bounds():
    a = RealGenotype(np.zeros(5, dtype=np.float64), bounds=(0.0, 2.0))
    b = RealGenotype(np.full(5, 2.0, dtype=np.float64), bounds=(0.0, 2.0))
    raw = genotype_distance(a, b)
    # Max L2 distance given bounds span=2 is sqrt(5)*2
    max_l2 = np.sqrt(5) * 2
    assert np.isclose(raw, max_l2)
    norm = normalized_genotype_distance(a, b)
    assert np.isclose(norm, 1.0)


def test_integer_distance_normalization():
    a = IntegerGenotype(np.array([0, 0, 0, 0], dtype=np.int32), bounds=(0, 9))
    b = IntegerGenotype(np.array([9, 9, 9, 9], dtype=np.int32), bounds=(0, 9))
    norm = normalized_genotype_distance(a, b)
    assert 0.99 <= norm <= 1.0  # allow floating tolerance


def test_permutation_distance_normalization():
    a = PermutationGenotype(np.array([0, 1, 2, 3], dtype=np.int32))
    b = PermutationGenotype(np.array([3, 2, 1, 0], dtype=np.int32))
    raw = genotype_distance(a, b)
    norm = normalized_genotype_distance(a, b)
    assert raw == 4  # all mismatched positions
    assert np.isclose(norm, 1.0)


def test_hybrid_distance_weighting():
    comp1_a = BinaryGenotype.random(10)
    comp1_b = BinaryGenotype.random(10)
    comp2_a = RealGenotype.random(5)
    comp2_b = RealGenotype.random(5)
    h1 = HybridGenotype({"bin": comp1_a, "real": comp2_a})
    h2 = HybridGenotype({"bin": comp1_b, "real": comp2_b})
    norm = normalized_genotype_distance(h1, h2)
    assert 0.0 <= norm <= 1.0
