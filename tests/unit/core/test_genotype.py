import numpy as np
import pytest

from evolib.core.genotype import (
    BinaryGenotype,
    HybridGenotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)


def test_binary_genotype():
    genlen = 10
    genotype = BinaryGenotype.random(length=genlen)
    assert len(genotype) == genlen
    assert genotype.genes.dtype == np.bool_
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (genlen,)
    assert str(genotype) == f"BinaryGenotype(shape=({genlen},))"


def test_binary_genotype_fail():
    with pytest.raises(TypeError, match="BinaryGenotype genes must be boolean"):
        BinaryGenotype(np.array([2, 3, 4]))


def test_binary_genotype_subtraction():
    g1 = BinaryGenotype(np.array([True, False, True, True]))
    g2 = BinaryGenotype(np.array([False, False, True, False]))
    diff = g1 - g2
    # Convert bool to int8: True->1 False->0
    assert np.array_equal(diff, np.array([1, 0, 0, 1], dtype=np.int8))


def test_binary_genotype_subtraction_fail():
    g1 = BinaryGenotype.random(5)
    with pytest.raises(TypeError):
        _ = g1 - RealGenotype.random(5)  # type: ignore[arg-type]


def test_binary_genotype_random_extremes():
    all_zero = BinaryGenotype.random(length=10, p=0.0)
    # All genes should be False
    assert not np.any(all_zero.genes)
    all_one = BinaryGenotype.random(length=10, p=1.0)
    # All genes should be True
    assert np.all(all_one.genes)


def test_zero_length_genotypes():
    b = BinaryGenotype.random(0)
    r = RealGenotype.random(0, bounds=(0.0, 1.0))
    assert len(b) == 0 and b.as_array().shape == (0,)
    assert len(r) == 0 and r.as_array().shape == (0,)


def test_real_genotype():
    genlen = 5
    low = 0.0
    high = 10.0
    genotype = RealGenotype.random(length=genlen, bounds=(low, high))
    assert len(genotype) == genlen
    assert genotype.genes.dtype in (np.float32, np.float64)
    assert np.all(genotype.genes >= low) and np.all(genotype.genes <= high)
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (genlen,)
    assert str(genotype) == f"RealGenotype(shape=({genlen},))"


def test_real_genotype_fail():
    with pytest.raises(TypeError, match="RealGenotype genes must be float32/float64"):
        RealGenotype(np.array([1, 2, 3], dtype=np.int32))


def test_real_genotype_subtraction():
    g1 = RealGenotype(np.array([1.0, 2.0, 3.0], dtype=np.float64), bounds=(0.0, 5.0))
    g2 = RealGenotype(np.array([0.5, 1.5, 2.5], dtype=np.float64), bounds=(0.0, 5.0))
    diff = g1 - g2
    assert np.allclose(diff, np.array([0.5, 0.5, 0.5]))


def test_real_genotype_subtraction_fail():
    g1 = RealGenotype.random(4, bounds=(0.0, 1.0))
    with pytest.raises(TypeError):
        _ = g1 - IntegerGenotype.random(4)  # type: ignore[arg-type]


def test_real_genotype_invalid_bounds():
    genes = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="Invalid bounds"):
        RealGenotype(genes, bounds=(1.0, 1.0))
    with pytest.raises(ValueError, match="Invalid bounds"):
        RealGenotype(genes, bounds=(5.0, 2.0))


def test_integer_genotype():
    genlen = 7
    low = 1
    high = 100
    genotype = IntegerGenotype.random(length=genlen, bounds=(low, high))
    assert len(genotype) == genlen
    assert np.issubdtype(genotype.genes.dtype, np.integer)
    assert np.all(genotype.genes >= low) and np.all(genotype.genes <= high)
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (genlen,)
    assert str(genotype) == f"IntegerGenotype(shape=({genlen},))"


def test_integer_genotype_fail():
    with pytest.raises(TypeError, match="IntegerGenotype genes must be integer dtype"):
        IntegerGenotype(np.array([1.5, 2.5, 3.5], dtype=np.float32))


def test_integer_genotype_subtraction():
    g1 = IntegerGenotype(np.array([10, 20, 30], dtype=np.int32), bounds=(0, 50))
    g2 = IntegerGenotype(np.array([1, 5, 10], dtype=np.int32), bounds=(0, 50))
    diff = g1 - g2
    assert np.array_equal(diff, np.array([9, 15, 20]))


def test_integer_genotype_subtraction_fail():
    g1 = IntegerGenotype.random(5, bounds=(0, 10))
    with pytest.raises(TypeError):
        _ = g1 - RealGenotype.random(5)  # type: ignore[arg-type]


def test_integer_genotype_invalid_bounds():
    genes = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="Invalid bounds"):
        IntegerGenotype(genes, bounds=(5, 2))


def test_permutation_genotype():
    genlen = 6
    genotype = PermutationGenotype.random(length=genlen)
    assert len(genotype) == genlen
    assert set(genotype.genes) == set(range(genlen))
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (genlen,)
    assert str(genotype) == f"PermutationGenotype(shape=({genlen},))"


def test_permutation_genotype_fail():
    with pytest.raises(TypeError, match="PermutationGenotype genes must be integer dtype"):
        PermutationGenotype(np.array([0.0, 2.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match=r"Genes must be a permutation of integers from 0 to len\(genes\)-1\."):
        PermutationGenotype(np.array([0, 2, 2, 3]))


def test_permutation_genotype_subtraction():
    g1 = PermutationGenotype(np.array([0, 2, 1, 3], dtype=np.int32))
    g2 = PermutationGenotype(np.array([1, 0, 2, 3], dtype=np.int32))
    diff = g1 - g2
    assert np.array_equal(diff, np.array([-1, 2, -1, 0]))


def test_permutation_genotype_subtraction_fail():
    g1 = PermutationGenotype.random(5)
    with pytest.raises(TypeError):
        _ = g1 - IntegerGenotype.random(5)  # type: ignore[arg-type]


def test_hybrid_genotype():
    components = {
        "comp1": BinaryGenotype.random(length=4),
        "comp2": RealGenotype.random(length=3, bounds=(-1.0, 1.0)),
        "comp3": IntegerGenotype.random(length=5, bounds=(0, 10)),
    }
    genotype = HybridGenotype(components=components)
    assert set(genotype.components.keys()) == set(components.keys())
    for key, comp in components.items():
        assert np.array_equal(genotype.components[key].genes, comp.genes)
    genotype_copy = genotype.copy()
    for key, _ in components.items():
        assert np.array_equal(genotype.components[key].genes, genotype_copy.components[key].genes)
    assert str(genotype) == "HybridGenotype(comp1:BinaryGenotype, comp2:RealGenotype, comp3:IntegerGenotype)"


def test_hybrid_genotype_fail():
    with pytest.raises(TypeError, match="Components must be a dictionary of Genotype instances"):
        HybridGenotype(components={"comp1": BinaryGenotype.random(4), "comp2": "not_a_genotype"})


def test_hybrid_genotype_length_and_array():
    comp1 = BinaryGenotype.random(3)
    comp2 = RealGenotype.random(2, bounds=(-1.0, 1.0))
    comp3 = IntegerGenotype.random(4, bounds=(0, 9))
    hg = HybridGenotype({"b": comp1, "r": comp2, "i": comp3})
    assert len(hg) == len(comp1) + len(comp2) + len(comp3)
    concat = hg.as_array()
    expected = np.concatenate([comp1.as_array().ravel(), comp2.as_array().ravel(), comp3.as_array().ravel()])
    assert np.array_equal(concat, expected)


def test_hybrid_genotype_subtraction():
    comp1a = BinaryGenotype(np.array([True, False]))
    comp1b = BinaryGenotype(np.array([False, True]))
    comp2a = RealGenotype(np.array([1.0, 2.0], dtype=np.float64), bounds=(0.0, 5.0))
    comp2b = RealGenotype(np.array([0.5, 1.5], dtype=np.float64), bounds=(0.0, 5.0))
    h1 = HybridGenotype({"bin": comp1a, "real": comp2a})
    h2 = HybridGenotype({"bin": comp1b, "real": comp2b})
    diff = h1 - h2
    # Binary diff first (int8), then real diff
    expected = np.concatenate(
        [
            comp1a.as_array().astype(np.int8) - comp1b.as_array().astype(np.int8),
            comp2a.as_array() - comp2b.as_array(),
        ]
    )
    assert np.array_equal(diff, expected)


def test_hybrid_genotype_subtraction_key_mismatch_fail():
    h1 = HybridGenotype({"a": BinaryGenotype.random(2), "b": RealGenotype.random(2)})
    h2 = HybridGenotype({"a": BinaryGenotype.random(2), "c": RealGenotype.random(2)})
    with pytest.raises(ValueError):
        _ = h1 - h2


def test_genotype_type_inequality():
    # Same numeric values but different genotype classes should not be equal
    real = RealGenotype(np.array([1.0, 2.0, 3.0], dtype=np.float64), bounds=(0.0, 5.0))
    integer = IntegerGenotype(np.array([1, 2, 3], dtype=np.int32), bounds=(0, 5))
    binary_like = BinaryGenotype(np.array([True, True, False]))
    perm = PermutationGenotype(np.array([0, 2, 1], dtype=np.int32))
    assert real != integer
    assert integer != perm
    assert binary_like != real
    assert perm != binary_like
