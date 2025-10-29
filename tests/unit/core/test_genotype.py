import numpy as np
import pytest

from evolib.core.genotype import (
    BinaryGenotype,
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
