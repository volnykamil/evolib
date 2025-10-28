import pytest
from evolib.core.genotype import (
    BinaryGenotype,
    RealGenotype,
    IntegerGenotype,
    PermutationGenotype,
)
import numpy as np

def test_binary_genotype():
    genotype = BinaryGenotype.random(length=10)
    assert len(genotype) == 10
    assert genotype.genes.dtype == np.bool_
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (10,)
    assert str(genotype) == "BinaryGenotype(shape=(10,))"

def test_binary_genotype_fail():
    with pytest.raises(TypeError, match="BinaryGenotype genes must be boolean"):
        BinaryGenotype(np.array([2, 3, 4]))

def test_real_genotype():
    genotype = RealGenotype.random(length=5, bounds=(0.0, 10.0))
    assert len(genotype) == 5
    assert genotype.genes.dtype in (np.float32, np.float64)
    assert np.all(genotype.genes >= 0.0) and np.all(genotype.genes <= 10.0)
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (5,)
    assert str(genotype) == "RealGenotype(shape=(5,))"

def test_real_genotype_fail():
    with pytest.raises(TypeError, match="RealGenotype genes must be float32/float64"):
        RealGenotype(np.array([1, 2, 3], dtype=np.int32))

def test_integer_genotype():
    genotype = IntegerGenotype.random(length=7, bounds=(1, 100))
    assert len(genotype) == 7
    assert np.issubdtype(genotype.genes.dtype, np.integer)
    assert np.all(genotype.genes >= 1) and np.all(genotype.genes <= 100)
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (7,)
    assert str(genotype) == "IntegerGenotype(shape=(7,))"

def test_integer_genotype_fail():
    with pytest.raises(TypeError, match="IntegerGenotype genes must be integer dtype"):
        IntegerGenotype(np.array([1.5, 2.5, 3.5], dtype=np.float32))

def test_permutation_genotype():
    genotype = PermutationGenotype.random(length=6)
    assert len(genotype) == 6
    assert set(genotype.genes) == set(range(6))
    genotype_copy = genotype.copy()
    assert np.array_equal(genotype.genes, genotype_copy.genes)
    assert genotype.as_array().shape == (6,)
    assert str(genotype) == "PermutationGenotype(shape=(6,))"

def test_permutation_genotype_fail():
    with pytest.raises(TypeError, match="PermutationGenotype genes must be integer dtype"):
        PermutationGenotype(np.array([0.0, 2.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="Genes must be a permutation of integers from 0 to len\\(genes\\)-1."):
        PermutationGenotype(np.array([0, 2, 2, 3]))
