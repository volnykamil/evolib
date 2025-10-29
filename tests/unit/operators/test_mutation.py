import numpy as np
import pytest

from evolib.core.genotype import (
    BinaryGenotype,
    Genotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)
from evolib.operators.mutation import (
    BitFlipMutation,
    CreepIntegerMutation,
    GaussianMutation,
    InsertMutation,
    InversionMutation,
    NonUniformIntegerMutation,
    NonUniformMutation,
    ScrambleMutation,
    SwapMutation,
    UniformIntegerMutation,
    UniformMutation,
)


def assert_change(mutated_genotype: Genotype, original_genotype: Genotype, original_genes: np.ndarray):
    """Helper function to assert that a change has occurred."""
    assert len(mutated_genotype) == len(original_genotype)
    num_changed = np.sum(mutated_genotype.genes != original_genes)
    assert num_changed > 0


def test_bit_flip_mutation():
    genotype = BinaryGenotype.random(length=100)
    mutation_operator = BitFlipMutation(probability=0.5)
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert mutated_genotype.genes.dtype == np.bool_


def test_bit_flip_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = BitFlipMutation(probability=0.5)
    with pytest.raises(TypeError, match=r"BitFlipMutation is only applicable to BinaryGenotype."):
        mutation_operator.mutate(genotype)


def test_gaussian_mutation():
    genotype = RealGenotype.random(length=100, bounds=(0.0, 1.0))
    mutation_operator = GaussianMutation(sigma=0.1, probability=0.5)
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert mutated_genotype.genes.dtype in (np.float32, np.float64)


def test_gaussian_mutation_fail():
    genotype = BinaryGenotype.random(length=10)
    mutation_operator = GaussianMutation(sigma=0.1, probability=0.5)
    with pytest.raises(TypeError, match=r"GaussianMutation is only applicable to RealGenotype."):
        mutation_operator.mutate(genotype)


def test_uniform_mutation():
    genotype = RealGenotype.random(length=100, bounds=(0.0, 1.0))
    mutation_operator = UniformMutation(probability=0.5)
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert mutated_genotype.genes.dtype in (np.float32, np.float64)


def test_uniform_mutation_fail():
    genotype = BinaryGenotype.random(length=10)
    mutation_operator = UniformMutation(probability=0.5)
    with pytest.raises(TypeError, match=r"UniformMutation is only applicable to RealGenotype."):
        mutation_operator.mutate(genotype)


def test_non_uniform_mutation():
    genotype = RealGenotype.random(length=100, bounds=(0.0, 1.0))
    mutation_operator = NonUniformMutation(progress=0.5, probability=0.5)
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert mutated_genotype.genes.dtype in (np.float32, np.float64)


def test_non_uniform_mutation_fail():
    genotype = BinaryGenotype.random(length=10)
    mutation_operator = NonUniformMutation(progress=0.5, probability=0.5)
    with pytest.raises(TypeError, match=r"NonUniformMutation is only applicable to RealGenotype."):
        mutation_operator.mutate(genotype)


def test_uniform_integer_mutation():
    genotype = IntegerGenotype.random(length=100, bounds=(0, 100))
    mutation_operator = UniformIntegerMutation(probability=0.5)
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert np.issubdtype(mutated_genotype.genes.dtype, np.integer)


def test_uniform_integer_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = UniformIntegerMutation(probability=0.5)
    with pytest.raises(TypeError, match=r"UniformIntegerMutation is only applicable to IntegerGenotype."):
        mutation_operator.mutate(genotype)


def test_creep_integer_mutation():
    genotype = IntegerGenotype.random(length=100, bounds=(0, 100))
    mutation_operator = CreepIntegerMutation(probability=0.5)
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert np.issubdtype(mutated_genotype.genes.dtype, np.integer)


def test_creep_integer_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = CreepIntegerMutation(probability=0.5)
    with pytest.raises(TypeError, match=r"CreepIntegerMutation is only applicable to IntegerGenotype."):
        mutation_operator.mutate(genotype)


def test_non_uniform_integer_mutation():
    genotype = IntegerGenotype.random(length=100, bounds=(0, 100))
    mutation_operator = NonUniformIntegerMutation(progress=0.5, probability=0.5)
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert np.issubdtype(mutated_genotype.genes.dtype, np.integer)


def test_non_uniform_integer_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = NonUniformIntegerMutation(progress=0.5, probability=0.5)
    with pytest.raises(TypeError, match=r"NonUniformIntegerMutation is only applicable to IntegerGenotype."):
        mutation_operator.mutate(genotype)


def test_swap_mutation():
    genotype = PermutationGenotype.random(length=100)
    mutation_operator = SwapMutation()
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert np.issubdtype(mutated_genotype.genes.dtype, np.integer)


def test_swap_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = SwapMutation()
    with pytest.raises(TypeError, match=r"SwapMutation is only applicable to PermutationGenotype."):
        mutation_operator.mutate(genotype)


def test_insert_mutation():
    genotype = PermutationGenotype.random(length=100)
    mutation_operator = InsertMutation()
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert np.issubdtype(mutated_genotype.genes.dtype, np.integer)


def test_insert_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = InsertMutation()
    with pytest.raises(TypeError, match=r"InsertMutation is only applicable to PermutationGenotype."):
        mutation_operator.mutate(genotype)


def test_scramble_mutation():
    genotype = PermutationGenotype.random(length=100)
    mutation_operator = ScrambleMutation()
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert np.issubdtype(mutated_genotype.genes.dtype, np.integer)


def test_scramble_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = ScrambleMutation()
    with pytest.raises(TypeError, match=r"ScrambleMutation is only applicable to PermutationGenotype."):
        mutation_operator.mutate(genotype)


def test_inversion_mutation():
    genotype = PermutationGenotype.random(length=100)
    mutation_operator = InversionMutation()
    mutated_genotype = mutation_operator.mutate(genotype)
    assert_change(mutated_genotype, genotype, genotype.genes.copy())
    assert np.issubdtype(mutated_genotype.genes.dtype, np.integer)


def test_inversion_mutation_fail():
    genotype = RealGenotype.random(length=10)
    mutation_operator = InversionMutation()
    with pytest.raises(TypeError, match=r"InversionMutation is only applicable to PermutationGenotype."):
        mutation_operator.mutate(genotype)
