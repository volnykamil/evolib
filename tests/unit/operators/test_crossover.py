"""
Unit tests for evolib.operators.crossover
"""

import numpy as np
import pytest

from evolib.core.genotype import (
    BinaryGenotype,
    HybridGenotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)
from evolib.operators.crossover import (
    ArithmeticCrossover,
    BlendCrossover,
    CycleCrossover,
    EdgeRecombinationCrossover,
    HybridCrossover,
    OnePointCrossover,
    OrderCrossover,
    ParallelHybridCrossover,
    PartiallyMappedCrossover,
    SimulatedBinaryCrossover,
    TwoPointCrossover,
    UniformCrossover,
)


# -----------------------------------------------------------------------------
# Helper factories
# -----------------------------------------------------------------------------
def make_binary_genotype(n=10):
    return BinaryGenotype.random(n)


def make_integer_genotype(n=10, low=0, high=10):
    return IntegerGenotype.random(n, (low, high))


def make_real_genotype(n=10, low=-1.0, high=1.0):
    return RealGenotype.random(n, (low, high))


def make_permutation_genotype(n=10):
    return PermutationGenotype.random(n)


# -----------------------------------------------------------------------------
# Generic checks
# -----------------------------------------------------------------------------
def check_offspring_type_and_shape(p1, p2, c1, c2):
    assert isinstance(c1, type(p1))
    assert isinstance(c2, type(p1))
    assert len(c1.genes) == len(p1.genes)
    assert len(c2.genes) == len(p2.genes)


# =============================================================================
# Binary / Integer / Real crossovers
# =============================================================================


@pytest.mark.parametrize("cls", [OnePointCrossover, TwoPointCrossover, UniformCrossover])
def test_basic_vector_crossovers(cls):
    for maker in [make_binary_genotype, make_integer_genotype, make_real_genotype]:
        p1, p2 = maker(), maker()
        cx = cls()
        c1, c2 = cx.crossover(p1, p2)
        check_offspring_type_and_shape(p1, p2, c1, c2)


# =============================================================================
# Real-valued crossovers
# =============================================================================


def test_arithmetic_crossover():
    p1, p2 = make_real_genotype(), make_real_genotype()
    cx = ArithmeticCrossover(alpha=0.5)
    c1, c2 = cx.crossover(p1, p2)
    check_offspring_type_and_shape(p1, p2, c1, c2)
    low, high = p1.bounds
    assert np.all((c1.genes >= low) & (c1.genes <= high))
    assert np.all((c2.genes >= low) & (c2.genes <= high))


def test_blend_crossover():
    p1, p2 = make_real_genotype(), make_real_genotype()
    cx = BlendCrossover(alpha=0.5)
    c1, c2 = cx.crossover(p1, p2)
    check_offspring_type_and_shape(p1, p2, c1, c2)


def test_simulated_binary_crossover():
    p1, p2 = make_real_genotype(), make_real_genotype()
    cx = SimulatedBinaryCrossover(eta=10, probability=1.0)
    c1, c2 = cx.crossover(p1, p2)
    check_offspring_type_and_shape(p1, p2, c1, c2)


# =============================================================================
# Permutation crossovers
# =============================================================================


@pytest.mark.parametrize("cls", [OrderCrossover, PartiallyMappedCrossover, CycleCrossover, EdgeRecombinationCrossover])
def test_permutation_crossovers(cls):
    p1, p2 = make_permutation_genotype(), make_permutation_genotype()
    cx = cls()
    c1, c2 = cx.crossover(p1, p2)
    check_offspring_type_and_shape(p1, p2, c1, c2)
    # offspring must be valid permutations
    for child in [c1, c2]:
        assert sorted(child.genes) == list(range(len(p1.genes)))
        assert len(np.unique(child.genes)) == len(p1.genes)


# =============================================================================
# Type safety
# =============================================================================


def test_invalid_type_combination_raises():
    p1, p2 = make_binary_genotype(), make_real_genotype()
    cx = OnePointCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_one_point_crossover_wrong_type():
    p1, p2 = make_integer_genotype(), make_real_genotype()
    cx = OnePointCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_two_point_crossover_wrong_type():
    p1, p2 = make_integer_genotype(), make_binary_genotype()
    cx = TwoPointCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_uniform_crossover_wrong_type():
    p1, p2 = make_real_genotype(), make_integer_genotype()
    cx = UniformCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_arithmetic_crossover_wrong_type():
    p1, p2 = make_integer_genotype(), make_integer_genotype()
    cx = ArithmeticCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_blend_crossover_wrong_type():
    p1, p2 = make_integer_genotype(), make_integer_genotype()
    cx = BlendCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_simulated_binary_crossover_wrong_type():
    p1, p2 = make_integer_genotype(), make_integer_genotype()
    cx = SimulatedBinaryCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_order_crossover_wrong_type():
    p1, p2 = make_permutation_genotype(), make_real_genotype()
    cx = OrderCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_partially_mapped_crossover_wrong_type():
    p1, p2 = make_permutation_genotype(), make_integer_genotype()
    cx = PartiallyMappedCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_cycle_crossover_wrong_type():
    p1, p2 = make_permutation_genotype(), make_binary_genotype()
    cx = CycleCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_edge_recombination_crossover_wrong_type():
    p1, p2 = make_permutation_genotype(), make_real_genotype()
    cx = EdgeRecombinationCrossover()
    with pytest.raises(TypeError):
        cx.crossover(p1, p2)


def test_hybrid_crossover():
    parent1 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    parent2 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    crossover_operator = HybridCrossover(
        operators={
            "comp1": BlendCrossover(alpha=0.5),
            "comp2": OrderCrossover(),
        }
    )
    child1, child2 = crossover_operator.crossover(parent1, parent2)
    assert isinstance(child1, HybridGenotype)
    assert isinstance(child2, HybridGenotype)
    assert "comp1" in child1.components
    assert "comp2" in child1.components
    assert "comp1" in child2.components
    assert "comp2" in child2.components
    # Check types of components
    assert isinstance(child1.components["comp1"], RealGenotype)
    assert isinstance(child1.components["comp2"], PermutationGenotype)
    assert isinstance(child2.components["comp1"], RealGenotype)
    assert isinstance(child2.components["comp2"], PermutationGenotype)


def test_hybrid_crossover_invalid_type():
    parent1 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    parent2 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    crossover_operator = HybridCrossover(
        operators={
            "comp1": OrderCrossover(),  # Invalid for RealGenotype
            "comp2": OrderCrossover(),
        }
    )
    with pytest.raises(TypeError, match=r"OrderCrossover is only applicable to .*PermutationGenotype.*"):
        crossover_operator.crossover(parent1, parent2)


def test_parallel_hybrid_crossover():
    parent1 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    parent2 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    crossover_operator = ParallelHybridCrossover(
        operators={
            "comp1": BlendCrossover(alpha=0.5),
            "comp2": OrderCrossover(),
        },
        max_workers=2,
    )
    child1, child2 = crossover_operator.crossover(parent1, parent2)
    assert isinstance(child1, HybridGenotype)
    assert isinstance(child2, HybridGenotype)
    assert "comp1" in child1.components
    assert "comp2" in child1.components
    assert "comp1" in child2.components
    assert "comp2" in child2.components
    # Check types of components
    assert isinstance(child1.components["comp1"], RealGenotype)
    assert isinstance(child1.components["comp2"], PermutationGenotype)
    assert isinstance(child2.components["comp1"], RealGenotype)
    assert isinstance(child2.components["comp2"], PermutationGenotype)


def test_parallel_hybrid_crossover_invalid_type():
    parent1 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    parent2 = HybridGenotype(
        {
            "comp1": make_real_genotype(),
            "comp2": make_permutation_genotype(),
        }
    )
    crossover_operator = ParallelHybridCrossover(
        operators={
            "comp1": OrderCrossover(),  # Invalid for RealGenotype
            "comp2": OrderCrossover(),
        },
        max_workers=2,
    )
    with pytest.raises(TypeError, match=r"OrderCrossover is only applicable to .*PermutationGenotype.*"):
        crossover_operator.crossover(parent1, parent2)
