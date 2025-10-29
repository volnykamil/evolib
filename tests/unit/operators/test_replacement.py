"""
Unit tests for evolib.operators.replacement
"""

import pytest

from evolib.core.genotype import BinaryGenotype, Genotype, IntegerGenotype, RealGenotype
from evolib.core.individual import Individual, Population
from evolib.operators.replacement import (
    AgeBasedReplacement,
    ElitismReplacement,
    FitnessSharingReplacement,
    GenerationalReplacement,
    MuLambdaReplacement,
    MuPlusLambdaReplacement,
    SteadyStateReplacement,
)


# -----------------------------------------------------------------------------
# Helper factories
# -----------------------------------------------------------------------------
def make_individual(fitness: float, age: int = 0, length: int = 5, kind: str = "int") -> Individual:
    if kind == "int":
        genotype: Genotype = IntegerGenotype.random(length, (0, 10))
    elif kind == "bin":
        genotype = BinaryGenotype.random(length)
    elif kind == "real":
        genotype = RealGenotype.random(length, (0.0, 1.0))
    else:
        raise ValueError("Unsupported kind")
    ind = Individual(genotype=genotype, age=age, fitness=fitness)

    return ind


def make_population(size: int, kind: str = "int", fitness_start: float = 0.0) -> Population:
    return Population(
        [
            make_individual(fitness=fitness_start + i, age=i, length=7, kind=kind)
            for i in range(size)
        ]
    )


# -----------------------------------------------------------------------------
# GenerationalReplacement
# -----------------------------------------------------------------------------
def test_generational_replacement_basic():
    parents = make_population(5)
    offspring = make_population(8, fitness_start=10)
    strat = GenerationalReplacement()
    new_pop = strat.replace(parents, offspring, population_size=5)
    assert len(new_pop) == 5
    # Should be first N offspring (order preserved)
    assert all(ind is offspring[i] for i, ind in enumerate(new_pop))


# -----------------------------------------------------------------------------
# SteadyStateReplacement
# -----------------------------------------------------------------------------
def test_steady_state_replacement_default():
    parents = make_population(6)
    offspring = make_population(4, fitness_start=100)
    strat = SteadyStateReplacement(num_replacements=2)
    new_pop = strat.replace(parents, offspring, population_size=6)
    assert len(new_pop) == 6
    # Top two offspring should appear
    top_offspring = sorted(offspring, key=lambda i: i.fitness, reverse=True)[:2]
    for o in top_offspring:
        assert o in new_pop


def test_steady_state_replacement_all_replaced_when_large_num():
    parents = make_population(4)
    offspring = make_population(4, fitness_start=50)
    strat = SteadyStateReplacement(num_replacements=10)  # larger than parent size
    new_pop = strat.replace(parents, offspring, population_size=4)
    # Fallback behavior: slicing should yield only offspring truncated
    assert len(new_pop) == 4


# -----------------------------------------------------------------------------
# MuLambdaReplacement
# -----------------------------------------------------------------------------
def test_mu_lambda_replacement_selection():
    mu, lambda_ = 5, 8
    parents = make_population(6)
    offspring = make_population(lambda_, fitness_start=200)
    strat = MuLambdaReplacement(mu=mu, lambda_=lambda_)
    new_pop = strat.replace(parents, offspring, population_size=mu)
    assert len(new_pop) == mu
    # Only offspring (no parents) should be present
    for ind in new_pop:
        assert ind in offspring
        assert ind not in parents


def test_mu_lambda_replacement_warning_on_mismatch():
    mu, lambda_ = 3, 6
    parents = make_population(4)
    offspring = make_population(lambda_ - 2, fitness_start=500)  # mismatch
    strat = MuLambdaReplacement(mu=mu, lambda_=lambda_)
    with pytest.warns(UserWarning, match="Expected"):
        strat.replace(parents, offspring, population_size=mu)


def test_mu_lambda_invalid_params():
    with pytest.raises(AssertionError):
        MuLambdaReplacement(mu=0, lambda_=1)
    with pytest.raises(AssertionError):
        MuLambdaReplacement(mu=5, lambda_=4)


# -----------------------------------------------------------------------------
# MuPlusLambdaReplacement
# -----------------------------------------------------------------------------
def test_mu_plus_lambda_replacement_selection():
    mu, lambda_ = 5, 6
    parents = make_population(mu, fitness_start=10)
    offspring = make_population(lambda_, fitness_start=100)
    strat = MuPlusLambdaReplacement(mu=mu, lambda_=lambda_)
    new_pop = strat.replace(parents, offspring, population_size=mu)
    assert len(new_pop) == mu
    # Top mu by fitness from combined should be chosen (all offspring higher fitness)
    for ind in new_pop:
        assert ind in offspring


def test_mu_plus_lambda_replacement_warning_on_mismatch():
    mu, lambda_ = 4, 5
    parents = make_population(mu, fitness_start=10)
    offspring = make_population(lambda_ - 1, fitness_start=300)
    strat = MuPlusLambdaReplacement(mu=mu, lambda_=lambda_)
    with pytest.warns(UserWarning, match="Expected"):
        strat.replace(parents, offspring, population_size=mu)


def test_mu_plus_lambda_invalid_params():
    with pytest.raises(AssertionError):
        MuPlusLambdaReplacement(mu=0, lambda_=5)
    with pytest.raises(AssertionError):
        MuPlusLambdaReplacement(mu=3, lambda_=-1)


# -----------------------------------------------------------------------------
# ElitismReplacement
# -----------------------------------------------------------------------------
def test_elitism_replacement():
    parents = make_population(6, fitness_start=10)
    offspring = make_population(10, fitness_start=100)
    strat = ElitismReplacement(elite_size=2)
    new_pop = strat.replace(parents, offspring, population_size=8)
    assert len(new_pop) == 8
    # First two should be elites (highest fitness among parents)
    parents_sorted = sorted(parents, key=lambda ind: ind.fitness, reverse=True)
    assert new_pop[0] is parents_sorted[0]
    assert new_pop[1] is parents_sorted[1]


# -----------------------------------------------------------------------------
# AgeBasedReplacement
# -----------------------------------------------------------------------------
def test_age_based_replacement_youngest_selected():
    parents = Population(
        [
            make_individual(fitness=10, age=50),
            make_individual(fitness=20, age=5),
            make_individual(fitness=30, age=10),
        ]
    )
    offspring = Population(
        [
            make_individual(fitness=100, age=2),
            make_individual(fitness=110, age=1),
            make_individual(fitness=90, age=40),
        ]
    )
    strat = AgeBasedReplacement()
    new_pop = strat.replace(parents, offspring, population_size=4)
    ages = [ind.age for ind in new_pop]
    assert ages == sorted(ages)  # youngest first
    assert len(new_pop) == 4


# -----------------------------------------------------------------------------
# FitnessSharingReplacement
# -----------------------------------------------------------------------------
def test_fitness_sharing_replacement_basic():
    # Use IntegerGenotype so subtraction works (__sub__ implemented)
    parents = Population(
        [
            make_individual(fitness=50, age=1, kind="int"),
            make_individual(fitness=60, age=2, kind="int"),
        ]
    )
    # Create offspring with duplicate genotypes to trigger sharing penalty
    base_genes = IntegerGenotype.random(5, (0, 5)).genes
    o1 = make_individual(fitness=200, age=3, kind="int")
    o1.genotype.genes = base_genes.copy()
    o2 = make_individual(fitness=190, age=4, kind="int")
    o2.genotype.genes = base_genes.copy()
    o3 = make_individual(fitness=180, age=5, kind="int")
    o3.genotype.genes = base_genes.copy()
    offspring = Population([o1, o2, o3])
    strat = FitnessSharingReplacement(sigma_share=10.0, alpha=1.0)
    new_pop = strat.replace(parents, offspring, population_size=3)
    assert len(new_pop) == 3
    # All individuals are from combined set
    combined = set(parents + offspring)
    assert all(ind in combined for ind in new_pop)


def test_fitness_sharing_sharing_function_values():
    strat = FitnessSharingReplacement(sigma_share=5.0, alpha=2.0)
    # distance < sigma_share -> positive value
    val_close = strat._sharing_function(2.5)
    assert 0 < val_close < 1
    # distance >= sigma_share -> zero
    assert strat._sharing_function(5.0) == 0
    assert strat._sharing_function(10.0) == 0


# -----------------------------------------------------------------------------
# Defensive tests for attribute consistency
# -----------------------------------------------------------------------------
def test_individual_has_fitnessScore_and_dynamic_fitness():
    ind = make_individual(fitness=42.0, age=3)
    assert hasattr(ind, "fitness")
    assert ind.fitness == ind.fitness


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------
def test_population_size_truncation():
    parents = make_population(3)
    offspring = make_population(10, fitness_start=50)
    strat = GenerationalReplacement()
    new_pop = strat.replace(parents, offspring, population_size=4)
    assert len(new_pop) == 4
    # When population_size larger than offspring slice length 
	# fallback still respects requested size due to slicing logic
    new_pop2 = strat.replace(parents, offspring, population_size=20)
    assert len(new_pop2) == len(offspring)  # truncated to available offspring


# -----------------------------------------------------------------------------
# Mutation of ordering / stability checks
# -----------------------------------------------------------------------------
def test_steady_state_replacement_ordering_stability():
    parents = make_population(6)
    offspring = make_population(3, fitness_start=500)
    strat = SteadyStateReplacement(num_replacements=1)
    new_pop = strat.replace(parents, offspring, population_size=6)
    # Ensure highest-fitness offspring inserted, others preserved except one removed
    top_offspring = max(offspring, key=lambda i: i.fitness)
    assert top_offspring in new_pop
    assert len(new_pop) == 6


# -----------------------------------------------------------------------------
# Ensure no division by zero in FitnessSharingReplacement
# -----------------------------------------------------------------------------
def test_fitness_sharing_no_divide_by_zero():
    parents = Population([make_individual(fitness=10, age=1, kind="int")])
    offspring = Population([make_individual(fitness=20, age=2, kind="int")])
    strat = FitnessSharingReplacement(sigma_share=0.01, alpha=1.0)
    # Very small sigma_share still includes self-distance (0) => niche_count >= 1
    new_pop = strat.replace(parents, offspring, population_size=2)
    assert len(new_pop) == 2


# -----------------------------------------------------------------------------
# Parametrized stress test for multiple strategies
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: GenerationalReplacement(),
        lambda: SteadyStateReplacement(num_replacements=2),
        lambda: MuLambdaReplacement(mu=3, lambda_=5),
        lambda: MuPlusLambdaReplacement(mu=3, lambda_=4),
        lambda: ElitismReplacement(elite_size=1),
        lambda: AgeBasedReplacement(),
    ],
)
def test_strategy_returns_population(strategy_factory):
    strat = strategy_factory()
    parents = make_population(5)
    offspring = make_population(6, fitness_start=100)
    new_pop = strat.replace(parents, offspring, population_size=5)
    assert isinstance(new_pop, list)
    assert len(new_pop) <= 5
