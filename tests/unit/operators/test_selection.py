import numpy as np
import pytest

from evolib.core.genotype import BinaryGenotype
from evolib.core.individual import Individual, Population
from evolib.operators.selection import (
    BoltzmannSelection,
    Elitism,
    FitnessSharingSelection,
    RandomSelection,
    RankSelection,
    RouletteWheelSelection,
    SelectionStrategy,
    StochasticUniversalSampling,
    TournamentSelection,
    TruncationSelection,
)


@pytest.fixture
def sample_population():
    return [Individual(genotype=None, fitness=float(i + 1)) for i in range(10)]


def _population(n: int) -> Population:
    # deterministic genotype pattern for stable fitness ranking
    pop: list[Individual] = []
    for i in range(n):
        genes = np.zeros(16, dtype=np.bool_)
        if i % 16 != 0:
            genes[: i % 16] = True
        fitness = float(genes.sum())
        pop.append(Individual(BinaryGenotype(genes), fitness=fitness))
    return Population(pop)


def validate_selection(selection: SelectionStrategy, population: Population, n: int) -> Population:
    result = selection.select(population, n)
    assert len(result) == n
    assert all(isinstance(ind, Individual) for ind in result)
    return result


def test_roulette_wheel_selection(sample_population):
    sel = RouletteWheelSelection()
    validate_selection(sel, sample_population, 5)
    many_trials = 1000
    counts = np.zeros(len(sample_population))
    for _ in range(many_trials):
        chosen = sel.select(sample_population, 1)[0]
        idx = sample_population.index(chosen)
        counts[idx] += 1
    assert counts[-1] > counts[0]  # Higher fitness picked more often


def test_roulette_wheel_selection_reproducible_with_seed():
    pop = _population(30)
    # Provide increasing fitness to ensure variety
    for i, ind in enumerate(pop):
        ind.fitness = float(i + 1)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    sel1 = RouletteWheelSelection(rng=rng1)
    sel2 = RouletteWheelSelection(rng=rng2)
    picks1 = sel1.select(pop, n_parents=15)
    picks2 = sel2.select(pop, n_parents=15)
    assert [ind.fitness for ind in picks1] == [ind.fitness for ind in picks2]


def test_stochastic_universal_sampling(sample_population):
    sel = StochasticUniversalSampling()
    validate_selection(sel, sample_population, 5)


def test_rank_selection(sample_population):
    sel = RankSelection()
    validate_selection(sel, sample_population, 5)


def test_tournament_selection(sample_population):
    sel = TournamentSelection(k=3)
    validate_selection(sel, sample_population, 5)


def test_tournament_selection_reproducible_with_seed():
    pop = _population(25)
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    # NOTE: constructor argument is 'k' (tournament size), adjust from previous incorrect name
    sel1 = TournamentSelection(k=3, rng=rng1)
    sel2 = TournamentSelection(k=3, rng=rng2)
    picks1 = sel1.select(pop, n_parents=12)
    picks2 = sel2.select(pop, n_parents=12)
    assert [ind.fitness for ind in picks1] == [ind.fitness for ind in picks2]


def test_truncation_selection(sample_population):
    sel = TruncationSelection(fraction=0.3)
    validate_selection(sel, sample_population, 5)


def test_boltzmann_selection(sample_population):
    sel = BoltzmannSelection(temperature=1.0)
    validate_selection(sel, sample_population, 5)


def test_fitness_sharing_selection(sample_population):
    sel = FitnessSharingSelection(sigma_share=2.0)
    validate_selection(sel, sample_population, 5)


def test_random_selection(sample_population):
    sel = RandomSelection()
    validate_selection(sel, sample_population, 5)


def test_random_selection_reproducible_with_seed():
    pop = _population(20)
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)
    sel1 = RandomSelection(rng=rng1)
    sel2 = RandomSelection(rng=rng2)
    picks1 = sel1.select(pop, n_parents=10)
    picks2 = sel2.select(pop, n_parents=10)
    assert [ind.fitness for ind in picks1] == [ind.fitness for ind in picks2]


def test_elitism_selection(sample_population):
    sel = Elitism(elite_size=2)
    elites = sel.select(sample_population)
    assert len(elites) == 2
    assert all(e.fitness >= elites[-1].fitness for e in elites)


def test_selection_output_type_consistency(sample_population):
    strategies = [
        RouletteWheelSelection(),
        StochasticUniversalSampling(),
        RankSelection(),
        TournamentSelection(),
        TruncationSelection(),
        BoltzmannSelection(),
        FitnessSharingSelection(),
        RandomSelection(),
        Elitism(),
    ]
    for strat in strategies:
        result = (
            strat.select(sample_population, 3) if not isinstance(strat, Elitism) else strat.select(sample_population)
        )
        assert all(isinstance(ind, Individual) for ind in result)
