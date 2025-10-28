import numpy as np
import pytest
from evolib.operators.selection import (
    SelectionStrategy,
    RouletteWheelSelection,
    StochasticUniversalSampling,
    RankSelection,
    TournamentSelection,
    TruncationSelection,
    BoltzmannSelection,
    FitnessSharingSelection,
    RandomSelection,
    Elitism
)
from evolib.core.individual import Individual, Population


@pytest.fixture
def sample_population():
    return [Individual(genotype=None, fitness=float(i + 1)) for i in range(10)]


def validate_selection(selection: SelectionStrategy, population: Population, n: int) -> Population:
    result = selection.select(population, n)
    assert len(result) == n
    assert all(isinstance(ind, Individual) for ind in result)
    return result


def test_roulette_wheel_selection(sample_population):
    sel = RouletteWheelSelection()
    selected = validate_selection(sel, sample_population, 5)
    many_trials = 1000
    counts = np.zeros(len(sample_population))
    for _ in range(many_trials):
        chosen = sel.select(sample_population, 1)[0]
        idx = sample_population.index(chosen)
        counts[idx] += 1
    assert counts[-1] > counts[0]  # Higher fitness picked more often


def test_stochastic_universal_sampling(sample_population):
    sel = StochasticUniversalSampling()
    selected = validate_selection(sel, sample_population, 5)
    assert len(selected) == 5


def test_rank_selection(sample_population):
    sel = RankSelection()
    selected = validate_selection(sel, sample_population, 5)
    assert len(selected) == 5


def test_tournament_selection(sample_population):
    sel = TournamentSelection(k=3)
    selected = validate_selection(sel, sample_population, 5)
    assert len(selected) == 5
    assert all(isinstance(ind, Individual) for ind in selected)


def test_truncation_selection(sample_population):
    sel = TruncationSelection(fraction=0.3)
    selected = validate_selection(sel, sample_population, 5)
    assert len(selected) == 5


def test_boltzmann_selection(sample_population):
    sel = BoltzmannSelection(temperature=1.0)
    selected = validate_selection(sel, sample_population, 5)
    assert len(selected) == 5


def test_fitness_sharing_selection(sample_population):
    sel = FitnessSharingSelection(sigma_share=2.0)
    selected = validate_selection(sel, sample_population, 5)
    assert len(selected) == 5


def test_random_selection(sample_population):
    sel = RandomSelection()
    selected = validate_selection(sel, sample_population, 5)
    assert len(selected) == 5


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
        Elitism()
    ]
    for strat in strategies:
        result = strat.select(sample_population, 3) \
                 if not isinstance(strat, Elitism) else strat.select(sample_population)
        assert all(isinstance(ind, Individual) for ind in result)