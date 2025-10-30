import time

import numpy as np

from evolib.core.genotype import BinaryGenotype
from evolib.core.individual import Individual, Population
from evolib.core.termination import (
    DiverseUnderMinimumTermination,
    FitnessThresholdTermination,
    HybridTermination,
    MaxGenerationsTermination,
    StagnationTermination,
    TimeLimitTermination,
)


def test_max_generations_termination():
    term = MaxGenerationsTermination(max_generations=10)
    assert not term.should_terminate(5, [], 0.0)
    assert term.should_terminate(10, [], 0.0)
    assert term.should_terminate(15, [], 0.0)


def test_fitness_threshold_termination():
    term = FitnessThresholdTermination(fitness_threshold=0.9)
    assert not term.should_terminate(5, [], 0.5)
    assert term.should_terminate(10, [], 0.9)
    assert term.should_terminate(15, [], 1.0)


def test_stagnation_termination():
    term = StagnationTermination(max_stagnant_generations=3)
    fitness_values = [0.5, 0.6, 0.6, 0.6, 0.6]
    results = []
    for gen, fit in enumerate(fitness_values):
        results.append(term.should_terminate(gen, [], fit))
    assert results == [False, False, False, True, True]


def test_time_limit_termination():
    term = TimeLimitTermination(time_limit_seconds=1.0)
    assert not term.should_terminate(0, [], 0.0)

    time.sleep(1.1)
    assert term.should_terminate(1, [], 0.0)


def test_diverse_under_minimum_termination():
    term = DiverseUnderMinimumTermination(min_diversity=0.5)
    population = Individual.create_population(lambda: BinaryGenotype.random(10), 4)
    assert not term.should_terminate(0, population, 0.0)
    # Artificially reduce diversity
    for ind in population:
        ind.genotype = BinaryGenotype(np.zeros(10, dtype=np.bool_))
    assert term.should_terminate(1, population, 0.0)


def test_hybrid_termination():
    term1 = MaxGenerationsTermination(max_generations=5)
    term2 = FitnessThresholdTermination(fitness_threshold=0.8)
    hybrid_term = HybridTermination([term1, term2])

    assert not hybrid_term.should_terminate(3, [], 0.5)
    assert hybrid_term.should_terminate(5, [], 0.5)
    assert hybrid_term.should_terminate(4, [], 0.8)
