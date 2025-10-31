import numpy as np

from evolib.core.genotype import BinaryGenotype
from evolib.core.individual import Individual
from evolib.core.termination import FitnessThresholdTermination
from evolib.engine import GAConfig, GAEngine
from evolib.operators.crossover import OnePointCrossover
from evolib.operators.mutation import BitFlipMutation
from evolib.operators.replacement import ElitismReplacement
from evolib.operators.selection import TournamentSelection


# Run a simple integration test on the OneMax problem
def test_onemax_integration():
    config = GAConfig(
        population_size=100,
        num_workers=4,
    )

    engine = GAEngine(
        config=config,
        selection=TournamentSelection(k=3),
        replacement=ElitismReplacement(elite_size=2),
        crossover=OnePointCrossover(),
        mutation=BitFlipMutation(probability=0.01),
        termination=FitnessThresholdTermination(fitness_threshold=1.0),
    )
    initial_population = Individual.create_population(lambda: BinaryGenotype.random(length=100), config.population_size)
    final_population = engine.run(
        fitness_fn=lambda ind: np.sum(ind.genotype.genes) / len(ind.genotype.genes),
        initial_population=initial_population,
    )
    best_individual = max(final_population, key=lambda ind: ind.fitness)
    assert best_individual.fitness == 1.0  # Ensure we found the optimal solution
    assert np.all(best_individual.genotype.genes)  # All genes should be True
