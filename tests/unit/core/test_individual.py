import numpy as np

from evolib.core.genotype import BinaryGenotype, RealGenotype
from evolib.core.individual import Individual


def test_individual_equality():
    g1 = BinaryGenotype.random(10)
    g2 = BinaryGenotype.random(10)

    ind1 = Individual(genotype=g1, age=5, fitness=10.0)
    ind2 = Individual(genotype=g1.copy(), age=5, fitness=10.0)
    ind3 = Individual(genotype=g2, age=5, fitness=10.0)
    ind4 = Individual(genotype=g1, age=6, fitness=10.0)
    ind5 = Individual(genotype=g1, age=5, fitness=12.0)

    assert ind1 == ind2, "Individuals with same attributes should be equal"
    assert ind1 != ind3, "Individuals with different genotypes should not be equal"
    assert ind1 != ind4, "Individuals with different ages should not be equal"
    assert ind1 != ind5, "Individuals with different fitness scores should not be equal"


def test_individual_copy():
    g1 = RealGenotype.random(10, (0.0, 1.0))
    ind1 = Individual(genotype=g1, age=3, fitness=15.0)
    ind2 = ind1.copy()

    assert ind1 == ind2, "Copied individual should be equal to the original"
    assert ind1.genotype is not ind2.genotype, "Genotypes should be different objects"
    assert np.array_equal(ind1.genotype.genes, ind2.genotype.genes), (
        "Genotype genes should be equal"
    )


def test_create_population():
    pop_size = 20
    population = Individual.create_population(lambda: BinaryGenotype.random(pop_size), pop_size)

    assert isinstance(population, list), "Population should be a list"
    assert len(population) == pop_size, f"Population size should be {pop_size}"
    for ind in population:
        assert isinstance(ind, Individual), "Each member of the population should be an Individual"
        assert len(ind.genotype.genes) == 20, "Each individual's genotype should have length 5"
