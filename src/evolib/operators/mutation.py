from typing import Any
import numpy as np
from evolib.core.genotype import Genotype, BinaryGenotype
from abc import ABC, abstractmethod


class MutationOperator(ABC):
    @abstractmethod
    def mutate(self, genotype: Genotype) -> Genotype:
        pass

class BitFlipMutation(MutationOperator):
    def __init__(self, probability: float = 0.01):
        self.probability = probability

    def mutate(self, genotype: BinaryGenotype) -> BinaryGenotype:
        mask = np.random.rand(len(genotype)) < self.probability
        mutated = genotype.genes.copy()
        mutated[mask] = ~mutated[mask]
        return BinaryGenotype(mutated)
