from __future__ import annotations

import dataclasses
import logging
import math
import os
import pickle
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from evolib.core.individual import Individual, Population  # Population = NewType("Population", list[Individual])
from evolib.core.termination import TerminationCondition
from evolib.operators.crossover import CrossoverOperator
from evolib.operators.mutation import MutationOperator
from evolib.operators.replacement import ReplacementStrategy
from evolib.operators.selection import SelectionStrategy

# ---------------------------------------------------------------------------
# Engine config & stats
# ---------------------------------------------------------------------------


@dataclass
class GAConfig:
    population_size: int = 100
    elitism: int = 0
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    num_workers: int | None = 0  # 0 => synchronous; None/1 => single-thread; >1 => use pool
    executor_type: str = "process"  # 'process' | 'thread' | 'custom'
    seed: int | None = None
    checkpoint_path: Path | None = None
    checkpoint_interval_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be > 0")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0.0, 1.0]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0.0, 1.0]")
        if self.elitism < 0 or self.elitism >= self.population_size:
            raise ValueError("elitism must be >= 0 and < population_size")
        if self.executor_type not in ("process", "thread", "custom"):
            raise ValueError("executor_type must be 'process', 'thread', or 'custom'")


@dataclass
class GAStats:
    generation: int = 0
    evaluations: int = 0
    best_fitness: float = float("-inf")
    mean_fitness: float = float("-inf")
    history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GAEngineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# GAEngine
# ---------------------------------------------------------------------------


class GAEngine:
    def __init__(  # noqa: PLR0913
        self,
        config: GAConfig,
        selection: SelectionStrategy,
        crossover: CrossoverOperator,
        mutation: MutationOperator,
        replacement: ReplacementStrategy,
        termination: TerminationCondition,
        evaluator: Callable[[Individual], float] | None = None,
        executor: Executor | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.termination = termination
        self._user_evaluator = evaluator
        self._external_executor = executor
        self._executor: Executor | None = None

        # population is a list-like (Population alias)
        self.population: Population = cast(Population, [])
        self.generation: int = 0
        self.stats = GAStats()

        self._stop_requested = threading.Event()
        self.logger = logger or logging.getLogger("evolib.gaengine")
        self.logger.setLevel(logging.INFO)

        # checkpoint management
        self._last_checkpoint = time.time()

    # -----------------------------
    # Public API
    # -----------------------------

    def run(
        self, fitness_fn: Callable[[Individual], float] | None = None, initial_population: Population | None = None
    ) -> Population:
        """Run the GA until termination condition triggers. Returns final population.

        fitness_fn: user-defined function that maps Individual -> fitness (float).
        initial_population: optional seed population; if None, engine must be seeded by user.
        """
        if initial_population is not None:
            self.population = initial_population
        if len(self.population) == 0:
            raise GAEngineError("Initial population is empty. Provide an initial_population.")

        if fitness_fn is not None:
            self._user_evaluator = fitness_fn
        elif self._user_evaluator is None:
            raise GAEngineError("No evaluator specified. Provide one via constructor or run().")

        self._prepare_executor()

        try:
            while (
                not self.termination.should_terminate(self.stats.generation, self.population, self.stats.best_fitness)
                and not self._stop_requested.is_set()
            ):
                self.logger.info("Generation %d start", self.generation)
                self._evaluate_population_parallel()
                self._update_stats()

                # create offspring (selection done internally to be consistent)
                offspring = self._create_offspring()

                # evaluate offspring
                self._evaluate_individuals_parallel(offspring)
                self.stats.evaluations += len(offspring)

                # replace - replacement strategy is responsible for keeping population_size
                self.population = self.replacement.replace(self.population, offspring, self.config.population_size)

                self.generation += 1
                self.stats.generation = self.generation
                self._checkpoint_if_needed()

            return self.population
        finally:
            self._shutdown_executor()

    def stop(self) -> None:
        """Request a graceful stop between generations."""
        self._stop_requested.set()

    def save_state(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        state = {
            "config": dataclasses.asdict(self.config),
            "generation": self.generation,
            "population": self.population,
            "stats": self.stats,
        }
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic rename
        self.logger.info("Saved checkpoint to %s", path)

    def load_state(self, path: Path) -> None:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        # Basic validation
        self.generation = state["generation"]
        self.population = state["population"]
        self.stats = state["stats"]
        self.logger.info("Loaded checkpoint from %s", path)

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _prepare_executor(self) -> None:
        if self._external_executor is not None:
            self._executor = self._external_executor
            self.logger.info("Using external executor provided by caller")
            return

        num_workers = self.config.num_workers
        if not num_workers or num_workers <= 0:
            self._executor = None
            self.logger.info("Running synchronously (no executor)")
            return

        # test whether evaluator is picklable
        evaluator_picklable = True
        if self._user_evaluator is None:
            evaluator_picklable = False
        else:
            try:
                pickle.dumps(self._user_evaluator)
            except Exception:
                evaluator_picklable = False

        if self.config.executor_type == "process" and evaluator_picklable:
            self._executor = ProcessPoolExecutor(max_workers=num_workers)
            self.logger.info("ProcessPoolExecutor prepared with %d workers", num_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
            if self.config.executor_type == "process" and not evaluator_picklable:
                self.logger.warning(
                    "Fitness function not picklable: falling back to ThreadPoolExecutor to avoid pickling errors."
                )
            else:
                self.logger.info("ThreadPoolExecutor prepared with %d workers", num_workers)

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True)
            except Exception:
                self.logger.exception("Executor shutdown failed")

    def _evaluate_population_parallel(self) -> None:
        # Evaluate individuals lacking fitness in the population
        to_eval = [ind for ind in self.population if ind.fitness is None or math.isnan(ind.fitness)]
        self._evaluate_individuals_parallel(cast(Population, to_eval))
        self.stats.evaluations += len(to_eval)

    def _evaluate_individuals_parallel(self, individuals: Population) -> None:
        if not individuals:
            return
        if self._user_evaluator is None:
            raise GAEngineError("No fitness evaluator set.")

        # If using ProcessPoolExecutor, map only genotypes to avoid pickling self/closures
        if isinstance(self._executor, ProcessPoolExecutor):
            genotypes = [ind.genotype for ind in individuals]
            results = list(self._executor.map(self._user_evaluator, genotypes))
            for ind, res in zip(individuals, results, strict=False):
                try:
                    ind.fitness = float(res)
                except Exception:
                    ind.fitness = float("-inf")
                    self.logger.exception("Fitness evaluation error; setting -inf")
            return

        # ThreadPoolExecutor or synchronous
        if isinstance(self._executor, ThreadPoolExecutor):
            genotypes = [ind.genotype for ind in individuals]
            results = list(self._executor.map(self._user_evaluator, individuals))
            for ind, res in zip(individuals, results, strict=False):
                try:
                    ind.fitness = float(res)
                except Exception:
                    ind.fitness = float("-inf")
                    self.logger.exception("Fitness evaluation error; setting -inf")
            return

        # synchronous fallback
        for ind in individuals:
            try:
                ind.fitness = float(self._user_evaluator(ind))
            except Exception:
                ind.fitness = float("-inf")
                self.logger.exception("Fitness evaluation failed; setting -inf")

    def _update_stats(self) -> None:
        scores = [ind.fitness for ind in self.population if ind.fitness is not None and not math.isnan(ind.fitness)]
        if scores:
            best = max(scores)
            mean = sum(scores) / len(scores)
            self.stats.best_fitness = best
            self.stats.mean_fitness = mean
        else:
            self.stats.best_fitness = float("-inf")
            self.stats.mean_fitness = float("-inf")

        snapshot = {
            "generation": self.generation,
            "best": self.stats.best_fitness,
            "mean": self.stats.mean_fitness,
            "evaluations": self.stats.evaluations,
            "time": time.time(),
        }
        self.stats.history.append(snapshot)
        self.logger.info(
            "Generation %d stats: best=%s mean=%s evals=%d",
            self.generation,
            self.stats.best_fitness,
            self.stats.mean_fitness,
            self.stats.evaluations,
        )

    def _create_offspring(self) -> Population:
        """
        Create offspring for the next generation.

        - selection.select(pop, k) is used to get parents_pool
        - parents are paired; if odd number, drop last
        - for each pair: with probability crossover_rate call crossover.crossover(p1.genotype, p2.genotype)
        else copy parent genotypes
        - apply mutation.mutate(...) with probability mutation_rate per child genotype
        - wrap resulting Genotype into Individual(age=0, fitness=nan)
        """
        num_parents = max(2, self.config.population_size)
        parents_pool: Population = self.selection.select(self.population, n_parents=num_parents)
        if not parents_pool:
            raise GAEngineError("Selection returned empty parent pool.")

        if len(parents_pool) % 2 != 0:
            self.logger.debug("Parent pool size odd (%d); dropping last to make it even.", len(parents_pool))
            parents_pool = cast(Population, parents_pool[:-1])

        offspring: Population = cast(Population, [])

        for i in range(0, len(parents_pool), 2):
            p1, p2 = parents_pool[i], parents_pool[i + 1]

            if random.random() <= self.config.crossover_rate:
                child1_gen, child2_gen = self.crossover.crossover(p1.genotype, p2.genotype)
            else:
                child1_gen = p1.genotype.copy()
                child2_gen = p2.genotype.copy()

            # mutation per-child
            if random.random() <= self.config.mutation_rate:
                child1_gen = self.mutation.mutate(child1_gen)
            if random.random() <= self.config.mutation_rate:
                child2_gen = self.mutation.mutate(child2_gen)

            child1 = Individual(genotype=child1_gen, age=0, fitness=float("nan"))
            child2 = Individual(genotype=child2_gen, age=0, fitness=float("nan"))
            offspring.extend([child1, child2])

        return offspring

    def _checkpoint_if_needed(self) -> None:
        if not self.config.checkpoint_path or not self.config.checkpoint_interval_seconds:
            return
        now = time.time()
        if now - self._last_checkpoint >= self.config.checkpoint_interval_seconds:
            try:
                p = Path(self.config.checkpoint_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                self.save_state(p)
                self._last_checkpoint = now
            except Exception:
                self.logger.exception("Checkpoint failed")
