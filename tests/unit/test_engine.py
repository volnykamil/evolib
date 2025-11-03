from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pytest

from evolib.core.genotype import BinaryGenotype
from evolib.core.individual import Individual, Population
from evolib.core.termination import MaxGenerationsTermination
from evolib.engine import GAConfig, GAEngine, GAEngineError, GAStats
from evolib.operators.crossover import OnePointCrossover
from evolib.operators.mutation import BitFlipMutation
from evolib.operators.replacement import GenerationalReplacement
from evolib.operators.selection import RandomSelection


class BaseTestGAEngine:
    def _population(self, n: int) -> Population:
        return Population([Individual(BinaryGenotype.random(8), fitness=0.0) for _ in range(n)])

    def _fitness(self, ind: Individual) -> float:
        return float(ind.genotype.genes.sum())


class TestGAConfig:
    def test_gaconfig_valid_defaults(self):
        cfg = GAConfig()
        assert cfg.population_size == 100
        assert cfg.elitism == 0
        assert 0.0 <= cfg.crossover_rate <= 1.0
        assert 0.0 <= cfg.mutation_rate <= 1.0

    @pytest.mark.parametrize("population_size", [0, -1])
    def test_gaconfig_invalid_population_size(self, population_size):
        with pytest.raises(ValueError) as e:
            GAConfig(population_size=population_size)
        assert "population_size" in str(e.value)

    @pytest.mark.parametrize("elitism", [-1])
    def test_gaconfig_invalid_elitism_negative(self, elitism):
        with pytest.raises(ValueError):
            GAConfig(elitism=elitism)

    def test_gaconfig_invalid_elitism_exceeds_population(self):
        with pytest.raises(ValueError):
            GAConfig(population_size=5, elitism=6)

    @pytest.mark.parametrize("crossover_rate", [-0.01, 1.01])
    def test_gaconfig_invalid_crossover_rate(self, crossover_rate):
        with pytest.raises(ValueError):
            GAConfig(crossover_rate=crossover_rate)

    @pytest.mark.parametrize("mutation_rate", [-0.01, 1.5])
    def test_gaconfig_invalid_mutation_rate(self, mutation_rate):
        with pytest.raises(ValueError):
            GAConfig(mutation_rate=mutation_rate)

    @pytest.mark.parametrize("num_workers", [-5])
    def test_gaconfig_invalid_num_workers(self, num_workers):
        with pytest.raises(ValueError):
            GAConfig(num_workers=num_workers)

    @pytest.mark.parametrize("executor_type", ["invalid", "PROCESS", "threads"])  # case-sensitive
    def test_gaconfig_invalid_executor_type(self, executor_type):
        with pytest.raises(ValueError):
            GAConfig(executor_type=executor_type)

    @pytest.mark.parametrize("checkpoint_interval_seconds", [0, -10])
    def test_gaconfig_invalid_checkpoint_interval(self, checkpoint_interval_seconds):
        with pytest.raises(ValueError):
            GAConfig(checkpoint_interval_seconds=checkpoint_interval_seconds)

    @pytest.mark.parametrize("seed", [-1])
    def test_gaconfig_invalid_seed(self, seed):
        with pytest.raises(ValueError):
            GAConfig(seed=seed)

    def test_gaconfig_valid_edge_values(self):
        cfg = GAConfig(
            population_size=1,
            elitism=0,
            crossover_rate=0.0,
            mutation_rate=1.0,
            num_workers=0,
            executor_type="thread",
            checkpoint_interval_seconds=None,
            seed=0,
        )
        assert cfg.population_size == 1
        assert cfg.crossover_rate == 0.0
        assert cfg.mutation_rate == 1.0
        assert cfg.executor_type == "thread"
        assert cfg.seed == 0


class TestStatsHistory(BaseTestGAEngine):
    def test_stats_history_pruned_to_cap(self):
        # Run for more generations than max_history and verify pruning keeps only newest entries.
        max_hist = 3
        gens = 7
        config = GAConfig(population_size=8, max_history=max_hist)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(gens),
            evaluator=self._fitness,
        )
        engine.population = self._population(config.population_size)
        engine.run()

        history = engine.stats.history
        assert len(history) <= max_hist
        # Ensure we kept the most recent snapshots by comparing generation numbers monotonically increasing
        gens_recorded = [h["generation"] for h in history]
        assert gens_recorded == sorted(gens_recorded)
        if len(gens_recorded) == max_hist:
            # Stats are captured before generation increment, so final generation value
            # may exceed last recorded snapshot by 1.
            assert gens_recorded[-1] in {engine.generation, engine.generation - 1}
            start_expected = engine.generation - max_hist + 1
            # Allow off-by-one due to capture timing
            assert gens_recorded[0] in {start_expected, start_expected - 1}

    def test_stats_history_unlimited_when_none(self):
        gens = 5
        config = GAConfig(population_size=6, max_history=None)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(gens),
            evaluator=self._fitness,
        )
        engine.population = self._population(config.population_size)
        engine.run()

        history = engine.stats.history
        # Should have one entry per generation processed
        assert len(history) == engine.generation + 1 or len(history) == engine.generation

    def test_stats_history_zero_cap_allows_latest_only(self):
        # Edge case: max_history = 0 means we expect empty or minimal retention
        # (implementation keeps none if excess > 0).
        config = GAConfig(population_size=5, max_history=0)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(3),
            evaluator=self._fitness,
        )
        engine.population = self._population(config.population_size)
        engine.run()
        assert len(engine.stats.history) == 0


class TestExecutors(BaseTestGAEngine):
    def test_thread_vs_process_executor_consistency(self, monkeypatch):
        # Ensure evaluator is picklable to allow process pool usage
        assert pickle.dumps(self._fitness)
        gens = 3
        pop_size = 12

        rng = np.random.default_rng(42)

        config_proc = GAConfig(
            population_size=pop_size,
            num_workers=2,
            executor_type="process",
            mutation_rate=0.0,
            crossover_rate=0.0,
        )
        engine_proc = GAEngine(
            config=config_proc,
            selection=RandomSelection(rng=rng),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(gens),
            evaluator=self._fitness,
        )
        # Shared initial population (deep copy for each engine)
        initial_pop = self._population(pop_size)
        engine_proc.population = Population([ind.copy() for ind in initial_pop])
        pop_final_proc = engine_proc.run()

        # Recreate identical RNG state for thread engine
        rng2 = np.random.default_rng(42)

        config_thread = GAConfig(
            population_size=pop_size,
            num_workers=2,
            executor_type="thread",
            mutation_rate=0.0,
            crossover_rate=0.0,
        )
        engine_thread = GAEngine(
            config=config_thread,
            selection=RandomSelection(rng=rng2),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(gens),
            evaluator=self._fitness,
        )
        # Use same original baseline (not the result of process run)
        engine_thread.population = Population([ind.copy() for ind in initial_pop])
        pop_final_thread = engine_thread.run()

        # With crossover and mutation disabled, only selection/replacement ordering could differ.
        # We compare sorted fitness lists for consistency.
        proc_fitness_sorted = sorted(ind.fitness for ind in pop_final_proc)
        thread_fitness_sorted = sorted(ind.fitness for ind in pop_final_thread)
        assert proc_fitness_sorted == thread_fitness_sorted


@pytest.fixture
def temp_checkpoint(tmp_path: Path):
    return tmp_path / "checkpoint.pkl"


class TestCheckpointing(BaseTestGAEngine):
    def test_checkpoint_round_trip(self, temp_checkpoint: Path):
        config = GAConfig(population_size=10, checkpoint_path=temp_checkpoint, checkpoint_interval_seconds=1)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),  # deterministic
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=self._fitness,
        )
        engine.population = self._population(config.population_size)
        # Force checkpoint eligibility before first generation completes
        assert config.checkpoint_interval_seconds is not None
        engine._last_checkpoint = time.time() - config.checkpoint_interval_seconds  # type: ignore[attr-defined]
        engine.run()
        assert temp_checkpoint.exists(), "Checkpoint file should exist after run with interval"

        # Load into a fresh engine
        new_engine = GAEngine(
            config=config,  # same config for simplicity
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=self._fitness,
        )
        new_engine.load_state(temp_checkpoint)
        assert new_engine.generation == engine.generation
        assert len(new_engine.population) == len(engine.population)
        assert new_engine.stats.best_fitness == engine.stats.best_fitness

    def test_checkpoint_version_mismatch_strict(self, temp_checkpoint: Path):
        config = GAConfig(population_size=4, checkpoint_path=temp_checkpoint, checkpoint_interval_seconds=1)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=self._fitness,
        )
        engine.population = self._population(config.population_size)
        assert config.checkpoint_interval_seconds is not None
        engine._last_checkpoint = time.time() - config.checkpoint_interval_seconds  # type: ignore[attr-defined]
        engine.run()
        assert temp_checkpoint.exists()

        # Corrupt version in checkpoint file
        raw = pickle.loads(temp_checkpoint.read_bytes())
        raw["version"] = 999  # mismatched
        with open(temp_checkpoint, "wb") as fh:
            pickle.dump(raw, fh)

        new_engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=self._fitness,
        )

        with pytest.raises(GAEngineError):
            new_engine.load_state(temp_checkpoint, strict=True)

    def test_checkpoint_version_mismatch_non_strict(self, temp_checkpoint: Path):
        config = GAConfig(population_size=4, checkpoint_path=temp_checkpoint, checkpoint_interval_seconds=1)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=self._fitness,
        )
        engine.population = self._population(config.population_size)
        assert config.checkpoint_interval_seconds is not None
        engine._last_checkpoint = time.time() - config.checkpoint_interval_seconds  # type: ignore[attr-defined]
        engine.run()
        assert temp_checkpoint.exists()

        # Corrupt version in checkpoint file
        raw = pickle.loads(temp_checkpoint.read_bytes())
        raw["version"] = 999  # mismatched
        with open(temp_checkpoint, "wb") as fh:
            pickle.dump(raw, fh)

        new_engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=self._fitness,
        )

        # Should not raise in non-strict mode
        new_engine.load_state(temp_checkpoint, strict=False)
        assert new_engine.generation == engine.generation
        assert len(new_engine.population) == len(engine.population)
        assert new_engine.stats.best_fitness == engine.stats.best_fitness


class BadNumber:
    """Helper object whose conversion to float raises an exception."""

    def __init__(self, value: float):
        self.value = value

    def __float__(self) -> float:  # pragma: no cover - exercised indirectly
        raise ValueError("cannot convert BadNumber")


class TestLifecycleHooks(BaseTestGAEngine):
    def test_generation_start_end_hooks_invoked(self, tmp_path: Path):
        starts: list[tuple[int, int]] = []  # (generation, pop_size)
        ends: list[GAStats] = []

        def on_start(gen: int, pop: Population) -> None:
            starts.append((gen, len(pop)))

        def on_end(stats: GAStats, pop: Population) -> None:
            # capture snapshot (copy best fitness to avoid mutation concerns)
            ends.append(stats)

        config = GAConfig(
            population_size=6,
            on_generation_start=on_start,
            on_generation_end=on_end,
            checkpoint_path=None,
        )

        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(2),
            evaluator=self._fitness,
        )
        engine.population = self._population(config.population_size)
        engine.run()

        # Since termination checks before loop, and generation increments at end:
        # We expect starts for generations 0 and 1, ends for generations 0 and 1.
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0][0] == 0
        assert starts[1][0] == 1
        # Population size remains constant
        assert all(sz == config.population_size for (_, sz) in starts)
        assert all(isinstance(s.generation, int) for s in ends)
        assert engine.generation == 2

    def test_evaluation_error_hook_invoked_per_individual(self):
        errors: list[tuple[int, Exception]] = []

        def evaluator(ind: Individual):  # returns either normal float or BadNumber
            # Mark half individuals for failure using first gene heuristic
            if ind.genotype.genes[0] == 1:
                return BadNumber(float(ind.genotype.genes.sum()))
            return float(ind.genotype.genes.sum())

        def on_error(ind: Individual, ex: Exception) -> None:
            errors.append((len(ind.genotype.genes), ex))

        config = GAConfig(population_size=8, on_evaluation_error=on_error, mutation_rate=0.0)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=evaluator,
        )
        engine.population = self._population(config.population_size)
        # Tag every even index to trigger BadNumber path
        for i, ind in enumerate(engine.population):
            if i % 2 == 0:
                ind.genotype.genes[0] = 1
            else:
                ind.genotype.genes[0] = 0
            # Force evaluation for all individuals by invalidating existing fitness
            ind.fitness = float("nan")

        engine.run()
        # Expect at least half the population to trigger conversion errors (initial population)
        # and at most all initial + all offspring evaluations.
        assert len(errors) >= config.population_size // 2
        assert len(errors) <= config.population_size * 2
        assert all(isinstance(ex, ValueError) for (_, ex) in errors)

    def test_evaluation_error_hook_not_invoked_on_batch_failure(self):
        """Raising inside evaluator for any individual aborts entire batch; no per-individual callback."""
        errors: list[tuple[int, Exception]] = []

        def evaluator(ind: Individual):  # raise for first individual only
            if ind.genotype.genes[0] == 1:
                raise RuntimeError("boom")
            return 0.0

        def on_error(ind: Individual, ex: Exception) -> None:
            errors.append((len(ind.genotype.genes), ex))

        config = GAConfig(population_size=6, on_evaluation_error=on_error)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=evaluator,
        )
        engine.population = self._population(config.population_size)
        # Ensure at least one raises
        engine.population[0].genotype.genes[0] = 1
        engine.run()
        # Batch failure path logs once and sets all -inf; callback not used
        assert len(errors) == 0
        assert all(ind.fitness == float("-inf") for ind in engine.population)
        assert all(isinstance(ex, RuntimeError) for (_, ex) in errors)

    def test_evaluation_error_no_hook_does_not_crash(self):
        def faulty_fitness(ind: Individual) -> float:
            raise ValueError("fail")

        config = GAConfig(population_size=4)
        engine = GAEngine(
            config=config,
            selection=RandomSelection(),
            crossover=OnePointCrossover(),
            mutation=BitFlipMutation(probability=0.0),
            replacement=GenerationalReplacement(),
            termination=MaxGenerationsTermination(1),
            evaluator=faulty_fitness,
        )
        engine.population = self._population(config.population_size)

        # Should not raise GAEngineError; individuals get -inf fitness
        engine.run()
        assert all(ind.fitness == float("-inf") for ind in engine.population)


class TestReproducibility(BaseTestGAEngine):
    def population_with_rng(self, n: int, rng: np.random.Generator) -> Population:
        # Deterministic initial population using rng bit draws
        pop: list[Individual] = []
        for _ in range(n):
            genes = rng.integers(
                0, 2, size=32, dtype=np.int8
            )  # BinaryGenotype expects bool-like; int8 works via asserts?
            genes = genes.astype(np.bool_)
            pop.append(Individual(BinaryGenotype(genes), fitness=float("nan")))
        return Population(pop)

    def test_two_runs_identical_trajectory_and_final_distribution(self):
        seed = 2024
        gens = 8
        pop_size = 20

        def run_engine(seed: int, gens: int, pop_size: int) -> tuple[list[float], list[float]]:
            rng_selection = np.random.default_rng(seed)
            config = GAConfig(
                population_size=pop_size,
                num_workers=0,  # synchronous for determinism (no scheduling variance)
                mutation_rate=0.05,
                crossover_rate=0.7,
                seed=seed,
                max_history=None,
            )
            engine = GAEngine(
                config=config,
                selection=RandomSelection(rng=rng_selection),
                crossover=OnePointCrossover(rng=np.random.default_rng(seed + 100)),
                mutation=BitFlipMutation(probability=0.05, rng=np.random.default_rng(seed + 200)),
                replacement=GenerationalReplacement(),
                termination=MaxGenerationsTermination(gens),
                evaluator=self._fitness,
            )
            # Use a dedicated RNG for genotype initialization to keep independence from selection RNG
            rng_pop = np.random.default_rng(seed + 10)
            engine.population = self.population_with_rng(pop_size, rng_pop)
            engine.run()
            # Extract trajectory of best fitness from stats history
            best_traj = [snap["best"] for snap in engine.stats.history]
            final_sorted = sorted(ind.fitness for ind in engine.population)
            return best_traj, final_sorted

        best1, final1 = run_engine(seed, gens, pop_size)
        best2, final2 = run_engine(seed, gens, pop_size)

        assert best1 == best2
        assert final1 == final2
