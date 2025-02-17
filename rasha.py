import math
import numpy as np
from dataclasses import dataclass
from base_opt import BaseOpt

@dataclass
class CandidateStats:
    n: int
    reward: float

    def update(self, r: float) -> None:
        self.n += 1
        self.reward += r

    @property
    def average_reward(self) -> float:
        return self.reward / self.n
    
def ucb1_sampler(keys: list[int], candidates: list[CandidateStats]) -> int:
    unexplored = [k for k, c in zip(keys, candidates) if c.n == 0]
    if unexplored:
        return np.random.choice(unexplored)

    total_n = sum(c.n for c in candidates)
    ucb_values = [c.average_reward + math.sqrt(2 * math.log(total_n) / c.n) for c in candidates]
    return keys[np.argmax(ucb_values)]

class Level:
    """
    A rung in the halving procedure.
    'capacity' indicates how many distinct configs we allow at this rung.
    """
    def __init__(self, max_resource: int) -> None:
        self.max_resource = max_resource
        self.candidates: dict[int, CandidateStats] = {}

    def __setitem__(self, combination_idx: int, candidate_stats: CandidateStats) -> None:
        self.candidates[combination_idx] = candidate_stats
        
    def __getitem__(self, combination_idx: int) -> CandidateStats:
        return self.candidates[combination_idx]
    
    def __delitem__(self, combination_idx: int) -> None:
        del self.candidates[combination_idx]
    
    def __len__(self) -> int:
        return len(self.candidates)

    def candidates_with_maxed_resource(self) -> list[int]:
        """
        Returns the indices of candidates that have used up their resource.
        """
        return [idx for idx, stats in self.candidates.items() if stats.n >= self.max_resource]

    def candidates_in_progress(self) -> list[int]:
        """
        Returns the indices of candidates that are still in progress.
        """
        return [idx for idx, stats in self.candidates.items() if stats.n < self.max_resource]

    def top_fraction(self, fraction: float) -> list[int]:
        """
        Returns the indices of the top fraction of candidates based on average reward.
        """
        candidates = self.candidates_with_maxed_resource()

        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=lambda idx: self.candidates[idx].average_reward, reverse=True)
        k = max(1, int(fraction * len(candidates)))
        return sorted_candidates[:k]

class LevelPromoter:
    def __init__(
        self,
        min_resource: int,
        max_resource: int,
        reduction_factor: int,
        early_stopping: int,
        num_combinations: int,
        max_finished_candidates: int | None = None
    ) -> None:
        """
        min_resource: smallest resource for level 0
        max_resource: largest resource for the last level
        reduction_factor: e.g. 4
        early_stopping: s
        num_combinations: total arms
        """

        self.reduction_factor = reduction_factor
        self.max_finished_candidates = max_finished_candidates

        max_level = int(math.floor(math.log(max_resource / min_resource, reduction_factor))) - early_stopping
        self.num_levels = max_level + 1

        self.resource_per_level = [
            int(min_resource * (reduction_factor ** (i + early_stopping)))
            for i in range(self.num_levels)
        ]

        self.levels = [Level(max_resource) for max_resource in self.resource_per_level]
        # Tracks which rung each combination_idx is currently in (-1 means not in any rung)
        self.current_levels = -1 * np.ones(num_combinations, dtype=int)
        self.is_done = False

    def _promote(self, combination_idx: int) -> None:
        """
        Move combination_idx up one level, if there's capacity.
        """
        old_level_idx = self.current_levels[combination_idx]
        old_stats = self.levels[old_level_idx][combination_idx]
        new_level_idx = old_level_idx + 1
        del self.levels[old_level_idx][combination_idx]
        self.levels[new_level_idx][combination_idx] = old_stats
        self.current_levels[combination_idx] = new_level_idx

    def _init_to_zero(self, combination_idx: int) -> None:
        """
        Put combination_idx into rung 0 with an initial reward sample.
        """
        self.levels[0][combination_idx] = CandidateStats(n=0, reward=0.0)
        self.current_levels[combination_idx] = 0

    def promote_if_eligible(self, combination_idx: int) -> None:
        """
        If current rung's resource is used up then promote to the next rung.
        """
        level_idx = self.current_levels[combination_idx]

        if level_idx < 0 or level_idx >= self.num_levels - 1:
            return

        stats = self.levels[level_idx][combination_idx]
        if stats.n >= self.resource_per_level[level_idx]:
            completed_candidated = self.levels[level_idx].candidates_with_maxed_resource()
            if len(completed_candidated) >= self.reduction_factor:
                fraction = 1.0 / self.reduction_factor
                top_fraction = self.levels[level_idx].top_fraction(fraction)
                if combination_idx in top_fraction:
                    self._promote(combination_idx)

    def sample(self) -> int:
        """
        Pick a configuration (by index) to sample next. We'll do:
          1) If there's a rung that has a partially trained config (n < resource for that rung), sample it.
          2) Else, if rung 0 has capacity and there's a config never added, add it.
          3) If no more arms are available, return -1.
        """
        # 1) Check each rung in order to see if there's a candidate that hasn't used up resource yet.
        undone = {}
        for level in reversed(self.levels):
            candidates_in_progress = level.candidates_in_progress()
            for arm in candidates_in_progress:
                undone[arm] = level[arm]

            candidates_done = level.candidates_with_maxed_resource()
            candidates_done_sorted = sorted(candidates_done, key=lambda idx: level[idx].average_reward, reverse=True)
            if len(candidates_done_sorted) > 0:
                self.promote_if_eligible(candidates_done_sorted[0])

        if len(undone) > 0:
            return ucb1_sampler(list(undone.keys()), list(undone.values()))
            
        # 2) If no rung has undone arms, try to add new arms to rung 0 if there's capacity
        unassigned = np.where(self.current_levels == -1)[0]
        if len(unassigned) > 0:
            combo_idx = np.random.choice(unassigned)
            self._init_to_zero(combo_idx)
            return combo_idx
        else:
            self.is_done = True
            return -1

    def update(self, combination_idx: int, reward: float) -> None:
        cur_lvl = self.current_levels[combination_idx]
        self.levels[cur_lvl][combination_idx].update(reward)
        self.promote_if_eligible(combination_idx)

    def is_done_processing(self) -> bool:
        if self.max_finished_candidates is None:
            return self.is_done
        return self.is_done or len(self.levels[-1].candidates_with_maxed_resource()) >= self.max_finished_candidates

class RandomAsynchronousSuccessiveHalvingAlgorithm(BaseOpt):
    def __init__(self, all_combinations: np.ndarray, **kwargs) -> None:
        super().__init__(all_combinations)
        self.reduction_factor = 4
        self.min_resource_r = 4
        self.max_resource_r = 2048
        self.early_stopping_rate = 0

        self.promoter = LevelPromoter(
            min_resource=self.min_resource_r,
            max_resource=self.max_resource_r,
            reduction_factor=self.reduction_factor,
            early_stopping=self.early_stopping_rate,
            num_combinations=self.num_arms
        )
        self.cached_best = None

    def sample(self) -> tuple[int, np.ndarray]:
        idx = self.promoter.sample()
        if self.promoter.is_done_processing():
            if self.cached_best is None:
                self.cached_best = self.best_known_combination()
            return self.cached_best
        return idx, self.all_combinations[idx]
    
    def update(self, combination_idx: int, reward: float) -> None:
        self.promoter.update(combination_idx, reward)

    def best_known_combination(self) -> tuple[int, np.ndarray]:
        """
        Among all rungs, returns the single best config's index and hyperparameter array 
        (based on highest average reward found so far).
        """
        for level in reversed(self.promoter.levels):
            candidates_won = level.candidates_with_maxed_resource()
            if not candidates_won:
                continue

            best_idx = max(candidates_won, key=lambda idx: level[idx].average_reward)
            return best_idx, self.all_combinations[best_idx]

        return 0, self.all_combinations[0]
