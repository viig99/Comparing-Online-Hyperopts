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

    def average_reward(self) -> float:
        return self.reward / self.n
    
def ucb1_sampler(keys: list[int], candidates: list[CandidateStats]) -> int:
    unexplored = [k for k, c in zip(keys, candidates) if c.n == 0]
    if unexplored:
        return np.random.choice(unexplored)

    total_n = sum(c.n for c in candidates)
    ucb_values = [c.average_reward() + math.sqrt(2 * math.log(total_n) / c.n) for c in candidates]
    return keys[np.argmax(ucb_values)]

class Level:
    """
    A rung in the halving procedure.
    'capacity' indicates how many distinct configs we allow at this rung.
    """
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.candidates: dict[int, CandidateStats] = {}

    def __setitem__(self, combination_idx: int, candidate_stats: CandidateStats) -> None:
        if len(self.candidates) < self.capacity:
            self.candidates[combination_idx] = candidate_stats
        else:
            raise ValueError("Level is full (capacity={})".format(self.capacity))
        
    def __getitem__(self, combination_idx: int) -> CandidateStats:
        return self.candidates[combination_idx]
    
    def __delitem__(self, combination_idx: int) -> None:
        del self.candidates[combination_idx]
    
    def __len__(self) -> int:
        return len(self.candidates)
    
    def has_capacity(self) -> bool:
        return len(self.candidates) < self.capacity

    def top_kth_reward(self, k: int) -> float:
        if len(self.candidates) == 0:
            return float('inf')

        # Sort by descending average reward
        sorted_candidates = sorted(self.candidates.items(),
                                   key=lambda x: x[1].average_reward(),
                                   reverse=True)
        k = max(1, min(k, len(sorted_candidates)))
        return sorted_candidates[k-1][1].average_reward()

    def best_candidate(self) -> int:
        keys, values = zip(*self.candidates.items())
        return keys[np.argmax([v.average_reward() for v in values])]

class LevelPromoter:
    def __init__(
        self,
        min_resource: int,
        max_resource: int,
        reduction_factor: int,
        early_stopping: int,
        num_combinations: int
    ) -> None:
        """
        min_resource: smallest resource for level 0
        max_resource: largest resource for the last level
        reduction_factor: e.g. 4
        early_stopping: s
        num_combinations: total arms
        """

        self.reduction_factor = reduction_factor

        max_level = int(math.floor(math.log(max_resource / min_resource, reduction_factor))) - early_stopping
        self.num_levels = max_level + 1

        self.resource_per_level = [
            int(min_resource * (reduction_factor ** (i + early_stopping)))
            for i in range(self.num_levels)
        ]

        # For rung i, capacity = int( num_combinations * (capacity_factor^(i+1)) ), clamped to at least 4
        self.level_capacity = [
            max(4, int(num_combinations * (1/reduction_factor ** (i + 1))))
            for i in range(self.num_levels)
        ]

        self.levels = [Level(cap) for cap in self.level_capacity]
        # Tracks which rung each combination_idx is currently in (-1 means not in any rung)
        self.current_levels = -1 * np.ones(num_combinations, dtype=int)

    def _promote(self, combination_idx: int) -> None:
        """
        Move combination_idx up one level, if there's capacity.
        """
        old_level_idx = self.current_levels[combination_idx]
        old_stats = self.levels[old_level_idx][combination_idx]

        new_level_idx = old_level_idx + 1
        if old_level_idx < self.num_levels - 1 and self.levels[new_level_idx].has_capacity():
            del self.levels[old_level_idx][combination_idx]
            self.levels[new_level_idx][combination_idx] = old_stats
            self.current_levels[combination_idx] = new_level_idx

    def _init_to_zero(self, combination_idx: int) -> None:
        """
        Put combination_idx into rung 0 with an initial reward sample.
        """
        if not self.levels[0].has_capacity():
            return
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
            rung_size = len(self.levels[level_idx])
            # k = max(1, self.level_capacity[level_idx] // self.reduction_factor)
            k = max(1, rung_size // self.reduction_factor)

            threshold = self.levels[level_idx].top_kth_reward(k)
            if rung_size >= k and stats.average_reward() >= threshold and self.levels[level_idx+1].has_capacity():
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
        for level_idx, level in enumerate(self.levels):
            for arm, st in level.candidates.items():
                if st.n < self.resource_per_level[level_idx]:
                    undone[arm] = st

        if len(undone) > 0:
            return ucb1_sampler(list(undone.keys()), list(undone.values()))
            
        # 2) If no rung has undone arms, try to add new arms to rung 0 if there's capacity
        level0 = self.levels[0]
        if level0.has_capacity():
            unassigned = np.where(self.current_levels == -1)[0]
            combo_idx = np.random.choice(unassigned)
            self._init_to_zero(combo_idx)
            return combo_idx
        else:
            return -1

    def update(self, combination_idx: int, reward: float) -> None:
        cur_lvl = self.current_levels[combination_idx]
        self.levels[cur_lvl][combination_idx].update(reward)
        self.promote_if_eligible(combination_idx)

class RandomAsynchronousSuccessiveHalvingAlgorithm(BaseOpt):
    def __init__(self, all_combinations: np.ndarray, **kwargs) -> None:
        super().__init__(all_combinations)
        self.reduction_factor = 2
        self.min_resource_r = 32
        self.max_resource_r = 4096
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
        if idx == -1:
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
        best_idx = 0
        best_combo = self.all_combinations[0]
        best_score = float('-inf')

        # Start from top rung down
        for level in reversed(self.promoter.levels):
            if len(level) > 0:
                candidate_idx = level.best_candidate()
                if candidate_idx != -1:
                    candidate_avg_reward = level[candidate_idx].average_reward()
                    if candidate_avg_reward > best_score:
                        best_score = candidate_avg_reward
                        best_idx = candidate_idx
                        best_combo = self.all_combinations[candidate_idx]
        return best_idx, best_combo