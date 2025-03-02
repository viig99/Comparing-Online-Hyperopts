from collections import Counter
import heapq
import random
import numpy as np
from typing import Literal
from dataclasses import dataclass
from algos.base_opt import BaseOpt

@dataclass
class Member:
    member_id: int
    num_obs: int = 0
    sum_rewards: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.sum_rewards / self.num_obs if self.num_obs > 0 else 0.0

    def update(self, reward: float) -> None:
        self.num_obs += 1
        self.sum_rewards += reward

    def __lt__(self, other: "Member") -> bool:
        return self.mean_reward < other.mean_reward
    
    def is_ready(self, ready_at: int) -> bool:
        return self.num_obs >= ready_at
    
    def __repr__(self) -> str:
        return f"Member(member_id={self.member_id}, num_obs={self.num_obs}, mean_reward={self.mean_reward})"

class Population:

    def __init__(self, population_size: int, frac_elites: float, ready_at: int):
        self.population_size = population_size
        self.frac_elites = frac_elites
        self.ready_at = ready_at
        self.members: list[Member] = []

    def __getitem__(self, idx: int) -> Member | None:
        member = [m for m in self.members if m.member_id == idx]
        return member[0] if member else None

    def add_member(self, member_id: int) -> None:
        member = Member(member_id)
        heapq.heappush(self.members, member)
        if len(self.members) > self.population_size:
            raise ValueError("Population size exceeded")

    def remove_member(self, member_id: int) -> None:
        self.members = list(filter(lambda x: x.member_id != member_id, self.members))
        heapq.heapify(self.members)

    def select_member(self, strategy: Literal["random", "ucb1"] = "random") -> Member:
        if strategy == "random":
            return random.choice(self.members)
        elif strategy == "ucb1":
            unselected_members = list(filter(lambda x: x.num_obs == 0, self.members))
            if unselected_members:
                return random.choice(unselected_members)
            else:
                total_obs = sum(m.num_obs for m in self.members)
                return max(self.members, key=lambda x: x.mean_reward + np.sqrt(2 * np.log(total_obs) / x.num_obs))
        else:
            raise ValueError("Invalid strategy")
    
    def top_members(self, frac: float | None = None) -> list[Member]:
        frac = frac or self.frac_elites
        return heapq.nlargest(int(frac * self.population_size), self._ready_members())
    
    def bottom_members(self, frac: float | None = None) -> list[Member]:
        frac = frac or self.frac_elites
        return heapq.nsmallest(int(frac * self.population_size), self._ready_members())
    
    def _ready_members(self) -> list[Member]:
        return list(filter(lambda x: x.is_ready(self.ready_at), self.members))
    
    def best_member(self) -> Member:
        ready_members = self._ready_members()
        return max(ready_members) if ready_members else max(self.members)

class PopulationBasedSearch(BaseOpt):

    def __init__(self, all_combinations: np.ndarray, **kwargs):
        super().__init__(all_combinations)
        self.population_size = kwargs.get("population_size", 100)
        self.frac_elites = kwargs.get("frac_elites", 0.2)
        self.ready_at = kwargs.get("ready_at", 150)
        self.param_values = kwargs["param_values"]
        self.population = Population(self.population_size, self.frac_elites, self.ready_at)
        self._initialize_population()

    def _initialize_population(self) -> None:
        initial_combinations = random.sample(range(self.all_combinations.shape[0]), self.population_size)
        for combination in initial_combinations:
            self.population.add_member(combination)

    def sample(self) -> tuple[int, np.ndarray]:
        member = self.population.select_member()
        return member.member_id, self.all_combinations[member.member_id]
    
    def update(self, combination_idx: int, reward: float) -> None:
        member = self.population[combination_idx]
        if member:
            member.update(reward)
            if member.is_ready(self.ready_at):
                new_combination_idx = self._exploit_ready_member(member)
                if new_combination_idx is not None and new_combination_idx != combination_idx:
                    new_combination_idx = self._explore_new_member(new_combination_idx)
                    if new_combination_idx not in map(lambda m: m.member_id, self.population.members):
                        self.population.remove_member(combination_idx)
                        self.population.add_member(new_combination_idx)
    
    def _list_contains(self, lst: list[Member], member: Member) -> bool:
        return len(list(filter(lambda m: m.member_id == member.member_id, lst))) > 0
    
    def _exploit_ready_member(self, member: Member) -> int | None:
        top_members = self.population.top_members()
        if (
            len(self.population._ready_members()) >= int(0.5 * self.population_size) 
            and not self._list_contains(top_members, member)
            and self._list_contains(self.population.bottom_members(), member)
        ):
            if not top_members:
                return None
            return random.choice(top_members).member_id
        return member.member_id
    
    def _explore_new_member(self, member_idx: int) -> int:
        top_ready_members = self.population.top_members(0.8)
        actual_combination_idx = [self._unravel_index(m.member_id) for m in top_ready_members]
        dim_counters = [Counter([comb[i] for comb in actual_combination_idx]) for i in range(len(self.param_values))]
        member_combination_idx = self._unravel_index(member_idx)
        perturbed_combination = [self._perturb_dimension(dim_index, dim_counter, orig_idx) for dim_index, (dim_counter, orig_idx) in enumerate(zip(dim_counters, member_combination_idx))]
        return self._ravel_multi_index(perturbed_combination)
    
    def _perturb_dimension(self, dim_index: int, counter: Counter, orig_idx: int) -> int:
        if random.random() < 0.5:
            return random.choices(list(counter.keys()), weights=list(counter.values()), k=1)[0]
        return orig_idx
    
    def _weighted_sample(self, counter: Counter) -> int:
        return random.choices(list(counter.keys()), weights=list(counter.values()))[0]
    
    def _unravel_index(self, idx: int) -> list[int]:
        unraveled_indices = np.unravel_index(idx, shape=[len(vals) for vals in self.param_values])
        return list(map(int, unraveled_indices))
    
    def _ravel_multi_index(self, indices: list[int]) -> int:
        return int(np.ravel_multi_index(indices, dims=[len(vals) for vals in self.param_values]))
    
    def best_known_combination(self) -> tuple[int, np.ndarray]:
        best_member = self.population.best_member()
        return best_member.member_id, self.all_combinations[best_member.member_id]
    
    def debug_internal_state(self) -> None:
        print("="*80)
        print(f"Population count: {len(self.population.members)}")
        for member in self.population.members:
            print(member)
        print("="*80)
    
if __name__ == "__main__":

    from itertools import product
    from tqdm import trange


    def all_hparam_combinations(hparam_dict: dict[str, list[int | float]]) -> np.ndarray:
        keys = list(hparam_dict.keys())
        combos = list(product(*(hparam_dict[k] for k in keys)))
        return np.array(combos, dtype=float)  # convert to float for kernel computations

    # Test the DiscreteCMAES class
    hyperparameters = {
        "var1": [7, 14, 30, 90, 120, 360],
        "var2": [0.1, 0.15, 0.2, 0.3, 0.49],
        "var3": [0, 1, 2, 4, 8],
        "var4": [0, 1, 2, 4, 8],
        "var5": [0, 1, 2, 4, 8],
        "var6": [0, 1, 2, 4, 8],
    }
    num_samples = 50000
    all_combinations = all_hparam_combinations(hyperparameters)
    pbs = PopulationBasedSearch(all_combinations, param_values=list(hyperparameters.values()))

    reward_means = np.random.uniform(0, 5, size=len(all_combinations))
    ranks = np.argsort(reward_means)[::-1]

    for i in trange(num_samples, leave=False):
        idx_to_try, _ = pbs.sample()
        reward = np.random.normal(reward_means[idx_to_try], 1)
        pbs.update(idx_to_try, reward)
        if i % 1000 == 0:
            pbs.debug_internal_state()
    
    predicted_best_arm = pbs.best_known_combination()[0]
    best_arm_rank_index = np.where(ranks == predicted_best_arm)[0][0] + 1
    print(f"#samples: {num_samples}, Predicted Best arm: {predicted_best_arm}, Rank of predicted best arm: {best_arm_rank_index}")