import numpy as np
from itertools import product
from tqdm.auto import trange
from typing import Literal
from algos.rasha import RandomAsynchronousSuccessiveHalvingAlgorithm
from algos.thompson_sampler import NormalInverseGammaThompsonSampler, TopTwoNormalInverseGammaThompsonSampler
from algos.factorized_bayesian import FactorizedThompsonSampler
from algos.popsearch import PopulationBasedSearch
from algos.ucb1 import UCB1Sampler
from rich.table import Table
from rich.console import Console

def all_hparam_combinations(hparam_dict: dict[str, list[int | float]]) -> np.ndarray:
    keys = list(hparam_dict.keys())
    combos = list(product(*(hparam_dict[k] for k in keys)))
    return np.array(combos, dtype=float)  # convert to float for kernel computations


class Config:
    hyperparameters = {
        "var1": [7, 14, 30, 90, 120, 360],
        "var2": [0.1, 0.15, 0.2, 0.3, 0.49],
        "var3": [0, 1, 2, 4, 8],
        "var4": [0, 1, 2, 4, 8],
        "var5": [0, 1, 2, 4, 8],
        "var6": [0, 1, 2, 4, 8],
    }
    num_items_shown: int = 25
    num_samples: list[int] = [100, 1000, 5000, 10000, 50000, 100000, 500000]
    click_prob: float = np.random.uniform(0.05, 0.25)
    thread_click_rate: float = np.random.uniform(0.05, 0.25)
    reward_strategy: Literal["real_world", "uniform"] = "real_world"


    def __init__(self, num_params: int = -1, reward_strategy: Literal["real_world", "uniform"] = "real_world"):
        self.reward_strategy = reward_strategy
        num_params = len(self.hyperparameters) if num_params == -1 else num_params
        self.hyperparameters = {k: self.hyperparameters[k] for k in list(self.hyperparameters.keys())[:num_params]}

    @property
    def highest_possible_reward(self) -> float:
        return sum(1/(i+1) for i in range(self.num_items_shown)) * self.thread_click_rate

    @property
    def num_arms(self) -> int:
        return int(np.prod([len(v) for v in self.hyperparameters.values()]))
    
    def compute_rewards(self, hparam_combinations: np.ndarray) -> np.ndarray:
        if self.reward_strategy == "real_world":
            return self._compute_real_world_rewards(hparam_combinations)
        else:
            return self._compute_uniform_rewards(hparam_combinations)

    def _compute_real_world_rewards(self, hparam_combinations: np.ndarray) -> np.ndarray:
        best_arm_idx = [np.random.randint(0, len(v)) for v in self.hyperparameters.values()]
        best_values = np.array([v[best_arm_idx[i]] for i, v in enumerate(self.hyperparameters.values())])
        stdevs = np.array([1.0, 0.05, 1.0, 1.0, 1.0, 1.0], dtype=float)[:len(self.hyperparameters.keys())]
        rho = np.array([
            [1.0,  0.8,  0.0,  0.0,  0.0,  0.0],
            [0.8,  1.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  1.0,  0.8,  0.5],
            [0.0,  0.0,  0.0,  0.8,  1.0,  0.5],
            [0.0,  0.0,  0.0,  0.5,  0.5,  1.0],
        ])[:len(self.hyperparameters.keys()), :len(self.hyperparameters.keys())]
        Sigma = np.outer(stdevs, stdevs) * rho
        Sigma_inv = np.linalg.inv(Sigma)
        mean_values = np.array([np.array(v).mean() for v in self.hyperparameters.values()])
        all_raw = np.array([np.exp(-0.01 * diff @ Sigma_inv @ diff) for combo in hparam_combinations if (diff := (combo - best_values) / mean_values) is not None])
        frac_low = (all_raw <= 0.05).sum() / len(all_raw)
        assert frac_low  <= 0.75, f"Too many low scores {frac_low}"
        max_raw = np.max(all_raw)
        # Scale so the best combination gets "highest_val"
        scaled_rewards = (all_raw / max_raw) * self.highest_possible_reward
        return scaled_rewards

    def _compute_uniform_rewards(self, hparam_combinations: np.ndarray) -> np.ndarray:
        return np.random.uniform(0, self.highest_possible_reward, size=len(hparam_combinations))

def test():
    conf = Config(num_params=4, reward_strategy="uniform")

    hparam_combinations = all_hparam_combinations(conf.hyperparameters)
    num_arms = conf.num_arms
    print(f"Number of arms: {num_arms}")

    reward_means = conf.compute_rewards(hparam_combinations)

    real_best_arm = np.argmax(reward_means)
    ranks = np.argsort(reward_means)[::-1]
    print(f"Real known best arm: {real_best_arm}")
    print(f"Best combination: {hparam_combinations[real_best_arm]}")
    print(f"CTR: {conf.click_prob:.4f}")
    print(f"Thread CTR: {conf.thread_click_rate:.4f}")

    algorithms_to_test = [
        PopulationBasedSearch,
        RandomAsynchronousSuccessiveHalvingAlgorithm,
        UCB1Sampler,
        FactorizedThompsonSampler,
        NormalInverseGammaThompsonSampler,
        TopTwoNormalInverseGammaThompsonSampler
    ]

    # Test performance of different algorithms
    for algorithm_cls in algorithms_to_test:
        headers = ["#samples", "Predicted Best arm", "Rank of predicted best arm", "Top %ile", "Total Regret"]
        table = Table(*headers, title=f"{algorithm_cls.__name__}", show_lines=True)
        for num_samples in conf.num_samples:
            hyper_param_tuner = algorithm_cls(hparam_combinations, param_values=list(conf.hyperparameters.values()))
            regret = 0
            for _ in trange(num_samples, leave=False):
                idx_to_try, _ = hyper_param_tuner.sample()
                would_click = int(np.random.random() < conf.click_prob)
                current_reward = max(np.random.normal(reward_means[idx_to_try], 1), 0) * would_click
                best_reward = max(np.random.normal(reward_means[real_best_arm], 1), 0) * would_click
                hyper_param_tuner.update(idx_to_try, current_reward)
                regret += (best_reward - current_reward)
            predicted_best_arm = hyper_param_tuner.best_known_combination()[0]
            best_arm_rank_index = np.where(ranks == predicted_best_arm)[0][0] + 1

            table.add_row(str(num_samples), str(predicted_best_arm),
                          str(best_arm_rank_index), f"{best_arm_rank_index / num_arms * 100:.4f}",
                          f"{regret:.4f}")
            if predicted_best_arm == real_best_arm:
                break
        console = Console()
        console.log(table)

if __name__ == "__main__":
    test()