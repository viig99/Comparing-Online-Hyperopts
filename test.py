import numpy as np
from itertools import product
from tqdm.auto import trange
from rasha import RandomAsynchronousSuccessiveHalvingAlgorithm
from thompson_sampler import NormalInverseGammaThompsonSampler
from factorized_bayesian import FactorizedThompsonSampler

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

    @property
    def highest_possible_reward(self) -> float:
        return sum(1/(i+1) for i in range(self.num_items_shown))

    @property
    def num_arms(self) -> int:
        return int(np.prod([len(v) for v in self.hyperparameters.values()]))

    def compute_rewards(self, hparam_combinations: np.ndarray) -> np.ndarray:
        best_arm_idx = [np.random.randint(0, len(v)) for v in self.hyperparameters.values()]
        best_values = np.array([v[best_arm_idx[i]] for i, v in enumerate(self.hyperparameters.values())])
        stdevs = np.array([1.0, 0.05, 1.0, 1.0, 1.0, 1.0], dtype=float)
        rho = np.array([
            [1.0,  0.8,  0.0,  0.0,  0.0,  0.0],
            [0.8,  1.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  1.0,  0.8,  0.5],
            [0.0,  0.0,  0.0,  0.8,  1.0,  0.5],
            [0.0,  0.0,  0.0,  0.5,  0.5,  1.0],
        ])
        Sigma = np.outer(stdevs, stdevs) * rho
        Sigma_inv = np.linalg.inv(Sigma)
        all_raw = []
        for combo in hparam_combinations:
            diff = combo - best_values
            raw_score = 1 / (1 + (diff @ Sigma_inv @ diff))
            all_raw.append(raw_score)
        all_raw = np.array(all_raw)
        max_raw = np.max(all_raw)
        # Scale so the best combination gets "highest_val"
        scaled_rewards = (all_raw / max_raw) * self.highest_possible_reward
        return scaled_rewards

    def compute_uniform_rewards(self, hparam_combinations: np.ndarray) -> np.ndarray:
        return np.random.uniform(0, self.highest_possible_reward, size=len(hparam_combinations))

def test():
    conf = Config()

    hparam_combinations = all_hparam_combinations(conf.hyperparameters)
    num_arms = conf.num_arms
    print(f"Number of arms: {num_arms}")

    reward_means = conf.compute_rewards(hparam_combinations)

    real_best_arm = np.argmax(reward_means)
    ranks = np.argsort(reward_means)[::-1]
    print(f"Real known best arm: {real_best_arm}")
    print(f"Best combination: {hparam_combinations[real_best_arm]}")

    algorithms_to_test = [
        RandomAsynchronousSuccessiveHalvingAlgorithm,
        FactorizedThompsonSampler,
        NormalInverseGammaThompsonSampler
    ]

    # Test performance of different algorithms
    for algorithm_cls in algorithms_to_test:
        print(f"Testing {algorithm_cls.__name__}")
        for num_samples in conf.num_samples:
            hyper_param_tuner = algorithm_cls(hparam_combinations, param_values=list(conf.hyperparameters.values()))
            for _ in trange(num_samples, leave=False):
                idx_to_try, _ = hyper_param_tuner.sample()
                current_reward = np.random.normal(reward_means[idx_to_try], 1)
                hyper_param_tuner.update(idx_to_try, current_reward)
            predicted_best_arm = hyper_param_tuner.best_known_combination()[0]
            best_arm_rank_index = np.where(ranks == predicted_best_arm)[0][0] + 1
            print(f"#samples: {num_samples}, Predicted Best arm: {predicted_best_arm}, Rank of predicted best arm: {best_arm_rank_index}")

if __name__ == "__main__":
    test()