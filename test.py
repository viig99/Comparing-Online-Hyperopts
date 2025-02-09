import numpy as np
from itertools import product
from tqdm.auto import trange
from rasha import RandomAsynchronousSuccessiveHalvingAlgorithm
from thompson_sampler import NormalInverseGammaThompsonSampler

def all_hparam_combinations(hparam_dict: dict[str, list[int | float]]) -> np.ndarray:
    keys = list(hparam_dict.keys())
    combos = list(product(*(hparam_dict[k] for k in keys)))
    return np.array(combos, dtype=float)  # convert to float for kernel computations


class Config:
    hyperparameters = {
        "maxAge": [7, 14, 30, 90, 120, 360],
        "halfAge": [0.1, 0.15, 0.2, 0.3, 0.49],
        "replyWeight": [0, 1, 2, 4, 8],
        "reactionWeight": [0, 1, 2, 4, 8],
        "followWeight": [0, 1, 2, 4, 8],
        "articleBoost": [0, 1, 2, 4, 8],
    }
    num_items_shown: int = 25
    num_samples: list[int] = [100, 1000, 5000, 10000, 50000, 100000, 500000]

def test():
    conf = Config()

    hparam_combinations = all_hparam_combinations(conf.hyperparameters)
    num_arms = hparam_combinations.shape[0]
    print(f"Number of arms: {num_arms}")
    highest_possible_reward = sum(1/(i+1) for i in range(conf.num_items_shown))

    reward_means = np.random.rand(num_arms) * highest_possible_reward

    real_best_arm = np.argmax(reward_means)
    ranks = np.argsort(reward_means)[::-1]
    print(f"Real known best arm: {real_best_arm}")
    print(f"Best combination: {hparam_combinations[real_best_arm]}")

    algorithms_to_test = [
        RandomAsynchronousSuccessiveHalvingAlgorithm,
        NormalInverseGammaThompsonSampler
    ]

    # Test performance of different algorithms
    for algorithm_cls in algorithms_to_test:
        print(f"Testing {algorithm_cls.__name__}")
        for num_samples in conf.num_samples:
            hyper_param_tuner = RandomAsynchronousSuccessiveHalvingAlgorithm(hparam_combinations)
            for _ in trange(num_samples, leave=False):
                idx_to_try, _ = hyper_param_tuner.sample()
                current_reward = np.random.normal(reward_means[idx_to_try], 1)
                hyper_param_tuner.update(idx_to_try, current_reward)
            predicted_best_arm = hyper_param_tuner.best_known_combination()[0]
            best_arm_rank_index = np.where(ranks == predicted_best_arm)[0][0] + 1
            print(f"#samples: {num_samples}, Predicted Best arm: {predicted_best_arm}, Rank of predicted best arm: {best_arm_rank_index}")

if __name__ == "__main__":
    test()