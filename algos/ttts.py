import numpy as np
from typing import Type
from algos.base_opt import BaseOpt

class TopTwoThomsonSampler:
    """
    Based on the Top-Two Thompson Sampling algorithm from
    "Simple Bayesian Algorithms for Best-Arm Identification" by Daniel Russo, 2018.
    Arxiv: https://arxiv.org/pdf/1602.08448
    
    Algorithm mentioned in section 3.4 of the paper.

    Primarily allows for better exploration, sampling efficiency by not letting the
    same arm be selected twice in a row.
    """
    def __init__(self, sampler: Type[BaseOpt], all_combinations: np.ndarray, **kwargs):
        self.rng = np.random.default_rng()
        self.beta = 0.5
        self.sampler = sampler(all_combinations, **kwargs)

    def sample(self) -> tuple[int, np.ndarray]:

        I_idx, I_values = self.sampler.sample()

        if self.rng.random() < self.beta:
            return I_idx, I_values
        else:
            for _ in range(100):
                J_idx, J_values = self.sampler.sample()
                if J_idx != I_idx:
                    return J_idx, J_values
            return I_idx, I_values

    def update(self, combination_idx: int, reward: float) -> None:
        self.sampler.update(combination_idx, reward)

    def best_known_combination(self) -> tuple[int, np.ndarray]:
        return self.sampler.best_known_combination()
