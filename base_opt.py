import numpy as np

class BaseOpt:

    def __init__(self, all_combinations: np.ndarray):
        self.all_combinations = all_combinations
        self.num_arms = len(all_combinations)

    def sample(self) -> tuple[int, np.ndarray]:
        raise NotImplementedError
    
    def update(self, combination_idx: int, reward: float) -> None:
        raise NotImplementedError
    
    def best_known_combination(self) -> tuple[int, np.ndarray]:
        raise NotImplementedError