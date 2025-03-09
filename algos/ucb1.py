import numpy as np
from algos.base_opt import BaseOpt

class UCB1Sampler(BaseOpt):
    def __init__(self, all_combinations: np.ndarray, **kwargs):
        super().__init__(all_combinations)
        self.rng = np.random.default_rng()
        self.reward_totals = np.zeros(self.all_combinations.shape[0])
        self.trials = np.zeros(self.all_combinations.shape[0])

    def sample(self) -> tuple[int, np.ndarray]:
        """
        Choose the combination with the highest UCB1 score.
        """
        never_tried = np.where(self.trials == 0)[0]
        if len(never_tried) > 0:
            return self.rng.choice(never_tried), self.all_combinations[never_tried]
        
        n = self.trials.sum()
        ucb1_scores = (self.reward_totals / self.trials) + np.sqrt(2 * np.log(n) / self.trials)
        best_idx = np.argmax(ucb1_scores)
        return best_idx, self.all_combinations[best_idx]

    def update(self, combination_idx: int, reward: float) -> None:
        self.reward_totals[combination_idx] += reward
        self.trials[combination_idx] += 1

    def best_known_combination(self) -> tuple[int, np.ndarray]:
        best_idx = np.argmax(self.reward_totals)
        return best_idx, self.all_combinations[best_idx]