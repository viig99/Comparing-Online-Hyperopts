import numpy as np
from algos.base_opt import BaseOpt
from algos.ttts import TopTwoThomsonSampler

class FactorizedThompsonSampler(BaseOpt):
    """
    A factorized Thompson sampler for hyper-parameters:
      - We have 'n_params' hyper-parameters.
      - Each hyper-parameter i can take on one of 'len(param_values[i])' discrete values.
      - We maintain a Normal-Inverse-Gamma posterior for each (param_i, value_j) pair.

    On each step:
      1. For each hyper-parameter i, we sample from each of its possible values' posterior
         and pick the value with the largest sampled mean for that parameter.
      2. That yields a full configuration.
      3. We evaluate the configuration's reward (train & validate model).
      4. We update the posterior for each (param_i, chosen_value) with the observed reward.

    NOTE: This assumes reward is reasonably approximated by a Normal distribution
    with unknown mean and variance. If your reward is in [0,1], consider Beta
    or Beta-Binomial approaches instead.
    """

    def __init__(self, all_combinations: np.ndarray, **kwargs):
        """
        param_values: list of lists
            param_values[i] is a list of discrete values for hyper-parameter i.
        """
        super().__init__(all_combinations)

        self.rng = np.random.default_rng()
        
        # For each param i, for each discrete choice j, maintain posterior parameters:
        #   mu0[i][j], lambda_[i][j], alpha[i][j], beta[i][j].
        # We'll initialize them with some "vague" or small-informative hyperparameters.
        self.mu0 = []
        self.lambda_ = []
        self.alpha = []
        self.beta = []

        self.n_params = self.all_combinations.shape[1]
        self.param_values = kwargs["param_values"]
        
        for i in range(self.n_params):
            n_vals = len(self.param_values[i])
            self.mu0.append(np.zeros(n_vals))        # prior mean
            self.lambda_.append(np.ones(n_vals))     # precision factor for mu
            self.alpha.append(np.ones(n_vals))       # shape for Inverse Gamma
            self.beta.append(np.ones(n_vals))        # scale for Inverse Gamma

    def sample(self) -> tuple[int, np.ndarray]:
        """
        For each hyper-parameter i:
          - sample from each discrete value's posterior for that param
          - pick the value that yields the largest sampled mean
        Return the resulting configuration (list of chosen param values),
        along with the indices chosen for each param's value.
        """
        chosen_indices = []
        
        for i in range(self.n_params):
            tau = self.rng.gamma(self.alpha[i], 1.0 / self.beta[i])
            sigma2 = 1.0 / tau
            mu_samp = self.rng.normal(
                loc=self.mu0[i],
                scale=np.sqrt(sigma2 / self.lambda_[i])
            )
            chosen_indices.append(int(np.argmax(mu_samp)))

        combination_idx = np.ravel_multi_index(chosen_indices, dims=[len(vals) for vals in self.param_values])
        return int(combination_idx), self.all_combinations[combination_idx]

    def update(self, combination_idx: int, reward: float) -> None:
        """
        Update the posterior for each (param_i, chosen_value_j) that was used
        to form the last configuration, based on the observed reward.
        """

        chosen_indices = np.unravel_index(combination_idx, shape=[len(vals) for vals in self.param_values])

        for i in range(self.n_params):
            j = chosen_indices[i]
            
            mu0_old = self.mu0[i][j]
            lambda_old = self.lambda_[i][j]
            alpha_old = self.alpha[i][j]
            beta_old = self.beta[i][j]
            
            # Posterior updates for Normal-Inverse-Gamma with one new data point:
            #   mu0' = (lambda_ * mu0 + x) / (lambda_ + 1)
            #   lambda_' = lambda_ + 1
            #   alpha' = alpha + 1/2
            #   beta'  = beta + 0.5 * (x - mu0)^2 * (lambda_/(lambda_+1))
            self.mu0[i][j] = (lambda_old * mu0_old + reward) / (lambda_old + 1.0)
            self.lambda_[i][j] = lambda_old + 1.0
            self.alpha[i][j] = alpha_old + 0.5
            
            B = 0.5 * (reward - mu0_old)**2 * (lambda_old / (lambda_old + 1.0))
            self.beta[i][j] = beta_old + B

    def best_known_combination(self) -> tuple[int, np.ndarray]:
        """
        Returns the posterior mean estimate of each param-value combination's
        mean reward. This is just mu0[i][j] under the factorized model.
        """
        best_indices = [np.argmax(self.mu0[i]) for i in range(self.n_params)]
        combination_idx = int(np.ravel_multi_index(best_indices, dims=[len(vals) for vals in self.param_values]))
        return combination_idx, self.all_combinations[combination_idx]

class TopTwoFactorizedThompsonSampler(TopTwoThomsonSampler):
    def __init__(self, all_combinations: np.ndarray, **kwargs):
        super().__init__(FactorizedThompsonSampler, all_combinations, **kwargs)