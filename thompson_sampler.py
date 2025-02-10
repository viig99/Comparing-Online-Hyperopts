import numpy as np
from base_opt import BaseOpt

class NormalInverseGammaThompsonSampler(BaseOpt):
    """
    Thompson sampler for continuous rewards assumed to follow
    Normal(mu, sigma^2), with a Normal-Inverse-Gamma conjugate prior.

    Each arm i has parameters (mu0[i], lambda_[i], alpha[i], beta[i]).
    """
    def __init__(self, all_combinations: np.ndarray, **kwargs):
        super().__init__(all_combinations)
        self.rng = np.random.default_rng()
        
        # Initialize prior hyperparameters for each arm.
        # Often you'd tune these or set them to something "vague."
        # mu0 = 0, alpha = 1, beta = 1, lambda_ = 1, for example
        self.mu0 = np.zeros(self.num_arms)       # prior mean
        self.lambda_ = np.ones(self.num_arms)    # precision factor on mu
        self.alpha = np.ones(self.num_arms)      # shape for InvGamma
        self.beta = np.ones(self.num_arms)       # scale for InvGamma

    def sample(self) -> tuple[int, np.ndarray]:
        """Sample (sigma^2, mu) from each arm's posterior and pick the arm
        with the largest sampled mu."""
        # Sample sigma^2 from InvGamma(alpha, beta):
        #   If X ~ Gamma(k=alpha, theta=1/beta), then 1/X ~ InvGamma(...)
        #   (But np.random.gamma uses shape & scale.)
        # We want sigma^2 ~ InvGamma(alpha, beta).
        # This is the same as gamma with shape=alpha, scale=1/beta, then invert:
        tau = self.rng.gamma(self.alpha, 1.0 / self.beta)
        sigma2 = 1.0 / tau
        # Now sample mu ~ N(mu0, sigma^2 / lambda_):
        mu_samp = self.rng.normal(
            loc=self.mu0,
            scale=np.sqrt(sigma2 / self.lambda_)
        )
        return int(np.argmax(mu_samp)), self.all_combinations[np.argmax(mu_samp)]

    def update(self, combination_idx: int, reward: float) -> None:
        """
        Given a new observed reward x for the selected arm, update
        that arm's posterior parameters.

        Posterior update from standard Normal-Inverse-Gamma formulas 
        (incremental case, n=1):
        
          mu0'     = (lambda_ * mu0 + x) / (lambda_ + 1)
          lambda_' = lambda_ + 1
          alpha'   = alpha + 1/2
          beta'    = beta + 0.5 * (x - mu0)^2 * (lambda_ / (lambda_ + 1))
        """
        mu0_old = self.mu0[combination_idx]
        lambda_old = self.lambda_[combination_idx]
        alpha_old = self.alpha[combination_idx]
        beta_old = self.beta[combination_idx]

        # Update
        self.mu0[combination_idx] = (lambda_old * mu0_old + reward) / (lambda_old + 1.0)
        self.lambda_[combination_idx] = lambda_old + 1.0
        self.alpha[combination_idx] = alpha_old + 0.5

        # The “B” term = 0.5 * (x - mu0_old)^2 * (lambda_/(lambda_+1))
        # captures the added contribution to beta from the new data point.
        B = 0.5 * (reward - mu0_old)**2 * (lambda_old / (lambda_old + 1.0))
        self.beta[combination_idx] = beta_old + B

    def best_known_combination(self) -> tuple[int, np.ndarray]:
        """
        The posterior mean of arm i's underlying "true mean" is just mu0[i].
        (One can also incorporate a correction for uncertain sigma^2, but
         mu0[i] is the standard posterior estimate of the mean.)
        """
        return int(np.argmax(self.mu0)), self.all_combinations[np.argmax(self.mu0)]