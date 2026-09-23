from multipledispatch import dispatch
import numpy as np
from numbers import Real

@dispatch(np.ndarray, np.ndarray)
def sumofsquares(sim, obs):
    """Sum of squares calculation for numpy arrays.

    Args:
        sim (np.ndarray):s Simulated values.
        obs (np.ndarray): Observed values.

    Returns:
        float: Sum of squared differences.

    Example:
        >>> sumofsquares(np.array([[0,1],[1,2]]), np.array([[0,2],[1,3]]))
    """

    return np.sum((sim[:,-1] - obs[:,-1])**2)


def nll_multinomial(sim, obs, eps=1e-12):
    """
    Negative multinomial log-likelihood for survival data.

    Parameters
    ----------
    obs : array-like, shape (n_times, n_columns)
        Last column contains observed survival counts or proportions.
        Rows must be ordered by increasing time.

    sim : array-like, shape (n_times, n_columns)
        Last column contains model-predicted survival probabilities.
        Model values must correspond to the observation times.

    eps : float
        Lower bound used only inside log() for numerical stability.

    check_times : bool
        Whether to require identical time grids.
    """

    s_obs = obs[:, -1]
    s_sim = sim[:, -1]

    # Observed death counts in each interval.
    observed_deaths = s_obs[:-1] - s_obs[1:]

    # Model probabilities for death in each interval.
    model_death_probs = s_sim[:-1] - s_sim[1:]
    # Final survivor bin: survivors after the final observation time.
    observed_tail = s_obs[-1]
    model_tail_prob = s_sim[-1]

    observed_counts = np.concatenate(
        [observed_deaths, [observed_tail]]
    )

    model_probs = np.concatenate(
        [model_death_probs, [model_tail_prob]]
    )

    # These should sum to the initial population/proportion.
    if np.any(observed_counts < 0):
        return np.inf

    if np.any(model_probs < 0):
        return np.inf

    # Use clipping only to avoid log(0); do not use it to repair
    # negative probabilities.
    log_probs = np.log(np.maximum(model_probs, eps))

    return -np.dot(observed_counts, log_probs)


@dispatch(np.ndarray, np.ndarray)
def negloglike(sim, obs, k):
    """Negative log-likelihood for numpy arrays, using an unbiased estimate of variance.

    Args:
        sim (np.ndarray): Simulated values.
        obs (np.ndarray): Observed values.
        k (Real): Number of fitted parameters in the model.

    Returns:
        float: Negative log-likelihood estimate.

    References:
        https://github.com/cvasi-tktd/cvasi/blob/main/R/lik_profile.R

    Example:
        >>> negloglike(np.array([[0,1],[1,2]]), np.array([[0,2],[1,3]]), 2)
    """

    SSE = sumofsquares(sim[:,-1], obs[:,-1])
    sigma = np.sqrt(SSE / (n - k))
    n = get_n(obs)
    sigma_unbiased = sigma * np.sqrt((n - k) / n)
    return sum(n.log(norm.pdf(obs[:,-1], mean = pred[:,-1], scale = sigma_unbiased)))

@dispatch(np.ndarray, np.ndarray, Real)
def euclidean(obs, sim, scale): 
    """Compute a Euclidean distance between observed and simulated values.

    Args:
        obs (np.ndarray): Observed values.
        sim (np.ndarray): Simulated values.
        scale (Real): Scaling factor.

    Returns:
        float: Euclidean distance.

    Example:
        >>> euclidean(np.array([1.0, 2.0]), np.array([1.1, 1.9]), 1.0)
    """
    return np.sqrt(np.sum((obs / scale - sim / scale)**2))

@dispatch(Real, Real, Real)
def euclidean(obs, sim, scale):
    """Compute Euclidean distance for scalar values.

    Args:
        obs (Real): Observed scalar.
        sim (Real): Simulated scalar.
        scale (Real): Scaling factor.

    Returns:
        float: Euclidean distance.

    Example:
        >>> euclidean(1.0, 0.8, 1.0)
    """
    return np.sqrt((obs/scale - sim/scale)**2)

