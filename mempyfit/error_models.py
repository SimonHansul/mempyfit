from multipledispatch import dispatch
import numpy as np
from numbers import Real

@dispatch(np.ndarray, np.ndarray)
def sumofsquares(sim, obs):
    """Sum of squares calculation for numpy arrays.

    Args:
        sim (np.ndarray): Simulated values.
        obs (np.ndarray): Observed values.

    Returns:
        float: Sum of squared differences.

    Example:
        >>> sumofsquares(np.array([[0,1],[1,2]]), np.array([[0,2],[1,3]]))
    """
    return np.sum((sim[:,-1] - obs[:,-1])**2)

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