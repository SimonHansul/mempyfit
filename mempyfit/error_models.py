from multipledispatch import dispatch
import numpy as np

@dispatch(np.ndarray, np.ndarray)
def sumofsquares(sim, obs):
    """
    Sum of squares calculation for numpy arrays.
    We assume the last column to contain the response variable.
    """
    return np.sum((sim[:,-1] - obs[:,-1])**2)

@dispatch(np.ndarray, np.ndarray)
def negloglike(sim, obs, k):
    """
    Negative log-likelihood for numpy arrays, using an unbiased estimate of the residual variance.
    We assume the last column to contain the response variable.

    References

    https://github.com/cvasi-tktd/cvasi/blob/main/R/lik_profile.R
    """

    SSE = sumofsquares(sim[:,-1], obs[:,-1])
    sigma = np.sqrt(SSE / (n - k))
    n = get_n(obs)
    sigma_unbiased = sigma * np.sqrt((n - k) / n)
    return sum(n.log(norm.pdf(obs[:,-1], mean = pred[:,-1], scale = sigma_unbiased)))
