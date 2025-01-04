import numpy as np
from scipy.special import loggamma


def str2int(l):
    """
    Convert a list of strings to a list of integers
    """
    d = dict([(y, x + 1) for x, y in enumerate(sorted(set(l)))])
    return [d[x] for x in l]


def random_key():
    """
    Generate a random key
    """
    return str(np.random.randint(0, 100000000000))


def logNcK(n, K):
    """
    Compute the log of the binomial coefficient N choose K
    """
    return loggamma(n + 1) - loggamma(n - K + 1) - loggamma(K + 1)


def logNmcK(n, K):
    """
    Compute the log of the multiset coefficient N multiset K
    """
    return loggamma(n + K) - loggamma(K + 1) - loggamma(n)


def logMult(N, ns):
    """
    Compute the log of the multinomial coefficient N multichoose ns
    """
    return loggamma(N + 1) - sum(loggamma(i + 1) for i in ns)


def effective_number_of_rankings(clusters: dict) -> float:
    """
    Compute the effective number of rankings given a dictionary of clusters

    Parameters
    ----------
    clusters : dict
        Dictionary of clusters

    Returns
    -------
    R_eff : float
        Effective number of rankings
    """

    N = np.sum([len(cluster) for cluster in clusters.values()])
    R_eff = 0
    for cluster in clusters.values():
        R_eff -= (len(cluster) / N) * np.log(len(cluster) / N)

    return np.exp(R_eff)
