import numpy as np

def minkowski_distance(v1,v2,p:int):
    """Compute the Minkowski distance of order p between two n dimensional vectors

    Args:
        v1 (np.ndarray): First vector
        v2 (np.ndarray): Second vector
        p (int): Order of the Minkowski distance, p < 1 is not a valid metric 

    Returns:
        float: Minkowski distance 
    """
    assert(p>=1)
    return np.power(np.sum(np.abs(v1-v2)**p), 1/p)

def kl_div(p, q):
    """Kullback-Leibler divergence between two probability distributions.

    Args:
        p (np.ndarray): Probability distribution
        q (np.ndarray): Probability distribution

    Returns:
        float: Kullback-Leibler divergence value
    """
    return np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)))


def js_div(p, q):
    """Jensen-Shannon divergence (nomalized and symmetric) between two probability distributions

    Args:
        p (np.ndarray): Probability distribution
        q (np.ndarray): Probability distribution

    Returns:
        float: Jensen-Shannon divergence value
    """
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p+q)/2
    return (kl_div(p, m) + kl_div(q, m))/2