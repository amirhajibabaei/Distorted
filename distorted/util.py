import numpy as np


def minimal_distance_perm(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find a permutation of rows of x that minimizes the sum of squared
    distances to y:
        x[perm] ~ y

    They should have the same length, but the order may be different.

    The assumption is that since these are vectors to voronoi neighbors, the
    vectors in x, y are well distinct, so a simple distance matrix is sufficient
    and we don't need to search all permutations.

    Returns:
        perm: permutation of x that minimizes the sum of squared distances
        diff: distances
    """
    assert x.shape == y.shape and x.ndim == 2 and x.shape[1] == 3
    d = x[:, None] - y[None, :]
    dm = np.linalg.norm(d, axis=-1)
    perm = np.argmin(dm, axis=0)
    diff = dm[perm].diagonal()

    # test there are no duplicates
    assert len(np.unique(perm)) == len(perm)

    return perm, diff
