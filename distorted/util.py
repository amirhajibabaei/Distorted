import numpy as np
from ase.geometry import Cell, get_distances


def minimal_distance_perm(
    x: np.ndarray, y: np.ndarray, cell: Cell = None, pbc: np.ndarray = None
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

    # Find distace matrix, taking into account periodic boundary conditions.
    # Without mic, this is equivalent to: d = y[None, :] - x[:, None]
    d, dm = get_distances(p1=x, p2=y, cell=cell, pbc=pbc)
    perm = np.argmin(dm, axis=0)
    diff = dm[perm].diagonal()

    # test there are no duplicates
    assert len(np.unique(perm)) == len(
        perm
    ), f"duplicate indices in permutation: {perm}"

    return perm, diff
