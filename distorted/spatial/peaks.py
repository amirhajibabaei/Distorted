import typing

import numpy as np
from ase.geometry import Cell, get_distances
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize


def find_peaks(
    f3d: np.ndarray,
    w: int,
    *,
    threshold: float = 0,
    dxdydz: float | tuple[float, float, float] = 1,
    shift: float | tuple[float, float, float] = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in a periodic 3D function discretized on a regular grid such that:
        f3d[i, j, k] = f(i*dx+shift[0], j*dy+shift[1], k*dz+shift[2])

    A shift in the range (-1, 1) can be applied to adjust the position of the peak.
    The two common cases are shift=0 and shift=0.5:
        1. If the function is calculated by a histogram, f3d represents the
           the values at the centre of the grid and therefore shift=0.5.
        2. If the function is directly calculated on the grid, then shift=0.

    :param f3d: 3D array
    :param w: half of the window size
    :param threshold: minimum value of the peak
    :param dxdydz: voxel size
    :param shift: shift of the center of the voxel
    :return: positions, values
    """
    coords = []
    values = []
    for (i, j, k), window in _iter3d(f3d, w):
        cond = window > window[w, w, w]
        cond[w, w, w] = False
        if cond.any():
            continue
        if window[w, w, w] < threshold:
            continue
        correction, value = _interpolated_peak(window, w)
        coords.append((i, j, k) + correction)
        values.append(value)
    positions = np.array(coords) + shift
    peaks = np.array(values)
    positions, peaks = _merge_close_peaks(positions, peaks, w, f3d.shape)
    return positions * dxdydz, peaks


def _get_range(w: int) -> np.ndarray:
    """
    Get range of size 2w+1.
    :param w: half of the window size
    :return: range
    """
    assert w > 0
    return np.array(range(-w, w + 1))


def _iter3d(
    f3d: np.ndarray, w: int
) -> typing.Iterable[tuple[tuple[int, int, int], np.ndarray]]:
    """
    Iterate over 3D array with a window of size 2w+1.
    :param f3d: 3D array
    :param w: half of the window size
    :return: (i, j, k), window
    """
    assert f3d.ndim == 3
    s = _get_range(w)
    for i, j, k in np.ndindex(f3d.shape):
        window = (
            f3d.take(i + s, axis=0, mode="wrap")
            .take(j + s, axis=1, mode="wrap")
            .take(k + s, axis=2, mode="wrap")
        )
        yield (i, j, k), window


def _interpolated_peak(
    f3d: np.ndarray,
    w: int,
    method: str = "cubic",
) -> tuple[np.ndarray, float]:
    range_ = _get_range(w)
    grid = 3 * (range_,)
    interpolator = RegularGridInterpolator(grid, -f3d, method=method)
    x0 = np.zeros(3)
    res = minimize(
        interpolator,
        x0,
    )
    assert res.success
    # The interpolated maximum should be near the center of the window
    assert abs(res.x).max() < 1.5
    return res.x, -res.fun


def _merge_close_peaks(
    positions: np.ndarray,
    peaks: np.ndarray,
    distance_threshold: float,
    wrap: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge peaks that are closer than the threshold.
    :param positions: positions of the peaks
    :param peaks: values of the peaks
    :param distance_threshold: minimum distance between peaks
    :param wrap: wrap distances (pbc)
    :return: positions, values
    """
    cell = Cell(np.diag(wrap))
    _, dm = get_distances(p1=positions, cell=cell, pbc=True)
    # if a few peaks are very close, we merge them
    # by taking the weighted average of their positions
    # and values
    clusters = np.zeros(len(positions), dtype=int)
    c = 0
    for i in range(len(positions)):
        if clusters[i] != 0:
            continue
        c += 1
        j = np.argwhere(dm[i] < distance_threshold).flatten()
        clusters[j] = c
    merged = []
    for c in range(1, clusters.max() + 1):
        mask = clusters == c
        weights = peaks[mask] / peaks[mask].sum()
        merged.append(
            (
                (positions[mask] * weights[:, None]).mean(axis=0),
                peaks[mask].mean(),
            )
        )
    positions_, peaks_ = zip(*merged)
    return np.array(positions_), np.array(peaks_)


def test_find_peaks():
    # Note that the following does not work:
    # bins = np.mgrid[0:10:51j]
    # x, y, z = np.meshgrid(bins, bins, bins)

    # We define a Gaussian peak near the center of the cell
    g = 0.2
    n = 31
    L = (n - 1) * g
    sigma = 5 * g
    a, b, c = np.random.uniform(-L * 0.1, L * 0.1, 3) + L / 2
    x, y, z = np.mgrid[0 : L : n * 1j, 0 : L : n * 1j, 0 : L : n * 1j]
    f = np.exp(-0.5 * ((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2) / sigma**2)
    pos, val = find_peaks(f, 5, dxdydz=g, shift=0)
    assert np.allclose(pos, (a, b, c), atol=0.01)


if __name__ == "__main__":
    test_find_peaks()
