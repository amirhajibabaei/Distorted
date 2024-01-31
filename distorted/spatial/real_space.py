import typing

import numpy as np
from ase.atoms import Atoms
from ase.cell import Cell

import distorted.spatial.peaks as _peaks
import distorted.spatial.util as _util
import distorted.symmetry as _sym


def get_real_space_projected_density(
    traj: typing.Sequence[Atoms],
    max_bin_width: float | tuple[float, float, float],
    data: _sym.SymmetryDataset,
    atomic_numbers: int | set[int] | None = None,
    max_cell_diff: float | None = None,
    adjust_cell: bool = True,
) -> dict[int, np.ndarray]:
    # get the unit cell for projection
    oriented_std_cell = Cell(data.get_oriented_std_lattice())
    recon_cell = Cell(data.transformation_matrix.T @ oriented_std_cell)
    if isinstance(atomic_numbers, int):
        atomic_numbers = {atomic_numbers}
    elif atomic_numbers is None:
        atomic_numbers = set(traj[0].numbers)

    # bins and histogram
    bin_sizes = _util.get_num_bins(oriented_std_cell, max_bin_width)
    edges = [np.linspace(0, 1, n + 1) for n in bin_sizes]
    hist = {z: np.zeros(bin_sizes) for z in atomic_numbers}
    c = {z: 0 for z in atomic_numbers}

    # loop over trajectory
    for atoms in traj:
        _sym._assert_oriented(atoms)
        if max_cell_diff is not None:
            assert abs(atoms.cell - recon_cell).max() < max_cell_diff
        for z in atomic_numbers:
            mask = atoms.numbers == z
            if adjust_cell:
                pos = recon_cell.cartesian_positions(atoms.get_scaled_positions()[mask])
            else:
                pos = atoms.get_positions()[mask]
            projected_scaled_positions = oriented_std_cell.scaled_positions(pos) % 1
            hist[z] += np.histogramdd(projected_scaled_positions, bins=edges)[0]
            c[z] += mask.sum()

    # normalization
    dv = oriented_std_cell.volume / np.prod(bin_sizes)
    for z in atomic_numbers:
        hist[z] /= c[z] * dv

    return hist


def hist_to_peaks(
    cell: Cell,
    hist: np.ndarray,
    min_radius: float,
    *,
    threshold: float = 0,
    symbol: str = "X",
) -> Atoms:
    delta = min(cell.cellpar()[:3] / hist.shape)
    w = np.ceil(min_radius / delta).astype(int)
    print(f"Using {w} as the peak width.")
    coords, peaks = _peaks.find_peaks(
        hist,
        w,
        threshold=threshold,
        dxdydz=1,
        shift=0.5,
    )
    atoms = Atoms(
        [symbol] * len(coords),
        scaled_positions=coords / hist.shape,
        cell=cell,
        pbc=True,
    )
    return atoms, peaks
