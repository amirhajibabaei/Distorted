import typing

import numpy as np
from ase.atoms import Atoms
from ase.cell import Cell
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor

import distorted.spatial.util as _util
import distorted.symmetry as _sym


def get_reciprocal_space_projected_density(
    traj: typing.Sequence[Atoms],
    max_bin_width: float | tuple[float, float, float],
    data: _sym.SymmetryDataset,
    atomic_numbers: int | set[int] | None = None,
    max_cell_diff: float | None = None,
    adjust_cell: bool = True,
) -> tuple[PWDescriptor, dict[int, np.ndarray]]:
    # get the unit cell for projection
    oriented_std_cell = Cell(data.get_oriented_std_lattice())
    recon_cell = Cell(data.transformation_matrix.T @ oriented_std_cell)
    if isinstance(atomic_numbers, int):
        atomic_numbers = {atomic_numbers}
    elif atomic_numbers is None:
        atomic_numbers = set(traj[0].numbers)

    # bins and histogram
    bin_sizes = _util.get_num_bins(oriented_std_cell, max_bin_width)
    gd = GridDescriptor(bin_sizes, cell_cv=oriented_std_cell)
    pw = PWDescriptor(None, gd)  # cutoff = None implies all k-points
    kpts = pw.get_reciprocal_vectors()
    rho_k = {z: np.zeros(kpts.shape[0], dtype=np.complex128) for z in atomic_numbers}
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

            # k.r
            kr = kpts @ pos.T
            rho_k[z] += np.cos(kr).sum(axis=1) - 1j * np.sin(kr).sum(axis=1)
            c[z] += mask.sum()

    # normalization
    dv = oriented_std_cell.volume / np.prod(bin_sizes)
    for z in atomic_numbers:
        rho_k[z] /= c[z] * dv
    return pw, rho_k


def inverse(pw: PWDescriptor, rho_k: np.ndarray) -> np.ndarray:
    rho_r = pw.ifft(rho_k)
    assert np.allclose(rho_r.imag, 0)
    return rho_r.real
