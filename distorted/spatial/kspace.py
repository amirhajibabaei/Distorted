import typing

import numpy as np
from ase.atoms import Atoms
from ase.cell import Cell

try:
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.pw.descriptor import PWDescriptor

    _has_gpaw = True
except ImportError:
    _has_gpaw = False


import distorted.spatial.util as _util
import distorted.symmetry as _sym


def get_hkl(
    traj: typing.Sequence[Atoms],
    q: np.ndarray,  # q points [nq, 3]
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

    # histogram
    q = np.array(q)
    assert q.shape[1] == 3
    s_q: dict[int, np.ndarray] = {
        z: np.zeros((len(traj), q.shape[0]), dtype=np.complex128)
        for z in atomic_numbers
    }

    # loop over trajectory
    for i, atoms in enumerate(traj):
        _sym._assert_standard(atoms)
        if max_cell_diff is not None:
            assert abs(atoms.cell - recon_cell).max() < max_cell_diff
        for z in atomic_numbers:
            mask = atoms.numbers == z
            if adjust_cell:
                pos = recon_cell.cartesian_positions(atoms.get_scaled_positions()[mask])
            else:
                pos = atoms.get_positions()[mask]

            scaled = oriented_std_cell.scaled_positions(pos)

            # k.r
            s_q[z][i] = np.exp(2j * np.pi * q @ scaled.T).mean(axis=1)

    return s_q


def get_kspace_projected_density(
    traj: typing.Sequence[Atoms],
    max_bin_width: float | tuple[float, float, float],
    data: _sym.SymmetryDataset,
    atomic_numbers: int | set[int] | None = None,
    max_cell_diff: float | None = None,
    adjust_cell: bool = True,
) -> tuple[PWDescriptor, dict[int, np.ndarray], dict[int, np.ndarray]]:
    if not _has_gpaw:
        raise ImportError("GPAW is not installed")

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
    s_k = {z: np.zeros(kpts.shape[0], dtype=np.complex128) for z in atomic_numbers}
    c = {z: 0 for z in atomic_numbers}

    # loop over trajectory
    for atoms in traj:
        _sym._assert_standard(atoms)
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
            psi_k = np.exp(-1j * kr).sum(axis=1)
            # rho_k[z] += np.cos(kr).sum(axis=1) - 1j * np.sin(kr).sum(axis=1)
            rho_k[z] += psi_k
            s_k[z] += np.abs(psi_k) ** 2
            c[z] += mask.sum()

    # normalization
    dv = oriented_std_cell.volume / np.prod(bin_sizes)
    for z in atomic_numbers:
        rho_k[z] /= c[z] * dv
        s_k[z] /= c[z] * dv
    return pw, rho_k, s_k


def kspace_to_real(pw: PWDescriptor, rho_k: np.ndarray) -> np.ndarray:
    if not _has_gpaw:
        raise ImportError("GPAW is not installed")
    rho_r = pw.ifft(rho_k)
    assert np.allclose(rho_r.imag, 0)
    return rho_r.real
