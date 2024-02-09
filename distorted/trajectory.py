import typing

import numpy as np
from ase.atoms import Atoms
from ase.cell import Cell


def get_standard_trajectory(
    traj: typing.Sequence[Atoms],
    fixed_com_atom_type: int,
    adjust_cells: bool = False,
) -> tuple[list[Atoms], float]:
    """
    Process configs in trajectory so that:
    1. The cell is oriented consistently.
    2. The center of mass of the fixed atoms remains constant.

    Args:
        traj: Trajectory to preprocess.
        fixed_com_atom_type: Atom type to fix the center of mass.
        adjust_cells: Adjust the cell to the mean cell.

    Returns:
        new standard trajectory.
        displacement of the center of mass of the fixed atoms.
    """
    if adjust_cells:
        mean_cell = Cell(np.mean([atoms.cell for atoms in traj], axis=0))

    mask = traj[0].numbers == fixed_com_atom_type
    com = traj[0][mask].get_center_of_mass()
    com_last = traj[-1][mask].get_center_of_mass()
    com_disp = np.linalg.norm(com_last - com)
    processed_traj = []
    for atoms in traj:
        # translate snapshots so that the center of mass of
        # the fixed atoms remains constant
        translation = com - atoms[mask].get_center_of_mass()
        positions = atoms.get_positions() + translation
        # generate from cellpar and scaled_positions so that the cell
        # is oriented correctly
        cell = atoms.cell if not adjust_cells else mean_cell
        new = get_standard_cell(cell, positions, atoms.numbers, atoms.pbc)
        processed_traj.append(new)
    return processed_traj, com_disp


def get_standard_cell(
    cell: Cell,
    positions: np.ndarray,
    numbers: typing.Sequence[int],
    pbc: bool | tuple[bool, bool, bool],
) -> Atoms:
    """
    Return an Atoms object with a consistent orientation of the cell.
    """
    return Atoms(
        cell=cell.cellpar(),
        scaled_positions=cell.scaled_positions(positions),
        numbers=numbers,
        pbc=pbc,
    )


def get_standard_atoms(atoms: Atoms) -> Atoms:
    """
    Return an Atoms object with a consistent orientation of the cell.
    """
    return Atoms(
        cell=atoms.cell.cellpar(),
        scaled_positions=atoms.cell.scaled_positions(atoms.positions),
        numbers=atoms.numbers,
        pbc=atoms.pbc,
    )


def _assert_standard(atoms: Atoms) -> None:
    """
    Check if the cell is oriented correctly.
    """
    cell = atoms.cell
    cell2 = atoms.cell.fromcellpar(cell.cellpar())
    if not np.allclose(cell, cell2):
        raise ValueError("The cell is not oriented correctly.")


def mean_of_traj(
    traj: typing.Sequence[Atoms], mask: typing.Sequence[bool] | None = None
) -> Atoms:
    """
    Return the mean positions of the trajectory.

    Args:
        traj: Trajectory to average.
        mask: Mask to select atoms.

    Returns:
        Atoms object with the mean positions.
    """
    # assert all cells are the same
    cell = traj[0].cell
    for atoms in traj:
        if not np.allclose(cell, atoms.cell):
            raise ValueError("All cells must be the same.")
    if mask is None:
        mask = np.ones(len(traj[0]), dtype=bool)
    positions = np.mean([atoms.positions[mask] for atoms in traj], axis=0)
    return Atoms(
        cell=traj[0].cell,
        positions=positions,
        numbers=traj[0].numbers[mask],
        pbc=traj[0].pbc,
    )
