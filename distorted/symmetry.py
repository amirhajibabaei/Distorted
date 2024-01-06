import dataclasses
import typing

import numpy as np
import spglib
from ase import Atoms
from ase.build import make_supercell
from numpy import ndarray

# SpgCell: (cell, scaled_positions, numbers)
SpgCell = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclasses.dataclass(kw_only=True)
class SymmetryDataset:
    number: int
    hall_number: int
    international: str
    hall: str
    choice: str
    transformation_matrix: ndarray
    origin_shift: ndarray
    rotations: ndarray
    translations: ndarray
    wyckoffs: list[str]
    site_symmetry_symbols: list[str]
    crystallographic_orbits: ndarray
    equivalent_atoms: ndarray
    primitive_lattice: ndarray
    mapping_to_primitive: ndarray
    std_lattice: ndarray
    std_types: ndarray
    std_positions: ndarray
    std_rotation_matrix: ndarray
    std_mapping_to_primitive: ndarray
    pointgroup: str

    def get_standard_unit_cell(
        self,
        origin_choice: typing.Literal["symmetric", "unchanged"] = "unchanged",
    ) -> Atoms:
        """
        Get the standard unit cell as an ASE Atoms object.

        Args:
            origin_choice:
                The origin choice for the standard unit cell.
                The default choice of spglib is 'symmetric', which
                shifts the origin so that atoms are located
                at symmetric positions.
                'unchanged' means discarding this shift which
                is more relevant for the reconstruction of the
                super cell.

        """

        if origin_choice == "symmetric":
            scaled_positions = self.std_positions
        elif origin_choice == "unchanged":
            scaled_positions = self.std_positions - self.origin_shift
        else:
            raise ValueError(
                f"origin_choice must be 'symmetric' or 'unchanged', got {origin_choice}"
            )

        spg_cell = (
            self.std_lattice,
            scaled_positions,
            self.std_types,
        )
        atoms = _spg_cell_to_ase_atoms(spg_cell)
        return atoms

    def get_super_cell(self, unit_cell: Atoms | None = None) -> Atoms:
        """Get the reconstructed super cell as an ASE Atoms object."""
        if unit_cell is None:
            unit_cell = self.get_standard_unit_cell(origin_choice="unchanged")
        # Note that the definition of the transformation matrix is different
        # from the one used in spglib (by a transpose).
        super_cell = make_supercell(unit_cell, self.transformation_matrix.T)
        return super_cell


def get_symmetry_dataset(atoms: Atoms, symprec: float) -> SymmetryDataset:
    """Get the symmetry dataset from an ASE Atoms object."""
    spg_cell = _ase_atoms_to_spg_cell(atoms)
    dataset = spglib.get_symmetry_dataset(spg_cell, symprec=symprec)
    return SymmetryDataset(**dataset)


def _ase_atoms_to_spg_cell(atoms: Atoms) -> SpgCell:
    """Get the cell arguments for spglib from an ASE Atoms object."""
    cell = atoms.get_cell()
    scaled_positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    spg_cell = (cell, scaled_positions, numbers)
    return spg_cell


def _spg_cell_to_ase_atoms(
    spg_cell: SpgCell,
) -> Atoms:
    """Get an ASE Atoms object from the cell arguments for spglib."""
    cell, scaled_positions, numbers = spg_cell
    atoms = Atoms(numbers, scaled_positions=scaled_positions, cell=cell, pbc=True)
    return atoms
