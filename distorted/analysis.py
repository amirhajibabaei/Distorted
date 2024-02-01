import numpy as np
from ase.atoms import Atoms

from .symmetry import get_symmetry_dataset
from .voronoi import VoronoiNeighborlist, get_neighborlist


def analyse_structure(
    atoms: Atoms,
    voronoi_neighborhood: float | VoronoiNeighborlist,
    clustering_threshold: float,
    symprec: float,
):
    if isinstance(voronoi_neighborhood, float):
        nl = VoronoiNeighborlist(get_neighborlist(len(atoms), voronoi_neighborhood))
        nl.update_(atoms)
    else:
        nl = voronoi_neighborhood

    ntags = nl.apply_clustering_(clustering_threshold)
    refined = nl.get_refined_structure()
    refined_dist = np.linalg.norm(refined.positions - atoms.positions, axis=1).max()
    tags = refined.get_tags()

    unit_cell = None
    _tags = []
    shifts = []
    for t in range(ntags):
        sub_latt = refined[tags == t]
        sub_sym = get_symmetry_dataset(
            sub_latt, symprec if symprec is not None else refined_dist
        )
        uc = sub_sym.get_standard_unit_cell(origin_choice="unchanged")
        if unit_cell is None:
            unit_cell = uc
            sym = sub_sym
        else:
            unit_cell += uc

        _tags.extend(len(uc) * [t])

        assert sub_sym.number == sym.number
        assert np.allclose(sub_sym.std_lattice, sym.std_lattice)

        shifts.append(sub_sym.origin_shift)

    assert unit_cell is not None
    unit_cell.set_tags(_tags)
    return unit_cell, sym, refined_dist, nl, shifts
