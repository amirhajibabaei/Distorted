# +
import typing
import warnings

import ase.neighborlist as _ase
import numpy as np
from ase import Atoms
from scipy.spatial import ConvexHull, Voronoi, distance_matrix


def get_neighborlist(
    natoms: int,
    cutoff: float,
    skin: float = 0.3,
    scaling: typing.Literal["linear", "quadratic"] = "linear",
) -> _ase.NeighborList:
    if scaling == "linear":
        primitive = _ase.NewPrimitiveNeighborList
    elif scaling == "quadratic":
        primitive = _ase.PrimitiveNeighborList
    return _ase.NeighborList(
        natoms * [cutoff / 2],
        skin=skin,
        bothways=True,
        self_interaction=False,
        primitive=primitive,
    )


class VoronoiNeighborlist:
    __slots__ = ["_nl", "_atoms", "_data", "_nei_count", "_types", "_centre"]

    nl: _ase.NeighborList
    _atoms: Atoms
    _data: typing.List[typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]
    _nei_count: list[int]
    _types: list[int]
    _centre: dict[int, np.ndarray]

    def __init__(self, nl: _ase.NeighborList) -> None:
        self._nl = nl

    def update_(self, atoms: Atoms) -> None:
        self._nl.update(atoms)
        self._data = []
        self._nei_count = []
        accounted_volume = 0.0
        for i in range(len(atoms)):
            j, offsets = self._nl.get_neighbors(i)
            rij = _get_displacements(atoms, i, j, offsets)
            indices, areas, volumes = _get_local_voronoi(rij)
            self._data.append((indices, areas, volumes))
            self._nei_count.append(len(indices))
            accounted_volume += volumes.sum()

        if not np.isclose(atoms.get_volume(), accounted_volume):
            warnings.warn(
                "The vornoi cells do not add up to the total volume of the "
                "system. This is likely due to small cutoff of the supplied "
                "neighborlist. Consider increasing the cutoff. "
                f"(volume: {atoms.get_volume()}, accounted: {accounted_volume}, "
                f"diff: {atoms.get_volume() - accounted_volume})"
            )

        self._atoms = atoms

    def apply_clustering_(self, prec) -> int:
        """
        Apply clustering to the voronoi cells.

        Args:
            prec: The precision of the clustering.

        Returns:
            The number of clusters.
        """
        self._centre = {}
        self._types = []
        for i in range(len(self._atoms)):
            rij = self.get_vor_nei_disp(i)
            n = rij.shape[0]
            # find the most aligned reference
            best_k = 0
            best_prec = np.inf
            best_perm = None
            for k, ref in self._centre.items():
                if ref.shape[0] != n:
                    continue
                perm, diffs = _min_dist_perm(rij, ref)
                x = diffs.max()
                if x < prec and x < best_prec:
                    best_k = k
                    best_prec = x
                    best_perm = perm
            if best_perm is None:
                self._centre[i] = rij
                self._types.append(i)
            else:
                # apply the permutation
                self._data[i] = tuple(v[best_perm] for v in self._data[i])
                self._types.append(best_k)
                # update the reference
                w = self._types.count(best_k)
                self._centre[best_k] = (
                    self._centre[best_k] * (w - 1) + rij[best_perm]
                ) / w
        return len(self._centre)

    def get_refined_structure(self, atoms: Atoms | None = None, max_iter=100) -> Atoms:
        if atoms is None:
            atoms = self._atoms
        refined = atoms.copy()

        def do(i):
            shifts = self._centre[self._types[i]]
            nei = self.get_vor_nei_indices(i)
            for j, vec in zip(nei, shifts):
                if not mask[j]:
                    refined.positions[j] = refined.positions[i] + vec
                    mask[j] = True

        # first atom
        mask = len(refined) * [False]
        mask[0] = True

        # rest
        for _ in range(max_iter):
            for i in range(len(refined)):
                if not mask[i]:
                    continue
                do(i)
            if all(mask):
                break

        assert all(mask), len(mask) - sum(mask)

        #  find minimal displacements
        dr = refined.positions - atoms.positions
        ds = (refined.cell.scaled_positions(dr) % 1.0).reshape(-1)
        ds[ds > 0.5] -= 1.0
        dr_minimal = refined.cell.cartesian_positions(ds.reshape(-1, 3))

        # apply minimal displacements
        refined.positions = atoms.positions + dr_minimal
        refined.set_tags(self.get_tags())

        return refined

    def get_tags(self) -> np.ndarray:
        tags = sorted(self._centre.keys())
        return np.array([tags.index(t) for t in self._types])

    def get_tag(self, i: int) -> int:
        tags = sorted(self._centre.keys())
        return tags.index(self._types[i])

    def get_vor_nei_count(self, i: int) -> int:
        return self._nei_count[i]

    def get_vor_nei_indices(self, i: int) -> np.ndarray:
        j, _ = self._nl.get_neighbors(i)
        indices, _, _ = self._data[i]
        return j[indices]

    def get_vor_nei_disp(self, i: int) -> np.ndarray:
        j, offsets = self._nl.get_neighbors(i)
        indices, _, _ = self._data[i]
        return _get_displacements(self._atoms, i, j[indices], offsets[indices])

    def get_vor_nei_data(
        self,
        i: int,
        quantity: typing.Literal["facet_area", "facet_volume"],
    ) -> np.ndarray:
        if quantity == "facet_area":
            _, x, _ = self._data[i]
        elif quantity == "facet_volume":
            _, _, x = self._data[i]
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        return x


def _get_local_voronoi(
    rij: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = rij.shape[1]
    orig = np.zeros((1, dim))
    # add origin to the beginning of rij -> c_rij
    c_rij = np.r_[orig, rij]
    vor = Voronoi(c_rij)
    indices = []
    areas = []
    volumes = []
    # loop over nn pairs
    for i, (_a, _b) in enumerate(vor.ridge_points):
        a, b = sorted((_a, _b))
        # only interested in neighbors of orig
        if a == 0:
            # vert: vertices for the face of vor cell
            # that devides the space between a, b
            vert = vor.ridge_vertices[i]
            # face: coordinates of the face
            face = vor.vertices[vert]
            # ph: orig+face polyhedron
            ph = np.r_[orig, face]
            vol = ConvexHull(ph).volume
            # h: length of the line segment from origin to the face
            h = np.linalg.norm(vor.points[b]) / 2
            # area of the face is obtained from vol and h
            area = dim * vol / h
            # note below: b-1, because c_rij is used instead of rij
            indices.append(b - 1)
            areas.append(area)
            volumes.append(vol)
    return np.array(indices), np.array(areas), np.array(volumes)


def _min_dist_perm(rij: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find a permutation of rows of rij that minimizes the sum of squared
    distances to ref:
        rij[perm] ~ ref

    They should have the same length, but the order may be different.

    The assumption is that since these are vectors to voronoi neighbors, the
    vectors in rij, ref are well distinct, so a simple distance matrix is sufficient
    and we don't need to search all permutations.

    Returns:
        perm: permutation of rij that minimizes the sum of squared distances
        diff: squared distances
    """
    assert rij.shape == ref.shape
    dm = distance_matrix(rij, ref)
    perm = np.argmin(dm, axis=0)
    diff = dm[perm].diagonal()
    return perm, diff


def _get_displacements(
    atoms: Atoms, i: int, j: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    shifts = (offsets[..., None] * atoms.cell).sum(axis=1)
    return atoms.positions[j] - atoms.positions[i] + shifts
