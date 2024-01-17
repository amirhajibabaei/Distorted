"""
Visualization of unit cells and atomic structures.

"""
import typing

import mpl_toolkits.mplot3d as mp3d
import numpy as np
from ase.atoms import Atoms

Coo3dType = tuple[float, float, float]
CellType = tuple[Coo3dType, Coo3dType, Coo3dType]
RepeatType = int | tuple[int, int, int]


def _draw_cell(
    ax3d: mp3d.Axes3D,
    cell: CellType,
    celldisp: Coo3dType | None = None,
    repeat: RepeatType = 1,
    color: str = "k",
    **plot_kwargs: dict,
):
    """
    Plot a unit cell.

    Args:
        ax3d: 3D axes.
        cell: Unit cell.
        celldisp: Displaced unit cell.
        color: Color of the cell.

    """
    if celldisp is None:
        celldisp = (0, 0, 0)

    if isinstance(repeat, int):
        nx = ny = nz = repeat
    else:
        nx, ny, nz = repeat

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                r0 = i * cell[0] + j * cell[1] + k * cell[2] + celldisp
                for u, (q, p) in enumerate(zip([i, j, k], [nx, ny, nz])):
                    if q + 1 <= p:
                        r1 = r0 + cell[u]
                        ax3d.plot(
                            [r0[0], r1[0]],
                            [r0[1], r1[1]],
                            [r0[2], r1[2]],
                            color=color,
                            **plot_kwargs,
                        )


def _draw_axes(
    ax3d: mp3d.Axes3D,
    cell: CellType,
    celldisp: Coo3dType | None = None,
    style: typing.Literal["origin", "corners"] = "origin",
):
    length = np.linalg.norm(cell, axis=1).min() / 5

    if celldisp is None:
        celldisp = (0, 0, 0)

    for i, (name, color) in enumerate(zip(["a", "b", "c"], ["r", "g", "b"])):
        _dir = np.array(cell[i])
        _norm = _dir / np.linalg.norm(_dir)
        vec = length * _norm
        if style == "corners":
            origin = celldisp + cell[i]
        else:
            origin = np.array(celldisp) - 2
        # draw arrow from celldisp to cell[i]
        ax3d.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            arrow_length_ratio=0.1,
            alpha=0.5,
            lw=1,
        )
        # draw text at celldisp + cell[i]
        ext = 1.0
        ax3d.text(
            origin[0] + _norm[0] * ext + vec[0],
            origin[1] + _norm[1] * ext + vec[1],
            origin[2] + _norm[2] * ext + vec[2],
            f"${name}$",
            color=color,
            fontsize=16,
        )


def _draw_atoms(
    ax3d: mp3d.Axes3D,
    atoms: typing.Sequence[Coo3dType],
    color: str = "k",
    s: float = 100,
    **kwargs: dict,
):
    """
    Plot atoms.

    Args:
        ax3d: 3D axes.
        atoms: Atomic positions.
        color: Color of the atoms.
        radius: Radius of the atoms.

    """
    z_max = max(atom[-1] for atom in atoms)
    z_min = min(atom[-1] for atom in atoms)
    w = z_max - z_min
    z_max += w / 2

    def get_alpha(z):
        return (z_max - z) / (z_max - z_min)

    for atom in atoms:
        alpha = get_alpha(atom[-1])
        ax3d.scatter(*atom, s=s * alpha, color=color, alpha=alpha, **kwargs)


def draw(atoms: Atoms, ax3d: mp3d.Axes3D | None = None, **kwargs: dict):
    """
    Draw unit cell and atomic positions.

    Args:
        atoms: Atomic structure.
        ax3d: 3D axes.

    """
    if ax3d is None:
        fig = mp3d.figure.Figure()
        ax3d = fig.add_subplot(111, projection="3d")
        ax3d.set_aspect("equal")
        ax3d.axis("off")

    cell = atoms.cell
    celldisp: Coo3dType | None = kwargs.pop("celldisp", None)  # type: ignore
    repeat: RepeatType = kwargs.pop("repeat", 1)  # type: ignore
    color: str = kwargs.pop("color", "k")  # type: ignore
    s: float = kwargs.pop("s", 100)  # type: ignore
    _draw_cell(ax3d, cell, celldisp, repeat, color, **kwargs)
    _draw_axes(ax3d, cell, celldisp)
    _draw_atoms(ax3d, atoms.positions, color, s)

    return ax3d
