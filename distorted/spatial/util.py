import numpy as np
from ase.cell import Cell


def get_num_bins(
    cell: Cell,
    max_bin_width: float | tuple[float, float, float],
) -> tuple[int, int, int]:
    """
    Get the number of bins for a given cell and max bin width.
    """
    lengths = cell.cellpar()[:3]
    return tuple(np.ceil(lengths / max_bin_width).astype(int))
