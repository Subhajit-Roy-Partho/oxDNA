"""Narrow compatibility shim for legacy OAT releases on modern NumPy."""

import numpy as np

# oxDNA_analysis_tools 1.0.18 still constructs cell counts with ``np.int``.
# Restoring this removed alias only in the output_bonds subprocess avoids
# modifying the user's Python installation or the numerical dtype of arrays.
if "int" not in np.__dict__:
    np.int = int
