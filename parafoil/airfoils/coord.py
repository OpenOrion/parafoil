from dataclasses import dataclass
from typing import List, cast
import numpy as np
from .airfoil import Airfoil


@dataclass
class CoordAirfoil(Airfoil):
    coords: List[List[np.float64]]

    def __post_init__(self):

        coords_x = np.array(self.coords)[:, 0]
        self.axial_chord_length = self.chord_length = cast(float, np.max(coords_x) - np.min(coords_x))
        
    def get_coords(self):
        return np.array(self.coords)