from dataclasses import dataclass
from typing import List, cast
import numpy as np
from .airfoil import Airfoil

@dataclass
class CoordAirfoil(Airfoil):
    coords: List[List[float]]

    def __post_init__(self):
        coords_array = np.array(self.coords)
        coords_x = coords_array[:, 0]
        coords_y = coords_array[:, 1]
        self.axial_chord_length = cast(float, np.max(coords_x) - np.min(coords_x))
        self.height = cast(float, np.max(coords_y) - np.min(coords_y))
        self.chord_length = np.sqrt(self.axial_chord_length**2 + self.height**2)

    def get_coords(self):
        return np.array(self.coords)
    
    @staticmethod
    def from_dat(file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Extract coordinates from lines
        coords = []
        for line in lines:
            if len(line) > 0:
                x, y = line.strip().split()
                coords.append([float(x), float(y)])
        return CoordAirfoil(coords)