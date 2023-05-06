from dataclasses import dataclass
from typing import List, cast
import numpy as np
from .airfoil import Airfoil

@dataclass
class CoordAirfoil(Airfoil):
    coords: List[List[float]]

    def __post_init__(self):
        self.coords = np.array(self.coords) # type: ignore
        coords_x = self.coords[:, 0]
        coords_y = self.coords[:, 1]
        self.axial_chord_length = cast(float, np.max(coords_x) - np.min(coords_x))
        self.chord_height = cast(float, np.max(coords_y) - np.min(coords_y))
        self.chord_length = np.sqrt(self.axial_chord_length**2 + self.chord_height**2)

    def get_coords(self):
        return self.coords
    
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