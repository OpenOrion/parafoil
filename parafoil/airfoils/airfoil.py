from typing import Protocol, Tuple
import numpy.typing as npt
import numpy as np

class Airfoil(Protocol):
    chord_length: float
    axial_chord_length: float
    height: float
    def get_coords(self) -> npt.NDArray[np.float64]: pass # type: ignore
