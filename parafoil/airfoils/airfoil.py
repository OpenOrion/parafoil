from typing import Protocol
import plotly.graph_objects as go
import numpy.typing as npt
import numpy as np

class Airfoil(Protocol):
    chord_length: float
    axial_chord_length: float
    height: float
    def get_coords(self) -> npt.NDArray[np.float64]: pass # type: ignore


    def visualize(self):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Airfoil"))
        )

        coords = self.get_coords()
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            name=f"Airfoil"
        ))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore

        fig.show()