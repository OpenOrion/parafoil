from dataclasses import dataclass
from functools import cached_property
import numpy as np
from plotly import graph_objects as go
from parafoil.airfoil import Airfoil
from parafoil.bspline import get_bspline
@dataclass
class Passage:
    "A row of blades for CFD simulation"
    top_outline: np.ndarray
    "top outline of the row"
    bottom_outline: np.ndarray
    "bottom outline of the row"
    airfoil_coords: list[np.ndarray]
    "airfoil coordinates for row"

    @cached_property
    def coords(self):
        return np.concatenate((self.top_outline, self.bottom_outline))

    def visualize(self):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Passage"))
        )
        fig.add_trace(go.Scatter(
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            fill="toself",
            legendgroup="passage",
            name=f"Passage"
        ))

        for i, airfoil_coord in enumerate(self.airfoil_coords):
            fig.add_trace(go.Scatter(
                x=airfoil_coord[:, 0],
                y=airfoil_coord[:, 1],
                fill="toself",
                legendgroup="airfoil",
                legendgrouptitle_text="Airfoils",
                name=f"Airfoil {i+1}"

            ))
        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()


class PassageBuilder:
    @staticmethod
    def airfoil_row_passage(
        airfoil: Airfoil,
        spacing: float,
        leading_edge_gap: float,
        trailing_edge_gap: float,
        num_blades: int = 2
    ):
        airfoil_coords = airfoil.coords
        height = spacing * num_blades
        airfoil_width: float = np.max(airfoil_coords[:, 0]) - np.min(airfoil_coords[:, 0])
        airfoil_leading_pnt = airfoil_coords[np.argmin(airfoil_coords[:, 0])]

        airfoil_row_coords = []
        airfoil_offset = np.array([
            leading_edge_gap-np.min(airfoil_coords[:, 0]),
            -airfoil_leading_pnt[1]+(height/2) - (spacing/2)
        ])
        for i in range(num_blades):
            airfoil_offseted_coords = airfoil_coords+airfoil_offset-np.array([0, i*spacing])
            airfoil_row_coords.append(airfoil_offseted_coords)

        passage_camber_ctrl_pnts = airfoil.camber_coords + np.array([leading_edge_gap, 0])
        outline_ctrl_pnts=np.concatenate([    
            np.array([[0, 0], [leading_edge_gap, 0]]),
            passage_camber_ctrl_pnts,
            np.array([[leading_edge_gap+airfoil_width+trailing_edge_gap, passage_camber_ctrl_pnts[-1][1]]])    # type: ignore
        ])

        passage_bspline = get_bspline(outline_ctrl_pnts, 3)

        return Passage(
            top_outline=passage_bspline(airfoil.sampling) + np.array([0, height/2]),
            bottom_outline=passage_bspline(airfoil.sampling_reversed) + np.array([0, -height/2]),
            airfoil_coords=airfoil_row_coords
        )