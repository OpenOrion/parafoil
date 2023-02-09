from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal, Optional, Union
import numpy as np
import numpy.typing as npt
from plotly import graph_objects as go
from parafoil.airfoil import Airfoil
from parafoil.bspline import get_bspline
from ezmesh import CurveLoop, PlaneSurface, BoundaryLayer

@dataclass
class PassageMeshConfig:
    airfoil_mesh_size: float = 0.001
    boundary_layer_thickness: float = airfoil_mesh_size
    boundary_wall_mesh_size: float = boundary_layer_thickness * 0.1
    passage_mesh_size: float = (airfoil_mesh_size * 10)/2

@dataclass
class Passage:
    "A row of blades for CFD simulation"
    top_outline: npt.NDArray[np.float64]
    "top outline of the row"
    bottom_outline: npt.NDArray[np.float64]
    "bottom outline of the row"
    airfoil_coords: list[npt.NDArray[np.float64]]
    "airfoil coordinates for row"
    
    @cached_property
    def coords(self):
        return np.concatenate((self.top_outline, self.bottom_outline))

    def get_mesh_surface(
        self,
        top_label: str = "top",
        bottom_label: str = "bottom",
        inlet_label: str = "inlet",
        outlet_label: str = "outlet",
        airfoil_label: Union[str, List[str]] = "airfoil",
        surface_label: Optional[str] = None,
        mesh_config: PassageMeshConfig = PassageMeshConfig(),
    ): 
        if isinstance(airfoil_label, List):
            assert len(airfoil_label) == len(self.airfoil_coords)
        
        
        passage_curve_loop = CurveLoop(
            self.coords,
            mesh_size=mesh_config.passage_mesh_size,
            labels={
                top_label: range(0, len(self.top_outline)-1),
                outlet_label: [len(self.top_outline)-1],
                bottom_label: range(len(self.top_outline), len(self.top_outline)+len(self.bottom_outline)-1),
                inlet_label: [len(self.top_outline)+len(self.bottom_outline)-1],
            },
        )

        boundary_layer = BoundaryLayer(
            hwall_n=mesh_config.boundary_wall_mesh_size,
            thickness=mesh_config.boundary_layer_thickness,
            quads=True,
            intersect_metrics=False
        )

        airfoil_curve_loops = [
            CurveLoop(
                airfoil_coords,
                mesh_size=mesh_config.airfoil_mesh_size,
                labels={airfoil_label[i] if isinstance(airfoil_label, List) else airfoil_label: "all"},
                fields=[boundary_layer],
            )
            for i, airfoil_coords in enumerate(self.airfoil_coords)
        ]

        surface = PlaneSurface(
            passage_curve_loop,
            holes=airfoil_curve_loops,
            label=surface_label
        )
        return surface

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
        num_blades: int = 2,
        offset: npt.NDArray[np.float64] = np.array([0, 0]),
        type: Literal['camber', 'line'] = 'camber',
    ):
        airfoil_coords = airfoil.coords
        height = spacing * num_blades
        airfoil_leading_pnt = airfoil_coords[np.argmin(airfoil_coords[:, 0])]

        airfoil_row_coords = []
        airfoil_offset = np.array([
            leading_edge_gap-np.min(airfoil_coords[:, 0]),
            -airfoil_leading_pnt[1]+(height/2) - (spacing/2)
        ])
        for i in range(num_blades):
            airfoil_offseted_coords = airfoil_coords+airfoil_offset-np.array([0, i*spacing]) + offset
            airfoil_row_coords.append(airfoil_offseted_coords)

        
        camber_coords = airfoil.camber_coords + np.array([leading_edge_gap, 0])

        outline_ctrl_pnts = np.concatenate([
            np.array([[0, 0], [leading_edge_gap, 0]]),
            camber_coords,
            np.array([camber_coords[-1] + np.array([trailing_edge_gap, 0])])    # type: ignore
        ])
        passage_bspline = get_bspline(outline_ctrl_pnts, 3)
        passage_bspline_coords = passage_bspline(airfoil.sampling)
        if type == 'camber':
            outline_coords = passage_bspline_coords
        else:
            
            start_coord = passage_bspline_coords[0]
            line_slope_start_coord: Optional[npt.NDArray[np.float64]] = None
            
            is_increasing = passage_bspline_coords[1][1] > start_coord[1]
            for i, coord in enumerate(passage_bspline_coords[1:]):
                if is_increasing and coord[1] <= start_coord[1]:
                    line_slope_start_coord = coord
                    break
                elif not is_increasing and coord[1] >= start_coord[1]:
                    line_slope_start_coord = coord
                    break
                
            assert line_slope_start_coord is not None
            outline_coords = np.concatenate([
                np.array([[0, line_slope_start_coord[1]], line_slope_start_coord]),
                np.array([camber_coords[-1]]),
                np.array([camber_coords[-1] + np.array([trailing_edge_gap, 0])])    # type: ignore
            ])

        return Passage(
            top_outline=outline_coords + np.array([0, height/2]) + offset,
            bottom_outline=np.flip(outline_coords, axis=0) + np.array([0, -height/2]) + offset,
            airfoil_coords=airfoil_row_coords
        )
