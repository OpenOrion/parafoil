from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal, Optional, Union
import numpy as np
import numpy.typing as npt
from plotly import graph_objects as go
from parafoil.airfoil import Airfoil
from parafoil.bspline import get_bspline
from ezmesh import Geometry, CurveLoop, PlaneSurface, BoundaryLayer

@dataclass
class PassageOutline:
    top_coords: npt.NDArray[np.float64]
    bottom_coords: npt.NDArray[np.float64]


@dataclass
class Passage:
    airfoil: Airfoil
    "airfoil for the passage"

    spacing: float
    "spacing between blades"

    leading_edge_gap: float
    "gap between the leading edge of the airfoil and the passage"

    trailing_edge_gap: float
    "gap between the trailing edge of the airfoil and the passage"

    num_blades: int = 2
    "number of blades in the passage"

    outline_type: Literal['camber', 'line'] = 'camber'
    "type of outline for the passage"

    offset: npt.NDArray[np.float64] = np.array([0, 0])
    "offset of the passage"

    @cached_property
    def total_spacing(self):
        return self.spacing * self.num_blades

    @cached_property
    def coords(self) -> npt.NDArray[np.float64]:
        return np.concatenate((self.outline.top_coords, self.outline.bottom_coords))

    @cached_property
    def outline(self):
        camber_coords = self.airfoil.camber_coords + np.array([self.leading_edge_gap, 0])
        outline_ctrl_pnts = np.concatenate([
            np.array([[0, 0], [self.leading_edge_gap, 0]]),
            camber_coords,
            np.array([camber_coords[-1] + np.array([self.trailing_edge_gap, 0])])    # type: ignore
        ])
        passage_bspline = get_bspline(outline_ctrl_pnts, 3)
        passage_bspline_coords = passage_bspline(self.airfoil.sampling)
        if self.outline_type == 'camber':
            outline_coords = passage_bspline_coords
        else:
            start_coord = passage_bspline_coords[0]
            line_slope_start_coord: Optional[npt.NDArray[np.float64]] = np.array([self.leading_edge_gap, 0])
            
            is_increasing = passage_bspline_coords[1][1] > start_coord[1]
            for coord in passage_bspline_coords[1:]:
                if is_increasing and coord[1] <= start_coord[1]:
                    line_slope_start_coord = coord
                    break
                elif not is_increasing and coord[1] >= start_coord[1]:
                    line_slope_start_coord = coord
                    break
            
            outline_coords: npt.NDArray[np.float64] = np.concatenate([
                np.array([[0, line_slope_start_coord[1]], line_slope_start_coord]),
                np.array([camber_coords[-1]]),
                np.array([camber_coords[-1] + np.array([self.trailing_edge_gap, 0])])    # type: ignore
            ])

        return PassageOutline(
            top_coords=outline_coords + np.array([0, self.total_spacing/2]) + self.offset,
            bottom_coords=np.flip(outline_coords, axis=0) + np.array([0, -self.total_spacing/2]) + self.offset
        )

    @cached_property
    def airfoils_coords(self):
        airfoil_coords = self.airfoil.coords
        airfoil_leading_pnt = airfoil_coords[np.argmin(airfoil_coords[:, 0])]

        airfoils_coords = []
        airfoil_offset = np.array([
            self.leading_edge_gap-np.min(airfoil_coords[:, 0]),
            (self.total_spacing/2) - (self.spacing/2) - airfoil_leading_pnt[1]
        ])
        for i in range(self.num_blades):
            airfoil_offseted_coords = airfoil_coords+airfoil_offset-np.array([0, i*self.spacing]) + self.offset
            airfoils_coords.append(airfoil_offseted_coords)
        return airfoils_coords

    def generate_mesh(
        self,
        top_label: str = "top",
        bottom_label: str = "bottom",
        inlet_label: str = "inlet",
        outlet_label: str = "outlet",
        airfoil_label: Union[str, List[str]] = "airfoil",
        surface_label: Optional[str] = None,
        airfoil_mesh_size: Optional[float] = None,
        boundary_layer_thickness: Optional[float] = None,
        boundary_wall_mesh_size: Optional[float] = None,
        passage_mesh_size: Optional[float] = None,
        output_path: Optional[str] = None
    ):
        if airfoil_mesh_size is None: 
            airfoil_mesh_size = 0.02 * self.airfoil.chord_length
        if boundary_layer_thickness is None: 
            boundary_layer_thickness = 0.01 * self.airfoil.chord_length
        if boundary_wall_mesh_size is None: 
            boundary_wall_mesh_size = 0.001 * self.airfoil.chord_length
        if passage_mesh_size is None: 
            passage_mesh_size = 0.05 * self.airfoil.chord_length


        with Geometry() as geometry:
            passage_curve_loop = CurveLoop.from_coords(
                self.coords, 
                mesh_size = passage_mesh_size,
                groups=[self.outline.top_coords, self.outline.bottom_coords],
                labels=[top_label, outlet_label, bottom_label, inlet_label],
            )

            boundary_layer = BoundaryLayer(
                hwall_n=boundary_wall_mesh_size,
                thickness=boundary_layer_thickness,
                is_quad_mesh=True,
                intersect_metrics=False
            )

            if isinstance(airfoil_label, List):
                assert len(airfoil_label) == len(self.airfoils_coords)



            airfoil_curve_loops = [
                CurveLoop.from_coords(
                    airfoil_coords, 
                    mesh_size = airfoil_mesh_size, 
                    labels=airfoil_label[i] if isinstance(airfoil_label, List) else airfoil_label,
                    fields=[boundary_layer],
                )
                for i, airfoil_coords in enumerate(self.airfoils_coords)
            ]

            surface = PlaneSurface(
                outlines=[passage_curve_loop],
                holes=airfoil_curve_loops,
                label=surface_label
            )
            if output_path is not None:
                geometry.write(output_path)
            return geometry.generate(surface)

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

        for i, airfoil_coord in enumerate(self.airfoils_coords):
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
