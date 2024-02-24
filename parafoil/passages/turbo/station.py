import numpy as np
import cadquery as cq
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, Union
import numpy.typing as npt
from paraflow.passages.passage import SimulationParams
from parafoil.airfoils import CamberThicknessAirfoil
from parafoil.metadata import opt_class, opt_constant
from paraflow import Passage, SimulationParams
from parafoil.utils import get_bspline, get_sampling
from plotly import graph_objects as go


@dataclass
class TurboMeshParameters:
    top_label: str = "top"
    bottom_label: str = "bottom"
    inlet_label: str = "inlet"
    outlet_label: str = "outlet"
    airfoil_label: Union[str, List[str]] = "airfoil"
    airfoil_mesh_size: Optional[float] = None
    boundary_wall_mesh_size: Optional[float] = None
    boundary_wall_ratio: float = 0.1
    boundary_num_layers: int = 4
    passage_mesh_size: Optional[float] = None


@dataclass
class TurboStationPassage(Passage):
    airfoil: CamberThicknessAirfoil = field(metadata=opt_class())
    "airfoil for the passage"

    spacing_to_chord: float = field(metadata=opt_constant())
    "spacing between blades"

    leading_edge_gap_to_chord: float = field(metadata=opt_constant())
    "gap between the leading edge of the airfoil and the passage"

    trailing_edge_gap_to_chord: float = field(metadata=opt_constant())
    "gap between the trailing edge of the airfoil and the passage"

    num_airfoils: int = 1
    "number of blades in the passage"

    offset: list[float] = field(default_factory=lambda: [0, 0])
    "offset of the passage"

    num_samples: int = 50
    "number of samples"

    is_cosine_sampling: bool = True
    "use cosine sampling"

    type: Literal["camber", "surface", "line"] = "camber"
    "type of the passage curve based on pressure or suction surface or camber"

    def __post_init__(self):
        self.degree = 3
        self.leading_edge_gap = (
            self.leading_edge_gap_to_chord * self.airfoil.chord_length
        )
        self.trailing_edge_gap = (
            self.trailing_edge_gap_to_chord * self.airfoil.chord_length
        )
        self.width = (
            self.airfoil.axial_chord_length
            + self.leading_edge_gap
            + self.trailing_edge_gap
        )
        self.spacing = self.airfoil.chord_length * self.spacing_to_chord
        self.height = self.spacing * self.num_airfoils
        self.sampling = get_sampling(self.num_samples, False)

    @cached_property
    def total_spacing(self):
        return self.spacing * self.num_airfoils

    def get_ctrl_pnts(
        self, type: Literal["top", "bottom", "camber"] = "top", is_centerd: bool = True
    ) -> npt.NDArray[np.float64]:
        if type == "camber":
            ctrl_coords = self.airfoil.camber_coords
        elif type == "top":
            ctrl_coords = self.airfoil.top_ctrl_pnts
        elif type == "bottom":
            ctrl_coords = self.airfoil.bottom_ctrl_pnts

        if self.type != "line" and type in ["top", "bottom"]:
            ctrl_coords = ctrl_coords[1:-1]

        if self.type == "line":
            ctrl_coords = np.array([ctrl_coords[0], ctrl_coords[-1]])

        passage_length = np.array([self.width, ctrl_coords[-1][1]])
        ctrl_pnts = np.array(
            [
                [0, 0],
                *(ctrl_coords + np.array([self.leading_edge_gap, 0])),
                passage_length,
            ]
        )

        return ctrl_pnts

    def get_airfoil_coords(self):
        top_coords = get_bspline(self.airfoil.top_ctrl_pnts, self.airfoil.degree)(
            self.airfoil.sampling
        )
        bottom_coords = get_bspline(self.airfoil.bottom_ctrl_pnts, self.airfoil.degree)(
            self.airfoil.sampling
        )

        # center the coordinates and apply +/- half spacing
        top_coords_centered = (
            top_coords
            - np.mean(top_coords, axis=0)
            + np.array(self.offset)

        )
        bottom_coords_centered = (
            bottom_coords
            - np.mean(top_coords, axis=0)
            + np.array(self.offset)

        )

        return top_coords_centered, bottom_coords_centered

    def get_airfoil_profile(self, workplane: Optional[cq.Workplane] = None):
        if workplane is None:
            workplane = cq.Workplane("XY")

        top_coords, bottom_coords = self.get_airfoil_coords()

        return workplane.spline(top_coords).spline(bottom_coords[::-1]).close()

    def get_coords(self):
        # control points of upper and lower passage wall
        if self.type == "camber":
            top_ctrl_pnts = bottom_ctrl_pnts = self.get_ctrl_pnts("camber")
        else:
            top_ctrl_pnts = self.get_ctrl_pnts("top")
            bottom_ctrl_pnts = self.get_ctrl_pnts("bottom")

        # evaluate the coordinates of the BSpline control points
        top_coords = get_bspline(top_ctrl_pnts, self.degree)(self.sampling)
        bottom_coords = get_bspline(bottom_ctrl_pnts, self.degree)(self.sampling)

        # center the coordinates and apply +/- half spacing and user specified offset
        top_coords_centered = (
            top_coords
            - np.mean(top_coords, axis=0)
            + np.array([0, self.total_spacing / 2])
            + np.array(self.offset)
        )
        bottom_coords_centered = (
            bottom_coords
            - np.mean(bottom_coords, axis=0)
            + np.array([0, -self.total_spacing / 2])
            + np.array(self.offset)
        )

        return top_coords_centered, bottom_coords_centered

    def get_profile(self, workplane: Optional[cq.Workplane] = None):
        if workplane is None:
            workplane = cq.Workplane("XY")

        top_coords, bottom_coords = self.get_coords()

        profile = (
            workplane.spline(top_coords)
            .lineTo(*bottom_coords[-1])
            .spline(bottom_coords[::-1])
            .close()
        )

        return profile

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self, airfoil: CamberThicknessAirfoil):
        return TurboStationPassage(
            airfoil=airfoil,
            spacing_to_chord=self.spacing_to_chord,
            leading_edge_gap_to_chord=self.leading_edge_gap_to_chord,
            trailing_edge_gap_to_chord=self.trailing_edge_gap_to_chord,
            num_airfoils=self.num_airfoils,
            offset=self.offset,
            num_samples=self.num_samples,
            is_cosine_sampling=self.is_cosine_sampling,
            type=self.type,
        )

    def to_unstructured_geo(
        self,
        mesh_params: TurboMeshParameters = TurboMeshParameters(),
        sim_params: Optional[SimulationParams] = None,
    ):
        if mesh_params.airfoil_mesh_size is None:
            mesh_params.airfoil_mesh_size = 0.02 * self.airfoil.chord_length
        if mesh_params.boundary_wall_mesh_size is None:
            mesh_params.boundary_wall_mesh_size = 0.01 * self.airfoil.chord_length
        if mesh_params.passage_mesh_size is None:
            mesh_params.passage_mesh_size = 0.025 * self.airfoil.chord_length

        # with GeometryQL() as geo:
        #     y_plus = get_y_plus(self.airfoil.chord_length, sim_params)
        #     profile = self.get_profile()
        #     return (
        #         geo
        #         .load(profile)

        #         .edges(type="interior")
        #         .setMeshSize(mesh_params.airfoil_mesh_size)
        #         .addBoundaryLayer(y_plus, mesh_params.boundary_wall_ratio, mesh_params.boundary_num_layers)
        #         .addPhysicalGroup(mesh_params.airfoil_label)
        #         .end()

        #         .edges(type="exterior")
        #         .setMeshSize(mesh_params.passage_mesh_size)
        #         .end()

        #         .generate(2)
        #         .show("gmsh")
        #     )

    def to_geo(
        self,
        mesh_params: TurboMeshParameters = TurboMeshParameters(),
        sim_params: Optional[SimulationParams] = None,
    ):
        return self.to_unstructured_geo(mesh_params, sim_params)

    def visualize(self, filename: Optional[str] = None):
        fig = go.Figure(layout=go.Layout(title=go.layout.Title(text="Airfoil")))

        top_coords, bottom_coords = self.get_coords()
        top_airfoil_coords, bottom_airfoil_coords = self.get_airfoil_coords()

        fig.add_trace(
            go.Scatter(x=top_coords[:, 0], y=top_coords[:, 1], name=f"Top Outline")
        )

        fig.add_trace(
            go.Scatter(
                x=bottom_coords[:, 0], y=bottom_coords[:, 1], name=f"Bottom Outline"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=top_airfoil_coords[:, 0],
                y=top_airfoil_coords[:, 1],
                name=f"Top Airfoil",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=bottom_airfoil_coords[:, 0],
                y=bottom_airfoil_coords[:, 1],
                name=f"Bottom Airfoil",
            )
        )

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        if filename:
            fig.write_image(filename, width=500, height=500)
        else:
            fig.show()
