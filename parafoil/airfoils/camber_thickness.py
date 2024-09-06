from dataclasses import dataclass, field, asdict
from functools import cached_property
from typing import List, Literal, Optional
import numpy as np
import plotly.graph_objects as go
import numpy.typing as npt
from parafoil.metadata import opt_constant, opt_range, opt_tol_range
from parafoil.utils import get_bspline, get_sampling
from .airfoil import Airfoil


def get_thickness_dist_ctrl_pnts(
    camber: npt.NDArray,
    camber_normal: npt.NDArray,
    thickness_dist: npt.NDArray,
    sampling: npt.NDArray,
    degree: int
):
    "get thickness distribution control points"
    camber_normal_thickness = get_bspline(thickness_dist, degree)(sampling)
    return np.concatenate([
        [camber[0]],
        camber + camber_normal*camber_normal_thickness,
        [camber[-1]]
    ])


@dataclass
class CamberThicknessAirfoil(Airfoil):
    "parametric airfoil using B-splines"

    inlet_angle: float = field(metadata=opt_tol_range(np.radians(-10), np.radians(10)))
    "inlet angle (deg or rad depending on angle_units)"

    outlet_angle: float = field(metadata=opt_tol_range(np.radians(-10), np.radians(10)))
    "outlet angle (deg or rad depending on angle_units)"

    upper_thick_prop: List[float] = field(metadata=opt_range(0.01, 0.05))
    "upper thickness proportion to chord length (length)"

    lower_thick_prop: List[float] = field(metadata=opt_range(0.01, 0.05))
    "lower thickness proportion to chord length (length)"

    leading_prop: float = field(metadata=opt_constant())
    "leading edge tangent line proportion [0.0-1.0] (dimensionless)"

    trailing_prop: float = field(metadata=opt_constant())
    "trailing edge tangent line proportion [0.0-1.0] (dimensionless)"

    chord_length: float = field(default=1.0, metadata=opt_constant())
    "chord length (length)"

    num_samples: int = 50
    "number of samples"

    is_cosine_sampling: bool = True
    "use cosine sampling"

    leading_ctrl_pnt: List[float] = field(default_factory=lambda: [0.0, 0.0])
    "leading control point (length)"

    angle_units: Literal["rad", "deg"] = "rad"
    "angle units"

    def __post_init__(self):
        self.inlet_angle = self.inlet_angle
        self.outlet_angle = self.outlet_angle

        self.inlet_angle_rad = np.radians(self.inlet_angle) if self.angle_units == "deg" else self.inlet_angle
        self.outlet_angle_rad = np.radians(self.outlet_angle) if self.angle_units == "deg" else self.outlet_angle



        if self.upper_thick_prop is not None:
            self.upper_thick_dist = [self.chord_length*prop for prop in self.upper_thick_prop]

        if self.lower_thick_prop is not None:
            self.lower_thick_dist = [self.chord_length*prop for prop in self.lower_thick_prop]

        assert self.upper_thick_dist is not None and self.lower_thick_dist is not None

        self.stagger_angle = (self.inlet_angle + self.outlet_angle)/2
        self.stagger_angle_rad = np.radians(self.stagger_angle) if self.angle_units == "deg" else self.stagger_angle


        self.degree = 3
        self.num_thickness_dist_pnts = len(self.upper_thick_dist) + 4
        self.thickness_dist_sampling = np.linspace(0, 1, self.num_thickness_dist_pnts, endpoint=True)
        self.camber_bspline = get_bspline(self.camber_ctrl_pnts, self.degree)
        self.sampling = get_sampling(self.num_samples, self.is_cosine_sampling)
        self.axial_chord_length = self.chord_length*np.cos(self.stagger_angle_rad)
        self.height = self.chord_length*np.sin(self.stagger_angle_rad)

        self.center_offset = np.array([
            -self.axial_chord_length/2, 
            -self.height/2
        ])


    def mutate(self, **kwargs):
        return CamberThicknessAirfoil(**{**asdict(self), **kwargs})

    @cached_property
    def camber_ctrl_pnts(self):
        p_le = np.array(self.leading_ctrl_pnt)

        p_te = p_le + np.array([
            self.chord_length*np.cos(self.stagger_angle_rad),
            self.chord_length*np.sin(self.stagger_angle_rad)
        ])

        # leading edge tangent control point
        p1 = p_le + self.leading_prop*np.array([
            self.chord_length*np.cos(self.inlet_angle_rad),
            self.chord_length*np.sin(self.inlet_angle_rad)
        ])

        # trailing edge tangent control point
        p2 = p_te - self.trailing_prop*np.array([
            self.chord_length*np.cos(self.outlet_angle_rad),
            self.chord_length*np.sin(self.outlet_angle_rad)
        ])

        camber_ctrl_pnts = np.vstack((p_le, p1, p2, p_te))
        return camber_ctrl_pnts

    @cached_property
    def top_ctrl_pnts(self):
        "upper side bspline"
        assert self.upper_thick_dist is not None, "upper thickness distribution is not defined"
        thickness_dist = np.vstack(self.upper_thick_dist)
        return get_thickness_dist_ctrl_pnts(
            self.camber_coords,
            self.camber_normal_coords,
            thickness_dist,
            self.thickness_dist_sampling,
            self.degree
        )

    @cached_property
    def bottom_ctrl_pnts(self):
        "lower side bspline"
        assert self.lower_thick_dist is not None, "lower thickness distribution is not defined"
        thickness_dist = -np.vstack(self.lower_thick_dist)
        return get_thickness_dist_ctrl_pnts(
            self.camber_coords,
            self.camber_normal_coords,
            thickness_dist,
            self.thickness_dist_sampling,
            self.degree
        )

    @cached_property
    def camber_coords(self) -> npt.NDArray:
        "camber line coordinates"
        return self.camber_bspline(self.thickness_dist_sampling)

    @cached_property
    def camber_normal_coords(self) -> npt.NDArray:
        "camber normal line coordinates"
        camber_prime = self.camber_bspline.derivative(1)(self.thickness_dist_sampling)
        normal = np.array([-camber_prime[:, 1], camber_prime[:, 0]]) / np.linalg.norm(camber_prime, axis=1)
        return normal.T

    def get_coords(self):
        "airfoil coordinates"
        top_coords = get_bspline(self.top_ctrl_pnts, self.degree)(self.sampling)
        bottom_coords = get_bspline(self.bottom_ctrl_pnts, self.degree)(self.sampling)
        return np.concatenate([top_coords[1:-1], bottom_coords[::-1]])

    def clone(
        self,
        inlet_angle: Optional[float] = None,
        outlet_angle: Optional[float] = None,
        upper_thick_prop: Optional[List[float]] = None,
        lower_thick_prop: Optional[List[float]] = None,
        leading_prop: Optional[float] = None,
        trailing_prop: Optional[float] = None,
        chord_length: Optional[float] = None,
        num_samples: Optional[int] = None,
        is_cosine_sampling: Optional[bool] = None,
        leading_ctrl_pnt: Optional[List[float]] = None,
        angle_units: Optional[Literal["rad", "deg"]] = None
    ):
        return CamberThicknessAirfoil(
            inlet_angle=inlet_angle if inlet_angle is not None else self.inlet_angle,
            outlet_angle=outlet_angle if outlet_angle is not None else self.outlet_angle,
            upper_thick_prop=upper_thick_prop if upper_thick_prop is not None else [*self.upper_thick_prop],
            lower_thick_prop=lower_thick_prop if lower_thick_prop is not None else [*self.lower_thick_prop],
            leading_prop=leading_prop if leading_prop is not None else self.leading_prop,
            trailing_prop=trailing_prop if trailing_prop is not None else self.trailing_prop,
            chord_length=chord_length if chord_length is not None else self.chord_length,
            num_samples=num_samples if num_samples is not None else self.num_samples,
            is_cosine_sampling=is_cosine_sampling if is_cosine_sampling is not None else self.is_cosine_sampling,
            leading_ctrl_pnt=leading_ctrl_pnt if leading_ctrl_pnt is not None else self.leading_ctrl_pnt,
            angle_units=angle_units if angle_units is not None else self.angle_units
        )

    def visualize(
        self,
        include_camber=True,
        include_camber_ctrl_pnts=False,
        filename: Optional[str] = None
    ):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Airfoil"))
        )
        if include_camber_ctrl_pnts:
            fig.add_trace(go.Scatter(
                x=self.camber_ctrl_pnts[:, 0],
                y=self.camber_ctrl_pnts[:, 1],
                name=f"Camber Control Points"
            ))

        if include_camber:
            camber_coords = self.camber_coords
            fig.add_trace(go.Scatter(
                x=camber_coords[:, 0],
                y=camber_coords[:, 1],
                name=f"Camber"
            ))

        coords = self.get_coords()
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            name=f"Airfoil"
        ))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        if filename:
            fig.write_image(filename, width=500, height=500)
        else:
            fig.show()
