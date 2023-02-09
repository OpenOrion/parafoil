from dataclasses import dataclass, field
from functools import cached_property
from typing import List
import numpy as np
import plotly.graph_objects as go
import numpy.typing as npt
from parafoil.bspline import get_bspline


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
class Airfoil:
    "parametric airfoil using B-splines"

    inlet_angle: float
    "inlet angle (rad)"

    outlet_angle: float
    "outlet angle (rad)"

    stagger_angle: float
    "stagger angle (rad)"

    upper_thick_dist: List[float]
    "upper thickness distribution (length)"

    lower_thick_dist: List[float]
    "lower thickness distribution (length)"

    leading_prop: float
    "leading edge tangent line proportion [0.0-1.0] (dimensionless)"

    trailing_prop: float
    "trailing edge tangent line proportion [0.0-1.0] (dimensionless)"

    chord_length: float = 1.0
    "chord length (length)"

    num_samples: int = 50
    "number of samples"

    degree: int = 3
    "degree of bspline"

    leading_ctrl_pnt: List[float] = field(default_factory=lambda: [0.0, 0.0])
    "leading control point (length)"

    def __post_init__(self):
        self.num_thickness_dist_pnts = len(self.upper_thick_dist) + 4
        self.thickness_dist_sampling = np.linspace(0, 1, self.num_thickness_dist_pnts, endpoint=True)
        self.sampling = np.linspace(0, 1, self.num_samples, endpoint=True)
    @cached_property
    def axial_chord_length(self):
        "axial chord length (length)"
        return self.chord_length*np.cos(self.stagger_angle)

    @cached_property
    def camber_bspline(self):
        "camber line bspline"
        p_le = np.array(self.leading_ctrl_pnt)

        p_te = p_le + np.array([
            self.chord_length*np.cos(self.stagger_angle),
            self.chord_length*np.sin(self.stagger_angle)
        ])

        # leading edge tangent control point
        p1 = p_le + self.leading_prop*np.array([
            self.chord_length*np.cos(self.inlet_angle),
            self.chord_length*np.sin(self.inlet_angle)
        ])

        # trailing edge tangent control point
        p2 = p_te - self.trailing_prop*np.array([
            self.chord_length*np.cos(self.outlet_angle),
            self.chord_length*np.sin(self.outlet_angle)
        ])

        ctrl_pnts = np.vstack((p_le, p1, p2, p_te))
        return get_bspline(ctrl_pnts, self.degree)

    @cached_property
    def upper_side_bspline(self):
        "upper side bspline"
        thickness_dist = np.vstack(self.upper_thick_dist)
        ctrl_pnts = get_thickness_dist_ctrl_pnts(
            self.camber_coords,
            self.camber_normal_coords,
            thickness_dist,
            self.thickness_dist_sampling,
            self.degree
        )
        return get_bspline(ctrl_pnts, self.degree)

    @cached_property
    def lower_side_bspline(self):
        "lower side bspline"
        thickness_dist = -np.vstack(self.lower_thick_dist)
        ctrl_pnts = get_thickness_dist_ctrl_pnts(
            self.camber_coords,
            self.camber_normal_coords,
            thickness_dist,
            self.thickness_dist_sampling,
            self.degree
        )
        return get_bspline(ctrl_pnts, self.degree)

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

    @cached_property
    def coords(self) -> npt.NDArray:
        "airfoil coordinates"
        upper_side = self.upper_side_bspline(self.sampling)
        lower_side = self.lower_side_bspline(self.sampling)
        return np.concatenate([upper_side[1:-1], np.flip(lower_side, axis=0)])

    def visualize(
        self,
        include_camber=True,
        include_camber_ctrl_pnts=False,
    ):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Airfoil"))
        )
        if include_camber_ctrl_pnts:
            camber_ctr_pnts = self.camber_bspline.c
            fig.add_trace(go.Scatter(
                x=camber_ctr_pnts[:, 0],
                y=camber_ctr_pnts[:, 1],
                name=f"Camber Control Points"

            ))

        if include_camber:
            camber_coords = self.camber_coords
            fig.add_trace(go.Scatter(
                x=camber_coords[:, 0],
                y=camber_coords[:, 1],
                name=f"Camber"
            ))

        coords = self.coords
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            name=f"Airfoil"
        ))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()
