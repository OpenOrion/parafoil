from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional
import numpy as np
from plotly import graph_objects as go
from paraflow import Passage, FlowState
from parafoil.airfoils import Airfoil
from ezmesh import Geometry, CurveLoop, PlaneSurface
from parafoil.utils import get_sampling


@dataclass
class CircularMeshParameters:
    farfield_label: str = "farfield"
    airfoil_label: str = "airfoil"
    airfoil_mesh_size: Optional[float] = None
    passage_mesh_size: Optional[float] = None


@dataclass
class CircularPassage(Passage):
    airfoil: Airfoil
    "airfoil for the passage"

    radius: float
    "radius of circular passage"

    num_samples: int = 50
    "number of samples"

    mesh_params: CircularMeshParameters = field(default_factory=CircularMeshParameters)
    "mesh parameters for the passage"

    def __post_init__(self):
        self.sampling = get_sampling(self.num_samples, is_cosine_sampling=False)

    def get_coords(self):
        theta = self.sampling * 2 * np.pi
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        return np.column_stack((x, y))

    @cached_property
    def surface(self):
        if self.mesh_params.airfoil_mesh_size is None:
            self.mesh_params.airfoil_mesh_size = 0.1 * self.airfoil.chord_length
        if self.mesh_params.passage_mesh_size is None:
            self.mesh_params.passage_mesh_size = self.airfoil.chord_length

        airfoil_coords = self.airfoil.get_coords()
        airfoil_curve_loop = CurveLoop.from_coords(
            airfoil_coords,
            mesh_size=self.mesh_params.airfoil_mesh_size,
            label=self.mesh_params.airfoil_label,
        )

        farfield_coords = self.get_coords()
        farfield_curve_loop = CurveLoop.from_coords(
            farfield_coords,
            mesh_size=self.mesh_params.passage_mesh_size,
            label=self.mesh_params.farfield_label,
            holes=[airfoil_curve_loop]
        )



        return PlaneSurface(
            outlines=[farfield_curve_loop],
        )

    def get_mesh(
        self,
        output_path: Optional[str] = None
    ):
        with Geometry() as geometry:
            if output_path is not None:
                geometry.write(output_path)

            mesh = geometry.generate(self.surface)
            return mesh

    def visualize(self, title: str = "Passage"):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text=title))
        )
        coords = self.get_coords()
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            fill="toself",
            legendgroup="passage",
            name=f"Passage"
        ))

        airfoil_coords = self.airfoil.get_coords()
        fig.add_trace(go.Scatter(
            x=airfoil_coords[:, 0],
            y=airfoil_coords[:, 1],
            fill="toself",
            legendgroup="airfoil",
            legendgrouptitle_text="Airfoils",
            name=f"Airfoil"

        ))
        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()

    def get_config(
            self, 
            inlet_total_state: FlowState, 
            working_directory: str, 
            id: str, 
            target_outlet_static_state: Optional[FlowState] = None, 
            angle_of_attack: float = 0.0
        ) -> Dict[str, Any]:
        return {
            "SOLVER": "EULER",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
            "MACH_NUMBER": inlet_total_state.mach_number,
            "AOA": angle_of_attack,
            "FREESTREAM_PRESSURE": inlet_total_state.P,
            "FREESTREAM_TEMPERATURE": inlet_total_state.T,
            "GAMMA_VALUE": inlet_total_state.gamma,
            "GAS_CONSTANT": inlet_total_state.gas_constant,
            "REF_ORIGIN_MOMENT_X": 0.25,
            "REF_ORIGIN_MOMENT_Y": 0.00,
            "REF_ORIGIN_MOMENT_Z": 0.00,
            "REF_LENGTH": 1.0,
            "REF_AREA": 1.0,
            "REF_DIMENSIONALIZATION": "DIMENSIONAL",
            "MARKER_EULER": f"({self.mesh_params.airfoil_label})",
            "MARKER_FAR":  f"({self.mesh_params.farfield_label})",
            "MARKER_PLOTTING": f"({self.mesh_params.airfoil_label})",
            "MARKER_MONITORING": f"({self.mesh_params.airfoil_label})",
            "MARKER_DESIGNING": f"({self.mesh_params.airfoil_label})",
            "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
            "OBJECTIVE_FUNCTION": "DRAG",
            "CFL_NUMBER": 1e3,
            "CFL_ADAPT": "NO",
            "CFL_ADAPT_PARAM": "(0.1, 2.0, 10.0, 1e10)",
            "ITER": 50,
            "LINEAR_SOLVER": "FGMRES",
            "LINEAR_SOLVER_PREC": "ILU",
            "LINEAR_SOLVER_ERROR": 1E-10,
            "LINEAR_SOLVER_ITER": 10,
            "MGLEVEL": 3,
            "MGCYCLE": "W_CYCLE",
            "MG_PRE_SMOOTH": (1, 2, 3, 3),
            "MG_POST_SMOOTH": (0, 0, 0, 0),
            "MG_CORRECTION_SMOOTH": (0, 0, 0, 0),
            "MG_DAMP_RESTRICTION": 1.0,
            "MG_DAMP_PROLONGATION": 1.0,
            "CONV_NUM_METHOD_FLOW": "JST",
            "SLOPE_LIMITER_FLOW": "VENKATAKRISHNAN_WANG",
            "JST_SENSOR_COEFF": (0.5, 0.02),
            "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
            "CONV_NUM_METHOD_ADJFLOW": "JST",
            "SLOPE_LIMITER_ADJFLOW": "NONE",
            "CFL_REDUCTION_ADJFLOW": 0.01,
            "TIME_DISCRE_ADJFLOW": "EULER_IMPLICIT",
            "CONV_FIELD": "RMS_DENSITY",
            "CONV_RESIDUAL_MINVAL": -8,
            "CONV_STARTITER": 10,
            "CONV_CAUCHY_ELEMS": 100,
            "CONV_CAUCHY_EPS": 1E-6,
            "SCREEN_OUTPUT": "(INNER_ITER, WALL_TIME, RMS_RES, LIFT, DRAG, CAUCHY_SENS_PRESS, CAUCHY_DRAG RMS_ADJ_DENSITY RMS_ADJ_ENERGY)",
            "MESH_FILENAME": f"{working_directory}/passage{id}.su2",
            "MESH_FORMAT": "SU2",
            "TABULAR_FORMAT": "CSV",
            "VOLUME_FILENAME": f"{working_directory}/flow{id}",
            "RESTART_FILENAME":  f"{working_directory}/restart_flow{id}.dat",
            "SURFACE_FILENAME":  f"{working_directory}/surface_flow{id}",
            "OUTPUT_WRT_FREQ": 250,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
