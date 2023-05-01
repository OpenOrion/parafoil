from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np
import numpy.typing as npt
from plotly import graph_objects as go
from parafoil.airfoils import Airfoil, BSplineAirfoil
from parafoil.utils import get_bspline, get_sampling
from ezmesh import Geometry, CurveLoop, PlaneSurface, BoundaryLayerField
from paraflow import Passage, FlowState


@dataclass
class TurboMeshParameters:
    top_label: str = "top"
    bottom_label: str = "bottom"
    inlet_label: str = "inlet"
    outlet_label: str = "outlet"
    airfoil_label: Union[str, List[str]] = "airfoil"
    airfoil_mesh_size: Optional[float] = None
    boundary_layer_thickness: Optional[float] = None
    boundary_wall_mesh_size: Optional[float] = None
    passage_mesh_size: Optional[float] = None


@dataclass
class TurboRowPassage(Passage):
    airfoil: BSplineAirfoil
    "airfoil for the passage"

    spacing: float
    "spacing between blades"

    leading_edge_gap: float
    "gap between the leading edge of the airfoil and the passage"

    trailing_edge_gap: float
    "gap between the trailing edge of the airfoil and the passage"

    num_airfoils: int = 1
    "number of blades in the passage"

    offset: List[float] = field(default_factory=lambda: [0, 0])
    "offset of the passage"

    num_samples: int = 50
    "number of samples"

    is_cosine_sampling: bool = True
    "use cosine sampling"

    mesh_params: TurboMeshParameters = field(default_factory=TurboMeshParameters)
    "mesh parameters for the passage"

    def __post_init__(self):
        self.sampling = get_sampling(self.num_samples, self.is_cosine_sampling)
        self.width = self.airfoil.axial_chord_length + self.leading_edge_gap + self.trailing_edge_gap

    @cached_property
    def total_spacing(self):
        return self.spacing * self.num_airfoils

    def get_coords(self) -> npt.NDArray[np.float64]:
        passage_ctrl_pnts = self.get_ctrl_pnts()
        passage_bspline = get_bspline(passage_ctrl_pnts, 3)
        passage_bspline_coords = passage_bspline(self.sampling)
        top_coords = passage_bspline_coords + np.array([0, self.total_spacing/2]) + self.offset
        bottom_coords = np.flip(passage_bspline_coords, axis=0) + np.array([0, -self.total_spacing/2]) + self.offset
        return np.concatenate((top_coords, bottom_coords))

    def get_ctrl_pnts(self):
        ctrl_coords = self.airfoil.camber_coords + np.array([self.leading_edge_gap, 0])
        return np.concatenate([
            np.array([[0, 0], [self.leading_edge_gap, 0]]),
            ctrl_coords,
            np.array([ctrl_coords[-1] + np.array([self.trailing_edge_gap, 0])])    # type: ignore
        ])

    @cached_property
    def airfoils_coords(self):
        airfoil_coords = self.airfoil.get_coords()
        airfoil_leading_pnt = airfoil_coords[np.argmin(airfoil_coords[:, 0])]

        airfoils_coords = []
        airfoil_offset = np.array([
            self.leading_edge_gap-np.min(airfoil_coords[:, 0]),
            (self.total_spacing/2) - (self.spacing/2) - airfoil_leading_pnt[1]
        ])
        for i in range(self.num_airfoils):
            airfoil_offseted_coords = airfoil_coords+airfoil_offset-np.array([0, i*self.spacing]) + self.offset  # type: ignore
            airfoils_coords.append(airfoil_offseted_coords)
        return airfoils_coords

    def get_mesh(
        self,
        output_path: Optional[str] = None
    ):
        if self.mesh_params.airfoil_mesh_size is None:
            self.mesh_params.airfoil_mesh_size = 0.02 * self.airfoil.chord_length
        if self.mesh_params.boundary_layer_thickness is None:
            self.mesh_params.boundary_layer_thickness = 0.01 * self.airfoil.chord_length
        if self.mesh_params.boundary_wall_mesh_size is None:
            self.mesh_params.boundary_wall_mesh_size = 0.001 * self.airfoil.chord_length
        if self.mesh_params.passage_mesh_size is None:
            self.mesh_params.passage_mesh_size = 0.05 * self.airfoil.chord_length

        with Geometry() as geometry:
            passage_ctrl_pnts = self.get_ctrl_pnts()
            passage_curve_loop = CurveLoop.from_coords(
                [
                    ("BSpline", passage_ctrl_pnts + np.array([0, self.total_spacing/2]) + self.offset),
                    ("BSpline", passage_ctrl_pnts[::-1] + np.array([0, -self.total_spacing/2]) + self.offset)
                ],
                mesh_size=self.mesh_params.passage_mesh_size,
                labels=[self.mesh_params.top_label, self.mesh_params.outlet_label, self.mesh_params.bottom_label, self.mesh_params.inlet_label],
            )

            if isinstance(self.mesh_params.airfoil_label, List):
                assert len(self.mesh_params.airfoil_label) == len(self.airfoils_coords)

            airfoil_curve_loops = [
                CurveLoop.from_coords(
                    airfoil_coords,
                    mesh_size=self.mesh_params.airfoil_mesh_size,
                    labels=self.mesh_params.airfoil_label[i] if isinstance(self.mesh_params.airfoil_label, List) else self.mesh_params.airfoil_label,
                    fields=[
                        BoundaryLayerField(
                            hwall_n=self.mesh_params.boundary_wall_mesh_size,
                            thickness=self.mesh_params.boundary_layer_thickness,
                            is_quad_mesh=True,
                            intersect_metrics=False
                        )
                    ],
                )
                for i, airfoil_coords in enumerate(self.airfoils_coords)
            ]

            surface = PlaneSurface(
                outlines=[passage_curve_loop],
                holes=airfoil_curve_loops
            )
            if output_path is not None:
                geometry.write(output_path)

            mesh = geometry.generate(surface)
            # mesh.add_target_point(f"mid_{self.mesh_params.outlet_label}", self.mesh_params.outlet_label, 0.5)
            return mesh

    def get_config(self, inlet_total_state: FlowState, working_directory: str, id: str, target_outlet_static_state: FlowState):
        return {
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SST",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
            "SYSTEM_MEASUREMENTS": "SI",
            "MACH_NUMBER": inlet_total_state.mach_number,
            "AOA": 0.0,
            "SIDESLIP_ANGLE": 0.0,
            "INIT_OPTION": "TD_CONDITIONS",
            "FREESTREAM_OPTION": "TEMPERATURE_FS",
            "FREESTREAM_PRESSURE": inlet_total_state.P,
            "FREESTREAM_TEMPERATURE": inlet_total_state.T,
            "FREESTREAM_DENSITY": inlet_total_state.rho_mass(),
            "REF_DIMENSIONALIZATION": "DIMENSIONAL",
            "FLUID_MODEL": "IDEAL_GAS",
            "GAMMA_VALUE": inlet_total_state.gamma,
            "GAS_CONSTANT": inlet_total_state.gas_constant,
            "VISCOSITY_MODEL": "SUTHERLAND",
            "MU_REF": 1.716E-5,
            "MU_T_REF": 273.15,
            "SUTHERLAND_CONSTANT": 110.4,
            "CONDUCTIVITY_MODEL": "CONSTANT_PRANDTL",
            "AVERAGE_PROCESS_KIND": "MIXEDOUT",
            "PERFORMANCE_AVERAGE_PROCESS_KIND": "MIXEDOUT",
            "MIXEDOUT_COEFF": "(1.0, 1.0E-05, 15)",
            "AVERAGE_MACH_LIMIT": 0.05,
            "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
            "CFL_NUMBER": 10.0,
            "CFL_ADAPT": "NO",
            "CFL_ADAPT_PARAM": "( 1.3, 1.2, 1.0, 10.0)",
            "LINEAR_SOLVER": "FGMRES",
            "LINEAR_SOLVER_PREC": "LU_SGS",
            "LINEAR_SOLVER_ERROR": 1E-6,
            "LINEAR_SOLVER_ITER": 10,
            "VENKAT_LIMITER_COEFF": 0.05,
            "LIMITER_ITER": 999999,
            "CONV_NUM_METHOD_FLOW": "ROE",
            "MUSCL_FLOW": "YES",
            "SLOPE_LIMITER_FLOW": "VAN_ALBADA_EDGE",
            "ENTROPY_FIX_COEFF": 0.1,
            "JST_SENSOR_COEFF": (0.5, 0.02),
            "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
            "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND",
            "TIME_DISCRE_TURB": "EULER_IMPLICIT",
            "CFL_REDUCTION_TURB": 1.0,
            "OUTER_ITER": 21,
            "CONV_RESIDUAL_MINVAL": -16,
            "CONV_STARTITER": 10,
            "CONV_CAUCHY_ELEMS": 100,
            "CONV_CAUCHY_EPS": 1E-6,
            "MESH_FILENAME": f"{working_directory}/passage{id}.su2",
            "MESH_FORMAT": "SU2",
            "TABULAR_FORMAT": "CSV",
            "VOLUME_FILENAME": f"{working_directory}/flow{id}",
            "RESTART_FILENAME":  f"{working_directory}/restart_flow{id}.dat",
            "SURFACE_FILENAME":  f"{working_directory}/surface_flow{id}",
            "CONV_FILENAME": f"{working_directory}/history{id}",
            "OUTPUT_WRT_FREQ": 1000,
            "SCREEN_OUTPUT": "OUTER_ITER, AVG_BGS_RES[0], AVG_BGS_RES[1], RMS_DENSITY[0], RMS_ENERGY[0], RMS_DENSITY[1], RMS_ENERGY[1], SURFACE_TOTAL_PRESSURE[1]"
        }

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurboStagePassage(Passage):
    inflow_passage: TurboRowPassage
    "Passage with inflow boundary condition"

    outflow_passage: TurboRowPassage
    "Passage with outflow boundary condition"

    def __post_init__(self):
        self.outflow_passage.offset = [self.inflow_passage.width, 0]

    def get_config(self, inlet_total_state: FlowState, working_directory: str, id: str, target_outlet_static_state: FlowState):
        inflow_config = self.inflow_passage.get_config(inlet_total_state, working_directory, id, target_outlet_static_state)
        inflow_mesh_params = self.inflow_passage.mesh_params
        outflow_mesh_params = self.outflow_passage.mesh_params
        return {
            **inflow_config,
            "MULTIZONE": "YES",
            "CONFIG_LIST": {
                "zone_1.cfg": {
                    **({
                        "GRID_MOVEMENT": "STEADY_TRANSLATION",
                        "MACH_MOTION": 0.35,
                        "TRANSLATION_RATE": f"0.0 {inlet_total_state.translation_velocity} 0.0",
                    } if inlet_total_state.translation_velocity else {"GRID_MOVEMENT": "NONE"})
                },
                "zone_2.cfg": {
                    **({
                        "GRID_MOVEMENT": "STEADY_TRANSLATION",
                        "MACH_MOTION": 0.35,
                        "TRANSLATION_RATE": f"0.0 {target_outlet_static_state.translation_velocity} 0.0",
                    } if target_outlet_static_state.translation_velocity else {"GRID_MOVEMENT": "NONE"})
                }
            },
            "MARKER_HEATFLUX": f"( {inflow_mesh_params.airfoil_label}, 0.0, {outflow_mesh_params.airfoil_label}, 0.0)",
            "MARKER_PERIODIC": f"( {inflow_mesh_params.bottom_label}, {inflow_mesh_params.top_label}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {self.inflow_passage.spacing}, 0.0, {outflow_mesh_params.bottom_label}, {outflow_mesh_params.top_label}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {self.outflow_passage.spacing}, 0.0)",
            "MARKER_TURBOMACHINERY": f"({inflow_mesh_params.inlet_label}, {inflow_mesh_params.outlet_label}, {outflow_mesh_params.inlet_label}, {outflow_mesh_params.outlet_label})",
            "MARKER_ANALYZE": "(outflow)",

            "MARKER_MIXINGPLANE_INTERFACE": f"({inflow_mesh_params.outlet_label}, {outflow_mesh_params.inlet_label})",
            "MARKER_GILES": f"({inflow_mesh_params.inlet_label}, TOTAL_CONDITIONS_PT, {inlet_total_state.P}, {inlet_total_state.T}, 1.0, 0.0, 0.0,1.0,1.0, {inflow_mesh_params.outlet_label}, MIXING_OUT, 0.0, 0.0, 0.0, 0.0, 0.0,1.0,1.0, {outflow_mesh_params.inlet_label}, MIXING_IN, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 1.0 {outflow_mesh_params.outlet_label}, STATIC_PRESSURE, {target_outlet_static_state.P}, 0.0, 0.0, 0.0, 0.0,1.0,1.0)",
            "SPATIAL_FOURIER": "NO",
            "TURBOMACHINERY_KIND": "AXIAL AXIAL",
            "TURBULENT_MIXINGPLANE": "YES",
            "RAMP_OUTLET_PRESSURE": "NO",
            "RAMP_OUTLET_PRESSURE_COEFF": "(140000.0, 10.0, 2000)",
            "MARKER_PLOTTING": f"({inflow_mesh_params.airfoil_label}, {outflow_mesh_params.airfoil_label})",
        }

    def get_mesh(self):
        return [self.inflow_passage.get_mesh(), self.outflow_passage.get_mesh()]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
