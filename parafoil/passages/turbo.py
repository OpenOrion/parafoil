from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import numpy as np
import numpy.typing as npt
from paraflow.passages.passage import SimulationParams
from plotly import graph_objects as go
from parafoil.airfoils import CamberThicknessAirfoil
from parafoil.metadata import opt_class, opt_constant
from paraflow import Passage, SimulationParams
from meshql import GeometryQL
import cadquery as cq
from parafoil.passages.utils import get_wall_distance
from parafoil.utils import get_bspline, get_sampling

def get_y_plus(chord_length: float, sim_params: Optional[SimulationParams] = None):
    if sim_params:
        return get_wall_distance(
            rho=sim_params.inlet_total_state.rho_mass(),
            Uf=sim_params.inlet_total_state.freestream_velocity,
            mu=sim_params.inlet_total_state.mu(), # type: ignore
            L=chord_length,
            y_plus_desired=1.0
        )
    else:
        return 0.001 * chord_length


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
class TurboRowPassage(Passage):
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
        self.leading_edge_gap = self.leading_edge_gap_to_chord * self.airfoil.chord_length
        self.trailing_edge_gap = self.trailing_edge_gap_to_chord * self.airfoil.chord_length
        self.width = self.airfoil.axial_chord_length + self.leading_edge_gap + self.trailing_edge_gap
        self.spacing = self.airfoil.chord_length * self.spacing_to_chord
        self.height = self.spacing * self.num_airfoils
        self.sampling = get_sampling(self.num_samples, False)

    @cached_property
    def total_spacing(self):
        return self.spacing * self.num_airfoils

    def get_ctrl_pnts(self, type: Literal["top", "bottom", "camber"] = "top", is_centerd: bool = True) -> npt.NDArray[np.float64]:
        if type == "camber":
            ctrl_coords = self.airfoil.camber_coords
        elif type == "top":
            ctrl_coords = self.airfoil.top_ctrl_pnts
        elif type == "bottom":
            ctrl_coords = self.airfoil.bottom_ctrl_pnts

        if self.type != "line" and type in ["top", "bottom"]:
            ctrl_coords = ctrl_coords[1:-1]

        if self.type == "line":
            ctrl_coords = np.array([
                ctrl_coords[0],
                ctrl_coords[-1]
            ])

        passage_length = np.array([self.width, ctrl_coords[-1][1]])
        ctrl_pnts =  np.array([
            [0, 0],
            *(ctrl_coords + np.array([self.leading_edge_gap, 0])),
            passage_length,
        ])

        return ctrl_pnts
    
    def get_airfoil_profile(self, workplane: cq.Workplane = cq.Workplane("XY")):
        top_coords = get_bspline(self.airfoil.top_ctrl_pnts, self.airfoil.degree)(self.airfoil.sampling)
        bottom_coords = get_bspline(self.airfoil.bottom_ctrl_pnts, self.airfoil.degree)(self.airfoil.sampling)
        center_offset = np.array([self.leading_edge_gap + self.offset[0], self.offset[1]])
        return (
            workplane
            .spline(top_coords + center_offset)
            .spline(bottom_coords[::-1] + center_offset)
            .close()
        )

    def get_profile(self, workplane: cq.Workplane = cq.Workplane("XY"), with_airfoils: bool = True):

        if self.type == "camber":
            top_ctrl_pnts = bottom_ctrl_pnts = self.get_ctrl_pnts("camber")
        else: 
            top_ctrl_pnts = self.get_ctrl_pnts("top")
            bottom_ctrl_pnts = self.get_ctrl_pnts("bottom")

        top_ctrl_pnts = top_ctrl_pnts + np.array([0, self.total_spacing/2])
        bottom_ctrl_pnts = bottom_ctrl_pnts + np.array([0, -self.total_spacing/2])


        top_coords = get_bspline(top_ctrl_pnts, self.degree)(self.sampling)
        bottom_coords = get_bspline(bottom_ctrl_pnts, self.degree)(self.sampling)
        center_offset = np.array(self.offset)
        profile = (
            workplane
            .spline(top_coords + center_offset)
            .lineTo(*bottom_coords[-1] + center_offset)
            .spline(bottom_coords[::-1] + center_offset)
            
            .close()
        )

        if with_airfoils:
            profile = self.get_airfoil_profile(profile)

        return profile

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self, airfoil: CamberThicknessAirfoil):
        return TurboRowPassage(
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

    def to_unstructured_geo(self, mesh_params: TurboMeshParameters = TurboMeshParameters(), sim_params: Optional[SimulationParams] = None) -> GeometryQL:
        if mesh_params.airfoil_mesh_size is None:
            mesh_params.airfoil_mesh_size = 0.02 * self.airfoil.chord_length
        if mesh_params.boundary_wall_mesh_size is None:
            mesh_params.boundary_wall_mesh_size = 0.01 * self.airfoil.chord_length
        if mesh_params.passage_mesh_size is None:
            mesh_params.passage_mesh_size = 0.025 * self.airfoil.chord_length

        with GeometryQL() as geo:
            y_plus = get_y_plus(self.airfoil.chord_length, sim_params)
            profile = self.get_profile()
            return (
                geo
                .load(profile)
                
                .edges(type="interior")
                .setMeshSize(mesh_params.airfoil_mesh_size)
                .addBoundaryLayer(y_plus, mesh_params.boundary_wall_ratio, mesh_params.boundary_num_layers)
                .addPhysicalGroup(mesh_params.airfoil_label)
                .end()
                
                .edges(type="exterior")
                .setMeshSize(mesh_params.passage_mesh_size)
                .end()

                .generate(2)
                .show("gmsh")
            )

    def to_geo(self, mesh_params: TurboMeshParameters = TurboMeshParameters(), sim_params: Optional[SimulationParams] = None):
        return self.to_unstructured_geo(mesh_params, sim_params)


@dataclass
class TurboRow3DPassage(Passage):
    row_passages: Sequence[TurboRowPassage] = field(metadata=opt_class())
    
    radii: Sequence[float]
    "radii of the passages"

    def get_meshes(self, params: SimulationParams | None = None):
        with GeometryQL() as geo:
            (
                geo
                .load(self.get_profile())
                .refine(1)
                .generate()
                .show("gmsh")
            )

    def get_profile(self, params: Optional[SimulationParams] = None):
        passage_profile = cq.Workplane("XY")
        airfoil_profile = cq.Workplane("XY")
        for i, row_passage in enumerate(self.row_passages):
            passage_profile = (
                row_passage.get_profile(
                    passage_profile
                    .transformed(offset=cq.Vector(0, 0, self.radii[i]-self.radii[i-1] if i else 0)),
                    with_airfoils=False
                )
            )

            airfoil_profile = (
                row_passage.get_airfoil_profile(
                    airfoil_profile
                    .transformed(offset=cq.Vector(0, 0, self.radii[i]-self.radii[i-1] if i else 0))
                )
            )

        passage_profile = passage_profile.sweep(
            cq.Workplane("XZ").lineTo(x=0, y=(self.radii[-1])-(self.radii[0])),
            multisection=True,
            makeSolid=True
        )

        airfoil_profile = airfoil_profile.sweep(
            cq.Workplane("XZ").lineTo(x=0, y=(self.radii[-1])-(self.radii[0])),
            multisection=True,
            makeSolid=True
        )

        profile = passage_profile.cut(airfoil_profile)
        return profile

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()


    def to_unstructured_geo(self, mesh_params: TurboMeshParameters = TurboMeshParameters(), sim_params: Optional[SimulationParams] = None) -> GeometryQL:
        if mesh_params.airfoil_mesh_size is None:
            mesh_params.airfoil_mesh_size = 0.02 * self.airfoil.chord_length
        if mesh_params.boundary_wall_mesh_size is None:
            mesh_params.boundary_wall_mesh_size = 0.01 * self.airfoil.chord_length
        if mesh_params.passage_mesh_size is None:
            mesh_params.passage_mesh_size = 0.025 * self.airfoil.chord_length

        with GeometryQL() as geo:
            y_plus = get_y_plus(self.airfoil.chord_length, sim_params)
            profile = self.get_profile()
            return (
                geo
                .load(profile)
                
                .edges(type="interior")
                .setMeshSize(mesh_params.airfoil_mesh_size)
                .addBoundaryLayer(y_plus, mesh_params.boundary_wall_ratio, mesh_params.boundary_num_layers)
                .addPhysicalGroup(mesh_params.airfoil_label)
                .end()
                
                .edges(type="exterior")
                .setMeshSize(mesh_params.passage_mesh_size)
                .end()

                .generate(2)
                .show("gmsh")
            )

    def to_geo(self, mesh_params: TurboMeshParameters = TurboMeshParameters(), sim_params: Optional[SimulationParams] = None):
        return self.to_unstructured_geo(mesh_params, sim_params)




@dataclass
class TurboStagePassage(Passage):
    inflow_passage: TurboRowPassage = field(metadata=opt_class())
    "Passage with inflow boundary condition"

    outflow_passage: TurboRowPassage = field(metadata=opt_class())
    "Passage with outflow boundary condition"

    def __post_init__(self):
        self.outflow_passage.offset = [self.inflow_passage.width, 0]

    # def get_profile(self, params: Optional[SimulationParams] = None):
    #     return [self.inflow_passage.get_profile(), self.outflow_passage.get_profile()]

    def visualize(self, title: str = "Passage"):
        self.inflow_passage.visualize(title)
        self.outflow_passage.visualize(title)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_config(
        self,
        params: SimulationParams,
        working_directory: str,
        id: str,
    ) -> Dict[str, Any]:
        assert params.translation is not None, "Translation must be specified for turbo stage passage"
        assert params.target_outlet_static_state is not None, "Target outlet static state must be specified for turbo stage passage"
        inflow_mesh_params = self.inflow_passage.mesh_params
        outflow_mesh_params = self.outflow_passage.mesh_params
        turb_kind = "TURBINE" if params.translation[0] is None else "COMPRESSOR"
        return {
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SST",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
            "SYSTEM_MEASUREMENTS": "SI",
            "MACH_NUMBER": params.inlet_total_state.mach_number,
            "AOA": params.angle_of_attack,
            "SIDESLIP_ANGLE": 0.0,
            "INIT_OPTION": "TD_CONDITIONS",
            "FREESTREAM_OPTION": "TEMPERATURE_FS",
            "FREESTREAM_PRESSURE": params.inlet_total_state.P,
            "FREESTREAM_TEMPERATURE": params.inlet_total_state.T,
            "FREESTREAM_DENSITY": params.inlet_total_state.rho_mass(),
            
            "FREESTREAM_TURBULENCEINTENSITY": 0.05, # standard for turbomachinery
            "FREESTREAM_TURB2LAMVISCRATIO": 100.0, # standard for turbomachinery
            "REF_ORIGIN_MOMENT_X": 0.00,
            "REF_ORIGIN_MOMENT_Y": 0.00,
            "REF_ORIGIN_MOMENT_Z": 0.00,
            "REF_LENGTH": 1.0,
            "REF_AREA": 1.0,
            "REF_DIMENSIONALIZATION": "DIMENSIONAL",
            "FLUID_MODEL": "IDEAL_GAS",
            "GAMMA_VALUE": params.inlet_total_state.gamma,
            "GAS_CONSTANT": params.inlet_total_state.gas_constant,
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
            "LINEAR_SOLVER_ERROR": 1E-4, # good estimate # "LINEAR_SOLVER_ERROR": 1E-6,

            "LINEAR_SOLVER_ITER": 10,
            "VENKAT_LIMITER_COEFF": 0.05,
            "LIMITER_ITER": 999999,
            "CONV_NUM_METHOD_FLOW": "ROE",
            "MUSCL_FLOW": "YES",
            "SLOPE_LIMITER_FLOW": "VAN_ALBADA_EDGE",
            "ENTROPY_FIX_COEFF": 0.1, # higher this the less accurate, 0.1 already pretty high, usually do 0.05
            "JST_SENSOR_COEFF": (0.5, 0.02),
            "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
            "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND",
            "TIME_DISCRE_TURB": "EULER_IMPLICIT",
            "CFL_REDUCTION_TURB": 1.0,
            "OUTER_ITER": 3000,
            "CONV_RESIDUAL_MINVAL": -10,
            "CONV_STARTITER": 10,
            "CONV_CAUCHY_ELEMS": 100,
            "CONV_CAUCHY_EPS": 1E-6,
            "MESH_FILENAME": f"{working_directory}/passage{id}.su2",
            "MESH_FORMAT": "SU2",
            "TABULAR_FORMAT": "CSV",
            **({} if id == "0" else {"SOLUTION_FILENAME": f"{working_directory}/restart_flow{int(id)-1}.dat"}),
            "VOLUME_FILENAME": f"{working_directory}/flow{id}.vtu",
            "RESTART_FILENAME":  f"{working_directory}/restart_flow{id}.dat",
            "SURFACE_FILENAME":  f"{working_directory}/surface_flow{id}.vtu",
            "CONV_FILENAME": f"{working_directory}/config{id}.csv",
            "HISTORY_OUTPUT": "TURBO_PERF",

            "MULTIZONE": "YES",
            "CONFIG_LIST": {
                f"{working_directory}/zone_{id}_1.cfg": {
                    **({
                        "GRID_MOVEMENT": "STEADY_TRANSLATION",
                        "MACH_MOTION": 0.35,
                        "TRANSLATION_RATE": f"{params.translation[0][0]} {params.translation[0][1]} {params.translation[0][2]}",
                    } if params.translation[0] is not None else {"GRID_MOVEMENT": "NONE"})
                },
                f"{working_directory}/zone_{id}_2.cfg": {
                    **({
                        "GRID_MOVEMENT": "STEADY_TRANSLATION",
                        "MACH_MOTION": 0.35,
                        "TRANSLATION_RATE": f"{params.translation[1][0]} {params.translation[1][1]} {params.translation[1][2]}",
                    } if params.translation[1] is not None else {"GRID_MOVEMENT": "NONE"})
                }
            },
            "MARKER_HEATFLUX": f"( {inflow_mesh_params.airfoil_label}, 0.0, {outflow_mesh_params.airfoil_label}, 0.0)",
            "MARKER_PERIODIC": f"( {inflow_mesh_params.bottom_label}, {inflow_mesh_params.top_label}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {self.inflow_passage.spacing}, 0.0, {outflow_mesh_params.bottom_label}, {outflow_mesh_params.top_label}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {self.outflow_passage.spacing}, 0.0)",
            "MARKER_TURBOMACHINERY": f"({inflow_mesh_params.inlet_label}, {inflow_mesh_params.outlet_label}, {outflow_mesh_params.inlet_label}, {outflow_mesh_params.outlet_label})",
            "MARKER_ANALYZE": "(outflow)",

            "MARKER_MIXINGPLANE_INTERFACE": f"({inflow_mesh_params.outlet_label}, {outflow_mesh_params.inlet_label})",
            "MARKER_ZONE_INTERFACE": f"({inflow_mesh_params.outlet_label}, {outflow_mesh_params.inlet_label})",

            "MARKER_GILES": f"({inflow_mesh_params.inlet_label}, TOTAL_CONDITIONS_PT, {params.inlet_total_state.P}, {params.inlet_total_state.T}, 1.0, 0.0, 0.0,1.0,1.0, {inflow_mesh_params.outlet_label}, MIXING_OUT, 0.0, 0.0, 0.0, 0.0, 0.0,1.0,1.0, {outflow_mesh_params.inlet_label}, MIXING_IN, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 1.0 {outflow_mesh_params.outlet_label}, STATIC_PRESSURE, {params.target_outlet_static_state.P}, 0.0, 0.0, 0.0, 0.0,1.0,1.0)",
            "SPATIAL_FOURIER": "NO", # YES if issues with wave reflection
            "TURBOMACHINERY_KIND": "AXIAL AXIAL",
            "TURBO_PERF_KIND": f"{turb_kind} {turb_kind}",

            "TURBULENT_MIXINGPLANE": "YES",
            "RAMP_OUTLET_PRESSURE": "NO", # YES can help with convergence
            "RAMP_OUTLET_PRESSURE_COEFF": "(140000.0, 10.0, 2000)",
            "MARKER_PLOTTING": f"({inflow_mesh_params.airfoil_label}, {outflow_mesh_params.airfoil_label})",
            "VOLUME_OUTPUT": "(SOLUTION, RESIDUAL, PRIMITIVE)"
        }

