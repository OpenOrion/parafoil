from dataclasses import asdict, dataclass, field
from typing import Any
from parafoil.metadata import opt_class
from paraflow import Passage
from parafoil.passages.turbo.row import TurboRowPassage
from paraflow import SimulationParams


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_config(
        self,
        params: SimulationParams,
        working_directory: str,
        id: str,
    ) -> dict[str, Any]:
        assert params.translation is not None, "Translation must be specified for turbo stage passage"
        assert params.target_outlet_static_state is not None, "Target outlet static state must be specified for turbo stage passage"
        inflow_mesh_params = self.inflow_passage.mesh_params
        outflow_mesh_params = self.outflow_passage.mesh_params
        turb_kind = "TURBINE" if params.translation[0] is None else "COMPRESSOR"
        return {
            # "SOLVER": "RANS",
            # "KIND_TURB_MODEL": "SST",
            # "MATH_PROBLEM": "DIRECT",
            # "RESTART_SOL": "NO",
            # "SYSTEM_MEASUREMENTS": "SI",
            
            # "SIDESLIP_ANGLE": 0.0,
            

            # "MACH_NUMBER": params.inlet_total_state.mach_number,
            # "AOA": params.angle_of_attack,
            # "INIT_OPTION": "TD_CONDITIONS",
            # "FREESTREAM_OPTION": "TEMPERATURE_FS",
            # "FREESTREAM_PRESSURE": params.inlet_total_state.P,
            # "FREESTREAM_TEMPERATURE": params.inlet_total_state.T,
            # "FREESTREAM_DENSITY": params.inlet_total_state.rho_mass(),
            
            # "FREESTREAM_TURBULENCEINTENSITY": 0.05, # standard for turbomachinery
            "FREESTREAM_TURB2LAMVISCRATIO": 100.0, # standard for turbomachinery

            "REF_ORIGIN_MOMENT_X": 0.00,
            # "REF_ORIGIN_MOMENT_Y": 0.00,
            # "REF_ORIGIN_MOMENT_Z": 0.00,
            # "REF_LENGTH": 1.0,
            # "REF_AREA": 1.0,
            # "REF_DIMENSIONALIZATION": "DIMENSIONAL",


            "FLUID_MODEL": "IDEAL_GAS",
            "GAMMA_VALUE": params.inlet_total_state.gamma,
            "GAS_CONSTANT": params.inlet_total_state.gas_constant,

            # "VISCOSITY_MODEL": "SUTHERLAND",
            # "MU_REF": 1.716E-5,
            # "MU_T_REF": 273.15,
            # "SUTHERLAND_CONSTANT": 110.4,
            # "CONDUCTIVITY_MODEL": "CONSTANT_PRANDTL",
            
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

            # "LINEAR_SOLVER_ITER": 10,
            # "VENKAT_LIMITER_COEFF": 0.05,
            # "LIMITER_ITER": 999999,
            
            "CONV_NUM_METHOD_FLOW": "ROE",
            # "MUSCL_FLOW": "YES",
            "SLOPE_LIMITER_FLOW": "VAN_ALBADA_EDGE",
            
            "ENTROPY_FIX_COEFF": 0.1, # higher this the less accurate, 0.1 already pretty high, usually do 0.05
            
            # "JST_SENSOR_COEFF": (0.5, 0.02),
            # "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
            # "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND",
            # "TIME_DISCRE_TURB": "EULER_IMPLICIT",
            # "CFL_REDUCTION_TURB": 1.0,
            "OUTER_ITER": 3000,
            
            "CONV_RESIDUAL_MINVAL": -10,
            
            "CONV_STARTITER": 10,
            
            # "CONV_CAUCHY_ELEMS": 100,
            "CONV_CAUCHY_EPS": 1E-6,
            
            # "MESH_FILENAME": f"{working_directory}/passage{id}.su2",
            # "MESH_FORMAT": "SU2",
            # "TABULAR_FORMAT": "CSV",
            # **({} if id == "0" else {"SOLUTION_FILENAME": f"{working_directory}/restart_flow{int(id)-1}.dat"}),
            # "VOLUME_FILENAME": f"{working_directory}/flow{id}.vtu",
            # "RESTART_FILENAME":  f"{working_directory}/restart_flow{id}.dat",
            # "SURFACE_FILENAME":  f"{working_directory}/surface_flow{id}.vtu",
            # "CONV_FILENAME": f"{working_directory}/config{id}.csv",
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

