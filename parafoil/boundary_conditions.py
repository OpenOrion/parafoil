from dataclasses import dataclass
from typing import Callable, Literal, Optional

from pydantic import BaseModel


@dataclass
class Su2BoundaryConditionOptions:
    to_tuple: Callable[[], tuple]
    cfg_name: str


class BoundaryCondition(BaseModel):
    type: str
    label: str
    su2: Optional[Su2BoundaryConditionOptions] = None


# Inlet Boundary Conditions
class InletTotalBoundaryCondition(BoundaryCondition):
    type: Literal["inlet-total"] = "inlet-total"
    total_temperature: float
    total_pressure: float
    flow_direction: tuple[float, float, float]

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (
                self.label,
                self.total_temperature,
                self.total_pressure,
                *self.flow_direction,
            ),
            cfg_name="marker_inlet",
        )


class TurbomachineryBoundaryCondition(BoundaryCondition):
    type: Literal["turbomachinery"] = "turbomachinery"
    inflow_label: str
    outflow_label: str

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.inflow_label, self.outflow_label),
            cfg_name="marker_turbomachinery",
        )

class ZoneInterfaceBoundaryCondition(BoundaryCondition):
    type: Literal["zone-interface"] = "zone-interface"
    outmix_label: str
    inmix_label: str

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.outmix_label, self.inmix_label),
            cfg_name="marker_zone_interface",
        )

class OutletGaugePressureBoundaryCondition(BoundaryCondition):
    """Outlet Gauge Pressure for Incompressible BCs"""

    type: Literal["outlet-gauge-pressure"] = "outlet-gauge-pressure"
    gauge_pressure: float

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.label, self.gauge_pressure),
            cfg_name="marker_outlet",
        )

class OutletStaticPressureBoundaryCondition(BoundaryCondition):
    """Outlet Statuc Pressure for Compressible BCs"""

    type: Literal["outlet-static-pressure"] = "outlet-static-pressure"
    static_pressure: float

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.label, self.static_pressure),
            cfg_name="marker_outlet",
        )

# Wall Boundary Conditions
class SymmetricBoundaryCondition(BoundaryCondition):
    type: Literal["symmetric"] = "symmetric"

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.label,),
            cfg_name="marker_symmetry",
        )

class EulerBoundaryCondition(BoundaryCondition):
    type: Literal["euler"] = "euler"

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.label,),
            cfg_name="marker_euler",
        )

class FarfieldBoundaryCondition(BoundaryCondition):
    type: Literal["farfield"] = "farfield"

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.label,),
            cfg_name="marker_farfield",
        )

class ConstantHeatfluxCondition(BoundaryCondition):
    type: Literal["heatflux"] = "heatflux"
    heatflux: float = 0.0
    """value for heatflux in W/m^2"""

    @property
    def su2(self):
        return Su2BoundaryConditionOptions(
            to_tuple=lambda: (self.label, self.heatflux),
            cfg_name="marker_heatflux",
        )