from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cadquery as cq
from paraflow import Passage, SimulationParams
from parafoil.metadata import opt_class
from parafoil.passages.turbo.station import TurboMeshParameters, TurboStationPassage
from parafoil.passages.utils import get_y_plus
from meshql import GeometryQL


@dataclass
class TurboRowPassage(Passage):
    stations: Sequence[TurboStationPassage] = field(metadata=opt_class())

    radii: Sequence[float]
    "radii of the passages"

    def get_profile(self):
        passage_profile = cq.Workplane("XY")
        airfoil_profile = cq.Workplane("XY")
        for i, row_passage in enumerate(self.stations):
            passage_profile = row_passage.get_profile(
                passage_profile.transformed(
                    offset=cq.Vector(
                        0, 0, self.radii[i] - self.radii[i - 1] if i else 0
                    )
                ),
            )

            airfoil_profile = row_passage.get_airfoil_profile(
                airfoil_profile.transformed(
                    offset=cq.Vector(
                        0, 0, self.radii[i] - self.radii[i - 1] if i else 0
                    )
                )
            )

        passage_profile = passage_profile.sweep(
            cq.Workplane("XZ").lineTo(x=0, y=(self.radii[-1]) - (self.radii[0])),
            multisection=True,
            makeSolid=True,
        )

        airfoil_profile = airfoil_profile.sweep(
            cq.Workplane("XZ").lineTo(x=0, y=(self.radii[-1]) - (self.radii[0])),
            multisection=True,
            makeSolid=True,
        )

        profile = passage_profile.cut(airfoil_profile)
        # profile = passage_profile
        # profile = airfoil_profile
        return profile

    # def get_meshes(self, params: SimulationParams | None = None):
    #     with GeometryQL() as geo:
    #         (geo.load(self.get_profile()).refine(1).generate().show("gmsh"))

    # def to_dict(self) -> dict[str, Any]:
    #     return super().to_dict()

    # def to_unstructured_geo(
    #     self,
    #     mesh_params: Optional[TurboMeshParameters] = None,
    #     sim_params: Optional[SimulationParams] = None,
    # ):
    #     if mesh_params is None:
    #         mesh_params = TurboMeshParameters()

    #     if mesh_params.airfoil_mesh_size is None:
    #         mesh_params.airfoil_mesh_size = 0.02 * self.airfoil.chord_length
    #     if mesh_params.boundary_wall_mesh_size is None:
    #         mesh_params.boundary_wall_mesh_size = 0.01 * self.airfoil.chord_length
    #     if mesh_params.passage_mesh_size is None:
    #         mesh_params.passage_mesh_size = 0.025 * self.airfoil.chord_length

    #     with GeometryQL() as geo:
    #         y_plus = get_y_plus(self.airfoil.chord_length, sim_params)
    #         profile = self.get_profile()
    #         return (
    #             geo.load(profile)
    #             .edges(type="interior")
    #             .setMeshSize(mesh_params.airfoil_mesh_size)
    #             .addBoundaryLayer(
    #                 y_plus,
    #                 mesh_params.boundary_wall_ratio,
    #                 mesh_params.boundary_num_layers,
    #             )
    #             .addPhysicalGroup(mesh_params.airfoil_label)
    #             .end()
    #             .edges(type="exterior")
    #             .setMeshSize(mesh_params.passage_mesh_size)
    #             .end()
    #             .generate(2)
    #             .show("gmsh")
    #         )

    # def to_geo(
    #     self,
    #     mesh_params: Optional[TurboMeshParameters] = None,
    #     sim_params: Optional[SimulationParams] = None,
    # ):
    #     return self.to_unstructured_geo(mesh_params, sim_params)
