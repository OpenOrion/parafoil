from dataclasses import dataclass
import dataclasses
import multiprocessing
import pickle
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, cast
import numpy as np
from paraflow import FlowState
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from paraflow.simulation.postprocessing import get_point_data
from paraflow.simulation.simulation import run_simulation
from paraflow.passages import SymmetricPassage, ConfigParameters
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from dacite.core import from_dict

from parafoil.passages.turbo import TurboStagePassage

MaxOrMin = Literal["max", "min"]

n_proccess = 1
pool = multiprocessing.Pool(n_proccess)
runner = StarmapParallelization(pool.starmap)

OBJECTIVE_TYPES = Literal["efficiency"]


class TurboPassageOptimizer(ElementwiseProblem):

    def __init__(
        self,
        working_directory: str,
        passage: TurboStagePassage,
        config_params: ConfigParameters,
        objectives: List[Tuple[OBJECTIVE_TYPES, MaxOrMin]],
    ):
        self.working_directory = working_directory
        self.passage = passage
        self.config_params = config_params
        self.objectives = objectives

            # label = metadata.get("label")
            # if label:
            #     print(f"{field_name}: {label}")

        # bounds = np.concatenate(list(self.variable_config.values()), axis=0)

        mins, maxs = get_opt_init(passage, TurboStagePassage)
        self.id = 0

        super().__init__(
            n_var=len(mins),
            n_obj=len(self.objectives),
            n_ieq_constr=0,
            xl=mins,
            xu=maxs,
            # elementwise_runner=runner
        )

    def _evaluate(self, x, out, *args, **kwargs):
        passage = cast(TurboStagePassage, create_opt_class_dict(self.passage, TurboStagePassage, x))
        self.id += 1
        sim_result = run_simulation(
            passage,
            config_params=self.config_params,
            working_directory="/workspaces/parafoil/simulation_out", 
            id=f"{self.id}",
            auto_delete=True
        )


        out["F"] = []
        out["G"] = []

    def optimize(self):
        algorithm = NSGA2(
            pop_size=40,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        res = minimize(
            self,
            algorithm,
            ("n_gen", 10000),
            seed=1,
            save_history=True,
            verbose=True
        )

        # X, F = res.opt.get("X", "F")

        with open(f'{self.working_directory}/optimization.pkl', 'wb') as optimization_result_file:
            pickle.dump(res, optimization_result_file, pickle.HIGHEST_PROTOCOL)



def get_opt_init(
    instance,
    cls,
    mins: List[float] = [],
    maxs: List[float] = []
):
    fields = dataclasses.fields(cls)
    for field in fields:
        instance_value = getattr(instance, field.name)
        if field.metadata and "type" in field.metadata:
            if field.metadata["type"] == "range":
                if isinstance(instance_value, Sequence):
                    mins += [field.metadata["min"]] * len(instance_value)
                    maxs += [field.metadata["max"]] * len(instance_value)
                else:
                    mins.append(field.metadata["min"])
                    maxs.append(field.metadata["max"])

            if dataclasses.is_dataclass(field.type) and field.metadata["type"] == "class":
                mins, maxs = get_opt_init(instance_value, field.type, mins, maxs)

    return mins, maxs

def create_opt_class_dict(
    instance,
    cls,
    flattened_values: List[float],
    current_index: int = 0,
):
    fields = dataclasses.fields(cls)
    class_dict = {}

    for field in fields:
        attribute_name = field.name
        attribute_metadata = field.metadata
        instance_value = getattr(instance, attribute_name)
        if attribute_metadata:
            if "type" in attribute_metadata:
                if attribute_metadata["type"] == "range":
                    if isinstance(instance_value, Sequence):
                        class_dict[attribute_name] = flattened_values[current_index:current_index + len(instance_value)]
                        current_index += len(instance_value)
                    else:
                        class_dict[attribute_name] = flattened_values[current_index]

                        current_index += 1
                if attribute_metadata["type"] == "constant":
                    class_dict[attribute_name] = instance_value

            if dataclasses.is_dataclass(field.type) and attribute_metadata["type"] == "class":
                nested_metadata = create_opt_class_dict(instance_value, field.type, flattened_values, current_index)
                class_dict[attribute_name] = nested_metadata
    
    if current_index == 0:
        return from_dict(cls, class_dict)
    return class_dict


#     if isinstance(instance_value, Sequence):
#         return mins[current_index:len(instance_value)]
#     else:
#         return mins[current_index]

