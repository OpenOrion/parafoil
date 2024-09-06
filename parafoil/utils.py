# %%
from typing import Iterable, Sequence
import numpy as np
import numpy.typing as npt
import cadquery as cq
from cadquery.cq import VectorLike
from jupyter_cadquery import show
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.Geom import Geom_BSplineCurve
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_HArray1OfReal, TColStd_HArray1OfInteger
import cadquery as cq
import numpy as np
from scipy.interpolate import BSpline

def get_sampling(num_samples: int, is_cosine_sampling: bool):
    if is_cosine_sampling:
        beta = np.linspace(0.0,np.pi, num_samples, endpoint=True)
        return 0.5*(1.0-np.cos(beta))
    else:
        return np.linspace(0.0, 1.0, num_samples, endpoint=True)


def get_bspline(ctrl_pnts: npt.NDArray, degree: int):
    "get a bspline with clamped knots"
    num_ctrl_pnts = ctrl_pnts.shape[0]
    knots = np.pad(
        array=np.linspace(0, 1, (num_ctrl_pnts + 1) - degree),
        pad_width=(degree, degree),
        mode='constant',
        constant_values=(0, 1)
    )
    return BSpline(knots, ctrl_pnts, degree, extrapolate=False)

def get_occ_pnt_array(listOfVector: list[cq.Vector]):
    arr = TColgp_HArray1OfPnt(1, len(listOfVector))
    for i, vector in enumerate(listOfVector):
        arr.SetValue(i + 1, vector.toPnt())
    return arr

def get_occ_real_array(listOfReal: Sequence[float]):
    arr = TColStd_HArray1OfReal(1, len(listOfReal))
    for i, real in enumerate(listOfReal):
        arr.SetValue(i + 1, real)
    return arr

def get_occ_int_array(listOfReal: Sequence[int]):
    arr = TColStd_HArray1OfInteger(1, len(listOfReal))
    for i, real in enumerate(listOfReal):
        arr.SetValue(i + 1, real)
    return arr

def makeBSplineEdge(ctrl_pnts: list[cq.Vector]):
    degree = 3
    num_ctrl_pnts = len(ctrl_pnts)
    assert num_ctrl_pnts > 2, "BSpline curve requires at least 2 control points"

    if degree > num_ctrl_pnts - 1:
        degree = num_ctrl_pnts - 1

    weights = np.ones(num_ctrl_pnts)
    # periodic = False 
    periodic = ctrl_pnts[0] == ctrl_pnts[-1]
    if not periodic:
        sum_of_all_mult = num_ctrl_pnts + degree + 1
        num_knots = sum_of_all_mult - 2 * degree
        knots = range(num_knots)
        multiplicities = [1] * num_knots
        multiplicities[0] = degree + 1
        multiplicities[-1] = degree + 1
    else:
        knots = range(num_ctrl_pnts)
        multiplicities = [1] * len(knots)

    ctrl_pnts_occ = get_occ_pnt_array(ctrl_pnts)
    multiplicities_occ = get_occ_int_array(multiplicities)
    knots_occ = get_occ_real_array(knots)
    weights_occ = get_occ_real_array(weights) # type: ignore
    spline_geom = Geom_BSplineCurve(ctrl_pnts_occ, weights_occ, knots_occ, multiplicities_occ, degree, periodic)
    return cq.Edge(BRepBuilderAPI_MakeEdge(spline_geom).Edge())

class ExtendedWorkplane(cq.Workplane):
    def bspline(
        self, 
        points: Iterable[VectorLike], 
        forConstruction: bool = False, 
        includeCurrent: bool = False, 
        makeWire: bool = False
    ):
        allPoints = self._toVectors(points, includeCurrent)
        e = makeBSplineEdge(allPoints)

        if makeWire:
            rv_w = cq.Wire.assembleEdges([e])
            if not forConstruction:
                self._addPendingWire(rv_w)
        else:
            if not forConstruction:
                self._addPendingEdge(e)

        return self.newObject([rv_w if makeWire else e])





