
def opt_class():
    return {"type": "class"}

def opt_range(min: float, max: float):
    return {"type": "range", "min": min, "max": max}
    
def opt_constant():
    return {"type": "constant"}

def opt_tol_range(min: float, max: float):
    return {"type": "tol_range", "min": min, "max": max}