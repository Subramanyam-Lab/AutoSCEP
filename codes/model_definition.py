from pyomo.environ import *

def define_model():
    model = AbstractModel()
    model.P = Set()
    model.C = Set()
    model.f = Param(model.P, within=NonNegativeReals)
    model.c = Param(model.P, within=NonNegativeReals)
    model.d = Param(model.C, within=NonNegativeReals)
    model.t = Param(model.C, model.P, within=NonNegativeReals)
    model.x = Var(model.C, model.P, within=NonNegativeReals, bounds=(0, 1))

    return model
