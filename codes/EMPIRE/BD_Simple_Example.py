# -*- coding: utf-8 -*-
"""
Developer: Ali Alizadeh Ph.D. Candidate at Laval University
"""
#Problem Statement
"""
MIN  obj=x1+3.x2+y1+4.y2
  s.t. 
  -2.x1-x2+y1-2.y2>=1
  2.x1+2.x2-y1+3.y2>=1
  x1,x2,y1,y2>=0
we want to decompose the above problem using benders decomposition

Master problem:
    
MIN   Z
   s.t.
   Z>=y1+4.y2
   Obj.Slack + Landa1(y1-yp1)+ Landa2(y2-yp2) =<0  #feasibility cut if subproblem is infeasible
   Z>=y1+4.y2 + Obj.Subproblem + phi1(y1-yp1)+ phi2(y2-yp2) #optimality cut if subproblem is feasible
SubProblem:
    
MIN   obj=x1+3.x2   
  s.t. 
  -2.x1-x2+y1-2.y2>=1
  2.x1+2.x2-y1+3.y2>=1
  y1=yp1   (phi1)
  y2=yp2   (phi2)
  
Slack Problem:
    
MIN   obj=s1+s2   
  s.t. 
  -2.x1-x2+y1-2.y2+s1>=1
  2.x1+2.x2-y1+3.y2+s2>=1
  y1=yp1   (landa1)
  y2=yp2   (landa2)
"""
#add packages
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

#--------------------------
#Master Problem

Master=pyo.ConcreteModel()
Master. Z = pyo.Var()
Z=Master.Z
Master.y=pyo.Var([1,2], within=NonNegativeReals)
y=Master.y
Master.obj = pyo.Objective(expr=Z, sense=minimize)
def rule_master (Master):
    return Z>=y[1]+4*y[2]
Master.Cons=Constraint(rule=rule_master)
Master.FECUT=ConstraintList()  #feasibility cut
Master.Opti=ConstraintList()   # optimality cut

"""
opt.solve(Master)
Master.pprint()
results=opt.solve(Master)
print(results.solver.status)
print(results.solver.termination_condition)
"""
#=====================================================
#Sub Problem
yp1=0
yp2=0
SubProb=pyo.ConcreteModel()
SubProb.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
SubProb.x=pyo.Var([1,2],within=NonNegativeReals)
x=SubProb.x
SubProb.y=pyo.Var([1,2])
yp=SubProb.y
SubProb.obj = pyo.Objective(expr=x[1]+3*x[2], sense=minimize)
def rule_C1 (SubProb):
    return -2*x[1]-x[2]+ yp[1] -2*yp[2] >=1
SubProb.Cons1=Constraint(rule=rule_C1)

def rule_C2 (SubProb):
    return 2*x[1]+2*x[2]- yp[1] +3*yp[2] >=1
SubProb.Cons2=Constraint(rule=rule_C2)

def rule_C3 (SubProb):
    return yp[1]  == yp1
SubProb.Cons3=Constraint(rule=rule_C3)

def rule_C4 (SubProb):
    return yp[2] == yp2
SubProb.Cons4=Constraint(rule=rule_C4)
"""
opt.solve(SubProb)
SubProb.pprint()
results1=opt.solve(SubProb)
print(results1.solver.status)
print(results1.solver.termination_condition)
"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Slack Problem
Slackprob=pyo.ConcreteModel()
Slackprob.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
Slackprob.x=pyo.Var([1,2],within=NonNegativeReals)
xs=Slackprob.x
Slackprob.y=pyo.Var([1,2])
yps=Slackprob.y
Slackprob.s=pyo.Var([1,2],within=NonNegativeReals)
s=Slackprob.s
Slackprob.obj = pyo.Objective(expr=s[1]+s[2], sense=minimize)

def rule_C1 (Slackprob):
    return -2*xs[1]-xs[2]+ yps[1] -2*yps[2] + s[1] >=1
Slackprob.Cons1=Constraint(rule=rule_C1)

def rule_C2 (Slackprob):
    return 2*xs[1]+2*xs[2]- yps[1] +3*yps[2] + s[2] >=1
Slackprob.Cons2=Constraint(rule=rule_C2)

def rule_C3 (Slackprob):
    return yps[1]  == yp1
Slackprob.Cons3=Constraint(rule=rule_C3)

def rule_C4 (Slackprob):
    return yps[2] == yp2
Slackprob.Cons4=Constraint(rule=rule_C4)
"""
opt.solve(Slackprob)
Slackprob.pprint()
results2=opt.solve(Slackprob)
print(results2.solver.status)
print(results2.solver.termination_condition)
"""
#=======================================================================================
#defining loop for iterative process
opt=SolverFactory('gurobi', solver_io='python')
LB=-1000000
UB=100000
K1=np.zeros(15)
K2=np.zeros(15)
K=0
print(abs(UB-LB))
while abs(UB-LB)>0.1:
    
    K+=1
    opt.solve(Master)
    LB=value(Master.obj)
    print('LB=')
    print(LB)
    results=opt.solve(Master)
    yp1=value(y[1])
    yp2=value(y[2])
    print("yp1 update:")
    print(yp1)
    print("yp2 update:")
    print(yp2)
    if K>=2:
        SubProb.Cons3.deactivate()
        SubProb.Cons4.deactivate()
        SubProb.Cons3=Constraint(expr=yp[1]  == yp1 )
        SubProb.Cons4=Constraint(expr=yp[2]  == yp2 )   
    opt.solve(SubProb)
    
    results1=opt.solve(SubProb)
    print(results1.solver.status)
    print(results1.solver.termination_condition)
    if (results1.solver.status == SolverStatus.ok) and (results1.solver.termination_condition == TerminationCondition.optimal):
        print('Subproblem is feasible now :)', "iteration #=", K)
        K1[K]= SubProb.dual[SubProb.Cons3]
        K2[K]= SubProb.dual[SubProb.Cons4]
        print("I added Optimality cut with following duals:", K1[K],K2[K])
        Master.Opti.add(expr= Z >= y[1]+4*y[2]+value(SubProb.obj)+K1[K]*(y[1]-yp1)+K2[K]*(y[2]-yp2) )
        UB=value(SubProb.obj) + value(y[1]) + 4*value(y[2])
        print('UB=')
        print(UB)
        print("print(abs(UB-LB)):",abs(UB-LB))
   
    elif (results1.solver.termination_condition == TerminationCondition.infeasible):
    
        print('Subproblem is infeasible now ):', "iteration #=", K)
        opt.solve(Slackprob)
        K1[K]= Slackprob.dual[Slackprob.Cons3]
        K2[K]= Slackprob.dual[Slackprob.Cons4]
        print("I added feasibility cut with following duals:", K1[K],K2[K])
        Master.FECUT.add(expr=value(Slackprob.obj) + K1[K]*(y[1]-0)+K2[K]*(y[2]-0)<=0)
    else:
        print('Problem is not Solvable')
   
    
print('x values:', 'x1=', value(x[1]), 'x2=', value(x[2]) )
print('y values:', 'y1=', value(y[1]), 'x2=', value(y[2]) )
print('Master Objective:',value(Master.obj))
print('Sub Problem Objective:',value(SubProb.obj))