import gurobipy as gp
from gurobipy import GRB
import numpy as np
import argparse

from main_gpu3 import read_instance


parser = argparse.ArgumentParser()
parser.add_argument('--instance', type=str, default="nug12")
parser.add_argument('--time_limit', type=int, default=100)


args = parser.parse_args()
instance = args.instance
time_limit = args.time_limit


n, F, D, obj_label, x_label = read_instance(f"QAPLIB/{instance}")


model = gp.Model("QAP")
model.Params.TimeLimit = time_limit
model.Params.OutputFlag = 1
model.Params.NonConvex = 2
model.Params.MIPGap = 0.0
model.Params.Threads = 8



x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
for i in range(n):
    model.addConstr(x.sum(i, '*') == 1)

for j in range(n):
    model.addConstr(x.sum('*', j) == 1)

obj_expr = gp.QuadExpr()
for i in range(n):
    for k in range(n):
        if F[i, k] == 0: continue
        for j in range(n):
            for l in range(n):
                if D[j, l] == 0: continue
                coeff = F[i, k] * D[j, l]
                obj_expr.add(x[i, j] * x[k, l], coeff)

model.setObjective(obj_expr, GRB.MINIMIZE)

model.optimize()
if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
    obj_val = model.ObjVal
    solve_time = model.Runtime

with open("result_gurobi.txt", "a") as f:
    f.write(f"{instance} {solve_time:.2f} {obj_val} {obj_label}\n")