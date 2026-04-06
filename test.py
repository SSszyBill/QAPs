# plot the histogram of the eigenvalue distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
import os


def read_instance(instance):
    # 请确保路径正确
    problem_file = f"./qaplibs/{instance}.dat"
    solution_file = f"./qaplibs/{instance}.sln"
    
    with open(problem_file, "r") as f:
        data = f.read().split()        
    data_iter = iter(map(int, data))
    n = next(data_iter)
    F_flat = [next(data_iter) for _ in range(n * n)]
    F_np = np.array(F_flat).reshape(n, n)
    D_flat = [next(data_iter) for _ in range(n * n)]
    D_np = np.array(D_flat).reshape(n, n)
    
    if not os.path.exists(solution_file):
        return n, F_np, D_np, 1.0, None
    
    
    with open(solution_file, "r") as f:
        sol_data = f.read().split()
    sol_data_iter = iter(map(int, sol_data))
    n_sol = next(sol_data_iter)
    obj_label = next(sol_data_iter)
    x_label = [next(sol_data_iter) - 1 for _ in range(n)]
    x_label_np = np.zeros((n, n))
    for i in range(n):
        x_label_np[i, x_label[i]] = 1

    return n, F_np, D_np, obj_label, x_label_np


n, F, D, obj_label, x_label = read_instance("QAPLIB/nug12")

# kronecker product
FD = np.kron(F, D)

eigenvalues = np.linalg.eigvalsh(FD)
# print(f"Eigenvalue stats: min {np.min(eigenvalues)}, max {np.max(eigenvalues)}, mean {np.mean(eigenvalues)}, std {np.std(eigenvalues)}")
# # calculate the density of eigenvalues greater than -200
# density = np.sum(eigenvalues > -10) / len(eigenvalues)
# print(f"Density of eigenvalues greater than -150: {density}")
# # filter outliers
# eigenvalues = eigenvalues[(eigenvalues > -3e3) & (eigenvalues < 2.5e3)]



# plt.hist(eigenvalues, bins=50, density=True)
# plt.title("Eigenvalue Distribution of Kronecker Product FD")
# plt.xlabel("Eigenvalue")
# plt.ylabel("Density")
# plt.grid(True)
# plt.savefig("eigenvalue_distribution.png")


import cvxpy as cp
Q = 1/2 * (FD + FD.T)  # Ensure Q is symmetric
n = Q.shape[0]
y = cp.Variable(n)

# Constraints: 
# 1. The resulting matrix must be PSD
# 2. Usually we want y to be non-negative (adding to diagonal)
constraints = [Q + cp.diag(y) >> 0]

# Objective: Minimize the sum of the shifts
prob = cp.Problem(cp.Minimize(cp.sum(y)), constraints)
prob.solve(verbose=True, solver='SCS', eps_infeas=1e-12)

print("Required diagonal shift y:", y.value)
# print("Eigenvalues of (Q + diag(y)):", np.linalg.eigvalsh(Q + np.diag(y.value)))

print(np.mean(y.value))