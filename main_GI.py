import numpy as np
import os

def read_dimacs_graph(filepath):
    """
    Reads a DIMACS file and returns the dense adjacency matrix (0-based).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_nodes = 0
    edges = []

    # 1. Parse Header
    for line in lines:
        line = line.strip()
        if line.startswith('p'):
            parts = line.split()
            # format: p edge <nodes> <edges>
            num_nodes = int(parts[2])
            break
            
    if num_nodes == 0:
        raise ValueError("Invalid DIMACS file: could not find 'p edge' line.")

    # 2. Build Matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32) # float32 is best for PyTorch
    
    for line in lines:
        if line.startswith('e'):
            parts = line.split()
            u, v = int(parts[1]), int(parts[2])
            
            # DIMACS is 1-based, convert to 0-based
            u -= 1
            v -= 1
            
            # Undirected graph
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            
    return adj

# def generate_isomorphic_batch(base_adj, num_instances):
#     """
#     Generates N randomly permuted copies of the base graph.
    
#     Returns:
#         dataset (list): A list of dictionaries, where each dict contains:
#             - 'adj': The permuted adjacency matrix
#             - 'perm_vector': The 1D permutation vector used (mapping old->new)
#             - 'perm_matrix': The explicit permutation matrix P
#     """
#     n = base_adj.shape[0]
#     dataset = []

#     for i in range(num_instances):
#         # 1. Generate random permutation vector
#         # perm[i] = j means node i in new graph maps to node j in base graph
#         p_vec = np.random.permutation(n)
        
#         # 2. Create Permutation Matrix P
#         # P[i, j] = 1 if p_vec[i] == j
#         P = np.zeros((n, n), dtype=np.float32)
#         P[np.arange(n), p_vec] = 1
        
#         # 3. Apply Permutation: A_new = P * A_base * P_transpose
#         # Using numpy slicing is faster: A_new = A_base[p_vec][:, p_vec]
#         # But let's follow the algebra for clarity:
#         permuted_adj = P @ base_adj @ P.T
        
#         instance = {
#             "id": i,
#             "adj": permuted_adj,
#             "perm_vector": p_vec, 
#             "perm_matrix": P 
#         }
#         dataset.append(instance)
        
#     return

def generate_isomorphic(filepath):
    '''
    min ||PA-BP||_F^2 = ||A||_F^2 + ||B||_F^2 - 2trace(BPA'P'),
    map to the qap, A = D, B = F
    '''
    filepath = "./qaplibs/" + filepath
    D = read_dimacs_graph(filepath)
    n = D.shape[0]
    
    # check if D is symmetric
    if not np.allclose(D, D.T):
        raise ValueError("The input graph must be undirected (symmetric adjacency matrix).")
    
    # generate random permutation matrix
    p_vec = np.random.permutation(n)
    P = np.zeros((n, n), dtype=np.float32)
    P[np.arange(n), p_vec] = 1
    
    # generate the isomorphic graph
    F = P @ D @ P.T
    
    # calculate the objective value
    obj_label = np.sum(D**2) + np.sum(F**2)
    
    return n, F, D, -obj_label/2, P

# --- Main Execution ---

# # 1. Setup
# instance = '29'
# input_file = f"./GI/paley/paley-{instance}"  # Replace with your actual file


# # base_adj = read_dimacs_graph(input_file)
# n, F_np, D_np, obj_val, x_label = generate_isomorphic(input_file)

    
# tmp = x_label @ D_np.T @ x_label.T
# obj2 = np.trace(F_np @ tmp)
# print("Objective check:", obj2 - obj_val)

#     # 2. Inspect the output
#     for item in data:
#         print(f"\nInstance {item['id']}:")
#         print(f"  - Adjacency Shape: {item['adj'].shape}")
#         # Show first 10 elements of permutation for brevity
#         print(f"  - Permutation Vector (first 10): {item['perm_vector'][:10]}...") 
        
#     # 3. Verification Example (Verify Instance 0 vs Base)
#     # If correct, P.T @ A_new @ P should equal A_base
#     inst_0 = data[0]
#     P = inst_0['perm_matrix']
#     A_new = inst_0['adj']
    
#     # Reconstruct base from new
#     reconstructed_base = P.T @ A_new @ P
    
#     # Check difference
#     diff = np.sum(np.abs(reconstructed_base - base_adj))
#     print(f"\nVerification check (Difference): {diff}")
#     if diff < 1e-5:
#         print(">> SUCCESS: The generated graph is perfectly isomorphic.")
#     else:
#         print(">> FAILURE: Math mismatch.")

# except Exception as e:
#     print(f"An error occurred: {e}")