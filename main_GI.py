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

def generate_isomorphic_batch(base_adj):
    """
    Generates N randomly permuted copies of the base graph.
    
    Returns:
        dataset (list): A list of dictionaries, where each dict contains:
            - 'adj': The permuted adjacency matrix
            - 'perm_vector': The 1D permutation vector used (mapping old->new)
            - 'perm_matrix': The explicit permutation matrix P
    """
    n = base_adj.shape[0]
    
    p_vec = np.random.permutation(n)
        
    # 2. Create Permutation Matrix P
    # P[i, j] = 1 if p_vec[i] == j
    P = np.zeros((n, n), dtype=np.float32)
    P[np.arange(n), p_vec] = 1
    
    # 3. Apply Permutation: A_new = P * A_base * P_transpose
    # Using numpy slicing is faster: A_new = A_base[p_vec][:, p_vec]
    # But let's follow the algebra for clarity:
    permuted_adj = P @ base_adj @ P.T
    
    # obj = frobenius norm(base_adj) **2 + frobenius norm(permuted_adj) ** 2
    obj = np.sum(base_adj**2) + np.sum(permuted_adj**2)
        
    return n, base_adj, permuted_adj, -obj/2, p_vec

# --- Main Execution ---

# 1. Setup
instance = '29'
input_file = f"./GI/paley/paley-{instance}"  # Replace with your actual file
N = 5                          # Input: Number of graphs to generate

try:
    print(f"--- Loading Base Graph from {input_file} ---")
    base_adj = read_dimacs_graph(input_file)
    print(f"Base Graph: {base_adj.shape[0]} nodes")

    print(f"\n--- Generating {N} Isomorphic Instances ---")
    data = generate_isomorphic_batch(base_adj, num_instances=N)

    # 2. Inspect the output
    for item in data:
        print(f"\nInstance {item['id']}:")
        print(f"  - Adjacency Shape: {item['adj'].shape}")
        # Show first 10 elements of permutation for brevity
        print(f"  - Permutation Vector (first 10): {item['perm_vector'][:10]}...") 
        
    # 3. Verification Example (Verify Instance 0 vs Base)
    # If correct, P.T @ A_new @ P should equal A_base
    inst_0 = data[0]
    P = inst_0['perm_matrix']
    A_new = inst_0['adj']
    
    # Reconstruct base from new
    reconstructed_base = P.T @ A_new @ P
    
    # Check difference
    diff = np.sum(np.abs(reconstructed_base - base_adj))
    print(f"\nVerification check (Difference): {diff}")
    if diff < 1e-5:
        print(">> SUCCESS: The generated graph is perfectly isomorphic.")
    else:
        print(">> FAILURE: Math mismatch.")

except Exception as e:
    print(f"An error occurred: {e}")