import numpy as np
import networkx as nx

def build_hamiltonian_graph(h1, h2, threshold=1e-5):
    """
    h1: One-body integrals (N, N)
    h2: Two-body integrals (N, N, N, N) in physicists' notation <ij|V|kl>
    """
    n_orb = h1.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n_orb))
    
    # 1. One-body hopping connectivity
    # Connect i <-> j if h_ij is significant
    rows, cols = np.where(np.abs(h1) > threshold)
    for p, q in zip(rows, cols):
        if p != q:
            G.add_edge(p, q)
            
    # 2. Two-body scattering connectivity
    # We want to connect indices that are coupled by V.
    # The term is c^dag_i c^dag_j c_k c_l. 
    # Conservative approach: Connect (i,k), (i,l), (j,k), (j,l)
    # This represents the "path" an electron takes during scattering.
    
    # This loop can be slow for large basis; vectorize in production!
    indices = np.argwhere(np.abs(h2) > threshold)
    for i, j, k, l in indices:
        # Add edges representing the transition 
        # An electron in k 'couples' to i (and j)
        edges = [(i, k), (i, l), (j, k), (j, l)]
        for u, v in edges:
            if u != v:
                G.add_edge(u, v)
                
    return G

import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

class GraphData(NamedTuple):
    senders: jnp.ndarray
    receivers: jnp.ndarray
    edge_types: jnp.ndarray # Optional: distinguish hopping vs interaction

def build_sparse_graph(h1, h2, threshold=1e-5):
    """
    Constructs sparse graph indices from Hamiltonian integrals.
    h1: (N_orb, N_orb)
    h2: (N_orb, N_orb, N_orb, N_orb)
    """
    n_orb = h1.shape[0]
    
    # --- 1. Identify Significant Interactions ---
    # We want edges where electrons can HOP or SCATTER.
    # A safe heuristic for backflow: connect (i, j) if they share a large V_ijkl 
    # This captures the "virtual excitation" pathways.
    
    # Find indices where interaction is strong
    # Note: For very large systems, avoid `np.abs(h2)` on full tensor.
    # Use PySCF's sparse iterators if N > 100.
    
    # Simplify scattering graph: Connect orbitals i,j if max_kl |V_{ijkl}| > tol
    # This reduces the 4-tensor to a 2-matrix of connectivity
    max_interaction = np.max(np.abs(h2), axis=(2, 3)) # Reduce over k,l
    adjacency = (np.abs(h1) > threshold) | (max_interaction > threshold)
    
    # Remove self-loops (handled by node update MLP)
    np.fill_diagonal(adjacency, False)
    
    # Convert to COO format (Senders -> Receivers)
    senders, receivers = np.where(adjacency)
    
    print(f"Graph built: {n_orb} nodes, {len(senders)} edges (Sparsity: {len(senders)/n_orb**2:.2%})")
    
    return GraphData(
        senders=jnp.array(senders, dtype=jnp.int32),
        receivers=jnp.array(receivers, dtype=jnp.int32),
        edge_types=jnp.zeros_like(senders) # Placeholder for more complex edge features
    )
