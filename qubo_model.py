"""QUBO formulation for choosing exactly one candidate route."""
import numpy as np

def create_qubo(costs, penalty=None):
    r"""Build upper-triangular Q for E(x)=sum_i cost_i*x_i + P(sum_i x_i - 1)^2.

    Energy convention: E(x) = sum_i Q[i,i] x_i + sum_{i<j} Q[i,j] x_i*x_j.
    """
    costs = np.asarray(costs, dtype=float)
    n = len(costs)
    if n == 0:
        raise ValueError("At least one candidate route is required.")
    if penalty is None:
        penalty = max(10.0, float(np.max(np.abs(costs))) * 2.0 + 1.0)
    q = np.zeros((n, n), dtype=float)
    # P(sum x - 1)^2 = P - P*sum(x_i) + 2P*sum_{i<j}x_i*x_j
    for i, cost in enumerate(costs):
        q[i, i] = float(cost) - float(penalty)
        for j in range(i + 1, n):
            q[i, j] = 2.0 * float(penalty)
    return q, float(penalty)

def qubo_energy(q, bits):
    """Evaluate upper-triangular QUBO energy for a binary vector."""
    x = np.asarray(bits, dtype=int)
    return float(np.dot(np.diag(q), x) + sum(q[i, j] * x[i] * x[j]
                     for i in range(len(x)) for j in range(i + 1, len(x))))

def exact_solve(q):
    """Exhaustive binary search; suitable only for small candidate counts."""
    import itertools
    n = q.shape[0]
    best_bits, best_energy = None, float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        energy = qubo_energy(q, bits)
        if energy < best_energy:
            best_bits, best_energy = bits, energy
    return list(best_bits), best_energy
