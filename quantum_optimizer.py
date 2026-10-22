"""Classical QUBO solver and a small QAOA circuit simulator implementation."""
import numpy as np
from qubo_model import exact_solve, qubo_energy

def solve_classical_qubo(q):
    bits, energy = exact_solve(q)
    return {"bits": bits, "energy": energy, "method": "Classical exact QUBO"}

def solve_qaoa(q, reps=1, maxiter=60, seed=7):
    """Run QAOA on Qiskit's statevector simulator for a small QUBO.

    This is a simulator-based research/demo implementation. It is not a claim
    of quantum advantage and scales poorly as the number of candidate paths grows.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError(
            "QAOA dependencies are missing. Install packages from requirements.txt."
        ) from exc

    n = int(q.shape[0])
    if n < 1:
        raise ValueError("QUBO matrix is empty.")
    if n > 10:
        raise ValueError("This demo QAOA simulator is limited to 10 route variables.")

    # Convert the QUBO into a diagonal cost for each computational basis state.
    # Qiskit statevector basis index uses qubit 0 as least-significant bit.
    states = np.arange(2**n, dtype=np.uint64)
    bit_matrix = np.array([[(int(s) >> i) & 1 for i in range(n)] for s in states], dtype=int)
    energies = np.array([qubo_energy(q, row) for row in bit_matrix], dtype=float)

    def probabilities(params):
        gammas = params[:reps]
        betas = params[reps:]
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for layer in range(reps):
            # Apply exp(-i gamma * cost) using a diagonal unitary. This is
            # a valid cost-layer implementation for this small simulator demo.
            phase = np.exp(-1j * gammas[layer] * energies)
            diagonal = np.diag(phase)
            qc.unitary(diagonal, list(range(n)), label="Cost")
            for qubit in range(n):
                qc.rx(2.0 * betas[layer], qubit)
        state = Statevector.from_instruction(qc)
        return np.abs(state.data) ** 2

    def objective(params):
        p = probabilities(params)
        return float(np.dot(p, energies))

    rng = np.random.default_rng(seed)
    initial = np.concatenate([rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi/2, reps)])
    result = minimize(objective, initial, method="COBYLA", options={"maxiter": int(maxiter), "rhobeg": 0.5})
    probs = probabilities(result.x)
    best_idx = int(np.argmax(probs))
    bits = bit_matrix[best_idx].tolist()
    # If the most likely sample violates one-hot selection, choose the lowest-energy
    # feasible sample from the final probability distribution when available.
    feasible = [i for i, row in enumerate(bit_matrix) if int(np.sum(row)) == 1]
    if feasible:
        chosen = max(feasible, key=lambda i: probs[i])
        bits = bit_matrix[chosen].tolist()
    return {
        "bits": bits,
        "energy": qubo_energy(q, bits),
        "method": f"QAOA statevector simulator (p={reps})",
        "expected_energy": float(result.fun),
        "optimizer_success": bool(result.success),
        "iterations": int(getattr(result, "nfev", 0)),
    }
