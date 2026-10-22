"""Run a small standalone experiment without the Streamlit UI."""
import time
import pandas as pd
from topology import create_topology
from traffic import apply_traffic
from congestion import update_congestion_scores
from routing import shortest_path_dijkstra, candidate_paths, path_metrics
from qubo_model import create_qubo
from quantum_optimizer import solve_classical_qubo, solve_qaoa
from performance import save_result

def run_experiment(level="Medium", seed=42, include_qaoa=False):
    graph = create_topology()
    apply_traffic(graph, level=level, seed=seed, burst=(level == "High"))
    update_congestion_scores(graph)
    records = []

    start = time.perf_counter()
    dijkstra_path = shortest_path_dijkstra(graph, "A", "F")
    elapsed = (time.perf_counter() - start) * 1000
    metrics = path_metrics(graph, dijkstra_path)
    records.append({"algorithm": "Dijkstra", "path": " -> ".join(dijkstra_path),
                    **metrics, "optimization_time_ms": elapsed, "traffic_level": level})

    paths = candidate_paths(graph, "A", "F", k=4)
    if paths:
        costs = [path_metrics(graph, p)["route_cost"] for p in paths]
        q, _ = create_qubo(costs)
        for name, solver in [("Classical QUBO", solve_classical_qubo)]:
            start = time.perf_counter()
            answer = solver(q)
            elapsed = (time.perf_counter() - start) * 1000
            idx = next((i for i, b in enumerate(answer["bits"]) if b), 0)
            path = paths[idx]
            records.append({"algorithm": name, "path": " -> ".join(path),
                            **path_metrics(graph, path), "optimization_time_ms": elapsed,
                            "traffic_level": level})
        if include_qaoa:
            try:
                start = time.perf_counter()
                answer = solve_qaoa(q)
                elapsed = (time.perf_counter() - start) * 1000
                idx = next((i for i, b in enumerate(answer["bits"]) if b), 0)
                path = paths[idx]
                records.append({"algorithm": "QAOA", "path": " -> ".join(path),
                                **path_metrics(graph, path), "optimization_time_ms": elapsed,
                                "traffic_level": level})
            except Exception as exc:
                print(f"QAOA was not run: {exc}")
    for row in records:
        save_result(row)
    return pd.DataFrame(records)

if __name__ == "__main__":
    results = run_experiment()
    print(results.to_string(index=False))
    print("\nResults appended to results/routing_results.csv")
