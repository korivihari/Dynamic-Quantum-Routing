"""Classical routing and candidate-path generation."""
import networkx as nx
from congestion import edge_cost

def shortest_path_dijkstra(graph, source, target):
    """Find route using estimated delay as the Dijkstra edge weight."""
    return nx.shortest_path(graph, source, target, weight="estimated_delay_ms")

def candidate_paths(graph, source, target, k=4):
    """Return up to k simple paths ranked by the project's composite edge cost."""
    if source == target:
        return [[source]]
    weighted = graph.copy()
    for u, v, data in weighted.edges(data=True):
        data["route_cost"] = edge_cost(data)
    try:
        paths = list(nx.shortest_simple_paths(weighted, source, target, weight="route_cost"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return paths[:max(1, int(k))]

def path_metrics(graph, path):
    """Calculate aggregate metrics for one path."""
    if not path or len(path) < 2:
        return {"latency_ms": 0.0, "congestion": 0.0, "packet_loss_pct": 0.0,
                "route_cost": 0.0, "bottleneck_utilization_pct": 0.0}
    links = [graph[path[i]][path[i+1]] for i in range(len(path)-1)]
    # Approximate end-to-end packet loss assuming independent link losses.
    success_prob = 1.0
    for link in links:
        success_prob *= (1.0 - float(link["packet_loss"]))
    return {
        "latency_ms": sum(float(d["estimated_delay_ms"]) for d in links),
        "congestion": sum(float(d["congestion"]) for d in links) / len(links),
        "packet_loss_pct": (1.0 - success_prob) * 100.0,
        "route_cost": sum(edge_cost(d) for d in links),
        "bottleneck_utilization_pct": max(float(d["utilization"]) for d in links) * 100.0,
    }
