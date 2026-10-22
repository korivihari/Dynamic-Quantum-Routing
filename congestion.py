"""Congestion scoring and edge cost calculations."""
import numpy as np

def update_congestion_scores(graph):
    """Add normalized congestion score (0..1) to each link."""
    edges = list(graph.edges(data=True))
    delays = [d["estimated_delay_ms"] for _, _, d in edges]
    max_delay = max(delays) if delays else 1.0
    for _, _, data in edges:
        util = float(data.get("utilization", 0))
        queue = float(data.get("queue", 0))
        delay_norm = float(data.get("estimated_delay_ms", 0)) / max(max_delay, 1e-9)
        loss_norm = min(float(data.get("packet_loss", 0)) / 0.08, 1.0)
        score = 0.30 * util + 0.25 * queue + 0.30 * delay_norm + 0.15 * loss_norm
        data["congestion"] = float(np.clip(score, 0, 1))
    return graph

def edge_cost(data):
    """Combined link cost in demonstration cost units."""
    return (
        0.40 * float(data["estimated_delay_ms"])
        + 35.0 * float(data["congestion"])
        + 12.0 * float(data["packet_loss"])
        + 0.10
    )
