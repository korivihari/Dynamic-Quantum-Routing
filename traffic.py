"""Generate synthetic, changing link traffic for the demo."""
import numpy as np

LEVELS = {"Low": (0.10, 0.35), "Medium": (0.30, 0.70), "High": (0.60, 0.95)}

def apply_traffic(graph, level="Medium", seed=None, burst=False):
    """Update each link with simulated utilization, queue, loss and delay.

    These are synthetic indicators, not measurements from real packets.
    """
    rng = np.random.default_rng(seed)
    low, high = LEVELS.get(level, LEVELS["Medium"])
    for _, _, data in graph.edges(data=True):
        util = float(np.clip(rng.uniform(low, high), 0.02, 0.98))
        if burst and rng.random() < 0.35:
            util = float(np.clip(util + rng.uniform(0.15, 0.35), 0.02, 0.99))
        data["utilization"] = util
        data["traffic_mbps"] = util * data["capacity_mbps"]
        # Normalized queue indicator (0..1), used as a simplified proxy.
        data["queue"] = float(np.clip(0.12 + 0.78 * util**1.7 + rng.normal(0, 0.04), 0, 1))
        data["packet_loss"] = float(np.clip(0.0005 + max(0, util - 0.65)**2 * 0.08, 0, 0.08))
        data["estimated_delay_ms"] = data["delay_ms"] * (1 + 4.0 * util**2) + data["queue"] * 5.0
    return graph
