"""Network topology creation and utility functions."""
import networkx as nx

def create_topology():
    """Create a small, reproducible demonstration network."""
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D", "E", "F"])
    # delay_ms and capacity_mbps are baseline link properties.
    edges = [
        ("A", "B", 5.0, 100.0),
        ("A", "C", 8.0, 100.0),
        ("A", "E", 12.0, 100.0),
        ("B", "F", 5.0, 100.0),
        ("C", "D", 4.0, 100.0),
        ("D", "F", 4.0, 100.0),
        ("E", "F", 5.0, 100.0),
        ("B", "C", 3.0, 100.0),
    ]
    for u, v, delay, capacity in edges:
        graph.add_edge(
            u, v, delay_ms=delay, capacity_mbps=capacity,
            utilization=0.15, queue=0.10, packet_loss=0.001,
            estimated_delay_ms=delay, congestion=0.0, traffic_mbps=15.0
        )
    return graph
