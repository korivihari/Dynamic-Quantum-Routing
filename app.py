"""Streamlit UI for Dynamic Congestion-Aware Packet Routing."""
import time
import os
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from topology import create_topology
from traffic import apply_traffic
from congestion import update_congestion_scores
from routing import shortest_path_dijkstra, candidate_paths, path_metrics
from qubo_model import create_qubo
from quantum_optimizer import solve_classical_qubo, solve_qaoa
from performance import save_result, load_results, save_comparison_graph

st.set_page_config(page_title="Dynamic Quantum Routing", page_icon="🌐", layout="wide")
st.title("🌐 Dynamic Congestion-Aware Packet Routing")
st.caption("QUBO-based route selection with classical baselines and a QAOA simulator")

if "graph" not in st.session_state:
    st.session_state.graph = create_topology()
    apply_traffic(st.session_state.graph, "Medium", seed=42)
    update_congestion_scores(st.session_state.graph)
if "last_runs" not in st.session_state:
    st.session_state.last_runs = pd.DataFrame()
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0

with st.sidebar:
    st.header("Simulation controls")
    source = st.selectbox("Source router", ["A", "B", "C", "D", "E", "F"], index=0)
    destination = st.selectbox("Destination router", ["A", "B", "C", "D", "E", "F"], index=5)
    traffic_level = st.select_slider("Traffic intensity", options=["Low", "Medium", "High"], value="Medium")
    burst = st.checkbox("Enable random traffic bursts", value=False)
    seed = st.number_input("Traffic seed", min_value=0, max_value=999999, value=42, step=1)
    k_paths = st.slider("Candidate routes for QUBO", min_value=2, max_value=6, value=4)
    qaoa_reps = st.slider("QAOA depth (p)", min_value=1, max_value=2, value=1)
    qaoa_iterations = st.slider("QAOA classical optimizer iterations", min_value=20, max_value=120, value=60, step=10)
    st.divider()
    generate = st.button("🔄 Generate traffic", use_container_width=True)
    optimize = st.button("⚡ Optimize route", type="primary", use_container_width=True)

if generate:
    st.session_state.run_counter += 1
    g = create_topology()
    apply_traffic(g, traffic_level, seed=int(seed) + st.session_state.run_counter, burst=burst)
    update_congestion_scores(g)
    st.session_state.graph = g
    st.success("Traffic conditions updated. Run optimization to calculate routes.")

graph = st.session_state.graph
if source not in graph or destination not in graph:
    st.error("Selected router is not in the topology.")
    st.stop()

# Metrics summarize the current simulated link conditions.
edge_data = [d for _, _, d in graph.edges(data=True)]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean link utilization", f"{np.mean([d['utilization'] for d in edge_data])*100:.1f}%")
c2.metric("Mean estimated link delay", f"{np.mean([d['estimated_delay_ms'] for d in edge_data]):.1f} ms")
c3.metric("Mean congestion score", f"{np.mean([d['congestion'] for d in edge_data]):.2f} / 1.00")
c4.metric("Links in topology", f"{graph.number_of_edges()}")

def make_topology_figure(g, selected_path=None):
    pos = nx.spring_layout(g, seed=17)
    fig = go.Figure()
    # Draw each edge separately so color can represent congestion.
    for u, v, d in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        score = float(d["congestion"])
        if score < 0.35:
            color = "#2ca25f"
        elif score < 0.60:
            color = "#f0ad4e"
        else:
            color = "#d62728"
        if selected_path and any(
            (selected_path[i] == u and selected_path[i+1] == v) or
            (selected_path[i] == v and selected_path[i+1] == u)
            for i in range(len(selected_path)-1)
        ):
            color = "#6f42c1"
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=4 if color == "#6f42c1" else 2, color=color),
            hoverinfo="text",
            text=f"{u}—{v}<br>Delay: {d['estimated_delay_ms']:.1f} ms"
                 f"<br>Utilization: {d['utilization']*100:.1f}%"
                 f"<br>Congestion: {score:.2f}",
            showlegend=False
        ))
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2,
            text=f"{d['utilization']*100:.0f}%", showarrow=False,
            font=dict(size=10, color="#333"), bgcolor="white", opacity=0.8)
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in g.nodes()],
        y=[pos[n][1] for n in g.nodes()],
        mode="markers+text", text=list(g.nodes()), textposition="top center",
        marker=dict(size=28, color="#e8f1fb", line=dict(width=2, color="#1f4e79")),
        hoverinfo="text", hovertext=[f"Router {n}" for n in g.nodes()],
        showlegend=False
    ))
    fig.update_layout(height=470, margin=dict(l=10,r=10,t=10,b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Network topology")
    st.plotly_chart(make_topology_figure(graph), use_container_width=True)
    st.caption("Green = lower congestion; amber = medium; red = high. Link labels show utilization.")

with right:
    st.subheader("Current link state")
    link_rows = []
    for u, v, d in graph.edges(data=True):
        link_rows.append({
            "Link": f"{u}—{v}", "Utilization %": round(d["utilization"]*100, 1),
            "Delay ms": round(d["estimated_delay_ms"], 2),
            "Queue indicator": round(d["queue"], 2),
            "Packet loss %": round(d["packet_loss"]*100, 3),
            "Congestion": round(d["congestion"], 2)
        })
    st.dataframe(pd.DataFrame(link_rows), use_container_width=True, hide_index=True)

if optimize:
    if source == destination:
        st.warning("Choose different source and destination routers.")
    else:
        try:
            paths = candidate_paths(graph, source, destination, k=k_paths)
            if not paths:
                st.error("No route exists between the selected routers.")
            else:
                costs = [path_metrics(graph, p)["route_cost"] for p in paths]
                q, penalty = create_qubo(costs)
                results = []

                # Dijkstra baseline uses estimated delay as its edge weight.
                start = time.perf_counter()
                dpath = shortest_path_dijkstra(graph, source, destination)
                dt = (time.perf_counter() - start) * 1000
                results.append({
                    "algorithm": "Dijkstra",
                    "path": " → ".join(dpath),
                    **path_metrics(graph, dpath),
                    "optimization_time_ms": dt,
                    "traffic_level": traffic_level,
                })

                start = time.perf_counter()
                classical = solve_classical_qubo(q)
                ct = (time.perf_counter() - start) * 1000
                cidx = next((i for i, bit in enumerate(classical["bits"]) if bit), 0)
                cpath = paths[cidx]
                results.append({
                    "algorithm": "Classical QUBO",
                    "path": " → ".join(cpath),
                    **path_metrics(graph, cpath),
                    "optimization_time_ms": ct,
                    "traffic_level": traffic_level,
                })

                qaoa_message = None
                try:
                    start = time.perf_counter()
                    quantum = solve_qaoa(q, reps=qaoa_reps, maxiter=qaoa_iterations, seed=int(seed))
                    qt = (time.perf_counter() - start) * 1000
                    qidx = next((i for i, bit in enumerate(quantum["bits"]) if bit), 0)
                    qpath = paths[qidx]
                    results.append({
                        "algorithm": "QAOA simulator",
                        "path": " → ".join(qpath),
                        **path_metrics(graph, qpath),
                        "optimization_time_ms": qt,
                        "traffic_level": traffic_level,
                    })
                except Exception as exc:
                    qaoa_message = str(exc)

                df = pd.DataFrame(results)
                st.session_state.last_runs = df
                for row in results:
                    save_result(row)
                st.session_state.selected_path = results[-1]["path"].split(" → ")
                st.session_state.candidate_paths = paths
                st.session_state.candidate_costs = costs
                st.session_state.qubo_matrix = q
                st.session_state.qubo_penalty = penalty
                st.session_state.qaoa_message = qaoa_message
                st.success("Routing optimization completed. Results were appended to results/routing_results.csv.")
        except Exception as exc:
            st.exception(exc)

if "last_runs" in st.session_state and not st.session_state.last_runs.empty:
    st.divider()
    st.subheader("Routing results")
    st.dataframe(st.session_state.last_runs.round(4), use_container_width=True, hide_index=True)
    selected = st.session_state.get("selected_path")
    if selected:
        st.markdown(f"**Last highlighted route:** `{' → '.join(selected)}`")
        st.plotly_chart(make_topology_figure(graph, selected), use_container_width=True)
    if st.session_state.get("qaoa_message"):
        st.warning(f"QAOA was unavailable for this run: {st.session_state.qaoa_message}")
    if "qubo_matrix" in st.session_state:
        with st.expander("View QUBO formulation"):
            st.write(f"One-hot penalty: {st.session_state.qubo_penalty:.3f}")
            st.write("Candidate paths:")
            for i, (path, cost) in enumerate(zip(st.session_state.candidate_paths, st.session_state.candidate_costs)):
                st.write(f"x{i}: {' → '.join(path)} | route cost = {cost:.3f}")
            st.write("Upper-triangular QUBO matrix (energy = diagonal terms + upper-triangle pair terms):")
            st.dataframe(pd.DataFrame(st.session_state.qubo_matrix).round(3), use_container_width=True)

st.divider()
st.subheader("Saved experiment history")
history = load_results()
if not history.empty:
    st.dataframe(history.tail(30), use_container_width=True, hide_index=True)
    if {"algorithm", "latency_ms"}.issubset(history.columns):
        plot_df = history.groupby("algorithm", as_index=False)["latency_ms"].mean()
        chart = go.Figure(data=[go.Bar(x=plot_df["algorithm"], y=plot_df["latency_ms"])])
        chart.update_layout(title="Mean estimated latency across saved runs",
                            xaxis_title="Algorithm", yaxis_title="Latency (ms)")
        st.plotly_chart(chart, use_container_width=True)
        if st.button("Save comparison graph as HTML"):
            path = save_comparison_graph(plot_df)
            if path:
                st.success(f"Saved graph to {path}")
    csv_bytes = history.to_csv(index=False).encode("utf-8")
    st.download_button("Download saved results CSV", data=csv_bytes,
                       file_name="routing_results.csv", mime="text/csv")
else:
    st.info("No saved runs yet. Select settings and click 'Optimize route' to create results.")

st.caption("Research prototype: traffic, queue and packet-loss values are simulated estimates. QAOA runs on a classical statevector simulator; no quantum hardware or real network is used.")
