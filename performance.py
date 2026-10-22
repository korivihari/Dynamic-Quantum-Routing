"""Performance metrics and persistent experiment results."""
import os
import pandas as pd
import plotly.express as px

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "routing_results.csv")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")

def ensure_result_dirs():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

def save_result(record):
    ensure_result_dirs()
    df = pd.DataFrame([record])
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)
    return CSV_PATH

def load_results():
    if os.path.exists(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH)
        except (pd.errors.EmptyDataError, OSError):
            pass
    return pd.DataFrame()

def save_comparison_graph(df, filename="latency_comparison.html"):
    ensure_result_dirs()
    if df.empty or "algorithm" not in df.columns or "latency_ms" not in df.columns:
        return None
    fig = px.bar(df, x="algorithm", y="latency_ms", color="algorithm",
                 title="End-to-end estimated latency by routing algorithm",
                 labels={"algorithm": "Algorithm", "latency_ms": "Estimated latency (ms)"})
    path = os.path.join(GRAPHS_DIR, filename)
    fig.write_html(path)
    return path
