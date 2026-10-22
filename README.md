# Dynamic Congestion-Aware Packet Routing Using QUBO and Quantum Optimization

A small M.Tech demonstration project with a Streamlit UI, simulated dynamic traffic,
Dijkstra baseline, QUBO route selection, a QAOA statevector simulator, and CSV/graph output.

## 1. Requirements
- Python 3.10 or newer recommended
- Internet access during initial package installation
- Qiskit is used for the QAOA statevector simulation

## 2. Setup

### Windows
```powershell
cd Dynamic-Quantum-Routing
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
cd Dynamic-Quantum-Routing
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit normally opens at http://localhost:8501.

## 3. UI workflow
1. Choose source and destination routers in the sidebar.
2. Select Low, Medium, or High traffic and optionally enable traffic bursts.
3. Click **Generate traffic** to create a new simulated network state.
4. Click **Optimize route**.
5. Inspect the highlighted route, QUBO model, and algorithm comparison.
6. Download saved experiment history as CSV.

## 4. Standalone run
```bash
python main.py
```
This runs a small experiment and appends records to `results/routing_results.csv`.

## 5. Architecture and module mapping
- `app.py`: Streamlit UI; calls shared project modules.
- `main.py`: standalone experiment entry point.
- `topology.py`: network graph.
- `traffic.py`: synthetic traffic generation.
- `congestion.py`: congestion scores.
- `routing.py`: Dijkstra, candidate routes, route metrics.
- `qubo_model.py`: exactly-one-candidate QUBO.
- `quantum_optimizer.py`: exhaustive classical QUBO and QAOA statevector simulation.
- `performance.py`: CSV history and HTML graph output.
- `results/`: saved experiment data and graphs.

## 6. Scientific limitations
- Traffic utilization, queue indicators, delay, and packet loss are simulated, not collected from real packet flows.
- QAOA runs on a classical simulator, not quantum hardware.
- The QAOA circuit is intended for a small number of route variables. It is not expected to outperform Dijkstra or exact classical search for this toy topology.
- The QUBO optimizer selects among a bounded set of candidate routes; it does not solve every possible network routing configuration.
- For a real packet-level evaluation, integrate Mininet and a controller, and measure traffic with tools such as ping/iperf. Do not describe this simulator alone as a live production network.

## 7. Troubleshooting
- If `streamlit` is not found, activate the virtual environment and rerun `pip install -r requirements.txt`.
- If QAOA fails due to a Qiskit installation issue, the UI still reports Dijkstra and classical QUBO results and shows the error.
- If package installation fails, check your Python version and install a compatible Python 3.10+ environment.
