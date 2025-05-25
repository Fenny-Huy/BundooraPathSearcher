# test_routes.py

import os
from algorithms.graph_builder import build_graph
from algorithms.astar_search   import astar
from utils.edge_mapper         import EdgeMapper
from models.predictor          import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor

# ─── Configuration ───────────────────────────────────────────────────────────────

NODE_CSV   = "data/scats_complete_average.csv"
VOLUME_PKL = "data/traffic_model_ready.pkl"

MODEL_CLASSES = {
    "LSTM": LSTMPredictor,
    "GRU":  GRUPredictor,
    "MLP":  MLPPredictor,
    "TCN":  TCNPredictor,
}

MODEL_DIRS = {
    "LSTM": "lstm_saved_models",
    "GRU":  "gru_saved_models",
    "MLP":  "mlp_saved_models",
    "TCN":  "tcn_saved_models",
}

# Define test cases: (source, target, timestamp, k_routes)
TEST_CASES = [
    ("970", "4030", "2006-10-08 14:45:00", 3),
    ("3682", "3126", "2006-10-15 08:30:00", 2),
    ("3804", "4264", "2006-10-20 17:00:00", 4),
    ("3120", "4057", "2006-10-25 12:15:00", 2),
]

# ─── Setup ──────────────────────────────────────────────────────────────────────

print("🔍 Building graph …")
centroids, edges = build_graph(NODE_CSV)

print("🗺️  Initializing edge→arm mapper …")
mapper = EdgeMapper(arms_pkl=VOLUME_PKL, nodes_csv="data/scats_complete.csv")

# ─── Run Tests ──────────────────────────────────────────────────────────────────

for src, dst, ts, k in TEST_CASES:
    print("\n" + "="*60)
    print(f"Test case: {src} → {dst} at {ts}, k={k}")
    print("="*60)
    
    for model_name, PredictorCls in MODEL_CLASSES.items():
        print(f"\n[Model: {model_name}]")
        # instantiate predictor
        models_dir = MODEL_DIRS[model_name]
        predictor = PredictorCls(data_pkl=VOLUME_PKL, models_dir=models_dir)
        
        # run A*
        try:
            results = astar(src, dst, centroids, edges, predictor, ts, k=k)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
        
        if not results:
            print("  ❌ No routes found.")
            continue
        
        # print each route
        for idx, (path, total_time, total_dist) in enumerate(results, start=1):
            print(f"  → Route #{idx}")
            print("    " + " → ".join(path))
            print(f"    Time: {total_time:.1f} min   Distance: {total_dist:.2f} km")
    print("\n" + "-"*60)

print("\nAll tests complete.")
