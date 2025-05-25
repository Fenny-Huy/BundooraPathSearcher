# test_routes.py

import os
import ast
from algorithms.graph_builder import build_graph
from algorithms.astar_search   import astar
from utils.edge_mapper         import EdgeMapper
from models.predictor          import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor

# ─── Configuration ───────────────────────────────────────────────────────────────

NODE_CSV   = "data/scats_complete_average.csv"
VOLUME_PKL = "data/traffic_model_ready.pkl"
TEST_CASES_FILE = "test_cases.txt"

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

# ─── Read Test Cases ─────────────────────────────────────────────────────────────

def load_test_cases(file_path):
    test_cases = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                case = ast.literal_eval(line)
                if isinstance(case, tuple) and len(case) == 4:
                    test_cases.append(case)
                else:
                    print(f"Skipping invalid test case: {line}")
    return test_cases

TEST_CASES = load_test_cases(TEST_CASES_FILE)

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
        
        try:
            results = astar(src, dst, centroids, edges, predictor, ts, k=k)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
        
        if not results:
            print("  ❌ No routes found.")
            continue
        
        for idx, (path, total_time, total_dist) in enumerate(results, start=1):
            print(f"  → Route #{idx}")
            print("    " + " → ".join(path))
            print(f"    Time: {total_time:.1f} min   Distance: {total_dist:.2f} km")
    print("\n" + "-"*60)

print("\n✅ All tests complete.")
