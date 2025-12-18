from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from network_builder import NetworkBuilder
from data_loader import DataLoader
from data_processor import DataProcessor
from movie_scorer import MovieScorer
from config import EXCEL_FILE

# =========================
# 创建 Flask 应用（只一次）
# =========================
app = Flask(__name__)
CORS(app)   # ⭐ 必须紧跟在这里

# =========================
# 初始化（启动时只跑一次）
# =========================
loader = DataLoader(EXCEL_FILE)
df = loader.load_data()
df = loader.rename_columns()
df = loader.clean_data()

processor = DataProcessor(df)
persons_dict = processor.extract_all_persons()
edges_df = processor.build_cooperation_edges()

builder = NetworkBuilder()
G = builder.build_network(edges_df, persons_dict)

person_stats = processor.get_person_stats()
scorer = MovieScorer(G, person_stats)

# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    directors = data.get("directors", [])
    actors = data.get("actors", [])
    weights = data.get("weights", {
        "actor": 0.4,
        "director": 0.4,
        "relation": 0.2
    })

    result = scorer.predict(directors, actors, weights)
    return jsonify(result)

# =========================
# 启动服务
# =========================
if __name__ == "__main__":
    app.run(debug=True)
