from flask import Flask, render_template, request, jsonify
import importlib

cr = importlib.import_module("src.content_recommender")
df, scaled_features = cr.build_model("data/tracks.csv")  # cached scaled features

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_route():
    song = request.form.get("song", "").strip()
    top_k = int(request.form.get("top_k", 10))
    # gather checked genres
    checked = request.form.getlist("filter_genre") or []
    # pass genres list into a new recommend_with_genres helper (below)
    results = cr.recommend_with_genres(song, df, scaled_features, genres=checked, top_k=top_k)
    return render_template("results.html", song=song, results=results)

@app.route("/suggest")
def suggest():
    q = request.args.get("q", "").strip()
    genres = request.args.get("genres", "")
    genres = [g for g in genres.split(",") if g]
    if not q:
        return jsonify([])
    # get top 25 suggestions from recommender helper
    names = cr.suggest_names(q, df, genres=genres, limit=25)
    return jsonify(names)
