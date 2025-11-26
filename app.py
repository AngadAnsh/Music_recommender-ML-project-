from flask import Flask, render_template, request, jsonify
import importlib
from urllib.parse import quote
from ytmusicapi import YTMusic

# Load the ML recommender
cr = importlib.import_module("src.content_recommender")

# Load dataset + model
df, scaled_features = cr.build_model()

# Initialize YTMusic (NO API KEY NEEDED)
ytmusic = YTMusic()

# ----------------------------------------------------
# Initialize Flask
# ----------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------
# Get YouTube Video ID using YTMusic (perfect results)
# ----------------------------------------------------
def get_youtube_video(title, artist):
    query = f"{title} {artist}"

    try:
        results = ytmusic.search(query, filter="songs")

        if results and "videoId" in results[0]:
            return results[0]["videoId"]
    except:
        return None

    return None


# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------
# RECOMMENDATION ROUTE
# ----------------------------------------------------
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == "POST":
        song = request.form.get("song", "").strip()
        selected_genres = request.form.getlist("filter_genre")
    else:
        song = request.args.get("song", "").strip()
        selected_genres = []

    # ML Recommendation
    if selected_genres:
        results = cr.recommend_with_genres(
            song, df, scaled_features, genres=selected_genres, top_k=10
        )
    else:
        results = cr.recommend(song, df, scaled_features, top_k=10)

    if not results:
        return "<h3 style='color:white;'>No songs found. Try another.</h3>"

    # ----------------------------------------------------
    # Spotify UI + Real YouTube Thumbnails
    # ----------------------------------------------------
    html = """
    <style>
        body { font-family: 'Inter', sans-serif; background: #121212; }
        .section-title { font-size: 26px; font-weight: 700; color: #fff; margin-bottom: 20px; }
        .song-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 20px; }
        .song-card { background: #181818; padding: 18px; border-radius: 12px; text-align: center;
                     transition: 0.2s; }
        .song-card:hover { transform: translateY(-4px); background: #242424; }
        .song-card img { width: 100%; height: 180px; border-radius: 10px;
                         object-fit: cover; margin-bottom: 12px; }
        .track-name { font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 4px; }
        .artist-name { font-size: 14px; color: #bbb; margin-bottom: 10px; }
        .open-btn { padding: 6px 12px; background: #1DB954; color: black;
                    border-radius: 20px; font-weight: bold;
                    text-decoration: none; display:inline-block; margin-top:6px; }
        .open-btn:hover { background: #1ed760; }
    </style>

    <h2 class='section-title'>Recommended Songs</h2>
    <div class='song-grid'>
    """

    # Loop over recommended songs
    for r in results:
        title = r["track_name"]
        artist = r["artists"]

        # Get REAL YouTube video ID
        vid = get_youtube_video(title, artist)

        if vid:
            thumbnail = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
            youtube_url = f"https://www.youtube.com/watch?v={vid}"
        else:
            thumbnail = "https://i.imgur.com/Q9JYbU8.png"
            youtube_url = "https://www.youtube.com/results?search_query=" + quote(f"{title} {artist}")

        # Spotify search link
        spotify_url = f"https://open.spotify.com/search/{quote(f'{title} {artist}')}"

        html += f"""
        <div class='song-card'>
            <img src="{thumbnail}">
            <div class='track-name'>{title}</div>
            <div class='artist-name'>{artist}</div>

            <a class='open-btn' href="{youtube_url}" target="_blank">YouTube</a>
            <a class='open-btn' href="{spotify_url}" target="_blank" style="margin-left:8px;">Spotify</a>
        </div>
        """

    html += "</div>"
    return html


# ----------------------------------------------------
# AUTOCOMPLETE SUGGESTIONS
# ----------------------------------------------------
@app.route("/suggest")
def suggest():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    names = cr.suggest_names(q, df, limit=25)
    return jsonify(names)


# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
