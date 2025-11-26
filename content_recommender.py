import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# FEATURES (lowercase)
# ----------------------------------
FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

CACHE_DIR = "data"
SCALER_PATH = os.path.join(CACHE_DIR, "scaler.joblib")
SCALED_PATH = os.path.join(CACHE_DIR, "scaled_features.npy")


# ------------------------------------------------------
# Load dataset & rename columns
# ------------------------------------------------------
def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    rename_map = {
        "SongName": "track_name",
        "ArtistName": "artists",
        "Danceability": "danceability",
        "Energy": "energy",
        "Speechiness": "speechiness",
        "Acousticness": "acousticness",
        "Instrumentalness": "instrumentalness",
        "Liveness": "liveness",
        "Valence": "valence",
        "Tempo": "tempo"
    }

    df = df.rename(columns=rename_map)

    if 'track_name' not in df.columns:
        raise ValueError("track_name missing after renaming")
    if 'artists' not in df.columns:
        df['artists'] = "Unknown"

    # Ensure ML features exist
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    df['track_name'] = df['track_name'].fillna("Unknown")
    df['artists'] = df['artists'].fillna("Unknown")

    return df


# ------------------------------------------------------
# AI GENRE CLASSIFIER (Hindi + Punjabi + others)
# ------------------------------------------------------
def classify_genre(row):
    name = row["artists"].lower()
    d = row["danceability"]
    e = row["energy"]
    s = row["speechiness"]
    a = row["acousticness"]
    v = row["valence"]
    t = row["tempo"]

    # -------------------------------
    # PUNJABI DETECTION
    # -------------------------------
    punjabi_artists = [
        "sidhu", "moose", "sidhu moose wala",
        "ap dhillon", "karan aujla", "diljit",
        "jass", "amrit maan", "shubh"
    ]
    if any(x in name for x in punjabi_artists):
        return "Punjabi"

    # -------------------------------
    # HINDI / BOLLYWOOD DETECTION
    # -------------------------------
    hindi_artists = [
        "arijit", "pritam", "badshah",
        "armaan", "jubin", "shreya",
        "atif", "amit trivedi", "darshan", "neha",
        "yo yo honey singh"
    ]
    if any(x in name for x in hindi_artists):
        return "Hindi"

    # -------------------------------
    # AUDIO FEATURE–BASED GENRES
    # -------------------------------

    # Rap
    if s > 0.40 and e > 0.55:
        return "Rap"

    # EDM
    if e > 0.75 and t > 120 and a < 0.3:
        return "EDM"

    # Acoustic
    if a > 0.6 and e < 0.5:
        return "Acoustic"

    # Sad
    if v < 0.3 and e < 0.5:
        return "Sad"

    # Bollywood (balanced)
    if 0.3 < d < 0.7 and 0.3 < e < 0.7 and 0.3 < v < 0.7:
        return "Bollywood"

    # Pop (default)
    if d > 0.55 and v > 0.45:
        return "Pop"

    return "Pop"


# ------------------------------------------------------
# Prepare features & caching
# ------------------------------------------------------
def prepare_and_cache_features(df, force_rebuild=False):
    os.makedirs(CACHE_DIR, exist_ok=True)

    if (not force_rebuild) and os.path.exists(SCALER_PATH) and os.path.exists(SCALED_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            scaled = np.load(SCALED_PATH)
            if scaled.shape[0] == len(df):
                return scaler, scaled
        except:
            pass

    X = df[FEATURES].astype(float).values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)
    np.save(SCALED_PATH, scaled)

    return scaler, scaled


# ------------------------------------------------------
# Build complete ML model + AI genre tags
# ------------------------------------------------------
def build_model(force_rebuild=False):
    df_list = []

    if os.path.exists("data/SpotifySongs.csv"):
        df_list.append(load_data("data/SpotifySongs.csv"))

    if not df_list:
        raise FileNotFoundError("No dataset found.")

    df = pd.concat(df_list, ignore_index=True)

    # ADD AI GENRE COLUMN
    df["track_genre"] = df.apply(classify_genre, axis=1)

    scaler, scaled = prepare_and_cache_features(df, force_rebuild)
    return df, scaled


# ------------------------------------------------------
# Autocomplete suggestions
# ------------------------------------------------------
def suggest_names(query, df, genres=None, limit=200):
    if not query:
        return []

    q = query.lower()
    subset = df

    if genres and "track_genre" in df.columns:
        genres = [g.lower() for g in genres]
        subset = df[df["track_genre"].str.lower().isin(genres)]

    mask = subset["track_name"].str.lower().str.contains(q)
    return subset[mask]["track_name"].unique().tolist()[:limit]


# ------------------------------------------------------
# Main Recommendation Engine
# ------------------------------------------------------
def recommend(song_name, df, scaled, top_k=10):
    if not song_name:
        return []

    s = song_name.lower()

    idx_list = df.index[df["track_name"].str.lower() == s].tolist()
    if not idx_list:
        idx_list = df.index[df["track_name"].str.lower().str.contains(s)].tolist()
    if not idx_list:
        return []

    idx = idx_list[0]
    vec = scaled[idx].reshape(1, -1)
    sims = cosine_similarity(vec, scaled)[0]

    pairs = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)

    results = []
    seen = set()

    for i, score in pairs:
        if i == idx:
            continue

        name = df.iloc[i]["track_name"]
        if name in seen:
            continue

        seen.add(name)

        results.append({
            "track_name": name,
            "artists": df.iloc[i]["artists"],
            "score": float(score)
        })

        if len(results) >= top_k:
            break

    return results


# ------------------------------------------------------
# Genre-Based Recommendations
# ------------------------------------------------------
def recommend_with_genres(song_name, df, scaled, genres=None, top_k=10):
    if not genres or "track_genre" not in df.columns:
        return recommend(song_name, df, scaled, top_k)

    genres = [g.lower() for g in genres]

    mask = df["track_genre"].str.lower().isin(genres)
    if mask.sum() == 0:
        return recommend(song_name, df, scaled, top_k)

    subdf = df[mask].reset_index(drop=True)
    sub_scaled = scaled[mask.values]

    return recommend_on_subset(song_name, subdf, sub_scaled, top_k)


def recommend_on_subset(song_name, subdf, scaled, top_k=10):
    s = song_name.lower()

    idx_list = subdf.index[subdf["track_name"].str.lower() == s].tolist()
    if not idx_list:
        idx_list = subdf.index[subdf["track_name"].str.lower().str.contains(s)].tolist()
    if not idx_list:
        return []

    idx = idx_list[0]
    sims = cosine_similarity(scaled[idx].reshape(1, -1), scaled)[0]

    pairs = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)

    results = []
    seen = set()

    for i, score in pairs:
        if i == idx:
            continue

        name = subdf.iloc[i]["track_name"]
        if name in seen:
            continue

        seen.add(name)

        results.append({
            "track_name": name,
            "artists": subdf.iloc[i]["artists"],
            "score": float(score)
        })

        if len(results) >= top_k:
            break

    return results
