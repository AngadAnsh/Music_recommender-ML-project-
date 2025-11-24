import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------
# LOAD + PREPARE MODEL
# ---------------------------------------------------------
def build_model(csv_path):
    df = pd.read_csv(csv_path)

    features = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    # Clean missing
    for f in features:
        df[f] = df[f].fillna(0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    return df, scaled


# ---------------------------------------------------------
# BASIC RECOMMENDATION (safe — no full similarity matrix)
# ---------------------------------------------------------
def recommend(song_name, df, scaled_features, top_k=10):

    # Try exact match
    matches = df.index[df['track_name'] == song_name].tolist()

    # Try lower-case match
    if not matches:
        matches = df.index[df['track_name'].str.lower() == song_name.lower()].tolist()

    # Try contains
    if not matches:
        matches = df.index[df['track_name'].str.contains(song_name, case=False, na=False)].tolist()

    # No match → return empty
    if not matches:
        return []

    idx = matches[0]
    q = scaled_features[idx].reshape(1, -1)

    sim = cosine_similarity(q, scaled_features)[0]
    scores = list(enumerate(sim))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, sc in scores:
        if i == idx: 
            continue
        results.append({
            "track_name": df.iloc[i]["track_name"],
            "artist": df.iloc[i]["artists"],
            "score": float(sc)
        })
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------
# AUTOCOMPLETE SEARCH
# ---------------------------------------------------------
def suggest_names(query, df, genres=None, limit=25):
    if not query:
        return []

    q = query.lower()

    subset = df
    if genres:
        if 'track_genre' in df.columns:
            mask = df['track_genre'].astype(str).str.lower().apply(
                lambda g: any(gg.lower() in g for gg in genres)
            )
            subset = df[mask]

    names = subset['track_name'].astype(str)
    matches = names[names.str.lower().str.contains(q)].unique().tolist()

    return matches[:limit]


# ---------------------------------------------------------
# FILTERED RECOMMENDATION (Hindi / Rap / Punjabi)
# ---------------------------------------------------------
def recommend_with_genres(song_name, df, scaled_features, genres=None, top_k=10):

    if genres and 'track_genre' in df.columns:

        mask = df['track_genre'].astype(str).str.lower().apply(
            lambda g: any(gg.lower() in g for gg in genres)
        )

        subdf = df[mask].reset_index(drop=True)
        original_idxs = df[mask].index.to_numpy()

        sub_scaled = scaled_features[original_idxs]

        return recommend_on_subset(song_name, subdf, sub_scaled, top_k)

    return recommend(song_name, df, scaled_features, top_k)


# ---------------------------------------------------------
# RECOMMENDATION ON SUBSET (genre filtered)
# ---------------------------------------------------------
def recommend_on_subset(song_name, subdf, scaled_sub, top_k=10):

    matches = subdf.index[subdf['track_name'] == song_name].tolist()
    if not matches:
        matches = subdf.index[subdf['track_name'].str.lower() == song_name.lower()].tolist()
    if not matches:
        matches = subdf.index[subdf['track_name'].str.contains(song_name, case=False)].tolist()

    if not matches:
        return []

    idx = matches[0]
    q = scaled_sub[idx].reshape(1, -1)

    sim = cosine_similarity(q, scaled_sub)[0]
    scores = sorted(list(enumerate(sim)), key=lambda x: x[1], reverse=True)

    results = []
    for i, sc in scores:
        if i == idx:
            continue

        results.append({
            "track_name": subdf.iloc[i]["track_name"],
            "artist": subdf.iloc[i]["artists"],
            "score": float(sc)
        })

        if len(results) >= top_k:
            break

    return results
