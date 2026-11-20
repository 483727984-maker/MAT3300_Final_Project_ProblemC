"""
Minimal data preprocessing for Model II (and shared later sections).

Reads only the four provided ICM files under ProblemC_Music_data/.
Writes clean tables + a short summary JSON/CSV for the report.

Usage (from repo root):
  /tmp/mat3300_venv/bin/python code/preprocess_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ProblemC_Music_data"
OUT = ROOT / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "duration_ms",
    "popularity",
]


def build_genre_map(influence: pd.DataFrame) -> dict[str, str]:
    genre_map: dict[str, str] = {}
    for _, row in influence.iterrows():
        genre_map[str(row["influencer_id"])] = str(row["influencer_main_genre"])
        genre_map[str(row["follower_id"])] = str(row["follower_main_genre"])
    return genre_map


def main() -> None:
    artists = pd.read_csv(DATA / "data_by_artist.csv")
    influence = pd.read_csv(DATA / "influence_data.csv")
    by_year = pd.read_csv(DATA / "data_by_year.csv")
    # full_music_data is large; only record its size for the exploration note
    full_path = DATA / "full_music_data.csv"
    n_full = sum(1 for _ in open(full_path, encoding="utf-8")) - 1

    artists["artist_id"] = artists["artist_id"].astype(str)
    genre_map = build_genre_map(influence)
    artists["genre"] = artists["artist_id"].map(genre_map)

    # Feature completeness
    missing_feat = artists[FEATURE_COLS].isna().sum().to_dict()

    with_genre = artists.dropna(subset=["genre"]).copy()
    with_genre = with_genre[with_genre["genre"].str.lower() != "unknown"].copy()

    # Influence network basic counts
    n_edges = len(influence)
    n_nodes = len(set(influence["influencer_id"].astype(str)) | set(influence["follower_id"].astype(str)))

    summary = {
        "n_songs_full_music_data": int(n_full),
        "n_artists_data_by_artist": int(len(artists)),
        "n_artists_with_genre_for_model_ii": int(len(with_genre)),
        "n_artists_missing_genre": int(artists["genre"].isna().sum()),
        "n_influence_edges": int(n_edges),
        "n_influence_nodes": int(n_nodes),
        "n_years_data_by_year": int(len(by_year)),
        "year_min": int(by_year["year"].min()),
        "year_max": int(by_year["year"].max()),
        "feature_cols": FEATURE_COLS,
        "missing_feature_counts": {k: int(v) for k, v in missing_feat.items()},
        "genre_counts_top10": with_genre["genre"].value_counts().head(10).astype(int).to_dict(),
        "n_genres_used": int(with_genre["genre"].nunique()),
    }

    # Clean table used by Model II
    cols_out = ["artist_id", "artist_name", "genre", "count"] + FEATURE_COLS
    cols_out = [c for c in cols_out if c in with_genre.columns]
    clean = with_genre[cols_out].reset_index(drop=True)
    clean.to_csv(OUT / "artists_features_with_genre.csv", index=False)

    with open(OUT / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    genre_counts = with_genre["genre"].value_counts().rename_axis("genre").reset_index(name="n_artists")
    genre_counts.to_csv(OUT / "genre_artist_counts.csv", index=False)

    print("=== Preprocess summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT / 'artists_features_with_genre.csv'}")
    print(f"Wrote {OUT / 'genre_artist_counts.csv'}")
    print(f"Wrote {OUT / 'preprocess_summary.json'}")


if __name__ == "__main__":
    main()
