"""
Model II: Music Similarity
- Baseline: z-score standardize -> Euclidean distance
- Robustness: z-score -> PCA -> Euclidean distance
- Test: within-genre mean distance vs between-genre mean distance

Data (only the four provided files):
  ProblemC_Music_data/data_by_artist.csv
  ProblemC_Music_data/influence_data.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
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


def load_artist_features_with_genre() -> pd.DataFrame:
    """Join artist-level Spotify features with main_genre from influence_data."""
    artists = pd.read_csv(DATA / "data_by_artist.csv")
    influence = pd.read_csv(DATA / "influence_data.csv")

    # Build artist_id -> genre map (later rows overwrite earlier; labels are stable)
    genre_map: dict[str, str] = {}
    for _, row in influence.iterrows():
        genre_map[str(row["influencer_id"])] = row["influencer_main_genre"]
        genre_map[str(row["follower_id"])] = row["follower_main_genre"]

    artists["artist_id"] = artists["artist_id"].astype(str)
    artists["genre"] = artists["artist_id"].map(genre_map)

    # Keep artists that have both features and a genre label
    df = artists.dropna(subset=["genre"]).copy()
    df = df[df["genre"].astype(str).str.lower() != "unknown"].copy()
    return df.reset_index(drop=True)


def pairwise_mean_distance(
    X: np.ndarray,
    genres: np.ndarray,
    max_pairs_per_bucket: int = 200_000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Estimate mean Euclidean distance within / between genres.

    Full all-pairs can be huge (n~5600 => ~15M pairs). We subsample pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = X.shape[0]
    # Sample random pairs and classify them
    # Aim for roughly max_pairs_per_bucket in each bucket (capped by availability)
    n_draw = min(n * (n - 1) // 2, max(2 * max_pairs_per_bucket, 500_000))

    i = rng.integers(0, n, size=n_draw)
    j = rng.integers(0, n, size=n_draw)
    mask = i < j
    i, j = i[mask], j[mask]

    same = genres[i] == genres[j]
    diff = ~same

    def mean_dist(idx_i: np.ndarray, idx_j: np.ndarray, cap: int) -> float:
        if len(idx_i) == 0:
            return float("nan")
        if len(idx_i) > cap:
            choose = rng.choice(len(idx_i), size=cap, replace=False)
            idx_i, idx_j = idx_i[choose], idx_j[choose]
        d = np.linalg.norm(X[idx_i] - X[idx_j], axis=1)
        return float(d.mean())

    d_within = mean_dist(i[same], j[same], max_pairs_per_bucket)
    d_between = mean_dist(i[diff], j[diff], max_pairs_per_bucket)
    return {
        "d_within": d_within,
        "d_between": d_between,
        "n_within_sampled_pool": int(same.sum()),
        "n_between_sampled_pool": int(diff.sum()),
    }


def genre_centroid_distance_matrix(X: np.ndarray, genres: np.ndarray) -> pd.DataFrame:
    """Mean feature vector per genre, then pairwise Euclidean distances between centroids."""
    uniq = np.array(sorted(pd.unique(genres)))
    centroids = []
    for g in uniq:
        centroids.append(X[genres == g].mean(axis=0))
    C = np.vstack(centroids)
    # pairwise distances between centroids
    # d_ab = ||c_a - c_b||
    diff = C[:, None, :] - C[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    return pd.DataFrame(D, index=uniq, columns=uniq)


def run_pipeline(
    X_raw: np.ndarray,
    genres: np.ndarray,
    pca_variance: float = 0.80,
) -> None:
    # ----- Baseline: standardize + Euclidean -----
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    base = pairwise_mean_distance(X_std, genres)
    print("=== Baseline: Standardized Euclidean ===")
    print(f"  mean within-genre distance : {base['d_within']:.4f}")
    print(f"  mean between-genre distance: {base['d_between']:.4f}")
    print(f"  within < between?          : {base['d_within'] < base['d_between']}")

    D_cent = genre_centroid_distance_matrix(X_std, genres)
    D_cent.to_csv(OUT / "genre_centroid_distances_baseline.csv")
    print(f"  saved centroid distance matrix -> {OUT / 'genre_centroid_distances_baseline.csv'}")

    # ----- Robustness: PCA then Euclidean -----
    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X_std)
    r = X_pca.shape[1]
    explained = float(pca.explained_variance_ratio_.sum())

    pca_res = pairwise_mean_distance(X_pca, genres)
    print("\n=== Robustness: PCA + Euclidean ===")
    print(f"  components kept (r)        : {r}")
    print(f"  variance explained         : {explained:.4f}")
    print(f"  mean within-genre distance : {pca_res['d_within']:.4f}")
    print(f"  mean between-genre distance: {pca_res['d_between']:.4f}")
    print(f"  within < between?          : {pca_res['d_within'] < pca_res['d_between']}")

    D_cent_pca = genre_centroid_distance_matrix(X_pca, genres)
    D_cent_pca.to_csv(OUT / "genre_centroid_distances_pca.csv")
    print(f"  saved PCA centroid matrix  -> {OUT / 'genre_centroid_distances_pca.csv'}")

    # Save a small summary table for the report
    summary = pd.DataFrame(
        [
            {
                "model": "euclidean_standardized",
                "d_within": base["d_within"],
                "d_between": base["d_between"],
                "within_lt_between": base["d_within"] < base["d_between"],
                "pca_r": np.nan,
                "pca_variance_explained": np.nan,
            },
            {
                "model": "pca_then_euclidean",
                "d_within": pca_res["d_within"],
                "d_between": pca_res["d_between"],
                "within_lt_between": pca_res["d_within"] < pca_res["d_between"],
                "pca_r": r,
                "pca_variance_explained": explained,
            },
        ]
    )
    summary.to_csv(OUT / "model_ii_summary.csv", index=False)
    print(f"\nSaved summary -> {OUT / 'model_ii_summary.csv'}")


def main() -> None:
    df = load_artist_features_with_genre()
    print(f"Artists with features + genre: {len(df)}")
    print("Genre counts (top 10):")
    print(df["genre"].value_counts().head(10).to_string())
    print()

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    genres = df["genre"].to_numpy()
    run_pipeline(X, genres, pca_variance=0.80)


if __name__ == "__main__":
    main()
