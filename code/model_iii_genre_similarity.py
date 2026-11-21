"""
Model III (Q3): genre-level similarity via influence-weighted feature centroids.

Pipeline
  1. Join Model-II artist features with Q1 influence scores (katz_out).
  2. Globally z-score the p=11 Spotify features (same space as Model II).
  3. For each genre g, form a centroid
       c_g = sum_i w_i x̃_i / sum_i w_i ,
     with primary weights w_i = Katz outgoing score from Q1.
     Equal weights are kept as a robustness baseline.
  4. Between-genre similarity: Euclidean distance between centroids
     (and s = 1/(1+d) for a bounded similarity score).
  5. Within-genre cohesion: mean distance of artists in g to c_g.

Usage (from repo root):
  export MPLBACKEND=Agg
  export MPLCONFIGDIR=code/.mplconfig
  mkdir -p "$MPLCONFIGDIR"
  /tmp/mat3300_venv/bin/python code/model_iii_genre_similarity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code" / "outputs"
FIG = ROOT / "code" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

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

WEIGHT_COL = "katz_out"  # Q1 primary influence measure


def load_joined() -> pd.DataFrame:
    feat = pd.read_csv(OUT / "artists_features_with_genre.csv")
    ind = pd.read_csv(ROOT / "code" / "q1_outputs" / "artist_network_indicators.csv")
    feat["artist_id"] = feat["artist_id"].astype(str)
    ind["artist_id"] = ind["artist_id"].astype(str)
    cols = ["artist_id", "katz_out", "out_degree", "pagerank_reverse"]
    df = feat.merge(ind[cols], on="artist_id", how="inner")
    if df["katz_out"].isna().any():
        raise ValueError("Missing Katz scores after join.")
    # Guarantee strictly positive weights (Katz baseline is already >= 1)
    df["weight_katz"] = df[WEIGHT_COL].clip(lower=1e-12)
    return df


def weighted_centroids(
    X: np.ndarray,
    genres: np.ndarray,
    weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return genre x feature centroid matrix and total weight per genre."""
    g_list = sorted(pd.unique(genres))
    rows = []
    totals = {}
    for g in g_list:
        mask = genres == g
        w = weights[mask]
        w_sum = float(w.sum())
        totals[g] = w_sum
        rows.append((w[:, None] * X[mask]).sum(axis=0) / w_sum)
    C = pd.DataFrame(rows, index=g_list, columns=FEATURE_COLS)
    return C, pd.Series(totals, name="weight_sum")


def pairwise_euclidean(C: pd.DataFrame) -> pd.DataFrame:
    labels = list(C.index)
    M = C.to_numpy(dtype=float)
    n = len(labels)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(M[i] - M[j]))
            D[i, j] = D[j, i] = d
    return pd.DataFrame(D, index=labels, columns=labels)


def distance_to_similarity(D: pd.DataFrame) -> pd.DataFrame:
    return 1.0 / (1.0 + D)


def within_genre_cohesion(
    X: np.ndarray,
    genres: np.ndarray,
    C: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Mean (optionally weighted) distance of artists to their genre centroid."""
    rows = []
    for g in C.index:
        mask = genres == g
        Xg = X[mask]
        cg = C.loc[g].to_numpy(dtype=float)
        dists = np.linalg.norm(Xg - cg, axis=1)
        if weights is None:
            mean_d = float(dists.mean())
            n = int(mask.sum())
        else:
            w = weights[mask]
            mean_d = float((w * dists).sum() / w.sum())
            n = int(mask.sum())
        rows.append({"genre": g, "n_artists": n, "mean_dist_to_centroid": mean_d})
    return pd.DataFrame(rows).sort_values("mean_dist_to_centroid")


def plot_heatmap(D: pd.DataFrame, title: str, out_stem: Path, cbar_label: str) -> None:
    labels = list(D.index)
    M = D.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(M, cmap="viridis_r", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Genre")
    if len(labels) <= 22:
        med = float(np.nanmedian(M))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if M[i, j] > med else "black",
                )
    fig.tight_layout()
    for folder in (OUT, FIG):
        png = folder / f"{out_stem.name}.png"
        pdf = folder / f"{out_stem.name}.pdf"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"saved {png}")
        print(f"saved {pdf}")
    plt.close(fig)


def offdiag_mean(D: pd.DataFrame) -> float:
    M = D.to_numpy(dtype=float)
    mask = ~np.eye(M.shape[0], dtype=bool)
    return float(M[mask].mean())


def main() -> None:
    df = load_joined()
    X_raw = df[FEATURE_COLS].to_numpy(dtype=float)
    genres = df["genre"].to_numpy()
    X = StandardScaler().fit_transform(X_raw)

    # --- Influence-weighted (primary) ---
    C_katz, wsum = weighted_centroids(X, genres, df["weight_katz"].to_numpy())
    D_katz = pairwise_euclidean(C_katz)
    S_katz = distance_to_similarity(D_katz)
    coh_katz = within_genre_cohesion(X, genres, C_katz, df["weight_katz"].to_numpy())

    # --- Equal-weight baseline ---
    C_eq, _ = weighted_centroids(X, genres, np.ones(len(df)))
    D_eq = pairwise_euclidean(C_eq)
    S_eq = distance_to_similarity(D_eq)
    coh_eq = within_genre_cohesion(X, genres, C_eq, None)

    # Save centroids (standardized feature space)
    C_katz.to_csv(OUT / "genre_feature_vectors_katz.csv")
    C_eq.to_csv(OUT / "genre_feature_vectors_equal.csv")
    wsum.to_csv(OUT / "genre_katz_weight_sums.csv", header=True)

    D_katz.to_csv(OUT / "genre_centroid_distances_katz.csv")
    S_katz.to_csv(OUT / "genre_centroid_similarity_katz.csv")
    D_eq.to_csv(OUT / "genre_centroid_distances_equal.csv")
    S_eq.to_csv(OUT / "genre_centroid_similarity_equal.csv")

    coh = coh_katz.merge(coh_eq, on=["genre", "n_artists"], suffixes=("_katz", "_equal"))
    coh.to_csv(OUT / "genre_within_cohesion.csv", index=False)

    # Closest / farthest genre pairs (katz)
    pairs = []
    labels = list(D_katz.index)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                pairs.append((a, b, float(D_katz.loc[a, b]), float(S_katz.loc[a, b])))
    pairs_df = pd.DataFrame(pairs, columns=["genre_a", "genre_b", "distance", "similarity"])
    pairs_df = pairs_df.sort_values("distance")
    pairs_df.to_csv(OUT / "genre_pair_similarity_katz_ranked.csv", index=False)

    mean_within = float(coh_katz["mean_dist_to_centroid"].mean())
    mean_between = offdiag_mean(D_katz)
    mean_within_eq = float(coh_eq["mean_dist_to_centroid"].mean())
    mean_between_eq = offdiag_mean(D_eq)

    # Correlation between katz-weighted and equal distance matrices (off-diagonal)
    mask = ~np.eye(len(labels), dtype=bool)
    corr = float(
        np.corrcoef(D_katz.to_numpy()[mask], D_eq.to_numpy()[mask])[0, 1]
    )

    summary = {
        "n_artists": int(len(df)),
        "n_genres": int(df["genre"].nunique()),
        "weight": WEIGHT_COL,
        "mean_within_cohesion_katz": mean_within,
        "mean_between_centroid_distance_katz": mean_between,
        "within_lt_between_katz": mean_within < mean_between,
        "mean_within_cohesion_equal": mean_within_eq,
        "mean_between_centroid_distance_equal": mean_between_eq,
        "within_lt_between_equal": mean_within_eq < mean_between_eq,
        "offdiag_distance_corr_katz_vs_equal": corr,
        "closest_3_pairs_katz": pairs_df.head(3).to_dict(orient="records"),
        "farthest_3_pairs_katz": pairs_df.tail(3).iloc[::-1].to_dict(orient="records"),
    }
    with open(OUT / "model_iii_genre_similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_heatmap(
        D_katz,
        "Genre centroid distances (Katz-weighted; z-scored Euclidean)",
        Path("heatmap_genre_distances_katz"),
        "Euclidean distance (smaller = more similar)",
    )
    # Similarity heatmap: larger = more similar → use viridis (not reversed)
    labels = list(S_katz.index)
    M = S_katz.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(M, cmap="viridis", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity s=1/(1+d) (larger = more similar)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Genre centroid similarity s=1/(1+d) (Katz-weighted)")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Genre")
    med = float(np.nanmedian(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if M[i, j] < med else "black",
            )
    fig.tight_layout()
    for folder in (OUT, FIG):
        png = folder / "heatmap_genre_similarity_katz.png"
        pdf = folder / "heatmap_genre_similarity_katz.pdf"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"saved {png}")
        print(f"saved {pdf}")
    plt.close(fig)

    print("=== Model III genre similarity summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote outputs under {OUT}")


if __name__ == "__main__":
    main()
