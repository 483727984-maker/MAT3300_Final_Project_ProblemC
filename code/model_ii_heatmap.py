"""
Model II: genre–genre Euclidean-distance heatmap (reproducible).

Produces:
  code/outputs/heatmap_genre_distances_baseline.png
  code/outputs/heatmap_genre_distances_pca.png
  (also reuses / writes the CSV matrices from model_ii_similarity.py)

Usage:
  python code/model_ii_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Allow `python code/model_ii_heatmap.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_ii_similarity import (  # noqa: E402
    FEATURE_COLS,
    OUT,
    genre_centroid_distance_matrix,
    load_artist_features_with_genre,
)

OUT.mkdir(parents=True, exist_ok=True)


def plot_distance_heatmap(
    D: pd.DataFrame,
    title: str,
    out_path: Path,
    figsize: tuple[float, float] = (10.5, 8.5),
) -> None:
    """Plot a symmetric genre–genre Euclidean distance heatmap."""
    labels = list(D.index)
    M = D.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="viridis_r", aspect="equal")
    # viridis_r: smaller distance (more similar) -> brighter / warmer end after reverse
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Euclidean distance (smaller = more similar)")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Genre")

    # Optional numeric annotations if the matrix is not too large
    if len(labels) <= 22:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if M[i, j] > np.nanmedian(M) else "black",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"saved {out_path.with_suffix('.pdf')}")


def main() -> None:
    df = load_artist_features_with_genre()
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    genres = df["genre"].to_numpy()

    # Baseline: standardized Euclidean on genre centroids
    X_std = StandardScaler().fit_transform(X)
    D_base = genre_centroid_distance_matrix(X_std, genres)
    D_base.to_csv(OUT / "genre_centroid_distances_baseline.csv")
    plot_distance_heatmap(
        D_base,
        title="Genre centroid distances (standardized Euclidean)",
        out_path=OUT / "heatmap_genre_distances_baseline.png",
    )

    # Robustness: PCA (80% variance) then Euclidean on centroids
    pca = PCA(n_components=0.80, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X_std)
    D_pca = genre_centroid_distance_matrix(X_pca, genres)
    D_pca.to_csv(OUT / "genre_centroid_distances_pca.csv")
    plot_distance_heatmap(
        D_pca,
        title=(
            f"Genre centroid distances (PCA→Euclidean; "
            f"r={X_pca.shape[1]}, var={pca.explained_variance_ratio_.sum():.1%})"
        ),
        out_path=OUT / "heatmap_genre_distances_pca.png",
    )

    # Quick sanity: diagonal should be ~0
    print("baseline diagonal max:", float(np.diag(D_base).max()))
    print("pca diagonal max:", float(np.diag(D_pca).max()))


if __name__ == "__main__":
    main()
