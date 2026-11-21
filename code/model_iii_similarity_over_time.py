"""
Model III: genre similarity change over time.

Same definition as the static analysis, with a time index:
  - Globally z-score artist features (fixed scaler; same space as c_g).
  - Assign artists to decades by active_start (Q1 `start`).
  - In each decade t, form Katz-weighted centroids c_g(t) for genres with
    at least MIN_ARTISTS artists starting in that decade.
  - Between-genre distance d_gh(t) = ||c_g(t) - c_h(t)||_2;
    similarity s = 1/(1+d).

Also reports:
  - mean off-diagonal between-genre distance per decade
  - self-drift ||c_g(t) - c_g|| vs the static full-sample c_g
  - time series for selected genre pairs

Usage:
  export MPLBACKEND=Agg
  export MPLCONFIGDIR=code/.mplconfig
  /tmp/mat3300_venv/bin/python code/model_iii_similarity_over_time.py
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

MIN_ARTISTS = 5
# Decades with extremely sparse genre coverage are flagged in the summary
CORE_DECADES = list(range(1950, 2010, 10))  # 1950–2000 main narrative

# Pairs of interest (static near neighbours + a few major cross-genre links)
FOCUS_PAIRS = [
    ("Blues", "R&B;"),
    ("Blues", "Country"),
    ("Jazz", "Pop/Rock"),
    ("Electronic", "Pop/Rock"),
    ("Folk", "Country"),
    ("Classical", "New Age"),
]


def load_joined() -> pd.DataFrame:
    feat = pd.read_csv(OUT / "artists_features_with_genre.csv")
    ind = pd.read_csv(ROOT / "code" / "q1_outputs" / "artist_network_indicators.csv")
    feat["artist_id"] = feat["artist_id"].astype(str)
    ind["artist_id"] = ind["artist_id"].astype(str)
    df = feat.merge(
        ind[["artist_id", "katz_out", "start"]],
        on="artist_id",
        how="inner",
    )
    df["decade"] = (df["start"] // 10) * 10
    df["weight_katz"] = df["katz_out"].clip(lower=1e-12)
    return df


def resolve_genre(name: str, available: set[str]) -> str | None:
    if name in available:
        return name
    for g in available:
        if g.replace(";", "") == name.replace(";", ""):
            return g
    return None


def weighted_centroids(
    X: np.ndarray,
    genres: np.ndarray,
    weights: np.ndarray,
    min_artists: int,
) -> pd.DataFrame:
    rows = {}
    for g in sorted(pd.unique(genres)):
        mask = genres == g
        if int(mask.sum()) < min_artists:
            continue
        w = weights[mask]
        rows[g] = (w[:, None] * X[mask]).sum(axis=0) / w.sum()
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS)
    return pd.DataFrame(rows, index=FEATURE_COLS).T  # genre x feature


def pairwise_distances(C: pd.DataFrame) -> pd.DataFrame:
    labels = list(C.index)
    M = C.to_numpy(dtype=float)
    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(M[i] - M[j]))
            D[i, j] = D[j, i] = d
    return pd.DataFrame(D, index=labels, columns=labels)


def offdiag_mean(D: pd.DataFrame) -> float:
    M = D.to_numpy()
    if M.size == 0 or M.shape[0] < 2:
        return float("nan")
    mask = ~np.eye(M.shape[0], dtype=bool)
    return float(M[mask].mean())


def main() -> None:
    df = load_joined()
    scaler = StandardScaler().fit(df[FEATURE_COLS].to_numpy(dtype=float))
    X_all = scaler.transform(df[FEATURE_COLS].to_numpy(dtype=float))
    df = df.copy()
    for i, col in enumerate(FEATURE_COLS):
        df[f"z_{col}"] = X_all[:, i]
    Z_COLS = [f"z_{c}" for c in FEATURE_COLS]

    # Static full-sample c_g (same definition as model_iii_genre_similarity)
    C_static = weighted_centroids(
        df[Z_COLS].to_numpy(),
        df["genre"].to_numpy(),
        df["weight_katz"].to_numpy(),
        min_artists=1,
    )
    C_static.columns = FEATURE_COLS

    decades = sorted(df["decade"].unique())
    decade_rows = []
    pair_rows = []
    drift_rows = []
    matrices: dict[int, pd.DataFrame] = {}

    for dec in decades:
        sub = df[df["decade"] == dec]
        C_t = weighted_centroids(
            sub[Z_COLS].to_numpy(),
            sub["genre"].to_numpy(),
            sub["weight_katz"].to_numpy(),
            min_artists=MIN_ARTISTS,
        )
        if C_t.empty or len(C_t) < 2:
            decade_rows.append(
                {
                    "decade": int(dec),
                    "n_artists": int(len(sub)),
                    "n_genres_used": int(len(C_t)),
                    "mean_between_distance": np.nan,
                    "mean_between_similarity": np.nan,
                }
            )
            continue
        C_t.columns = FEATURE_COLS
        D = pairwise_distances(C_t)
        matrices[int(dec)] = D
        mean_d = offdiag_mean(D)
        decade_rows.append(
            {
                "decade": int(dec),
                "n_artists": int(len(sub)),
                "n_genres_used": int(len(C_t)),
                "genres_used": "|".join(C_t.index.astype(str)),
                "mean_between_distance": mean_d,
                "mean_between_similarity": float(1.0 / (1.0 + mean_d)),
            }
        )

        # Self-drift vs static c_g
        for g in C_t.index:
            if g in C_static.index:
                drift = float(np.linalg.norm(C_t.loc[g].to_numpy() - C_static.loc[g].to_numpy()))
                drift_rows.append(
                    {
                        "decade": int(dec),
                        "genre": g,
                        "n_artists_in_decade": int((sub["genre"] == g).sum()),
                        "drift_vs_static_cg": drift,
                    }
                )

        # Focus pairs present in this decade
        avail = set(C_t.index.astype(str))
        for a0, b0 in FOCUS_PAIRS:
            a = resolve_genre(a0, avail)
            b = resolve_genre(b0, avail)
            if a is None or b is None:
                continue
            d = float(D.loc[a, b])
            pair_rows.append(
                {
                    "decade": int(dec),
                    "genre_a": a,
                    "genre_b": b,
                    "pair": f"{a} — {b}",
                    "distance": d,
                    "similarity": float(1.0 / (1.0 + d)),
                }
            )

        D.to_csv(OUT / f"genre_centroid_distances_katz_{dec}.csv")

    decade_df = pd.DataFrame(decade_rows)
    pair_df = pd.DataFrame(pair_rows)
    drift_df = pd.DataFrame(drift_rows)

    decade_df.to_csv(OUT / "genre_similarity_over_time_summary.csv", index=False)
    pair_df.to_csv(OUT / "genre_pair_similarity_over_time.csv", index=False)
    drift_df.to_csv(OUT / "genre_self_drift_over_time.csv", index=False)

    # --- Figures ---
    # 1) Mean between-genre distance over decades
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    plot_df = decade_df.dropna(subset=["mean_between_distance"])
    ax.plot(plot_df["decade"], plot_df["mean_between_distance"], "o-", color="#1b4965", lw=2)
    ax.set_xlabel("Artist active-start decade")
    ax.set_ylabel("Mean between-genre centroid distance")
    ax.set_title(r"Genre–genre sonic separation over time (Katz-weighted $c_g(t)$)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for folder in (OUT, FIG):
        fig.savefig(folder / "genre_mean_between_distance_over_time.png", dpi=200, bbox_inches="tight")
        fig.savefig(folder / "genre_mean_between_distance_over_time.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved mean-between-distance time series")

    # 2) Focus-pair similarity trajectories
    if not pair_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for pair_name, sub in pair_df.groupby("pair"):
            sub = sub.sort_values("decade")
            ax.plot(sub["decade"], sub["similarity"], "o-", label=pair_name, lw=1.8)
        ax.set_xlabel("Artist active-start decade")
        ax.set_ylabel(r"Similarity $s=1/(1+d)$")
        ax.set_title("Selected genre-pair similarity over time")
        ax.legend(fontsize=7, loc="best", frameon=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for folder in (OUT, FIG):
            fig.savefig(folder / "genre_pair_similarity_over_time.png", dpi=200, bbox_inches="tight")
            fig.savefig(folder / "genre_pair_similarity_over_time.pdf", bbox_inches="tight")
        plt.close(fig)
        print("saved pair similarity time series")

    # 3) Self-drift heatmap (genre x decade)
    if not drift_df.empty:
        pivot = drift_df.pivot(index="genre", columns="decade", values="drift_vs_static_cg")
        # order genres by mean drift
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(9.5, 7.0))
        M = pivot.to_numpy(dtype=float)
        im = ax.imshow(M, aspect="auto", cmap="YlOrRd")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(r"$\|c_g(t)-c_g\|$ (static full-sample)")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
        ax.set_title("Genre self-drift over time vs static Katz-weighted profile")
        ax.set_xlabel("Active-start decade")
        fig.tight_layout()
        for folder in (OUT, FIG):
            fig.savefig(folder / "genre_self_drift_over_time.png", dpi=200, bbox_inches="tight")
            fig.savefig(folder / "genre_self_drift_over_time.pdf", bbox_inches="tight")
        plt.close(fig)
        print("saved self-drift heatmap")

    # Narrative helpers for core decades
    core = decade_df[decade_df["decade"].isin(CORE_DECADES)].dropna(subset=["mean_between_distance"])
    pair_trends = []
    for pair_name, sub in pair_df.groupby("pair"):
        sub = sub.sort_values("decade")
        if len(sub) >= 3:
            # simple endpoint comparison on overlapping core decades
            sub_core = sub[sub["decade"].isin(CORE_DECADES)]
            if len(sub_core) >= 2:
                first, last = sub_core.iloc[0], sub_core.iloc[-1]
                pair_trends.append(
                    {
                        "pair": pair_name,
                        "first_decade": int(first["decade"]),
                        "last_decade": int(last["decade"]),
                        "similarity_first": float(first["similarity"]),
                        "similarity_last": float(last["similarity"]),
                        "delta_similarity": float(last["similarity"] - first["similarity"]),
                    }
                )
    trends_df = pd.DataFrame(pair_trends).sort_values("delta_similarity")
    if not trends_df.empty:
        trends_df.to_csv(OUT / "genre_pair_similarity_trends.csv", index=False)

    summary = {
        "definition": (
            "c_g(t): Katz-weighted mean of globally z-scored features for artists "
            f"with active_start in decade t; genres need >= {MIN_ARTISTS} artists."
        ),
        "min_artists_per_genre_decade": MIN_ARTISTS,
        "decades": [int(d) for d in decades],
        "mean_between_distance_by_decade": decade_df[
            ["decade", "n_artists", "n_genres_used", "mean_between_distance", "mean_between_similarity"]
        ].to_dict(orient="records"),
        "core_decades_note": "Main interpretation: 1950–2000; 1930s/2010s are sparse.",
        "pair_similarity_trends": trends_df.to_dict(orient="records") if not trends_df.empty else [],
        "largest_mean_self_drift_genres": (
            drift_df.groupby("genre")["drift_vs_static_cg"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .round(3)
            .to_dict()
            if not drift_df.empty
            else {}
        ),
    }
    with open(OUT / "model_iii_similarity_over_time_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Similarity over time summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
