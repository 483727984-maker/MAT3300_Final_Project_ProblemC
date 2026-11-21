"""
Model III: what distinguishes a genre — read off the SAME Katz-weighted
genre vectors c_g used for genre–genre similarity.

In the globally z-scored space, each coordinate of c_g is already a signed
deviation from the all-artist mean. We therefore:

  1. Profile each genre by its top-high / top-low features on c_g.
  2. Rank features by how much they vary across genre centroids
     (discriminating power shared across many genres).
  3. For the closest genre pairs, decompose distance into per-feature |Δ|
     to show what still separates near neighbours.

Usage (from repo root; run after model_iii_genre_similarity.py):
  export MPLBACKEND=Agg
  export MPLCONFIGDIR=code/.mplconfig
  /tmp/mat3300_venv/bin/python code/model_iii_genre_distinction.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code" / "outputs"
FIG = ROOT / "code" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

N_TOP = 2
N_CLOSE_PAIRS = 5


def load_centroids() -> pd.DataFrame:
    path = OUT / "genre_feature_vectors_katz.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run code/model_iii_genre_similarity.py first."
        )
    C = pd.read_csv(path, index_col=0)
    return C


def genre_profiles(C: pd.DataFrame, n_top: int = N_TOP) -> pd.DataFrame:
    rows = []
    for g, row in C.iterrows():
        order = row.sort_values(ascending=False)
        high = order.head(n_top)
        low = order.tail(n_top).sort_values()  # most negative first
        rec = {"genre": g}
        for k, (feat, val) in enumerate(high.items(), start=1):
            rec[f"high_{k}_feature"] = feat
            rec[f"high_{k}_z"] = float(val)
        for k, (feat, val) in enumerate(low.items(), start=1):
            rec[f"low_{k}_feature"] = feat
            rec[f"low_{k}_z"] = float(val)
        # L2 norm of profile = how far from global mean (distinctiveness)
        rec["profile_norm"] = float(np.linalg.norm(row.to_numpy()))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("profile_norm", ascending=False)


def feature_discriminability(C: pd.DataFrame) -> pd.DataFrame:
    """How much each feature separates genres (variance / range across c_g)."""
    rows = []
    for feat in C.columns:
        vals = C[feat].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feat,
                "across_genre_std": float(vals.std(ddof=0)),
                "across_genre_range": float(vals.max() - vals.min()),
                "max_genre": C[feat].idxmax(),
                "max_z": float(C[feat].max()),
                "min_genre": C[feat].idxmin(),
                "min_z": float(C[feat].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("across_genre_std", ascending=False)


def close_pair_feature_gaps(
    C: pd.DataFrame,
    pairs_path: Path,
    n_pairs: int = N_CLOSE_PAIRS,
    n_top: int = 3,
) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_path).head(n_pairs)
    rows = []
    for _, p in pairs.iterrows():
        a, b = p["genre_a"], p["genre_b"]
        # CSV may have R&B; from pandas display — match index
        if a not in C.index or b not in C.index:
            # try stripping trailing semicolon artifacts
            a2 = a.rstrip(";")
            b2 = b.rstrip(";")
            a = a if a in C.index else next(g for g in C.index if g.startswith(a2[:3]) and "R" in g) if "R&B" in a else a2
            b = b if b in C.index else next(g for g in C.index if g.startswith(b2[:3]) and "R" in g) if "R&B" in b else b2
        # Robust match for R&B
        def resolve(name: str) -> str:
            if name in C.index:
                return name
            for g in C.index:
                if g.replace(";", "") == name.replace(";", ""):
                    return g
            raise KeyError(name)

        a, b = resolve(str(a)), resolve(str(b))
        delta = (C.loc[a] - C.loc[b]).abs().sort_values(ascending=False)
        for k, (feat, val) in enumerate(delta.head(n_top).items(), start=1):
            signed = float(C.loc[a, feat] - C.loc[b, feat])
            rows.append(
                {
                    "genre_a": a,
                    "genre_b": b,
                    "pair_distance": float(p["distance"]),
                    "rank_gap": k,
                    "feature": feat,
                    "abs_delta_z": float(val),
                    "signed_a_minus_b": signed,
                }
            )
    return pd.DataFrame(rows)


def plot_discriminability(disc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(disc))[::-1]
    ax.barh(y, disc["across_genre_std"].to_numpy(), color="#2a6f97")
    ax.set_yticks(y)
    ax.set_yticklabels(disc["feature"].tolist())
    ax.set_xlabel("Std. of Katz-weighted genre centroids (z-space)")
    ax.set_title("Which features most distinguish genres?")
    fig.tight_layout()
    for folder in (OUT, FIG):
        png = folder / "genre_feature_discriminability.png"
        pdf = folder / "genre_feature_discriminability.pdf"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"saved {png}")
    plt.close(fig)


def plot_profile_heatmap(C: pd.DataFrame) -> None:
    """Genre × feature signed z heatmap of the same c_g used for similarity."""
    # order genres by profile norm (most distinctive first)
    norms = np.linalg.norm(C.to_numpy(), axis=1)
    order = np.argsort(-norms)
    C_ord = C.iloc[order]
    M = C_ord.to_numpy(dtype=float)
    lim = float(np.percentile(np.abs(M), 95))
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    im = ax.imshow(M, cmap="RdBu_r", aspect="auto", vmin=-lim, vmax=lim)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Katz-weighted centroid (z-score)")
    ax.set_xticks(np.arange(C.shape[1]))
    ax.set_xticklabels(C.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(C_ord)))
    ax.set_yticklabels(C_ord.index.tolist(), fontsize=8)
    ax.set_title(r"Genre profiles $c_g$ (same vectors as similarity)")
    fig.tight_layout()
    for folder in (OUT, FIG):
        fig.savefig(folder / "genre_profile_heatmap_katz.png", dpi=200, bbox_inches="tight")
        fig.savefig(folder / "genre_profile_heatmap_katz.pdf", bbox_inches="tight")
        print(f"saved {folder / 'genre_profile_heatmap_katz.png'}")
    plt.close(fig)


def main() -> None:
    C = load_centroids()
    # Normalize index quirks if any
    C.index = C.index.map(lambda s: str(s))

    profiles = genre_profiles(C)
    disc = feature_discriminability(C)
    gaps = close_pair_feature_gaps(C, OUT / "genre_pair_similarity_katz_ranked.csv")

    profiles.to_csv(OUT / "genre_distinction_profiles_katz.csv", index=False)
    disc.to_csv(OUT / "genre_feature_discriminability_katz.csv", index=False)
    gaps.to_csv(OUT / "genre_close_pair_feature_gaps_katz.csv", index=False)

    summary = {
        "definition": "Profiles read from Katz-weighted centroids c_g in global z-space (same as genre similarity).",
        "n_genres": int(len(C)),
        "n_features": int(C.shape[1]),
        "most_discriminating_features": disc.head(4)["feature"].tolist(),
        "most_distinctive_genres_by_profile_norm": profiles.head(5)[
            ["genre", "profile_norm", "high_1_feature", "low_1_feature"]
        ].to_dict(orient="records"),
        "least_distinctive_genres_by_profile_norm": profiles.tail(3)[
            ["genre", "profile_norm", "high_1_feature", "low_1_feature"]
        ].to_dict(orient="records"),
        "example_close_pair_gaps": gaps.head(6).to_dict(orient="records"),
    }
    with open(OUT / "model_iii_genre_distinction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_discriminability(disc)
    plot_profile_heatmap(C)

    print("=== Genre distinction (from Katz-weighted c_g) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPer-genre high/low features:")
    show = profiles[
        [
            "genre",
            "high_1_feature",
            "high_1_z",
            "high_2_feature",
            "high_2_z",
            "low_1_feature",
            "low_1_z",
            "low_2_feature",
            "low_2_z",
            "profile_norm",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
