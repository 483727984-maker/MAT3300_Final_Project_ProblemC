"""Regenerate fig4 with an improved left panel.

Left panel: radial layout --- the 8 selected Jazz hubs sit on an inner
circle, their selected followers radiate outward, and every hub is labelled.
Right panel: unchanged within-genre Katz bar chart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigs, spsolve

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ProblemC_Music_data" / "influence_data.csv"
OUT = ROOT / "q1_outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def repair_name(value: str) -> str:
    if "Ã" in value or "Â" in value:
        try:
            return value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
    return value


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype={"influencer_id": "string", "follower_id": "string"}, encoding="utf-8")
    for col in ["influencer_id", "follower_id"]:
        df[col] = df[col].str.strip()
    for col in ["influencer_name", "follower_name"]:
        df[col] = df[col].astype("string").str.strip().map(repair_name)
    for col in ["influencer_main_genre", "follower_main_genre"]:
        df[col] = df[col].astype("string").str.strip().str.replace(";", "", regex=False)
    for col in ["influencer_active_start", "follower_active_start"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["influencer_id", "follower_id"]).copy()
    df = df.drop_duplicates(["influencer_id", "follower_id"]).reset_index(drop=True)
    return df


def make_adjacency(df: pd.DataFrame, all_ids: list[str]) -> sparse.csr_matrix:
    index = {a: i for i, a in enumerate(all_ids)}
    row = df["influencer_id"].map(index).to_numpy()
    col = df["follower_id"].map(index).to_numpy()
    return sparse.csr_matrix((np.ones(len(df)), (row, col)), shape=(len(all_ids), len(all_ids)))


def outgoing_katz(A: sparse.csr_matrix, alpha: float) -> np.ndarray:
    try:
        rho = float(abs(eigs(A, k=1, which="LM", return_eigenvectors=False)[0]))
    except ArpackNoConvergence:
        g = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
        rho = 0.0 if nx.is_directed_acyclic_graph(g) else float(
            abs(eigs(A, k=1, which="LM", tol=1e-6, maxiter=50000, return_eigenvectors=False)[0]))
    bound = float("inf") if rho < 1e-12 else 1 / rho
    if alpha <= 0 or alpha >= bound:
        raise ValueError(f"alpha must be in (0, {bound:.6g})")
    return spsolve(sparse.eye(A.shape[0], format="csc") - alpha * A.tocsc(), np.ones(A.shape[0]))


raw = load_data()
name = (pd.concat([
    raw[["influencer_id", "influencer_name"]].rename(columns={"influencer_id": "artist_id", "influencer_name": "name"}),
    raw[["follower_id", "follower_name"]].rename(columns={"follower_id": "artist_id", "follower_name": "name"}),
]).dropna().drop_duplicates("artist_id").set_index("artist_id")["name"].to_dict())
start = (pd.concat([
    raw[["influencer_id", "influencer_active_start"]].rename(columns={"influencer_id": "artist_id", "influencer_active_start": "start"}),
    raw[["follower_id", "follower_active_start"]].rename(columns={"follower_id": "artist_id", "follower_active_start": "start"}),
]).dropna().drop_duplicates("artist_id").set_index("artist_id")["start"].to_dict())

jazz = raw.loc[(raw["influencer_main_genre"] == "Jazz") & (raw["follower_main_genre"] == "Jazz")]
jazz_ids = sorted(set(jazz["influencer_id"]) | set(jazz["follower_id"]))
xj = outgoing_katz(make_adjacency(jazz, jazz_ids), ALPHA)
Gj = nx.DiGraph()
Gj.add_nodes_from(jazz_ids)
Gj.add_edges_from(zip(jazz["influencer_id"], jazz["follower_id"]))
jazz_index = {a: i for i, a in enumerate(jazz_ids)}
jazz_top = np.argsort(-xj)[:15]

# ---- selected ego view: top-8 hubs, up to 3 high-Katz direct followers each
core = [jazz_ids[i] for i in jazz_top[:8]]
selected_edges = []
for artist in core:
    targets = sorted(Gj.successors(artist), key=lambda v: xj[jazz_index[v]], reverse=True)[:3]
    selected_edges.extend((artist, t) for t in targets)

# ---- radial layout: hubs on inner circle, followers on outer rays
R_HUB, R_FOLLOW = 1.0, 1.72
hub_angle = {a: 2 * np.pi * k / len(core) + np.pi / 8 for k, a in enumerate(core)}
pos: dict[str, tuple[float, float]] = {a: (R_HUB * np.cos(t), R_HUB * np.sin(t)) for a, t in hub_angle.items()}
placed_followers: set[str] = set()
for hub in core:  # deterministic iteration order keeps layout stable
    targets = [t for u, t in selected_edges if u == hub]
    for m, t in enumerate(targets):
        if t in pos:
            continue
        spread = (m - (len(targets) - 1) / 2) * 0.30
        ang = hub_angle[hub] + spread * (R_HUB / R_FOLLOW)
        pos[t] = (R_FOLLOW * np.cos(ang), R_FOLLOW * np.sin(ang))
        placed_followers.add(t)

x_max = max(xj.max(), 1)
size = {v: 140 + 1100 * xj[jazz_index[v]] / x_max for v in pos}
followers_all = [v for v in pos if v not in core]
view_decades = [int(start[v]) for v in pos if v in start and pd.notna(start[v])]
dmin, dmax = min(view_decades), max(view_decades)
norm = plt.Normalize(dmin, dmax if dmax > dmin else dmin + 10)
color = {v: plt.cm.viridis(norm(int(start[v]))) for v in pos if v in start and pd.notna(start[v])}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
# edges (hub -> follower)
for u, v in selected_edges:
    pu, pv = np.array(pos[u]), np.array(pos[v])
    # shrink arrow endpoints toward node centres so arrows do not cover markers
    du = (pv - pu) / np.linalg.norm(pv - pu)
    pu2, pv2 = pu + du * 0.09, pv - du * 0.13
    ax.annotate("", xy=tuple(pv2), xytext=tuple(pu2),
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#53616d", "alpha": 0.55,
                            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 11})
# follower nodes
ax.scatter([pos[v][0] for v in followers_all], [pos[v][1] for v in followers_all],
           s=[size[v] for v in followers_all],
           c=[color.get(v, "#999") for v in followers_all],
           edgecolors="white", linewidths=0.6, zorder=3)
# hub nodes (diamonds)
ax.scatter([pos[v][0] for v in core], [pos[v][1] for v in core],
           s=[size[v] for v in core], marker="D",
           c=[color[v] for v in core],
           edgecolors="#222222", linewidths=1.1, zorder=4)
# label every hub
for v in core:
    pv = np.array(pos[v])
    off = pv / np.linalg.norm(pv) * 30
    ax.annotate(name.get(v, v), pos[v], xytext=(off[0], off[1]), textcoords="offset points",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
                zorder=5)
# label the six highest-Katz followers
top_followers = sorted(followers_all, key=lambda v: xj[jazz_index[v]], reverse=True)[:6]
for v in top_followers:
    pv = np.array(pos[v])
    off = pv / np.linalg.norm(pv) * 13
    ax.annotate(name.get(v, v), pos[v], xytext=(off[0], off[1]), textcoords="offset points",
                ha="center", va="center", fontsize=6.8, color="#333333", zorder=5)
ax.set_title("Selected Jazz influence ties\n(top-8 hubs on inner circle; arrows point to selected followers)")
ax.set_xlim(-2.55, 2.55); ax.set_ylim(-2.15, 2.15)
ax.set_aspect("equal"); ax.axis("off")
ax.scatter([], [], marker="D", s=50, c="#777", edgecolors="#222", label="Top-8 Jazz hubs (Katz)")
ax.scatter([], [], marker="o", s=40, c="#999", edgecolors="white", label="Selected direct followers")
ax.legend(loc="lower left", frameon=False, fontsize=7.5)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="Active-start decade")

# ---- right panel: within-genre Katz bar chart (unchanged logic)
outj = np.asarray(make_adjacency(jazz, jazz_ids).sum(axis=1)).ravel()
topj = jazz_top[:12][::-1]
axes[1].barh(range(len(topj)), xj[topj], color="#bd4b55")
axes[1].set_yticks(range(len(topj)), [name.get(jazz_ids[i], jazz_ids[i]) for i in topj])
axes[1].set_xlabel("Katz score within Jazz-induced subnetwork")
axes[1].set_title("Top Jazz artists by within-genre Katz")
for y, i in enumerate(topj):
    axes[1].text(xj[i] + max(xj[topj]) * 0.015, y, f"d={int(outj[i])}", va="center", fontsize=8)

fig.tight_layout()
fig.savefig(FIG / "fig4_jazz_subnetwork.png")
plt.close(fig)
print("fig4 regenerated:", FIG / "fig4_jazz_subnetwork.png")
