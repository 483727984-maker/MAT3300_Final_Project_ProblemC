"""Regenerate fig2 with both sensitivity curves (Spearman + top-20 overlap).

Mirrors the fig2 block of q1_analysis.py after the fix; standalone so the
full pipeline does not need to be re-run just for this figure.
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ProblemC_Music_data" / "influence_data.csv"
FIG = ROOT / "q1_outputs" / "figures"
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


rank_corr = lambda a, b: float(spearmanr(a, b).statistic)

raw = load_data()
ids = sorted(set(raw["influencer_id"]) | set(raw["follower_id"]))
A = make_adjacency(raw, ids)
x = outgoing_katz(A, ALPHA)

# alpha sweep: Spearman + top-20 overlap at each attenuation level
alphas = [0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
alpha_corr, alpha_top20 = [], []
for a in alphas:
    xa = outgoing_katz(A, a)
    alpha_corr.append(rank_corr(x, xa))
    alpha_top20.append(len(set(np.argsort(-x)[:20]) & set(np.argsort(-xa)[:20])))

# time-filter sensitivity: drop edges whose follower started before the influencer
filtered = raw.loc[~(raw["follower_active_start"] < raw["influencer_active_start"])]
x_filtered = outgoing_katz(make_adjacency(filtered, ids), ALPHA)
rho_filt = rank_corr(x, x_filtered)
top20_filt = len(set(np.argsort(-x)[:20]) & set(np.argsort(-x_filtered)[:20]))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
axes[0].plot(alphas, alpha_corr, "o-", color="#4776a8")
axes[0].axvline(ALPHA, color="#777", linestyle=":")
axes[0].set(xlabel="Katz attenuation α", ylabel="Spearman correlation with α=0.10",
            ylim=(.9, 1.01), title="Parameter sensitivity")
axes[0].grid(alpha=.2)
ax2 = axes[0].twinx()
ax2.plot(alphas, alpha_top20, "s--", color="#bd4b55")
ax2.set_ylabel("Common artists in top-20 vs α=0.10", color="#bd4b55")
ax2.tick_params(axis="y", colors="#bd4b55")
ax2.set_ylim(0, 21)
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="o", ls="-", color="#4776a8", label="Spearman ρ (left axis)"),
    Line2D([], [], marker="s", ls="--", color="#bd4b55", label="Top-20 overlap (right axis)"),
]
axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5)

axes[1].scatter(x_filtered, x, alpha=.45, s=15, color="#bd4b55", edgecolors="none")
lim = max(float(x.max()), float(x_filtered.max()))
axes[1].plot([1, lim], [1, lim], linestyle="--", color="#777", linewidth=1)
axes[1].set(xscale="log", yscale="log",
            xlabel="Katz score after optional time filtering", ylabel="Katz score with all edges",
            title=f"Time-filter sensitivity (ρ={rho_filt:.3f}; top-20 overlap={top20_filt}/20)")
axes[1].grid(alpha=.2, which="both")

fig.tight_layout()
fig.savefig(FIG / "fig2_sensitivity.png")
plt.close(fig)
print("fig2 regenerated:", FIG / "fig2_sensitivity.png")
print("alpha_top20:", dict(zip(alphas, alpha_top20)))
print("filtered: rho=%.3f top20=%d/20" % (rho_filt, top20_filt))
