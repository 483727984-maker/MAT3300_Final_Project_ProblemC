"""Question 1: build and analyze the reported musical-influence network.

Primary analysis retains every valid reported edge. Active-start decades are
descriptive attributes; the 733 time-order reversals are assessed separately
as a sensitivity check, not removed from the main graph.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Select a non-interactive backend before pyplot import.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigs, spsolve
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "ProblemC_Music_data" / "influence_data.csv",
    ROOT.parent / "ProblemC_Music_data" / "influence_data.csv",
]
DATA = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA is None:
    raise FileNotFoundError(
        "Cannot find influence_data.csv. Expected it in ProblemC_Music_data/ "
        "inside this project folder (or its parent)."
    )

OUT = ROOT / "q1_outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10
TOP_N = 15

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def repair_name(value: str) -> str:
    """Repair common UTF-8/Latin-1 mojibake without changing valid names."""
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
    # The data represent a binary reported relation; duplicate pairs, if any,
    # are represented once rather than interpreted as repeated strength.
    df = df.drop_duplicates(["influencer_id", "follower_id"]).reset_index(drop=True)
    return df


def artist_metadata(df: pd.DataFrame) -> pd.DataFrame:
    left = df[["influencer_id", "influencer_name", "influencer_main_genre", "influencer_active_start"]].rename(
        columns={"influencer_id": "artist_id", "influencer_name": "name", "influencer_main_genre": "genre", "influencer_active_start": "start"}
    )
    right = df[["follower_id", "follower_name", "follower_main_genre", "follower_active_start"]].rename(
        columns={"follower_id": "artist_id", "follower_name": "name", "follower_main_genre": "genre", "follower_active_start": "start"}
    )
    records = pd.concat([left, right], ignore_index=True).sort_values("artist_id")
    conflicts = records.groupby("artist_id")[["name", "genre", "start"]].nunique(dropna=True).gt(1).sum()
    print("Metadata conflicts by field (artist IDs with multiple values):")
    print(conflicts.to_string())
    return records.groupby("artist_id", as_index=False).agg(
        name=("name", "first"), genre=("genre", "first"), start=("start", "first")
    )


def make_adjacency(df: pd.DataFrame, all_ids: list[str]) -> sparse.csr_matrix:
    index = {artist_id: i for i, artist_id in enumerate(all_ids)}
    row = df["influencer_id"].map(index).to_numpy()
    col = df["follower_id"].map(index).to_numpy()
    return sparse.csr_matrix((np.ones(len(df)), (row, col)), shape=(len(all_ids), len(all_ids)))


def outgoing_katz(A: sparse.csr_matrix, alpha: float) -> tuple[np.ndarray, float]:
    """Katz over paths leaving each influencer: x=(I-alpha*A)^(-1) 1."""
    try:
        rho = float(abs(eigs(A, k=1, which="LM", return_eigenvectors=False)[0]))
    except ArpackNoConvergence:
        # Cohort graphs can be DAGs, whose spectral radius is exactly zero;
        # ARPACK may not return an eigenvector for this all-zero spectrum.
        graph = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
        if nx.is_directed_acyclic_graph(graph):
            rho = 0.0
        else:
            rho = float(abs(eigs(A, k=1, which="LM", tol=1e-6, maxiter=50000,
                                 return_eigenvectors=False)[0]))
    # For a directed acyclic graph rho can be zero; then A is nilpotent and
    # the Katz series is finite for every positive alpha.
    bound = float("inf") if rho < 1e-12 else 1 / rho
    if alpha <= 0 or alpha >= bound:
        raise ValueError(f"Katz alpha must be in (0, {bound:.6g}); got {alpha}.")
    score = spsolve(sparse.eye(A.shape[0], format="csc") - alpha * A.tocsc(), np.ones(A.shape[0]))
    return np.asarray(score), rho


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.0); ax.axis("off")
    boxes = [
        (0.20, "Reported influence\npairs"),
        (2.65, "Directed graph\nall edges; weight = 1"),
        (5.10, "Measures\nout-degree\nKatz\nreverse PageRank"),
        (7.55, "Analysis\nfull network\ndecade cohorts\nJazz subnetwork"),
    ]
    for x, label in boxes:
        ax.text(x + 0.98, 1.72, label, ha="center", va="center", fontsize=8.2,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "#edf2f7", "edgecolor": "#56708f", "linewidth": 1.1})
    for x in [2.17, 4.62, 7.07]:
        ax.annotate("", xy=(x + 0.36, 1.72), xytext=(x, 1.72),
                    arrowprops={"arrowstyle": "->", "color": "#52616b", "lw": 1.3})
    ax.text(5, 0.28, "Active-start decade is descriptive; time-order filtering is a sensitivity check only.",
            ha="center", va="center", fontsize=8.5, color="#454545")
    fig.savefig(FIG / "fig0_analysis_workflow.png")
    plt.close(fig)


def main() -> None:
    raw = load_data()
    metadata = artist_metadata(raw)
    ids = sorted(metadata["artist_id"].tolist())
    idx = {artist_id: i for i, artist_id in enumerate(ids)}
    name = metadata.set_index("artist_id")["name"].to_dict()
    genre = metadata.set_index("artist_id")["genre"].to_dict()
    start = metadata.set_index("artist_id")["start"].to_dict()

    # Primary model: retain all observed relationships. No causal timing claim
    # is inferred from decade-level active-start fields.
    edges = raw
    A = make_adjacency(edges, ids)
    x, rho = outgoing_katz(A, ALPHA)
    katz_bound = float("inf") if rho < 1e-12 else 1 / rho
    outdeg = np.asarray(A.sum(axis=1)).ravel()
    indeg = np.asarray(A.sum(axis=0)).ravel()
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    G.add_edges_from(zip(edges["influencer_id"], edges["follower_id"]))

    # Reverse PageRank makes score flow from each reported follower back to
    # the artist(s) they name as influences.
    reverse_pr_dict = nx.pagerank(G.reverse(copy=False), alpha=0.85)
    reverse_pr = np.array([reverse_pr_dict[a] for a in ids])
    top_katz = np.argsort(-x)
    top_out = np.argsort(-outdeg)
    top_pr = np.argsort(-reverse_pr)

    # Sensitivity only: compare the all-edge model with a version that omits
    # records whose decade ordering is reversed. This filtered graph is not
    # the primary result.
    time_known = raw.dropna(subset=["influencer_active_start", "follower_active_start"])
    reversed_time = time_known["influencer_active_start"] > time_known["follower_active_start"]
    filtered_edges = raw.loc[~raw.index.isin(time_known.index[reversed_time])]
    A_filtered = make_adjacency(filtered_edges, ids)
    x_filtered, _ = outgoing_katz(A_filtered, ALPHA)
    top20_all = set(top_katz[:20])
    top20_filtered = set(np.argsort(-x_filtered)[:20])
    timing_gap = (time_known["follower_active_start"].astype(int)
                  - time_known["influencer_active_start"].astype(int))

    # Summary table and full-network indicator table.
    indicators = metadata.copy()
    indicators["out_degree"] = outdeg
    indicators["in_degree"] = indeg
    indicators["katz_out"] = x
    indicators["pagerank_reverse"] = reverse_pr
    indicators["rank_katz"] = pd.Series(np.argsort(np.argsort(-x)) + 1)
    indicators["rank_out_degree"] = pd.Series(np.argsort(np.argsort(-outdeg)) + 1)
    indicators["rank_reverse_pagerank"] = pd.Series(np.argsort(np.argsort(-reverse_pr)) + 1)
    indicators.to_csv(OUT / "artist_network_indicators.csv", index=False)

    data_summary = {
        "raw_records": int(len(pd.read_csv(DATA))),
        "unique_directed_edges": int(len(raw)),
        "artists": int(len(ids)),
        "weak_components": int(nx.number_weakly_connected_components(G)),
        "strong_components": int(nx.number_strongly_connected_components(G)),
        "largest_weak_component": int(max(map(len, nx.weakly_connected_components(G)))),
        "largest_strong_component": int(max(map(len, nx.strongly_connected_components(G)))),
        "out_degree_zero": int((outdeg == 0).sum()),
        "in_degree_zero": int((indeg == 0).sum()),
        "time_reversed_edges_retained": int(reversed_time.sum()),
        "time_reversed_share": float(reversed_time.mean()),
        "katz_alpha": ALPHA,
        "spectral_radius": rho,
        "katz_alpha_upper_bound": katz_bound if np.isfinite(katz_bound) else None,
        "katz_vs_outdegree_spearman": rank_corr(x, outdeg),
        "katz_vs_reverse_pagerank_spearman": rank_corr(x, reverse_pr),
        "top20_overlap_all_vs_filtered": len(top20_all & top20_filtered),
        "all_vs_filtered_spearman": rank_corr(x, x_filtered),
        "start_decade_gap_counts": {
            "influencer_earlier": int((timing_gap > 0).sum()),
            "same_decade": int((timing_gap == 0).sum()),
            "influencer_later": int((timing_gap < 0).sum()),
        },
    }
    (OUT / "q1_results.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    # Figure 1: overall rankings and what Katz adds beyond direct degree.
    colors = {"Pop/Rock": "#4776a8", "Blues": "#d17a43", "Jazz": "#bd4b55",
              "Country": "#5b9b6d", "R&B": "#8773ad", "Folk": "#9a8066"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.1, 1.0]})
    top = top_katz[:TOP_N][::-1]
    axes[0].barh(range(len(top)), x[top], color=[colors.get(genre[ids[i]], "#91a0ae") for i in top])
    axes[0].set_yticks(range(len(top)), [name[ids[i]] for i in top])
    axes[0].set_xlabel("Outgoing Katz score (α = 0.10)")
    axes[0].set_title("Top artists by multi-step influence")
    for y, i in enumerate(top):
        axes[0].text(x[i] + max(x[top]) * .012, y, f"out-degree {int(outdeg[i])}", va="center", fontsize=8)
    axes[0].set_xlim(0, max(x[top]) * 1.42)
    axes[1].scatter(outdeg + 1, x, c=[colors.get(genre[a], "#aab3bb") for a in ids], s=18, alpha=.55, edgecolors="none")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Direct followers (out-degree + 1)")
    axes[1].set_ylabel("Outgoing Katz score")
    axes[1].set_title(f"Direct vs multi-step influence (ρ = {data_summary['katz_vs_outdegree_spearman']:.3f})")
    label_offsets = {"Chuck Berry": (-38, -24), "The Beatles": (9, 5),
                     "Muddy Waters": (9, 17), "Bob Dylan": (10, -20)}
    for artist, offset in label_offsets.items():
        matches = [i for i, a in enumerate(ids) if name[a] == artist]
        if matches:
            i = matches[0]
            axes[1].annotate(artist, (outdeg[i] + 1, x[i]), xytext=offset,
                             textcoords="offset points", fontsize=8,
                             arrowprops={"arrowstyle": "-", "lw": .6, "color": "#555"})
    axes[1].grid(alpha=.2, which="both")
    fig.tight_layout(); fig.savefig(FIG / "fig1_global_influence.png"); plt.close(fig)

    # Figure 2: alpha stability and sensitivity to the optional time filter.
    alphas = [0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
    alpha_corr, alpha_top20 = [], []
    for a in alphas:
        xa, _ = outgoing_katz(A, a)
        alpha_corr.append(rank_corr(x, xa))
        alpha_top20.append(len(set(np.argsort(-x)[:20]) & set(np.argsort(-xa)[:20])))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(alphas, alpha_corr, "o-", color="#4776a8")
    axes[0].axvline(ALPHA, color="#777", linestyle=":")
    axes[0].set(xlabel="Katz attenuation α", ylabel="Spearman correlation with α=0.10", ylim=(.9, 1.01), title="Parameter sensitivity")
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
    axes[1].set(xscale="log", yscale="log", xlabel="Katz score after optional time filtering", ylabel="Katz score with all edges", title=f"Time-filter sensitivity (ρ={data_summary['all_vs_filtered_spearman']:.3f}; top-20 overlap={data_summary['top20_overlap_all_vs_filtered']}/20)")
    axes[1].grid(alpha=.2, which="both")
    fig.tight_layout(); fig.savefig(FIG / "fig2_sensitivity.png"); plt.close(fig)

    # Follower-start-decade cohorts: descriptive groups of reported edges,
    # not claims that influence occurred in the start decade.
    cohort_decades = sorted(int(v) for v in raw["follower_active_start"].dropna().unique())
    cohort_top: dict[str, dict[int, int]] = collections.defaultdict(dict)
    cohort_rows = []
    for decade in cohort_decades:
        cohort = raw.loc[raw["follower_active_start"] == decade]
        cohort_ids = sorted(set(cohort["influencer_id"]) | set(cohort["follower_id"]))
        Ac = make_adjacency(cohort, cohort_ids)
        xc, _ = outgoing_katz(Ac, ALPHA)
        order = np.argsort(-xc)
        for rank, j in enumerate(order[:10], 1):
            artist = cohort_ids[j]
            cohort_top[artist][decade] = rank
        cohort_rows.append({"decade": decade, "edges": len(cohort), "artists": len(cohort_ids),
                            "top5": [name.get(cohort_ids[j], cohort_ids[j]) for j in order[:5]]})
    persistence = collections.Counter({artist: len(ranks) for artist, ranks in cohort_top.items()})
    heat_artists = [a for a, _ in persistence.most_common(15)]
    mat = np.full((len(heat_artists), len(cohort_decades)), np.nan)
    for i, artist in enumerate(heat_artists):
        for j, decade in enumerate(cohort_decades):
            if decade in cohort_top[artist]:
                mat[i, j] = cohort_top[artist][decade]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    cmap = plt.cm.Blues_r.copy(); cmap.set_bad("white")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=1, vmax=10)
    ax.set_xticks(range(len(cohort_decades)), [f"{d}s" for d in cohort_decades])
    ax.set_yticks(range(len(heat_artists)), [name.get(a, a) for a in heat_artists])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] <= 5 else "black")
    ax.set_title("Artists repeatedly ranked in the top 10 by follower-start-decade cohort")
    ax.set_xlabel("Follower active-start decade (cohort label)")
    ax.set_ylabel("Artist")
    fig.colorbar(im, ax=ax, label="Katz rank within cohort (1 = highest)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_cohort_heatmap.png"); plt.close(fig)
    pd.DataFrame(cohort_rows).to_json(OUT / "cohort_results.json", orient="records", indent=2)

    # Jazz-induced subnetwork statistics use all Jazz -> Jazz edges.
    jazz = raw.loc[(raw["influencer_main_genre"] == "Jazz") & (raw["follower_main_genre"] == "Jazz")]
    jazz_ids = sorted(set(jazz["influencer_id"]) | set(jazz["follower_id"]))
    AJ = make_adjacency(jazz, jazz_ids)
    xj, _ = outgoing_katz(AJ, ALPHA)
    outj = np.asarray(AJ.sum(axis=1)).ravel()
    Gj = nx.DiGraph(); Gj.add_nodes_from(jazz_ids); Gj.add_edges_from(zip(jazz["influencer_id"], jazz["follower_id"]))
    jazz_meta = metadata.set_index("artist_id")
    jazz_stats = {
        "artists": len(jazz_ids), "edges": len(jazz),
        "density": nx.density(Gj),
        "weak_components": nx.number_weakly_connected_components(Gj),
        "strong_components": nx.number_strongly_connected_components(Gj),
        "reciprocal_pairs": sum(1 for u, v in Gj.edges if Gj.has_edge(v, u)) // 2,
        "jazz_to_pop_rock_edges": int(((raw["influencer_main_genre"] == "Jazz") & (raw["follower_main_genre"] == "Pop/Rock")).sum()),
    }
    jazz_index = {a: i for i, a in enumerate(jazz_ids)}
    jazz_top = np.argsort(-xj)[:15]
    jazz_top_rows = [{"rank": r + 1, "artist": name.get(jazz_ids[i], jazz_ids[i]),
                      "out_degree_within_jazz": int(outj[i]), "katz_within_jazz": float(xj[i])}
                     for r, i in enumerate(jazz_top)]

    # Focused directed view: top Jazz hubs and up to three high-Katz direct
    # Jazz followers each. The chart is explicitly a selected ego subnetwork.
    selected_edges = []
    core = [jazz_ids[i] for i in jazz_top[:8]]
    for artist in core:
        targets = sorted(Gj.successors(artist), key=lambda v: xj[jazz_index[v]], reverse=True)[:3]
        selected_edges.extend((artist, target) for target in targets)
    view = nx.DiGraph(); view.add_edges_from(selected_edges)
    pos = nx.spring_layout(view, seed=24, k=.85, iterations=120)
    decades = sorted({int(v) for v in jazz_meta.loc[list(view.nodes()), "start"].dropna()})
    decade_norm = plt.Normalize(min(decades), max(decades) if len(decades) > 1 else min(decades) + 10)
    node_colors = [plt.cm.viridis(decade_norm(int(jazz_meta.loc[v, "start"]))) for v in view.nodes()]
    node_sizes = [120 + 900 * xj[jazz_index[v]] / max(xj.max(), 1) for v in view.nodes()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    nx.draw_networkx_edges(view, pos, ax=axes[0], arrows=True, arrowsize=12, width=1.0, alpha=.5, edge_color="#53616d")
    core_present = [v for v in core if v in view]
    followers_present = [v for v in view if v not in core]
    nx.draw_networkx_nodes(view, pos, nodelist=followers_present, ax=axes[0],
                           node_color=[plt.cm.viridis(decade_norm(int(jazz_meta.loc[v, "start"]))) for v in followers_present],
                           node_size=[node_sizes[list(view.nodes()).index(v)] for v in followers_present],
                           edgecolors="white", linewidths=.5, node_shape="o")
    nx.draw_networkx_nodes(view, pos, nodelist=core_present, ax=axes[0],
                           node_color=[plt.cm.viridis(decade_norm(int(jazz_meta.loc[v, "start"]))) for v in core_present],
                           node_size=[node_sizes[list(view.nodes()).index(v)] for v in core_present],
                           edgecolors="#222222", linewidths=1.0, node_shape="D")
    label_offsets = {core[0]: (14, -22), core[1]: (-48, 20)}
    for artist, offset in label_offsets.items():
        if artist in view:
            axes[0].annotate(name.get(artist, artist), pos[artist], xytext=offset,
                             textcoords="offset points", fontsize=7.5,
                             bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": .82},
                             arrowprops={"arrowstyle": "-", "lw": .5, "color": "#555"})
    axes[0].set_title("Selected Jazz influence ties\n(top hubs + up to 3 direct followers each)")
    axes[0].axis("off")
    axes[0].scatter([], [], marker="D", s=50, c="#777", edgecolors="#222", label="Selected top-8 hubs")
    axes[0].scatter([], [], marker="o", s=40, c="#777", edgecolors="white", label="Selected followers")
    axes[0].legend(loc="lower left", frameon=False, fontsize=7.5)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=decade_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], fraction=.035, pad=.02, label="Active-start decade")
    topj = jazz_top[:12][::-1]
    axes[1].barh(range(len(topj)), xj[topj], color="#bd4b55")
    axes[1].set_yticks(range(len(topj)), [name.get(jazz_ids[i], jazz_ids[i]) for i in topj])
    axes[1].set_xlabel("Katz score within Jazz-induced subnetwork")
    axes[1].set_title("Top Jazz artists by within-genre Katz")
    for y, i in enumerate(topj):
        axes[1].text(xj[i] + max(xj[topj]) * .015, y, f"d={int(outj[i])}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_jazz_subnetwork.png"); plt.close(fig)
    (OUT / "jazz_results.json").write_text(json.dumps({"summary": jazz_stats, "top": jazz_top_rows}, indent=2), encoding="utf-8")

    draw_workflow()
    print(json.dumps(data_summary, indent=2))
    print(f"Outputs saved under: {OUT}")
    print("Jazz subnetwork:", json.dumps(jazz_stats))


if __name__ == "__main__":
    main()
