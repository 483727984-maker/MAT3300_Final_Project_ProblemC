"""MAT3300 Final Project, Problem C, Question 6.

How one genre evolves: the jazz influence process, 1930--2010.

The script builds a sonic timeline of jazz from the per-song Spotify features,
detects periods of rapid change, and scores every jazz artist with four
indicators of dynamic influence:

    N_i  sonic originality   Mahalanobis length of the artist's time-adjusted
                             deviation from the jazz sound of their own era;
    A_i  anticipation        cosine between that deviation and the direction in
                             which the genre's centroid moved over the next
                             ten years;
    H_i  transmission        mean cosine between an artist's deviation and the
                             deviations of the followers who name them;
    K_i  network reach       out-going Katz centrality inside the jazz
                             influence subnetwork;
    B_i  boundary reach      number of non-jazz artists naming the artist.

The transmission indicator is validated against a within-decade permutation
null.  All numbers and figures used in the report are written to
q6_results.json and figures/.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8.5,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT.parent / "ProblemC_Music_data",       # q6 run from the project folder
    ROOT.parents[1] / "ProblemC_Music_data",   # q6 run from code/q6 inside the repo
    ROOT / "ProblemC_Music_data",
]
DATA = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA is None:
    raise FileNotFoundError("Cannot find ProblemC_Music_data next to q6_analysis.py")
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RESULT = ROOT / "q6_results.json"

# liveness is excluded: on these records it mostly separates live albums from
# studio albums, not musical styles (it forms a one-cluster artefact in k-means).
FEATURES = ["danceability", "energy", "valence", "tempo", "loudness",
            "acousticness", "instrumentalness", "speechiness", "duration_ms"]
PRETTY = {
    "danceability": "danceability", "energy": "energy", "valence": "valence",
    "tempo": "tempo", "loudness": "loudness", "acousticness": "acousticness",
    "instrumentalness": "instrumentalness", "speechiness": "speechiness",
    "duration_ms": "duration (log ms)",
}
ALPHA = 0.1
MIN_SONGS_YEAR = 20
MIN_SONGS_ARTIST = 3
N_PERM = 2000
RNG = np.random.default_rng(20211101)


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def fix_mojibake(text: str) -> str:
    if isinstance(text, str) and ("Ã" in text or "Â" in text):
        try:
            return text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def clean_genre(text: str) -> str:
    return (text or "").replace(";", "").strip()


def load_influence():
    inf = pd.read_csv(DATA / "influence_data.csv", encoding="utf-8")
    inf["influencer_name"] = inf["influencer_name"].map(fix_mojibake)
    inf["follower_name"] = inf["follower_name"].map(fix_mojibake)
    inf["igenre"] = inf["influencer_main_genre"].map(clean_genre)
    inf["fgenre"] = inf["follower_main_genre"].map(clean_genre)
    inf["iy"] = inf["influencer_active_start"].astype(int)
    inf["fy"] = inf["follower_active_start"].astype(int)
    # Assumption A3 of the joint report: all 42,770 reported pairs are retained
    # in the primary network; time-order reversals are a sensitivity check, not
    # a filter.
    rows = inf.copy()
    names = {}
    for r in inf.itertuples():
        names[r.influencer_id] = r.influencer_name
        names[r.follower_id] = r.follower_name
    return inf, rows, names


def artist_start(rows, artist_id):
    hit = rows[rows["influencer_id"] == artist_id]
    if len(hit):
        return int(hit["influencer_active_start"].iloc[0])
    hit = rows[rows["follower_id"] == artist_id]
    return int(hit["follower_active_start"].iloc[0])


def load_songs(artist_ids):
    cols = ["artists_id", "year"] + FEATURES + ["liveness"]
    full = pd.read_csv(DATA / "full_music_data.csv", usecols=cols)
    ids = full["artists_id"].astype(str).str.findall(r"\d+")
    full = full.assign(artist_id=ids).explode("artist_id")
    full = full.dropna(subset=["artist_id"])
    full["artist_id"] = full["artist_id"].astype(int)
    full = full.drop(columns=["artists_id"])
    full = full[full["artist_id"].isin(artist_ids)].copy()
    full["duration_ms"] = np.log1p(full["duration_ms"])
    return full.reset_index(drop=True)


def feature_scaler():
    art = pd.read_csv(DATA / "data_by_artist.csv", encoding="utf-8")
    art = art.dropna(subset=FEATURES).copy()
    art["duration_ms"] = np.log1p(art["duration_ms"])
    return StandardScaler().fit(art[FEATURES]), art


# ----------------------------------------------------------------------
# sonic timeline
# ----------------------------------------------------------------------
def timeline_matrix(df, years, feats):
    grouped = df.groupby("year")[feats]
    mean = grouped.mean()
    smooth = mean.rolling(3, center=True, min_periods=2).mean()
    out = np.column_stack([
        np.interp(years, smooth.index.values, smooth[f].values) for f in feats
    ])
    return out


def build_timeline(songs_z, feats):
    """Yearly smoothed centroid of the genre, one row per year."""
    counts = songs_z.groupby("year").size()
    good = counts[counts >= MIN_SONGS_YEAR].index
    years = np.arange(int(good.min()), int(good.max()) + 1)
    M = timeline_matrix(songs_z, years, feats)
    return years, M, counts


def interpolate_row(years, M, t):
    return np.array([np.interp(t, years, M[:, k]) for k in range(M.shape[1])])


def pca_timeline(years, M):
    pca = PCA(n_components=4).fit(M)
    coords = pca.transform(M)
    loadings = {f: pca.components_[0, k] for k, f in enumerate(FEATURES)}
    loadings2 = {f: pca.components_[1, k] for k, f in enumerate(FEATURES)}
    return pca, coords, loadings, loadings2


def velocity_series(years, M, half=2):
    rows = []
    for k in range(half, len(years) - half):
        v = float(np.linalg.norm(M[k + half] - M[k - half]))
        rows.append({"year": int(years[k]), "velocity": v})
    return rows


def find_episodes(vel, year_counts, top=4):
    """Local maxima of the centroid speed, well separated and away from gaps."""
    dense = {int(y) for y, n in year_counts.items() if n >= 5}
    n = len(vel)
    cand = []
    for k in range(2, n - 2):
        y = vel[k]["year"]
        if not all((y + d) in dense for d in range(-2, 3)):
            continue
        left = max(vel[k - 1]["velocity"], vel[k - 2]["velocity"])
        right = max(vel[k + 1]["velocity"], vel[k + 2]["velocity"])
        if vel[k]["velocity"] >= left and vel[k]["velocity"] >= right:
            cand.append(vel[k])
    cand.sort(key=lambda r: -r["velocity"])
    chosen = []
    for r in cand:
        if all(abs(r["year"] - c["year"]) > 4 for c in chosen):
            chosen.append(r)
        if len(chosen) == top:
            break
    return sorted(chosen, key=lambda r: r["year"]), float(np.quantile(
        [r["velocity"] for r in vel], 0.90))


# ----------------------------------------------------------------------
# indicators
# ----------------------------------------------------------------------
def residuals(songs_z, artist_ids, years, M, feats):
    """Time-adjusted vector r_i = mean over songs of (z_song - centroid(year))."""
    out = {}
    counts = songs_z.groupby("artist_id").size()
    for a in artist_ids:
        sub = songs_z[songs_z["artist_id"] == a]
        if len(sub) < MIN_SONGS_ARTIST:
            continue
        diffs = np.array([
            sub.iloc[k][feats].values - interpolate_row(years, M, int(y))
            for k, y in enumerate(sub["year"].values)
        ])
        out[a] = diffs.mean(axis=0)
    return out, counts


def loo_residuals(songs_z, artist_ids, years, feats):
    out = {}
    base = songs_z
    for a in artist_ids:
        sub = base[base["artist_id"] == a]
        if len(sub) < MIN_SONGS_ARTIST:
            continue
        other = base[base["artist_id"] != a]
        M_loo = timeline_matrix(other, years, feats)
        diffs = np.array([
            sub.iloc[k][feats].values - interpolate_row(years, M_loo, int(y))
            for k, y in enumerate(sub["year"].values)
        ])
        out[a] = diffs.mean(axis=0)
    return out


def maha_norm(vec, inv_cov):
    return float(np.sqrt(vec @ inv_cov @ vec))


def katz(A, alpha=ALPHA):
    n = A.shape[0]
    return spsolve(sparse.eye(n, format="csc") - alpha * A.tocsc(), np.ones(n))


# ----------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------
def main():
    inf, rows, names = load_influence()
    jazz_ids = sorted(set(rows.loc[rows["igenre"] == "Jazz", "influencer_id"])
                      | set(rows.loc[rows["fgenre"] == "Jazz", "follower_id"]))
    start = {a: artist_start(rows, a) for a in jazz_ids}

    scaler, art_df = feature_scaler()
    songs = load_songs(set(jazz_ids))
    songs_z = songs.copy()
    songs_z[FEATURES] = scaler.transform(songs[FEATURES])
    songs_z["year"] = songs_z["year"].astype(int)

    years, M, year_counts = build_timeline(songs_z, FEATURES)
    pca, coords, load1, load2 = pca_timeline(years, M)
    vel = velocity_series(years, M)
    episodes, vel_q90 = find_episodes(vel, year_counts)

    # ---- decade centroids, raw and standardised
    raw = songs.copy()
    raw["duration_min"] = raw["duration_ms"].apply(np.expm1) / 60000.0
    raw["decade"] = (raw["year"] // 10 * 10).astype(int)
    zdf = songs_z.copy()
    zdf["decade"] = (zdf["year"] // 10 * 10).astype(int)
    songs_z["decade"] = zdf["decade"].values
    raw_feats = [f for f in FEATURES if f != "duration_ms"] + ["duration_min"]
    decade_raw = {}
    decade_z = {}
    for d, sub in raw.groupby("decade"):
        if len(sub) < 20:
            continue
        decade_raw[int(d)] = {**{f: round(float(sub[f].mean()), 3) for f in raw_feats},
                             "songs": int(len(sub))}
        decade_z[int(d)] = {f: round(float(zdf.loc[sub.index, f].mean()), 3)
                            for f in FEATURES}
    # decades beyond the last reliable year (from raw counts)
    last_decade = int(raw["decade"].max())

    # ---- network
    J = rows[(rows["igenre"] == "Jazz") & (rows["fgenre"] == "Jazz")]
    net_ids = sorted(set(J["influencer_id"]) | set(J["follower_id"]))
    index = {a: k for k, a in enumerate(net_ids)}
    ii = [index[r.influencer_id] for r in J.itertuples()]
    jj = [index[r.follower_id] for r in J.itertuples()]
    A = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(net_ids), len(net_ids)))
    katz_vec = katz(A)
    K = {a: float(katz_vec[index[a]]) for a in net_ids}
    B = collections.Counter(r.influencer_id for r in rows.itertuples()
                            if r.igenre == "Jazz" and r.fgenre != "Jazz")
    future = collections.Counter()
    for r in rows.itertuples():
        if (r.igenre == "Jazz" and r.follower_id in start
                and start.get(r.influencer_id) is not None
                and start[r.follower_id] >= start[r.influencer_id] + 10):
            future[r.influencer_id] += 1

    # ---- residuals and indicators
    R, nsongs = residuals(songs_z, jazz_ids, years, M, FEATURES)
    R_loo = loo_residuals(songs_z, list(R.keys()), years, FEATURES)
    artist_cov = np.cov(scaler.transform(art_df[FEATURES]), rowvar=False)
    inv_cov = np.linalg.inv(artist_cov + 1e-6 * np.eye(len(FEATURES)))

    novelty = {a: maha_norm(R[a], inv_cov) for a in R}
    novelty_loo = {a: maha_norm(R_loo[a], inv_cov) for a in R_loo}
    anticipation = {}
    anticipation_loo = {}
    anticipation5 = {}
    for a in R:
        t = start[a]
        if t + 10 > years.max():
            continue
        drift = interpolate_row(years, M, t + 10) - interpolate_row(years, M, t)
        anticipation[a] = float(R[a] @ drift /
                                (np.linalg.norm(R[a]) * np.linalg.norm(drift) + 1e-12))
        if a in R_loo:
            anticipation_loo[a] = float(R_loo[a] @ drift /
                                        (np.linalg.norm(R_loo[a]) * np.linalg.norm(drift) + 1e-12))
        if t + 5 <= years.max():
            drift5 = interpolate_row(years, M, t + 5) - interpolate_row(years, M, t)
            anticipation5[a] = float(R[a] @ drift5 /
                                     (np.linalg.norm(R[a]) * np.linalg.norm(drift5) + 1e-12))

    pairs = [(r.influencer_id, r.follower_id) for r in J.itertuples()
             if (r.influencer_id in R and r.follower_id in R
                 and start[r.follower_id] > start[r.influencer_id])]
    align = {}
    per_pair = []
    for i, j in pairs:
        u = R[i] / (np.linalg.norm(R[i]) + 1e-12)
        v = float(R[j] @ u)
        align.setdefault(i, []).append(v)
        per_pair.append(v)
    H_raw = {a: float(np.mean(v)) for a, v in align.items()}
    H = {a: H_raw[a] * len(align[a]) / (len(align[a]) + 3.0) for a in H_raw}
    nfol = {a: len(v) for a, v in align.items()}
    shrink_rho = round(float(spearmanr(
        [H_raw[a] for a in H_raw if nfol[a] >= 5],
        [H[a] for a in H_raw if nfol[a] >= 5]).correlation), 3)

    # ---- permutation null for transmission
    by_decade = collections.defaultdict(list)
    for a in R:
        by_decade[start[a]].append(a)
    observed = float(np.mean(per_pair))
    null = np.empty(N_PERM)
    null_pair_pos = np.empty(N_PERM)
    for b in range(N_PERM):
        vals = []
        for i, j in pairs:
            jj = by_decade[start[j]][RNG.integers(len(by_decade[start[j]]))]
            vals.append(float(R[jj] @ (R[i] / (np.linalg.norm(R[i]) + 1e-12))))
        null[b] = np.mean(vals)
        null_pair_pos[b] = np.mean(np.asarray(vals) > 0)
    p_value = float((1 + np.sum(null >= observed)) / (1 + N_PERM))
    boot = np.empty(3000)
    for b in range(3000):
        idx = RNG.integers(0, len(per_pair), len(per_pair))
        boot[b] = np.mean(np.asarray(per_pair)[idx])
    pos_share = float(np.mean(np.asarray(per_pair) > 0))
    null_pos = float(np.mean(null_pair_pos))

    # ---- tableau
    records = []
    for a in R:
        records.append({
            "artist_id": int(a), "artist": names.get(a, str(a)), "start": start[a],
            "songs": int(nsongs.get(a, 0)), "novelty": novelty[a],
            "anticipation": anticipation.get(a, None),
            "H": H_raw.get(a, None), "H_shrunk": H.get(a, 0.0), "followers": nfol.get(a, 0),
            "katz": K.get(a, 1.0), "boundary": int(B.get(a, 0)),
            "future_followers": int(future.get(a, 0)),
        })
    T = pd.DataFrame(records)
    T["anticipation"] = T["anticipation"].astype(float)
    T["H"] = T["H"].astype(float)
    T.to_csv(ROOT / "q6_artists.csv", index=False)

    # z-scores for the composite (over artists with a defined anticipation)
    elig = T[T["anticipation"].notna()].copy()
    for col, z in [("anticipation", "zA"), ("H_shrunk", "zH"),
                   ("novelty", "zN"), ("katz", "zK"), ("boundary", "zB")]:
        mu, sd = elig[col].mean(), elig[col].std(ddof=0)
        elig[z] = (elig[col] - mu) / (sd if sd else 1.0)
    eligible_rank = elig[elig["followers"] >= 5].copy()
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        eligible_rank[f"S{int(w * 10)}"] = w * eligible_rank["zA"] + (1 - w) * eligible_rank["zH"]
    eligible_rank["S"] = eligible_rank["S5"]
    top_dynamic = eligible_rank.sort_values("S", ascending=False).head(15)
    weight_rho = {str(w): round(float(spearmanr(
        eligible_rank["S5"].rank(ascending=False),
        eligible_rank[f"S{int(w * 10)}"].rank(ascending=False)).correlation), 3)
        for w in (0.3, 0.4, 0.6, 0.7)}

    # ---- archetypes
    canon = elig.sort_values("katz", ascending=False).head(8)
    bridges = elig.sort_values("boundary", ascending=False).head(8)
    prophets = elig[(elig["zA"] > elig["zA"].quantile(0.75))
                    & (elig["H_shrunk"] < elig["H_shrunk"].median())
                    & (elig["followers"] >= 1)].sort_values("zA", ascending=False).head(8)
    catalysts = eligible_rank.sort_values("S", ascending=False).head(8)

    def slim(df, n=8):
        cols = ["artist", "start", "novelty", "anticipation", "H", "H_shrunk",
                "followers", "katz", "boundary", "future_followers", "S"]
        cols = [c for c in cols if c in df.columns]
        return df.head(n)[cols].round(3).to_dict("records")

    # ---- correlations
    corr_cols = ["novelty", "anticipation", "H_shrunk", "katz", "boundary", "future_followers"]
    corr = elig[corr_cols].corr(method="spearman")

    # ---- robustness: leave-one-out, five-year horizon, liveness included
    loo_rho = {
        "novelty": round(float(spearmanr(
            [novelty[a] for a in novelty_loo],
            [novelty_loo[a] for a in novelty_loo]).correlation), 4),
        "anticipation": round(float(spearmanr(
            [anticipation[a] for a in anticipation_loo],
            [anticipation_loo[a] for a in anticipation_loo]).correlation), 4),
    }
    horizon5_rho = round(float(spearmanr(
        [anticipation[a] for a in anticipation5],
        [anticipation5[a] for a in anticipation5]).correlation), 4)

    feats10 = FEATURES + ["liveness"]
    scaler10 = StandardScaler().fit(art_df[feats10])
    songs10_z = songs.copy()
    songs10_z[feats10] = scaler10.transform(songs[feats10])
    years10, M10, _ = build_timeline(songs10_z, feats10)
    R10, _ = residuals(songs10_z, jazz_ids, years10, M10, feats10)
    inv10 = np.linalg.inv(np.cov(scaler10.transform(art_df[feats10]), rowvar=False)
                          + 1e-6 * np.eye(len(feats10)))
    N10 = {a: maha_norm(R10[a], inv10) for a in R10}
    ant10 = {}
    for a in R10:
        t = start[a]
        if t + 10 > years10.max():
            continue
        drift = interpolate_row(years10, M10, t + 10) - interpolate_row(years10, M10, t)
        ant10[a] = float(R10[a] @ drift /
                         (np.linalg.norm(R10[a]) * np.linalg.norm(drift) + 1e-12))
    liveness_rho = {
        "novelty": round(float(spearmanr(
            [N10[a] for a in N10], [novelty[a] for a in N10]).correlation), 4),
        "anticipation": round(float(spearmanr(
            [ant10[a] for a in ant10], [anticipation[a] for a in ant10]).correlation), 4),
    }
    best10 = None
    X10 = songs10_z[feats10].values
    for seed in range(20):
        c, lab = kmeans2(X10, 4, minit="++", seed=seed)
        sse = ((X10 - c[lab]) ** 2).sum()
        if best10 is None or sse < best10[0]:
            best10 = (sse, c, lab)
    _, c10, lab10 = best10
    lv = c10[:, feats10.index("liveness")]
    artifact_id = int(np.argmax(lv))
    liveness_artifact = {
        "cluster_size": int((lab10 == artifact_id).sum()),
        "liveness_z": round(float(lv[artifact_id]), 2),
    }

    # ---- style clusters (descriptive; silhouette reported honestly)
    cluster_data = songs_z[FEATURES].values
    best = None
    for seed in range(30):
        c, lab = kmeans2(cluster_data, 4, minit="++", seed=seed)
        sse = ((cluster_data - c[lab]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, c, lab)
    _, centroids, labels = best
    sil = float(silhouette_score(cluster_data, labels))
    songs_z["cluster"] = labels
    cluster_label = {0: "A up-tempo acoustic", 1: "B short-form / songbook",
                     2: "C quiet ballad", 3: "D long-form modern"}
    cluster_desc = []
    for cid in range(4):
        sub = songs_z[songs_z["cluster"] == cid]
        top = sub["artist_id"].value_counts().head(5).index
        cluster_desc.append({
            "cluster": cluster_label[cid],
            "share": round(float(len(sub) / len(songs_z)), 3),
            "centroid": {f: round(float(centroids[cid][k]), 2) for k, f in enumerate(FEATURES)},
            "artists": [names.get(a, str(a)) for a in top],
        })
    dec_share = pd.crosstab(songs_z["decade"], songs_z["cluster"], normalize="index")
    style_share = {int(d): {cluster_label[c]: round(float(row.get(c, 0.0)), 3)
                            for c in range(4)} for d, row in dec_share.iterrows()}

    # ======================================================================
    # figures
    # ======================================================================
    # ---- fig 1: trajectory + decade heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4),
                            gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    dec_pts = [d for d in sorted(decade_z) if 1930 <= d <= 2000]
    D = np.array([[decade_z[d][f] for f in FEATURES] for d in dec_pts])
    DP = pca.transform(D)
    cols = plt.cm.viridis(np.linspace(0.04, 0.94, len(dec_pts)))
    ax.scatter(DP[:, 0], DP[:, 1], c=cols, s=62, zorder=3,
               edgecolors="white", linewidths=0.7)
    for k in range(len(dec_pts) - 1):
        ax.annotate("", xy=(DP[k + 1, 0], DP[k + 1, 1]),
                    xytext=(DP[k, 0], DP[k, 1]),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.1,
                                    shrinkA=8, shrinkB=8))
    for k, d in enumerate(dec_pts):
        ax.annotate(f"{d}s", (DP[k, 0], DP[k, 1]), fontsize=8.5, zorder=4,
                    xytext=(6, 4), textcoords="offset points",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.8))
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% of trajectory variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}%)")
    ax.set_title("(a) Decade centroids in feature space")
    ax.margins(0.14)

    ax = axes[1]
    decs = [d for d in sorted(decade_z) if 1930 <= d <= 2000]
    mat = np.array([[decade_z[d][f] for d in decs] for f in FEATURES])
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(decs)))
    ax.set_xticklabels([f"{d}s" for d in decs], rotation=45)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([PRETTY[f] for f in FEATURES], fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(decs)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7.2,
                    color="white" if abs(mat[i, j]) > 1.2 else "#222222")
    ax.set_title("(b) Standardised decade centroids")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="z-score vs all artists")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_genre_timeline.png", dpi=200)
    fig.savefig(FIG / "fig1_genre_timeline.pdf")
    plt.close(fig)

    # ---- fig 2: velocity + episode decomposition
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1),
                            gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    vy = [r["year"] for r in vel]
    vv = [r["velocity"] for r in vel]
    ax.plot(vy, vv, color="#4C72B0", lw=1.4)
    for ep in episodes:
        ax.axvspan(ep["year"] - 2, ep["year"] + 2, color="#C44E52", alpha=0.12)
        ax.annotate(f"{ep['year']}", (ep["year"], ep["velocity"]), fontsize=9,
                    xytext=(0, 7), textcoords="offset points", ha="center", color="#C44E52")
    ax.set_xlabel("year")
    ax.set_ylabel(r"$\|m(y+2)-m(y-2)\|$ in z-units")
    ax.set_title("(a) Speed of the genre centroid")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ep_rows = []
    for ep in episodes:
        y = ep["year"]
        delta = interpolate_row(years, M, y + 2) - interpolate_row(years, M, y - 2)
        ep_rows.append(delta)
    ep_mat = np.array(ep_rows).T
    im = ax.imshow(ep_mat, aspect="auto", cmap="RdBu_r", vmin=-1.1, vmax=1.1)
    ax.set_xticks(range(len(episodes)))
    ax.set_xticklabels([f"{e['year']}" for e in episodes])
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([PRETTY[f] for f in FEATURES], fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(episodes)):
            ax.text(j, i, f"{ep_mat[i, j]:+.2f}", ha="center", va="center", fontsize=7.4,
                    color="white" if abs(ep_mat[i, j]) > 0.6 else "#222222")
    ax.set_title("(b) Feature change around each episode")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="change in z-units")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_revolutions.png", dpi=200)
    fig.savefig(FIG / "fig2_revolutions.pdf")
    plt.close(fig)

    # ---- fig 3: style shares + centroids
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1),
                            gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    dds = [d for d in sorted(style_share) if 1930 <= d <= 2000]
    bottom = np.zeros(len(dds))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
    for cid in range(4):
        vals = np.array([style_share[d][cluster_label[cid]] for d in dds])
        ax.bar(range(len(dds)), vals, bottom=bottom, color=colors[cid],
               label=cluster_label[cid], width=0.78)
        bottom += vals
    ax.set_xticks(range(len(dds)))
    ax.set_xticklabels([f"{d}s" for d in dds], rotation=45)
    ax.set_ylabel("share of the decade's songs")
    ax.set_title(f"(a) Sonic profile shares (k=4, silhouette {sil:.2f})")
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    im = ax.imshow(centroids, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
    ax.set_yticks(range(4))
    ax.set_yticklabels([cluster_label[c] for c in range(4)], fontsize=8.5)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([PRETTY[f] for f in FEATURES], rotation=60, ha="right", fontsize=7.5)
    for i in range(4):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{centroids[i, j]:.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(centroids[i, j]) > 1.2 else "#222222")
    ax.set_title("(b) Cluster centroids")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.004))
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(FIG / "fig3_styles.png", dpi=200)
    fig.savefig(FIG / "fig3_styles.pdf")
    plt.close(fig)

    # ---- fig 4: influencer map + ranking
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6),
                            gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    q = eligible_rank.copy()
    sizes = 18 + 12 * q["katz"]
    sc = ax.scatter(q["anticipation"], q["H_shrunk"], s=sizes, c=q["start"],
                    cmap="viridis", alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    labels = ["Miles Davis", "John Coltrane", "Charlie Parker", "Thelonious Monk",
              "Herbie Hancock", "George Duke", "Grover Washington, Jr.",
              "Tal Farlow", "Ornette Coleman"]
    offsets = {"Miles Davis": (-52, -20), "John Coltrane": (-18, 16),
               "Charlie Parker": (-30, -22), "Thelonious Monk": (12, 12),
               "Herbie Hancock": (10, -20),
               "George Duke": (10, -12), "Grover Washington, Jr.": (-28, 14),
               "Tal Farlow": (-58, 12), "Ornette Coleman": (-72, -8)}
    for nm in labels:
        hit = q[q["artist"] == nm]
        if len(hit):
            r = hit.iloc[0]
            ax.annotate(nm, (r["anticipation"], r["H_shrunk"]), fontsize=7.2,
                        xytext=offsets.get(nm, (4, 4)), textcoords="offset points",
                        bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.6))
    ax.set_xlabel("anticipation  A  (ahead of their time)")
    ax.set_ylabel("transmission  H*  (followers moved toward them)")
    ax.set_title("(a) Four indicator map (bubble = within-jazz Katz)")
    ax.set_xlim(q["anticipation"].min() - 0.12, q["anticipation"].max() + 0.38)
    ax.margins(0.10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("active start (decade)")

    ax = axes[1]
    top12 = top_dynamic.head(12).iloc[::-1]
    ax.barh(range(len(top12)), top12["S"], color="#C44E52", alpha=0.85)
    ax.set_yticks(range(len(top12)))
    ax.set_yticklabels([f"{r.artist} ({r.start}s)" for r in top12.itertuples()], fontsize=8.5)
    for k, r in enumerate(top12.itertuples()):
        ax.text(r.S + 0.03, k, f"A {r.anticipation:+.2f}  H {r.H_shrunk:+.2f}",
                va="center", fontsize=7.2, color="#333333")
    ax.set_xlabel("dynamic-influence score  z(A) + z(H*)")
    ax.set_title("(b) Top dynamic influencers")
    ax.set_xlim(0, top12["S"].max() * 1.55)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_influencers.png", dpi=200)
    fig.savefig(FIG / "fig5_influencers.pdf")
    plt.close(fig)

    # ---- fig 5: permutation null + correlation matrix
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    ax = axes[0]
    ax.hist(null, bins=30, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(observed, color="#C44E52", lw=2)
    ax.annotate(f"observed {observed:.3f}\n(99% null < {np.quantile(null, .99):.3f})",
                (observed, ax.get_ylim()[1] * 0.6), fontsize=8.5, color="#C44E52",
                xytext=(-4, 0), textcoords="offset points", ha="right")
    ax.set_xlabel("mean follower alignment (cosine)")
    ax.set_ylabel("permutations")
    p_txt = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.set_title(f"(a) Reported influence is visible: {p_txt}")

    ax = axes[1]
    short = {"novelty": "N", "anticipation": "A", "H_shrunk": "H", "katz": "K",
             "boundary": "B", "future_followers": "F"}
    cm = corr.rename(index=short, columns=short)
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm)))
    ax.set_yticklabels(cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(cm.values[i, j]) > 0.55 else "#222222")
    ax.set_title("(b) Spearman correlations of indicators")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_validation.png", dpi=200)
    fig.savefig(FIG / "fig4_validation.pdf")
    plt.close(fig)

    # ======================================================================
    # results
    # ======================================================================
    result = {
        "data": {
            "raw_edges": int(len(inf)),
            "network_edges": int(len(rows)),
            "network_nodes": int(len(set(rows["influencer_id"]) | set(rows["follower_id"]))),
            "same_decade_edges": int((rows["iy"] == rows["fy"]).sum()),
            "reversed_edges": int((rows["iy"] > rows["fy"]).sum()),
            "jazz_edges_all": int(((rows.igenre == "Jazz") | (rows.fgenre == "Jazz")).sum()),
            "jazz_jazz_edges": int(len(J)),
            "jazz_to_pop_rock_edges": int(((rows["igenre"] == "Jazz")
                                           & (rows["fgenre"] == "Pop/Rock")).sum()),
            "jazz_network_nodes": int(len(net_ids)),
            "jazz_artists": int(len(jazz_ids)),
            "jazz_artists_used": int(len(R)),
            "jazz_songs": int(len(songs)),
            "jazz_song_artists": int(songs.artist_id.nunique()),
            "year_min": int(songs.year.min()), "year_max": int(songs.year.max()),
            "timeline_min": int(years.min()), "timeline_max": int(years.max()),
            "min_songs_year": MIN_SONGS_YEAR,
            "min_songs_artist": MIN_SONGS_ARTIST,
            "features": FEATURES,
            "year_song_counts": {int(k): int(v) for k, v in year_counts.items()},
        },
        "timeline": {
            "pca_explained": [round(float(x), 4) for x in pca.explained_variance_ratio_],
            "pca_loadings_pc1": {f: round(float(load1[f]), 3) for f in FEATURES},
            "pca_loadings_pc2": {f: round(float(load2[f]), 3) for f in FEATURES},
            "trajectory": [{"year": int(y), "pc1": round(float(coords[k, 0]), 3),
                            "pc2": round(float(coords[k, 1]), 3), "songs": int(year_counts.get(y, 0))}
                           for k, y in enumerate(years)],
            "decade_raw": {str(k): v for k, v in decade_raw.items()},
            "decade_z": {str(k): v for k, v in decade_z.items()},
            "velocity": [{"year": r["year"], "velocity": round(r["velocity"], 3)} for r in vel],
            "velocity_q90": round(vel_q90, 3),
            "episodes": [{"year": e["year"], "velocity": round(e["velocity"], 3)}
                         for e in episodes],
            "episode_feature_change": {
                str(e["year"]): {f: round(float(interpolate_row(years, M, e["year"] + 2)[k]
                                               - interpolate_row(years, M, e["year"] - 2)[k]), 3)
                                 for k, f in enumerate(FEATURES)}
                for e in episodes},
        },
        "adoption": {
            "pairs": int(len(pairs)),
            "mean_alignment": round(observed, 4),
            "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std()), 4),
            "null_q95": round(float(np.quantile(null, 0.95)), 4),
            "null_q99": round(float(np.quantile(null, 0.99)), 4),
            "p_value": round(p_value, 4),
            "bootstrap_ci95": [round(float(np.quantile(boot, 0.025)), 4),
                               round(float(np.quantile(boot, 0.975)), 4)],
            "positive_share": round(pos_share, 4),
            "null_positive_share": round(null_pos, 4),
        },
        "indicators": {
            "top_dynamic": slim(top_dynamic, 12),
            "weight_robustness_spearman": weight_rho,
            "archetypes": {
                "catalysts_ahead_and_copied": slim(catalysts, 8),
                "canon_high_katz": slim(canon, 8),
                "bridges_cross_genre": slim(bridges, 8),
                "prophets_ahead_not_canonised": slim(prophets, 8),
            },
            "spearman": {r: {c: round(float(corr.loc[r, c]), 3) for c in corr_cols}
                         for r in corr_cols},
            "with_anticipation": int(len(elig)),
            "with_followers": int(T["H"].notna().sum()),
            "eligible": int(len(eligible_rank)),
            "shrinkage_spearman": shrink_rho,
        },
        "styles": {
            "silhouette": round(sil, 3),
            "clusters": cluster_desc,
            "share_by_decade": {str(k): v for k, v in style_share.items()},
        },
        "robustness": {
            "loo_spearman": loo_rho,
            "horizon5_spearman": horizon5_rho,
            "liveness_included_spearman": liveness_rho,
            "liveness_artifact": liveness_artifact,
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"jazz artists {len(jazz_ids)}, used {len(R)}, songs {len(songs)}")
    print(f"timeline {years.min()}--{years.max()}, episodes {[e['year'] for e in episodes]}")
    print(f"adoption {observed:.3f} vs null {null.mean():.3f}+-{null.std():.3f}, p={p_value:.4f}")
    print(f"loo rho {loo_rho}, liveness rho {liveness_rho}")
    print(top_dynamic.head(12)[["artist", "start", "novelty", "anticipation",
                                 "H_shrunk", "followers", "katz", "boundary", "S"]].round(2).to_string(index=False))
    print(f"wrote {RESULT}")


if __name__ == "__main__":
    main()
