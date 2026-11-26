"""Q4: Do reported influencers actually leave an acoustic trace on followers?

Design (matched-control excess, coherent with Models I-III):
  For every reported edge (i, j):
    - edge similarity      s(i, j)   = 1 / (1 + ||x_i - x_j||_2)   in Model II
      z-space (11 features, global column-wise z-score, winsorised at +-4 sd);
    - K = 10 matched controls j': same main genre AND same active-start decade
      as j, not reported as influenced by i, with features available;
    - excess similarity  s_e = s(i, j) - mean_controls s(i, j').
  Overall: mean excess + bootstrap CI + Wilcoxon signed-rank (paired design).
  Per-feature contagion a_f: mean/median of |x_i^f - x_j^f| on controls minus
  on edges; plus directional regression  x_j^f ~ x_i^f + genre FE + decade FE
  fitted separately on edges (beta_f) and on pooled controls (beta0_f).
  Dose-response: mean excess by influencer-Katz quintile (Model I link).

Outputs -> C/q4_outputs/ (json, csv, edge-level table, figures fig11/fig12).
Gates: counts of dropped edges disclosed; control coverage reported.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)
DATA = "ProblemC_Music_data"
OUT = "q4_outputs"
K_CONTROLS = 10
N_BOOT = 2000

FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo", "loudness", "acousticness",
    "instrumentalness", "liveness", "speechiness", "duration_ms", "popularity",
]

# ------------------------------------------------------------------ load edges
edges = pd.read_csv(f"{DATA}/influence_data.csv",
                    dtype={"influencer_id": "string", "follower_id": "string"},
                    encoding="utf-8")
for c in ["influencer_id", "follower_id"]:
    edges[c] = edges[c].str.strip()
for c in ["influencer_main_genre", "follower_main_genre"]:
    edges[c] = edges[c].astype("string").str.strip().str.replace(";", "", regex=False)
for c in ["influencer_active_start", "follower_active_start"]:
    edges[c] = pd.to_numeric(edges[c], errors="coerce").astype("Int64")
edges = (edges.dropna(subset=["influencer_id", "follower_id"])
              .drop_duplicates(["influencer_id", "follower_id"]))
n_edges_total = len(edges)
print(f"edges: {n_edges_total}")

# genre / start dictionary from influence_data (as in section 4.1)
inf_meta = (edges[["influencer_id", "influencer_main_genre",
                   "influencer_active_start"]]
            .rename(columns={"influencer_id": "artist_id",
                             "influencer_main_genre": "genre",
                             "influencer_active_start": "start"}))
fol_meta = (edges[["follower_id", "follower_main_genre", "follower_active_start"]]
            .rename(columns={"follower_id": "artist_id",
                             "follower_main_genre": "genre",
                             "follower_active_start": "start"}))
meta = (pd.concat([inf_meta, fol_meta])
          .dropna(subset=["artist_id"])
          .sort_values(["artist_id", "start"])
          .drop_duplicates("artist_id", keep="first")   # influencer role wins
          .set_index("artist_id"))
print(f"genre dictionary: {len(meta)} artists")

# ------------------------------------------------- features (Model II recipe)
feat = pd.read_csv(f"{DATA}/data_by_artist.csv")
feat = feat.dropna(subset=FEATURE_COLS).drop_duplicates("artist_id")
feat["artist_id"] = feat["artist_id"].astype(str).str.strip()
X = StandardScaler().fit_transform(feat[FEATURE_COLS].to_numpy(dtype=float))
X = np.clip(X, -4, 4)                                    # winsorise heavy tails
Z = pd.DataFrame(X, index=feat["artist_id"], columns=FEATURE_COLS)
print(f"feature table: {Z.shape[0]} artists")

# join edges to features + follower meta
e = edges.copy()
e["i"] = e["influencer_id"].astype(str)
e["j"] = e["follower_id"].astype(str)
e = e[e["i"].isin(Z.index) & e["j"].isin(Z.index)].copy()
e = e.merge(meta[["genre", "start"]], left_on="j", right_index=True, how="left")
n_with_features = len(e)
print(f"edges with both endpoints in feature table: {n_with_features} "
      f"(dropped {n_edges_total - n_with_features})")
e = e[e["start"].notna()].copy()
e["jdec"] = e["start"].astype(int)
n_matched = len(e)
print(f"edges with follower genre+decade known: {n_matched}")

edge_ids = set(zip(e["i"], e["j"]))
edge_pair_str = {f"{a}|{b}" for a, b in edge_ids}

# candidate control pool per (genre, decade)
pool = (Z.join(meta[["genre", "start"]], how="inner").reset_index()
         .rename(columns={"index": "artist_id"}))
pool = pool[pool["genre"].notna() & pool["start"].notna()].copy()
pool["dec"] = pool["start"].astype(int)
pool_by_cell = {k: g["artist_id"].to_numpy()
                for k, g in pool.groupby(["genre", "dec"])}
feat_by_id = Z.loc[pool["artist_id"]].to_numpy(dtype=float)
id_pos = {a: p for p, a in enumerate(pool["artist_id"])}

Xi = Z.loc[e["i"]].to_numpy(dtype=float)
Xj = Z.loc[e["j"]].to_numpy(dtype=float)

# ---------------------------------------------------------- matched controls
ctrl_rows = []          # (edge position, list of control pool positions)
n_no_ctrl = 0
e_ids = e["j"].to_numpy()
i_ids = e["i"].to_numpy()
j_genre = e["genre"].to_numpy()
j_dec = e["jdec"].to_numpy()
for r in range(len(e)):
    cell = (j_genre[r], j_dec[r])
    cand = pool_by_cell.get(cell)
    if cand is None:
        n_no_ctrl += 1
        continue
    cand = cand[cand != e_ids[r]]
    if len(cand) == 0:
        n_no_ctrl += 1
        continue
    if len(cand) <= K_CONTROLS + 50:
        ok = [a for a in cand if f"{i_ids[r]}|{a}" not in edge_pair_str]
    else:
        ok = []
        for _ in range(6):                      # sampled rejection; fast path
            draw = rng.choice(cand, size=K_CONTROLS + 50, replace=False)
            draw = [a for a in draw if f"{i_ids[r]}|{a}" not in edge_pair_str]
            if len(draw) >= K_CONTROLS:
                ok = draw[:K_CONTROLS]
                break
            ok.extend(draw)
        ok = list(dict.fromkeys(ok))[:K_CONTROLS]
    if not ok:
        n_no_ctrl += 1
        continue
    ctrl_rows.append((r, [id_pos[a] for a in ok]))
print(f"edges with >=1 control: {len(ctrl_rows)} (no control: {n_no_ctrl})")

pos = np.array([r for r, _ in ctrl_rows])
Xi = Xi[pos]
Xj = Xj[pos]
ex = e.iloc[pos].copy().reset_index(drop=True)

# distances
d_edge = np.linalg.norm(Xi - Xj, axis=1)
s_edge = 1.0 / (1.0 + d_edge)
d_ctrl = np.empty(len(ex))
s_ctrl = np.empty(len(ex))
for k, (_, sel) in enumerate(ctrl_rows):
    Xc = feat_by_id[sel]
    dc = np.linalg.norm(Xi[k] - Xc, axis=1)
    d_ctrl[k] = dc.mean()
    s_ctrl[k] = (1.0 / (1.0 + dc)).mean()
excess = s_edge - s_ctrl
dist_excess = d_ctrl - d_edge
ex["excess"] = excess

print(f"mean excess similarity: {excess.mean():.4f} "
      f"(edge {s_edge.mean():.4f} vs control {s_ctrl.mean():.4f})")
print(f"mean distance excess:   {dist_excess.mean():.4f}")

# overall inference
boot = np.array([rng.choice(excess, size=len(excess), replace=True).mean()
                 for _ in range(N_BOOT)])
ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
stat, p_wil = wilcoxon(excess)
print(f"bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]; "
      f"Wilcoxon p={p_wil:.2e}; share>0: {(excess > 0).mean():.3f}")

# ------------------------------------------------- per-feature contagion a_f
abs_edge = np.abs(Xi - Xj)
per_f = []
for f, col in enumerate(FEATURE_COLS):
    ctrl_abs = []
    for k, (_, sel) in enumerate(ctrl_rows):
        ctrl_abs.append(np.abs(Xi[k, f] - feat_by_id[sel, f]))
    ctrl_abs = np.concatenate(ctrl_abs)
    a_mean = ctrl_abs.mean() - abs_edge[:, f].mean()
    a_med = np.median(ctrl_abs) - np.median(abs_edge[:, f])
    per_f.append({"feature": col, "excess_mean": round(a_mean, 4),
                  "excess_median": round(a_med, 4)})
    print(f"{col:>18}: a_mean={a_mean:+.4f}  a_med={a_med:+.4f}")

# ------------------------------------------- directional regression beta_f
p_dim = len(FEATURE_COLS)
n_e = len(ex)
genre_code = pd.Categorical(ex["genre"]).codes
dec_code = pd.Categorical(ex["jdec"]).codes
G = np.zeros((n_e, genre_code.max() + 1)); G[np.arange(n_e), genre_code] = 1
Dm = np.zeros((n_e, dec_code.max() + 1)); Dm[np.arange(n_e), dec_code] = 1
A_edge = np.hstack([np.ones((n_e, 1)), Xi, G[:, 1:], Dm[:, 1:]])

ctrl_i, ctrl_j, ctrl_g, ctrl_d = [], [], [], []
pool_genres = pool["genre"].to_numpy()
pool_decs = pool["dec"].to_numpy()
pool_ids = pool["artist_id"].to_numpy()
for k, (_, sel) in enumerate(ctrl_rows):
    for pos in sel:
        ctrl_i.append(k); ctrl_j.append(pos)
        ctrl_g.append(str(pool_genres[pos])); ctrl_d.append(int(pool_decs[pos]))
n_c = len(ctrl_i)
Xi_c = Xi[np.asarray(ctrl_i)]
Xj_c = feat_by_id[np.asarray(ctrl_j)]
g_code_c = pd.Categorical(pd.Series(ctrl_g)).codes
d_code_c = pd.Categorical(pd.Series(ctrl_d)).codes
Gc = np.zeros((n_c, g_code_c.max() + 1)); Gc[np.arange(n_c), g_code_c] = 1
Dc = np.zeros((n_c, d_code_c.max() + 1)); Dc[np.arange(n_c), d_code_c] = 1
A_ctrl = np.hstack([np.ones((n_c, 1)), Xi_c, Gc[:, 1:], Dc[:, 1:]])

betas_e, betas_c = {}, {}
for f, col in enumerate(FEATURE_COLS):
    y_e = Xj[:, f]
    y_c = Xj_c[:, f]
    be, *_ = np.linalg.lstsq(A_edge, y_e, rcond=None)
    bc, *_ = np.linalg.lstsq(A_ctrl, y_c, rcond=None)
    betas_e[col] = float(be[1])
    betas_c[col] = float(bc[1])
    print(f"{col:>18}: beta_edge={be[1]:+.3f}  beta_ctrl={bc[1]:+.3f}  "
          f"delta={be[1]-bc[1]:+.3f}")

# ------------------------------------------------- Katz dose-response (Model I)
katz = pd.read_csv("q1_outputs/artist_network_indicators.csv",
                   usecols=["artist_id", "katz_out"], dtype={"artist_id": "string"})
katz["artist_id"] = katz["artist_id"].str.strip()
kmap = katz.set_index("artist_id")["katz_out"]
ex["katz_i"] = ex["i"].str.strip().map(kmap)
rho_k, p_k = spearmanr(ex["katz_i"], excess)
ex["katz_bin"] = pd.qcut(ex["katz_i"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
dose = ex.groupby("katz_bin", observed=True)["excess"].mean()
print(f"Spearman(excess, influencer Katz): rho={rho_k:.3f}, p={p_k:.2e}")
print("dose-response:", {k: round(v, 4) for k, v in dose.items()})

# decade curve
ex["decade"] = ex["jdec"]
dec_curve = ex.groupby("decade")["excess"].agg(["mean", "size"])
print(dec_curve.round(4).to_string())

# ------------------------------------------------------------------- save
import os
os.makedirs(f"{OUT}/figures", exist_ok=True)
per_f_df = pd.DataFrame(per_f)
per_f_df["beta_edge"] = [betas_e[c] for c in FEATURE_COLS]
per_f_df["beta_ctrl"] = [betas_c[c] for c in FEATURE_COLS]
per_f_df["beta_delta"] = per_f_df["beta_edge"] - per_f_df["beta_ctrl"]
per_f_df.sort_values("excess_median", ascending=False).to_csv(
    f"{OUT}/per_feature_contagion.csv", index=False)
ex.assign(excess=excess, s_edge=s_edge, s_ctrl=s_ctrl,
          dist_edge=d_edge, dist_ctrl=d_ctrl)[
    ["influencer_id", "follower_id", "influencer_main_genre",
     "follower_main_genre", "jdec", "katz_i", "s_edge", "s_ctrl",
     "excess", "dist_edge", "dist_ctrl"]
].to_csv(f"{OUT}/edge_excess.csv", index=False)

results = {
    "edges_total": int(n_edges_total),
    "edges_used": int(len(ex)),
    "edges_dropped_no_features": int(n_with_features - n_matched) + (n_edges_total - n_with_features),
    "edges_no_control": int(n_no_ctrl),
    "k_controls": K_CONTROLS,
    "edge_sim_mean": float(s_edge.mean()),
    "control_sim_mean": float(s_ctrl.mean()),
    "excess_mean": float(excess.mean()),
    "excess_ci95": [float(ci[0]), float(ci[1])],
    "wilcoxon_p": float(p_wil),
    "share_positive": float((excess > 0).mean()),
    "distance_excess_mean": float(dist_excess.mean()),
    "spearman_excess_katz": {"rho": float(rho_k), "p": float(p_k)},
    "dose_response": {str(k): float(v) for k, v in dose.items()},
    "decade_curve": {int(k): {"mean": float(v["mean"]), "n": int(v["size"])}
                     for k, v in dec_curve.iterrows()},
    "per_feature": per_f,
    "beta_edge": betas_e,
    "beta_ctrl": betas_c,
}
with open(f"{OUT}/contagion_results.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("saved outputs to", OUT)

# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

pf = per_f_df.sort_values("excess_median")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
colors = ["#0F6E56" if v > 0.10 else "#5DCAA5" if v > 0.05 else "#B4B2A9"
          for v in pf["excess_median"]]
ax.barh(pf["feature"], pf["excess_median"], color=colors, alpha=0.9,
        label="median excess |z-gap| (edges vs controls)")
ax.scatter(pf["excess_mean"], pf["feature"], color="#712B13", s=14, zorder=3,
           label="mean excess (mean is tail-sensitive)")
ax.axvline(0, color="#444441", lw=0.8)
ax.set_xlabel("contagion: control |z-gap| minus edge |z-gap| (sd units)")
ax.set_title("Which features are contagious?  All positive; stylistic >> technical")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig11_feature_contagion.png", dpi=200,
            bbox_inches="tight")
fig.savefig(f"{OUT}/figures/fig11_feature_contagion.pdf", bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
dose.plot(kind="bar", color="#378ADD", alpha=0.9, ax=axes[0],
          rot=0, legend=False)
axes[0].set_xlabel("influencer outgoing-Katz quintile (Model I)")
axes[0].set_ylabel("mean excess similarity s*")
axes[0].set_title("(a) Dose-response: hubs do not transmit more sound\n"
                  rf"Spearman $\rho$={rho_k:.2f} (p={p_k:.1e})", fontsize=9)
dec_curve["mean"].plot(ax=axes[1], color="#D85A30", marker="o", ms=3,
                       lw=1.2, legend=False)
axes[1].set_xlabel("follower active-start decade")
axes[1].set_title("(b) Acoustic trace by cohort\npeak 1980s, decline thereafter",
                  fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig12_dose_response.png", dpi=200,
            bbox_inches="tight")
fig.savefig(f"{OUT}/figures/fig12_dose_response.pdf", bbox_inches="tight")
print("figures saved")
