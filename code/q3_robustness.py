"""Add-on robustness for the q3 genre-flow analysis.

(a) Permutation test for the Katz-weighted cross-genre share c_K(t):
    within each follower-decade, the influencer genre labels are permuted
    (keeping the decade's edge count and the follower Katz weights fixed),
    giving a null distribution of c_K(t); p-value = share of permutations
    with c_K >= observed (one-sided, large = unusually cross-genre-heavy).

(b) Mantel test between sonic distance D_gh (Katz-weighted centroids,
    teammate's genre_centroid_distances_katz.csv) and symmetric cross-genre
    flow Fsym = F + F^T (upper triangle, log1p-transformed). Significance by
    row/column permutation of one matrix.

Outputs: C/q3_outputs/robustness_results.json
"""
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DATA = "ProblemC_Music_data/influence_data.csv"
OUT = "q3_outputs/robustness_results.json"

# ---------------------------------------------------------------- load edges
df = pd.read_csv(DATA, dtype={"influencer_id": "string", "follower_id": "string"},
                 encoding="utf-8")
for c in ["influencer_id", "follower_id"]:
    df[c] = df[c].str.strip()
for c in ["influencer_main_genre", "follower_main_genre"]:
    df[c] = df[c].astype("string").str.strip().str.replace(";", "", regex=False)
for c in ["influencer_active_start", "follower_active_start"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
df = (df.dropna(subset=["influencer_id", "follower_id"])
        .drop_duplicates(["influencer_id", "follower_id"]))

katz = pd.read_csv("q1_outputs/artist_network_indicators.csv",
                   usecols=["artist_id", "katz_out"], dtype={"artist_id": "string"})
katz["artist_id"] = katz["artist_id"].str.strip()
kmap = katz.set_index("artist_id")["katz_out"]
kw = df["follower_id"].str.strip().map(kmap).fillna(1.0).to_numpy()

dec = df["follower_active_start"].to_numpy()
gcol = df["influencer_main_genre"].to_numpy()
fcol = df["follower_main_genre"].to_numpy()

# ------------------------------------------------ (a) permutation test for c_K
PERMS = 1000
perm_results = {}
for t in [1950, 1960, 1970, 1980, 1990, 2000]:
    m = dec == t
    if m.sum() < 50:
        continue
    cross = (gcol[m] != fcol[m])
    obs = float(kw[m][cross].sum() / kw[m].sum())          # observed c_K(t)
    gm, fm, wm = gcol[m], fcol[m], kw[m]
    nulls = np.empty(PERMS)
    for k in range(PERMS):
        pm = rng.permutation(len(gm))                       # shuffle influencer genres
        nulls[k] = wm[gm[pm] != fm].sum() / wm.sum()
    mu, sd = float(nulls.mean()), float(nulls.std())
    perm_results[int(t)] = {"c_K_obs": round(obs, 4),
                            "null_mean": round(mu, 4), "null_sd": round(sd, 4),
                            "z": round((obs - mu) / sd, 3),
                            "p_lower": round(float((nulls <= obs).mean()), 4),
                            "p_upper": round(float((nulls >= obs).mean()), 4)}
    print(f"{t}: c_K={obs:.3f}  null={mu:.3f}±{sd:.3f}  z={(obs-mu)/sd:.2f}")

# ---------------------------------------------------------------- (b) Mantel
F = pd.read_csv("q3_outputs/genre_flow_matrix_F.csv", index_col=0)
D = pd.read_csv("outputs/genre_centroid_distances_katz.csv",
                index_col=0)
D.columns = [c.strip().replace("R&B;", "R&B") for c in D.columns]
D.index = [i.strip().replace("R&B;", "R&B") for i in D.index]
genres = [g for g in D.index if g in F.columns and g != "Unknown"]
F19 = F.loc[genres, genres].to_numpy()
D19 = D.loc[genres, genres].to_numpy()

Fsym = F19 + F19.T                       # symmetric cross-flow
n = len(genres)
iu = np.triu_indices(n, k=1)
x = np.log1p(Fsym[iu])                   # flow (log scale)
y = D19[iu]                              # sonic distance
r_obs = float(np.corrcoef(x, y)[0, 1])   # Mantel r (Pearson on log flow)
from scipy.stats import spearmanr
rho_obs = float(spearmanr(x, y).statistic)

MANT = 5000
hits = 0
for _ in range(MANT):
    p = rng.permutation(n)
    Dp = D19[np.ix_(p, p)]               # permute rows+cols of distance matrix
    r_null = float(np.corrcoef(x, Dp[iu])[0, 1])
    if r_null <= r_obs:
        hits += 1
mantel = {"mantel_r": round(r_obs, 4), "spearman_rho": round(rho_obs, 4),
          "p_one_sided": round(hits / MANT, 5), "perms": MANT, "pairs": int(n * (n - 1) // 2)}
print(f"Mantel r={r_obs:.3f} (log flow vs distance), p={hits/MANT:.4f}")

with open(OUT, "w") as fh:
    json.dump({"cK_permutation": perm_results, "mantel": mantel}, fh, indent=2)
print("saved", OUT)
