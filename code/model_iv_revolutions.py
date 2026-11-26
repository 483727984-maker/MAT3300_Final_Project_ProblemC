"""
Model IV / Task 5 — Dual-channel musical revolution detection.

Notation (shared with Models I–III and q3_genre_flows.py):
  x(t)      = z-scored yearly feature vector from data_by_year (p=11, Model II coords)
  E(t)      = ||x(t) - x(t-1)||_2                     sonic leap size
  F(t)      = genre-flow matrix in follower active-start decade t   (teammate)
  c_cross(t)= cross-genre edge share in decade t
  c_K(t)    = Katz-weighted cross-genre share in decade t
  K_i       = outgoing Katz (Model I)
  N(t)      = network restructuring score for decade t
              = 0.5 * z(|Δ c_cross|) + 0.5 * z(|Δ c_K|)
                (within the decade panel; equal weight)

Revolution (operational):
  Evolution  = gradual drift of x(t) / slow change of c_cross, c_K.
  Revolution = short window where sonic rate E(t) is anomalous relative to
               its own baseline, optionally reinforced by elevated N(t)
               in the enclosing follower-start decade (dual-channel).
  Strong     = sonic anomaly years whose decade also has N(t) above baseline.
  Weak       = sonic-only or network-only anomaly.

Revolutionaries (in a flagged decade d):
  high K_i; feature displacement from the previous-decade mean aligned with
  the decade's mean sonic Δx; many followers whose active_start falls in d
  or d+10.

Usage:
  export MPLBACKEND=Agg MPLCONFIGDIR=code/.mplconfig
  /opt/anaconda3/bin/python code/model_iv_revolutions.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ProblemC_Music_data"
Q1 = ROOT / "code" / "q1_outputs"
Q3 = ROOT / "code" / "q3_outputs"
OUT = ROOT / "code" / "outputs"
FIG = ROOT / "code" / "figures"
REP_FIG = ROOT / "report" / "figures"
for p in (OUT, FIG, REP_FIG):
    p.mkdir(parents=True, exist_ok=True)

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

# Match teammate MAIN_WINDOW; early Spotify aggregates are thin.
YEAR_MIN, YEAR_MAX = 1930, 2010
# Local (rolling) z-score of E(t): detects leaps relative to recent evolution,
# avoiding early-year corpus noise dominating a global threshold.
ROLL_WIN = 12
SONIC_Z = 2.0          # rolling z_E anomaly threshold
NETWORK_Z = 0.75       # N(t) elevated if z-score of leap >= this
# Years before this are plotted but treated as composition-caution for naming.
SONIC_TRUST_YEAR = 1948
TOP_DRIVERS = 4
TOP_REVOLUTIONARIES = 8
MIN_FOLLOWERS = 5


def decade_of(year: int) -> int:
    return int(year) // 10 * 10


def load_sonic() -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    raw = pd.read_csv(DATA / "data_by_year.csv")
    raw = raw[(raw["year"] >= YEAR_MIN) & (raw["year"] <= YEAR_MAX)].copy()
    raw = raw.sort_values("year").reset_index(drop=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(raw[FEATURE_COLS].to_numpy(dtype=float))
    return raw, X, scaler


def sonic_channel(years: np.ndarray, X: np.ndarray) -> pd.DataFrame:
    """Yearly leap size E(t), drivers, and anomaly flags."""
    dX = np.diff(X, axis=0)
    E = np.linalg.norm(dX, axis=1)
    # Foote-style local novelty: distance between mean of past w and future w
    w = 3
    novelty = np.full(len(E), np.nan)
    for i in range(w, len(X) - w):
        past = X[i - w : i].mean(axis=0)
        fut = X[i : i + w].mean(axis=0)
        novelty[i - 1] = float(np.linalg.norm(fut - past))  # aligned to year[i]

    # Rolling z-score of E: compare each leap to the preceding ROLL_WIN leaps
    zE = np.full(len(E), np.nan)
    for i in range(len(E)):
        lo = max(0, i - ROLL_WIN)
        hist = E[lo:i]
        if len(hist) < max(5, ROLL_WIN // 2):
            continue
        mu_i, sd_i = float(hist.mean()), float(hist.std(ddof=1))
        zE[i] = (E[i] - mu_i) / sd_i if sd_i > 1e-12 else 0.0

    mu, sd = float(E.mean()), float(E.std(ddof=1))
    rows = []
    for i, e in enumerate(E):
        year = int(years[i + 1])  # leap ending at this year
        abs_share = np.abs(dX[i]) / (e + 1e-12)
        order = np.argsort(-abs_share)
        drivers = [
            {"feature": FEATURE_COLS[j], "share": float(abs_share[j]), "delta": float(dX[i, j])}
            for j in order[:TOP_DRIVERS]
        ]
        z = zE[i]
        trusted = year >= SONIC_TRUST_YEAR
        anom = bool(trusted and np.isfinite(z) and z >= SONIC_Z)
        rows.append(
            {
                "year": year,
                "decade": decade_of(year),
                "E": float(e),
                "z_E": None if not np.isfinite(z) else float(z),
                "novelty": None if np.isnan(novelty[i]) else float(novelty[i]),
                "sonic_anomaly": anom,
                "composition_caution": bool(year < SONIC_TRUST_YEAR),
                "top_drivers": drivers,
                "delta_vec": dX[i].tolist(),
            }
        )
    return pd.DataFrame(rows), {
        "E_mean": mu,
        "E_sd": sd,
        "sonic_z": SONIC_Z,
        "roll_win": ROLL_WIN,
        "sonic_trust_year": SONIC_TRUST_YEAR,
    }


def network_channel() -> tuple[pd.DataFrame, dict]:
    """Decade panel from teammate q3 outputs + Katz-mass turnover."""
    cross = pd.read_csv(Q3 / "decade_cross_genre_share.csv")
    cross = cross[(cross["decade"] >= YEAR_MIN) & (cross["decade"] <= YEAR_MAX)].copy()
    cross = cross.sort_values("decade").reset_index(drop=True)

    # Katz-mass share vector per decade from artist indicators (all genres)
    ind = pd.read_csv(Q1 / "artist_network_indicators.csv")
    ind["decade"] = (ind["start"] // 10) * 10
    ind["genre"] = ind["genre"].astype(str).str.strip().str.replace(";", "", regex=False)
    genres = sorted(ind["genre"].dropna().unique())
    mass_rows = []
    for d, sub in ind.groupby("decade"):
        if d < YEAR_MIN or d > YEAR_MAX:
            continue
        w = sub.groupby("genre")["katz_out"].sum()
        w = w.reindex(genres, fill_value=0.0)
        share = w / max(w.sum(), 1e-12)
        mass_rows.append((int(d), share.to_numpy(dtype=float)))
    mass_rows.sort(key=lambda x: x[0])
    mass = {d: v for d, v in mass_rows}

    records = []
    prev = None
    for _, r in cross.iterrows():
        d = int(r["decade"])
        c_cross = float(r["cross_genre_share"])
        c_K = float(r["cross_genre_share_katz_weighted"])
        if prev is None:
            d_cross = 0.0
            d_K = 0.0
            mass_turn = 0.0
        else:
            d_cross = c_cross - prev["c_cross"]
            d_K = c_K - prev["c_K"]
            if prev["decade"] in mass and d in mass:
                mass_turn = float(0.5 * np.linalg.norm(mass[d] - mass[prev["decade"]], ord=1))
            else:
                mass_turn = 0.0
        rec = {
            "decade": d,
            "edges": int(r["edges"]),
            "c_cross": c_cross,
            "c_K": c_K,
            "delta_c_cross": d_cross,
            "delta_c_K": d_K,
            "katz_mass_turnover": mass_turn,
        }
        records.append(rec)
        prev = {"decade": d, "c_cross": c_cross, "c_K": c_K}

    net = pd.DataFrame(records)
    # Network restructuring score from absolute leaps (skip first decade in z)
    leap = 0.5 * net["delta_c_cross"].abs() + 0.5 * net["delta_c_K"].abs()
    # blend in mass turnover (scale to similar magnitude)
    leap = leap + 0.25 * net["katz_mass_turnover"]
    mu, sd = float(leap.iloc[1:].mean()), float(leap.iloc[1:].std(ddof=1))
    zN = (leap - mu) / (sd if sd > 1e-12 else 1.0)
    net["N"] = leap
    net["z_N"] = zN
    net["network_anomaly"] = (net.index > 0) & (zN >= NETWORK_Z)
    meta = {"N_mean": mu, "N_sd": sd, "network_z": NETWORK_Z}
    return net, meta


def dual_classify(sonic: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    net_map = net.set_index("decade")
    rows = []
    # Decade-level synthesis
    decades = sorted(set(sonic["decade"]).intersection(set(net["decade"])))
    for d in decades:
        s_sub = sonic[sonic["decade"] == d]
        n_row = net_map.loc[d]
        sonic_hit = bool(s_sub["sonic_anomaly"].any())
        net_hit = bool(n_row["network_anomaly"])
        if sonic_hit and net_hit:
            label = "strong"
        elif sonic_hit or net_hit:
            label = "weak"
        else:
            label = "evolution"
        peak = s_sub.loc[s_sub["E"].idxmax()] if len(s_sub) else None
        z_num = pd.to_numeric(s_sub["z_E"], errors="coerce")
        # aggregate drivers over anomaly years, else over peak year
        use = s_sub[s_sub["sonic_anomaly"]] if sonic_hit else s_sub.nlargest(1, "E")
        share_acc = {f: 0.0 for f in FEATURE_COLS}
        for _, r in use.iterrows():
            for dr in r["top_drivers"]:
                share_acc[dr["feature"]] += dr["share"]
        drivers_sorted = sorted(share_acc.items(), key=lambda kv: -kv[1])[:TOP_DRIVERS]
        rows.append(
            {
                "decade": int(d),
                "label": label,
                "sonic_anomaly": sonic_hit,
                "network_anomaly": net_hit,
                "max_E": float(s_sub["E"].max()),
                "max_z_E": float(z_num.max()) if z_num.notna().any() else None,
                "anomaly_years": sorted(s_sub.loc[s_sub["sonic_anomaly"], "year"].astype(int).tolist()),
                "peak_year": int(peak["year"]) if peak is not None else None,
                "c_cross": float(n_row["c_cross"]),
                "c_K": float(n_row["c_K"]),
                "N": float(n_row["N"]),
                "z_N": float(n_row["z_N"]),
                "top_drivers": [{"feature": f, "score": float(s)} for f, s in drivers_sorted],
            }
        )
    return pd.DataFrame(rows)


def find_revolutionaries(
    dual: pd.DataFrame,
    sonic: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Name influencers of major change in strong/weak-sonic decades."""
    ind = pd.read_csv(Q1 / "artist_network_indicators.csv")
    ind["artist_id"] = ind["artist_id"].astype(str).str.strip()
    ind["genre"] = ind["genre"].astype(str).str.strip().str.replace(";", "", regex=False)
    feat = pd.read_csv(OUT / "artists_features_with_genre.csv")
    feat["artist_id"] = feat["artist_id"].astype(str).str.strip()
    artists = feat.merge(
        ind[["artist_id", "katz_out", "out_degree", "start", "name", "genre"]].rename(
            columns={"genre": "genre_net", "name": "name_net"}
        ),
        on="artist_id",
        how="inner",
    )
    if "main_genre" in artists.columns:
        artists["genre"] = (
            artists["main_genre"].astype(str).str.strip().str.replace(";", "", regex=False)
        )
    else:
        artists["genre"] = artists["genre_net"]
    artists["name"] = artists["name_net"]
    X_art = scaler.transform(artists[FEATURE_COLS].to_numpy(dtype=float))

    edges = pd.read_csv(
        DATA / "influence_data.csv",
        dtype={"influencer_id": "string", "follower_id": "string"},
        encoding="utf-8",
    )
    edges["influencer_id"] = edges["influencer_id"].str.strip()
    edges["follower_id"] = edges["follower_id"].str.strip()
    edges["follower_active_start"] = pd.to_numeric(
        edges["follower_active_start"], errors="coerce"
    )

    year_raw = pd.read_csv(DATA / "data_by_year.csv")
    year_raw = year_raw.sort_values("year")
    X_year = scaler.transform(year_raw[FEATURE_COLS].to_numpy(dtype=float))
    year_to_x = dict(zip(year_raw["year"].astype(int), X_year))

    out_rows = []
    targets = dual[dual["label"].isin(["strong", "weak"])]
    for _, win in targets.iterrows():
        d = int(win["decade"])
        # mean sonic direction over anomaly years (or peak year in the decade)
        years = list(win["anomaly_years"] or [])
        if not years and win["peak_year"] is not None:
            years = [int(win["peak_year"])]
        deltas = []
        for y in years:
            sub = sonic[sonic["year"] == int(y)]
            if len(sub):
                deltas.append(np.asarray(sub.iloc[0]["delta_vec"], dtype=float))
        if not deltas:
            continue
        direction = np.mean(deltas, axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        # previous-decade sonic mean as baseline style
        prev_years = [y for y in range(d - 10, d) if y in year_to_x]
        if not prev_years:
            prev_years = [y for y in range(d - 20, d) if y in year_to_x]
        x_prev = np.mean([year_to_x[y] for y in prev_years], axis=0)

        # influencers active before/at window who send edges into d or d+10 followers
        foll_mask = edges["follower_active_start"].isin([d, d + 10])
        into = edges.loc[foll_mask]
        follow_counts = into.groupby("influencer_id").size().rename("followers_in_window")

        cand = artists[(artists["start"] <= d) & (artists["start"] >= d - 30)].copy()
        cand = cand.merge(follow_counts, left_on="artist_id", right_index=True, how="left")
        cand["followers_in_window"] = cand["followers_in_window"].fillna(0).astype(int)
        cand = cand[cand["followers_in_window"] >= MIN_FOLLOWERS]
        if cand.empty:
            continue

        idx = cand.index.to_numpy()
        # map to X_art rows via positional alignment on artists index
        pos = artists.index.get_indexer(idx)
        Xi = X_art[pos]
        disp = Xi - x_prev
        disp_norm = np.linalg.norm(disp, axis=1) + 1e-12
        align = (disp * direction).sum(axis=1) / disp_norm
        # require positive alignment with the sonic leap
        cand = cand.copy()
        cand["displacement"] = disp_norm
        cand["alignment"] = align
        cand = cand[cand["alignment"] > 0]
        if cand.empty:
            continue

        # score: rank-normalize Katz, followers, alignment
        def _rank01(s: pd.Series) -> pd.Series:
            return s.rank(method="average", pct=True)

        cand["score"] = (
            0.45 * _rank01(cand["katz_out"])
            + 0.30 * _rank01(cand["followers_in_window"])
            + 0.25 * _rank01(cand["alignment"])
        )
        top = cand.nlargest(TOP_REVOLUTIONARIES, "score")
        for rank, (_, a) in enumerate(top.iterrows(), start=1):
            out_rows.append(
                {
                    "decade": d,
                    "window_label": win["label"],
                    "rank": rank,
                    "artist_id": a["artist_id"],
                    "name": a["name"],
                    "genre": a["genre"],
                    "start": int(a["start"]),
                    "K_i": float(a["katz_out"]),
                    "out_degree": int(a["out_degree"]),
                    "followers_in_window": int(a["followers_in_window"]),
                    "alignment": float(a["alignment"]),
                    "displacement": float(a["displacement"]),
                    "score": float(a["score"]),
                }
            )
    return pd.DataFrame(out_rows)


def plot_dual(sonic: pd.DataFrame, net: pd.DataFrame, dual: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=False,
                             gridspec_kw={"height_ratios": [1.15, 1.0]})

    ax = axes[0]
    ax.plot(sonic["year"], sonic["E"], color="#1f4e79", lw=1.6, label=r"$E(t)=\|x(t)-x(t-1)\|_2$")
    ax.axvline(SONIC_TRUST_YEAR, color="gray", ls=":", lw=1, label=f"trust cutoff {SONIC_TRUST_YEAR}")
    anom = sonic[sonic["sonic_anomaly"]]
    ax.scatter(anom["year"], anom["E"], color="#c0392b", s=28, zorder=3,
               label=fr"local sonic anomaly (rolling $z\geq{SONIC_Z}$)")
    for _, r in dual[dual["label"] == "strong"].iterrows():
        ax.axvspan(r["decade"], r["decade"] + 9, color="#f4d03f", alpha=0.25, lw=0)
    for _, r in dual[(dual["label"] == "weak") & dual["sonic_anomaly"]].iterrows():
        ax.axvspan(r["decade"], r["decade"] + 9, color="#aed6f1", alpha=0.25, lw=0)
    ax.set_ylabel("Sonic leap size")
    ax.set_title("Dual-channel revolution scan")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_xlim(YEAR_MIN, YEAR_MAX)

    ax2 = axes[1]
    ax2.plot(net["decade"], net["c_cross"], "o-", color="#1f4e79", lw=1.5, ms=5,
             label=r"$c_{\mathrm{cross}}(t)$")
    ax2.plot(net["decade"], net["c_K"], "s--", color="#117a65", lw=1.5, ms=5,
             label=r"$c_K(t)$ (Katz-weighted)")
    for _, r in net[net["network_anomaly"]].iterrows():
        ax2.axvspan(r["decade"] - 2, r["decade"] + 2, color="#e74c3c", alpha=0.12, lw=0)
    ax2.set_xlabel("Follower active-start decade $t$")
    ax2.set_ylabel("Cross-genre edge share")
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    ax2.set_xlim(YEAR_MIN, YEAR_MAX)

    # legend notes for spans
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#f4d03f", alpha=0.35, label="strong (sonic+network)"),
        Patch(facecolor="#aed6f1", alpha=0.35, label="weak (sonic-only)"),
        Patch(facecolor="#e74c3c", alpha=0.15, label="network anomaly decade"),
    ]
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + handles,
                   loc="upper right", fontsize=7.5, frameon=False)

    fig.tight_layout()
    out = FIG / "fig_t5_dual_channel_revolutions.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "fig_t5_dual_channel_revolutions.png", dpi=160, bbox_inches="tight")
    fig.savefig(REP_FIG / "fig_t5_dual_channel_revolutions.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def plot_drivers(dual: pd.DataFrame) -> Path | None:
    strong = dual[dual["label"].isin(["strong", "weak"]) & dual["sonic_anomaly"]]
    if strong.empty:
        return None
    # pick up to 4 windows
    strong = strong.sort_values(["label", "max_z_E"], ascending=[True, False]).head(4)
    fig, axes = plt.subplots(1, len(strong), figsize=(2.6 * len(strong), 3.4), sharey=True)
    if len(strong) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, strong.iterrows()):
        feats = [d["feature"] for d in r["top_drivers"]]
        scores = [d["score"] for d in r["top_drivers"]]
        ax.barh(feats[::-1], scores[::-1], color="#1f4e79")
        ax.set_title(f"{int(r['decade'])}s ({r['label']})", fontsize=9)
        ax.set_xlabel("driver mass")
    fig.suptitle("Features driving sonic leaps in flagged decades", fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG / "fig_t5_revolution_drivers.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "fig_t5_revolution_drivers.png", dpi=160, bbox_inches="tight")
    fig.savefig(REP_FIG / "fig_t5_revolution_drivers.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    raw, X, scaler = load_sonic()
    years = raw["year"].to_numpy(dtype=int)
    sonic, sonic_meta = sonic_channel(years, X)
    net, net_meta = network_channel()
    dual = dual_classify(sonic, net)
    revs = find_revolutionaries(dual, sonic, scaler)

    # flatten sonic for CSV (drop heavy fields)
    sonic_csv = sonic.drop(columns=["top_drivers", "delta_vec"]).copy()
    sonic_csv.to_csv(OUT / "t5_sonic_leaps.csv", index=False)
    net.to_csv(OUT / "t5_network_panel.csv", index=False)

    dual_flat = dual.copy()
    dual_flat["anomaly_years"] = dual_flat["anomaly_years"].apply(
        lambda xs: ";".join(str(x) for x in xs)
    )
    dual_flat["top_drivers"] = dual_flat["top_drivers"].apply(
        lambda ds: ";".join(f"{d['feature']}:{d['score']:.3f}" for d in ds)
    )
    dual_flat.to_csv(OUT / "t5_dual_decade_labels.csv", index=False)
    if len(revs):
        revs.to_csv(OUT / "t5_revolutionaries.csv", index=False)

    summary = {
        "definition": {
            "evolution": "gradual drift of yearly z-feature state x(t) and slow change in c_cross, c_K",
            "revolution": "anomalous sonic leap E(t) relative to baseline; strong if enclosing decade also has elevated network score N(t)",
            "citations": [
                "Mauch et al. 2015 Royal Society Open Science (punctuated change / Foote novelty)",
                "Savage 2019 Humanities & Social Sciences Communications (cultural evolution as variant-frequency change)",
            ],
        },
        "params": {
            "year_window": [YEAR_MIN, YEAR_MAX],
            "features": FEATURE_COLS,
            **sonic_meta,
            **net_meta,
        },
        "sonic_anomaly_years": sonic.loc[sonic["sonic_anomaly"], "year"].astype(int).tolist(),
        "decade_labels": dual[["decade", "label", "sonic_anomaly", "network_anomaly",
                               "max_E", "max_z_E", "c_cross", "c_K", "N", "z_N"]].to_dict(
            orient="records"
        ),
        "strong_decades": dual.loc[dual["label"] == "strong", "decade"].astype(int).tolist(),
        "weak_decades": dual.loc[dual["label"] == "weak", "decade"].astype(int).tolist(),
        "top_revolutionaries_by_decade": {},
    }
    if len(revs):
        for d, sub in revs.groupby("decade"):
            summary["top_revolutionaries_by_decade"][str(int(d))] = sub.head(5)[
                ["rank", "name", "genre", "start", "K_i", "followers_in_window",
                 "alignment", "score"]
            ].to_dict(orient="records")

    # driver table for strong/weak sonic windows
    driver_rows = []
    for _, r in dual[dual["sonic_anomaly"]].iterrows():
        for d in r["top_drivers"]:
            driver_rows.append(
                {"decade": int(r["decade"]), "label": r["label"],
                 "feature": d["feature"], "score": d["score"]}
            )
    if driver_rows:
        pd.DataFrame(driver_rows).to_csv(OUT / "t5_driver_features.csv", index=False)

    with open(OUT / "t5_revolution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig1 = plot_dual(sonic, net, dual)
    fig2 = plot_drivers(dual)

    print("=== T5 dual-channel summary ===")
    print(dual[["decade", "label", "max_z_E", "z_N", "c_cross", "c_K"]].to_string(index=False))
    print("sonic anomaly years:", summary["sonic_anomaly_years"])
    print("strong:", summary["strong_decades"], "weak:", summary["weak_decades"])
    print("wrote", fig1)
    if fig2:
        print("wrote", fig2)
    if len(revs):
        print(revs.groupby("decade")["name"].apply(lambda s: list(s.head(3))))


if __name__ == "__main__":
    main()
