"""Q3 / Section 7.4 -- How genre influence changes over time.

Aggregation layer on top of Model I (no new network model):
  F[g,h]  = number of reported edges from genre g artists to genre h artists
  P       = row-normalized F  (export profile: where g's influence lands)
  Q       = column-normalized F (source profile: where h's influence comes from)
  F(t)    = the same group-by restricted to edges whose FOLLOWER started in decade t
  K_g(t)  = share of decade t's outgoing-Katz mass held by genre g's new artists

Correctness gates:
  sum(F) == |E|                      (conservation)
  F['Jazz','Pop/Rock'] == 409        (must match Model I section 5.6)

Outputs -> q3_outputs/ (JSON, CSVs, figures). Not uploaded to GitHub.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ProblemC_Music_data" / "influence_data.csv"
KATZ = ROOT / "q1_outputs" / "artist_network_indicators.csv"
OUT = ROOT / "q3_outputs"
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DECADE_MIN_EDGES = 50      # decade needs this many edges to enter the time plots
ROW_MIN_EDGES = 20         # P(t)[g,h] plotted only if genre-decade row has enough edges
MAIN_WINDOW = (1930, 2010) # follower active-start decades shown in figures

SELECTED_FLOWS = [
    ("Jazz", "Pop/Rock"),
    ("Blues", "R&B"),
    ("Electronic", "Pop/Rock"),
    ("Folk", "Country"),
    ("Blues", "Country"),
]


def load_edges() -> pd.DataFrame:
    """Same cleaning pipeline as q1_analysis.load_data (Model I, section 4.2)."""
    df = pd.read_csv(DATA, dtype={"influencer_id": "string", "follower_id": "string"},
                     encoding="utf-8")
    for col in ["influencer_id", "follower_id"]:
        df[col] = df[col].str.strip()
    for col in ["influencer_main_genre", "follower_main_genre"]:
        df[col] = df[col].astype("string").str.strip().str.replace(";", "", regex=False)
    for col in ["influencer_active_start", "follower_active_start"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["influencer_id", "follower_id"]).copy()
    df = df.drop_duplicates(["influencer_id", "follower_id"]).reset_index(drop=True)
    return df


def block_matrix(edges: pd.DataFrame, genres: list[str],
                 weight: pd.Series | None = None) -> pd.DataFrame:
    """Group edges by (influencer genre, follower genre). weight=None -> counts."""
    key = pd.crosstab(edges["influencer_main_genre"], edges["follower_main_genre"])
    key = key.reindex(index=genres, columns=genres, fill_value=0)
    if weight is None:
        return key.astype(int)
    w = pd.DataFrame({"g": edges["influencer_main_genre"].values,
                      "h": edges["follower_main_genre"].values,
                      "w": weight.values})
    agg = w.groupby(["g", "h"])["w"].sum().unstack(fill_value=0.0)
    return agg.reindex(index=genres, columns=genres, fill_value=0.0)


def main() -> None:
    edges = load_edges()
    genres = sorted(set(edges["influencer_main_genre"].dropna())
                    | set(edges["follower_main_genre"].dropna()))
    n_genres = len(genres)

    # ---------- static flow matrix + gates ----------
    F = block_matrix(edges, genres)
    total = int(F.values.sum())
    conservation_ok = total == len(edges)
    jazz_poprock = int(F.loc["Jazz", "Pop/Rock"])
    gate = {"conservation_sumF_vs_edges": {"sum_F": total, "n_edges": len(edges),
                                           "ok": conservation_ok},
            "jazz_to_poprock": {"value": jazz_poprock, "expected_5_6": 409,
                                "ok": jazz_poprock == 409}}
    print("[gate] conservation:", gate["conservation_sumF_vs_edges"])
    print("[gate] Jazz->Pop/Rock:", gate["jazz_to_poprock"])
    assert conservation_ok and jazz_poprock == 409, "aggregation gates failed"

    P = F.div(F.sum(axis=1), axis=0)
    Q = F.div(F.sum(axis=0), axis=1)
    F.to_csv(OUT / "genre_flow_matrix_F.csv")
    P.to_csv(OUT / "genre_export_profile_P.csv")
    Q.to_csv(OUT / "genre_source_profile_Q.csv")

    same_share = float(np.trace(F.values) / total)
    top_flows = (F.stack().sort_values(ascending=False)
                 .head(15).rename("edges").reset_index()
                 .rename(columns={F.index.name or "level_0": "from",
                                  F.columns.name or "level_1": "to"}))
    print(f"same-genre edge share: {same_share:.3f}")

    # ---------- time dimension: follower active-start decade ----------
    t = edges["follower_active_start"]
    tknown = edges[t.notna()].copy()
    tknown["decade"] = tknown["follower_active_start"].astype(int)

    katz = pd.read_csv(KATZ, usecols=["artist_id", "genre", "start", "katz_out"],
                       dtype={"artist_id": "string"})
    katz["artist_id"] = katz["artist_id"].str.strip()
    katz_map = katz.set_index("artist_id")["katz_out"]

    # follower-Katz weights for the Katz-weighted robustness variant
    fw = tknown["follower_id"].str.strip().map(katz_map)
    n_matched = int(fw.notna().sum())
    fw = fw.fillna(1.0)
    print(f"followers matched to Katz table: {n_matched}/{len(tknown)}")

    decade_rows = []
    Ft = {}
    Fkatz_t = {}

    for dec, sub in tknown.groupby("decade"):
        if not (MAIN_WINDOW[0] <= dec <= MAIN_WINDOW[1]):
            continue
        Fd = block_matrix(sub, genres)
        Fkd = block_matrix(sub, genres, weight=fw.loc[sub.index])
        n = int(Fd.values.sum())
        if n < DECADE_MIN_EDGES:
            continue
        row = {"decade": dec, "edges": n,
               "same_genre_share": float(np.trace(Fd.values) / n),
               "cross_genre_share": 1.0 - float(np.trace(Fd.values) / n)}
        wk = float(np.nan_to_num(Fkd.values.sum()))
        row["cross_genre_share_katz_weighted"] = (
            1.0 - float(np.trace(Fkd.values)) / wk) if wk > 0 else np.nan
        decade_rows.append(row)
        Ft[dec] = Fd
        Fkatz_t[dec] = Fkd

    dec_df = pd.DataFrame(decade_rows).set_index("decade")
    dec_df.to_csv(OUT / "decade_cross_genre_share.csv")
    print(dec_df[["edges", "cross_genre_share", "cross_genre_share_katz_weighted"]])

    # selected pair export share P[g,h](t)
    pair_rows = []
    for dec, Fd in Ft.items():
        rowtot = Fd.sum(axis=1)
        for g, h in SELECTED_FLOWS:
            if rowtot[g] >= ROW_MIN_EDGES and rowtot[g] > 0:
                pair_rows.append({"decade": dec, "from": g, "to": h,
                                  "share": float(Fd.loc[g, h] / rowtot[g]),
                                  "row_edges": int(rowtot[g])})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT / "selected_pair_export_share.csv", index=False)

    # ---------- Katz mass by (genre, own start decade) ----------
    kstart = katz.dropna(subset=["start", "katz_out"]).copy()
    kstart["decade"] = kstart["start"].astype(int)
    kstart = kstart[(kstart["decade"] >= MAIN_WINDOW[0])
                    & (kstart["decade"] <= MAIN_WINDOW[1])]
    mass_rows = []
    for dec, sub in kstart.groupby("decade"):
        tot = float(sub["katz_out"].sum())
        if tot <= 0:
            continue
        gm = sub.groupby("genre")["katz_out"].sum()
        top = gm.sort_values(ascending=False).head(5)
        mass_rows.append({"decade": dec, "artists": int(len(sub)),
                          "top5_genres_by_katz_mass": {k: float(v / tot)
                                                       for k, v in top.items()}})
    mass_df = pd.DataFrame(mass_rows).set_index("decade")
    mass_df.to_json(OUT / "katz_mass_by_genre_decade.json", indent=2,
                    orient="index", force_ascii=False)

    # trend of a few genres' Katz mass share
    mass_series = {}
    for g in ["Pop/Rock", "Electronic", "Jazz", "Hip Hop", "Electronic/Dance"]:
        vals = {}
        for dec, sub in kstart.groupby("decade"):
            tot = float(sub["katz_out"].sum())
            if tot > 0:
                vals[dec] = float(sub.loc[sub["genre"] == g, "katz_out"].sum() / tot)
        if any(v > 0 for v in vals.values()):
            mass_series[g] = vals

    results = {
        "genres": n_genres,
        "n_edges": len(edges),
        "same_genre_share_overall": same_share,
        "gates": gate,
        "top_flows": top_flows.to_dict("records"),
        "decade_cross_genre_share": dec_df.reset_index().to_dict("records"),
        "selected_pair_export_share": pair_df.to_dict("records"),
        "katz_mass_top5_by_decade": mass_df.reset_index().to_dict("records"),
        "katz_mass_share_series": mass_series,
    }
    (OUT / "q3_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ---------- figure 9 (compact companion to fig 8) ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    ax = axes[0]
    ax.plot(dec_df.index, dec_df["cross_genre_share"], "o-", color="#d85a30",
            label="cross-genre edge share")
    ax.plot(dec_df.index, dec_df["same_genre_share"], "s-", color="#5f5e5a",
            label="same-genre edge share")
    ax.plot(dec_df.index, dec_df["cross_genre_share_katz_weighted"], "^--",
            color="#378add", alpha=0.9,
            label="cross-genre share (Katz-weighted)")
    ax.axhline(same_share, color="#5f5e5a", ls=":", lw=1)
    ax.set_xlabel("Follower active-start decade")
    ax.set_ylabel("Share of reported edges")
    ax.set_title("(a) Cross-genre influence over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    palette = ["#1d9e75", "#d85a30", "#378add", "#7f77dd", "#ba7517"]
    for (g, h), c in zip(SELECTED_FLOWS, palette):
        sub = pair_df[(pair_df["from"] == g) & (pair_df["to"] == h)]
        ax.plot(sub["decade"], sub["share"], "o-", ms=4, color=c, label=f"{g} -> {h}")
    ax.set_xlabel("Follower active-start decade")
    ax.set_ylabel(r"Export share $P_{g,h}(t)$")
    ax.set_title("(b) Selected genre flows over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Genre influence flows by follower cohort decade", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig9_genre_influence_time.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "fig9_genre_influence_time.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 10: full Q heatmap (for section 7.5) ----------
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    Qt = Q.copy()
    Qt.index = [i.replace("/", "/\n") for i in Qt.index]
    Qt.columns = [c.replace("/", "/\n") for c in Qt.columns]
    im = ax.imshow(Qt.values, cmap="YlGnBu", vmin=0)
    ax.set_xticks(range(n_genres), Qt.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(n_genres), Qt.index, fontsize=7)
    ax.set_xlabel("Influenced genre (follower)")
    ax.set_ylabel("Influencing genre (source)")
    ax.set_title("Source profile Q[g,h]: share of h's absorbed influence\ncoming from genre g (column-normalized)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Share of inflow")
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_Q_source_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("saved:", sorted(p.name for p in OUT.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
