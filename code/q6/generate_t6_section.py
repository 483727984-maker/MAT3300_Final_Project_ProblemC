"""Build the Task-6 (jazz evolution) subsection of Model IV.

Reads q6_results.json and q6_artists.csv, writes model4_t6.tex in the
joint report's notation, and stages the three figures for
report/figures/ under names fig_t6_*.pdf.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "q6_results.json"
ARTISTS = ROOT / "q6_artists.csv"
FIG = ROOT / "figures"
OUT_TEX = ROOT / "model4_t6.tex"
OUT_FIG = ROOT / "t6_figures"

REPL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
}


def esc(text) -> str:
    return "".join(REPL.get(ch, ch) for ch in str(text))


def main() -> None:
    r = json.loads(RESULT.read_text(encoding="utf-8"))
    d, tl, ad = r["data"], r["timeline"], r["adoption"]
    ind, rob = r["indicators"], r["robustness"]
    corr = ind["spearman"]
    art = pd.read_csv(ARTISTS).set_index("artist")

    def v(name, col, fmt=".2f"):
        return format(float(art.loc[name, col]), fmt)

    def K(name):
        return v(name, "katz", ".1f")

    def A(name):
        return v(name, "anticipation", "+.2f")

    def H(name):
        return v(name, "H_shrunk", "+.2f")

    def B(name):
        return str(int(art.loc[name, "boundary"]))

    def D(dec, col, fmt=".2f"):
        return format(float(tl["decade_raw"][str(dec)][col]), fmt)

    ep = {e["year"]: e["velocity"] for e in tl["episodes"]}
    ep_years = [e["year"] for e in tl["episodes"]]
    ep_txt = ", ".join(str(y) for y in ep_years[:-1]) + " and " + str(ep_years[-1])
    changes = tl["episode_feature_change"]

    def ep_items(year, n=3):
        row = changes[str(year)]
        top = sorted(row.items(), key=lambda kv: -abs(kv[1]))[:n]
        return ", ".join(f"{f.replace('_ms', '')} ${val:+.2f}$" for f, val in top)

    top_rows = "\n".join(
        f"{i} & {esc(x['artist'])} & {x['start']}s & {x['novelty']:.2f} & "
        f"{x['anticipation']:+.2f} & {x['H_shrunk']:+.2f} & {x['followers']} & "
        f"{x['katz']:.2f} & {x['boundary']} \\\\"
        for i, x in enumerate(ind["top_dynamic"][:12], 1)
    )

    load1 = ", ".join(f"{k.replace('_ms', '')} ${val:+.2f}$" for k, val in
                      sorted(tl["pca_loadings_pc1"].items(), key=lambda kv: -abs(kv[1]))[:3])
    load2 = ", ".join(f"{k.replace('_ms', '')} ${val:+.2f}$" for k, val in
                      sorted(tl["pca_loadings_pc2"].items(), key=lambda kv: -abs(kv[1]))[:3])

    repl = {
        "J_A": f"{d['jazz_artists']}",
        "JNET": f"{d['jazz_network_nodes']}",
        "JJ": f"{d['jazz_jazz_edges']:,}",
        "J_SONGS": f"{d['jazz_songs']:,}",
        "J_SONG_ARTISTS": f"{d['jazz_song_artists']}",
        "J_USED": f"{d['jazz_artists_used']}",
        "J_POP": f"{d['jazz_to_pop_rock_edges']}",
        "EDGES": f"{d['network_edges']:,}",
        "REV": f"{d['reversed_edges']}",
        "TL_MIN": f"{d['timeline_min']}",
        "TL_MAX": f"{d['timeline_max']}",
        "PC1": f"{100 * tl['pca_explained'][0]:.1f}",
        "PC2": f"{100 * tl['pca_explained'][1]:.1f}",
        "LOAD1": load1,
        "LOAD2": load2,
        "EP_LIST": ep_txt,
        "EP49": ep_items(1949),
        "EP54": ep_items(1954),
        "EP69": ep_items(1969),
        "EP80": ep_items(1980),
        "D1930AC": D(1930, "acousticness"),
        "D1930DUR": D(1930, "duration_min", ".1f"),
        "D1930TEMPO": D(1930, "tempo", ".0f"),
        "D1930VAL": D(1930, "valence"),
        "D1970AC": D(1970, "acousticness"),
        "D1970EN": D(1970, "energy"),
        "D1970DUR": D(1970, "duration_min", ".1f"),
        "D1980AC": D(1980, "acousticness"),
        "D1990AC": D(1990, "acousticness"),
        "D2000AC": D(2000, "acousticness"),
        "D1960VAL": D(1960, "valence"),
        "D1960AC": D(1960, "acousticness"),
        "N_PAIRS": f"{ad['pairs']:,}",
        "N_ANT": f"{ind['with_anticipation']}",
        "N_FOL": f"{ind['with_followers']}",
        "N_ELIG": f"{ind['eligible']}",
        "ADOPT_OBS": f"{ad['mean_alignment']:.3f}",
        "ADOPT_NULL": f"{ad['null_mean']:.3f}",
        "ADOPT_SD": f"{ad['null_sd']:.3f}",
        "ADOPT_LO": f"{ad['bootstrap_ci95'][0]:.3f}",
        "ADOPT_HI": f"{ad['bootstrap_ci95'][1]:.3f}",
        "ADOPT_P": ("$p<0.001$" if ad["p_value"] < 0.001 else f"$p={ad['p_value']:.3f}$"),
        "ADOPT_POS": f"{100 * ad['positive_share']:.1f}\\%",
        "ADOPT_NULLPOS": f"{100 * ad['null_positive_share']:.1f}\\%",
        "CORR_NA": f"{corr['novelty']['anticipation']:+.2f}",
        "CORR_NK": f"{corr['novelty']['katz']:+.2f}",
        "CORR_AK": f"{corr['anticipation']['katz']:+.2f}",
        "CORR_HK": f"{corr['H_shrunk']['katz']:+.2f}",
        "CORR_NF": f"{corr['novelty']['future_followers']:+.2f}",
        "CORR_AF": f"{corr['anticipation']['future_followers']:+.2f}",
        "CORR_HF": f"{corr['H_shrunk']['future_followers']:+.2f}",
        "TOP_ROWS": top_rows,
        "WEIGHT_RHO": f"{min(ind['weight_robustness_spearman'].values()):.3f}--"
                      f"{max(ind['weight_robustness_spearman'].values()):.3f}",
        "SHRINK": f"{ind['shrinkage_spearman']:.3f}",
        "LOO_N": f"{rob['loo_spearman']['novelty']:.3f}",
        "LOO_A": f"{rob['loo_spearman']['anticipation']:.3f}",
        "H5": f"{rob['horizon5_spearman']:.3f}",
        "LIV_N": f"{rob['liveness_included_spearman']['novelty']:.3f}",
        "LIV_A": f"{rob['liveness_included_spearman']['anticipation']:.3f}",
        "LIV_SIZE": f"{rob['liveness_artifact']['cluster_size']}",
        "LIV_Z": f"{rob['liveness_artifact']['liveness_z']:+.1f}",
        "A_YOUNG": A("Lester Young"),
        "K_PARKER": K("Charlie Parker"),
        "K_YOUNG": K("Lester Young"),
        "K_MILES": K("Miles Davis"),
        "K_COLTRAINE": K("John Coltrane"),
        "K_ORNETTE": K("Ornette Coleman"),
        "K_CECIL": K("Cecil Taylor"),
        "A_PARKER": A("Charlie Parker"),
        "H_PARKER": H("Charlie Parker"),
        "A_POWELL": A("Bud Powell"),
        "H_GILLESPIE": H("Dizzy Gillespie"),
        "A_BRUBECK": A("Dave Brubeck"),
        "H_BRUBECK": H("Dave Brubeck"),
        "A_FARLOW": A("Tal Farlow"),
        "A_SILVER": A("Horace Silver"),
        "H_SILVER": H("Horace Silver"),
        "A_BROWN": A("Clifford Brown"),
        "A_ORNETTE": A("Ornette Coleman"),
        "H_ORNETTE": H("Ornette Coleman"),
        "A_CECIL": A("Cecil Taylor"),
        "A_HANCOCK": A("Herbie Hancock"),
        "H_HANCOCK": H("Herbie Hancock"),
        "A_HILL": A("Andrew Hill"),
        "H_HILL": H("Andrew Hill"),
        "A_TONY": A("Tony Williams"),
        "A_MCLAUGHLIN": A("John McLaughlin"),
        "H_MCLAUGHLIN": H("John McLaughlin"),
        "A_DUKE": A("George Duke"),
        "H_DUKE": H("George Duke"),
        "A_GROVER": A("Grover Washington, Jr."),
        "H_GROVER": H("Grover Washington, Jr."),
        "A_JAMES": A("Bob James"),
        "H_JAMES": H("Bob James"),
        "A_SANBORN": A("David Sanborn"),
        "H_SANBORN": H("David Sanborn"),
        "A_MARSALIS": A("Wynton Marsalis"),
        "B_MILES": B("Miles Davis"),
        "B_NAT": B("Nat King Cole"),
        "B_COLTRAINE": B("John Coltrane"),
        "B_QUINCY": B("Quincy Jones"),
        "B_SUNRA": B("Sun Ra"),
        "B_CALLOWAY": B("Cab Calloway"),
        "B_ORNETTE": B("Ornette Coleman"),
        "B_MCLAUGHLIN": B("John McLaughlin"),
        "B_WEATHER": B("Weather Report"),
    }

    text = TEMPLATE
    for k, val in repl.items():
        text = text.replace(f"@@{k}@@", str(val))
    OUT_TEX.write_text(text, encoding="utf-8")

    OUT_FIG.mkdir(exist_ok=True)
    for src, dst in [("fig1_genre_timeline.pdf", "fig_t6_timeline.pdf"),
                     ("fig2_revolutions.pdf", "fig_t6_episodes.pdf"),
                     ("fig5_influencers.pdf", "fig_t6_influencers.pdf")]:
        shutil.copyfile(FIG / src, OUT_FIG / dst)
    print(f"Wrote {OUT_TEX}")
    print(f"Staged figures in {OUT_FIG}")


TEMPLATE = r"""
\subsection{The Jazz Influence Process over Time (Task~6)}\label{subsec:t6-jazz}

Task~6 asks for the influence process of musical evolution over time in one genre, for indicators
that reveal the dynamic influencers, and for an explanation of how the genre and its artists
changed. We study jazz. The choice continues \cref{subsec:q1-jazz}: the Jazz-induced subnetwork
there (@@JNET@@ artists, @@JJ@@ edges) is the genre subnetwork examined in Model~I, and it spans the
whole record. It also gives the revolution windows of T5 (\cref{subsec:t5-revolutions}) a
genre-level test: the question is not only whether a decade is a revolution in the all-genre
catalogue, but which jazz artists moved, and whether their followers moved with them.

\paragraph{A release-year timeline.}
We use the same @@EDGES@@ edges as Model~I under Assumption~A3, including the @@REV@@ time-order
reversals. @@J_A@@ artists carry the Jazz label; @@JNET@@ of them form the both-endpoints-jazz
subnetwork of \cref{subsec:q1-jazz}. The Spotify files contain @@J_SONGS@@ jazz songs by
@@J_SONG_ARTISTS@@ of these artists; @@J_USED@@ artists have at least three songs and can be given
a stable feature vector. Three clocks are kept separate: a song's release year \(y\), an artist's
active-start decade \(t_i\), and a follower's active-start decade (the cohort clock of
\cref{subsec:q1-cohorts}). The first describes the genre's sound, the second places an artist's
generation, and the third is used to validate transmission.

Model~II represents an artist by eleven Spotify coordinates (\cref{sec:model-similarity}); we use
its nine \emph{sound} coordinates, dropping \texttt{liveness} and \texttt{popularity}. Liveness
separates live from studio recordings --- a \(k\)-means run that includes it sets apart a
@@LIV_SIZE@@-song cluster almost entirely by liveness ($@@LIV_Z@@$ standard deviations) --- and
popularity is not a property of the sound. All coordinates are \(z\)-scored across the artists of
\texttt{data\_by\_artist.csv}. This is a deliberate difference from T5, which uses all eleven
yearly aggregates and finds liveness the dominant 1970s driver; the robustness paragraph below
shows that adding liveness back changes the indicator rankings only marginally.

For each year with at least 20 jazz songs we compute the mean \(z\)-vector
\(\mathbf{x}_J(y)\), smooth it with a centred three-year window, and interpolate between years.
This leaves a reliable series from @@TL_MIN@@ to @@TL_MAX@@; earlier and later years are too sparse
for a yearly curve, and the 2010s (22 songs) are excluded from the decade summaries. A principal
component analysis of the yearly centroids gives PC1 @@PC1@@\% (acoustic and short recordings at
the positive end, long and electric ones at the negative end; loadings @@LOAD1@@) and PC2
@@PC2@@\% (loadings @@LOAD2@@). \cref{fig:t6-timeline} shows the decade trajectory and the
standardized decade centroids.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_t6_timeline.pdf}
\caption{Jazz sonic timeline. (a) Decade centroids in the first two principal components of the
yearly release-year centroids; arrows follow the decades. (b) Standardized decade centroids of the
nine sound features.}
\label{fig:t6-timeline}
\end{figure}

The trajectory is a loop. The 1930s centroid is the most acoustic and the shortest of the series
(acousticness @@D1930AC@@, duration @@D1930DUR@@ min, tempo @@D1930TEMPO@@ BPM, valence
@@D1930VAL@@); the 1950s--60s lengthen and darken; the 1970s is the most electric, highest-energy
and longest decade (acousticness @@D1970AC@@, energy @@D1970EN@@, duration @@D1970DUR@@ min); and
the 1980s--2000s recover part way towards an acoustic sound (@@D1980AC@@, @@D1990AC@@,
@@D2000AC@@) without returning to the 1930s values.

\paragraph{Four jazz-specific episodes.}
The speed \(\|\mathbf{x}_J(y+2)-\mathbf{x}_J(y-2)\|\) has four well-separated local maxima, at
@@EP_LIST@@ (\cref{fig:t6-episodes}). Their feature decompositions separate a stylistic break, a
format change, an electric turn and a crossover reversal:
\begin{itemize}[leftmargin=1.2em,itemsep=0.15em]
\item 1949: @@EP49@@ --- the small-group bebop break with swing;
\item 1954: @@EP54@@ --- consistent with the LP replacing the 78-rpm single;
\item 1969: @@EP69@@ --- fusion and electric instruments;
\item 1980: @@EP80@@ --- crossover, smooth jazz and the start of the acoustic revival.
\end{itemize}
The 1969 episode is the jazz face of T5's strong 1970s window (\cref{fig:t5-dual}), and the 1954
format shift corresponds to T5's network-only weak 1950s window. T5 reaches the 1970s through
liveness (a production shift); the jazz-only space reaches it through acousticness, energy and
duration (a style shift). The two readings are complementary.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_t6_episodes.pdf}
\caption{Four jazz-specific episodes. (a) Speed of the jazz centroid; shaded bands mark the
episodes. (b) Change in each feature around each episode, $\mathbf{x}_J(y+2)-\mathbf{x}_J(y-2)$ in
\(z\)-units.}
\label{fig:t6-episodes}
\end{figure}

\paragraph{Indicators of dynamic influence.}
For artist \(i\) with \(n_i\ge3\) songs, let
\begin{equation}\label{eq:t6-r}
r_i=\frac{1}{n_i}\sum_{s\in i}\bigl(\tilde{\mathbf{x}}_s-\mathbf{x}_J(y_s)\bigr)
\end{equation}
be the \emph{time-adjusted deviation}: the average of the song vectors, each centred on the jazz
sound of its release year. The four indicators are:
\begin{align}
N_i&=\|r_i\|_\Sigma, \label{eq:t6-N}\\
A_i&=\cos(r_i,\Delta_i),\qquad \Delta_i=\mathbf{x}_J(t_i+10)-\mathbf{x}_J(t_i),
\label{eq:t6-A}\\
H_i&=\frac{1}{n_i^H}\sum_{j:\,i\to j,\ t_j>t_i}\cos(r_j,r_i),\qquad
H_i^*=H_i\,\frac{n_i^H}{n_i^H+3}, \label{eq:t6-H}
\end{align}
where \(N_i\) is originality (Mahalanobis length), \(A_i\) is anticipation (does the artist's
deviation point where the genre went over the next decade?), and \(H_i^*\) is transmission (did the
followers who name \(i\) move in \(i\)'s direction?), shrunk for small follower counts. Reach is
measured by \(K_i\), the within-jazz Katz score of \cref{subsec:q1-jazz}, and by \(B_i\), the number
of non-jazz followers; the Jazz$\to$Pop/Rock block alone contains @@J_POP@@ edges
(\cref{subsec:q3-inf-time}). The sonic indicators are defined for @@N_ANT@@ artists; @@N_FOL@@ have
a usable transmission score and @@N_ELIG@@ have at least five jazz followers.

\paragraph{Validation.}
There are @@N_PAIRS@@ jazz$\to$jazz pairs with a later follower and at least three songs for both
artists. The mean alignment is @@ADOPT_OBS@@. A within-decade permutation null (2{,}000 draws)
gives @@ADOPT_NULL@@ \(\pm\) @@ADOPT_SD@@, so @@ADOPT_P@@; the bootstrap 95\% interval is
\([@@ADOPT_LO@@,@@ADOPT_HI@@]\). @@ADOPT_POS@@ of the real pairs are positive against
@@ADOPT_NULLPOS@@ in the null. The effect is modest but real: reported influence is visible as a
small average movement of the follower's music in the influencer's direction. This complements the
matched-control similarity test of Task~4, which asks whether followers are closer to their
influencers than to controls; the present test asks whether they move \emph{towards} the
influencer's deviation from their own era.

The indicators are not redundant: \(N\)--\(A\) @@CORR_NA@@, \(N\)--\(K\) @@CORR_NK@@,
\(A\)--\(K\) @@CORR_AK@@, \(H^*\)--\(K\) @@CORR_HK@@. Originality is negatively related to citation
(@@CORR_NF@@ with future followers), anticipation is nearly independent of it (@@CORR_AF@@), and
transmission is the closest to future citation (@@CORR_HF@@). The most novel artists are not the
most cited, and being ahead is not the same as being central.

\paragraph{Dynamic influencers.}
We rank artists by \(S_i=z(A_i)+z(H_i^*)\), the equal-weight sum of the two standardized sonic
indicators, among the @@N_ELIG@@ artists with at least five jazz followers. Ranks are stable to
the weights (Spearman @@WEIGHT_RHO@@ for \(A\) weights 0.3--0.7) and to the transmission shrinkage
(raw and shrunk \(H\) correlate at @@SHRINK@@). \cref{tab:t6-top} gives the top twelve;
\cref{fig:t6-influencers} maps anticipation against transmission.

\begin{table}[htbp]
\centering
\small
\caption{Top twelve dynamic influencers. \(N\) originality, \(A\) anticipation, \(H^*\)
transmission, ``Followers'' the number of usable jazz followers, \(K\) within-jazz Katz, \(B\)
non-jazz followers.}
\label{tab:t6-top}
\begin{tabular}{rlccccccc}
\toprule
\# & Artist & Start & \(N\) & \(A\) & \(H^*\) & Followers & \(K\) & \(B\) \\
\midrule
@@TOP_ROWS@@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_t6_influencers.pdf}
\caption{(a) Anticipation against transmission for artists with at least five jazz followers;
bubble area is within-jazz Katz and colour is the active-start decade. (b) The top twelve by
\(S=z(A)+z(H^*)\), annotated with the two component scores.}
\label{fig:t6-influencers}
\end{figure}

\paragraph{Four roles.}
The \emph{canon} (high \(K\)) is Charlie Parker (@@K_PARKER@@), Lester Young (@@K_YOUNG@@), Miles
Davis (@@K_MILES@@), Dizzy Gillespie, Thelonious Monk and John Coltrane (@@K_COLTRAINE@@). Their
anticipation is mixed and often negative, because they define the centre of their generation. The
\emph{catalysts} have high \(A\) and high \(H^*\): Dave Brubeck (@@A_BRUBECK@@/@@H_BRUBECK@@), George
Duke, Tal Farlow (@@A_FARLOW@@), Grover Washington, Jr., David Sanborn and Bob James. The
\emph{prophets} are ahead but not copied: Duke Jordan, Shelly Manne, Jimmy Giuffre and John Lewis,
with Herbie Hancock (@@A_HANCOCK@@/@@H_HANCOCK@@), Lee Morgan and Andrew Hill in the 1960s
generation. The \emph{bridges} export influence beyond jazz: Miles Davis (@@B_MILES@@ non-jazz
followers), Nat King Cole (@@B_NAT@@), John Coltrane (@@B_COLTRAINE@@), Quincy Jones (@@B_QUINCY@@)
and Sun Ra (@@B_SUNRA@@).

\paragraph{How jazz changed.}
The swing baseline is acoustic and short (1930s: @@D1930AC@@, @@D1930DUR@@ min, @@D1930TEMPO@@
BPM); Lester Young is the pure canon type ($K=@@K_YOUNG@@$, $A=@@A_YOUNG@@$), and Cab Calloway is a
bridge (@@B_CALLOWAY@@ non-jazz followers). The 1949 bebop break
is visible in the cohort, not in Parker's distance from it: Parker has the highest within-jazz Katz
(@@K_PARKER@@) but only $A=@@A_PARKER@@$ and $H^*=@@H_PARKER@@$; Bud Powell ($A=@@A_POWELL@@$) and
Gillespie ($H^*=@@H_GILLESPIE@@$) carry more of the directional signal. The 1950s add the LP and
hard bop: Brubeck (@@A_BRUBECK@@/@@H_BRUBECK@@), Farlow ($A=@@A_FARLOW@@$), Horace Silver
($A=@@A_SILVER@@$/$H^*=@@H_SILVER@@$) and Clifford Brown ($A=@@A_BROWN@@$). The late 1950s--60s
separate the canon from the avant-garde: Coltrane and Miles remain central ($K=@@K_COLTRAINE@@$
and @@K_MILES@@) with negative anticipation, while Ornette Coleman
($A=@@A_ORNETTE@@$/$H^*=@@H_ORNETTE@@$, @@B_ORNETTE@@ non-jazz followers), Cecil Taylor
($A=@@A_CECIL@@$, $K=@@K_CECIL@@$), Herbie Hancock ($A=@@A_HANCOCK@@$/$H^*=@@H_HANCOCK@@$) and
Andrew Hill ($A=@@A_HILL@@$/$H^*=@@H_HILL@@$) are ahead of the centre. The 1970s is fusion:
Hancock, Tony Williams ($A=@@A_TONY@@$), John McLaughlin ($A=@@A_MCLAUGHLIN@@$/$H^*=@@H_MCLAUGHLIN@@$,
@@B_MCLAUGHLIN@@ non-jazz followers) and Weather Report (@@B_WEATHER@@ non-jazz followers), with
Miles as the boundary bridge. The 1980s--2000s turn to crossover: George Duke
(@@A_DUKE@@/@@H_DUKE@@), Grover Washington, Jr. (@@A_GROVER@@/@@H_GROVER@@), Bob James
(@@A_JAMES@@/@@H_JAMES@@) and David Sanborn (@@A_SANBORN@@/@@H_SANBORN@@), with Wynton Marsalis
($A=@@A_MARSALIS@@$) marking the neoclassical reaction but having no usable jazz follower in this
edge set.

Across the period the influence process has two mechanisms. \emph{Canonical transmission}
reproduces the centre: the most cited artists define what the next generation sounds like, which
shows up as high \(K\) with low \(A\) and \(H^*\). \emph{Catalytic transmission} moves the centre: an
artist's deviation points where the genre goes and the followers move towards it, which shows up as
high \(A\) and \(H^*\). The largest movements occur when the second mechanism meets an opening that
is not itself musical --- the LP in the 1950s, electric instruments and amplification in the late
1960s, and multitrack production in the 1980s. This is also where the social and technological
dimension of the problem enters the network: it enters as a change of recording format and
production that shifts the feature centroid before the influence network catches up.

\paragraph{Limits and robustness.}
Only @@J_USED@@ of the @@J_A@@ jazz artists have at least three songs, @@N_FOL@@ have a usable
transmission score, and @@N_ELIG@@ have at least five followers; the ranking is therefore
descriptive rather than definitive. The 2010s (22 songs) are excluded; the 1920s contain only 9.
Active-start is a decade, it comes from AllMusic while the features come from Spotify, and a
song's year is its release year, not its composition year. Career averages are replaced by
song-level residuals, but the genre label remains coarse: ``Jazz'' covers swing, bebop, free jazz
and smooth jazz. Leave-one-out centroids change the originality and anticipation rankings by
Spearman @@LOO_N@@ and @@LOO_A@@; adding liveness gives @@LIV_N@@ and @@LIV_A@@; a five-year
anticipation horizon gives @@H5@@. The transmission test controls for decade composition but not
for shared scene or label, and it is an association between reported influence and sonic
direction, not a causal effect.
"""


if __name__ == "__main__":
    main()
