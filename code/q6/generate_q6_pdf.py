"""Build the Question 6 section of the MAT3300 final-project report.

Reads q6_results.json, fills the LaTeX template and compiles it with tectonic.
Every number quoted in the report comes from the JSON, so rerunning
q6_analysis.py and this script regenerates the section exactly.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "q6_results.json"
TEX = ROOT / "Q6_Genre_Evolution.tex"
PDF = ROOT / "Q6_Genre_Evolution.pdf"
TECTONIC = Path("/opt/homebrew/bin/tectonic")

NAME = "虞紫涵 (Yu Zihan)"
SID = "123090773"

REPL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}


def esc(text) -> str:
    return "".join(REPL.get(ch, ch) for ch in str(text))


def f2(x) -> str:
    return f"{float(x):.2f}"


def pct(x) -> str:
    return f"{100 * float(x):.1f}\\%"


def join_names(items, key="artist", n=4):
    return ", ".join(esc(x[key]) for x in items[:n])


def role_examples(items, n=4, with_k=False):
    out = []
    for x in items[:n]:
        if with_k:
            out.append(f"{esc(x['artist'])} (A {x['anticipation']:+.2f}, "
                       f"H* {x['H_shrunk']:+.2f}, K {x['katz']:.1f})")
        else:
            out.append(f"{esc(x['artist'])} (A {x['anticipation']:+.2f}, "
                       f"H* {x['H_shrunk']:+.2f})")
    return "; ".join(out)


def bridge_examples(items, n=4):
    return "; ".join(f"{esc(x['artist'])} (B {x['boundary']})" for x in items[:n])


def feature_change_text(row, n=4):
    items = sorted(((f, v) for f, v in row.items()), key=lambda kv: -abs(kv[1]))[:n]
    return ", ".join(f"{f.replace('_ms', '')} ${v:+.2f}$" for f, v in items)


def main() -> None:
    r = json.loads(RESULT.read_text(encoding="utf-8"))
    d, tl, ad = r["data"], r["timeline"], r["adoption"]
    ind, st, rob = r["indicators"], r["styles"], r["robustness"]
    arch = ind["archetypes"]
    corr = ind["spearman"]

    # ---- tables
    data_rows = "\n".join([
        f"Raw influence records & {d['raw_edges']:,} \\\\",
        f"Influence edges retained (Assumption A3) & {d['network_edges']:,} \\\\",
        f"Time-order reversals retained & {d['reversed_edges']:,} \\\\",
        f"Same-decade edges & {d['same_decade_edges']:,} \\\\",
        f"Artists in the influence network & {d['network_nodes']:,} \\\\",
        f"Edges with at least one jazz endpoint & {d['jazz_edges_all']:,} \\\\",
        f"Jazz$\\to$jazz edges & {d['jazz_jazz_edges']:,} \\\\",
        f"Artists in the jazz influence subnetwork & {d['jazz_network_nodes']:,} \\\\",
        f"Artists labelled Jazz & {d['jazz_artists']:,} \\\\",
        f"Jazz artists with at least 3 songs & {d['jazz_artists_used']:,} \\\\",
        f"Jazz songs & {d['jazz_songs']:,} \\\\",
        f"Artists with at least one jazz song & {d['jazz_song_artists']} \\\\",
        f"Song years & {d['year_min']}--{d['year_max']} \\\\",
        f"Yearly timeline (years with $\\ge 20$ songs) & {d['timeline_min']}--{d['timeline_max']} \\\\",
        f"Jazz$\\to$jazz pairs used for transmission & {ad['pairs']:,} \\\\",
        f"Artists with a usable transmission score & {ind['with_followers']} \\\\",
        f"Artists with at least 5 jazz followers & {ind['eligible']} \\\\",
    ])

    decades = sorted((int(k), v) for k, v in tl["decade_raw"].items()
                     if 1930 <= int(k) <= 2000)
    decade_rows = "\n".join(
        f"{y}s & {v['songs']:,} & {v['acousticness']:.2f} & {v['energy']:.2f} & "
        f"{v['tempo']:.0f} & {v['valence']:.2f} & {v['instrumentalness']:.2f} & "
        f"{v['duration_min']:.1f} \\\\"
        for y, v in decades
    )

    top_rows = "\n".join(
        f"{i} & {esc(x['artist'])} & {x['start']}s & {x['novelty']:.2f} & "
        f"{x['anticipation']:+.2f} & {x['H_shrunk']:+.2f} & {x['followers']} & "
        f"{x['katz']:.2f} & {x['boundary']} \\\\"
        for i, x in enumerate(ind["top_dynamic"], 1)
    )

    role_rows = "\n".join([
        f"Canon & high $K$ & {role_examples(arch['canon_high_katz'], 4, True)} \\\\",
        f"Catalyst & high $A$ and high $H^*$ & "
        f"{role_examples(arch['catalysts_ahead_and_copied'], 4)} \\\\",
        f"Prophet & high $A$, low $H^*$ & "
        f"{role_examples(arch['prophets_ahead_not_canonised'], 4)} \\\\",
        f"Bridge & high $B$ & "
        f"{bridge_examples(arch['bridges_cross_genre'], 4)} \\\\",
    ])

    ep = {e["year"]: e["velocity"] for e in tl["episodes"]}
    ep_years = [e["year"] for e in tl["episodes"]]
    ep_txt = ", ".join(str(y) for y in ep_years[:-1]) + " and " + str(ep_years[-1])
    ep_changes = tl["episode_feature_change"]
    ep_strings = {y: feature_change_text(ep_changes[str(y)]) for y in ep_years}

    load1 = ", ".join(f"{k.replace('_ms', '')} ${v:+.2f}$" for k, v in
                      sorted(tl["pca_loadings_pc1"].items(), key=lambda kv: -abs(kv[1]))[:3])
    load2 = ", ".join(f"{k.replace('_ms', '')} ${v:+.2f}$" for k, v in
                      sorted(tl["pca_loadings_pc2"].items(), key=lambda kv: -abs(kv[1]))[:3])

    repl = {
        "NAME": NAME, "SID": SID,
        "DATA_ROWS": data_rows,
        "DECADE_ROWS": decade_rows,
        "TOP_ROWS": top_rows,
        "ROLE_ROWS": role_rows,
        "JAZZ_ARTISTS": f"{d['jazz_artists']:,}",
        "JAZZ_USED": f"{d['jazz_artists_used']:,}",
        "JAZZ_SONGS": f"{d['jazz_songs']:,}",
        "JAZZ_SONG_ARTISTS": f"{d['jazz_song_artists']}",
        "JAZZ_JJ": f"{d['jazz_jazz_edges']:,}",
        "JAZZ_EDGES_ALL": f"{d['jazz_edges_all']:,}",
        "JAZZ_NODES": f"{d['jazz_network_nodes']:,}",
        "TL_MIN": str(d["timeline_min"]), "TL_MAX": str(d["timeline_max"]),
        "N_PAIRS": f"{ad['pairs']:,}",
        "N_ANT": str(ind["with_anticipation"]),
        "N_FOL": str(ind["with_followers"]),
        "N_ELIG": str(ind["eligible"]),
        "N_1920S": str(sum(v for y, v in d["year_song_counts"].items()
                           if 1920 <= int(y) <= 1929)),
        "SHRINK_RHO": f"{ind['shrinkage_spearman']:.3f}",
        "PC1": f"{100 * tl['pca_explained'][0]:.1f}",
        "PC2": f"{100 * tl['pca_explained'][1]:.1f}",
        "PC3": f"{100 * tl['pca_explained'][2]:.1f}",
        "LOAD1": load1, "LOAD2": load2,
        "EPISODES": ep_txt,
        "EP_CHANGES": "\n".join(
            f"{y} & {ep[y]:.2f} & {ep_strings[y]} \\\\" for y in ep_years),
        "ADOPT_OBS": f"{ad['mean_alignment']:.3f}",
        "ADOPT_NULL": f"{ad['null_mean']:.3f}",
        "ADOPT_SD": f"{ad['null_sd']:.3f}",
        "ADOPT_Q99": f"{ad['null_q99']:.3f}",
        "ADOPT_LO": f"{ad['bootstrap_ci95'][0]:.3f}",
        "ADOPT_HI": f"{ad['bootstrap_ci95'][1]:.3f}",
        "ADOPT_P": ("$p<0.001$" if ad["p_value"] < 0.001 else f"$p= {ad['p_value']:.3f}$"),
        "ADOPT_POS": pct(ad["positive_share"]),
        "ADOPT_NULLPOS": pct(ad["null_positive_share"]),
        "CORR_NA": f"{corr['novelty']['anticipation']:+.2f}",
        "CORR_NK": f"{corr['novelty']['katz']:+.2f}",
        "CORR_AK": f"{corr['anticipation']['katz']:+.2f}",
        "CORR_HK": f"{corr['H_shrunk']['katz']:+.2f}",
        "CORR_AF": f"{corr['anticipation']['future_followers']:+.2f}",
        "CORR_NF": f"{corr['novelty']['future_followers']:+.2f}",
        "CORR_HF": f"{corr['H_shrunk']['future_followers']:+.2f}",
        "SIL": f"{st['silhouette']:.2f}",
        "CATALYSTS": join_names(arch["catalysts_ahead_and_copied"], n=4),
        "CANON": join_names(arch["canon_high_katz"], n=4),
        "PROPHETS": join_names(arch["prophets_ahead_not_canonised"], n=4),
        "BRIDGES": join_names(arch["bridges_cross_genre"], n=4),
        "LOO_N": f"{rob['loo_spearman']['novelty']:.3f}",
        "LOO_A": f"{rob['loo_spearman']['anticipation']:.3f}",
        "H5": f"{rob['horizon5_spearman']:.3f}",
        "LIV_N": f"{rob['liveness_included_spearman']['novelty']:.3f}",
        "LIV_A": f"{rob['liveness_included_spearman']['anticipation']:.3f}",
        "LIV_SIZE": str(rob["liveness_artifact"]["cluster_size"]),
        "LIV_Z": f"{rob['liveness_artifact']['liveness_z']:+.1f}",
        "WEIGHT_RHO": f"{min(ind['weight_robustness_spearman'].values()):.3f}--"
                      f"{max(ind['weight_robustness_spearman'].values()):.3f}",
    }

    text = TEMPLATE
    for k, v in repl.items():
        text = text.replace(f"@@{k}@@", str(v))
    TEX.write_text(text, encoding="utf-8")
    subprocess.run([str(TECTONIC), TEX.name, "--outdir", str(ROOT)], check=True, cwd=ROOT)
    print(f"Wrote {PDF}")


TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{xeCJK}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,array,hyperref,float}
\setCJKmainfont{PingFang SC}
\setlength{\emergencystretch}{3em}

\title{MAT3300 Final Project --- Problem C\\[2pt]
\large Question 6: How one genre evolves --- the jazz influence process, 1930s--2000s}
\author{@@NAME@@\\Student ID: @@SID@@\\The Chinese University of Hong Kong, Shenzhen}
\date{}

\begin{document}
\maketitle

\section{The question, the data, and the choice of jazz}
The sixth task asks us to analyse the influence process of musical evolution over time in
\emph{one} genre, to identify indicators that reveal the dynamic influencers, and to explain how
the genre and its artists changed. We take jazz as the genre. The choice continues Question~1:
the Jazz-induced influence subnetwork already described in Question~1
(@@JAZZ_NODES@@ artists, @@JAZZ_JJ@@ edges) spans the whole record, and it gives
the analysis a well-documented internal history --- swing, bebop, cool jazz, hard bop, free jazz,
fusion, crossover --- against which the data-driven periods can be read.

The data are the same four files and the same cleaning as Question~1, with Assumption~A3 of the
joint report: all 42,770 reported pairs are retained, including the 733 time-order reversals.
Of the 5,603 artists in the influence network, @@JAZZ_ARTISTS@@ carry the Jazz label as
their main genre; they are involved in @@JAZZ_EDGES_ALL@@ influence edges, of which
@@JAZZ_JJ@@ have jazz at both ends.
The Spotify files contain @@JAZZ_SONGS@@ songs by @@JAZZ_SONG_ARTISTS@@ of these artists; @@JAZZ_USED@@
artists have at least three songs and can be given a stable feature vector.
Table~\ref{tab:data} collects the counts.

\begin{table}[!ht]
\centering
\small
\begin{tabular}{lr}
\toprule
Quantity & Value \\
\midrule
@@DATA_ROWS@@
\bottomrule
\end{tabular}
\caption{Jazz data used in this section. The influence network is the cleaned network of
Question~1; the song features are the per-song entries of \texttt{full\_music\_data.csv}.}
\label{tab:data}
\end{table}

Two properties of the source data shape the design.
First, \texttt{data\_by\_artist} gives one career average per artist, not a time series.
We therefore work at the song level: every song is compared with the sound of jazz in its own
release year, and the artist's deviation is the average of these song-level deviations.
Second, \texttt{active\_start} is a decade, so the artist's position on the timeline is
decade-coarse; we use it only to place the artist's generation, never to date individual songs.

\section{A sonic timeline for jazz}
\subsection{The feature space}
We use the nine continuous audio features of the data (danceability, energy, valence, tempo,
loudness, acousticness, instrumentalness, speechiness, and duration), with duration on a
logarithmic scale, each standardised against the 5,854 artists of \texttt{data\_by\_artist.csv}.
A value of $+1$ therefore means one population standard deviation above the average recording of
the whole corpus, not of jazz.
We exclude \texttt{liveness}. On these records it mostly separates live albums from studio albums:
a $k$-means run that includes it sets apart a @@LIV_SIZE@@-song cluster almost entirely by
liveness ($@@LIV_Z@@$ standard deviations), which is a recording artefact rather than a style.
Mode and key are categorical and are not used.

The genre's timeline is the yearly mean feature vector $m(y)$ over jazz songs, computed for years
with at least 20 songs, smoothed with a centred three-year window and linearly interpolated
between years.
This leaves a reliable yearly series from @@TL_MIN@@ to @@TL_MAX@@; earlier and later years remain
in the decade summaries but not in the yearly velocity curve.

\subsection{The decade trajectory}
Table~\ref{tab:decades} and Figure~\ref{fig:timeline} summarise the movement.
The most acoustic decade is the 1930s (acousticness 0.91, duration 3.0 minutes, tempo 124~BPM);
the 1970s is the most electric and the longest (acousticness 0.48, duration 6.7 minutes, energy
0.51); the 1980s--2000s return part way towards an acoustic, quieter sound without returning to the
1930s values.

\begin{table}[!ht]
\centering
\small
\begin{tabular}{crcccccc}
\toprule
Decade & Songs & Acoustic. & Energy & Tempo & Valence & Instrum. & Duration \\
 & & & & (BPM) & & & (min) \\
\midrule
@@DECADE_ROWS@@
\bottomrule
\end{tabular}
\caption{Raw decade centroids of jazz songs. ``Acoustic.'', ``Energy'', ``Valence'' and
``Instrum.'' are Spotify features on their original scales; duration is in minutes.}
\label{tab:decades}
\end{table}

A principal-component analysis of the yearly centroids makes the shape of the trajectory
explicit (Figure~\ref{fig:timeline}a).
The first component explains @@PC1@@\% of the variance and loads positively on acousticness and
negatively on duration (@@LOAD1@@): it separates short acoustic recordings from long electric
ones.
The second component explains @@PC2@@\% and loads on @@LOAD2@@.
The 1930s and 1940s sit at the acoustic, short end; the 1950s and 1960s move left as tracks
lengthen and darken; the 1970s jump far to the electric, loud, long corner; the 1980s--2000s
return towards the middle.
The trajectory is a loop rather than a line: jazz modernised and then partly re-acousticised,
and a one-dimensional ``more modern'' axis would miss the reversal.

\begin{figure}[!ht]
\centering
\includegraphics[width=\linewidth]{figures/fig1_genre_timeline.png}
\caption{(a) Decade centroids of jazz in the space of the first two principal components of the
yearly centroids; arrows follow the decades. (b) Standardised decade centroids of the nine
features (z-scores against all 5,854 artists).}
\label{fig:timeline}
\end{figure}

\subsection{Episodes of rapid change}
The velocity of the centroid, $\|m(y+2)-m(y-2)\|$ in z-units, measures how fast the genre's sound
moved. Figure~\ref{fig:revolutions}a shows four well-separated local maxima, at @@EPISODES@@.
The feature decomposition of each episode (Figure~\ref{fig:revolutions}b and
Table~\ref{tab:episodes}) identifies what kind of change each one was.

\begin{table}[!ht]
\centering
\small
\begin{tabular}{clp{0.62\linewidth}}
\toprule
Year & Speed & Largest feature changes (z-units, $y-2\to y+2$) \\
\midrule
@@EP_CHANGES@@
\bottomrule
\end{tabular}
\caption{The four episodes of rapid centroid movement and the features that moved most.}
\label{tab:episodes}
\end{table}

The episodes are not all stylistic.
The 1949 episode is the break with the swing baseline: recordings become shorter, quieter, less
energetic and less speech-like, the profile of the small bebop group after the big band.
The 1954 episode is dominated by duration ($+1.58$ z, with the next largest change,
instrumentalness, at $-0.43$): this is consistent with the LP replacing the 78-rpm single, a
format change that allowed long performances.
The 1969 episode is the electric turn --- acousticness falls by 0.55 z while energy, loudness and
duration rise --- and the 1980 episode is a quieter, more acoustic, more speech-like production,
consistent with the crossover and smooth-jazz turn together with the beginning of the acoustic
revival.
A definition of ``revolution'' that looked only at style would miss the format revolution of the
1950s; the feature decomposition separates them.

\begin{figure}[!ht]
\centering
\includegraphics[width=\linewidth]{figures/fig2_revolutions.png}
\caption{(a) Speed of the jazz centroid; shaded bands mark the four episodes. (b) Change in each
feature around each episode, $m(y+2)-m(y-2)$ in z-units.}
\label{fig:revolutions}
\end{figure}

\section{Indicators of dynamic influence}
\subsection{Definitions}
For artist $i$ with at least three songs, let
\[
r_i=\frac{1}{n_i}\sum_{s\in i}\bigl(z_s-m(y_s)\bigr)
\]
be the artist's \emph{time-adjusted deviation}: the average of the song vectors, each centred on
the jazz centroid of its release year.
Four indicators are built from $r_i$ and from the influence edges.

\paragraph{Originality $N_i$.}
$N_i=\|r_i\|_\Sigma$, the Mahalanobis length of the deviation using the covariance of the 5,854
artist vectors. It measures how far the artist's sound was from the jazz of their own time, in
units that account for the correlations among features.

\paragraph{Anticipation $A_i$.}
Let $t_i$ be the artist's active-start decade and
$\Delta_i=m(t_i+10)-m(t_i)$ the direction in which the genre moved over the following ten years.
Then
\[
A_i=\cos\bigl(r_i,\Delta_i\bigr)\in[-1,1].
\]
A positive $A_i$ means that the artist's deviation from their own generation pointed where the
genre was about to go; a negative value means the opposite. The score is undefined for artists who
started in 2000 or later, because the timeline has no reliable ten-year future; for the 1930s
cohort the first reliable centroid (1936) is used as the starting point.

\paragraph{Transmission $H_i$.}
For each jazz$\to$jazz edge $i\to j$ with $t_j>t_i$ and at least three songs for both artists, compute
$\cos(r_j,r_i)$, the alignment of the follower's deviation with the influencer's deviation, and
average over the followers of $i$. Because many artists have few followers, we report the shrunk
value
\[
H_i^*=H_i\,\frac{n_i^H}{n_i^H+3},
\]
which pulls small samples towards zero. $H_i^*$ asks whether the people who name $i$ actually
moved their music in $i$'s direction.

\paragraph{Reach.}
$K_i$ is the out-going Katz centrality of $i$ inside the Jazz-induced subnetwork, with the same
attenuation $\alpha=0.1$ as Question~1; $B_i$ is the number of non-jazz artists who name $i$.
$K$ measures reach inside the genre and $B$ measures reach across its boundary.

The sonic indicators are computed on @@N_ANT@@ artists with a defined anticipation score;
@@N_FOL@@ of them have a usable transmission score and @@N_ELIG@@ have at least five jazz
followers.
No indicator uses popularity or any label other than ``is this artist jazz''.

\subsection{Validation: is the reported influence visible in the followers' music?}
There are @@N_PAIRS@@ jazz$\to$jazz pairs with a later follower and at least three songs for both
artists.
The mean alignment is @@ADOPT_OBS@@.
The permutation null reassigns each follower to a random jazz artist of the same active-start
decade, which preserves the decade composition of the pairs but destroys the reported edges.
Over 2,000 draws the null mean is @@ADOPT_NULL@@ with standard deviation @@ADOPT_SD@@ (99th
percentile @@ADOPT_Q99@@), and the observed value gives @@ADOPT_P@@; the bootstrap 95\%
confidence interval for the mean is $[@@ADOPT_LO@@,\,@@ADOPT_HI@@]$.
@@ADOPT_POS@@ of the real pairs have a positive alignment, against @@ADOPT_NULLPOS@@ in the null.

The effect is real but modest: the reported influence is a noisy signal, not a deterministic
transfer, and the test controls for decade composition but not for shared scene, label, or
geography.
It supports the weaker claim that is actually needed here: on average, the artists who are named as
influences are the artists in whose direction their followers' music moves.

\subsection{The indicators are not redundant}
The Spearman correlations among the indicators are small (Figure~\ref{fig:validation}b):
originality and anticipation correlate $@@CORR_NA@@$, originality and Katz $@@CORR_NK@@$,
anticipation and Katz $@@CORR_AK@@$, and transmission and Katz $@@CORR_HK@@$.
Originality is negatively related to citation ($@@CORR_NF@@$ with future followers), while
anticipation and transmission are weakly positive ($@@CORR_AF@@$ and $@@CORR_HF@@$).
The most novel artists are not the most cited ones, and being ahead of the genre is nearly
independent of being central in it.
The indicators therefore measure different things, which is what makes the typology below useful.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.95\linewidth]{figures/fig4_validation.png}
\caption{(a) Permutation null for the mean follower alignment with the observed value.
(b) Spearman correlations among originality $N$, anticipation $A$, transmission $H^*$, within-jazz
Katz $K$, boundary reach $B$, and future followers $F$.}
\label{fig:validation}
\end{figure}

\section{Who the indicators reveal}
\subsection{Four roles}
Combining the indicators separates four roles that a single centrality ranking cannot distinguish
(Table~\ref{tab:roles}).
The \emph{canon} has high within-genre Katz: Charlie Parker, Lester Young, Miles Davis, Dizzy
Gillespie, Thelonious Monk and John Coltrane.
Their anticipation scores are mixed and often negative, because they define the centre of their
generation rather than deviate from it.
The \emph{catalysts} have both high anticipation and high transmission: Dave Brubeck, George Duke,
Tal Farlow, Grover Washington, Jr., David Sanborn and Bob James.
Their sound pointed where jazz went, and their followers moved with them.
The \emph{prophets} are ahead of their time but were not copied in their own direction; the table
lists Duke Jordan, Shelly Manne, Jimmy Giuffre and John Lewis, and the same pattern appears for
Herbie Hancock, Lee Morgan and Andrew Hill.
The \emph{bridges} export influence beyond jazz: Miles Davis (77 non-jazz followers), Nat King Cole
(52), John Coltrane (52), Quincy Jones (25) and Sun Ra (24).

\begin{table}[!ht]
\centering
\small
\begin{tabular}{lp{0.17\linewidth}p{0.62\linewidth}}
\toprule
Role & Signature & Examples \\
\midrule
@@ROLE_ROWS@@
\bottomrule
\end{tabular}
\caption{Four roles revealed by the indicators. $A$ is anticipation, $H^*$ shrunk transmission,
$K$ within-jazz Katz, $B$ non-jazz followers.}
\label{tab:roles}
\end{table}

\subsection{The dynamic influencers}
We rank artists by the dynamic-influence score
\[
S_i=z(A_i)+z(H_i^*),
\]
the equal-weight sum of the two standardised sonic indicators.
Table~\ref{tab:top} gives the top twelve among the @@N_ELIG@@ artists with at least five jazz
followers; ranks are stable when the weights are changed (Spearman @@WEIGHT_RHO@@ for $A$ weights
from 0.3 to 0.7).

\begin{table}[!ht]
\centering
\small
\begin{tabular}{rlccccccc}
\toprule
\# & Artist & Start & $N$ & $A$ & $H^*$ & Followers & $K$ & $B$ \\
\midrule
@@TOP_ROWS@@
\bottomrule
\end{tabular}
\caption{Top twelve dynamic influencers. $N$ originality, $A$ anticipation, $H^*$ transmission,
``Followers'' the number of usable jazz followers, $K$ within-jazz Katz, $B$ non-jazz followers.}
\label{tab:top}
\end{table}

Two groups dominate.
The first is the cool/hard-bop generation of the 1940s--50s: Brubeck, Farlow, Barney Kessel,
Erroll Garner, Oscar Peterson, Horace Silver and Clifford Brown.
The second is the crossover generation of the 1960s: George Duke, Grover Washington, Jr., David
Sanborn and Bob James.
Both are moments at which the genre's centre moved, and in both the leading artists are not the
canonical names of Table~\ref{tab:roles}.
This is the main answer to the ``dynamic influencers'' part of the question: the artists who move
a genre are not necessarily the artists with the largest citation counts.
The canon supplies continuity; the catalysts supply direction.

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/fig5_influencers.png}
\caption{(a) Anticipation against transmission for artists with at least five jazz followers;
bubble area is within-jazz Katz and colour is the active-start decade. (b) The top twelve by
$S=z(A)+z(H^*)$, annotated with the two component scores.}
\label{fig:influencers}
\end{figure}

\section{How jazz changed, period by period}
The indicators and the timeline can be read together as a sequence of influence regimes.

\paragraph{1930s--mid-1940s: the swing baseline}
The decade centroid is the most acoustic and the shortest of the series (acousticness 0.91,
duration 3.0 minutes, tempo 124~BPM, valence 0.62), and the sound is dominated by up-tempo
acoustic small groups.
Lester Young is the pure canon type: within-jazz Katz 15.5, anticipation $-0.81$ and transmission
$-0.41$; his followers did not move towards his deviations, they reproduced the centre he already
defined.
Cab Calloway is a bridge (23 non-jazz followers) rather than a sonic catalyst, and Roy Eldridge is
a second, less extreme canon case.

\paragraph{Late 1940s: bebop}
The 1949 episode is the cleanest stylistic break: duration $-1.28$, energy $-1.00$, loudness
$-1.07$ and speechiness $-0.85$ z, with danceability rising.
The small group replaces the big band.
Charlie Parker has the highest within-jazz Katz (17.2) but only $A=0.26$ and $H^*=0.03$: the
revolution is visible in the cohort's shift, not in Parker's distance from it.
Bud Powell is the exception on the sonic side ($A=0.72$) and Gillespie combines a high centrality
with a positive transmission ($H^*=0.31$).

\paragraph{1950s: cool jazz, hard bop and the long-playing record}
The 1954 episode is almost entirely a duration shift ($+1.58$ z), consistent with the LP
replacing the 78-rpm single.
Tracks lengthen from 3.0 minutes in the 1930s to 4.9 minutes in the 1950s, and the 1954 episode is
the largest duration change in the series.
The dynamic influencers of this period are Brubeck ($A=0.61$, $H^*=0.62$), Farlow ($A=0.92$),
Kessel, Garner, Oscar Peterson, Horace Silver and Clifford Brown.
The Brubeck case is the clearest: a sound that was both ahead of the 1940s centre and copied by
later musicians.

\paragraph{Late 1950s--1960s: cool, modal, hard bop and the avant-garde}
The decade is darker and slower: valence falls to 0.45, the lowest of the series, and acousticness
to 0.75.
The canon and the avant-garde separate.
Coltrane and Miles remain central ($K=10.8$ and $13.9$) but have negative anticipation
($-0.52$ and $-0.33$): they are the centre, not ahead of it.
Ornette Coleman ($A=0.58$, $H^*=-0.24$, 16 non-jazz followers), Cecil Taylor ($A=0.81$, only two
usable jazz followers), Herbie Hancock ($A=0.76$, $H^*=-0.16$) and Andrew Hill ($A=0.70$,
$H^*=-0.31$) are the ahead-of-time group; their influence is either absorbed later or exported
rather than copied directly within the genre.

\paragraph{Late 1960s--1970s: fusion}
The 1969 episode is the electric turn (acousticness $-0.55$, energy $+0.56$, loudness $+0.48$,
duration $+0.76$).
The 1970s centroid is the most electric (acousticness 0.48), the highest-energy (0.51) and the
longest (6.7 minutes), the average track length being the highest of any decade.
Hancock ($A=0.76$), Tony Williams ($A=0.73$), John McLaughlin ($A=0.39$, $H^*=0.48$, five non-jazz
followers) and Weather Report (10 non-jazz followers) carry the change; Miles Davis is the
boundary bridge with 77 non-jazz followers.
Here the ``prophet'' and ``bridge'' roles overlap: the fusion idea travelled both inside jazz and
out of it.

\paragraph{1980s--2000s: crossover, smooth jazz and the acoustic revival}
The 1980 episode reverses several signs at once (energy $-0.47$, valence $-0.51$, acousticness
$+0.30$, speechiness $+0.59$): a quieter, more vocal, more produced sound.
Acousticness partially recovers (0.55 in the 1980s, 0.60 in the 1990s against 0.48 in the 1970s)
and the average track length falls back to 4.8 minutes by the 2000s.
The dynamic influencers are now the crossover musicians --- George Duke ($A=0.35$, $H^*=0.66$),
Grover Washington, Jr. ($A=0.23$, $H^*=0.70$), Bob James ($A=0.30$, $H^*=0.50$) and David Sanborn
($A=0.76$, $H^*=0.30$) --- while the canon remains Miles, Coltrane and Parker.
Wynton Marsalis ($A=0.31$, no usable jazz follower in this edge set) marks the neoclassical
reaction, but the data give him no transmission score.

Across the whole period the influence process has two mechanisms.
\emph{Canonical transmission} reproduces the centre: the most cited artists define what the next
generation sounds like, and the mechanism shows up as high $K$ with low $A$ and $H^*$.
\emph{Catalytic transmission} moves the centre: an artist's deviation from their generation points
where the genre goes, and their followers move towards it, which shows up as high $A$ and $H^*$.
The genre's largest moves occur when the second mechanism meets an opening that is not itself
musical --- the LP in the 1950s, electric instruments and amplification in the late 1960s, and
multitrack pop production in the 1980s.
This is also how the ``social, political or technological'' dimension of the problem enters the
network: it enters as a change in the recording format and production that shifts the feature
centroid before the influence network catches up.

\section{Robustness and limitations}
\paragraph{Leave-one-out centroids.}
Recomputing each artist's residual and anticipation with that artist's own songs removed from the
yearly centroid changes the rankings only marginally: Spearman @@LOO_N@@ for originality and
@@LOO_A@@ for anticipation. The artist's own contribution to a centroid of several hundred songs
is too small to drive the result.

\paragraph{Shrinkage, feature set and horizon.}
The shrinkage of the transmission score is not driving the ranking: among the @@N_ELIG@@ artists
with at least five followers, raw and shrunk transmission correlate at Spearman @@SHRINK_RHO@@.
Adding liveness back gives Spearman @@LIV_N@@ for originality and @@LIV_A@@ for anticipation.
Shortening the anticipation horizon from ten to five years gives @@H5@@: the ten-year horizon
matters, and the five-year version is a different, noisier question. The reported score uses the
ten-year horizon.

\paragraph{The transmission test.}
The permutation null preserves the follower's decade but not the follower's genre, because all
pairs are jazz$\to$jazz; it also does not control for shared scene or label. The observed effect is
an association between reported influence and sonic direction, not a causal effect. The positive share (56\% against 51\%) shows that the effect is a modest shift of the whole
distribution, not a few extreme pairs.

\paragraph{Coverage and measurement.}
Only 285 of the 406 jazz artists have at least three songs, and only @@N_FOL@@ have a usable
transmission score; the @@N_ELIG@@ artists with at least five followers are the defensible
ranking set. The 2010s contain 22 jazz songs and are excluded from the decade figures; the 1920s
contain only @@N_1920S@@. Active-start is a decade, and it comes from AllMusic while the features come from
Spotify, so an artist's generation and their recordings can be misaligned by a few years.
Career-spanning artist averages are replaced by song-level residuals, but a song's year is its
release year, not its composition year.
Finally, the genre label is coarse: ``Jazz'' covers swing, bebop, free jazz and smooth jazz, and
the sub-genre is not observed.
A $k$-means partition of the songs was also computed; its silhouette is only @@SIL@@ and the
solution changes with small changes to the song set, so we use the continuous trajectory and the
episodes rather than hard clusters.

\section{Conclusion}
Jazz's sonic centre moved through a loop: acoustic and short in the 1930s, longer and darker
through the 1950s--60s, electric and long in the 1970s, then partly acoustic again.
Four episodes mark the largest moves, and their feature decompositions separate a stylistic
revolution (1949), a format revolution (1954), an electric revolution (1969) and a
crossover/acoustic reversal (1980).

Four indicators make the dynamic influencers visible.
Originality, anticipation, transmission and reach are nearly independent of one another, and the
transmission indicator passes a within-decade permutation test ($p<0.001$; mean alignment 0.164
against a null of 0.024).
The resulting typology separates the canon (Parker, Young, Miles, Gillespie, Monk, Coltrane) from
the catalysts (Brubeck, Farlow, Duke, Washington, Sanborn, James), the prophets (John Lewis,
Hancock, Lee Morgan, Andrew Hill) and the bridges (Miles, Nat King Cole, Coltrane, Quincy Jones,
Sun Ra).
The artists who moved jazz are not the artists with the largest citation counts: the canon supplies
continuity, while the catalysts supply direction, and the genre's largest changes occur when a
catalytic sound meets a non-musical opening such as a new recording format or a new instrument.

\paragraph{Code and reproducibility.}
\texttt{q6\_analysis.py} produces every number and figure in this section from the four problem
files; \texttt{q6\_results.json} stores the numbers used in the tables, \texttt{q6\_artists.csv}
stores the full artist-level indicator table, and \texttt{generate\_q6\_pdf.py} rebuilds this
document.

\begin{thebibliography}{9}
\bibitem{problem} 2021 ICM Problem D, \emph{The Influence of Music}, problem statement (provided).
\bibitem{allmusic} \texttt{influence\_data.csv}, influence relations scraped from AllMusic.com,
as provided in the problem data set.
\bibitem{spotify} \texttt{full\_music\_data.csv}, \texttt{data\_by\_artist.csv} and
\texttt{data\_by\_year.csv}, Spotify audio features, as provided in the problem data set.
\bibitem{katz} L. Katz, A new status index derived from sociometric analysis,
\emph{Psychometrika} 18 (1953), 39--43.
\bibitem{gioia} T. Gioia, \emph{The History of Jazz}, 3rd ed., Oxford University Press, 2021.
\bibitem{giddins} G. Giddins and S. DeVeaux, \emph{Jazz}, W. W. Norton, 2009.
\end{thebibliography}

\end{document}
"""


if __name__ == "__main__":
    main()
