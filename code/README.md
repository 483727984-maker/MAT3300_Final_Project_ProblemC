# Code

Reproducible scripts for the MAT3300 Problem C report. Prefer running from the repo root.

## Setup (once)

```bash
python3 -m venv /tmp/mat3300_venv
/tmp/mat3300_venv/bin/pip install -r code/requirements.txt
```

## Preprocessing (shared)

```bash
/tmp/mat3300_venv/bin/python code/preprocess_data.py
```

Writes:
- `code/outputs/artists_features_with_genre.csv`
- `code/outputs/genre_artist_counts.csv`
- `code/outputs/preprocess_summary.json`

## Model II — similarity numbers

```bash
/tmp/mat3300_venv/bin/python code/model_ii_similarity.py
```

Writes `code/outputs/model_ii_summary.csv` and genre-centroid distance CSVs.

## Model II — heatmaps (for Overleaf)

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=code/.mplconfig
mkdir -p "$MPLCONFIGDIR"
/tmp/mat3300_venv/bin/python code/model_ii_heatmap.py
```

Figures (upload these two PDFs into the Overleaf project root):

- `code/figures/heatmap_genre_distances_baseline.pdf`
- `code/figures/heatmap_genre_distances_pca.pdf`

Also mirrored under `code/outputs/` as `.png` / `.pdf`.

## Model III — genre similarity (Katz-weighted centroids)

Uses Q1 `katz_out` as within-genre weights to form each genre's feature vector,
then Euclidean distance / \(s=1/(1+d)\) between genres (equal-weight baseline included).

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=code/.mplconfig
mkdir -p "$MPLCONFIGDIR"
/tmp/mat3300_venv/bin/python code/model_iii_genre_similarity.py
```

Writes under `code/outputs/` (figures also in `code/figures/`):
- `genre_feature_vectors_katz.csv` / `genre_feature_vectors_equal.csv`
- `genre_centroid_distances_katz.csv` / `genre_centroid_similarity_katz.csv`
- `genre_within_cohesion.csv`, `genre_pair_similarity_katz_ranked.csv`
- `model_iii_genre_similarity_summary.json`
- heatmaps `heatmap_genre_distances_katz.*`, `heatmap_genre_similarity_katz.*`

## Model III — what distinguishes a genre (same \(c_g\))

```bash
/tmp/mat3300_venv/bin/python code/model_iii_genre_distinction.py
```

Writes profile / discriminability tables and:
- `genre_profile_heatmap_katz.*`
- `genre_feature_discriminability.*`

## Model III — genre similarity over time

Same \(c_g(t)\) definition by artist `active_start` decade (Katz-weighted, global z-space).

```bash
/tmp/mat3300_venv/bin/python code/model_iii_similarity_over_time.py
```

Key outputs:
- `genre_similarity_over_time_summary.csv`
- `genre_pair_similarity_over_time.csv` / `genre_pair_similarity_trends.csv`
- `genre_self_drift_over_time.csv`
- figures: `genre_mean_between_distance_over_time.*`, `genre_pair_similarity_over_time.*`, `genre_self_drift_over_time.*`
