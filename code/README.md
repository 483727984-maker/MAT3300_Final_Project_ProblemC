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
