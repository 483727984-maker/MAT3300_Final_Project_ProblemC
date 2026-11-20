# MAT3300 Final Project — Problem C (ICM 2021-D: Influence of Music)

Team workspace for the music-influence modeling project.

## Current progress (synced from local work)

| Area | Status | Notes |
|------|--------|--------|
| Q1 Network / influence measures | Not started in code | Deferred; discuss later |
| **Q2 Music similarity (Model II)** | **In progress / runnable** | Main focus so far |
| Q3–Q7 | Conceptual only | Not coded yet |
| Overleaf write-up | Partial | Model II section + notation/data snippets |

### Model II (Q2) — what is implemented

- **Baseline:** z-score standardize → Euclidean distance  
- **Robustness:** z-score → PCA → Euclidean distance  
- **Check:** within-genre mean distance vs between-genre mean distance  
- **Extras:** genre-centroid distance heatmaps for the report  

Code lives under `code/`:

| File | Role |
|------|------|
| `code/preprocess_data.py` | Join artist features with genre labels |
| `code/model_ii_similarity.py` | Similarity metrics + summary tables |
| `code/model_ii_heatmap.py` | Genre distance heatmaps |
| `code/requirements.txt` | Python deps |
| `code/README.md` | How to run |
| `code/outputs/` | Generated CSVs / figures |
| `code/figures/` | Heatmap PDFs for Overleaf |

LaTeX snippets (for Overleaf paste/sync): `overleaf_snippets/`  
Overleaf project (external): https://www.overleaf.com/project/6afc1695ba902a83a9c02e24

## Data (required)

Use **only** the four ICM-provided files in `ProblemC_Music_data/`:

- `influence_data.csv`
- `full_music_data.csv`
- `data_by_artist.csv`
- `data_by_year.csv`

References / problem PDFs are also in this repo for convenience.

## Setup & run

```bash
python3 -m venv /tmp/mat3300_venv
/tmp/mat3300_venv/bin/pip install -r code/requirements.txt

/tmp/mat3300_venv/bin/python code/preprocess_data.py
/tmp/mat3300_venv/bin/python code/model_ii_similarity.py

export MPLBACKEND=Agg
export MPLCONFIGDIR=code/.mplconfig
mkdir -p "$MPLCONFIGDIR"
/tmp/mat3300_venv/bin/python code/model_ii_heatmap.py
```

## Collaborators

Repo owner invites teammates via GitHub → **Settings → Collaborators → Add people**.  
After accepting the invite: `git clone` this repo, work on a branch or coordinate file ownership, then `git pull` / `git push`.
