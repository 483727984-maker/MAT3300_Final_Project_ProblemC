# MAT3300 Final Project — Problem C (ICM 2021-D: Influence of Music)

Team workspace for the music-influence modeling project.

## Progress

| Area | Status | Location |
|------|--------|----------|
| **Q1** Directed influence network | Done (teammate) | `code/q1_*.py`, `code/q1_outputs/`, `report/model1.tex` |
| **Q2** Music similarity | Done | `code/model_ii_*.py`, `report/model2.tex` |
| **Q3** Genre similarity / distinction / sim-over-time | Done (similarity side) | `code/model_iii_*.py`, `report/model3.tex` |
| Q3 genre **influence** over time | Placeholder | `report/model3.tex` §`subsec:q3-inf-time` |
| Q3 “Are some related?” | Placeholder | `report/model3.tex` §`subsec:q3-related` |
| **Q5** Revolutions (T5) | Done | `code/model_iv_revolutions.py`, `report/model4.tex` |
| **Q6** One-genre evolution (T6) | Done | `code/q6/`, `report/model4_t6.tex` |
| Q4, Q7 / memo / summary | Not done | `report/main.tex` TODOs |

## Report (LaTeX)

Compile from `report/`:

```bash
cd report
pdflatex main.tex   # run twice for ToC / refs
```

Main inputs: `model1.tex`, `model2.tex`, `model3.tex`, `model4.tex` (T5 and
`model4_t6.tex` for T6). Figures in `report/figures/`.

**Teammate hooks in Model III**
- `\ref{subsec:q3-inf-time}` — genre influence over time (network side)
- `\ref{subsec:q3-related}` — synthesize sonic relatedness with cross-genre influence

## Code

See `code/README.md`. Model III entry points:

```bash
export MPLBACKEND=Agg MPLCONFIGDIR=code/.mplconfig
mkdir -p "$MPLCONFIGDIR"
/tmp/mat3300_venv/bin/python code/model_iii_genre_similarity.py
/tmp/mat3300_venv/bin/python code/model_iii_genre_distinction.py
/tmp/mat3300_venv/bin/python code/model_iii_similarity_over_time.py
```

## Data

Only the four ICM files under `ProblemC_Music_data/`.

## Overleaf

https://www.overleaf.com/project/6afc1695ba902a83a9c02e24  
Snippets mirror: `overleaf_snippets/` (including `model3.tex`).
