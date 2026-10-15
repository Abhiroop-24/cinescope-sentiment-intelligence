# CINESCOPE

**Explainable Movie Review Sentiment Intelligence**  
PES University · UE24CS352A Machine Learning · Project 20

CINESCOPE is a reproducible, dependency-light reproduction attempt of Wu and
Shin's *Machine Learning Based Classification for Sentimental Analysis of IMDb
Reviews*. It downloads the official IMDb Large Movie Review Dataset, applies
the paper-inspired cleaning and TF-IDF representation, trains a logistic
regression classifier, evaluates on the untouched test split, and exposes the
model through a small web application with word-level explanations.

## Quick start (one command)

```bash
source .venv/bin/activate
python -m cinescope.cli run --device auto
```

This single command downloads the official IMDb data if needed, trains the
model with CUDA when PyTorch detects your GPU, evaluates the full test split,
and starts the website at <http://127.0.0.1:8000>. Use `--device cuda` to
require the GPU or `--device cpu` to force CPU execution. Install the
dependencies once with:

```bash
pip install -r requirements.txt
pip install -r requirements-cuda.txt
```

The separate commands remain available for debugging. For the optional
six-model comparison, install the packages in
`requirements-optional.txt` and run:

```bash
python3 -m cinescope.compare_cli --data-root data/raw/aclImdb
```

The FastAPI adapter and its test client are available after
`pip install -r requirements.txt`:

```bash
uvicorn cinescope.api:app --reload
```

The React source is in `frontend/`; `npm install --prefix frontend` followed
by `npm run build --prefix frontend` creates its production bundle. The
dependency-free `web/index.html` remains the fallback served by the standard
library server.

CUDA status can be checked directly:

```bash
python3 -m cinescope.cli gpu-status
```

Requesting CUDA when no compatible runtime exists fails explicitly.

## Project layout

* `cinescope/` — data loading, preprocessing, vectorization, model, metrics,
  explanation, CLI, and HTTP server.
* `web/` — responsive black-and-white frontend.
* `frontend/` — React/Vite frontend source.
* `notebooks/` — executable baseline notebook.
* `tests/` — unit and API tests.
* `docs/` — dataset datasheet, research reproduction report, and demo/viva
  support.
* `models/` — the lightweight serialized baseline artifact tracked for the
  demo.
* `results/` — measured baseline and bounded comparison results.
* `data/` — downloaded dataset files, ignored by Git.

The tracked baseline was trained on all 25,000 IMDb training reviews with
20,000 TF-IDF unigram-and-bigram features and reaches 0.8846 accuracy on the untouched 25,000
review test split. Re-run the commands above to regenerate the artifact and
results locally.

The CUDA-enabled PyTorch run was also verified on an NVIDIA GeForce RTX 2050:
the phrase-aware 20,000-feature setup reached 0.8846 accuracy and is recorded in
`results/gpu_baseline_metrics.json`. The GPU model is stored in
`models/cinescope_gpu.json`.

## Research fidelity and limitations

The paper describes filtering ratings below 4 as negative and above 7 as
positive, omitting intermediate ratings. The public Maas dataset distributes
already-balanced train/test folders with binary labels, but does not expose
the original numeric rating for every review. Therefore this repository labels
the official split as a **reproduction attempt**, not an exact replication.
This distinction, preprocessing choices, and known limitations are recorded in
[`docs/reproduction_report.md`](docs/reproduction_report.md) and
[`docs/dataset_datasheet.md`](docs/dataset_datasheet.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests use temporary fixtures and never require downloading 50,000 reviews.

## Citation

Wu, C.-L. and Shin, S.-L. *Machine Learning based classification for
Sentimental analysis of IMDb reviews*, Stanford CS229 project report, 2020.
