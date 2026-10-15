# CINESCOPE

**Explainable Movie Review Sentiment Intelligence**  
PES University · UE24CS352A Machine Learning · Project 20

CINESCOPE is a reproducible, dependency-light reproduction attempt of Wu and
Shin's *Machine Learning Based Classification for Sentimental Analysis of IMDb
Reviews*. It downloads the official IMDb Large Movie Review Dataset, applies
the paper-inspired cleaning and TF-IDF representation, trains a logistic
regression classifier, evaluates on the untouched test split, and exposes the
model through a small web application with word-level explanations.

## Quick start

```bash
python3 -m cinescope.cli download
python3 -m cinescope.cli train --max-features 12000
python3 -m cinescope.cli evaluate
python3 -m cinescope.cli serve
```

Open <http://127.0.0.1:8000>. The commands use only the Python standard
library. For the optional six-model comparison, install the packages in
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

CUDA is optional and detected without changing the CPU baseline:

```bash
python3 -m cinescope.cli gpu-status
```

Install a CUDA-matched PyTorch wheel using the official PyTorch selector and
then use `cinescope.gpu.select_device("auto")` in optional training code.
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
* `models/` and `data/` — generated artifacts (ignored by Git).

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
