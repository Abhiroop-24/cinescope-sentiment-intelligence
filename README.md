# CINESCOPE

**Explainable Movie Review Sentiment Intelligence**  
PES University · UE24CS352A Machine Learning · Project 20

CINESCOPE is a reproducible, dependency-light reproduction attempt of Wu and
Shin's *Machine Learning Based Classification for Sentimental Analysis of IMDb
Reviews*. It downloads the official IMDb Large Movie Review Dataset, applies
paper-inspired cleaning, learns TF-IDF unigram and bigram features, trains a
logistic regression classifier, evaluates on the untouched test split, and
exposes the model through a web application with word-level explanations.

## What is included

### Dataset

* Official IMDb Large Movie Review Dataset.
* 25,000 labeled training reviews and 25,000 labeled test reviews.
* Balanced classes: 12,500 positive and 12,500 negative reviews in each split.
* The raw dataset is downloaded locally to `data/` and excluded from Git.

### Text processing

Each review goes through the same preprocessing path during training and
prediction:

1. Convert text to lowercase.
2. Extract alphabetic tokens and preserve contractions such as `isn't`.
3. Remove common stopwords.
4. Build unigram and adjacent bigram features.
5. Calculate normalized TF-IDF values.
6. Keep the 20,000 most frequent features with minimum document frequency 2.

Bigrams allow the model to learn short sentiment phrases such as `not worth`,
`poor acting`, and `very good`, instead of treating every word independently.

### Classifier and acceleration

The final classifier is binary logistic regression. Its learned feature weights
produce both the prediction and the explanation shown in the dashboard.
Training uses batched PyTorch tensors and can run on CUDA; the persisted model
and vectorizer are JSON artifacts, so inference remains lightweight.

The project also includes:

* A dependency-free Python HTTP server.
* An optional FastAPI adapter.
* A React/Vite dashboard and a dependency-free fallback dashboard.
* Confidence visualization, influential-language explanations, prediction
  history, model comparison bars, dataset statistics, and dark mode.
* An executable notebook with dataset statistics, charts, training metrics,
  confusion matrix, and CUDA checks.

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

## API

The default one-command server exposes:

* `GET /api/health` — confirms that the model server is running.
* `GET /api/metadata` — returns dataset size, model configuration, metrics,
  and comparison results.
* `POST /api/predict` — accepts `{"text": "..."}` and returns the predicted
  label, positive probability, and influential feature contributions.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"A thoughtful film with brilliant acting."}'
```

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

## Verified results

The final artifact was trained on all 25,000 training reviews and evaluated on
all 25,000 test reviews. These are measured results from
`results/gpu_baseline_metrics.json`, not values hard-coded into the interface:

| Metric | Result |
|---|---:|
| Accuracy | **88.46%** |
| Precision | **86.66%** |
| Recall | **90.92%** |
| F1 score | **88.74%** |
| True positives | 11,365 |
| True negatives | 10,750 |
| False positives | 1,750 |
| False negatives | 1,135 |

Accuracy means that 22,115 of the 25,000 held-out reviews were classified
correctly. Precision measures how often a review predicted as positive was
actually positive. Recall measures how many of all positive reviews were
found. F1 is the harmonic mean of precision and recall, balancing both
measures.

The earlier unigram-only CUDA baseline reached 87.78% accuracy. Adding bigrams
increased accuracy to 88.46%, a measured improvement of 0.68 percentage points.
The CUDA-enabled run was verified with an Nvidia Geforce RTX GPU. The trained
artifacts are stored in `models/cinescope.json` and
`models/cinescope_gpu.json`.

To reproduce the evaluation after downloading the dataset:

```bash
python -m cinescope.cli run --device cuda
```

The command prints the evaluation metrics before starting the dashboard.

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
