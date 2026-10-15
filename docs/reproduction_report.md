# Research reproduction and comparison report

## Reference

Wu and Shin, *Machine Learning Based Classification for Sentimental Analysis
of IMDb Reviews*, Stanford CS229 project report (2020).

The report proposes cleaning IMDb reviews (lowercasing, removing punctuation,
line breaks, numbers, stopwords, and stemming), then comparing binary,
word-count, n-gram, and TF-IDF vectorization with logistic regression, SVM,
Naive Bayes, random forest, boosting, and deep neural networks. Its dataset
description says ratings below 4 are negative, above 7 positive, and
intermediate ratings are omitted.

Table 2 reports 90.6% as the best accuracy (DNN with binary trigram features).
Logistic regression and SVM reach approximately 90.0–90.1%, Naive Bayes 88.1%,
random forest 85.2%, and boosting 83.4% on their strongest listed
configurations. These are paper-reported values, not results silently
substituted for this project's run.

## Reproduction implemented here

This repository uses the canonical 25,000/25,000 labeled train/test folders,
without merging or randomly repartitioning them. It implements the paper's
core cleaning idea and TF-IDF representation, then trains deterministic sparse
logistic regression. The model and vectorizer are saved together so the web
demo uses the measured artifact rather than a mock result.

This is explicitly a **reproduction attempt**. The public benchmark does not
include the original numeric rating in each review, so the paper's threshold
filter cannot be reconstructed exactly. Stemming is also omitted from the
dependency-free baseline; this is documented rather than silently claimed as
identical.

## Evaluation

Run:

```bash
python3 -m cinescope.cli download
python3 -m cinescope.cli train
python3 -m cinescope.cli evaluate
```

The command reports accuracy, precision, recall, F1, and confusion-matrix
counts on the untouched test split. Run with `--limit` for a quick smoke test;
reported final results should use the full split and the generated artifact.
No result is hard-coded in the application.

### Verified baseline run

The full 25,000-review training split and full 25,000-review test split were
CPU baseline run on 2026-10-15 with 12,000 maximum TF-IDF unigram features:

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.87908 | 0.91037 | 0.84096 | 0.87429 |

Confusion counts were TP=10,512, TN=11,465, FP=1,035, and FN=1,988. The
serialized artifact is generated under `models/cinescope.json` and is ignored
by Git because it is reproducible from the downloaded dataset.

## Comparison plan

The paper's six algorithm families and four representations remain useful
comparison axes. The optional scientific environment
(`requirements-optional.txt`) can be used to extend the pipeline with
scikit-learn's LogisticRegression, LinearSVC, MultinomialNB, RandomForest,
boosting, and a neural baseline. That extension must record parameters,
runtime, random seeds, and the exact split before results are presented.
The executable implementation is `cinescope.compare`; it fails explicitly
when the optional dependency is absent and writes measured JSON results when
the comparison is run.
An 18-configuration smoke run (six classifier families × three
representations) on 1,000 reviews per class produced measured results in
`results/model_comparison_smoke.json`; its leading configuration was DNN with
binary 1–3-gram features at 0.8870 accuracy. These bounded results are not
presented as full-split paper replication results.

CUDA is not assumed for the published baseline. `cinescope.gpu` reports
PyTorch/CUDA availability and selects `cuda` only when the runtime is
actually available; otherwise the project remains reproducible on CPU.

The accelerated implementation was verified on an NVIDIA GeForce RTX 2050
with PyTorch 2.11.0+cu128. Full-split training used 25,000 reviews and 20,000
TF-IDF unigram-and-bigram features, reaching 0.8846 accuracy on the
25,000-review test split.
