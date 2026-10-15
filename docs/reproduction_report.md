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

CUDA is not assumed for the published baseline. `cinescope.gpu` reports
PyTorch/CUDA availability and selects `cuda` only when the runtime is
actually available; otherwise the project remains reproducible on CPU.
