# Dataset datasheet

## Identity and source

**IMDb Large Movie Review Dataset (Maas et al., 2011)**  
Canonical sources: <https://ai.stanford.edu/~amaas/data/sentiment/> and
<https://huggingface.co/datasets/stanfordnlp/imdb>.

The corpus contains 50,000 labeled reviews: 25,000 train and 25,000 test,
balanced between positive and negative classes. Each review is plain text; the
label is `0` (negative) or `1` (positive).

## Collection and labeling

Reviews were collected from IMDb and labeled from user ratings. The assigned
Wu–Shin paper describes negative reviews as ratings below 4 and positive
reviews as ratings above 7, omitting intermediate ratings. The canonical
dataset documentation describes the polarized rule as negative ≤4 and
positive ≥7, with 5 and 6 omitted; the paper's prose is ambiguous at the
boundaries. The distributed benchmark exposes binary folders rather than each
original rating, so this project cannot verify that filtering decision review
by review.

## Intended use and limitations

The dataset is intended for educational binary sentiment classification.
It is not a representative sample of all movie audiences, languages, eras, or
opinions. Reviews can contain sarcasm, spoilers, quotations, and rating
language. Predictions must not be used for employment, credit, medical, or
other high-impact decisions.

## License and access

Use the canonical source terms and IMDb terms. This repository does not
redistribute the corpus; `cinescope download` retrieves it at runtime. Users
must check current source terms before publishing or sharing downloaded data.
The dataset's research citation and licensing conditions should be treated as
separate from this project's MIT-style source code.

## Preprocessing and splits

Punctuation, numbers, line breaks, and a compact English stopword list are
removed; text is lowercased and tokenized. Training vocabulary and IDF values
are fit only on `train`; evaluation uses untouched `test`. Generated data stays
outside version control. The baseline uses normalized TF-IDF unigrams and a
deterministic sparse logistic regression optimizer.

## Risks

Duplicate or near-duplicate reviews, leakage through recurring movie names,
and temporal/domain drift can inflate or reduce measured performance.
IMDb users are self-selecting, and English-only processing excludes many
linguistic contexts. Reviews may contain usernames or other personal
information; avoid exposing raw examples unnecessarily.

## Citation

Maas, A. L. et al. (2011), “Learning Word Vectors for Sentiment Analysis,”
ACL. See the source pages above for the original download and citation.
