# Demo and viva support

## Live demo

1. `python3 -m cinescope.cli download`
2. `python3 -m cinescope.cli train --max-features 12000`
3. `python3 -m cinescope.cli evaluate`
4. `python3 -m cinescope.cli serve`
5. Open `http://127.0.0.1:8000`, paste one clearly positive and one clearly
   negative review, and point out the explanation terms.

## Likely questions

**Why TF-IDF?** It downweights terms common across documents and emphasizes
terms concentrated in a smaller number of reviews.

**Why is this not an exact replication?** The public folders retain binary
labels but not the original numeric ratings used by the paper's threshold
description.

**What is leakage?** Fitting vocabulary or IDF on test text would let test
information influence training. CINESCOPE fits those artifacts on train only.

**What does an explanation mean?** A term's contribution is its normalized
TF-IDF value multiplied by the learned logistic-regression weight; it is a
local feature attribution, not a causal claim.
