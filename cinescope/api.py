"""FastAPI application for CINESCOPE.

Install ``requirements.txt`` to run this adapter. The dependency-free server
remains available through ``python3 -m cinescope.cli serve``.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

from .explain import explain
from .persistence import load_artifact

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as error:  # pragma: no cover - exercised in dependency checks
    raise RuntimeError(
        "FastAPI is not installed. Install requirements.txt before importing cinescope.api."
    ) from error

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path(os.getenv("CINESCOPE_ARTIFACT", ROOT / "models" / "cinescope.json"))
vectorizer, model = load_artifact(ARTIFACT)
app = FastAPI(title="CINESCOPE API", version="0.1.0")
GPU_RESULTS = ROOT / "results" / "gpu_baseline_metrics.json"
COMPARISON_RESULTS = ROOT / "results" / "model_comparison_smoke.json"


class ReviewRequest(BaseModel):
    text: str


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "cinescope-logistic-regression"}


@app.get("/api/metadata")
def metadata() -> dict:
    gpu_result = json.loads(GPU_RESULTS.read_text()) if GPU_RESULTS.exists() else {}
    comparison = json.loads(COMPARISON_RESULTS.read_text()) if COMPARISON_RESULTS.exists() else {"results": []}
    return {
        "dataset": {"train": 25000, "test": 25000, "positive": 12500, "negative": 12500},
        "model": gpu_result,
        "comparison": comparison["results"],
    }


@app.post("/api/predict")
def predict(request: ReviewRequest) -> dict:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must be a non-empty string")
    probability = model.predict_proba_one(vectorizer.transform([request.text])[0])
    return {
        "label": "positive" if probability >= 0.5 else "negative",
        "probability": round(probability, 4),
        "explanation": explain(request.text, vectorizer, model),
    }
