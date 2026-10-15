"""Command line interface for CINESCOPE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download_dataset
from .explain import explain
from .persistence import load_artifact
from .pipeline import evaluate, train, train_accelerated
from .server import run
from .gpu import cuda_status

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="CINESCOPE IMDb sentiment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--max-features", type=int, default=20000)
    train_parser.add_argument("--limit", type=int)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--limit", type=int)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("text")
    sub.add_parser("serve")
    sub.add_parser("gpu-status")
    run_parser = sub.add_parser("run", help="download, train with GPU when available, evaluate, and serve")
    run_parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    run_parser.add_argument("--max-features", type=int, default=20000)
    run_parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    data_root = ROOT / "data" / "raw"
    artifact = ROOT / "models" / "cinescope.json"
    if args.command == "download":
        print(download_dataset(data_root))
    elif args.command == "train":
        print(json.dumps(train(data_root / "aclImdb", artifact, args.max_features, args.limit), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(data_root / "aclImdb", artifact, args.limit), indent=2))
    elif args.command == "predict":
        vectorizer, model = load_artifact(artifact)
        row = vectorizer.transform([args.text])[0]
        probability = model.predict_proba_one(row)
        print(json.dumps({"label": "positive" if probability >= 0.5 else "negative",
                          "probability": round(probability, 4),
                          "explanation": explain(args.text, vectorizer, model)}, indent=2))
    elif args.command == "gpu-status":
        print(json.dumps(cuda_status(), indent=2))
    elif args.command == "run":
        if not (data_root / "aclImdb").exists():
            print("Downloading IMDb dataset...")
            download_dataset(data_root)
        print("Training accelerated model...")
        print(json.dumps(train_accelerated(
            data_root / "aclImdb", artifact, max_features=args.max_features,
            limit=args.limit, device=args.device, ngram_range=(1, 2),
        ), indent=2))
        metrics = evaluate(data_root / "aclImdb", artifact, args.limit)
        print("Evaluation:")
        print(json.dumps(metrics, indent=2))
        print("Starting CINESCOPE at http://127.0.0.1:8000")
        run_server = run
        run_server(artifact)
    else:
        run(artifact)


if __name__ == "__main__":
    main()
