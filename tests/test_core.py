import json
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

from cinescope.explain import explain
from cinescope.metrics import classification_metrics
from cinescope.model import LogisticRegression
from cinescope.persistence import load_artifact, save_artifact
from cinescope.preprocess import normalize, tokenize
from cinescope.server import create_server
from cinescope.vectorizer import TfidfVectorizer


class CoreTests(unittest.TestCase):
    def test_preprocessing(self):
        self.assertEqual(tokenize("The GREAT movie! 42"), ["great", "movie"])
        self.assertEqual(normalize("A film, with joy."), "film joy")

    def test_training_persistence_and_explanation(self):
        texts = ["excellent joyful film", "wonderful acting", "awful boring film", "terrible acting"]
        labels = [1, 1, 0, 0]
        vectorizer = TfidfVectorizer(max_features=20, min_df=1)
        rows = vectorizer.fit_transform(texts)
        model = LogisticRegression(epochs=20).fit(rows, labels, len(vectorizer.vocabulary))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            save_artifact(path, vectorizer, model)
            loaded_vectorizer, loaded_model = load_artifact(path)
            row = loaded_vectorizer.transform(["excellent joyful film"])[0]
            self.assertEqual(loaded_model.predict_one(row), 1)
            self.assertTrue(explain("excellent film", loaded_vectorizer, loaded_model))

    def test_metrics(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["false_positive"], 1)

    def test_api_endpoints(self):
        vectorizer = TfidfVectorizer(max_features=10, min_df=1)
        rows = vectorizer.fit_transform(["good film", "bad film"])
        model = LogisticRegression(epochs=8).fit(rows, [1, 0], len(vectorizer.vocabulary))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            save_artifact(path, vectorizer, model)
            server = create_server(path, "127.0.0.1", 0)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            address = f"http://127.0.0.1:{server.server_port}"
            with urlopen(address + "/api/health") as response:
                self.assertEqual(json.load(response)["status"], "ok")
            request = Request(
                address + "/api/predict",
                data=json.dumps({"text": "good film"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                self.assertIn(json.load(response)["label"], {"positive", "negative"})
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
