"""Evaluation metrics without third-party dependencies."""

def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("prediction and target lengths must match and be non-empty")
    tp = sum(actual == predicted == 1 for actual, predicted in zip(y_true, y_pred))
    tn = sum(actual == predicted == 0 for actual, predicted in zip(y_true, y_pred))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}
