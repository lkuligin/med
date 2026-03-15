CLUSTER_ALIASES = {
    "Clinical Decision-Making (including Risk Reduction, Surgery, and Surveillance) AND Peer experience seeking": (
        "Clinical Decision-Making (including Risk Reduction, Surgery, and Surveillance)"
    ),
}


def parse_labels(raw: str) -> set[str]:
    """Split a potentially multi-label ground truth string into a set of individual labels."""
    return CLUSTER_ALIASES.get(raw.strip(), raw.strip())


def pairwise_precision_recall(
    predicted: list[str], true_labels: list[str]
) -> tuple[float, float]:
    """Compute pairwise precision and recall for multi-label clustering evaluation.

    Two samples 'belong together' if they share at least one true label.
    Pairwise Precision: fraction of same-predicted-cluster pairs that share at least one true label.
    Pairwise Recall: fraction of truly-related pairs that share the same predicted cluster.
    """
    n = len(predicted)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = predicted[i] == predicted[j]
            true_pred = true_labels[i] == true_labels[j]
            if same_pred and true_pred:
                tp += 1
            elif same_pred and not true_pred:
                fp += 1
            elif not same_pred and true_pred:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def eval_results(predicted_labels: list[str], true_labels: list[str]) -> None:
    true_labels = [parse_labels(raw) for raw in true_labels]
    predicted_labels = [l for l in predicted_labels]

    # Pairwise metrics (multi-label aware)
    precision, recall = pairwise_precision_recall(predicted_labels, true_labels)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n=== Pairwise Clustering Metrics (multi-label) ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    # Hit rate: prediction counts as correct if it appears in any of the true labels
    hits = sum(
        1
        for pred, true_label in zip(predicted_labels, true_labels)
        if pred == true_label
    )
    total = len(predicted_labels)
    print("\n=== Multi-Label Hit Rate ===")
    print(f"Hit rate  : {hits}/{total} = {hits / total:.4f}")

    # Per-class metrics and micro-averaging
    all_labels = sorted(set(true_labels))

    total_tp = total_fp = total_fn = 0

    print("\n=== Per-class Metrics ===")
    print(f"  {'Label':<70}  {'P':>6}  {'R':>6}  {'F1':>6}  support")
    for label in all_labels:
        tp = sum(
            1
            for p, s in zip(predicted_labels, true_labels)
            if p == label and s == label
        )
        fp = sum(
            1
            for p, s in zip(predicted_labels, true_labels)
            if p == label and s != label
        )
        fn = sum(
            1
            for p, s in zip(predicted_labels, true_labels)
            if p != label and s == label
        )
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(
            f"  {label:<70}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  [{tp:2d}/{support:2d}]"
        )

    micro_acc = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    print(total_tp, total_fp, total_fn)
    print("\n=== Micro-Averaged Metrics ===")
    print(f"Accuracy : {micro_acc:.4f}")
