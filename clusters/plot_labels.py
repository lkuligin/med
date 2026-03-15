import argparse
import json
import textwrap
from collections import Counter
from datetime import datetime, timezone

import matplotlib.pyplot as plt


def load_counts(filename):
    with open(filename) as f:
        data = json.load(f)
    labels = [item["predicted_label"] for item in data]
    return Counter(labels), len(labels)


def load_date_range(submissions_file, results_files):
    """Return (earliest, latest) datetime strings from submissions matching results."""
    with open(submissions_file) as f:
        raw = json.load(f)
    # submissions JSON is either a list or a dict with a single key holding a list
    if isinstance(raw, dict):
        submissions = next(iter(raw.values()))
    else:
        submissions = raw

    # build url -> created_utc lookup (match on path suffix)
    url_map = {}
    for item in submissions:
        url = item.get("url", "")
        # normalise to the path portion so it matches submission_id values
        path = url.replace("https://www.reddit.com", "").rstrip("/")
        url_map[path] = item.get("created_utc")

    # collect timestamps for all submission_ids present in any results file
    timestamps = []
    for rf in results_files:
        if rf is None:
            continue
        with open(rf) as f:
            data = json.load(f)
        for item in data:
            sid = item.get("submission_id", "").rstrip("/")
            if sid in url_map and url_map[sid] is not None:
                timestamps.append(url_map[sid])

    if not timestamps:
        return None, None

    def fmt(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    return fmt(min(timestamps)), fmt(max(timestamps))


def main():
    parser = argparse.ArgumentParser(description="Plot predicted label distributions.")
    parser.add_argument(
        "--gemini_file", required=True, help="Path to Gemini results JSON file"
    )
    parser.add_argument(
        "--anthropic_file", default=None, help="Path to Anthropic results JSON file"
    )
    parser.add_argument(
        "--output_path",
        default="label_distribution.png",
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--submissions_file",
        default=None,
        help="Path to submissions JSON for date range in title",
    )
    args = parser.parse_args()

    gemini_file = args.gemini_file
    anthropic_file = args.anthropic_file
    output_path = args.output_path

    date_suffix = ""
    if args.submissions_file:
        earliest, latest = load_date_range(
            args.submissions_file, [gemini_file, anthropic_file]
        )
        if earliest and latest:
            date_suffix = f"\n{earliest} – {latest}"

    gemini_counts, gemini_total = load_counts(gemini_file)

    if anthropic_file:
        anthropic_counts, anthropic_total = load_counts(anthropic_file)
        all_labels = sorted(
            set(gemini_counts) | set(anthropic_counts),
            key=lambda l: gemini_counts.get(l, 0) + anthropic_counts.get(l, 0),
        )

        display_labels = ["\n".join(textwrap.wrap(l, width=60)) for l in all_labels]
        gem_vals = [gemini_counts.get(l, 0) for l in all_labels]
        anth_vals = [anthropic_counts.get(l, 0) for l in all_labels]

        row_height = 0.9
        fig, ax = plt.subplots(figsize=(16, max(6, len(all_labels) * row_height)))

        bar_h = 0.35
        import numpy as np

        y = np.arange(len(all_labels))

        bars_gem = ax.barh(
            y + bar_h / 2, gem_vals, height=bar_h, color="steelblue", label="Gemini"
        )
        bars_anth = ax.barh(
            y - bar_h / 2,
            anth_vals,
            height=bar_h,
            color="darkorange",
            label="Anthropic",
        )

        max_val = max(max(gem_vals, default=0), max(anth_vals, default=0))
        for bar, val, total in [
            *zip(bars_gem, gem_vals, [gemini_total] * len(gem_vals)),
            *zip(bars_anth, anth_vals, [anthropic_total] * len(anth_vals)),
        ]:
            pct = val / total * 100
            ax.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=8,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(display_labels)
        ax.set_xlabel("Count")
        ax.set_title(
            f"Predicted Label Distribution\nGemini: {gemini_file}  |  Anthropic: {anthropic_file}{date_suffix}"
        )
        ax.set_xlim(0, max_val * 1.15)
        ax.legend()
    else:
        counts = gemini_counts
        total = gemini_total
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        labels_sorted, values_sorted = zip(*sorted_items)

        display_labels = ["\n".join(textwrap.wrap(l, width=60)) for l in labels_sorted]

        row_height = 0.7
        fig, ax = plt.subplots(figsize=(16, max(6, len(labels_sorted) * row_height)))
        bars = ax.barh(
            display_labels[::-1], values_sorted[::-1], color="steelblue", height=0.6
        )

        vals_reversed = list(values_sorted[::-1])
        pcts = [v / total * 100 for v in vals_reversed]
        for bar, val, pct in zip(bars, vals_reversed, pcts):
            ax.text(
                bar.get_width() + max(values_sorted) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.set_xlabel("Count")
        ax.set_title(f"Predicted Label Distribution ({gemini_file}){date_suffix}")
        ax.set_xlim(0, max(values_sorted) * 1.12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    n_labels = len(
        set(gemini_counts) | (set(anthropic_counts) if anthropic_file else set())
    )
    print(f"Saved to {output_path} ({n_labels} unique labels)")
    plt.show()


if __name__ == "__main__":
    main()
