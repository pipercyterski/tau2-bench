"""Plot accuracy-vs-cost and accuracy-vs-time Pareto views for text submissions.

Reads the patched submission.json files under web/leaderboard/public/submissions/
and produces scatter plots with the Pareto frontier highlighted:

  - pass^1 vs agent cost (USD/task), core domains and banking_knowledge
  - pass^1 vs agent execution time (s/task), core domains and banking_knowledge

Core = mean over airline, retail, telecom (only submissions covering all three).

Usage:
    uv run --with matplotlib python scripts/pareto_cost_time.py [--out-dir DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "web" / "leaderboard" / "public" / "submissions"

CORE_DOMAINS = ["airline", "retail", "telecom"]


def load_text_submissions() -> list[dict]:
    with open(SUBMISSIONS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    submissions = []
    for name in manifest["submissions"]:
        with open(SUBMISSIONS_DIR / name / "submission.json") as f:
            data = json.load(f)
        data["_dir"] = name
        submissions.append(data)
    return submissions


def mean_or_none(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def submission_label(sub: dict, name_counts: dict[str, int]) -> str:
    """Model name, disambiguated with reasoning effort when duplicated."""
    name = sub["model_name"]
    if name_counts.get(name, 0) > 1 and sub.get("reasoning_effort"):
        return f"{name} ({sub['reasoning_effort']})"
    return name


def extract_points(
    submissions: list[dict], bucket: str, x_field: str
) -> list[tuple[str, float, float]]:
    """Return (label, x, pass_1) points for a bucket ('core' or 'banking')."""
    name_counts: dict[str, int] = {}
    for sub in submissions:
        name_counts[sub["model_name"]] = name_counts.get(sub["model_name"], 0) + 1
    points = []
    for sub in submissions:
        results = sub.get("results", {})
        if bucket == "core":
            domains = [results.get(d) for d in CORE_DOMAINS]
            if any(d is None for d in domains):
                continue
            x = mean_or_none([d.get(x_field) for d in domains])
            y = mean_or_none([d.get("pass_1") for d in domains])
        else:
            d = results.get("banking_knowledge")
            if d is None:
                continue
            x, y = d.get(x_field), d.get("pass_1")
        if x is None or y is None or x <= 0:
            continue
        points.append((submission_label(sub, name_counts), x, y))
    return points


def pareto_frontier(
    points: list[tuple[str, float, float]],
) -> list[tuple[float, float]]:
    """Non-dominated points: no other point has lower x and higher y."""
    frontier = []
    best_y = -1.0
    for _, x, y in sorted(points, key=lambda p: (p[1], -p[2])):
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def plot_view(submissions: list[dict], x_field: str, x_label: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, bucket, title in [
        (axes[0], "core", "Core domains (airline / retail / telecom)"),
        (axes[1], "banking", "banking_knowledge"),
    ]:
        points = extract_points(submissions, bucket, x_field)
        if not points:
            ax.set_title(f"{title} — no data")
            continue
        frontier = pareto_frontier(points)
        frontier_set = set(frontier)
        for label, x, y in points:
            on_frontier = (x, y) in frontier_set
            ax.scatter(
                x,
                y,
                s=70 if on_frontier else 45,
                color="#d62728" if on_frontier else "#1f77b4",
                zorder=3,
            )
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
        fx, fy = zip(*frontier)
        ax.step(fx, fy, where="post", color="#d62728", alpha=0.5, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel(f"{x_label} (log scale)")
        ax.set_ylabel("pass^1 (%)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"pass^1 vs {x_label} — text track", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default="/tmp/tau2_pareto", help="Output directory"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    submissions = load_text_submissions()
    print(f"Loaded {len(submissions)} text submissions")

    plot_view(
        submissions,
        "cost",
        "agent LLM cost per task (USD)",
        out_dir / "pareto_pass1_vs_cost.png",
    )
    plot_view(
        submissions,
        "agent_time_seconds",
        "agent execution time per task (s)",
        out_dir / "pareto_pass1_vs_agent_time.png",
    )
    plot_view(
        submissions,
        "duration_seconds",
        "total task duration (s)",
        out_dir / "pareto_pass1_vs_duration.png",
    )


if __name__ == "__main__":
    main()
