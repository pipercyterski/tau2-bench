"""
Maintainer tool: backfill cost/time breakdown metrics for text submissions.

For each text submission in the leaderboard manifest, downloads the public
trajectories from S3 (cached locally), recomputes the per-domain cost and
time breakdown (see AgentMetrics / build_domain_results), and patches the
corresponding web/leaderboard/public/submissions/<dir>/submission.json.
Trajectories on S3 are never modified; the patched submission.json files are
committed via PR and synced to S3 by CI.

Only the cost/time fields are touched; pass^k values and everything else in
the submission are left as-is. Existing cost values that differ from the
recomputed ones are overwritten (the old value is printed for review).

Usage:
    python -m tau2.scripts.leaderboard.backfill_cost_time_metrics \\
        [--submissions DIR ...] [--cache-dir PATH] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.metrics.agent_metrics import compute_metrics
from tau2.scripts.leaderboard.prepare_submission import build_domain_results
from tau2.scripts.leaderboard.submission import (
    MANIFEST_FILE_NAME,
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
)

S3_BUCKET = "sierra-tau-bench-public"
S3_PREFIX = "submissions"

REPO_ROOT = Path(__file__).resolve().parents[4]
SUBMISSIONS_DIR = REPO_ROOT / "web" / "leaderboard" / "public" / "submissions"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tau2" / "backfill_trajectories"

# The cost/time breakdown fields owned by this script (DomainResults fields).
COST_TIME_FIELDS = [
    "cost",
    "user_cost",
    "total_cost",
    "duration_seconds",
    "agent_time_seconds",
    "user_time_seconds",
    "tool_time_seconds",
]

console = Console()


def sync_trajectories(submission_name: str, cache_dir: Path) -> Path:
    """Download a submission's trajectories from public S3 (idempotent sync)."""
    dest = cache_dir / submission_name / TRAJECTORY_FILES_DIR_NAME
    dest.mkdir(parents=True, exist_ok=True)
    s3_url = (
        f"s3://{S3_BUCKET}/{S3_PREFIX}/{submission_name}/{TRAJECTORY_FILES_DIR_NAME}/"
    )
    console.print(f"  Syncing {s3_url}")
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            s3_url,
            str(dest),
            "--delete",
            "--no-sign-request",
            "--only-show-errors",
        ],
        check=True,
    )
    return dest


def _repair_trimmed_messages(data: dict) -> int:
    """Fix messages broken by older trim_trajectories passes.

    Some older S3 trajectory files were trimmed with the message ``id`` field
    stripped, but ``ToolMessage.id`` is required by the data model. Restore a
    placeholder so validation succeeds; the id is not used by any metric.
    Returns the number of repaired messages.
    """
    repaired = 0
    for sim in data.get("simulations", []):
        for msg in sim.get("messages") or []:
            if msg.get("role") == "tool" and "id" not in msg:
                msg["id"] = "trimmed"
                repaired += 1
    return repaired


def compute_domain_cost_time(results_path: Path) -> tuple[str, dict]:
    """Compute the cost/time field values for one domain's trajectory file.

    Returns (domain_name, {field: value}) with None values included so a
    field that can't be derived is explicitly recorded as absent.
    """
    with open(results_path) as f:
        raw = json.load(f)
    _repair_trimmed_messages(raw)
    results = TrajectoryResults.model_validate(raw)
    domain = results.info.environment_info.domain_name
    metrics = compute_metrics(results)
    domain_results = build_domain_results(metrics, include_time=True)
    values = {field: getattr(domain_results, field) for field in COST_TIME_FIELDS}
    # An agent cost of exactly 0.0 means per-message costs were never recorded
    # (e.g. self-hosted models); report the cost as unknown, not free.
    if values["cost"] == 0.0:
        values["cost"] = None
        values["total_cost"] = None
    return domain, values


def backfill_submission(submission_name: str, cache_dir: Path, dry_run: bool) -> bool:
    """Compute and patch cost/time metrics for one submission."""
    submission_file = SUBMISSIONS_DIR / submission_name / SUBMISSION_FILE_NAME
    if not submission_file.exists():
        console.print(f"  [red]No submission.json at {submission_file}[/red]")
        return False

    with open(submission_file) as f:
        data = json.load(f)

    if data.get("modality", "text") != "text":
        console.print("  [yellow]Skipping: not a text submission[/yellow]")
        return True
    if not data.get("trajectories_available", False):
        console.print("  [yellow]Skipping: trajectories not available[/yellow]")
        return True

    trajectories_dir = sync_trajectories(submission_name, cache_dir)
    results_files = sorted(trajectories_dir.glob("*.json"))
    if not results_files:
        console.print(
            "  [yellow]Skipping: no trajectory files on S3 "
            "(submission never uploaded trajectories)[/yellow]"
        )
        return True

    changed = False
    for results_path in results_files:
        try:
            domain, values = compute_domain_cost_time(results_path)
        except Exception as e:
            # Truncate: pydantic errors on big files can be megabytes long,
            # and rich takes minutes to render them.
            console.print(f"  [red]{results_path.name}: {str(e)[:500]}[/red]")
            return False

        domain_block = data.get("results", {}).get(domain)
        if domain_block is None:
            console.print(
                f"  [yellow]{domain}: in trajectories but not in submission.json; "
                "skipping[/yellow]"
            )
            continue

        summary = []
        for field, value in values.items():
            old = domain_block.get(field)
            if value is None:
                # Never erase an existing manually-provided value — except a
                # bogus 0.0 written by an earlier run of this script.
                if old == 0:
                    domain_block[field] = None
                    summary.append(f"{field}: {old} -> null")
                    changed = True
                continue
            if old is not None and old != value:
                summary.append(f"{field}: {old} -> {value}")
            elif old is None:
                summary.append(f"{field}: {value}")
            domain_block[field] = value
            changed = changed or old != value

        console.print(
            f"  {domain}: " + ("; ".join(summary) if summary else "no change")
        )

    if dry_run:
        console.print("  [yellow]Dry run: submission.json not modified[/yellow]")
        return True

    if changed:
        with open(submission_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        console.print(f"  [green]Patched {submission_file}[/green]")
    else:
        console.print("  No changes needed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Backfill cost/time metrics for text submissions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submissions",
        nargs="*",
        default=None,
        help="Submission directory names to backfill "
        "(default: all text submissions in the manifest)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Local cache for downloaded trajectories (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print metrics without modifying submission.json files",
    )
    args = parser.parse_args()

    if args.submissions:
        submission_names = args.submissions
    else:
        with open(SUBMISSIONS_DIR / MANIFEST_FILE_NAME) as f:
            manifest = json.load(f)
        submission_names = manifest.get("submissions", [])

    console.print(
        f"[bold blue]Backfilling {len(submission_names)} text submission(s)[/bold blue]"
    )

    failures = []
    for name in submission_names:
        console.print(f"\n[bold]{name}[/bold]")
        try:
            if not backfill_submission(name, Path(args.cache_dir), args.dry_run):
                failures.append(name)
        except Exception as e:
            console.print(f"  [red]FAILED: {str(e)[:500]}[/red]")
            failures.append(name)

    if failures:
        console.print(f"\n[red bold]Failed: {failures}[/red bold]")
        sys.exit(1)
    console.print("\n[green bold]All submissions backfilled.[/green bold]")


if __name__ == "__main__":
    main()
