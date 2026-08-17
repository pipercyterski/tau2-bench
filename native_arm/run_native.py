"""Arm B of the port experiment: the same agent, run in tau2's own environment.

The experiment is a two-arm comparison on identical tasks:

* **Arm A** -- the agent driven by Dystopic against the declared, simulated
  world exported by ``dystopic_export/``, scored by Dystopic scorers.
* **Arm B** -- the same agent driven by ``tau2 run`` against tau2's real
  Python environment, scored by tau2's own reward.

Arm B is this module. We are *not* reimplementing tau2's reward: we shell out
to the benchmark's own CLI and read back the artifact it writes. Everything
here is either (a) building the exact command, or (b) parsing
``tau2.data_model.simulation.Results`` into a task-keyed shape the comparison
layer can join against Arm A.

Why a wrapper at all, when ``tau2 run`` is one line? Three reasons, and they
are the whole value of this file:

1. **The join key.** tau2 writes a flat list of ``SimulationRun``s, one per
   (task, trial). The comparison joins on ``task_id``, so somebody has to
   regroup and aggregate (mean reward, pass^k) -- and it should be done once,
   in the same way, forever.
2. **Parity is not self-evident.** A native run and a Dystopic run are only
   comparable if a specific list of knobs matches (see ``PARITY_KNOBS``).
   Nothing in tau2 checks that for you, and a mismatched user-simulator model
   silently produces a *plausible* number that means nothing. So the loader
   refuses to normalize an artifact whose ``Info`` disagrees with the cell it
   claims to be.
3. **The split is not in the artifact.** ``Info`` records the domain, both
   LLMs, trials, steps and seed -- but *not* ``task_split_name``. It is
   recoverable only by comparing the artifact's task ids against
   ``split_tasks.json``, which ``infer_split()`` does. Without that, "did we
   run test or base?" is unanswerable after the fact.

Running a real cell costs real money. ``run`` refuses to execute without
``--yes``; ``command`` prints the exact line a human would run plus a cost and
wall-clock estimate derived from the shipped paper artifacts, and ``normalize``
works entirely offline.

    # what would it cost / what is the command
    python -m native_arm.run_native command --cell retail-test

    # actually spend money
    python -m native_arm.run_native run --cell retail-test --yes

    # parse an artifact (offline; works on the shipped paper results too)
    python -m native_arm.run_native normalize \
        --in data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json \
        --adopt paper-retail-base-gpt41 --allow-commit-drift
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tau2.data_model.simulation import Info, Results, TerminationReason  # noqa: E402
from tau2.metrics.agent_metrics import is_successful, pass_hat_k  # noqa: E402
from tau2.runner.helpers import load_task_splits  # noqa: E402
from tau2.utils.utils import DATA_DIR  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The fork is pinned. tau2 stamps ``info.git_commit`` into every artifact, so a
# rebase that changes the environment, the tools or the evaluator is detectable
# after the fact -- which matters, because Arm A's world was exported from this
# exact tree. Drift here invalidates the comparison even if every other knob
# matches.
PINNED_TAU2_COMMIT = "5ebebbe827b455b3ed04fcb9294235c6ef4e5fd6"
PINNED_TAU2_VERSION = "1.0.0"  # pyproject [project].version -- "tau3-bench v1.0.0"


#: The knobs that must be identical in Arm A for a comparison to mean anything.
#: Each entry is (knob, where it lives in the native artifact, why it matters).
PARITY_KNOBS: tuple[tuple[str, str, str], ...] = (
    (
        "agent model",
        "info.agent_info.llm",
        "The thing under test. Different model, different experiment.",
    ),
    (
        "agent implementation",
        "info.agent_info.implementation",
        "llm_agent is the plain ReAct-ish loop. llm_agent_solo/llm_agent_gt "
        "are ablations that remove the user or hand over an oracle plan.",
    ),
    (
        "agent temperature",
        "info.agent_info.llm_args.temperature",
        "tau2 runs 0.0. Sampling temperature is a confound across arms.",
    ),
    (
        "user-simulator model",
        "info.user_info.llm",
        "The user sim IS half the environment: it decides what gets revealed, "
        "when, and when to say ###STOP###. A weaker user sim caps the agent's "
        "achievable reward. Arm A must drive the same simulator model.",
    ),
    (
        "user implementation",
        "info.user_info.implementation",
        "dummy_user is the no-user ablation; user_simulator is the real one.",
    ),
    (
        "task split",
        "NOT STORED -- recover via infer_split()",
        "retail ships base(114) / train(74) / test(40). Comparing a test-split "
        "Arm A against a base-split Arm B compares different task mixes.",
    ),
    (
        "task ids",
        "results.tasks[].id",
        "The join is per task_id, so any task present in one arm and absent "
        "from the other must be dropped explicitly, not averaged over.",
    ),
    (
        "num_trials",
        "info.num_trials",
        "pass^k is only defined at equal k. Mean reward over 1 trial and over "
        "4 trials are different estimators with different variance.",
    ),
    (
        "max_steps",
        "info.max_steps",
        "The truncation budget. A tighter budget converts solvable tasks into "
        "max_steps terminations, which score 0.",
    ),
    (
        "max_errors",
        "info.max_errors",
        "Consecutive tool-error budget before the run is abandoned.",
    ),
    (
        "seed",
        "info.seed",
        "Seeds the per-trial seeds. Same seed does NOT make LLM sampling "
        "deterministic, but it does keep trial indexing aligned.",
    ),
    (
        "tau2 commit",
        "info.git_commit",
        "Pins the domain DB, the tool implementations and the evaluator. "
        "Arm A's world was exported from this tree.",
    ),
    (
        "reward_basis",
        "derived: tasks[].evaluation_criteria.reward_basis, per task id",
        "The scoring contract itself, not a version knob. DB+COMMUNICATE and "
        "DB+NL_ASSERTION grade different things (the paper-era artifacts "
        "shipped with this repo were scored under the former; this pinned "
        "checkout scores retail under the latter on 112/114 base tasks). A "
        "pass rate computed under one basis is not comparable to a pass rate "
        "computed under the other no matter how well every other knob lines "
        "up, so this is checked even when --allow-commit-drift is passed.",
    ),
)


class ParityError(RuntimeError):
    """An artifact's recorded configuration disagrees with the cell it claims."""


@dataclass(frozen=True)
class Cell:
    """One comparable experiment cell. Named, so both arms can refer to it.

    Defaults are tau2's own CLI defaults, spelled out rather than imported so
    the record of what we ran survives a dependency bump.
    """

    name: str
    domain: str = "retail"
    task_split: str = "test"
    agent_llm: str = "gpt-4.1-2025-04-14"
    user_llm: str = "gpt-4.1-2025-04-14"
    num_trials: int = 4
    agent: str = "llm_agent"
    user: str = "user_simulator"
    agent_temperature: float = 0.0
    user_temperature: float = 0.0
    max_steps: int = 200
    max_errors: int = 10
    seed: int = 300
    num_tasks: Optional[int] = None
    task_ids: tuple[str, ...] = field(default_factory=tuple)
    # Overridden only when adopting an artifact produced on another tree; a
    # cell we run ourselves is always on the pin.
    tau2_commit: str = PINNED_TAU2_COMMIT

    @property
    def save_to(self) -> str:
        """The ``--save-to`` name, hence ``data/simulations/<save_to>/``."""
        return f"dystopic_native_{self.name}"

    @property
    def artifact_path(self) -> Path:
        return RESULTS_DIR / f"{self.name}.json"

    @property
    def normalized_path(self) -> Path:
        return RESULTS_DIR / f"{self.name}.normalized.json"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "task_split": self.task_split,
            "agent": self.agent,
            "agent_llm": self.agent_llm,
            "agent_temperature": self.agent_temperature,
            "user": self.user,
            "user_llm": self.user_llm,
            "user_temperature": self.user_temperature,
            "num_trials": self.num_trials,
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
            "seed": self.seed,
            "num_tasks": self.num_tasks,
            "task_ids": list(self.task_ids) or None,
            "tau2_commit": self.tau2_commit,
            "tau2_commit_is_pinned": self.tau2_commit == PINNED_TAU2_COMMIT,
            "tau2_version": PINNED_TAU2_VERSION,
        }


#: Pre-registered cells. Retail only -- airline is deferred (see contract.py).
CELLS: dict[str, Cell] = {
    # Plumbing check. Five tasks, one trial: enough to prove the command, the
    # artifact path and the loader, cheap enough to not need a conversation.
    "retail-smoke": Cell(
        name="retail-smoke", task_split="test", num_trials=1, num_tasks=5
    ),
    # The comparison cell. 40 held-out tasks x 4 trials, so pass^1..pass^4 are
    # all defined and the headline can be pass^1 against the published number.
    "retail-test": Cell(name="retail-test", task_split="test", num_trials=4),
    # Full paper parity: the 114-task base split, directly comparable to the
    # shipped data/tau2/results/final/ artifacts.
    "retail-base": Cell(name="retail-base", task_split="base", num_trials=4),
}


# --- cost / wall-clock model ------------------------------------------------
#
# Measured, not guessed: these are the per-simulation means over all 456
# simulations of the shipped paper artifact
# data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json
# (retail, base split, gpt-4.1 agent + gpt-4.1 user, 4 trials).
MEASURED_RETAIL_GPT41 = {
    "agent_cost_usd": 0.0582,
    "user_cost_usd": 0.0133,
    "duration_s": 63.3,
    "num_messages": 26.6,
    "num_tool_calls": 7.7,
}


def split_size(domain: str, split: str) -> int:
    splits = load_task_splits(domain)
    if splits is None or split not in splits:
        raise ValueError(f"no split {split!r} for domain {domain!r}")
    return len(splits[split])


def estimate(cell: Cell, max_concurrency: int) -> dict[str, Any]:
    """Rough cost and wall clock for a cell, from measured per-sim means.

    Only honest for gpt-4.1-class models: the per-simulation cost scales with
    the agent model's price, and reasoning models also inflate the message
    count. Treat the dollar figure as an order of magnitude for anything else.
    """
    n_tasks = cell.num_tasks or split_size(cell.domain, cell.task_split)
    if cell.task_ids:
        n_tasks = len(cell.task_ids)
    n_sims = n_tasks * cell.num_trials
    m = MEASURED_RETAIL_GPT41
    return {
        "num_tasks": n_tasks,
        "num_trials": cell.num_trials,
        "num_simulations": n_sims,
        "est_agent_cost_usd": round(n_sims * m["agent_cost_usd"], 2),
        "est_user_cost_usd": round(n_sims * m["user_cost_usd"], 2),
        "est_total_cost_usd": round(
            n_sims * (m["agent_cost_usd"] + m["user_cost_usd"]), 2
        ),
        "est_serial_hours": round(n_sims * m["duration_s"] / 3600, 2),
        "est_wallclock_hours": round(
            n_sims * m["duration_s"] / 3600 / max(1, max_concurrency), 2
        ),
        "est_llm_turns": int(n_sims * m["num_messages"]),
        "est_tool_calls": int(n_sims * m["num_tool_calls"]),
        "basis": "measured per-simulation means from the shipped gpt-4.1 retail paper artifact",
    }


# --- running ----------------------------------------------------------------


def tau2_entrypoint() -> list[str]:
    """``tau2`` if the console script is on PATH, else the module."""
    exe = shutil.which("tau2")
    return [exe] if exe else [sys.executable, "-m", "tau2.cli"]


def build_command(
    cell: Cell,
    *,
    max_concurrency: int = 3,
    log_level: str = "ERROR",
    auto_resume: bool = True,
) -> list[str]:
    """The exact ``tau2 run`` invocation for a cell.

    Every knob in PARITY_KNOBS that the CLI can set is set explicitly, even
    where it equals the current default -- a default that moves must not
    silently move the experiment.
    """
    argv = [
        *tau2_entrypoint(),
        "run",
        "--domain",
        cell.domain,
        "--task-split-name",
        cell.task_split,
        "--agent",
        cell.agent,
        "--agent-llm",
        cell.agent_llm,
        "--agent-llm-args",
        json.dumps({"temperature": cell.agent_temperature}),
        "--user",
        cell.user,
        "--user-llm",
        cell.user_llm,
        "--user-llm-args",
        json.dumps({"temperature": cell.user_temperature}),
        "--num-trials",
        str(cell.num_trials),
        "--max-steps",
        str(cell.max_steps),
        "--max-errors",
        str(cell.max_errors),
        "--seed",
        str(cell.seed),
        "--max-concurrency",
        str(max_concurrency),
        "--log-level",
        log_level,
        "--save-to",
        cell.save_to,
    ]
    if cell.num_tasks is not None:
        argv += ["--num-tasks", str(cell.num_tasks)]
    if cell.task_ids:
        argv += ["--task-ids", *cell.task_ids]
    if auto_resume:
        # Without this the runner blocks on an interactive prompt when a
        # partial artifact exists, which deadlocks any non-tty invocation.
        argv += ["--auto-resume"]
    return argv


def native_artifact_path(cell: Cell) -> Path:
    """Where ``tau2 run`` itself writes: ``DATA_DIR/simulations/<save_to>/``.

    Text runs use the monolithic format, so it is a single ``results.json``
    (``runner/batch.py`` picks "dir" only for voice runs).
    """
    return DATA_DIR / "simulations" / cell.save_to / "results.json"


def run_cell(
    cell: Cell,
    *,
    max_concurrency: int = 3,
    log_level: str = "ERROR",
    auto_resume: bool = True,
) -> Path:
    """Run the cell natively and copy the artifact under ``native_arm/results``.

    tau2 owns its own output tree and its own checkpointing; we copy rather
    than redirect so a resumed or re-run cell keeps working and this directory
    stays a snapshot rather than live state.
    """
    argv = build_command(
        cell,
        max_concurrency=max_concurrency,
        log_level=log_level,
        auto_resume=auto_resume,
    )
    print("+", shlex.join(argv))
    subprocess.run(argv, cwd=REPO_ROOT, check=True)

    produced = native_artifact_path(cell)
    if not produced.exists():
        raise FileNotFoundError(f"tau2 run produced no artifact at {produced}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, cell.artifact_path)
    return cell.artifact_path


# --- parity -----------------------------------------------------------------


def infer_split(domain: str, task_ids: set[str]) -> Optional[str]:
    """Recover ``task_split_name``, which the artifact does not record.

    Returns the split whose id set equals the artifact's, or -- when the run
    was narrowed by ``--num-tasks``/``--task-ids`` -- the unique split that
    contains all of them. ``None`` if that is ambiguous or unsatisfiable.
    """
    splits = load_task_splits(domain)
    if not splits:
        return None
    for name, ids in splits.items():
        if set(ids) == task_ids:
            return name
    supersets = [name for name, ids in splits.items() if task_ids <= set(ids)]
    # "base" contains train+test, so a train-only subset is in both; prefer the
    # tightest containing split, and give up if that is still not unique.
    if supersets:
        supersets.sort(key=lambda n: len(splits[n]))
        if len(supersets) == 1 or len(splits[supersets[0]]) < len(splits[supersets[1]]):
            return supersets[0]
    return None


def _reward_basis_key(basis: Any) -> str:
    """``[RewardType.DB, RewardType.NL_ASSERTION]`` -> ``"DB,NL_ASSERTION"``.

    A sorted, comma-joined string rather than a tuple so it serializes
    cleanly into the normalized artifact and diffs as plain text in a
    failure message; sorted so equal-but-differently-ordered lists don't
    read as a mismatch.
    """
    return ",".join(sorted(b.value if hasattr(b, "value") else b for b in (basis or []))) or "(none)"


def reward_basis_distribution(task_bases: list[Any]) -> dict[str, int]:
    """``{"DB,NL_ASSERTION": 112, "DB": 2}`` -- how many tasks carry each
    ``reward_basis`` combination, one entry per element of ``task_bases``.

    This *is* the scoring contract a run was evaluated under: two task sets
    that differ here are not measuring the same thing, no matter how many of
    the other knobs line up.
    """
    counts: Counter[str] = Counter(_reward_basis_key(basis) for basis in task_bases)
    return dict(sorted(counts.items()))


def _pinned_reward_basis_by_id(domain: str) -> dict[str, Any]:
    """task_id -> reward_basis, as declared on the pinned tau2 checkout.

    Loaded from retail's ``base`` split -- the train+test superset -- rather
    than a single named split, so it resolves any task id an artifact might
    carry, including an arbitrary ``--task-ids``/``--num-tasks`` subset that
    lines up with no named split.
    """
    if domain != "retail":  # pragma: no cover - guarded by CELLS
        raise ValueError(f"unsupported domain {domain!r}")
    from tau2.domains.retail.environment import get_tasks

    return {
        str(t.id): (t.evaluation_criteria.reward_basis if t.evaluation_criteria else [])
        for t in get_tasks("base")
    }


def check_reward_basis(cell: Cell, meta: Results) -> Optional[str]:
    """The one parity problem that must never be waived.

    Compares the ``reward_basis`` an artifact was actually scored under
    (recorded per task on the artifact itself, in ``meta.tasks``) against
    what the pinned tau2 checkout declares for those same task ids today.
    They can disagree even when every other knob matches: the paper-era
    result artifact checked into this repo is scored ``[DB, COMMUNICATE]``,
    while this checkout scores retail ``[DB, NL_ASSERTION]`` on 112/114
    tasks -- upstream changed which fields gate the reward between those two
    points, and averaging pass rates across that boundary is comparing two
    different metrics, not two runs of the same experiment.
    """
    artifact_basis = reward_basis_distribution(
        [t.evaluation_criteria.reward_basis if t.evaluation_criteria else [] for t in meta.tasks]
    )
    try:
        pinned_by_id = _pinned_reward_basis_by_id(cell.domain)
    except Exception as exc:  # pragma: no cover - env problem, not a defect
        return f"reward_basis: could not load the pinned task set to compare against: {exc!r}"

    task_ids = [str(t.id) for t in meta.tasks]
    pinned_basis = reward_basis_distribution(
        [pinned_by_id[tid] for tid in task_ids if tid in pinned_by_id]
    )
    if artifact_basis == pinned_basis:
        return None
    return (
        f"reward_basis: artifact was scored under {artifact_basis} but the "
        f"pinned tau2 checkout ({PINNED_TAU2_COMMIT[:12]}) grades these same "
        f"{len(task_ids)} task ids under {pinned_basis} today -- these are "
        "different scoring metrics (e.g. DB+COMMUNICATE vs DB+NL_ASSERTION), "
        "so a pass rate under one is not comparable to a pass rate under the "
        "other; --allow-commit-drift does not waive this"
    )


def check_parity(cell: Cell, meta: Results) -> list[str]:
    """Return the list of parity violations between a cell and an artifact.

    Empty list means the artifact really is the cell it claims to be, on every
    knob tau2 records plus the split we recover ourselves.
    """
    info: Info = meta.info
    problems: list[str] = []

    def cmp(label: str, want: Any, got: Any) -> None:
        if want != got:
            problems.append(f"{label}: cell={want!r} artifact={got!r}")

    cmp("domain", cell.domain, info.environment_info.domain_name)
    cmp("agent implementation", cell.agent, info.agent_info.implementation)
    cmp("agent model", cell.agent_llm, info.agent_info.llm)
    cmp(
        "agent temperature",
        cell.agent_temperature,
        (info.agent_info.llm_args or {}).get("temperature"),
    )
    cmp("user implementation", cell.user, info.user_info.implementation)
    cmp("user model", cell.user_llm, info.user_info.llm)
    cmp(
        "user temperature",
        cell.user_temperature,
        (info.user_info.llm_args or {}).get("temperature"),
    )
    cmp("num_trials", cell.num_trials, info.num_trials)
    cmp("max_steps", cell.max_steps, info.max_steps)
    cmp("max_errors", cell.max_errors, info.max_errors)
    cmp("seed", cell.seed, info.seed)
    cmp("tau2 commit", cell.tau2_commit, info.git_commit)
    if cell.tau2_commit != PINNED_TAU2_COMMIT:
        problems.append(
            f"tau2 commit pin: cell was produced at {cell.tau2_commit} but the "
            f"port is pinned at {PINNED_TAU2_COMMIT}; Arm A's world was "
            "exported from the pinned tree"
        )

    task_ids = {str(t.id) for t in meta.tasks}
    found_split = infer_split(cell.domain, task_ids)
    if cell.num_tasks is None and not cell.task_ids:
        cmp("task split", cell.task_split, found_split)
    elif found_split not in (None, cell.task_split):
        problems.append(
            f"task split: cell={cell.task_split!r} artifact tasks live in {found_split!r}"
        )
    if cell.task_ids and set(cell.task_ids) != task_ids:
        problems.append(
            f"task ids: cell={sorted(cell.task_ids)} artifact={sorted(task_ids)}"
        )
    if cell.num_tasks is not None and len(task_ids) != cell.num_tasks:
        problems.append(f"num_tasks: cell={cell.num_tasks} artifact={len(task_ids)}")

    if reward_basis_problem := check_reward_basis(cell, meta):
        problems.append(reward_basis_problem)
    return problems


def cell_from_results(name: str, meta: Results) -> Cell:
    """Reconstruct the cell an existing artifact represents.

    For adopting artifacts nobody ran through this wrapper -- the shipped
    paper results, or a run someone did by hand.
    """
    info = meta.info
    task_ids = {str(t.id) for t in meta.tasks}
    return Cell(
        name=name,
        domain=info.environment_info.domain_name,
        task_split=infer_split(info.environment_info.domain_name, task_ids)
        or "unknown",
        agent=info.agent_info.implementation,
        agent_llm=info.agent_info.llm,
        agent_temperature=(info.agent_info.llm_args or {}).get("temperature"),
        user=info.user_info.implementation,
        user_llm=info.user_info.llm,
        user_temperature=(info.user_info.llm_args or {}).get("temperature"),
        num_trials=info.num_trials,
        max_steps=info.max_steps,
        max_errors=info.max_errors,
        seed=info.seed,
        tau2_commit=info.git_commit,
    )


# --- loading ----------------------------------------------------------------


def _enum_value(value: Any) -> Any:
    """``RewardType.DB`` -> ``"DB"``.

    tau2's enums subclass ``(str, Enum)``, not ``StrEnum``, so ``str()`` on
    them yields ``"RewardType.DB"`` under Python 3.12. The wire format must
    carry the value, because Arm A's side of the join has never heard of
    tau2's class names.
    """
    return value.value if hasattr(value, "value") else value


def _normalize_message(msg: Any) -> dict[str, Any]:
    """One trajectory entry, provider noise stripped.

    ``raw_data`` is the verbatim provider response and roughly doubles the
    artifact size while carrying nothing the comparison uses; token ``usage``
    is kept because per-turn tokens are the only cost signal on the Arm A side.
    """
    out: dict[str, Any] = {
        "role": msg.role,
        "turn_idx": getattr(msg, "turn_idx", None),
        "content": getattr(msg, "content", None),
    }
    calls = getattr(msg, "tool_calls", None)
    if calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "name": c.name,
                "arguments": c.arguments,
                "requestor": c.requestor,
            }
            for c in calls
        ]
    if msg.role == "tool":
        out["tool_call_id"] = getattr(msg, "id", None)
        out["requestor"] = getattr(msg, "requestor", None)
        out["error"] = getattr(msg, "error", None)
    for optional in ("cost", "usage"):
        value = getattr(msg, optional, None)
        if value is not None:
            out[optional] = value
    return out


def load_trials(
    path: Path, *, include_messages: bool = True
) -> Iterator[dict[str, Any]]:
    """Stream one record per (task, trial) simulation out of a tau2 artifact.

    Streams rather than ``Results.load()`` because a 4-trial retail artifact is
    ~22 MB and telecom's are ~50 MB; ``iter_simulations`` keeps peak memory at
    one simulation. Handles both the monolithic and directory formats.
    """
    for sim in Results.iter_simulations(path):
        messages = sim.get_messages()
        tool_calls = [
            c for m in messages for c in (getattr(m, "tool_calls", None) or [])
        ]
        reward = sim.reward_info

        record: dict[str, Any] = {
            "task_id": str(sim.task_id),
            "trial": sim.trial,
            "simulation_id": sim.id,
            "seed": sim.seed,
            "reward": reward.reward if reward else None,
            "success": is_successful(reward.reward) if reward else False,
            # The basis IS the scoring contract: reward is the product of these
            # components, and they are per task, not per run.
            "reward_basis": (
                [_enum_value(b) for b in (reward.reward_basis or [])] if reward else []
            ),
            "reward_breakdown": (
                {_enum_value(k): v for k, v in (reward.reward_breakdown or {}).items()}
                if reward
                else {}
            ),
            "termination_reason": _enum_value(sim.termination_reason),
            "num_messages": len(messages),
            "num_agent_turns": sum(1 for m in messages if m.role == "assistant"),
            "num_user_turns": sum(1 for m in messages if m.role == "user"),
            "num_tool_calls": len(tool_calls),
            "num_agent_tool_calls": sum(
                1 for c in tool_calls if c.requestor == "assistant"
            ),
            "num_user_tool_calls": sum(1 for c in tool_calls if c.requestor == "user"),
            "num_tool_errors": sum(1 for m in messages if getattr(m, "error", False)),
            "agent_cost_usd": sim.agent_cost,
            "user_cost_usd": sim.user_cost,
            "duration_s": sim.duration,
        }

        if reward:
            if reward.db_check:
                record["db_check"] = {
                    "db_match": reward.db_check.db_match,
                    "db_reward": reward.db_check.db_reward,
                }
            if reward.communicate_checks:
                record["communicate_checks"] = [
                    {"info": c.info, "met": c.met} for c in reward.communicate_checks
                ]
            if reward.env_assertions:
                record["env_assertions"] = [
                    {"met": c.met, "reward": c.reward} for c in reward.env_assertions
                ]
            if reward.nl_assertions:
                record["nl_assertions"] = [
                    {"nl_assertion": c.nl_assertion, "met": c.met}
                    for c in reward.nl_assertions
                ]
            # Diagnostic only: ACTION is never in retail's reward_basis, so this
            # is similarity to ONE reference trajectory, not a correctness
            # verdict (docs/evaluation.md). Carried because it is the closest
            # native analogue to Arm A's per-tool-call scorers.
            if reward.action_checks:
                record["partial_action_reward"] = reward.partial_action_reward

        if include_messages:
            record["messages"] = [_normalize_message(m) for m in messages]
        yield record


def group_by_task(trials: Iterator[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Regroup per-trial records into the task-keyed shape the join needs."""
    tasks: dict[str, dict[str, Any]] = {}
    for record in trials:
        tasks.setdefault(
            record["task_id"], {"task_id": record["task_id"], "trials": []}
        )["trials"].append(record)

    for task in tasks.values():
        task["trials"].sort(key=lambda r: (r["trial"] is None, r["trial"]))
        # tau2 drops infrastructure errors before computing metrics -- a
        # simulation that never ran is not a failure, and averaging it in as a
        # zero would understate the arm.
        scored = [
            t
            for t in task["trials"]
            if t["termination_reason"] != TerminationReason.INFRASTRUCTURE_ERROR.value
        ]
        rewards = [t["reward"] for t in scored if t["reward"] is not None]
        successes = sum(1 for t in scored if t["success"])
        task["num_trials"] = len(scored)
        task["num_infra_error_trials"] = len(task["trials"]) - len(scored)
        task["rewards"] = rewards
        task["avg_reward"] = sum(rewards) / len(rewards) if rewards else None
        task["num_successes"] = successes
        task["pass_hat_k"] = {
            str(k): pass_hat_k(len(scored), successes, k)
            for k in range(1, len(scored) + 1)
        }
        task["reward_basis"] = scored[0]["reward_basis"] if scored else []
    return tasks


def normalize(
    cell: Cell,
    path: Path,
    *,
    include_messages: bool = True,
    parity_problems: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the task-keyed document the comparison layer joins Arm A against."""
    meta = Results.load_metadata(path)
    tasks = group_by_task(load_trials(path, include_messages=include_messages))
    scored = [t for t in tasks.values() if t["avg_reward"] is not None]

    return {
        "arm": "native",
        "cell": cell.as_dict(),
        "parity": {
            "knobs": [
                {"knob": k, "recorded_at": w, "why": why} for k, w, why in PARITY_KNOBS
            ],
            "problems": parity_problems or [],
            "artifact_commit": meta.info.git_commit,
            "commit_matches_pin": meta.info.git_commit == PINNED_TAU2_COMMIT,
            # The reward_basis distribution this artifact's own task set was
            # scored under (see check_reward_basis) -- recorded here, not just
            # checked, so a downstream comparison can tell at a glance which
            # metric a normalized artifact represents.
            "artifact_reward_basis": reward_basis_distribution(
                [t.evaluation_criteria.reward_basis if t.evaluation_criteria else [] for t in meta.tasks]
            ),
        },
        "source": {
            "path": str(path),
            "timestamp": meta.timestamp,
            "domain": meta.info.environment_info.domain_name,
            "agent_llm": meta.info.agent_info.llm,
            "user_llm": meta.info.user_info.llm,
            "num_trials": meta.info.num_trials,
        },
        "summary": {
            "num_tasks": len(tasks),
            "num_simulations": sum(len(t["trials"]) for t in tasks.values()),
            "avg_reward": (
                sum(t["avg_reward"] for t in scored) / len(scored) if scored else None
            ),
            "pass_hat_1": (
                sum(t["pass_hat_k"].get("1", 0.0) for t in scored) / len(scored)
                if scored
                else None
            ),
            "includes_messages": include_messages,
        },
        "tasks": tasks,
    }


# --- cli --------------------------------------------------------------------


def _resolve_cell(args: argparse.Namespace) -> Cell:
    if args.cell not in CELLS:
        raise SystemExit(f"unknown cell {args.cell!r}; known: {', '.join(CELLS)}")
    cell = CELLS[args.cell]
    overrides = {
        k: getattr(args, k)
        for k in ("agent_llm", "user_llm", "num_trials", "task_split", "num_tasks")
        if getattr(args, k, None) is not None
    }
    if getattr(args, "task_ids", None):
        overrides["task_ids"] = tuple(args.task_ids)
    return replace(cell, **overrides) if overrides else cell


def normalize_to_disk(
    cell: Cell,
    path: Path,
    out: Optional[Path] = None,
    *,
    include_messages: bool = True,
    allow_commit_drift: bool = False,
    allow_mismatch: bool = False,
) -> Path:
    """Check parity, normalize, write, and print a one-screen summary.

    Shared by ``run`` and ``normalize`` so a run whose cell was overridden on
    the command line is checked against the cell it actually ran, not against
    the registry entry of the same name.
    """
    meta = Results.load_metadata(path)
    problems = check_parity(cell, meta)
    if allow_commit_drift:
        problems = [p for p in problems if not p.startswith("tau2 commit pin:")]
    if problems and not allow_mismatch:
        raise ParityError(
            f"{path} is not comparable as cell {cell.name!r}:\n  "
            + "\n  ".join(problems)
            + "\n(pass --allow-mismatch to normalize anyway; the problems are "
            "recorded in the output)"
        )

    doc = normalize(
        cell, path, include_messages=include_messages, parity_problems=problems
    )
    out = out or cell.normalized_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True))

    summary = doc["summary"]
    print(f"wrote {out} ({out.stat().st_size / 1_000_000:.1f} MB)")
    print(f"  cell: {cell.name}  split: {cell.task_split}  trials: {cell.num_trials}")
    print(f"  tasks: {summary['num_tasks']}  simulations: {summary['num_simulations']}")
    print(
        f"  avg_reward: {summary['avg_reward']:.4f}  pass^1: {summary['pass_hat_1']:.4f}"
    )
    for problem in problems:
        print(f"  PARITY: {problem}")
    return out


def _add_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cell", required=True, choices=sorted(CELLS))
    parser.add_argument("--agent-llm")
    parser.add_argument("--user-llm")
    parser.add_argument("--task-split")
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--max-concurrency", type=int, default=3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_cmd = sub.add_parser("command", help="print the command and a cost estimate")
    _add_cell_args(p_cmd)

    p_run = sub.add_parser("run", help="actually run the cell (SPENDS MONEY)")
    _add_cell_args(p_run)
    p_run.add_argument("--log-level", default="ERROR")
    p_run.add_argument(
        "--yes", action="store_true", help="confirm the spend; required to execute"
    )
    p_run.add_argument(
        "--no-normalize", action="store_true", help="skip the normalize step"
    )

    p_norm = sub.add_parser(
        "normalize", help="parse an artifact into the joinable shape"
    )
    p_norm.add_argument("--cell", choices=sorted(CELLS))
    p_norm.add_argument(
        "--adopt",
        metavar="NAME",
        help="derive the cell from the artifact itself, under this name",
    )
    p_norm.add_argument("--in", dest="input", type=Path, help="artifact path")
    p_norm.add_argument("--out", type=Path, help="output path")
    p_norm.add_argument("--no-messages", action="store_true")
    p_norm.add_argument(
        "--allow-commit-drift",
        action="store_true",
        help="tolerate an adopted artifact built off the pinned commit "
        "(does not excuse a registered cell claiming the pin)",
    )
    p_norm.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="record parity problems instead of refusing",
    )

    args = ap.parse_args(argv)

    if args.command == "command":
        cell = _resolve_cell(args)
        # shlex.join, not " ".join: the --*-llm-args values are JSON and must
        # survive a copy-paste into a shell intact.
        print(shlex.join(build_command(cell, max_concurrency=args.max_concurrency)))
        print()
        for key, value in estimate(cell, args.max_concurrency).items():
            print(f"  {key}: {value}")
        return 0

    if args.command == "run":
        cell = _resolve_cell(args)
        est = estimate(cell, args.max_concurrency)
        if not args.yes:
            print(f"refusing to run {cell.name} without --yes.")
            print(
                f"  {est['num_simulations']} simulations, "
                f"~${est['est_total_cost_usd']} of API spend, "
                f"~{est['est_wallclock_hours']}h wall clock at concurrency {args.max_concurrency}"
            )
            return 1
        path = run_cell(
            cell, max_concurrency=args.max_concurrency, log_level=args.log_level
        )
        print(f"artifact: {path}")
        if args.no_normalize:
            return 0
        normalize_to_disk(cell, path)
        return 0

    # normalize
    if bool(args.cell) == bool(args.adopt):
        raise SystemExit("pass exactly one of --cell or --adopt")

    if args.cell:
        cell = CELLS[args.cell]
        path = args.input or cell.artifact_path
    else:
        if not args.input:
            raise SystemExit("--adopt requires --in")
        path = args.input
    if not path.exists():
        raise SystemExit(f"no artifact at {path}")

    if args.adopt:
        cell = cell_from_results(args.adopt, Results.load_metadata(path))

    normalize_to_disk(
        cell,
        path,
        args.out,
        include_messages=not args.no_messages,
        allow_commit_drift=args.allow_commit_drift,
        allow_mismatch=args.allow_mismatch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
