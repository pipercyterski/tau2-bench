"""tau2 tasks -> Dystopic scenario payloads (+ a golden sidecar).

    python -m dystopic_export.export_scenarios --domain retail --split test \
        --out scenarios/retail.test.json

One tau2 ``Task`` becomes one scenario POST body (``POST
/api/agents/{id}/scenarios``). The mapping is not mechanical everywhere, so the
three judgement calls are spelled out here rather than buried in the code:

**1. Where the user-simulator prompt goes, and what turn 0 may contain.**
tau2 builds its user simulator's system prompt as ``simulation_guidelines.md``
+ ``<scenario>str(UserScenario)</scenario>`` (``user_simulator.py::system_prompt``).
That whole block goes into ``conversation.user_simulator_persona``, verbatim --
it is what drives the model-as-user turns, and reproducing it unparaphrased is
the only way our simulator withholds information on the same schedule tau2's
does (progressive disclosure is a *guideline*, not task data; drop it and the
simulator volunteers the order id on turn one and every task gets easier).

Two axes it must NOT go in, both of which are easy to reach for and wrong:

* ``user_instruction`` is documented as "the per-row prompt **the agent
  receives**" and is seeded as the user's opening message. Putting
  ``reason_for_call`` there hands the agent the entire goal -- constraints, ids
  and all -- before it asks anything, which is a materially easier benchmark
  than tau2 and would inflate every pass rate. So turn 0 is a fixed contentless
  greeting (``OPENER``) and the goal is elicited, as upstream intends.
* ``behavior_instructions`` is "free-text guidance fed into the Odyssey
  simulator" -- the **world** simulator, the component that invents tool
  responses. User-roleplay rules aimed there are both the wrong consumer and a
  route for the user's goal to leak into the world's behaviour. We emit none.

The one disclosed deviation: tau2's opener is *generated* by the simulator and
so carries some signal ("I'd like to exchange a water bottle"); ours carries
none. That makes our version marginally harder than tau2, not easier -- the
safe direction to err in.

**2. completion vs refusal.** tau2 has no refusal flag, so we derive it from
the reference trajectory, using tau2's own ``mutates_state`` marking (the
decorator attribute that decides which calls are re-executed during evaluation
replay -- i.e. exactly the calls that move the DB hash the reward compares).
A task is a **refusal** when its reference trajectory contains no
state-mutating call *and* the user was not merely asking a question: either the
trajectory ends in ``transfer_to_human_agents`` (tau2's canonical "the agent
cannot help"), or the request itself names a mutation. A task that does mutate
is a completion even if it *also* transfers afterwards (task 26 returns items
and then hands off over a payment-method the agent may not use) -- calling that
a refusal would tell the judge to penalize the half the agent is supposed to
do. The partial-refusal nuance rides in ``expected_outcome_detail`` instead.
Across the full 114-task retail set this rule splits 10 zero-mutation tasks
into 8 refusals and 2 informational completions, which matches a read of every
one of them.

**3. What carries the reward.** We do not reconstruct tau2's DB-hash reward.
Instead the ground truth is split across the axes the platform already grades:
``nl_assertions`` become one inline behavior scorer binding each (the judged,
gate-visible half), and the reference trajectory + ``communicate_info`` are
rendered into ``expected_outcome_detail`` (the judge's reference for
``task_completion``). ``communicate_info`` is deliberately *not* turned into
scorers: on this checkout retail's ``reward_basis`` is ``[DB, NL_ASSERTION]``,
so upstream itself treats those strings as diagnostics. The raw criteria are
emitted untouched to ``<out>.golden.json`` so a native tau2 run can still be
diffed against our verdicts later.

A behavior scorer binding is emitted only when ``NL_ASSERTION`` is actually in
*that task's* ``reward_basis`` -- tau2 gates ``nl_assertions`` per task
(``EvaluationCriteria`` docstring: "other populated fields run as diagnostics
only"), and a handful of tasks carry a leftover ``nl_assertions`` list under a
``[DB]``-only basis. Grading those anyway would judge the agent on a criterion
upstream itself never scores. The raw ``nl_assertions`` still ride in the
golden sidecar regardless of basis, since they remain diagnostically useful
even when ungated.

**4. Scenario naming.** ``name`` is ``"<domain>/<task_id>"`` rather than the
bare tau2 id. It is still a deterministic function of the task id -- so a
base/head review still lines up scenario-for-scenario -- but it stops
colliding once a second domain (airline) lands, and reads sanely in a
fleet-wide dashboard where "38" on its own says nothing.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Verbs that make a zero-mutation request a refusal rather than a question.
# Only consulted for trajectories that mutate nothing, so a false positive on a
# task that legitimately writes is impossible.
_MUTATION_REQUEST = re.compile(
    r"\b(cancel|return|exchange|modify|change|update|swap|replace|refund|add|remove)\w*",
    re.IGNORECASE,
)

# tau2's user simulator emits one of these once it is done; the platform watches
# a single keyword, so we arm the one this task is expected to end on.
STOP = "###STOP###"
TRANSFER = "###TRANSFER###"

# The orchestrator caps max_turns at 50 and defaults to 10. A tau2 retail
# conversation needs roughly one user turn per sub-request plus haggling room,
# and trajectory length is the only per-task proxy we have for that.
MIN_TURNS = 10
MAX_TURNS = 50
TURN_SLACK = 8

PERSONA_FALLBACK = (
    "An ordinary retail customer contacting customer service. The task "
    "specifies no distinctive personality; behave as a cooperative, "
    "unremarkable customer."
)

# Turn 0, identical for every task. It must reveal nothing: the whole point of
# tau2's interaction model is that the agent elicits the goal from the user.
# See _opener() for why this is a constant rather than the task's own
# reason_for_call.
OPENER = "Hi, I need some help with my account please."


def _load_tasks(domain: str, split: str) -> list[Any]:
    """Load a split through tau2's own loader, so we inherit its validation."""
    if domain == "retail":
        from tau2.domains.retail.environment import get_tasks
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unsupported domain {domain!r}")
    return get_tasks(split)


def _mutating_tools(domain: str) -> set[str]:
    """The tool names tau2 marks ``mutates_state=True``.

    This is the set replayed by ``set_state`` during evaluation, so it is
    precisely the set that can move the DB hash tau2's reward compares --
    ``transfer_to_human_agents`` is a WRITE-ish signal but explicitly not a
    mutation, and it falls out of this set for free.
    """
    from tau2.environment.toolkit import MUTATES_STATE_ATTR, TOOL_ATTR

    if domain == "retail":
        from tau2.domains.retail.tools import RetailTools as toolkit
    else:  # pragma: no cover
        raise ValueError(f"unsupported domain {domain!r}")

    return {
        name
        for name in dir(toolkit)
        if getattr(getattr(toolkit, name, None), TOOL_ATTR, False)
        and getattr(getattr(toolkit, name), MUTATES_STATE_ATTR, True)
    }


@lru_cache(maxsize=1)
def _user_sim_guidelines() -> str:
    """tau2's global user-simulator guidelines, persona slot collapsed.

    The runtime persona slot is a *simulation-time* knob (``PersonaConfig``),
    not task data; tau2 substitutes an empty string when none is configured and
    so do we.
    """
    from tau2.user.user_simulator import get_global_user_sim_guidelines

    return get_global_user_sim_guidelines(use_tools=False).replace(
        "<PERSONA_GUIDELINES>", ""
    )


def _goal_text(task: Any) -> str:
    """What the user actually wants, as free text.

    Never shown to the agent -- it is read only by the export's own heuristics
    (refusal detection, scenario grouping), which need the goal even though the
    opener deliberately withholds it.
    """
    instructions = task.user_scenario.instructions
    reason = getattr(instructions, "reason_for_call", None)
    return reason if reason is not None else str(instructions)


def _opener(task: Any) -> str:
    """Turn 0: what the agent actually receives, and it must NOT be the goal.

    In tau2 the user simulator holds ``reason_for_call`` in its system prompt
    and reveals it conversationally; the agent has to elicit it. On this
    platform ``user_instruction`` is documented as "the per-row prompt **the
    agent receives**" and is seeded as the user's first message -- so putting
    ``reason_for_call`` here would hand the agent the entire goal, constraints
    and ids included, before it has asked anything. That is a materially easier
    benchmark, and it would inflate every pass rate we report.

    So turn 0 is a fixed, contentless greeting and the goal lives only in the
    persona. One deviation to disclose: tau2's opener is *generated* by the
    simulator and therefore carries some signal ("I'd like to exchange a water
    bottle"), whereas ours carries none. Our version is slightly HARDER than
    tau2, not easier -- the safe direction to err in, and the agent still has to
    do the eliciting either way.
    """
    return OPENER


def _persona(task: Any) -> str:
    """Everything the simulated user knows -- tau2's own UserSimulator prompt.

    This is the field the model-as-user turns are driven from, so it carries
    what tau2 puts in the simulator's system prompt: the global simulation
    guidelines, then the whole ``<scenario>`` block (reason_for_call, known and
    unknown info, and the personality trait). Reproducing tau2's own prompt
    verbatim is the point -- the user side is not something we should be
    paraphrasing if we want the interaction to be comparable.

    Deliberately NOT ``behavior_instructions``: that axis is documented as
    "free-text guidance fed into the Odyssey simulator", i.e. the WORLD
    simulator. Putting user-roleplay rules there would aim them at the component
    that invents tool responses, which is both the wrong consumer and an
    invitation to leak the user's goal into the world's behaviour.
    """
    scenario = task.user_scenario
    trait = (getattr(scenario.instructions, "task_instructions", "") or "").strip()
    if scenario.persona:
        trait = scenario.persona
    elif len(trait.strip(".")) <= 1:
        # 20 of 114 retail tasks leave the trait as a bare "."; persona mode
        # fails closed on an empty persona, so give it a neutral one.
        trait = PERSONA_FALLBACK

    return (
        f"{_user_sim_guidelines()}\n\n"
        f"<scenario>\n{scenario}\n</scenario>\n\n"
        f"<persona>\n{trait}\n</persona>"
    )


def _render_call(action: Any) -> str:
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in action.arguments.items())
    return f"{action.name}({args})"


def _tally(names: list[str]) -> str:
    """``[a, a, b]`` -> ``"a x2, b"`` -- lookups matter by set, not by count."""
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return ", ".join(
        f"{name} x{names.count(name)}" if names.count(name) > 1 else name
        for name in seen
    )


def _expected_outcome_detail(
    task: Any, mutating: set[str], refusal: bool
) -> str | None:
    """Render the reference resolution as prose ground truth for the judge.

    The reward we cannot port is a hash over the DB after replaying
    ``actions``; the closest honest substitute is to *describe* that end state,
    including the explicit "any equivalent path counts" that tau2's own
    docstring attaches to the action list.
    """
    criteria = task.evaluation_criteria
    if criteria is None:
        return None

    actions = criteria.actions or []
    writes = [a for a in actions if a.name in mutating]
    reads = [
        a.name
        for a in actions
        if a.name not in mutating and a.name != "transfer_to_human_agents"
    ]
    transfers = [a for a in actions if a.name == "transfer_to_human_agents"]

    lines: list[str] = []
    if refusal:
        lines.append(
            "Correct end state: the world is UNCHANGED. No order, user record "
            "or item may be created, modified, cancelled, returned or "
            "exchanged. The agent is expected to decline the request (or the "
            "part of it that policy forbids) rather than act on it."
        )
    else:
        lines.append(
            "Correct end state, as produced by the reference resolution. Any "
            "path that leaves the world in this state is correct; these exact "
            "calls are not required:"
        )
        lines.extend(f"  {i}. {_render_call(a)}" for i, a in enumerate(writes, 1))
        lines.append("Nothing else in the world may change.")

    if transfers:
        summary = transfers[-1].arguments.get("summary", "")
        lines.append(
            "The reference resolution then hands the conversation off to a "
            f"human agent: {summary}"
        )
    if reads:
        lines.append(f"Lookups the reference used to get there: {_tally(reads)}.")
    if criteria.communicate_info:
        told = ", ".join(json.dumps(s) for s in criteria.communicate_info)
        lines.append(f"The agent must also tell the user, verbatim: {told}.")

    return "\n".join(lines)


def _conversation(task: Any, ends_in_transfer: bool) -> dict[str, Any]:
    n_actions = len(((task.evaluation_criteria and task.evaluation_criteria.actions) or []))
    return {
        "turn_mode": "model_as_user",
        "simulator_mode": "persona",
        "user_simulator_persona": _persona(task),
        "memory_mode": "replay",
        "max_turns": min(MAX_TURNS, max(MIN_TURNS, n_actions + TURN_SLACK)),
        "termination_keyword": TRANSFER if ends_in_transfer else STOP,
    }


def _is_refusal(task: Any, mutating: set[str]) -> bool:
    criteria = task.evaluation_criteria
    actions = (criteria.actions if criteria else None) or []
    names = [a.name for a in actions]
    if any(n in mutating for n in names):
        return False
    if "transfer_to_human_agents" in names:
        return True
    return bool(_MUTATION_REQUEST.search(_goal_text(task)))


def _is_defective(task: Any) -> bool:
    """A task carrying an unresolved upstream issue is expected to fail.

    ``wont_fix`` counts: the defect stands, upstream has just declined to fix
    it. Only ``resolved`` clears the marker.
    """
    return any(issue.status.value != "resolved" for issue in task.issues or [])


def _scenario_group(task: Any, mutating: set[str]) -> str:
    """Coverage taxonomy: the capability the task exercises."""
    criteria = task.evaluation_criteria
    writes = [a.name for a in (criteria.actions if criteria else None) or [] if a.name in mutating]
    return writes[0] if writes else "read_only"


def _nl_assertions_graded(criteria: Any) -> list[str]:
    """``nl_assertions`` gated by tau2's own reward_basis for this task.

    tau2 only judges ``nl_assertions`` when ``NL_ASSERTION`` is in
    ``reward_basis`` (see the class docstring on ``EvaluationCriteria``); a
    task can carry a populated ``nl_assertions`` list under a ``[DB]``-only
    basis, where upstream treats it as a diagnostic, not a grading criterion.
    Binding a scorer to it anyway would judge the agent more strictly than
    tau2 does.
    """
    if criteria is None:
        return []
    basis = {b.value if hasattr(b, "value") else b for b in (criteria.reward_basis or [])}
    if "NL_ASSERTION" not in basis:
        return []
    return criteria.nl_assertions or []


def build_scenario(
    task: Any, mutating: set[str], domain: str = "retail"
) -> dict[str, Any]:
    criteria = task.evaluation_criteria
    actions = (criteria.actions if criteria else None) or []
    refusal = _is_refusal(task, mutating)
    ends_in_transfer = bool(actions) and actions[-1].name == "transfer_to_human_agents"

    scenario: dict[str, Any] = {
        "name": f"{domain}/{task.id}",
        "scenario_group": _scenario_group(task, mutating),
        # No behavior_instructions: that axis feeds the WORLD simulator, and
        # everything the simulated USER knows belongs in the persona instead.
        "user_instruction": _opener(task),
        "expected_outcome": "refusal" if refusal else "completion",
        "expected_tool_sequence": [
            a.name for a in actions if a.requestor == "assistant"
        ],
        "conversation": _conversation(task, ends_in_transfer),
        "scorers": [
            {"target": "behavior", "constraint": assertion}
            for assertion in _nl_assertions_graded(criteria)
        ],
        "xfail": _is_defective(task),
    }
    detail = _expected_outcome_detail(task, mutating, refusal)
    if detail:
        scenario["expected_outcome_detail"] = detail
    return scenario


def build_golden(task: Any, domain: str = "retail") -> dict[str, Any]:
    """The raw criteria, untransformed, for diffing against a native tau2 run."""
    criteria = task.evaluation_criteria
    name = f"{domain}/{task.id}"
    if criteria is None:
        return {"name": name, "evaluation_criteria": None}
    return {
        "name": name,
        "evaluation_criteria": criteria.model_dump(mode="json"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=["retail"])
    ap.add_argument("--split", required=True, choices=["test", "train", "base"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    tasks = _load_tasks(args.domain, args.split)
    mutating = _mutating_tools(args.domain)

    scenarios = [build_scenario(t, mutating, args.domain) for t in tasks]
    golden = [build_golden(t, args.domain) for t in tasks]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scenarios, indent=2))
    golden_path = args.out.with_suffix(".golden.json")
    golden_path.write_text(json.dumps(golden, indent=2))

    # How many tasks the reward_basis gate (FIX 2) actually changed: a task
    # whose raw nl_assertions are non-empty but got dropped because
    # NL_ASSERTION was not in that task's reward_basis.
    gated_out = sum(
        bool((t.evaluation_criteria.nl_assertions if t.evaluation_criteria else None))
        and not _nl_assertions_graded(t.evaluation_criteria)
        for t in tasks
    )

    refusals = sum(s["expected_outcome"] == "refusal" for s in scenarios)
    print(f"wrote {args.out} ({len(scenarios)} scenarios)")
    print(f"wrote {golden_path}")
    print(f"  refusal: {refusals}  completion: {len(scenarios) - refusals}")
    print(f"  xfail: {sum(s['xfail'] for s in scenarios)}")
    print(
        f"  nl_assertions ungraded by reward_basis (FIX 2, dropped from scorers): "
        f"{gated_out}"
    )
    print(f"  with behavior scorers: {sum(bool(s['scorers']) for s in scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
