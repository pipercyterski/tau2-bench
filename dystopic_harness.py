"""``harness_variant.knob_values`` -> a constructed tau2 agent.

This is the adapter the dispatch contract asks for
(docs/reference/port-agent#the-harness_variant-block-the-adapter-you-write).
It lives at the fork root, NOT in a ``dystopic/`` package directory: the SDK
installs a real package called ``dystopic`` and a same-named directory here
would shadow it (or be shadowed by it) with an import error that reads like a
platform fault.

The whole point of the module is to FAIL CLOSED. Config loading is itself part
of what a check exercises, so a knob we cannot honor must raise
:class:`HarnessVariantBreach` rather than silently fall back to a default --
otherwise the platform grades a green run for an agent that was never
configured, and every verdict in the suite is about the wrong subject.

Knobs
-----
``agent_llm``       litellm model string, e.g. ``"gpt-4.1-2025-04-14"`` or
                    ``"anthropic/claude-sonnet-4-5"``. Defaults to tau2's own
                    ``DEFAULT_LLM_AGENT`` so an unconfigured run reproduces
                    tau2's published default.
``max_tool_iters``  Tool-calling iterations allowed within ONE conversation
                    turn. Not tau2's ``max_steps`` (which bounds the whole
                    conversation) -- the platform owns turn count.
``tau2_domain``     Which tau2 domain to build. Only ``retail`` is buildable:
                    the world/ontology port is retail-only, and running the
                    airline agent against a retail world would answer "not
                    found" to every call and read as an agent regression.
``agent_variant``   ``plain``  -> tau2's ``LLMAgent`` (the benchmarked agent).
                    ``gt``     -> tau2's ``LLMGTAgent``, which is handed the
                                  task's resolution steps. This is tau2's own
                                  user-simulator sanity check, and it needs the
                                  tau2 task -- see ``_resolve_tau2_task``.
``policy_variant``  ``default`` -> the domain's ``policy.md``, verbatim.
                    ``none``    -> policy withheld (the ablation that measures
                                   how much of the score the policy carries).

Everything else about the agent -- ``SYSTEM_PROMPT``, ``AGENT_INSTRUCTION``,
the policy text, the tool signatures -- is tau2's, untouched. We construct it;
we do not rewrite it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# litellm fetches a model-cost map over the network on first import. That is a
# documented sandbox cold-start failure, and tau2 imports litellm transitively
# from ``tau2.utils.llm_utils``, so the flag has to be set before ANY tau2
# import -- module scope here, and again in dystopic_entry, since either module
# may be imported first.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

REPO_ROOT = Path(__file__).resolve().parent
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    # Put tau2 on the path directly rather than relying on an editable install:
    # the sandbox installs `requirements`, not this repo's own package.
    sys.path.insert(0, _SRC)

# Every name the frozen-variant block has travelled under on the wire. The
# accessor tries them in order; a hand-rolled lookup of one name fails SILENTLY
# when the name moves, because absence legitimately means "no variant".
_VARIANT_KEYS = ("harness_variant", "configuration", "harness_config")

SUPPORTED_DOMAINS = ("retail",)
AGENT_VARIANTS = ("plain", "gt")
POLICY_VARIANTS = ("default", "none")

DEFAULT_MAX_TOOL_ITERS = 30


class HarnessVariantBreach(RuntimeError):
    """A knob we cannot honor. Raised, never swallowed: the run fails."""


@dataclass(frozen=True)
class HarnessSpec:
    """The resolved harness, plus what to acknowledge on the way out."""

    agent_llm: str
    llm_args: dict[str, Any]
    max_tool_iters: int
    tau2_domain: str
    agent_variant: str
    policy_variant: str
    source: str  # "harness_variant" | "static_default"
    snapshot_id: int | None = None
    fingerprint: str | None = None
    variant_name: str | None = None
    knob_values: dict[str, Any] = field(default_factory=dict)

    def acknowledgement(self) -> dict[str, Any]:
        """The ``metadata.harness_variant`` block the platform checks.

        A ``source`` other than ``"harness_variant"``, or a snapshot/fingerprint
        that disagrees with what the check froze, fails the run with
        ``harness_variant_breach`` instead of grading it -- which is the point:
        it turns a silent misconfiguration into a loud one. So we report
        ``static_default`` honestly when no variant reached us.
        """
        if self.source != "harness_variant":
            return {"source": self.source}
        return {
            "source": "harness_variant",
            "snapshot_id": self.snapshot_id,
            "fingerprint": self.fingerprint,
        }


def harness_variant_from_input(task_input: Any) -> dict[str, Any] | None:
    """Read the frozen variant block, or ``None`` when the check froze none.

    Prefers the SDK accessor when the SDK happens to be installed (it knows the
    canonical key list); falls back to the same presence test over every name
    the block has used. A check with no variant OMITS the key entirely -- it
    never sends ``null`` -- so this is a presence test, not a truthiness test
    on a known key.
    """
    if not isinstance(task_input, dict):
        return None
    try:
        from dystopic.odyssey import harness_variant_from_input as _sdk_accessor
    except Exception:
        pass
    else:
        block = _sdk_accessor(task_input)
        if isinstance(block, dict) and block:
            return block

    for key in _VARIANT_KEYS:
        block = task_input.get(key)
        if isinstance(block, dict) and block:
            return block
    return None


def _require(name: str, value: Any, kind: type | tuple[type, ...]) -> Any:
    if isinstance(value, bool) and kind is not bool:
        # bool is an int subclass; a `true` where a number belongs is a typo,
        # not a 1.
        raise HarnessVariantBreach(f"knob {name!r}: expected {kind}, got a boolean")
    if not isinstance(value, kind):
        raise HarnessVariantBreach(
            f"knob {name!r}: expected {kind}, got {type(value).__name__} ({value!r})"
        )
    return value


def _enum(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    _require(name, value, str)
    if value not in allowed:
        raise HarnessVariantBreach(
            f"knob {name!r}: {value!r} is not one of {list(allowed)}"
        )
    return value


def resolve_harness(task_input: Any) -> HarnessSpec:
    """Map ``knob_values`` onto a :class:`HarnessSpec`, or raise.

    An explicit ``null`` always conforms -- it means "leave this knob at the
    harness default" -- but an unknown knob name, a wrong type, or a value
    outside our enum is a breach.
    """
    from tau2.config import DEFAULT_LLM_AGENT, DEFAULT_LLM_ARGS_AGENT

    agent_llm = DEFAULT_LLM_AGENT
    max_tool_iters = DEFAULT_MAX_TOOL_ITERS
    tau2_domain = "retail"
    agent_variant = "plain"
    policy_variant = "default"

    block = harness_variant_from_input(task_input)
    if block is None:
        return HarnessSpec(
            agent_llm=agent_llm,
            llm_args=dict(DEFAULT_LLM_ARGS_AGENT),
            max_tool_iters=max_tool_iters,
            tau2_domain=tau2_domain,
            agent_variant=agent_variant,
            policy_variant=policy_variant,
            source="static_default",
        )

    knob_values = block.get("knob_values")
    if knob_values is None:
        knob_values = {}
    if not isinstance(knob_values, dict):
        raise HarnessVariantBreach(
            f"harness_variant.knob_values must be an object, got {type(knob_values).__name__}"
        )

    for name, value in knob_values.items():
        if value is None:
            continue  # "leave at the harness default"
        if name == "agent_llm":
            agent_llm = _require(name, value, str).strip()
            if not agent_llm:
                raise HarnessVariantBreach("knob 'agent_llm': empty model string")
        elif name == "max_tool_iters":
            max_tool_iters = int(_require(name, value, int))
            if max_tool_iters < 1:
                raise HarnessVariantBreach(
                    f"knob 'max_tool_iters': must be >= 1, got {max_tool_iters}"
                )
        elif name == "tau2_domain":
            tau2_domain = _enum(name, value, SUPPORTED_DOMAINS)
        elif name == "agent_variant":
            agent_variant = _enum(name, value, AGENT_VARIANTS)
        elif name == "policy_variant":
            policy_variant = _enum(name, value, POLICY_VARIANTS)
        else:
            raise HarnessVariantBreach(
                f"unknown knob {name!r}: this harness honors "
                f"{['agent_llm', 'max_tool_iters', 'tau2_domain', 'agent_variant', 'policy_variant']}"
            )

    return HarnessSpec(
        agent_llm=agent_llm,
        llm_args=dict(DEFAULT_LLM_ARGS_AGENT),
        max_tool_iters=max_tool_iters,
        tau2_domain=tau2_domain,
        agent_variant=agent_variant,
        policy_variant=policy_variant,
        source="harness_variant",
        snapshot_id=block.get("snapshot_id"),
        fingerprint=block.get("fingerprint"),
        variant_name=block.get("name"),
        knob_values=dict(knob_values),
    )


def _get_environment(tau2_domain: str):
    """Build the domain environment for its tools + policy ONLY.

    The environment carries a live tau2 DB, and tau2's orchestrator would
    execute tools against it. We never do: every call goes to the Odyssey
    proxy, which owns the world. We import the domain module directly rather
    than through ``tau2.registry`` so building retail does not drag in the
    airline / telecom / banking domains.
    """
    if tau2_domain == "retail":
        from tau2.domains.retail.environment import get_environment

        return get_environment()
    raise HarnessVariantBreach(  # pragma: no cover - resolve_harness gates this
        f"tau2_domain {tau2_domain!r} is not buildable by this port"
    )


def _resolve_tau2_task(task_input: Any, tau2_domain: str):
    """Find the tau2 task the GT agent needs its resolution steps from.

    The seed deliberately does not forward the answer key, so the GT variant is
    only buildable when the scenario names the tau2 task it was generated from.
    We look for that id in the places a generated scenario can put it, and
    raise when it is absent -- an ungrounded GT agent is exactly the silently
    misconfigured harness the fail-closed rule exists to prevent.
    """
    candidates: list[Any] = []
    if isinstance(task_input, dict):
        candidates.append(task_input.get("tau2_task_id"))
        for holder in (task_input.get("input"), task_input.get("metadata")):
            if isinstance(holder, dict):
                candidates.append(holder.get("tau2_task_id"))
    task_id = next((c for c in candidates if isinstance(c, str) and c), None)
    if task_id is None:
        raise HarnessVariantBreach(
            "knob 'agent_variant'='gt' needs the tau2 task id (the GT agent is "
            "built from that task's resolution steps); none of "
            "task_input['tau2_task_id'], ['input']['tau2_task_id'] or "
            "['metadata']['tau2_task_id'] was present"
        )

    if tau2_domain == "retail":
        from tau2.domains.retail.environment import get_tasks
    else:  # pragma: no cover - resolve_harness gates this
        raise HarnessVariantBreach(f"no task set for domain {tau2_domain!r}")

    for task in get_tasks(None):
        if task.id == task_id:
            return task
    raise HarnessVariantBreach(
        f"tau2 task {task_id!r} not found in the {tau2_domain} task set"
    )


def build_agent(spec: HarnessSpec, task_input: Any):
    """Construct the tau2 agent this variant describes.

    Returns ``(agent, tools)``. ``tools`` is tau2's own ``Tool`` list -- the
    same objects tau2 hands its LLM, so the tool names the model emits
    byte-match the ``tools_schema`` we exported from ``get_info()``.
    """
    from tau2.agent.llm_agent import LLMAgent, LLMGTAgent

    environment = _get_environment(spec.tau2_domain)
    tools = environment.get_tools()
    policy = environment.get_policy() if spec.policy_variant == "default" else ""

    if spec.agent_variant == "plain":
        agent = LLMAgent(
            tools=tools,
            domain_policy=policy,
            llm=spec.agent_llm,
            llm_args=dict(spec.llm_args),
        )
    else:
        task = _resolve_tau2_task(task_input, spec.tau2_domain)
        if not LLMGTAgent.check_valid_task(task):
            raise HarnessVariantBreach(
                f"tau2 task {task.id!r} declares no expected actions, so the GT "
                "agent cannot be built from it"
            )
        agent = LLMGTAgent(
            tools=tools,
            domain_policy=policy,
            task=task,
            llm=spec.agent_llm,
            llm_args=dict(spec.llm_args),
        )
    return agent, tools
