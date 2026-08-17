"""The platform entrypoint: tau2's agent, driven by us, tools routed to Odyssey.

    run(task_input, *, proxy_url, run_token) -> dict

The signature is fixed by the dispatch contract
(docs/reference/port-agent/code-mode.mdx, "The entrypoint": ``proxy_url`` and
``run_token`` are keyword-only), and this file + the callable name are what the
agent registers as ``entrypoint_file`` / ``entrypoint``.

WHAT THIS REPLACES
------------------
tau2's ``LLMAgent`` never executes a tool -- ``src/tau2/orchestrator/`` does,
against tau2's in-process DB. So the piece we replace is the **orchestrator**,
not the agent. tau2's ``LLMAgent`` / ``LLMGTAgent``, its ``SYSTEM_PROMPT``, its
``AGENT_INSTRUCTION`` and the domain's ``policy.md`` are used verbatim; we drive
the loop and send every tool call to the Odyssey proxy, which owns the world.
That is the whole port: same agent, different world.

ONE CALL = ONE CONVERSATION TURN
--------------------------------
The platform runs the user side (an LLM user-simulator holding the task
instruction, tau2's own interaction model) and calls this entrypoint once per
agent turn. In the default ``replay`` memory mode it re-feeds the prior
transcript in ``task_input["messages"]`` and puts the current user turn in
``task_input["user_instruction"]`` (alias ``latest_user_prompt``). So the
entrypoint is stateless per turn: rebuild, run tool calls until the model
answers, return that one reply. World state persists in the Odyssey ledger
between turns, so read tools always see the evolved world.

FAILURE POSTURE
---------------
Two rules, and they point opposite ways on purpose:

* ``final_response`` must be a NON-EMPTY string on every path that returns --
  an empty one is rejected outright (``ResponseShapeError``) and the judge gets
  nothing to grade. So every exception below is converted into a one-sentence
  summary of what went wrong.
* A :class:`HarnessVariantBreach` is the ONE exception that propagates. A run
  built from a variant we could not honor has produced a verdict about the
  wrong subject; it must fail, not be scored.

No ``asyncio.run`` appears anywhere here -- the sandbox already runs the
entrypoint inside a live event loop, and this loop is synchronous throughout,
so there is no thread to spawn and no ContextVar to re-bind.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Must precede any tau2 import (tau2.utils.llm_utils imports litellm, which
# otherwise fetches its cost map over the network -- a documented sandbox
# cold-start failure).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Anchor tau2 to THIS checkout, wherever the platform cloned it. tau2 resolves
# its corpus as ``DATA_DIR / "tau2" / "domains" / <domain>``, so a TAU2_DATA_DIR
# of "data/tau2" silently doubles into "data/tau2/tau2/domains/..." -- and left
# unset it derives a path from the installed package, which need not be the
# commit under test. Deciding it here, from the entrypoint's own location, means
# the agent always reads the corpus that shipped with the code being graded.
_REPO_ROOT = Path(__file__).resolve().parent
os.environ["TAU2_DATA_DIR"] = str(_REPO_ROOT / "data")
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dystopic_harness import (  # noqa: E402 - after the env guard, on purpose
    HarnessSpec,
    HarnessVariantBreach,
    build_agent,
    resolve_harness,
)
from dystopic_proxy import OdysseyProxy, StaleRunTokenError  # noqa: E402

_NO_INSTRUCTION = (
    "No user instruction reached the agent for this turn, so it had nothing to "
    "act on and made no tool calls."
)
_REPLAY_GAP = json.dumps({"error": "tool_result_unavailable_in_replay"})


# --------------------------------------------------------------------------- #
# Rendering a projection back into tau2's response shape
# --------------------------------------------------------------------------- #
# A declared ``ledger_read`` projection ALWAYS returns an object -- that is the
# grammar. Three retail tools do not: ``find_user_id_by_*`` return a bare user-id
# string and ``list_all_product_types`` returns a JSON string of {name: id}. The
# schema exporter deliberately kept the projections honest (an object) and left
# the re-rendering here, rather than lying in the ``output_schema``.
#
# This is not cosmetic. Without it the model sees ``{"user_id": "sara_doe_496"}``
# where tau2 shows ``sara_doe_496`` -- on the authentication step that begins
# essentially every retail task -- so the port would be measuring a different
# prompt than the benchmark it claims to reproduce.
#
# The same applies to a projection miss: ``not_found`` yields ``{"error": ...}``
# while tau2's environment renders a raised ``ValueError`` as ``f"Error: {e}"``
# with ``ToolMessage.error=True`` (``src/tau2/environment/environment.py``).
_BARE_STRING_TOOLS = {
    "find_user_id_by_email": "user_id",
    "find_user_id_by_name_zip": "user_id",
}


def render_retail(tool_name: str, payload: Any) -> tuple[Any, bool]:
    """Restate an Odyssey payload in the shape tau2's retail tool returns.

    Returns ``(payload, is_error)``. Anything not named here passes through
    untouched -- the entity ``get_*`` projections are already key-identical to
    tau2's models, which ``dystopic_export/validate.py`` checks against the live
    tau2 tools.
    """
    if isinstance(payload, dict) and set(payload) == {"error"}:
        # A projection miss. tau2 raises; its environment renders the raise.
        return f"Error: {payload['error']}", True

    if isinstance(payload, dict):
        key = _BARE_STRING_TOOLS.get(tool_name)
        if key is not None and key in payload:
            return payload[key], False
        if tool_name == "list_all_product_types" and "product_types" in payload:
            rows = payload.get("product_types") or []
            return (
                json.dumps(
                    {
                        r["name"]: r["product_id"]
                        for r in rows
                        if isinstance(r, dict) and "name" in r and "product_id" in r
                    },
                    sort_keys=True,
                ),
                False,
            )
    return payload, False


# --------------------------------------------------------------------------- #
# Tools that execute IN the sandbox rather than at the proxy
# --------------------------------------------------------------------------- #
# `calculate` is registered with default_execution_mode "executed"
# (stored: "code_intercepted"), which by contract means its real code runs here
# and only its DATA operations would go to /odyssey-proxy/data. Dispatching it
# to /odyssey-proxy/tools/{name} is a harness misroute that the proxy REFUSES
# with a 409 `tool_executes_in_sandbox` — the model would get an error string
# where tau2 gives it arithmetic. So we run tau2's own implementation.
#
# It is bound unbound-with-None on purpose: tau2's `calculate` never touches
# `self.db` (it is a pure function of its argument, which is exactly why the
# schema exporter marked it executed), so this needs no retail DB in the
# sandbox and cannot drift from tau2's semantics — including its rounding and
# its ValueError, which tau2's environment renders as f"Error: {e}".
def _sandbox_calculate(arguments: dict[str, Any]) -> tuple[str, bool]:
    from tau2.domains.retail.tools import RetailTools

    try:
        return RetailTools.calculate(None, **arguments), False
    except Exception as exc:
        return f"Error: {exc}", True


SANDBOX_EXECUTED = {"calculate": _sandbox_calculate}


# --------------------------------------------------------------------------- #
# Rebuilding the conversation
# --------------------------------------------------------------------------- #
def _tool_calls_from(entry: dict[str, Any]) -> list[Any]:
    """Normalize either tool-call wire shape into tau2 ``ToolCall`` objects.

    Accepts the flat platform shape ``{"id", "name", "arguments"}`` and the
    OpenAI-nested ``{"function": {"name", "arguments": "<json>"}}``.
    """
    from tau2.data_model.message import ToolCall

    calls = []
    for raw in entry.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = raw.get("name") or fn.get("name")
        if not name:
            continue
        arguments = raw.get("arguments", fn.get("arguments"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(
                id=str(raw.get("id") or ""),
                name=str(name),
                arguments=arguments,
                requestor="assistant",
            )
        )
    return calls


def rebuild_history(task_input: Any) -> list[Any]:
    """Rebuild the prior conversation as tau2 messages.

    The platform's ``replay`` mode projects its stored transcript down to
    ``{role, content}`` user/assistant turns, which is the common case. But the
    rich ``messages`` array THIS entrypoint returns can also be re-fed, so the
    richer shape is handled too: assistant ``tool_calls`` and ``role: "tool"``
    entries are rebuilt into real ``ToolCall`` / ``ToolMessage`` objects so a
    replayed turn sees the actual tool state it saw the first time.

    Two invariants the providers enforce and tau2 does not, so we do:

    * every message must carry content or tool calls (empty ones are dropped);
    * every assistant tool call must be answered by a tool message with the
      same id. A call whose result was not replayed gets an explicitly-labelled
      placeholder rather than fabricated output -- dropping the call instead
      would erase the fact that the agent acted.
    """
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

    entries = task_input.get("messages") if isinstance(task_input, dict) else None
    if not isinstance(entries, list):
        return []

    history: list[Any] = []
    pending: list[str] = []  # tool-call ids still awaiting a tool message

    def close_pending() -> None:
        for call_id in pending:
            history.append(
                ToolMessage(
                    id=call_id,
                    role="tool",
                    content=_REPLAY_GAP,
                    requestor="assistant",
                    error=True,
                )
            )
        pending.clear()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, default=str)

        if role == "tool":
            call_id = str(entry.get("tool_call_id") or entry.get("id") or "")
            if call_id in pending:
                pending.remove(call_id)
            elif not pending:
                continue  # orphan tool message: nothing called it
            else:
                call_id = pending.pop(0)
            history.append(
                ToolMessage(
                    id=call_id,
                    role="tool",
                    content=content if content is not None else "",
                    requestor="assistant",
                    error=bool(entry.get("error", False)),
                )
            )
            continue

        close_pending()
        if role == "assistant":
            calls = _tool_calls_from(entry)
            if not calls and not (content or "").strip():
                continue
            history.append(
                AssistantMessage(
                    role="assistant",
                    content=content,
                    tool_calls=calls or None,
                )
            )
            pending.extend(call.id for call in calls)
        elif role == "user":
            if not (content or "").strip():
                continue
            history.append(UserMessage(role="user", content=content))
        # "system" is skipped: tau2 owns the system prompt and re-deriving it
        # from the variant is the whole point of building the agent.

    close_pending()
    return history


def _current_user_message(task_input: Any) -> Any | None:
    """The user turn this call has to answer, if there is one."""
    from tau2.data_model.message import UserMessage

    if not isinstance(task_input, dict):
        return None
    for key in ("user_instruction", "latest_user_prompt"):
        value = task_input.get(key)
        if isinstance(value, str) and value.strip():
            return UserMessage(role="user", content=value)
    return None


# --------------------------------------------------------------------------- #
# The transcript we hand back
# --------------------------------------------------------------------------- #
def _transcript(state: Any) -> list[dict[str, Any]]:
    """tau2 message objects -> the platform's ``messages`` schema.

    Roles are restricted to system/user/assistant/tool and tool results carry
    ``tool_call_id``, so the trace renders the real trajectory and a replayed
    turn can re-feed genuine tool state. A message that does not validate
    platform-side is dropped to ``null`` with a soft warning rather than
    failing the run, so this stays deliberately close to the documented shape.
    """
    from tau2.data_model.message import (
        AssistantMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )

    out: list[dict[str, Any]] = []
    for message in list(state.system_messages) + list(state.messages):
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            out.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.is_tool_call():
                entry["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in message.tool_calls
                ]
            out.append(entry)
        elif isinstance(message, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.id,
                    "content": message.content,
                }
            )
    return out


def _token_totals(state: Any) -> tuple[int, int]:
    from tau2.data_model.message import AssistantMessage

    prompt = completion = 0
    for message in state.messages:
        if isinstance(message, AssistantMessage) and isinstance(message.usage, dict):
            prompt += int(message.usage.get("prompt_tokens") or 0)
            completion += int(message.usage.get("completion_tokens") or 0)
    return prompt, completion


# --------------------------------------------------------------------------- #
# The turn loop (the orchestrator we replace)
# --------------------------------------------------------------------------- #
def _dispatch_tool_calls(
    proxy: OdysseyProxy, tool_calls: list[Any]
) -> tuple[Any, bool]:
    """Route one assistant turn's tool calls to the proxy.

    Returns ``(message_for_the_agent, run_is_closed)``. A tool failure comes
    back as a structured error payload the model can reason about -- exactly
    what tau2's own environment does when a tool raises -- so one bad call
    never kills the turn. A stale run token is different: the run has closed,
    so we stop rather than feed the model N identical auth failures.
    """
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    messages: list[Any] = []
    closed = False
    for call in tool_calls:
        local = SANDBOX_EXECUTED.get(call.name)
        if local is not None:
            content, is_error = local(call.arguments or {})
            # The proxy never saw this call, so it cannot count it, but the
            # model did make it -- keep `n_tool_calls` an honest total.
            proxy.n_calls += 1
            messages.append(
                ToolMessage(
                    id=call.id,
                    role="tool",
                    content=content,
                    requestor="assistant",
                    error=is_error,
                )
            )
            continue
        try:
            content, is_error = proxy.call_as_content(
                call.name, call.arguments or {}, renderer=render_retail
            )
        except StaleRunTokenError as exc:
            content, is_error, closed = json.dumps({"error": str(exc)}), True, True
        messages.append(
            ToolMessage(
                id=call.id,
                role="tool",
                content=content,
                requestor="assistant",
                error=is_error,
            )
        )
        if closed:
            break
    if len(messages) == 1:
        return messages[0], closed
    return MultiToolMessage(role="tool", tool_messages=messages), closed


def _run_turn(
    agent: Any, state: Any, first_message: Any, proxy: OdysseyProxy, max_tool_iters: int
) -> tuple[str, dict[str, Any], Any]:
    """Drive tau2's agent until it produces a message for the user.

    ``max_tool_iters`` bounds tool-calling rounds within THIS turn; the number
    of turns is the platform's business, not ours. The (mutated) agent state is
    returned alongside so the caller renders the transcript from the same
    object the loop advanced.
    """
    message = first_message
    final = ""
    stats = {"n_llm_calls": 0, "iters_exhausted": False, "run_closed": False}

    for _ in range(max_tool_iters):
        assistant, state = agent.generate_next_message(message, state)
        stats["n_llm_calls"] += 1
        if not assistant.is_tool_call():
            final = (assistant.content or "").strip()
            break
        message, closed = _dispatch_tool_calls(proxy, assistant.tool_calls)
        if closed:
            stats["run_closed"] = True
            final = (assistant.content or "").strip()
            break
    else:
        stats["iters_exhausted"] = True

    if not final:
        if stats["run_closed"]:
            final = (
                "The run's token expired mid-turn, so the agent could not complete "
                "the tool calls it had started or reply to the user."
            )
        elif stats["iters_exhausted"]:
            final = (
                f"The agent used its full budget of {max_tool_iters} tool-calling "
                "iterations for this turn without producing a reply to the user."
            )
        else:
            final = "The agent produced no message for the user on this turn."
    return final, stats, state


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def _unbuilt_spec() -> HarnessSpec:
    """The spec to acknowledge when construction failed before one existed."""
    return HarnessSpec(
        agent_llm="unknown",
        llm_args={},
        max_tool_iters=0,
        tau2_domain="unknown",
        agent_variant="unknown",
        policy_variant="unknown",
        source="harness_build_failed",
    )


def run(task_input: dict, *, proxy_url: str, run_token: str) -> dict:
    """Produce the agent's reply for ONE conversation turn.

    ``task_input`` is the scenario's payload (or ``{}``); ``proxy_url`` and
    ``run_token`` are the per-run proxy base and bearer, keyword-only per the
    dispatch contract.
    """
    started = time.perf_counter()
    proxy = OdysseyProxy(proxy_url, run_token)

    # Fail closed BEFORE anything else: a harness built from a variant we could
    # not honor produces a verdict about the wrong subject, so the breach is
    # the one exception that escapes `run`. Any OTHER construction failure
    # (a missing data file, a broken import) still answers -- but it
    # acknowledges `harness_build_failed`, which is itself a breach when a
    # variant was frozen, and honest evidence when one was not.
    spec = None
    try:
        spec = resolve_harness(task_input)
        agent, _tools = build_agent(spec, task_input)
    except HarnessVariantBreach:
        raise
    except Exception as exc:  # noqa: BLE001
        return _respond(
            f"The agent could not be constructed for this turn: {type(exc).__name__}: {exc}",
            spec or _unbuilt_spec(),
            None,
            proxy,
            started,
            {"error": f"{type(exc).__name__}: {exc}"},
        )

    state = None
    try:
        state = agent.get_init_state(message_history=rebuild_history(task_input))
        first = _current_user_message(task_input)
        if first is None:
            return _respond(_NO_INSTRUCTION, spec, state, proxy, started, {})
        final, stats, state = _run_turn(agent, state, first, proxy, spec.max_tool_iters)
        return _respond(final, spec, state, proxy, started, stats)
    except HarnessVariantBreach:
        raise
    except Exception as exc:  # noqa: BLE001 - every other failure must still answer
        # An empty final_response fails the run outright and tells the judge
        # nothing. A one-sentence account of the failure is strictly better
        # evidence, and the partial transcript still renders.
        return _respond(
            f"The agent failed while handling this turn: {type(exc).__name__}: {exc}",
            spec,
            state,
            proxy,
            started,
            {"error": f"{type(exc).__name__}: {exc}"},
        )


def _respond(
    final_response: str,
    spec: HarnessSpec,
    state: Any,
    proxy: OdysseyProxy,
    started: float,
    stats: dict[str, Any],
) -> dict:
    """Assemble the response, guaranteeing a non-empty ``final_response``."""
    prompt_tokens = completion_tokens = 0
    messages: list[dict[str, Any]] = []
    if state is not None:
        try:
            messages = _transcript(state)
            prompt_tokens, completion_tokens = _token_totals(state)
        except Exception as exc:  # noqa: BLE001 - a bad transcript must not fail the run
            stats = dict(stats, transcript_error=f"{type(exc).__name__}: {exc}")

    metadata: dict[str, Any] = {
        "model": spec.agent_llm,
        "total_input_tokens": prompt_tokens,
        "total_output_tokens": completion_tokens,
        "agent_runtime_ms": int((time.perf_counter() - started) * 1000),
        # The platform compares this against the snapshot the check froze and
        # fails the run on a mismatch. Reporting it honestly is what makes a
        # misconfiguration loud instead of silently green.
        "harness_variant": spec.acknowledgement(),
        "tau2_domain": spec.tau2_domain,
        "agent_variant": spec.agent_variant,
        "policy_variant": spec.policy_variant,
        "max_tool_iters": spec.max_tool_iters,
        "n_tool_calls": proxy.n_calls,
        **stats,
    }
    if spec.variant_name:
        metadata["harness_variant_name"] = spec.variant_name

    response: dict[str, Any] = {
        "final_response": final_response.strip()
        or "The agent produced no output for this turn.",
        "metadata": metadata,
    }
    if messages:
        response["messages"] = messages
    return response
