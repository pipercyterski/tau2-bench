"""Contract tests for the Dystopic sandbox entrypoint.

The live platform is not reachable from here, so both edges are faked:

* the Odyssey proxy, by monkeypatching ``urllib.request.urlopen`` inside
  ``dystopic_proxy`` -- which means the REAL request object is built and
  asserted on (URL, bearer header, JSON body), not a stub of it;
* the LLM, by monkeypatching ``tau2.agent.llm_agent.generate`` -- the exact
  symbol tau2's ``LLMAgent`` calls, so everything between the prompt assembly
  and the tool dispatch is the real tau2 code path.

What is being pinned here is the contract, not the behaviour of a model:
a non-empty ``final_response`` on every path out, the byte shape of the proxy
call, multi-turn rebuild of real tool state, and fail-closed on a knob the
harness cannot honor.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dystopic_entry  # noqa: E402
import dystopic_proxy  # noqa: E402
from dystopic_harness import HarnessVariantBreach, resolve_harness  # noqa: E402
from tau2.data_model.message import AssistantMessage, ToolCall  # noqa: E402

PROXY_URL = "https://proxy.dystopic.test/odyssey/run/abc"
RUN_TOKEN = "rt_test_token"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeResponse(io.BytesIO):
    """Minimal stand-in for the object ``urlopen`` yields as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class FakeProxy:
    """Records every request and replays a scripted payload per tool."""

    def __init__(self, payloads: dict[str, object], status: int | None = None):
        self.payloads = payloads
        self.status = status
        self.requests: list[dict[str, object]] = []

    def __call__(self, request, timeout=None):
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if self.status is not None:
            raise urllib.error.HTTPError(
                request.full_url,
                self.status,
                "boom",
                {},
                io.BytesIO(json.dumps({"detail": {"error_class": "boom"}}).encode()),
            )
        tool_name = request.full_url.rsplit("/", 1)[-1]
        envelope = {
            "tool_name": tool_name,
            "response": self.payloads.get(tool_name, {}),
            "source": "ledger_read",
            "latency_ms": 1.0,
        }
        return FakeResponse(json.dumps(envelope).encode("utf-8"))


class FakeLLM:
    """Replays a scripted list of assistant turns and records what it was fed."""

    def __init__(self, turns: list[AssistantMessage]):
        self.turns = list(turns)
        self.seen: list[list[object]] = []

    def __call__(self, model, messages, tools=None, **kwargs):
        self.seen.append(list(messages))
        if not self.turns:
            return AssistantMessage(role="assistant", content="Anything else?")
        return self.turns.pop(0)


@pytest.fixture
def install(monkeypatch):
    """Install both fakes; returns ``(llm, proxy)``."""

    def _install(turns, payloads=None, status=None):
        llm = FakeLLM(turns)
        proxy = FakeProxy(payloads or {}, status=status)
        monkeypatch.setattr("tau2.agent.llm_agent.generate", llm)
        monkeypatch.setattr(dystopic_proxy.urllib.request, "urlopen", proxy)
        return llm, proxy

    return _install


def text(content: str) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=content)


def call(name: str, arguments: dict, call_id: str = "c1") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def go(task_input: dict) -> dict:
    return dystopic_entry.run(task_input, proxy_url=PROXY_URL, run_token=RUN_TOKEN)


# --------------------------------------------------------------------------- #
# 1. A single-turn run answers
# --------------------------------------------------------------------------- #
def test_single_turn_returns_non_empty_final_response(install):
    install([text("Hi! How can I help you today?")])

    result = go({"user_instruction": "Hello", "task_id": 7})

    assert result["final_response"] == "Hi! How can I help you today?"
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "Hi! How can I help you today?",
    }
    # No variant was frozen, so the acknowledgement must say so honestly.
    assert result["metadata"]["harness_variant"] == {"source": "static_default"}


def test_missing_instruction_still_answers(install):
    install([text("unused")])
    result = go({})
    assert result["final_response"].strip()
    assert result["metadata"]["n_tool_calls"] == 0


# --------------------------------------------------------------------------- #
# 2. A tool call round-trips to the proxy with the right URL/headers/body
# --------------------------------------------------------------------------- #
def test_tool_call_round_trips_to_the_proxy(install):
    llm, proxy = install(
        [
            call("get_user_details", {"user_id": "sara_doe_496"}),
            text("Your account is under Sara Doe."),
        ],
        payloads={"get_user_details": {"name": {"first_name": "Sara"}}},
    )

    result = go({"user_instruction": "Who am I?"})

    assert len(proxy.requests) == 1
    request = proxy.requests[0]
    assert request["url"] == f"{PROXY_URL}/tools/get_user_details"
    assert request["method"] == "POST"
    assert request["headers"]["authorization"] == f"Bearer {RUN_TOKEN}"
    assert request["headers"]["content-type"] == "application/json"
    assert request["body"] == {"user_id": "sara_doe_496"}

    # Only the `response` field reaches the model -- never the envelope.
    fed = llm.seen[-1]
    tool_message = fed[-1]
    assert tool_message.role == "tool"
    assert json.loads(tool_message.content) == {"name": {"first_name": "Sara"}}
    assert tool_message.id == "c1"

    assert result["final_response"] == "Your account is under Sara Doe."
    assert result["metadata"]["n_tool_calls"] == 1
    tool_entries = [m for m in result["messages"] if m["role"] == "tool"]
    assert tool_entries[0]["tool_call_id"] == "c1"


def test_tool_failure_is_handed_to_the_model_not_raised(install):
    llm, proxy = install(
        [
            call("get_order_details", {"order_id": "#W1"}),
            text("I could not read that order."),
        ],
        status=500,
    )

    result = go({"user_instruction": "Check my order"})

    assert result["final_response"] == "I could not read that order."
    fed = llm.seen[-1]
    assert json.loads(fed[-1].content)["error"] == "boom"


# --------------------------------------------------------------------------- #
# 3. Multi-turn replay rebuilds prior tool state
# --------------------------------------------------------------------------- #
def test_multi_turn_replay_rebuilds_prior_messages(install):
    llm, _ = install([text("It shipped on the 3rd.")])

    result = go(
        {
            "user_instruction": "And when did it ship?",
            "messages": [
                {"role": "user", "content": "Where is order #W1?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t0",
                            "name": "get_order_details",
                            "arguments": {"order_id": "#W1"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t0",
                    "content": '{"status": "delivered"}',
                },
                {"role": "assistant", "content": "It was delivered."},
            ],
        }
    )

    fed = llm.seen[0]
    roles = [m.role for m in fed]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert fed[2].tool_calls[0].name == "get_order_details"
    assert fed[2].tool_calls[0].arguments == {"order_id": "#W1"}
    assert fed[3].id == "t0"
    assert json.loads(fed[3].content) == {"status": "delivered"}
    assert fed[-1].content == "And when did it ship?"
    assert result["final_response"] == "It shipped on the 3rd."


def test_replayed_tool_call_without_a_result_gets_a_labelled_placeholder(install):
    llm, _ = install([text("ok")])

    go(
        {
            "user_instruction": "carry on",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "t9", "name": "calculate", "arguments": {}}],
                },
                {"role": "user", "content": "hello?"},
            ],
        }
    )

    fed = llm.seen[0]
    assert [m.role for m in fed] == ["system", "assistant", "tool", "user", "user"]
    assert json.loads(fed[2].content) == {"error": "tool_result_unavailable_in_replay"}


# --------------------------------------------------------------------------- #
# 4. Exceptions still produce a non-empty final_response
# --------------------------------------------------------------------------- #
def test_llm_exception_still_yields_a_final_response(install, monkeypatch):
    install([])

    def explode(*args, **kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr("tau2.agent.llm_agent.generate", explode)

    result = go({"user_instruction": "Hello"})

    assert result["final_response"].strip()
    assert "provider is down" in result["final_response"]
    assert "provider is down" in result["metadata"]["error"]


def test_construction_failure_still_answers_and_owns_up(install, monkeypatch):
    install([text("never reached")])

    def broken(*args, **kwargs):
        raise FileNotFoundError("data/tau2/domains/retail/policy.md")

    monkeypatch.setattr(dystopic_entry, "build_agent", broken)

    result = go({"user_instruction": "Hello"})

    assert "could not be constructed" in result["final_response"]
    # Not "harness_variant": a harness we failed to build must never claim it
    # built the frozen one.
    assert result["metadata"]["harness_variant"] == {"source": "static_default"}


def test_iteration_budget_exhaustion_still_answers(install):
    install(
        [call("calculate", {"expression": "1+1"}, call_id=f"c{i}") for i in range(4)],
        payloads={"calculate": 2},
    )

    result = go(
        {
            "user_instruction": "Add some numbers",
            "harness_variant": {
                "snapshot_id": 1,
                "fingerprint": "ff",
                "knob_values": {"max_tool_iters": 2},
            },
        }
    )

    assert "2 tool-calling iterations" in result["final_response"]
    assert result["metadata"]["iters_exhausted"] is True
    assert result["metadata"]["n_tool_calls"] == 2


# --------------------------------------------------------------------------- #
# 5. Harness variants: honored, acknowledged, or failed closed
# --------------------------------------------------------------------------- #
def test_variant_is_honored_and_acknowledged(install):
    llm, _ = install([text("done")])

    result = go(
        {
            "user_instruction": "Hi",
            "harness_variant": {
                "snapshot_id": 8412,
                "variant_id": 7,
                "name": "eu-tenant",
                "fingerprint": "9f2c",
                "knob_values": {
                    "agent_llm": "anthropic/claude-sonnet-4-5",
                    "max_tool_iters": 5,
                    "tau2_domain": "retail",
                    "agent_variant": "plain",
                    "policy_variant": "none",
                },
            },
        }
    )

    assert result["metadata"]["harness_variant"] == {
        "source": "harness_variant",
        "snapshot_id": 8412,
        "fingerprint": "9f2c",
    }
    assert result["metadata"]["model"] == "anthropic/claude-sonnet-4-5"
    assert result["metadata"]["max_tool_iters"] == 5
    # policy_variant="none" must actually withhold the policy, not just be recorded.
    system_prompt = llm.seen[0][0].content
    assert "<policy>\n\n</policy>" in system_prompt


def test_default_policy_variant_ships_the_real_policy(install):
    llm, _ = install([text("done")])
    go({"user_instruction": "Hi"})
    assert "# Retail agent policy" in llm.seen[0][0].content


@pytest.mark.parametrize(
    "knobs",
    [
        {"cart_enabled": False},  # a knob this harness does not have
        {"agent_variant": "solo"},  # outside our enum
        {"tau2_domain": "airline"},  # not buildable by this port
        {"max_tool_iters": "lots"},  # wrong type
        {"max_tool_iters": 0},  # out of range
    ],
    ids=["unknown", "bad-enum", "unported-domain", "wrong-type", "out-of-range"],
)
def test_unhonorable_knob_fails_closed(install, knobs):
    install([text("must never be reached")])

    with pytest.raises(HarnessVariantBreach):
        go(
            {
                "user_instruction": "Hi",
                "harness_variant": {
                    "snapshot_id": 1,
                    "fingerprint": "ff",
                    "knob_values": knobs,
                },
            }
        )


def test_gt_variant_without_a_task_id_fails_closed(install):
    install([text("must never be reached")])

    with pytest.raises(HarnessVariantBreach):
        go(
            {
                "user_instruction": "Hi",
                "harness_variant": {
                    "snapshot_id": 1,
                    "fingerprint": "ff",
                    "knob_values": {"agent_variant": "gt"},
                },
            }
        )


def test_gt_variant_builds_from_the_named_tau2_task(install):
    llm, _ = install([text("Step 1 done.")])

    result = go(
        {
            "user_instruction": "Hi",
            "tau2_task_id": "0",
            "harness_variant": {
                "snapshot_id": 2,
                "fingerprint": "cc",
                "knob_values": {"agent_variant": "gt"},
            },
        }
    )

    # The GT agent's distinguishing feature is the resolution steps in its
    # system prompt -- that is what makes it a user-simulator sanity check
    # rather than the benchmarked agent.
    assert "<resolution_steps>" in llm.seen[0][0].content
    assert result["metadata"]["agent_variant"] == "gt"


def test_explicit_null_knob_leaves_the_default(install):
    spec = resolve_harness(
        {
            "harness_variant": {
                "snapshot_id": 3,
                "fingerprint": "aa",
                "knob_values": {"agent_llm": None},
            }
        }
    )
    assert spec.source == "harness_variant"
    assert spec.agent_llm  # tau2's own default, not an empty string


def test_legacy_variant_key_is_still_read():
    spec = resolve_harness(
        {
            "configuration": {
                "snapshot_id": 5,
                "fingerprint": "bb",
                "knob_values": {"max_tool_iters": 3},
            }
        }
    )
    assert spec.source == "harness_variant"
    assert spec.snapshot_id == 5
    assert spec.max_tool_iters == 3


# --------------------------------------------------------------------------- #
# 6. Proxy transport policy
# --------------------------------------------------------------------------- #
def test_a_stale_run_token_ends_the_turn_with_an_answer(install, monkeypatch):
    install([call("get_user_details", {"user_id": "x"}), text("never reached")])

    def expired(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "expired",
            {},
            io.BytesIO(
                json.dumps({"detail": {"error_class": "run_token_expired"}}).encode()
            ),
        )

    monkeypatch.setattr(dystopic_proxy.urllib.request, "urlopen", expired)

    result = go({"user_instruction": "Who am I?"})

    assert "token expired" in result["final_response"]
    assert result["metadata"]["run_closed"] is True


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    attempts = {"n": 0}
    payload = {"tool_name": "calculate", "response": 4, "source": "odyssey"}

    def flaky(request, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(
                request.full_url, 429, "slow down", {}, io.BytesIO(b"{}")
            )
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(dystopic_proxy.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(dystopic_proxy.time, "sleep", lambda _s: None)

    proxy = dystopic_proxy.OdysseyProxy(PROXY_URL, RUN_TOKEN)
    assert proxy.call("calculate", {"expression": "2+2"}) == 4
    assert attempts["n"] == 3


def test_a_terminal_status_is_not_retried(monkeypatch):
    attempts = {"n": 0}

    def refused(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad args",
            {},
            io.BytesIO(
                json.dumps({"detail": {"error_class": "invalid_args"}}).encode()
            ),
        )

    monkeypatch.setattr(dystopic_proxy.urllib.request, "urlopen", refused)

    proxy = dystopic_proxy.OdysseyProxy(PROXY_URL, RUN_TOKEN)
    with pytest.raises(dystopic_proxy.ProxyError) as excinfo:
        proxy.call("calculate", {})
    assert excinfo.value.error_class == "invalid_args"
    assert attempts["n"] == 1


# --------------------------------------------------------------------------- #
# 8. A projection is re-rendered into tau2's own response shape
# --------------------------------------------------------------------------- #
# A ledger_read projection always returns an OBJECT. Several tau2 retail tools
# return a bare string. Without `render_retail` the model sees
# {"user_id": "sara_doe_496"} where tau2 shows sara_doe_496 -- on the
# authentication step that opens essentially every retail task.
def test_find_user_id_is_unwrapped_to_a_bare_string(install):
    llm, _ = install(
        [
            call("find_user_id_by_email", {"email": "sara@example.com"}),
            text("You are sara_doe_496."),
        ],
        payloads={"find_user_id_by_email": {"user_id": "sara_doe_496"}},
    )

    go({"user_instruction": "Look me up"})

    tool_message = llm.seen[-1][-1]
    assert tool_message.content == "sara_doe_496"
    assert tool_message.error is False


def test_list_all_product_types_is_rendered_as_tau2s_name_to_id_map(install):
    llm, _ = install(
        [
            call("list_all_product_types", {}),
            text("Here they are."),
        ],
        payloads={
            "list_all_product_types": {
                "product_types": [
                    {"name": "Backpack", "product_id": "2524789262"},
                    {"name": "Action Camera", "product_id": "3377618313"},
                ]
            }
        },
    )

    go({"user_instruction": "What do you sell?"})

    tool_message = llm.seen[-1][-1]
    assert tool_message.content == json.dumps(
        {"Action Camera": "3377618313", "Backpack": "2524789262"}, sort_keys=True
    )


def test_a_projection_miss_is_rendered_as_tau2s_error_string(install):
    llm, _ = install(
        [
            call("get_order_details", {"order_id": "#WNOPE"}),
            text("I could not find that order."),
        ],
        payloads={"get_order_details": {"error": "Order not found"}},
    )

    go({"user_instruction": "Check #WNOPE"})

    tool_message = llm.seen[-1][-1]
    # tau2's environment renders a raised ValueError as f"Error: {e}" and sets
    # ToolMessage.error -- both halves matter, the second drives its metrics.
    assert tool_message.content == "Error: Order not found"
    assert tool_message.error is True


def test_entity_projections_pass_through_untouched(install):
    payload = {"user_id": "sara_doe_496", "email": "sara@example.com"}
    llm, _ = install(
        [
            call("get_user_details", {"user_id": "sara_doe_496"}),
            text("Got it."),
        ],
        payloads={"get_user_details": payload},
    )

    go({"user_instruction": "Who am I?"})

    assert json.loads(llm.seen[-1][-1].content) == payload


def test_renderer_leaves_a_payload_that_merely_contains_an_error_key(install):
    # {"error": ...} ALONE is a projection miss; an object that happens to carry
    # an `error` field beside real data is not, and must not be flattened.
    payload = {"error": "partial", "user_id": "sara_doe_496"}
    assert dystopic_entry.render_retail("get_user_details", payload) == (payload, False)


# --------------------------------------------------------------------------- #
# 9. An `executed` tool runs in the sandbox, never at the proxy
# --------------------------------------------------------------------------- #
# `calculate` is declared executed/code_intercepted. The proxy REFUSES
# /tools/{name} for such a tool with a 409 `tool_executes_in_sandbox`, so
# dispatching it would hand the model an error instead of arithmetic.
def test_calculate_runs_in_sandbox_and_never_reaches_the_proxy(install):
    llm, proxy = install(
        [
            call("calculate", {"expression": "481.50 * 3"}),
            text("That comes to $1444.50."),
        ]
    )

    result = go({"user_instruction": "What is three cameras?"})

    assert proxy.requests == []  # the misroute never happens
    tool_message = llm.seen[-1][-1]
    assert tool_message.content == "1444.5"  # tau2's exact rounding
    assert tool_message.error is False
    assert result["metadata"]["n_tool_calls"] == 1


def test_calculate_rejects_a_bad_expression_the_way_tau2_does(install):
    llm, proxy = install(
        [
            call("calculate", {"expression": "__import__('os')"}),
            text("I cannot evaluate that."),
        ]
    )

    go({"user_instruction": "compute this"})

    assert proxy.requests == []
    tool_message = llm.seen[-1][-1]
    assert tool_message.content == "Error: Invalid characters in expression"
    assert tool_message.error is True


def test_every_executed_tool_in_the_schema_has_a_sandbox_implementation():
    schema = json.loads(
        (
            Path(__file__).resolve().parent.parent / "schemas" / "retail.tools.json"
        ).read_text()
    )
    executed = {
        t["name"]
        for t in schema
        if t.get("default_execution_mode") in ("executed", "code_intercepted")
    }
    assert executed, "the fixture should declare at least one executed tool"
    assert executed <= set(dystopic_entry.SANDBOX_EXECUTED)
