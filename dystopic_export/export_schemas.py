"""tau2 retail toolkit -> Dystopic ``tools_schema``.

    python -m dystopic_export.export_schemas --domain retail --out schemas/retail.tools.json
    python -m dystopic_export.export_schemas --domain retail --level all --out schemas/retail.tools.json

This is where the "declared environment" thesis lives. tau2's retail toolkit is
16 functions over a frozen DB; on the platform each one becomes a declaration:

* a **pure read** becomes a ``ledger_read`` projection -- the world engine
  answers it straight off the ledger, deterministically, with no LLM. Same seed
  and same action history give a byte-identical response, and the trace row is
  stamped ``source: "ledger_read"`` at zero token cost.
* a **write** becomes a ``ledger_adapter`` -- the state change is declared
  rather than inferred, so base and head checks mutate the world identically.
* the two tools that are neither (``calculate`` is a pure function,
  ``transfer_to_human_agents`` is a terminal signal) are handled explicitly
  rather than left to default into simulation.

Three levels are emitted so the port can be *ablated*, not just asserted:

    v0  tau2's literal name + description + input_schema. Nothing else. The
        simulator invents both the shape and the content of every response.
    v1  + the response contract. For a read that is ``output_schema``,
        synthesized from tau2's own pydantic return models: the shape is pinned
        and a violation is regenerated, though the content is still invented.
        For a *write* it is that schema made permissive (failure has to be
        expressible), a worked ``examples`` payload, and the preconditions
        tau2's own source enumerates appended to the description -- so the
        description differs from v0 on the seven writes, and only there.
    v2  + ledger_read on every read and ledger_adapter on every write. The
        reads stop being generated at all; the writes stop being guessed.

v2 is the real artifact. v0/v1 exist so the eventual pass-rate delta can be
attributed to the declaration rather than to the model.

Three deliberate shape divergences (v2 only)
--------------------------------------------
A projection returns a JSON object; three tau2 tools return a bare scalar or a
JSON *string*. Rather than bend the grammar, the projection returns the honest
object and the sandbox agent re-renders tau2's shape in one line:

* ``find_user_id_by_email`` / ``find_user_id_by_name_zip`` project
  ``{"user_id": ...}``; the agent returns ``payload["user_id"]``.
* ``list_all_product_types`` projects ``{"product_types": [{name, product_id}]}``;
  the agent returns ``json.dumps({r["name"]: r["product_id"] for r in rows},
  sort_keys=True)`` -- byte-identical to tau2, because the 50 retail product
  names are unique (asserted below, which is also why the dedup tau2's dict
  comprehension performs is a no-op here and a ``list`` projection suffices).

Likewise a ``not_found`` payload is ``{"error": "Order not found"}`` where tau2
raises ``ValueError("Order not found")`` and its environment renders
``f"Error: {e}"``. The agent reconstructs that exact string, so the text the
model under test sees is unchanged.

What could not be declared
--------------------------
``user.payment_methods{}`` is a map of payment-method objects nested inside the
user entity. Four write tools adjust a gift card's ``balance`` inside that map,
and ``field_map`` sets a *declared field* from a *single path* -- it cannot
address a value nested under a map key. Those balance changes are therefore
left to the simulator: an adapter is authoritative only for the entity types it
writes, so declaring the ``order`` mutation keeps the simulator's inferred
``user`` ops intact. Promoting ``payment_method`` to a first-class entity (the
way ``item`` was promoted out of ``products[].variants``) would close this, and
is the obvious next move if gift-card balance turns out to be graded.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dystopic_export import contract  # noqa: E402

LEVELS = ("v0", "v1", "v2")


# --- tau2 side -------------------------------------------------------------


def _tau2_toolkit(domain: str) -> Any:
    """A toolkit bound to a *fresh* copy of tau2's shipped DB.

    Fresh per call on purpose: ``_success_examples`` invokes the real write
    tools, which mutate the DB in place. Sharing one instance would make the
    second example depend on the first.
    """
    if domain != "retail":  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unsupported domain {domain!r}")

    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    return RetailTools(RetailDB.load(RETAIL_DB_PATH))


def _tau2_tools(domain: str) -> dict[str, Any]:
    """tau2's own ``Tool`` objects, so names and schemas byte-match upstream."""
    return _tau2_toolkit(domain).get_tools()


def _write_tool_names(toolkit: Any) -> set[str]:
    """The tools tau2 itself classifies as writes.

    Derived from ``@is_tool(ToolType.WRITE)`` rather than listed here, so a
    toolkit that grows a write cannot quietly skip the response contract below.
    """
    from tau2.environment.toolkit import ToolType, get_tool_types

    return {
        name for name, kind in get_tool_types(toolkit).items() if kind is ToolType.WRITE
    }


def _name_desc_params(tool: Any) -> tuple[str, str, dict[str, Any]]:
    """tau2's own OpenAI function schema, verbatim.

    Taken from ``openai_schema`` rather than rebuilt, so the name byte-matches
    the ``proxy_call`` routing key and the description/parameters are exactly
    what tau2 hands its own agent (``title`` noise included). Any pass-rate
    delta between the two harnesses is then attributable to the world, not to a
    reworded tool definition.
    """
    fn = copy.deepcopy(tool.openai_schema["function"])
    return fn["name"], fn["description"], fn["parameters"]


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Resolve ``$ref``/``$defs`` into a self-contained schema.

    The platform validates ``output_schema`` as a plain JSON Schema object and
    caps nesting at 24 levels; retail's deepest inlined return (``User`` ->
    ``payment_methods`` -> ``anyOf`` -> ``CreditCard``) sits at 8.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            rest = {k: v for k, v in node.items() if k != "$ref"}
            target = defs[node["$ref"].rsplit("/", 1)[-1]]
            resolved = _inline_refs(copy.deepcopy(target), defs)
            resolved.update(_inline_refs(rest, defs))
            return resolved
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _output_schema(tool: Any) -> dict[str, Any]:
    """Synthesize ``output_schema`` from tau2's pydantic return annotation.

    ``Tool`` wraps the return type in a one-field ``returns`` model, so the
    tool's actual payload schema is that field's schema with the domain's
    ``$defs`` inlined.
    """
    raw = tool.returns.model_json_schema()
    defs = raw.get("$defs", {})
    return _inline_refs(copy.deepcopy(raw["properties"]["returns"]), defs)


# --- the write-tool response contract --------------------------------------
#
# A write tool has TWO legal responses and tau2's return annotation only
# describes one of them. `cancel_pending_order -> Order` is the success case;
# the failure case is a raised ValueError that tau2's Environment renders as
# `f"Error: {e}"` and hands back as an errored tool message.
#
# Synthesizing `output_schema` straight off the return model therefore declares
# a world in which failure is *unrepresentable*: a `type: object` with a large
# `required` array admits nothing but a fully-populated Order. The simulator
# validates against it and regenerates until it conforms, so a call tau2 would
# have rejected -- an item id that exists nowhere, a payment method the user
# does not hold -- comes back as a fabricated success. The adapter then writes
# it, and the agent never gets the error it would have self-corrected from.
# (The same over-constraint is why a write's first generation is often
# malformed: a deeply-required nested object is hard to emit in one shot.)
#
# So the write schemas are made PERMISSIVE, which the platform explicitly
# allows -- `output_schema` is validated only as "a JSON Schema object" within
# the nesting cap (`_validate_tools_schema` in apps/api/app/schemas/agent.py);
# nothing requires a top-level `type` or any `required`. Dropping both lets one
# schema admit tau2's success object AND the `{"error": ...}` failure body,
# while every field `description` survives -- and those descriptions, plus a
# real worked `examples` entry, are what actually ground the simulator's
# field-level output. Guidance is kept; the straitjacket is removed.
#
# The same three-part treatment is applied to all of them:
#   1. permissive schema      -> failure is expressible at all
#   2. derived preconditions  -> the simulator knows WHEN to fail, and how to
#                                phrase it in tau2's own words
#   3. a `when`-guarded adapter (see `retail_ledger_adapters`)
#                             -> a failure body writes nothing


def _fn_of(tool: Any) -> Any:
    """The plain function behind a tau2 ``Tool``, for source reading."""
    fn = getattr(tool, "_func", None)
    if fn is None:  # pragma: no cover - every retail tool wraps a function
        raise AssertionError(f"tau2 Tool {tool.name!r} exposes no underlying function")
    return inspect.unwrap(fn)


def _render_message(node: ast.expr) -> str | None:
    """The text a ``raise ValueError(<node>)`` would produce.

    Interpolations become ``<name>`` placeholders -- the *shape* of the message
    is what the simulator needs, and the runtime value is call data it already
    has.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.FormattedValue):
                parts.append(f"<{ast.unparse(piece.value)}>")
            else:  # pragma: no cover - retail has no other f-string parts
                return None
        return "".join(parts)
    return None


class _RaiseCollector(ast.NodeVisitor):
    """Every ``ValueError`` a tool body can raise, in source order.

    Follows ``self._get_*`` helper calls, because that is where the *existence*
    checks live -- ``_get_payment_method`` raising "Payment method not found" is
    the exact precondition the live bug walked straight through, and it appears
    in no tool's own docstring.
    """

    def __init__(self, owner: type, seen: set[str] | None = None) -> None:
        self.owner = owner
        self.seen = seen if seen is not None else set()
        self.messages: list[str] = []

    def _add(self, message: str) -> None:
        if message not in self.messages:
            self.messages.append(message)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        if (
            isinstance(exc, ast.Call)
            and isinstance(exc.func, ast.Name)
            and exc.func.id == "ValueError"
            and len(exc.args) == 1
        ):
            rendered = _render_message(exc.args[0])
            if rendered is not None:
                self._add(rendered)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and func.attr.startswith("_get_")
            and func.attr not in self.seen
        ):
            self.seen.add(func.attr)
            helper = getattr(self.owner, func.attr, None)
            if helper is not None:
                for message in _raise_messages(helper, self.owner, self.seen):
                    self._add(message)
        self.generic_visit(node)


def _raise_messages(fn: Any, owner: type, seen: set[str] | None = None) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    collector = _RaiseCollector(owner, seen)
    collector.visit(tree)
    return collector.messages


def _declared_raises(tool: Any) -> list[str]:
    """tau2's OWN parse of the docstring's ``Raises:`` block (``Tool.raises``).

    Reusing upstream's parse rather than re-parsing the docstring here means the
    conditions are worded exactly as tau2 words them. It is not sufficient on
    its own, though, which is why the raise-message walk above exists too:
    ``cancel_pending_order`` documents no ``Raises:`` section at all, and no
    docstring anywhere mentions the helper existence checks (``_get_order``,
    ``_get_payment_method``) that a bad id actually trips.
    """
    return [
        " ".join(str(entry.get("desc") or "").split())
        for entry in (tool.raises or [])
        if str(entry.get("desc") or "").strip()
    ]


# The one piece of hand-written prose in this file. It is tool-INDEPENDENT on
# purpose: everything tool-specific below is derived from tau2's source, and
# this frame only states the platform-side contract those derived facts feed --
# what shape a failure takes, and that a failure must not be dressed as success.
#
# `{"error": <reason>}` rather than a bare `Error: <reason>` string because that
# is the shape `dystopic_entry.render_retail` already converts into tau2's exact
# `f"Error: {e}"` rendering *and* flags as an errored tool message. A bare string
# would pass through unrendered and un-flagged -- right text, wrong error bit.
#
# The second paragraph is a correction, not decoration. The first version said an
# argument naming something absent from the world "IS a violation", and a live
# run showed the cost: the simulator refused two valid desk-lamp exchanges in a
# row with a system error, having just reported the same item as available. The
# working set is built by matching ARGUMENT NAMES against a declared type's id
# tokens (item_id / item_ids / itemid / item), so `new_item_ids` binds to
# nothing, and `payment_method_id` binds to nothing either because payment
# methods live nested inside `user` rather than as their own entity type. Those
# are exactly the two arguments an exchange must validate. Fail-closed plus
# unresolvable equals a false error, so the contract now distinguishes "looked up
# and found absent" from "could not look up".
_FAILURE_CONTRACT = """\
Preconditions. On a violation return exactly {{"error": "<reason>"}} -- an \
object with that single key and nothing else -- choosing <reason> verbatim from \
the message list. The caller renders it as the string `Error: <reason>`, which \
is exactly what the real environment produces (`f"Error: {{e}}"` over the raised \
ValueError). Never answer a violating call with a success object.

Fail only on a violation you can POSITIVELY establish from world state -- an \
entity you looked up and found absent, or a value you looked up and found to \
conflict. If you cannot resolve an argument either way, treat its precondition \
as SATISFIED and return the success response. Not every argument is retrievable: \
only ids whose argument NAME matches a declared entity's id tokens land in your \
working set, so a qualified name like `new_item_ids`, or an id for something the \
ontology holds nested inside another entity rather than as its own type (a \
payment method), can be unresolvable while being perfectly valid. Treating "I \
could not look this up" as "this does not exist" turns valid calls into system \
errors and strands the conversation.

Conditions checked (from the upstream implementation):
{conditions}

Exact failure messages this call can produce:
{messages}\
"""


def _write_contract_description(description: str, tool: Any, owner: type) -> str:
    """tau2's description, plus the preconditions its own source enumerates."""
    fn = _fn_of(tool)
    conditions = _declared_raises(tool) or [
        "(upstream documents no Raises: section; the messages below are the checks)"
    ]
    messages = _raise_messages(fn, owner)
    if not messages:
        raise AssertionError(f"{fn.__name__} raises nothing -- is it really a write?")
    block = _FAILURE_CONTRACT.format(
        conditions="\n".join(f"  - {c}" for c in conditions),
        messages="\n".join(f"  - {m!r}" for m in messages),
    )
    return f"{description.strip()}\n\n{block}"


def _permissive(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop the top-level ``type`` and every ``required``, keep the guidance.

    What is left is a pure *description* of the success payload -- property
    names, types, enums and prose -- with no assertion that a response must BE
    that payload. The simulator still gets every field-level hint; it is simply
    no longer forbidden from reporting a failure.
    """

    def strip_required(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip_required(v) for k, v in node.items() if k != "required"}
        if isinstance(node, list):
            return [strip_required(v) for v in node]
        return node

    out = strip_required(copy.deepcopy(schema))
    out.pop("type", None)
    return out


def _success_example_args(db: Any) -> dict[str, dict[str, Any]]:
    """Arguments that make each write tool *succeed* against tau2's shipped DB.

    Searched, not written down: every candidate is drawn from the corpus in
    sorted-key order, so the choice is deterministic and re-derives itself if
    the DB ever moves. Payment methods are picked non-gift-card wherever tau2
    guards on a gift-card balance, so the example never depends on a balance.
    """

    def by_status(status: str) -> list[Any]:
        return [db.orders[k] for k in sorted(db.orders) if db.orders[k].status == status]

    def cards(user_id: str) -> list[str]:
        methods = db.users[user_id].payment_methods
        return [p for p in sorted(methods) if methods[p].source != "gift_card"]

    def swap_target(item: Any) -> str | None:
        variants = db.products[item.product_id].variants
        return next(
            (v for v in sorted(variants) if variants[v].available and v != item.item_id),
            None,
        )

    def address_args(addr: Any) -> dict[str, Any]:
        return {
            "address1": addr.address1,
            "address2": addr.address2,
            "city": addr.city,
            "state": addr.state,
            "country": addr.country,
            "zip": addr.zip,
        }

    def swap_call(candidates: list[Any]) -> dict[str, Any] | None:
        for order in candidates:
            payable = cards(order.user_id)
            if not payable:
                continue
            for item in order.items:
                target = swap_target(item)
                if target is not None:
                    return {
                        "order_id": order.order_id,
                        "item_ids": [item.item_id],
                        "new_item_ids": [target],
                        "payment_method_id": payable[0],
                    }
        return None

    pending, delivered = by_status("pending"), by_status("delivered")
    order, user = pending[0], db.users[sorted(db.users)[0]]

    args: dict[str, dict[str, Any] | None] = {
        "cancel_pending_order": {"order_id": order.order_id, "reason": "no longer needed"},
        # Re-stating the order's own address keeps every value a real corpus
        # value; the example exists to show the response SHAPE, not a diff.
        "modify_pending_order_address": {"order_id": order.order_id, **address_args(order.address)},
        "modify_user_address": {"user_id": user.user_id, **address_args(user.address)},
        "modify_pending_order_items": swap_call(pending),
        "modify_pending_order_payment": next(
            (
                {"order_id": o.order_id, "payment_method_id": alt[0]}
                for o in pending
                if len(o.payment_history) == 1
                and o.payment_history[0].transaction_type == "payment"
                for alt in [
                    [p for p in cards(o.user_id) if p != o.payment_history[0].payment_method_id]
                ]
                if alt
            ),
            None,
        ),
        "exchange_delivered_order_items": swap_call(delivered),
        # The original payment method is always legal for a return.
        "return_delivered_order_items": next(
            (
                {
                    "order_id": o.order_id,
                    "item_ids": [o.items[0].item_id],
                    "payment_method_id": o.payment_history[0].payment_method_id,
                }
                for o in delivered
                if o.payment_history and o.items
            ),
            None,
        ),
    }
    if unfound := sorted(k for k, v in args.items() if v is None):
        raise AssertionError(f"no succeeding example call found in the corpus for {unfound}")
    return {k: v for k, v in args.items() if v is not None}  # narrows the type


def _success_examples(domain: str) -> dict[str, Any]:
    """``{tool_name: response}``, produced by RUNNING tau2's own write tools.

    Not hand-authored and not invented: each example is the literal object
    upstream returns for a real, succeeding call against the shipped corpus. An
    example that drifts from tau2's semantics is therefore impossible -- if the
    call stopped succeeding, the export fails here rather than shipping a lie.
    """
    args_for = _success_example_args(_tau2_toolkit(domain).db)
    out: dict[str, Any] = {}
    for name, args in sorted(args_for.items()):
        toolkit = _tau2_toolkit(domain)  # fresh DB: writes must not compound
        result = getattr(toolkit, name)(**args)
        out[name] = json.loads(json.dumps(result.model_dump(), default=str))
    return out


# --- ontology --------------------------------------------------------------

# The ontology agent writes schemas/<domain>.ledger.json in parallel. When it is
# present we validate every reference against the real declaration; when it is
# not we derive the identical field sets from tau2's models + contract.py, so
# this exporter is never blocked on it.
_RETAIL_MODELS = {
    "product": "Product",
    "item": "Variant",
    "user": "User",
    "order": "Order",
}


def _derived_fields() -> dict[str, list[str]]:
    """Declared field sets, derived the way ``export_ontology.py`` derives them."""
    from tau2.domains.retail import data_model

    out: dict[str, list[str]] = {}
    for entity_type, model_name in _RETAIL_MODELS.items():
        fields = list(getattr(data_model, model_name).model_fields)
        for extra in contract.RETAIL_EXTRA_FIELDS.get(entity_type, []):
            if extra not in fields:
                fields.append(extra)
        out[entity_type] = fields
    return out


def load_declared_fields(ledger_path: Path) -> tuple[dict[str, list[str]], str]:
    """``{entity_type: [field, ...]}`` plus where it came from."""
    if not ledger_path.exists():
        return _derived_fields(), f"derived (no {ledger_path})"

    schema = json.loads(ledger_path.read_text())
    declared = {
        entity["type"]: [f["name"] for f in entity.get("fields", [])]
        for entity in schema["entities"]
    }
    derived = _derived_fields()
    missing = {
        t: sorted(set(derived[t]) - set(declared.get(t, ())))
        for t in derived
        if set(derived[t]) - set(declared.get(t, ()))
    }
    if missing:
        raise AssertionError(f"{ledger_path} is missing derived fields: {missing}")
    return declared, str(ledger_path)


def _payload_fields(entity_type: str) -> list[str]:
    """The fields tau2's *return model* carries -- the projection's shape.

    Driving the projection off the return model (not off the declared field
    set) is what makes a projected response byte-identical to tau2's. The
    declared set is a superset: it also holds the promoted backrefs
    (``item.product_id``) and the flattened filter keys tau2 never returns.
    """
    from tau2.domains.retail import data_model

    return list(getattr(data_model, _RETAIL_MODELS[entity_type]).model_fields)


def _identity_projection(entity_type: str) -> dict[str, str]:
    return {f: f for f in _payload_fields(entity_type)}


# --- v2 declarations -------------------------------------------------------


def _get(entity_type: str, where: list[dict[str, Any]], not_found: str) -> dict[str, Any]:
    return {
        "op": "get",
        "entity_type": entity_type,
        "where": where,
        "project": _identity_projection(entity_type),
        "not_found": {"error": not_found},
    }


def _by_id(entity_type: str, param: str, not_found: str) -> dict[str, Any]:
    id_field = contract.RETAIL_ID_FIELDS[entity_type]
    return _get(entity_type, [{"field": id_field, "cmp": "eq", "value": {"param": param}}], not_found)


def retail_ledger_reads() -> dict[str, dict[str, Any]]:
    """The seven pure reads, as projections.

    Two of them only exist because of decisions made in ``export_world.py``:
    ``get_item_details`` projects the *promoted* ``item`` type (tau2 stores
    variants nested under a product, where nothing could address them), and
    ``find_user_id_by_name_zip`` filters on the *flattened* ``first_name`` /
    ``last_name`` / ``zip`` keys, because ``where.field`` names a declared
    scalar and cannot walk into ``name{}`` / ``address{}``.

    ``ilike`` rather than ``eq`` on the name and email filters: tau2 compares
    ``.lower() == .lower()``, and ``ilike`` with a wildcard-free pattern is
    exactly a case-insensitive equality. The pattern passes through verbatim
    (no escaping), so this is only safe because no retail email or name
    contains ``%`` or ``_`` -- asserted in ``_assert_world_assumptions``.
    """
    return {
        "get_order_details": _by_id("order", "order_id", "Order not found"),
        "get_product_details": _by_id("product", "product_id", "Product not found"),
        "get_item_details": _by_id("item", "item_id", "Item not found"),
        "get_user_details": _by_id("user", "user_id", "User not found"),
        "find_user_id_by_email": {
            "op": "get",
            "entity_type": "user",
            "where": [{"field": "email", "cmp": "ilike", "value": {"param": "email"}}],
            "project": {"user_id": "user_id"},
            "not_found": {"error": "User not found"},
        },
        "find_user_id_by_name_zip": {
            "op": "get",
            "entity_type": "user",
            "where": [
                {"field": "first_name", "cmp": "ilike", "value": {"param": "first_name"}},
                {"field": "last_name", "cmp": "ilike", "value": {"param": "last_name"}},
                {"field": "zip", "cmp": "eq", "value": {"param": "zip"}},
            ],
            "project": {"user_id": "user_id"},
            "not_found": {"error": "User not found"},
        },
        # A no-argument `list` with no `where` is explicitly legal, and is the
        # one tool the simulator's working set could never have bound.
        "list_all_product_types": {
            "op": "list",
            "entity_type": "product",
            "order_by": [{"field": "name", "dir": "asc"}],
            "limit": 200,
            "project": {"name": "name", "product_id": "product_id"},
            "wrap": "product_types",
        },
    }


# `when` is the declared precondition. tau2's `_is_pending_order` accepts both
# pending statuses, and `when.eq` takes a scalar constant only -- so a tool
# guarded by that predicate becomes two effects, one per legal status. Exactly
# one can fire; a domain-failure response (which carries no `status`) fires
# neither, and writes nothing.
_PENDING_STATUSES = ("pending", "pending (item modified)")


def _pending_guarded(field_map: dict[str, str], order_param: str = "order_id") -> dict[str, Any]:
    return {
        "effects": [
            {
                "op": "update",
                "entity_type": "order",
                "id_from": f"$args.{order_param}",
                "field_map": dict(field_map),
                "when": {"field": "status", "eq": status},
            }
            for status in _PENDING_STATUSES
        ]
    }


def retail_ledger_adapters() -> dict[str, dict[str, Any]]:
    """The seven writes, as declared effects.

    Values come from ``$args`` wherever the argument *is* the stored value, and
    from the response only where tau2 computes something (a status transition,
    a refund appended to ``payment_history``, the sorted item lists, the
    exchange price difference). That split is deliberate: an ``$args`` path is
    deterministic, a response path is only as good as the simulator's grounding.

    Every effect is guarded by a ``when`` predicate keyed on a value only a
    SUCCESS payload carries -- for the six order writes, the status tau2
    transitions *to*; for the one user write, the presence of ``user_id``. A
    call the domain rejects (a non-pending order, an unavailable variant, an id
    that exists nowhere) comes back as ``{"error": ...}``, which carries none of
    those, so the predicate is false and the adapter performs zero operations
    rather than half-applying a write tau2 never performed.

    That guard is only as good as the response it reads, which is the other half
    of the fix: see the write-tool response contract above for why the
    ``output_schema`` had to stop making failure unrepresentable before this
    predicate could ever be false.
    """
    return {
        "cancel_pending_order": {
            "op": "update",
            "entity_type": "order",
            "id_from": "$args.order_id",
            "field_map": {
                "status": "status",
                "cancel_reason": "$args.reason",
                "payment_history": "payment_history",
            },
            "flags": ["order_cancelled:{order_id}"],
            "when": {"field": "status", "eq": "cancelled"},
        },
        # No status transition, so the pending precondition is the guard.
        # `address` is copied wholesale from the response: field_map sets one
        # declared field from one path, so a six-argument object cannot be
        # assembled from $args.
        "modify_pending_order_address": _pending_guarded({"address": "address"}),
        "modify_pending_order_items": {
            "op": "update",
            "entity_type": "order",
            "id_from": "$args.order_id",
            "field_map": {
                "status": "status",
                "items": "items",
                "payment_history": "payment_history",
            },
            "flags": ["order_items_modified:{order_id}"],
            "when": {"field": "status", "eq": "pending (item modified)"},
        },
        "modify_pending_order_payment": _pending_guarded(
            {"payment_history": "payment_history"}
        ),
        # The only write on `user`. It must also refresh the flattened `zip`
        # filter key, or find_user_id_by_name_zip would keep resolving on the
        # pre-move zip.
        #
        # This is the one write with no status transition to key a guard on: it
        # returns a User, and no scalar on a User is CONSTANT across users, so
        # `eq` has nothing to compare against. The discriminator is therefore
        # the *presence* of `user_id` -- a User payload always carries it, and
        # the `{"error": ...}` failure body never does. It has to be a presence
        # test rather than a value test, which is why this one predicate uses
        # `exists` (`_validate_ledger_adapter` accepts eq|neq|exists|empty; the
        # tools.mdx table documents only `eq`).
        #
        # Leaving it unguarded is not the harmless option it looks like: `zip`
        # is written straight from `$args`, so a rejected call would still move
        # the user's zip -- and silently break the name+zip lookup.
        "modify_user_address": {
            "op": "update",
            "entity_type": "user",
            "id_from": "$args.user_id",
            "field_map": {"address": "address", "zip": "$args.zip"},
            "flags": ["user_address_modified:{user_id}"],
            "when": {"field": "user_id", "exists": True},
        },
        "exchange_delivered_order_items": {
            "op": "update",
            "entity_type": "order",
            "id_from": "$args.order_id",
            "field_map": {
                "status": "status",
                # tau2 stores these *sorted*; $args carries call order.
                "exchange_items": "exchange_items",
                "exchange_new_items": "exchange_new_items",
                "exchange_payment_method_id": "$args.payment_method_id",
                "exchange_price_difference": "exchange_price_difference",
            },
            "flags": ["order_exchange_requested:{order_id}"],
            "when": {"field": "status", "eq": "exchange requested"},
        },
        "return_delivered_order_items": {
            "op": "update",
            "entity_type": "order",
            "id_from": "$args.order_id",
            "field_map": {
                "status": "status",
                "return_items": "return_items",
                "return_payment_method_id": "$args.payment_method_id",
            },
            "flags": ["order_return_requested:{order_id}"],
            "when": {"field": "status", "eq": "return requested"},
        },
    }


# The two tools that are neither a read nor a write.
#
# `calculate` is a pure function over its own argument -- it touches no state,
# so simulating it would spend a model call to guess arithmetic. Executed mode
# runs tau2's real body in the sandbox: exact semantics, deterministic, no
# ledger involvement at all (it has no data operations, so there is nothing to
# route through the /data plane).
#
# `transfer_to_human_agents` returns a constant and mutates nothing, but the
# *fact* of the transfer is graded -- transferring is correct on some tau2
# tasks and a failure on others. Executed mode would make the response
# deterministic but drops the ledger write policy entirely, leaving no world
# state for a scorer to bind to. So it stays Simulated with a `set_flag`-only
# adapter: the one thing that matters lands deterministically on the ledger,
# and the cost is a single model call for a fixed string no scorer reads.
_RETAIL_SPECIAL: dict[str, dict[str, Any]] = {
    "calculate": {
        "default_execution_mode": "executed",
        "ledger_write_policy": "none",
    },
    "transfer_to_human_agents": {
        "ledger_write_policy": "adapter",
        "ledger_adapter": {"op": "set_flag", "flags": ["transferred_to_human"]},
    },
}

# v2 projections whose payload cannot be tau2's literal return type, so the
# level-2 output_schema describes the projection instead of the return model.
_RETAIL_PROJECTED_OUTPUTS: dict[str, dict[str, Any]] = {
    "find_user_id_by_email": {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    },
    "find_user_id_by_name_zip": {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    },
    "list_all_product_types": {
        "type": "object",
        "properties": {
            "product_types": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "product_id": {"type": "string"},
                    },
                    "required": ["name", "product_id"],
                },
            }
        },
        "required": ["product_types"],
    },
}


# --- assembly --------------------------------------------------------------


def build_entries(domain: str, level: str, declared: dict[str, list[str]]) -> list[dict[str, Any]]:
    toolkit = _tau2_toolkit(domain)
    tools = toolkit.get_tools()
    writes = _write_tool_names(toolkit)
    reads = retail_ledger_reads()
    adapters = retail_ledger_adapters()

    unknown = (set(reads) | set(adapters) | set(_RETAIL_SPECIAL)) - set(tools)
    if unknown:
        raise AssertionError(f"declared tools tau2 does not have: {sorted(unknown)}")
    uncovered = set(tools) - set(reads) - set(adapters) - set(_RETAIL_SPECIAL)
    if uncovered:
        raise AssertionError(f"tau2 tools with no declared treatment: {sorted(uncovered)}")
    if undeclared := sorted(writes - set(adapters)):
        raise AssertionError(f"tau2 write tools with no ledger_adapter: {undeclared}")

    # Only needed for the levels that ship a response contract; running tau2's
    # write tools is the expensive part of an export, so it is not paid at v0.
    examples = _success_examples(domain) if level in ("v1", "v2") else {}

    entries: list[dict[str, Any]] = []
    for name in sorted(tools):
        tool_name, description, input_schema = _name_desc_params(tools[name])
        assert tool_name == name, f"tau2 tool key {name!r} != schema name {tool_name!r}"
        entry: dict[str, Any] = {
            "name": tool_name,
            "description": description,
            "input_schema": input_schema,
        }
        if level in ("v1", "v2"):
            schema = _output_schema(tools[name])
            if name in writes:
                # The three-part write contract: a schema that can express
                # failure, a worked success payload so the permissiveness costs
                # no field-level grounding, and the preconditions that say when
                # to fail and in whose words.
                schema = _permissive(schema)
                schema["examples"] = [examples[name]]
                entry["description"] = _write_contract_description(
                    description, tools[name], type(toolkit)
                )
            entry["output_schema"] = schema
        if level == "v2":
            if name in _RETAIL_PROJECTED_OUTPUTS:
                entry["output_schema"] = copy.deepcopy(_RETAIL_PROJECTED_OUTPUTS[name])
            if name in reads:
                entry["ledger_read"] = copy.deepcopy(reads[name])
                entry["ledger_write_policy"] = "none"
            elif name in adapters:
                entry["ledger_write_policy"] = "adapter"
                entry["ledger_adapter"] = copy.deepcopy(adapters[name])
            entry.update(copy.deepcopy(_RETAIL_SPECIAL.get(name, {})))
        entries.append(entry)

    if level == "v2":
        assert_references_resolve(entries, declared, writes)
    return entries


# --- export-time validation ------------------------------------------------
#
# Every rule below is a 422 at registration. Failing here instead means the
# error names the tool and the field rather than arriving from the API.

_GET_ONLY = {"not_found"}
_LIST_ONLY = {"order_by", "limit", "wrap"}
_READ_KEYS = {"op", "entity_type", "where", "project", "nest"} | _GET_ONLY | _LIST_ONLY
_PARAM_ONLY_CMP = {"ilike", "icontains"}
# Exactly one of these per `when`, per _validate_ledger_adapter.
_WHEN_OPS = {"eq", "neq", "exists", "empty"}


def _check_read(name: str, read: dict[str, Any], props: set[str], declared: dict[str, list[str]]) -> None:
    def fail(msg: str) -> None:
        raise AssertionError(f"{name}.ledger_read: {msg}")

    if unknown := set(read) - _READ_KEYS:
        fail(f"unknown keys {sorted(unknown)}")
    op = read.get("op")
    if op not in ("get", "list"):
        fail(f"op must be get|list, got {op!r}")
    if op == "get" and (bad := set(read) & _LIST_ONLY):
        fail(f"{sorted(bad)} is list-only")
    if op == "list" and (bad := set(read) & _GET_ONLY):
        fail(f"{sorted(bad)} is get-only")

    entity_type = read["entity_type"]
    if entity_type not in declared:
        fail(f"entity_type {entity_type!r} is not in ledger_schema.entities")
    fields = set(declared[entity_type]) | {"id", contract.RETAIL_ID_FIELDS[entity_type]}

    if not read.get("project"):
        fail("project is required and non-empty")
    for key, source in read["project"].items():
        if source not in fields:
            fail(f"project[{key!r}] -> undeclared field {source!r}")
    filter_only = contract.RETAIL_FILTER_ONLY_FIELDS.get(entity_type, set())
    if leaked := filter_only & set(read["project"].values()):
        fail(f"projects filter-only field(s) {sorted(leaked)} -- would change tau2's payload")

    for cond in read.get("where", []):
        if cond["field"] not in fields:
            fail(f"where on undeclared field {cond['field']!r}")
        value = cond["value"]
        if "param" in value:
            if value["param"] not in props:
                fail(f"where binds {value['param']!r}, absent from input_schema.properties")
        elif "const" not in value:
            fail(f"where value must be a param ref or a const, got {value!r}")
        elif cond["cmp"] in _PARAM_ONLY_CMP:
            fail(f"cmp {cond['cmp']!r} takes param refs only")

    for key, block in (read.get("nest") or {}).items():
        if "nest" in block:
            fail(f"nest[{key!r}] nests again -- one level only")
        if block.get("where"):
            fail(f"nest[{key!r}] is projection-only")

    for cond in read.get("order_by", []):
        if cond["field"] not in fields:
            fail(f"order_by on undeclared field {cond['field']!r}")


def _check_adapter(
    name: str,
    adapter: dict[str, Any],
    props: set[str],
    declared: dict[str, list[str]],
    require_when: bool = False,
) -> None:
    def fail(msg: str) -> None:
        raise AssertionError(f"{name}.ledger_adapter: {msg}")

    effects = adapter.get("effects", [adapter])
    if "effects" in adapter and set(adapter) != {"effects"}:
        fail("'effects' must not sit beside top-level effect keys")
    if len(effects) > 10:
        fail(f"{len(effects)} effects, cap is 10")
    for effect in effects:
        for template in effect.get("flags", []):
            for slot in re.findall(r"\{([^}]*)\}", template):
                if slot not in props:
                    fail(f"flag {template!r} interpolates {slot!r}, not a call argument")
        op = effect.get("op", "update")
        if op == "set_flag":
            if effect.get("entity_type"):
                fail("set_flag must not set entity_type")
            if not effect.get("flags"):
                fail("set_flag requires a non-empty flags list")
            continue
        entity_type = effect.get("entity_type")
        if entity_type not in declared:
            fail(f"entity_type {entity_type!r} is not in ledger_schema.entities")
        if not effect.get("id_from"):
            fail(f"op={op!r} requires id_from")
        fields = set(declared[entity_type]) | {contract.RETAIL_ID_FIELDS[entity_type]}
        for target, path in {**effect.get("field_map", {}), **effect.get("list_append", {})}.items():
            if target not in fields:
                fail(f"writes undeclared field {target!r} on {entity_type}")
            if path.startswith("$args.") and path[len("$args."):] not in props:
                fail(f"{path!r} is not an input_schema property")
        for path in [effect["id_from"]]:
            if path.startswith("$args.") and path[len("$args."):] not in props:
                fail(f"id_from {path!r} is not an input_schema property")
        when = effect.get("when")
        if when is None:
            if require_when:
                fail(
                    "a write effect must carry a `when` guard, or a response that "
                    "reports a domain failure still mutates the ledger"
                )
        elif set(when) - {"field", *_WHEN_OPS} or len(set(when) & _WHEN_OPS) != 1:
            fail(f"when must be {{field, <one of {sorted(_WHEN_OPS)}>}}, got {sorted(when)}")


def assert_references_resolve(
    entries: list[dict[str, Any]],
    declared: dict[str, list[str]],
    writes: set[str] | None = None,
) -> None:
    writes = writes or set()
    for entry in entries:
        name = entry["name"]
        props = set(entry["input_schema"].get("properties", {}))
        read = entry.get("ledger_read")
        adapter = entry.get("ledger_adapter")
        if read is not None:
            if entry.get("default_execution_mode", "simulated") != "simulated":
                raise AssertionError(f"{name}: ledger_read is Simulated-mode only")
            effects = (adapter or {}).get("effects", [adapter]) if adapter else []
            if any(e.get("op", "update") != "set_flag" for e in effects):
                raise AssertionError(f"{name}: ledger_read composes only with set_flag adapters")
            _check_read(name, read, props, declared)
        if adapter is not None:
            if entry.get("ledger_write_policy") != "adapter":
                raise AssertionError(f"{name}: ledger_adapter needs ledger_write_policy='adapter'")
            _check_adapter(name, adapter, props, declared, require_when=name in writes)
        elif entry.get("ledger_write_policy") == "adapter":
            raise AssertionError(f"{name}: ledger_write_policy='adapter' needs a ledger_adapter")

        # Fidelity: a projection must reproduce tau2's payload exactly.
        if read is not None and read["project"] == _identity_projection(read["entity_type"]):
            allowed = set(contract.projected_fields(read["entity_type"], declared))
            if leaked := set(read["project"].values()) - allowed:
                raise AssertionError(f"{name}: projects {sorted(leaked)} outside the payload set")


def _assert_world_assumptions(domain: str) -> None:
    """The three data facts the projections lean on. Cheap to check, fatal to assume.

    1. ``ilike`` passes its pattern through verbatim, so a stored value or an
       argument containing ``%`` / ``_`` would turn an equality into a wildcard.
    2. ``find_user_id_by_name_zip`` is a ``get``; ambiguity would silently
       return the first row under the total order.
    3. ``list_all_product_types`` builds a name-keyed dict upstream, which
       dedups. A ``list`` projection cannot dedup, so it is only equivalent
       while product names are unique.
    """
    if domain != "retail":
        return
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    db = RetailDB.load(RETAIL_DB_PATH)
    users = db.users.values()
    patterned = [
        v
        for u in users
        for v in (u.email, u.name.first_name, u.name.last_name)
        if "%" in v or "_" in v
    ]
    assert not patterned, f"ilike filter values contain SQL wildcards: {patterned[:5]}"

    emails = [u.email.lower() for u in users]
    assert len(set(emails)) == len(emails), "emails are not unique -- find_user_id_by_email is ambiguous"

    name_zip = [(u.name.first_name.lower(), u.name.last_name.lower(), u.address.zip) for u in users]
    assert len(set(name_zip)) == len(name_zip), "name+zip is not unique -- the get would be ambiguous"

    product_names = [p.name for p in db.products.values()]
    assert len(set(product_names)) == len(product_names), (
        "product names collide -- a list projection cannot dedup them the way "
        "tau2's name-keyed dict does; fall back to simulation for "
        "list_all_product_types"
    )


# --- cli -------------------------------------------------------------------


def _out_path(base: Path, level: str, single: bool) -> Path:
    if single:
        return base
    return base.with_name(f"{base.name.removesuffix('.json')}.{level}.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=["retail"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level", default="v2", choices=[*LEVELS, "all"])
    ap.add_argument(
        "--ledger",
        type=Path,
        help="ledger_schema JSON to validate references against "
        "(default schemas/<domain>.ledger.json, derived from tau2's models if absent)",
    )
    args = ap.parse_args(argv)

    _assert_world_assumptions(args.domain)
    ledger_path = args.ledger or REPO_ROOT / "schemas" / f"{args.domain}.ledger.json"
    declared, source = load_declared_fields(ledger_path)
    print(f"ledger_schema fields: {source}")

    levels = LEVELS if args.level == "all" else (args.level,)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for level in levels:
        entries = build_entries(args.domain, level, declared)
        path = _out_path(args.out, level, single=args.level != "all")
        path.write_text(json.dumps(entries, indent=2))
        widest = max(len(json.dumps(e)) for e in entries)
        reads = sum("ledger_read" in e for e in entries)
        writes = sum(e.get("ledger_write_policy") == "adapter" for e in entries)
        print(
            f"wrote {path} ({level}: {len(entries)} tools, "
            f"{reads} projected, {writes} adapter-backed, "
            f"{path.stat().st_size / 1024:.1f} KB total, {widest / 1024:.1f} KB widest entry)"
        )
        if args.level == "all" and level == "v2":
            # --level all fans out to the three .v0/.v1/.v2 files, but --out
            # itself is the canonical path everything else (validate.py's
            # TOOLS_PATH, the platform registration payload) reads -- v2 is
            # the real artifact (see the module docstring), so --out must end
            # up byte-identical to retail.tools.v2.json, not stale.
            args.out.write_text(json.dumps(entries, indent=2))
            print(f"wrote {args.out} (canonical copy of v2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
