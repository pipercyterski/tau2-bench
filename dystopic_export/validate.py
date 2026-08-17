"""Cross-check the exported artifacts against *each other* and against tau2.

Each exporter validates its own output in isolation. That is not enough: the
artifacts are only useful as a *set*, and every interesting failure mode lives
in the seams between them --

* a ``ledger_read`` projecting a field the ontology never declared,
* a ``{"param": x}`` that no ``input_schema`` property can bind,
* a projection leaking a filter-only key (which silently changes the payload
  shape the model sees, and would only be caught by a byte-diff against tau2),
* a tool name that drifted from ``Environment.get_info()``,
* a scenario naming a tool that no longer exists,
* a write tool whose adapter has no ``when`` guard, so a simulated response
  reporting a domain failure is written to the ledger anyway.

None of those are visible from inside a single exporter. All of them cost a
live check run to discover. So they are checked here, independently -- this
module re-derives everything from the *emitted artifacts* plus tau2 itself and
deliberately does not import the exporters' own helpers (importing them would
make the check agree with the bug).

Usage::

    python -m dystopic_export.validate                 # structural cross-checks
    python -m dystopic_export.validate --platform PATH # + the real platform validators

Exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

LEDGER_PATH = REPO_ROOT / "schemas" / "retail.ledger.json"
TOOLS_PATH = REPO_ROOT / "schemas" / "retail.tools.json"
TOOLS_LEVEL_PATHS = {
    "v0": REPO_ROOT / "schemas" / "retail.tools.v0.json",
    "v1": REPO_ROOT / "schemas" / "retail.tools.v1.json",
    "v2": REPO_ROOT / "schemas" / "retail.tools.v2.json",
}
WORLD_PATH = REPO_ROOT / "worlds" / "retail.json"
SCENARIO_SPLITS = ("test", "train", "base")

LEGAL_FIELD_TYPES = {"string", "number", "integer", "boolean", "object", "array"}
IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# A `when` predicate carries a `field` plus exactly one of these operators
# (``_validate_ledger_adapter`` in apps/api/app/schemas/agent.py). Restated here
# rather than imported, for the same reason FILTER_ONLY is.
WHEN_OPS = {"eq", "neq", "exists", "empty"}

# The filter-only keys, restated here rather than imported from ``contract`` so
# a change to the contract cannot silently retire the leak check.
FILTER_ONLY = {
    "user": {"first_name", "last_name", "zip"},
    "item": {"product_name"},
}

# initial_state key -> declared singular type. export_world.py seeds the
# singular, declared-type spelling directly (its own comment: this is what
# makes seed key and declared type identical, so the ledger's plural->singular
# canonicalization never has to fire and never emits an advisory) -- but the
# ledger accepts either spelling per contract.py rule #1, so a lookup here
# must too, or a correctly-seeded world reads as ungrounded.
WORLD_KEY_FOR_TYPE = {
    "product": "products",
    "item": "items",
    "user": "users",
    "order": "orders",
}


def _world_key(etype: str, state: dict) -> str | None:
    """The key ``etype``'s rows actually live under in an exported world.

    Tries the plural tau2-shaped key first, then the singular declared-type
    key -- the two spellings the real ledger bootstrap canonicalizes between.
    """
    plural = WORLD_KEY_FOR_TYPE.get(etype)
    if plural is not None and plural in state:
        return plural
    if etype in state:
        return etype
    return None


class Violations:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []

    def add(self, check: str, msg: str) -> None:
        self.rows.append((check, msg))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rows)


def load(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# ontology index
# --------------------------------------------------------------------------


def index_ledger(ledger: dict) -> dict[str, dict]:
    """{entity_type: {"fields": {name: fielddict}, "id_field": str}}."""
    out: dict[str, dict] = {}
    for ent in ledger.get("entities") or []:
        fields = {f["name"]: f for f in ent.get("fields") or [] if isinstance(f, dict)}
        out[ent["type"]] = {"fields": fields, "id_field": ent.get("id_field")}
    return out


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_ledger_self(ledger: dict, idx: dict[str, dict], v: Violations) -> None:
    c = "ledger.self"
    policy = ledger.get("field_policy", "open")
    if policy not in ("open", "closed"):
        v.add(c, f"field_policy={policy!r} is not open|closed")
    seen: set[str] = set()
    for etype, ent in idx.items():
        if not IDENT_RE.match(etype):
            v.add(c, f"entity type {etype!r} is not ^[a-z][a-z0-9_]*$")
        if etype in seen:
            v.add(c, f"duplicate entity type {etype!r}")
        seen.add(etype)
        if ent["id_field"] not in ent["fields"]:
            v.add(c, f"{etype}.id_field={ent['id_field']!r} is not a declared field")
        for fname, f in ent["fields"].items():
            if not IDENT_RE.match(fname):
                v.add(c, f"{etype}.{fname} is not ^[a-z][a-z0-9_]*$")
            if f.get("type") not in LEGAL_FIELD_TYPES:
                v.add(
                    c,
                    f"{etype}.{fname}.type={f.get('type')!r} is not one of {sorted(LEGAL_FIELD_TYPES)}",
                )
            if "enum" in f and fname.endswith("_id"):
                v.add(c, f"{etype}.{fname} carries an enum on an id field")


def check_tool_names_vs_tau2(tools: list[dict], v: Violations) -> set[str]:
    """Tool names must byte-match ``Environment.get_info()``; so must inputs."""
    c = "tools.tau2_parity"
    names = [t["name"] for t in tools]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        v.add(c, f"duplicate tool names in tools schema: {dupes}")
    try:
        from tau2.domains.retail.environment import get_environment

        defs = get_environment().get_info(include_tool_info=True).tool_defs
    except Exception as exc:  # pragma: no cover - env problem, not a defect
        v.add(c, f"could not load tau2 retail environment: {exc!r}")
        return set(names)

    tau2_names = set(defs)
    ours = set(names)
    for missing in sorted(tau2_names - ours):
        v.add(c, f"tau2 exposes tool {missing!r} but the tools schema does not")
    for extra in sorted(ours - tau2_names):
        v.add(c, f"tools schema declares tool {extra!r} which tau2 does not expose")

    for t in tools:
        sig = defs.get(t["name"])
        if sig is None:
            continue
        if t["name"] != sig.name:
            v.add(c, f"tool name {t['name']!r} != tau2 ToolSignature.name {sig.name!r}")
        if t.get("input_schema") != sig.params:
            v.add(
                c,
                f"{t['name']}.input_schema is not byte-identical to tau2's params "
                f"(ours={json.dumps(t.get('input_schema'), sort_keys=True)[:200]} ...)",
            )
    return ours


def _param_refs(node: Any) -> set[str]:
    """Every ``{"param": "x"}`` name anywhere under ``node``."""
    found: set[str] = set()
    if isinstance(node, dict):
        if isinstance(node.get("param"), str) and set(node) == {"param"}:
            found.add(node["param"])
        for val in node.values():
            found |= _param_refs(val)
    elif isinstance(node, list):
        for item in node:
            found |= _param_refs(item)
    return found


def check_ledger_reads(tools: list[dict], idx: dict[str, dict], v: Violations) -> None:
    c = "ledger_read"
    for t in tools:
        read = t.get("ledger_read")
        if read is None:
            continue
        name = t["name"]
        props = set((t.get("input_schema") or {}).get("properties") or {})

        # Both spellings: the public vocabulary is "executed", the stored/legacy
        # one is "code_intercepted", and the platform accepts either.
        if t.get("default_execution_mode") in ("executed", "code_intercepted"):
            v.add(c, f"{name}: ledger_read declared on an executed tool")

        etype = read.get("entity_type")
        if etype not in idx:
            v.add(
                c, f"{name}: entity_type={etype!r} is not declared in the ledger schema"
            )
            continue
        declared = set(idx[etype]["fields"])
        filter_only = FILTER_ONLY.get(etype, set())
        op = read.get("op")

        # where.field must be declared
        for i, cond in enumerate(read.get("where") or []):
            fld = cond.get("field")
            if fld not in declared and fld not in ("id", idx[etype]["id_field"]):
                v.add(
                    c,
                    f"{name}: where[{i}].field={fld!r} is not a declared field of {etype!r}",
                )

        # project sources must be declared and must not leak a filter-only key
        for out_key, src in (read.get("project") or {}).items():
            if src not in declared and src not in ("id", idx[etype]["id_field"]):
                v.add(
                    c,
                    f"{name}: project[{out_key!r}] reads undeclared field {src!r} of {etype!r}",
                )
            if src in filter_only:
                v.add(
                    c,
                    f"{name}: project[{out_key!r}] leaks filter-only field {src!r} of {etype!r} "
                    "-- the payload would no longer match tau2's",
                )

        # order_by fields must be declared
        for i, ob in enumerate(read.get("order_by") or []):
            fld = ob.get("field")
            if fld not in declared:
                v.add(
                    c,
                    f"{name}: order_by[{i}].field={fld!r} is not a declared field of {etype!r}",
                )

        # every {"param": x} must be bindable
        for ref in sorted(_param_refs(read)):
            if ref not in props:
                v.add(
                    c,
                    f"{name}: {{'param': {ref!r}}} names no input_schema property "
                    f"(declared: {sorted(props)})",
                )

        # op-shape exclusions
        if op == "get":
            if "order_by" in read:
                v.add(c, f"{name}: op='get' must not carry order_by")
            if "limit" in read:
                v.add(c, f"{name}: op='get' must not carry limit")
            if "wrap" in read:
                v.add(c, f"{name}: op='get' must not carry wrap")
        elif op == "list":
            if "not_found" in read:
                v.add(c, f"{name}: op='list' must not carry not_found")
        else:
            v.add(c, f"{name}: ledger_read.op={op!r} is neither 'get' nor 'list'")


def _adapter_effects(adapter: dict) -> list[dict]:
    raw = adapter.get("effects")
    return raw if isinstance(raw, list) else [adapter]


def check_ledger_adapters(
    tools: list[dict], idx: dict[str, dict], ledger: dict, v: Violations
) -> None:
    c = "ledger_adapter"
    flags_block = ledger.get("flags")
    flag_closure: set[str] | None = None
    if isinstance(flags_block, dict) and flags_block.get("policy") == "closed":
        flag_closure = set(
            flags_block.get("declared") or flags_block.get("names") or []
        )

    for t in tools:
        name = t["name"]
        adapter = t.get("ledger_adapter")
        policy = t.get("ledger_write_policy")
        if adapter is None:
            if policy == "adapter":
                v.add(c, f"{name}: ledger_write_policy='adapter' but no ledger_adapter")
            continue
        if policy != "adapter":
            v.add(
                c, f"{name}: declares ledger_adapter but ledger_write_policy={policy!r}"
            )
        props = set((t.get("input_schema") or {}).get("properties") or {})

        for i, eff in enumerate(_adapter_effects(adapter)):
            loc = f"{name}.effects[{i}]" if "effects" in adapter else name
            op = eff.get("op", "update")

            for flag in eff.get("flags") or []:
                for ref in re.findall(r"\{([^}]*)\}", flag):
                    if ref not in props:
                        v.add(
                            c,
                            f"{loc}: flag template {flag!r} interpolates {ref!r}, "
                            f"which is not an input_schema property (declared: {sorted(props)})",
                        )
                if flag_closure is not None and flag not in flag_closure:
                    v.add(c, f"{loc}: flag {flag!r} is not in the closed flag closure")

            if op == "set_flag":
                continue

            etype = eff.get("entity_type")
            if etype not in idx:
                v.add(
                    c,
                    f"{loc}: entity_type={etype!r} is not declared in the ledger schema",
                )
                continue
            declared = set(idx[etype]["fields"])

            id_from = eff.get("id_from")
            if isinstance(id_from, str) and id_from.startswith("$args."):
                ref = id_from[len("$args.") :]
                if ref not in props:
                    v.add(
                        c, f"{loc}: id_from={id_from!r} names no input_schema property"
                    )

            for map_key in ("field_map", "list_append"):
                for fld, src in (eff.get(map_key) or {}).items():
                    if fld not in declared:
                        v.add(
                            c,
                            f"{loc}: {map_key} writes undeclared field {fld!r} on {etype!r} "
                            f"(declared: {sorted(declared)})",
                        )
                    if isinstance(src, str) and src.startswith("$args."):
                        ref = src[len("$args.") :]
                        if ref not in props:
                            v.add(
                                c,
                                f"{loc}: {map_key}[{fld!r}]={src!r} names no input_schema property",
                            )

            when = eff.get("when")
            if isinstance(when, dict):
                ops = sorted(set(when) - {"field"})
                if len(ops) != 1 or ops[0] not in WHEN_OPS:
                    v.add(
                        c,
                        f"{loc}: when declares {ops} -- exactly one of {sorted(WHEN_OPS)} "
                        "is allowed beside 'field'",
                    )
                wf = when.get("field")
                if wf not in declared:
                    v.add(
                        c,
                        f"{loc}: when.field={wf!r} is not a declared field of {etype!r}",
                    )
                else:
                    enum = idx[etype]["fields"][wf].get("enum")
                    if enum is not None and "eq" in when and when["eq"] not in enum:
                        v.add(
                            c,
                            f"{loc}: when.eq={when['eq']!r} is outside the declared enum "
                            f"for {etype}.{wf} ({enum})",
                        )


def check_write_tools_are_guarded(tools: list[dict], v: Violations) -> None:
    """Every tau2 WRITE tool must gate its adapter on a success discriminator.

    This is the regression guard for the failure that motivated the write-tool
    response contract: an unguarded adapter applies its ``field_map`` to
    whatever the simulator returned, so a call the real domain would have
    rejected still mutates the world -- and the agent, told the write succeeded,
    never gets the ``Error: ...`` it would have self-corrected from.

    The write set is taken from tau2's own ``@is_tool(ToolType.WRITE)``
    classification, not from the exporter, so a tool that changes category
    upstream is caught here rather than inherited.
    """
    c = "write_guard"
    try:
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.tools import RetailTools
        from tau2.domains.retail.utils import RETAIL_DB_PATH
        from tau2.environment.toolkit import ToolType, get_tool_types

        types = get_tool_types(RetailTools(RetailDB.load(RETAIL_DB_PATH)))
    except Exception as exc:  # pragma: no cover - env problem, not a defect
        v.add(c, f"could not classify tau2's tools: {exc!r}")
        return

    writes = {name for name, kind in types.items() if kind is ToolType.WRITE}
    by_name = {t["name"]: t for t in tools}
    for name in sorted(writes):
        entry = by_name.get(name)
        if entry is None:
            continue  # already reported by check_tool_names_vs_tau2
        adapter = entry.get("ledger_adapter")
        if adapter is None:
            v.add(c, f"{name}: tau2 classifies it a WRITE but it declares no ledger_adapter")
            continue
        for i, eff in enumerate(_adapter_effects(adapter)):
            loc = f"{name}.effects[{i}]" if "effects" in adapter else name
            when = eff.get("when")
            if not isinstance(when, dict) or not (set(when) & WHEN_OPS):
                v.add(
                    c,
                    f"{loc}: a write effect with no `when` guard -- a response that "
                    "reports a domain failure would still be written to the ledger",
                )


def check_execution_mode_routing(tools: list[dict], v: Violations) -> None:
    """An ``executed`` tool must NOT be dispatched to /odyssey-proxy/tools.

    The proxy refuses that route with a 409 ``tool_executes_in_sandbox`` ("its
    code runs in the sandbox and its data access goes to /odyssey-proxy/data"),
    so a tool declared ``executed`` that the entrypoint does not run locally
    hands the model an error string instead of a result -- silently, and only
    on the tasks that happen to call it.
    """
    c = "execution_mode"
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from dystopic_entry import SANDBOX_EXECUTED
    except Exception as exc:  # pragma: no cover
        v.add(c, f"could not import the entrypoint's sandbox-executed table: {exc!r}")
        return
    declared_executed = {
        t["name"]
        for t in tools
        if t.get("default_execution_mode") in ("executed", "code_intercepted")
    }
    for name in sorted(declared_executed - set(SANDBOX_EXECUTED)):
        v.add(
            c,
            f"{name} is declared executed but the entrypoint routes it to the proxy, "
            "which refuses it with 409 tool_executes_in_sandbox",
        )
    for name in sorted(set(SANDBOX_EXECUTED) - declared_executed):
        v.add(
            c,
            f"{name} is executed locally by the entrypoint but is not declared "
            "executed in the tools schema (the world never sees the call)",
        )


def check_levels(v: Violations) -> None:
    c = "levels"
    levels = {}
    for lvl, path in TOOLS_LEVEL_PATHS.items():
        if not path.exists():
            v.add(c, f"{path.name} is missing")
            return
        levels[lvl] = {t["name"]: t for t in load(path)}
    if not (set(levels["v0"]) == set(levels["v1"]) == set(levels["v2"])):
        v.add(c, "v0/v1/v2 do not declare the same tool set")
    for lower, higher in (("v0", "v1"), ("v1", "v2")):
        for name, entry in levels[lower].items():
            up = levels[higher].get(name)
            if up is None:
                continue
            for key, val in entry.items():
                if key not in up:
                    v.add(c, f"{higher}.{name} dropped key {key!r} present in {lower}")
                elif up[key] == val or key == "output_schema":
                    continue
                elif key == "description":
                    # A level may APPEND to the description -- v1 tells a write
                    # tool's simulator which preconditions to enforce -- but it
                    # must never reword tau2's own text, or the pass-rate delta
                    # between levels stops being attributable to the
                    # declaration. Additive here means "keeps v0 as its prefix".
                    if not isinstance(up[key], str) or not up[key].startswith(val):
                        v.add(
                            c,
                            f"{higher}.{name}.description does not extend {lower}'s "
                            "(a level may append to it, never rewrite it)",
                        )
                else:
                    v.add(
                        c,
                        f"{higher}.{name}.{key} differs from {lower} (levels must be additive)",
                    )

    if TOOLS_PATH.exists():
        if load(TOOLS_PATH) != load(TOOLS_LEVEL_PATHS["v2"]):
            v.add(
                c,
                "schemas/retail.tools.json is not identical to retail.tools.v2.json (stale?)",
            )


def check_scenarios(tool_names: set[str], v: Violations) -> None:
    c = "scenarios"
    for split in SCENARIO_SPLITS:
        spath = REPO_ROOT / "scenarios" / f"retail.{split}.json"
        gpath = REPO_ROOT / "scenarios" / f"retail.{split}.golden.json"
        if not spath.exists():
            v.add(c, f"{spath.name} is missing")
            continue
        scenarios = load(spath)
        names = [s["name"] for s in scenarios]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            v.add(c, f"{split}: duplicate scenario names {dupes}")
        for s in scenarios:
            for tool in s.get("expected_tool_sequence") or []:
                if tool not in tool_names:
                    v.add(
                        c,
                        f"{split}/{s['name']}: expected_tool_sequence names undeclared tool {tool!r}",
                    )
        if not gpath.exists():
            v.add(c, f"{gpath.name} is missing")
            continue
        golden = load(gpath)
        if [g["name"] for g in golden] != names:
            v.add(
                c,
                f"{split}: golden sidecar names/order do not match the scenarios file",
            )


def check_world_grounding(ledger: dict, idx: dict[str, dict], v: Violations) -> None:
    c = "world"
    if not WORLD_PATH.exists():
        v.add(c, "worlds/retail.json is missing")
        return
    world = load(WORLD_PATH)
    state = world.get("initial_state", world)
    closed = ledger.get("field_policy") == "closed"
    for etype, ent in idx.items():
        key = _world_key(etype, state)
        if key is None:
            v.add(
                c,
                f"declared type {etype!r} has no {WORLD_KEY_FOR_TYPE.get(etype)!r} "
                f"or {etype!r} key in the exported world",
            )
            continue
        rows = state[key]
        rows_iter = rows.values() if isinstance(rows, dict) else rows
        declared = set(ent["fields"])
        for row in rows_iter:
            if not isinstance(row, dict):
                continue
            if closed:
                undeclared = set(row) - declared
                if undeclared:
                    v.add(
                        c,
                        f"{etype}: world row carries undeclared field(s) {sorted(undeclared)}",
                    )
                    break
            bad_enum = False
            for fname, f in ent["fields"].items():
                enum = f.get("enum")
                if enum is not None and fname in row and row[fname] is not None:
                    if row[fname] not in enum:
                        v.add(
                            c,
                            f"{etype}.{fname}: world value {row[fname]!r} is outside the declared enum",
                        )
                        bad_enum = True
            if bad_enum:
                break


# The platform accepts these product-facing spellings and stores the legacy
# one (``_LEGACY_EXECUTION_MODE_ALIASES`` in apps/api/app/schemas/agent.py).
# Normalizing across them is documented behaviour, not a rewrite of our intent.
_MODE_ALIASES = {"simulated": "sandbox", "executed": "code_intercepted"}

_PLATFORM_PROBE = r"""
import json, sys
from app.schemas.agent import (_validate_ledger_schema, _validate_tools_schema,
    validate_adapter_ontology_consistency, validate_ledger_read_consistency)
led = json.load(open(sys.argv[1])); tools = json.load(open(sys.argv[2]))
out = {"errors": [], "warnings": [], "rewrites": []}
try:
    nl = _validate_ledger_schema(led)
except Exception as exc:
    out["errors"].append("_validate_ledger_schema rejected the ontology: %s" % exc)
    print(json.dumps(out)); raise SystemExit(0)
if nl != led:
    out["rewrites"].append(["<ledger_schema>", "normalized", None, None])
try:
    nt = _validate_tools_schema(tools)
except Exception as exc:
    out["errors"].append("_validate_tools_schema rejected the tools schema: %s" % exc)
    print(json.dumps(out)); raise SystemExit(0)
for a, b in zip(tools, nt):
    for k in a:                       # only keys WE emitted; added defaults are fine
        if k in b and a[k] != b[k]:
            out["rewrites"].append([a["name"], k, a[k], b[k]])
for fn in (validate_adapter_ontology_consistency, validate_ledger_read_consistency):
    try:
        for w in fn(nl, nt) or []:
            out["warnings"].append("%s: %s" % (fn.__name__, w))
    except Exception as exc:
        out["errors"].append("%s raised: %s" % (fn.__name__, exc))

# --- scenarios ------------------------------------------------------------
# ScenarioBase types `conversation` as an opaque dict, so the multi-turn knobs
# it carries are NOT validated by the model. They ARE honoured by the run
# executor, so a bad value is a silent behaviour change; check them here
# against the platform's own vocabularies.
from app.constants import (TURN_MODE_VALUES, SIMULATOR_MODE_VALUES,
                           MEMORY_MODE_VALUES, EXPECTED_OUTCOME_VALUES)
from app.schemas.suite import ScenarioBase
VOCAB = {"turn_mode": TURN_MODE_VALUES, "simulator_mode": SIMULATOR_MODE_VALUES,
         "memory_mode": MEMORY_MODE_VALUES}
for path in sys.argv[3:]:
    split = path.rsplit("/", 1)[-1]
    for row in json.load(open(path)):
        nm = row.get("name")
        try:
            ScenarioBase(**row)
        except Exception as exc:
            out["errors"].append("%s/%s: ScenarioBase rejected it: %s" % (split, nm, exc))
            continue
        if row.get("expected_outcome") not in EXPECTED_OUTCOME_VALUES:
            out["errors"].append("%s/%s: expected_outcome=%r is outside %s"
                                 % (split, nm, row.get("expected_outcome"), sorted(EXPECTED_OUTCOME_VALUES)))
        conv = row.get("conversation") or {}
        for knob, allowed in VOCAB.items():
            if conv.get(knob) not in allowed:
                out["errors"].append("%s/%s: conversation.%s=%r is outside %s"
                                     % (split, nm, knob, conv.get(knob), sorted(allowed)))
        turns = conv.get("max_turns")
        if not isinstance(turns, int) or turns < 1:
            out["errors"].append("%s/%s: conversation.max_turns=%r" % (split, nm, turns))
        if conv.get("simulator_mode") == "persona" and not (conv.get("user_simulator_persona") or "").strip():
            out["errors"].append("%s/%s: persona mode with an empty persona (fails closed)" % (split, nm))
print(json.dumps(out))
"""


def _replay_read(read: dict, args: dict, world_rows: list[dict]) -> Any:
    """A minimal, independent interpreter for the subset of the read grammar
    retail uses (eq / ilike on declared scalars, project, order_by, limit, wrap,
    not_found). Deliberately re-implemented here rather than imported, so it can
    disagree with the exporter."""

    def resolve(val: Any) -> Any:
        if isinstance(val, dict) and "param" in val:
            return args.get(val["param"])
        if isinstance(val, dict) and "const" in val:
            return val["const"]
        return val

    rows = []
    for row in world_rows:
        keep = True
        for cond in read.get("where") or []:
            have, want = row.get(cond["field"]), resolve(cond.get("value"))
            cmp = cond.get("cmp", "eq")
            if cmp == "eq":
                keep = have == want
            elif cmp == "ilike":
                keep = (
                    isinstance(have, str)
                    and isinstance(want, str)
                    and have.lower() == want.lower()
                )
            else:
                raise NotImplementedError(f"cmp {cmp!r} not replayed")
            if not keep:
                break
        if keep:
            rows.append(row)

    for ob in reversed(read.get("order_by") or []):
        rows.sort(
            key=lambda r: r.get(ob["field"]) or "", reverse=ob.get("dir") == "desc"
        )

    def project(row: dict) -> dict:
        return {out: row.get(src) for out, src in (read.get("project") or {}).items()}

    if read.get("op") == "get":
        if not rows:
            return read.get("not_found") or None
        return project(rows[0])
    limit = read.get("limit")
    out = [project(r) for r in (rows[:limit] if isinstance(limit, int) else rows)]
    wrap = read.get("wrap")
    return {wrap: out} if wrap else out


def check_response_fidelity(tools: list[dict], v: Violations) -> None:
    """The check that would otherwise cost a live run: does what the MODEL sees
    match what tau2's own tool returns, byte for byte?

    A projection always yields an object; several tau2 tools return a bare
    string. The gap between the two is closed by ``dystopic_entry.render_retail``
    -- and nothing else verifies that the renderer and the projections agree.
    """
    c = "response_fidelity"
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from dystopic_entry import render_retail
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.tools import RetailTools

        real = RetailTools(
            RetailDB.load(str(REPO_ROOT / "data/tau2/domains/retail/db.json"))
        )
    except Exception as exc:  # pragma: no cover - env problem, not a defect
        v.add(c, f"could not set up the fidelity replay: {exc!r}")
        return

    state = load(WORLD_PATH)
    state = state.get("initial_state", state)
    rows_for = {}
    for etype in WORLD_KEY_FOR_TYPE:
        key = _world_key(etype, state)
        if key is not None:
            rows = state[key]
            rows_for[etype] = list(rows.values()) if isinstance(rows, dict) else rows
    user = rows_for["user"][0]
    order = rows_for["order"][0]
    product = rows_for["product"][0]
    item = rows_for["item"][0]

    cases = [
        ("get_user_details", {"user_id": user["user_id"]}),
        ("get_order_details", {"order_id": order["order_id"]}),
        ("get_product_details", {"product_id": product["product_id"]}),
        ("get_item_details", {"item_id": item["item_id"]}),
        ("find_user_id_by_email", {"email": user["email"]}),
        # case-mangled on purpose: `ilike` must behave like tau2's .lower() ==
        ("find_user_id_by_email", {"email": user["email"].upper()}),
        (
            "find_user_id_by_name_zip",
            {
                "first_name": user["first_name"],
                "last_name": user["last_name"],
                "zip": user["zip"],
            },
        ),
        (
            "find_user_id_by_name_zip",
            {
                "first_name": user["first_name"].upper(),
                "last_name": user["last_name"].lower(),
                "zip": user["zip"],
            },
        ),
        ("list_all_product_types", {}),
        # misses -> tau2 raises; we must render "Error: ..."
        ("get_order_details", {"order_id": "#WDOESNOTEXIST"}),
        ("find_user_id_by_email", {"email": "nobody@example.com"}),
    ]

    by_name = {t["name"]: t for t in tools}
    for name, args in cases:
        # This check runs AFTER the ones that diagnose a missing tool or an
        # undeclared entity type, so it must degrade rather than crash -- a
        # traceback here would swallow the violations already collected.
        entry = by_name.get(name)
        if entry is None:
            continue  # already reported by check_tool_names_vs_tau2
        read = entry.get("ledger_read")
        if read is None or read.get("entity_type") not in rows_for:
            continue  # already reported by check_ledger_reads
        try:
            payload = _replay_read(read, args, rows_for[read["entity_type"]])
        except Exception as exc:
            v.add(c, f"{name}: could not replay the projection: {exc}")
            continue
        ours, ours_err = render_retail(name, payload)
        if not isinstance(ours, str):
            ours = json.dumps(ours, default=str)

        try:
            theirs = getattr(real, name)(**args)
            theirs_err = False
        except Exception as exc:
            theirs, theirs_err = f"Error: {exc}", True
        if hasattr(theirs, "model_dump"):
            theirs = json.dumps(theirs.model_dump(), default=str)
        elif not isinstance(theirs, str):
            theirs = json.dumps(theirs, default=str)

        if ours_err != theirs_err:
            v.add(c, f"{name}{args}: is_error {ours_err} but tau2 says {theirs_err}")
        if (
            json.loads(ours) != json.loads(theirs)
            if _both_json(ours, theirs)
            else ours != theirs
        ):
            v.add(
                c,
                f"{name}{args}: the model would see {ours[:120]!r} "
                f"but tau2 returns {theirs[:120]!r}",
            )


def _both_json(a: str, b: str) -> bool:
    try:
        json.loads(a), json.loads(b)
        return True
    except Exception:
        return False


def check_platform(monorepo: Path, python: str | None, v: Violations) -> None:
    """Run the platform's *real* registration validators over the artifacts.

    This is the check that matters most: it is the same code path that returns
    a 422 at agent registration, so a pass here means the artifacts install.
    """
    import subprocess

    c = "platform"
    api = monorepo / "apps" / "api"
    if not api.exists():
        v.add(c, f"--platform {monorepo} has no apps/api")
        return
    interp = python or str(api / ".venv" / "bin" / "python")
    if not Path(interp).exists():
        interp = sys.executable
    scenario_files = [
        str(REPO_ROOT / "scenarios" / f"retail.{s}.json")
        for s in SCENARIO_SPLITS
        if (REPO_ROOT / "scenarios" / f"retail.{s}.json").exists()
    ]
    proc = subprocess.run(
        [
            interp,
            "-c",
            _PLATFORM_PROBE,
            str(LEDGER_PATH),
            str(TOOLS_PATH),
            *scenario_files,
        ],
        cwd=api,
        capture_output=True,
        text=True,
    )
    line = next(
        (ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None
    )
    if line is None:
        v.add(
            c,
            f"platform probe produced no result (rc={proc.returncode}): {proc.stderr.strip()[-300:]}",
        )
        return
    res = json.loads(line)
    for err in res["errors"]:
        v.add(c, err)
    for warn in res["warnings"]:
        v.add(c, warn)
    for name, key, ours, theirs in res["rewrites"]:
        if key == "default_execution_mode" and _MODE_ALIASES.get(ours) == theirs:
            continue  # documented alias, not a rewrite
        v.add(c, f"the platform rewrites {name}.{key}: {ours!r} -> {theirs!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--platform", type=Path, default=None, help="path to the pipelines monorepo"
    )
    ap.add_argument(
        "--platform-python",
        default=None,
        help="interpreter that can import apps/api (defaults to apps/api/.venv/bin/python)",
    )
    args = ap.parse_args()

    v = Violations()
    for path in (LEDGER_PATH, TOOLS_PATH):
        if not path.exists():
            v.add("artifacts", f"{path} is missing")
    if len(v):
        for check, msg in v.rows:
            print(f"FAIL [{check}] {msg}")
        return 1

    ledger = load(LEDGER_PATH)
    tools = load(TOOLS_PATH)
    idx = index_ledger(ledger)

    check_ledger_self(ledger, idx, v)
    tool_names = check_tool_names_vs_tau2(tools, v)
    check_ledger_reads(tools, idx, v)
    check_ledger_adapters(tools, idx, ledger, v)
    check_write_tools_are_guarded(tools, v)
    check_levels(v)
    check_scenarios(tool_names, v)
    check_world_grounding(ledger, idx, v)
    check_response_fidelity(tools, v)
    check_execution_mode_routing(tools, v)
    if args.platform:
        check_platform(args.platform, args.platform_python, v)

    if v.rows:
        for check, msg in v.rows:
            print(f"FAIL [{check}] {msg}")
        print(f"\n{len(v.rows)} violation(s)")
        return 1
    print(
        f"OK: {len(tools)} tools, {len(idx)} entity types, "
        f"{sum(len(e['fields']) for e in idx.values())} declared fields -- all cross-checks pass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
