"""tau2 pydantic models -> Dystopic ``ledger_schema`` ontology.

    python -m dystopic_export.export_ontology --domain retail --out schemas/retail.ledger.json

tau2's ``src/tau2/domains/retail/data_model.py`` already *is* a closed-world
ontology: ``Order`` declares every post-action field (``cancel_reason``,
``return_items``, ``return_payment_method_id``, ``exchange_items``,
``exchange_new_items``, ``exchange_payment_method_id``,
``exchange_price_difference``) as an optional, because tau2's own write tools
have to populate them. So the field set is *derived* from the models rather
than discovered by replaying golden actions -- the earlier attempt's replay was
compensating for a missing schema, not for a missing field list.

What the models cannot tell us is the two things we added in ``export_world``,
so ``contract.RETAIL_EXTRA_FIELDS`` is unioned on top: the promoted ``item``
type's ``product_id`` / ``product_name`` backrefs, and the user's flattened
``first_name`` / ``last_name`` / ``zip`` filter keys. Even those get their
*types* from a tau2 model (``Product.product_id``, ``UserName.first_name``, ...)
rather than a hand-written guess -- see ``EXTRA_FIELD_SOURCES``.

Two deliberate choices:

* **Enums only where genuinely categorical.** ``order.status`` and
  ``order.cancel_reason``, and nothing else. Enum members are interpolated
  verbatim into the simulator's world-state prompt and matched verbatim by the
  strict-mode consistency check, so an enum on an id field would be a closed
  set of 1000 order ids in the prompt and a regeneration every time the world
  grew one. The status enum is the union of tau2's ``OrderStatus`` literal, the
  contract list, and the statuses actually present in the exported world, so
  post-action-only values like ``pending (item modified)`` are covered even
  though no seeded order starts there.

* **``field_policy: "closed"``.** The field set is complete by construction
  (it came from the models that tau2 itself validates against), so a write
  outside it is a genuine fidelity bug worth surfacing rather than absorbing.
  Closed also turns an adapter ``field_map`` onto an undeclared field into a
  registration-time ``422`` instead of a silently dropped op at runtime. If
  strict-mode regeneration churn ever shows up in a run, this is the one knob
  to flip back to ``"open"``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dystopic_export import contract

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The platform's validators (app/schemas/agent.py::_validate_ledger_schema).
ENTITY_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FIELD_TYPES = {"string", "number", "integer", "boolean", "object", "array"}

# Declared type -> the tau2 model whose fields define it. ``item`` is the
# promoted ``products[].variants`` value, whose model is ``Variant``.
RETAIL_BASE_MODELS: dict[str, str] = {
    "product": "Product",
    "item": "Variant",
    "user": "User",
    "order": "Order",
}

# The fields in ``contract.RETAIL_EXTRA_FIELDS`` that the base model does NOT
# have, mapped to the tau2 model field they were copied from -- so even our
# additions take their declared type from tau2 rather than from a guess.
EXTRA_FIELD_SOURCES: dict[tuple[str, str], tuple[str, str]] = {
    ("item", "product_id"): ("Product", "product_id"),
    ("item", "product_name"): ("Product", "name"),
    ("user", "first_name"): ("UserName", "first_name"),
    ("user", "last_name"): ("UserName", "last_name"),
    ("user", "zip"): ("UserAddress", "zip"),
}

# Descriptions for the added fields: their tau2 description describes the field
# in its original home, not the role it plays here.
EXTRA_FIELD_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("item", "product_id"): "ID of the product this item is a variant of.",
    ("item", "product_name"): (
        "Name of the parent product, denormalized onto the item so it can be "
        "filtered without a join. Filter key only -- never projected."
    ),
    ("user", "first_name"): (
        "Flattened copy of name.first_name so a ledger_read `where` can filter "
        "on it. Filter key only -- never projected; the nested name{} object is "
        "what tools return."
    ),
    ("user", "last_name"): (
        "Flattened copy of name.last_name so a ledger_read `where` can filter "
        "on it. Filter key only -- never projected; the nested name{} object is "
        "what tools return."
    ),
    ("user", "zip"): (
        "Flattened copy of address.zip so a ledger_read `where` can filter on "
        "it. Filter key only -- never projected; the nested address{} object is "
        "what tools return."
    ),
}

ENTITY_DESCRIPTIONS: dict[str, str] = {
    "product": (
        "A retail product line. Its variants{} map holds the purchasable items, "
        "each of which is also a first-class `item` entity."
    ),
    "item": (
        "One purchasable variant of a product (a specific colour/size/...), "
        "promoted out of products[].variants so it can be looked up directly."
    ),
    "user": "A customer, with their address, payment methods and order history.",
    "order": (
        "A customer order and its lifecycle: items, fulfillments, payment "
        "history, and the cancel/return/exchange fields written by the "
        "post-purchase tools."
    ),
}

# Where an enum is genuinely categorical. Never an *_id field.
RETAIL_ENUM_FIELDS: dict[tuple[str, str], list[str]] = {
    ("order", "status"): contract.RETAIL_ORDER_STATUS,
    ("order", "cancel_reason"): contract.RETAIL_CANCEL_REASON,
}


def _retail_models() -> dict[str, type[BaseModel]]:
    from tau2.domains.retail import data_model

    return {
        name: getattr(data_model, name)
        for name in ("Product", "Variant", "User", "UserName", "UserAddress", "Order")
    }


def _properties(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """The model's own top-level JSON-schema properties, in declaration order."""
    return model.model_json_schema(mode="serialization")["properties"]


_JSON_TO_LEDGER = {
    "string": "string",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
    "null": "string",
}


def _ledger_type(schema: dict[str, Any]) -> str:
    """Collapse a pydantic JSON-schema property onto one of the six ledger types.

    ``Optional[X]`` arrives as an ``anyOf`` with a ``null`` branch (we take the
    real branch -- the ledger has no nullable type, and an unset optional is
    simply an absent field), ``Dict[...]`` and a nested model both arrive as an
    object, and a ``Literal`` arrives as a bare ``enum``.
    """
    if "anyOf" in schema:
        branches = [b for b in schema["anyOf"] if b.get("type") != "null"]
        if len(branches) != 1:
            raise ValueError(f"cannot collapse a union of {len(branches)} branches: {schema}")
        return _ledger_type(branches[0])
    if "$ref" in schema or "allOf" in schema or "additionalProperties" in schema:
        return "object"
    if "type" in schema:
        return _JSON_TO_LEDGER[schema["type"]]
    values = schema.get("enum") or ([schema["const"]] if "const" in schema else None)
    if values:
        sample = values[0]
        if isinstance(sample, bool):
            return "boolean"
        if isinstance(sample, int):
            return "integer"
        if isinstance(sample, float):
            return "number"
        return "string"
    raise ValueError(f"untyped property schema: {schema}")


def _observed_statuses(world: dict[str, Any]) -> list[str]:
    return sorted({order["status"] for order in world["orders"].values() if order.get("status")})


def build_retail_ontology(world: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Return ``(ledger_schema, enums_declared)`` for the retail domain."""
    models = _retail_models()
    props = {name: _properties(model) for name, model in models.items()}

    # Union the enum with what the seeded world actually contains, so a status
    # the data uses but the contract forgot can never make a seeded order look
    # like a violation.
    status_enum = list(RETAIL_ENUM_FIELDS[("order", "status")])
    for status in _observed_statuses(world):
        if status not in status_enum:
            status_enum.append(status)
    enum_by_field = dict(RETAIL_ENUM_FIELDS)
    enum_by_field[("order", "status")] = status_enum

    entities: list[dict[str, Any]] = []
    declared: dict[str, list[str]] = {}
    enums_declared: dict[str, list[str]] = {}

    for state_key, etype in contract.RETAIL_TYPES.items():
        base = props[RETAIL_BASE_MODELS[etype]]
        names = list(base)
        for extra in contract.RETAIL_EXTRA_FIELDS.get(etype, []):
            if extra not in names:
                names.append(extra)

        fields: list[dict[str, Any]] = []
        for name in names:
            if name in base:
                source = base[name]
                description = source.get("description")
            else:
                src_model, src_field = EXTRA_FIELD_SOURCES[(etype, name)]
                source = props[src_model][src_field]
                description = EXTRA_FIELD_DESCRIPTIONS[(etype, name)]

            field: dict[str, Any] = {"name": name, "type": _ledger_type(source)}
            values = enum_by_field.get((etype, name))
            if values:
                field["enum"] = list(values)
                enums_declared[f"{etype}.{name}"] = list(values)
            if description:
                field["description"] = description
            fields.append(field)

        entities.append(
            {
                "type": etype,
                "id_field": contract.RETAIL_ID_FIELDS[etype],
                "description": ENTITY_DESCRIPTIONS[etype],
                "fields": fields,
            }
        )
        declared[etype] = names
        assert state_key in world  # checked properly in validate()

    schema = {"entities": entities, "field_policy": "closed"}
    return schema, enums_declared


def declared_fields(schema: dict[str, Any]) -> dict[str, list[str]]:
    """``{entity_type: [field names]}`` -- what ``contract.projected_fields`` takes.

    The tools exporter needs the same field lists this module derived, and a
    ledger_read whose ``project`` or ``where`` names a field outside them is a
    422, so it reads them back off the emitted ontology rather than re-deriving.
    """
    return {e["type"]: [f["name"] for f in e["fields"]] for e in schema["entities"]}


def validate(schema: dict[str, Any], world: dict[str, Any], types: dict[str, str]) -> None:
    """Fail loudly here rather than with a 422 at registration time."""
    state_key_by_type = {etype: key for key, etype in types.items()}

    for entity in schema["entities"]:
        etype = entity["type"]
        if not ENTITY_TYPE_RE.match(etype):
            raise ValueError(f"entity type {etype!r} is not canonical singular snake_case")

        names = [f["name"] for f in entity["fields"]]
        if len(names) != len(set(names)):
            raise ValueError(f"{etype}: duplicate field names")
        for field in entity["fields"]:
            if not FIELD_NAME_RE.match(field["name"]):
                raise ValueError(f"{etype}.{field['name']} is not a legal identifier")
            if field["type"] not in FIELD_TYPES:
                raise ValueError(f"{etype}.{field['name']}: bad type {field['type']!r}")
            if "enum" in field and field["name"].endswith("_id"):
                raise ValueError(f"{etype}.{field['name']}: an id field must never carry an enum")

        if entity["id_field"] not in names:
            raise ValueError(f"{etype}: id_field {entity['id_field']!r} is not a declared field")

        state_key = state_key_by_type[etype]
        if state_key not in world:
            raise ValueError(f"{etype}: the world has no {state_key!r} map")

        # Closed field policy cuts both ways: a field the world carries but the
        # ontology never declares would be an undeclared write on every run.
        seen: set[str] = set()
        for row in world[state_key].values():
            seen.update(row)
        undeclared = seen - set(names)
        if undeclared:
            raise ValueError(f"{etype}: world rows carry undeclared fields {sorted(undeclared)}")

        # An enum the seeded world already violates would be a permanent
        # strict-mode finding on every scenario.
        for field in entity["fields"]:
            if "enum" not in field:
                continue
            allowed = set(field["enum"])
            bad = {
                row[field["name"]]
                for row in world[state_key].values()
                if row.get(field["name"]) is not None and row[field["name"]] not in allowed
            }
            if bad:
                raise ValueError(f"{etype}.{field['name']}: world uses {sorted(bad)} outside enum")


BUILDERS = {"retail": build_retail_ontology}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--world",
        type=Path,
        help="exported world JSON; enums are widened to cover it and every "
        "declared type is checked against it (default worlds/<domain>.json)",
    )
    args = ap.parse_args(argv)

    world_path = args.world or REPO_ROOT / "worlds" / f"{args.domain}.json"
    world = json.loads(world_path.read_text())

    schema, enums = BUILDERS[args.domain](world)
    validate(schema, world, contract.RETAIL_TYPES)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, indent=2) + "\n")

    print(f"wrote {args.out} (grounded against {world_path})")
    print(f"  field_policy: {schema['field_policy']}")
    for entity in schema["entities"]:
        print(
            f"  {entity['type']}: {len(entity['fields'])} fields, "
            f"id_field={entity['id_field']}"
        )
    for field, values in enums.items():
        print(f"  enum {field}: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
