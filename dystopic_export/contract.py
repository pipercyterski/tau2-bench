"""The naming contract every exporter keys off.

Two rules decide every name in here, and both come from the platform:

1. **The world's ``initial_state`` top-level key is authoritative.** The ledger
   is physically keyed by it (``services/odyssey/ledger.py::_bootstrap_from_seed``),
   and a read or write arriving under a singular/plural/case variant of a known
   type canonicalizes onto that stored key. So we ship tau2's DB keyed exactly
   as it ships -- ``products`` / ``users`` / ``orders`` -- and declare the
   singular types the ontology requires. No transform, no drift.

2. **Declared entity types are canonical singular snake_case** (``^[a-z][a-z0-9_]*$``).

Where tau2's DB nests, we *promote* rather than flatten-in-place:

* ``products[].variants{item_id: {...}}`` becomes a first-class ``item`` entity
  carrying a ``product_id`` backref. Without this, ``get_item_details`` has no
  entity to project from and would have to fall to the world-simulator LLM.

* ``users[].name{first,last}`` and ``users[].address{...,zip}`` stay nested in
  the stored entity (so a projection returns the byte-identical tau2 payload),
  but we *additionally* store flattened ``first_name`` / ``last_name`` / ``zip``
  fields, because ``ledger_read.where.field`` must name a declared scalar field
  and cannot walk a path. The flattened fields are filter keys only -- no tool
  projects them, so no tool response changes shape because of them.

That asymmetry is the whole design: the world is modelled for *querying*, the
projections are shaped for *fidelity*.
"""

from __future__ import annotations

# --- retail ----------------------------------------------------------------

# initial_state key -> declared singular ledger type
RETAIL_TYPES: dict[str, str] = {
    "products": "product",
    "items": "item",
    "users": "user",
    "orders": "order",
}

# Declared field sets are DERIVED from tau2's own pydantic models rather than
# hand-listed -- see ``export_ontology.py``. tau2's ``Order`` model already
# declares every post-action field (``cancel_reason``, ``return_items``,
# ``return_payment_method_id``, ``exchange_items``, ``exchange_new_items``,
# ``exchange_payment_method_id``, ``exchange_price_difference``) as optionals,
# so the model *is* the closed-world ontology. That is why this port does not
# need the golden-action replay the previous attempt used to discover
# post-action fields: the fields were declared upstream all along, and the
# earlier gap was a missing schema, not a missing replay.
#
# Only the fields tau2's models do NOT have -- because we added them -- are
# listed here. The exporter unions these onto the derived set.
RETAIL_EXTRA_FIELDS: dict[str, list[str]] = {
    # promoted out of products[].variants
    "item": ["item_id", "product_id", "product_name", "options", "available", "price"],
    # flattened filter keys (never projected)
    "user": ["first_name", "last_name", "zip"],
}

RETAIL_ID_FIELDS: dict[str, str] = {
    "product": "product_id",
    "item": "item_id",
    "user": "user_id",
    "order": "order_id",
}

# Fields that exist purely so `where` can filter on them. Excluded from every
# projection so tool responses keep tau2's exact shape.
RETAIL_FILTER_ONLY_FIELDS: dict[str, set[str]] = {
    "user": {"first_name", "last_name", "zip"},
    "item": {"product_name"},
}

# The projection field list for a "return the whole entity" read -- the declared
# fields minus the filter-only ones. This is what makes `get_user_details`
# return exactly what tau2's own tool returns.
def projected_fields(entity_type: str, declared: dict[str, list[str]]) -> list[str]:
    skip = RETAIL_FILTER_ONLY_FIELDS.get(entity_type, set())
    return [f for f in declared[entity_type] if f not in skip]


# Status values tau2's retail order lifecycle actually uses. Declared as an enum
# so the ontology is closed where it is genuinely categorical -- and nowhere
# else. Never put an enum on an id field.
RETAIL_ORDER_STATUS = [
    "pending",
    "processed",
    "delivered",
    "cancelled",
    "pending (item modified)",
    "return requested",
    "exchange requested",
]

RETAIL_CANCEL_REASON = ["no longer needed", "ordered by mistake"]


# --- airline ---------------------------------------------------------------
# Airline nests per-date availability under ``flights[].dates{}``, which a
# projection cannot slice. Deferred until retail proves the loop end to end;
# see the notes in export_world.py.

AIRLINE_TYPES: dict[str, str] = {
    "flights": "flight",
    "users": "user",
    "reservations": "reservation",
}
