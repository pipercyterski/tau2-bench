"""tau2 domain DB -> Dystopic World variant ``initial_state``.

    python -m dystopic_export.export_world --domain retail --out worlds/retail.json

Retail ships as three top-level maps (``products`` 50, ``users`` 500,
``orders`` 1000). We emit those verbatim -- the shape is already the canonical
``{type: {id: {fields}}}`` the ledger bootstraps from -- plus two additions
justified in ``contract.py``:

* a promoted ``items`` map lifted out of ``products[].variants``
* flattened filter keys on each user (``first_name`` / ``last_name`` / ``zip``)

Airline is deliberately not implemented yet. Its ``flights[].dates{}`` nests
per-date availability, and ``search_direct_flight(origin, destination, date)``
needs to project one date's sub-object -- which a ``project`` (which selects
declared fields, not paths) cannot express. The options are to promote
flight x date into ~9000 ``flight_instance`` entities, or to let the search
tools fall to the simulator. That is a real design decision, not a mechanical
port, and it should be made after retail has proven the loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_tau2_db(domain: str) -> dict[str, Any]:
    """Load a domain's initial DB through tau2's own pydantic loader.

    Going through the domain's own ``DB.load`` rather than reading db.json
    directly means we inherit tau2's validation and field defaults, so the
    world we seed is the world tau2 itself would have started from.
    """
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.utils import RETAIL_DB_PATH

        db = RetailDB.load(RETAIL_DB_PATH)
    elif domain == "airline":
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.airline.utils import AIRLINE_DB_PATH

        db = FlightDB.load(AIRLINE_DB_PATH)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unsupported domain {domain!r}")

    return copy.deepcopy(db.model_dump(mode="json"))


def _promote_items(products: dict[str, Any]) -> dict[str, Any]:
    """Lift ``products[].variants{}`` into a top-level ``items`` map.

    Each variant already carries its own ``item_id``; we add the ``product_id``
    backref and the denormalized ``product_name`` so an item can be found
    without a join (``nest`` is one level and projection-only, so a filter on
    the parent's name would otherwise be inexpressible).
    """
    items: dict[str, Any] = {}
    for product_id, product in products.items():
        for item_id, variant in (product.get("variants") or {}).items():
            row = dict(variant)
            row["item_id"] = item_id
            row["product_id"] = product_id
            row["product_name"] = product.get("name")
            items[item_id] = row
    return items


def _flatten_user_filter_keys(users: dict[str, Any]) -> None:
    """Add scalar filter keys alongside (never replacing) the nested originals.

    ``find_user_id_by_name_zip`` filters on first name, last name and zip. Those
    live at ``name.first_name`` / ``name.last_name`` / ``address.zip``, and a
    ``where.field`` names a declared field rather than a path -- so we surface
    them as scalars. Nothing projects them, so no tool response changes.
    """
    for user in users.values():
        name = user.get("name") or {}
        address = user.get("address") or {}
        user["first_name"] = name.get("first_name")
        user["last_name"] = name.get("last_name")
        user["zip"] = address.get("zip")


def build_retail_world(db: dict[str, Any]) -> dict[str, Any]:
    products = copy.deepcopy(db["products"])
    users = copy.deepcopy(db["users"])
    orders = copy.deepcopy(db["orders"])

    items = _promote_items(products)
    _flatten_user_filter_keys(users)

    # Keyed by the DECLARED SINGULAR entity type, not tau2's plural map names.
    # The ledger treats the initial_state key as authoritative and will fold a
    # plural key onto a declared singular type -- but it emits a seeding
    # advisory when it has to ("prefer the declared singular form"), and a
    # showcase build should not ship four advisories on every check. Keying
    # singular makes seed key and declared type identical, so no normalization
    # happens at all.
    return {
        "product": products,
        "item": items,
        "user": users,
        "order": orders,
    }


BUILDERS = {"retail": build_retail_world}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    db = _load_tau2_db(args.domain)
    world = BUILDERS[args.domain](db)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(world, indent=2, sort_keys=True))

    total = sum(len(v) for v in world.values())
    size_mb = args.out.stat().st_size / 1_000_000
    print(f"wrote {args.out} ({size_mb:.1f} MB, {total} entities)")
    for key, rows in world.items():
        print(f"  {key}: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
