"""Push exported scenarios onto a platform agent, and bind them to a suite.

    python -m dystopic_export.upload --agent 101 --split test \
        --suite retail-test --world-id 13

Scenarios go over REST rather than the CLI because the interesting fields --
``conversation`` (the multi-turn/persona block), ``scorers`` (the inline
behaviour bindings derived from tau2's nl_assertions), ``expected_outcome_detail``
and ``xfail`` -- are JSON-shaped and the CLI's ``scenarios create`` only exposes
the scalar ones.

Idempotent by scenario ``name``: a re-run PATCHes the scenario that already
carries that name rather than minting a duplicate, so re-exporting after a
tweak does not leave the agent littered with orphans. Names are namespaced
(``retail/12``) and stable, which is also what lets a base-vs-head review line
the two sides up case by case.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fields the platform accepts on a scenario create/update. Anything else in the
# exported payload is ours (bookkeeping) and must not be sent.
SCENARIO_FIELDS = (
    "name",
    "scenario_group",
    "user_instruction",
    "behavior_instructions",
    "expected_outcome",
    "expected_outcome_detail",
    "expected_tool_sequence",
    "conversation",
    "scorers",
    "xfail",
    "world_id",
)


def _request(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    *,
    retries: int = 4,
) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {api_key}")
        if body:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:600]
            # 429 carries Retry-After; 5xx is worth one more try. A 4xx is ours
            # to fix, so surface it immediately rather than burning retries.
            if exc.code == 429 or exc.code >= 500:
                if attempt < retries - 1:
                    wait = float(exc.headers.get("Retry-After") or 2**attempt)
                    time.sleep(wait)
                    continue
            raise SystemExit(f"{method} {url} -> {exc.code}\n{detail}") from exc
    raise SystemExit(f"{method} {url}: exhausted retries")


def existing_by_name(base: str, agent: int, api_key: str) -> dict[str, int]:
    """Map scenario name -> id for everything already on the agent."""
    found: dict[str, int] = {}
    page = 1
    while True:
        got = _request(
            "GET", f"{base}/agents/{agent}/scenarios?page={page}&page_size=100", api_key
        )
        rows = got.get("items", got) if isinstance(got, dict) else got
        if not rows:
            break
        for row in rows:
            found[row["name"]] = row["id"]
        if len(rows) < 100:
            break
        page += 1
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True, type=int)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--split", default="test")
    ap.add_argument("--suite", help="suite name to create (or reuse) and bind to")
    ap.add_argument("--world-id", type=int)
    ap.add_argument("--limit", type=int, help="upload only the first N (smoke runs)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    api_key = os.environ.get("DYSTOPIC_API_KEY")
    if not api_key:
        raise SystemExit("DYSTOPIC_API_KEY is not set")
    base = os.environ.get("DYSTOPIC_BASE", "https://api-staging.pipelines.tech/api")

    path = REPO_ROOT / "scenarios" / f"{args.domain}.{args.split}.json"
    scenarios = json.loads(path.read_text())
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"{path.name}: {len(scenarios)} scenarios")

    if args.dry_run:
        print(json.dumps(scenarios[0], indent=2)[:2000])
        return 0

    known = existing_by_name(base, args.agent, api_key)
    ids: list[int] = []
    created = updated = 0

    for scenario in scenarios:
        payload = {k: scenario[k] for k in SCENARIO_FIELDS if k in scenario}
        if args.world_id:
            payload["world_id"] = args.world_id
        name = payload["name"]
        if name in known:
            sid = known[name]
            _request(
                "PATCH", f"{base}/agents/{args.agent}/scenarios/{sid}", api_key, payload
            )
            updated += 1
        else:
            got = _request(
                "POST", f"{base}/agents/{args.agent}/scenarios", api_key, payload
            )
            sid = got["id"]
            created += 1
        ids.append(sid)
        print(f"  {name} -> #{sid}", flush=True)

    print(f"{created} created, {updated} updated")

    if args.suite:
        suites = _request("GET", f"{base}/agents/{args.agent}/suites", api_key)
        rows = suites.get("items", suites) if isinstance(suites, dict) else suites
        match = next((s for s in rows if s["name"] == args.suite), None)
        if match:
            suite_id = match["id"]
            print(f"reusing suite #{suite_id} {args.suite!r}")
        else:
            body: dict[str, Any] = {"name": args.suite}
            if args.world_id:
                body["world_id"] = args.world_id
            suite_id = _request(
                "POST", f"{base}/agents/{args.agent}/suites", api_key, body
            )["id"]
            print(f"created suite #{suite_id} {args.suite!r}")

        # Binding is an ordered replace-all: ids omitted here are UNBOUND.
        _request(
            "PUT",
            f"{base}/agents/{args.agent}/suites/{suite_id}/scenarios",
            api_key,
            {"scenario_ids": ids},
        )
        print(f"bound {len(ids)} scenarios to suite #{suite_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
