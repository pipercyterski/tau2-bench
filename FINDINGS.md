# Platform findings from porting τ-bench

Running τ-bench through Dystopic is, incidentally, a hard test of the platform
itself: τ-bench is a stateful, multi-turn, tool-using benchmark with a
deterministic reward, authored by people who had never heard of us. Anything it
cannot express is something a customer with a stateful agent will also hit.

These are the gaps found while porting the retail domain. Ordered by how much
they'd cost a real customer.

---

## 1. There is no state/ledger scorer target

**What happened.** τ-bench's reward is deterministic: the product of a DB-hash
equivalence check against a gold replay and a substring match on required
utterances. It is exactly the kind of thing a scorer should express.

It cannot be expressed. `ScorerTarget` is `code | behavior` only. Every `code`
scorer type except `llm_judge` is `WORKSPACE_DEPENDENT` and is rejected at
seeding on a no-workspace simulated run. So on precisely the runs where the
platform owns the world state, there is **no way to assert anything about that
state deterministically**. The only verdict mechanisms available are an LLM
judge and LLM-judged behaviour constraints.

**Why it matters beyond us.** Any customer whose agent's correctness is "the
database ended up in the right state" — order systems, booking, ticketing,
provisioning, anything transactional — has the same problem. They can watch the
ledger in a trace, and they can ask an LLM whether it looks right, but they
cannot write `assert order.status == "cancelled"` and gate on it. That is the
single most natural assertion for a stateful agent, and it is the one shape the
scorer library cannot hold.

**Shape of the ask.** A `state` scorer target that runs against the final
ledger, with the deterministic-predicate flavour the `code` types already have,
but without the workspace dependency.

---

## 2. A nested map's member cannot be declaratively updated

**What happened.** Four retail write tools adjust a gift card's balance, which
lives at `user.payment_methods[<id>].balance`. A `ledger_adapter`'s `field_map`
addresses declared fields on an entity; it cannot address a member of a map held
*inside* one, and it cannot compute a replacement map. So those mutations fall
back to the world simulator, and the user row can silently diverge from what
τ-bench's own reward would compute.

The escape hatch that worked elsewhere does not work here. `products[].variants{}`
had the same problem and was solved by **promoting** it to a first-class `item`
entity. That fix is unavailable for payment methods, because `get_user_details`
must return the nested `payment_methods` map to keep response fidelity, and
`nest` assembles one sub-object per join — it cannot reconstruct a map of N.

**Scope here.** 234 of 695 retail payment methods carry a balance; 5 of the 40
test-split tasks touch one.

**Why it matters.** "Promote the nested thing to an entity" is the platform's
implicit answer to nesting, and it is a good answer — but it only works when no
tool has to return the nested shape back. Whenever a customer's API returns a
map of sub-objects *and* mutates its members, they are stuck between response
fidelity and declarative writes.

---

## 3. The CLI reports a successful dispatch as a failure

**What happened.** `odyssey test` printed:

```
Test run created, but its agent runs were not queued:
  - row 1: Odyssey seed column 'state' initial_state key 'items' normalized ... (advisory)
No runs will progress. Resolve the dispatch failure, then re-run
```

The run was dispatched and executing at that moment, and completed normally.
Every line listed is an **advisory**, not an error. The CLI appears to treat a
non-empty preflight-advisory list as a dispatch failure.

**Why it matters.** It tells the operator to stop and debug a problem that does
not exist, and it does so on the *first* run of any suite whose world uses plural
seed keys — i.e. exactly when a new user is least able to tell the difference.

---

## 4. Connected-repo agents still need hand-written `requirements`

**What happened.** The agent clones a repo that has a `pyproject.toml` declaring
its 19 dependencies, but `config.requirements` is authored separately by hand. A
hand-picked list missed `addict` (a transitive import), and the failure surfaced
only at run time, 129 seconds in, as `ModuleNotFoundError` inside the entrypoint.

Worse than the missing package: the hand-written list pinned `litellm>=1.60`
while the repo pins `litellm>=1.80.15,<1.82.7`. That would have resolved to a
version the code under test does not support — a *silent* version skew rather
than a loud import error.

**Shape of the ask.** For a connected repo, default `requirements` to the repo's
own declared dependencies, or offer `requirements: "from-pyproject"`. The repo
is already the source of truth for the code; it should be the source of truth
for the code's dependencies.

---

## 5. In-app checks clone the default branch, with no way to target a ref

**What happened.** A check on a connected-repo agent clones the repo's default
branch. With the port on a side branch, every run failed
`agent_code_fetch_failed: entrypoint_file not found`. The error names the missing
file but not the ref it looked on, which is the fact that would have explained it.

Working on a branch before merging is the normal state of affairs while porting.
The workaround — merge to the default branch — means the default branch has to
carry work-in-progress for it to be testable in-app at all.

**Shape of the ask.** A `--ref` on `odyssey test`, and the resolved ref in the
`agent_code_fetch_failed` message.

---

## 6. Seeding advisories fire on the natural key spelling

`initial_state` keyed `orders` against a declared `order` normalizes correctly
and emits an advisory preferring the singular. That is a fine default, but the
natural spelling of a seed dump — including τ-bench's, and most real exports —
is plural, so the advisory fires on the common case. It is also the *opposite*
of the failure mode a previous generation of this port hit, where the mismatch
silently zero-scored; the canonicalizer is a genuine fix and the advisory is
cheap. Noted only because four advisories on every check read as alarming.

---

## Not a finding: what worked

Worth recording, because the port's core bet paid off. Of retail's 16 tools,
**14 are fully declarative** — 7 deterministic `ledger_read` projections and 7
`ledger_adapter` writes — leaving the world-simulator LLM essentially out of the
loop. The whole tool surface, the 4-type closed ontology and a 2141-entity world
registered without a single validator warning. Declared reads also eliminate, by
construction, the context-window blowup that made large worlds unusable before:
a projection never enters the LLM's prompt, so world size stops being a limit on
what can be simulated faithfully.
