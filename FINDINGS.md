# Platform findings from porting τ-bench

Running τ-bench through Dystopic is, incidentally, a hard test of the platform
itself: τ-bench is a stateful, multi-turn, tool-using benchmark with a
deterministic reward, authored by people who had never heard of us. Anything it
cannot express is something a customer with a stateful agent will also hit.

These are the gaps found while porting the retail domain. Ordered by how much
they'd cost a real customer. Everything here is platform-facing — our own port
mistakes are not findings, and where a problem was partly ours (§6) the split is
stated rather than blurred.

| # | Finding | Kind | Hurts |
|---|---|---|---|
| 1 | No state/ledger scorer target | product gap | any stateful agent |
| 2 | Nested-map members can't be declaratively updated | product gap | APIs returning maps of sub-objects |
| 3 | CLI reports a successful dispatch as a failure | **bug** | every first-time suite run |
| 4 | Connected repos still need hand-written `requirements` | papercut | every code-agent port |
| 5 | In-app checks clone the default branch, no `--ref` | product gap | every port, before merge |
| 6 | A refused write is indistinguishable from a success | product gap | any validated write path |
| 7 | Only the final turn's tool-call bodies are retrievable | observability | every multi-turn triage |
| 8 | `behavior_instructions` names the wrong consumer | **naming / silent** | multi-turn ports |
| 9 | `tools.mdx` documents 1 `when` operator, validator accepts 4 | **docs bug** | anyone guarding a write |
| 10 | An API key can create a model but never fix it | **bug** | key-driven setup |
| 11 | Check page prints raw model ids, not names | **bug** | reading any result |
| 12 | Seeding advisories fire on the natural key spelling | papercut | cosmetic |

The two worth acting on first are different in character. **§1** is the one that
blocks selling this evaluation shape to anyone else — a customer whose
correctness *is* the end state cannot assert on it. **§8** is the one most
likely to be silently mis-set by the next person, and its only symptom is an
inflated pass rate, which is the worst failure an eval platform can have.

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

## 6. A refused write is indistinguishable from a successful one

**What happened.** An agent called

```
return_delivered_order_items(order_id="#W6390527",
                             item_ids=["1003829102"],
                             payment_method_id="paypal_0000001")
```

Neither id exists anywhere in the world (the real ones are `8538875209` and
`paypal_7644869`). Real τ-bench rejects this twice over — `_get_payment_method`
raises on an unknown method, and the item-existence loop raises "Some item not
found" — and renders the raise as the string `Error: …`, which the agent sees
and can correct from.

**What our world did, precisely.** The simulator returned the order **echoed
back unchanged**: `status: "delivered"` (τ-bench would have set `"return
requested"`), `return_items: null`, `return_payment_method_id: null`. The
adapter's `when: {field: "status", eq: "return requested"}` therefore evaluated
false and **skipped the effect entirely — zero ops**. The final ledger confirms
it: 2141 entries, **0 mutations**, order `#W6390527` still at
`updated_at_call: -1`.

So the write gating worked exactly as designed. Nothing bad was written. The
platform behaved correctly at every step we can inspect.

**The gap is the response channel, and it is narrow but consequential.** The
tool's reply carried no success/failure signal at all — it was a well-formed
`Order` object that happens to be unmutated. The agent read it as success and
told the user the return had been processed. A refused write and a successful
write are the same shape, and only a field-by-field comparison against
pre-call state distinguishes them.

Our `output_schema` is what closed off the alternative: a full nested Order with
large `required` arrays and no error variant leaves the simulator no legal way
to emit `Error: …`. The corroborating trace detail is that
`schema_compliance` logged *"First response was malformed; recovered after 1
retry"* — consistent with the simulator attempting to signal the failure, being
rejected by the schema, and retrying into a compliant success-shaped object.
(Inference: the malformed body is not retained.)

**Two asks, in priority order.**

1. **An error channel for simulated tools that survives `output_schema`.** A
   declared error variant, or an explicit "this call failed: `<reason>`" return
   the engine renders in the tool's own error idiom. Without one, a strict
   output schema silently converts every domain refusal into a fake success.
2. **Declarative preconditions on an adapter** — a `require` clause using the
   `ledger_read` `where` vocabulary (`{"entity_type": "order", "where": [...]}`)
   that produces a configurable error when unmet. `when` resolves only against
   the tool's own response or arguments; it cannot query ledger state, so
   "the item must exist in this order" is not currently expressible and
   validation rests on the simulator noticing. The grammar to express it already
   exists on the read side.

---

## 7. A run's detail only exposes the final turn's tool calls

Diagnosing the above needed the *arguments and responses* of 12 `ledger_read`
calls spread across a multi-turn conversation. The counts are aggregated across
turns (`tool_calls: {total: 13, ledger_read: 12}`), but `GET /agents/{id}/runs/
{run_id}` and `…/runs/{run_id}/ledger` both return only the **last** turn's call
bodies — one entry, and in the ledger view with `arguments: null` and
`response: None`. The per-turn session endpoint is namespaced under
`/projects/{id}/workflows/{id}/tasks/{id}/agent-sessions`, which a suite-based
check does not have ids for.

So on a multi-turn run there is no reachable answer to "what did that read
actually return on turn 3" — the exact question every simulator-fidelity triage
starts from. We settled the case by reasoning from the transcript and the world
file instead, which worked here only because the fabricated ids were absent from
the world entirely.

---

## 8. `behavior_instructions` names the wrong thing, and the failure is silent

**What happened.** Porting τ-bench's user simulator, the obvious home for "what
the simulated counterpart knows and how it must behave" is
`behavior_instructions`. It is the wrong field. Its documented meaning is
"free-text guidance fed into the Odyssey simulator" — the **world** simulator,
the component that invents tool responses. The simulated *user* is driven from
`conversation.user_simulator_persona`.

The neighbouring axis has a symmetric trap. `user_instruction` reads like "the
instruction describing the user's task", and τ-bench has a field of exactly that
name and meaning (`reason_for_call`). But it is documented as "the per-row
prompt **the agent receives**" and is seeded as the user's opening message. Put
the task goal there and the agent is handed the entire objective — constraints
and ids included — before it asks a single question.

**Why it matters.** Neither mistake errors. Nothing warns. The suite runs, the
traces look plausible, and the only symptom is that **every pass rate is
inflated**, because the agent no longer has to elicit anything. For an eval
platform this is the worst possible failure shape: a silent, one-directional
bias in the headline number. We caught it only by reading the platform source
to find out which component consumes which axis.

**Shape of the ask.** Rename or alias `behavior_instructions` to say *world*
(`world_behavior_instructions`, `simulator_guidance`), and warn at seeding when
`user_instruction` is long enough to be a task description rather than an
opener — in multi-turn persona mode a 500-character opener is almost always
this mistake.

---

## 9. `tools.mdx` documents one `when` operator; the validator accepts four

The reference documents the adapter's `when` predicate as `{"field", "eq"}` and
nothing else. The platform's own validator (`_validate_ledger_adapter`,
`apps/api/app/schemas/agent.py`) accepts **`eq | neq | exists | empty`**, with
`exists`/`empty` documented in-code as taking a boolean and treating a missing
field as the answer.

This is not academic. `modify_user_address` returns a `User` and undergoes no
status transition, so no scalar on it is constant across users — `eq` has
nothing to compare against. The only honest success discriminator is *presence*:
a `User` payload always carries `user_id`, an `{"error": …}` body never does.
With `eq` alone that write is unguardable, and an unguarded write there is not
harmless — `zip` is mapped straight from `$args`, so a rejected call would still
move the user's zip and silently break `find_user_id_by_name_zip` for every
later task touching that user.

We only found `exists` by reading the validator. Anyone working from the docs
would conclude the tool cannot be guarded.

---

## 10. An API key can create an org model but can never fix one

`POST /orgs/{id}/models` accepts a `pk_live_` key (`get_authorized_user_or_api_key`).
`PATCH`, `DELETE` and set-default on that same resource all require a Firebase
user (`get_authorized_user`) and `401` for a key.

So the one credential that can *create* a model cannot correct a typo in it,
retire it, or change which model is default. Every registration made by a key is
permanent and immutable to that key.

That asymmetry has already cost this org something. Its single model (`321`,
`claude-sonnet-4-5`) carries a stored `max_tokens: 4096` — the stale form
default from before that field became derivable — and it backs the judge, the
user simulator and the world simulator simultaneously. It cannot be cleared with
a key. The only workaround is to register a *second* model with `max_tokens:
null` and re-point every role at it, which leaves the broken row in the
catalogue forever.

We deliberately deferred registering anything for this experiment for exactly
that reason: with a key, model registration is a one-way door.

**Shape of the ask.** Either let a key that may create a model also PATCH and
DELETE it, or reject key-authenticated creates entirely. Create-but-never-amend
is the one combination that accumulates permanent mistakes.

---

## 11. The check page prints raw model ids where the model name belongs

The check detail page's config subheader renders the frozen model ids verbatim:

```
judge model 321  ·  simulators 321
```

`321` is not a fact about the run anyone can read. It is `claude-sonnet-4-5`,
and the whole reason those rows exist is so a reader can see which model judged
and which simulated — the two facts a result is least interpretable without.

Source: `apps/web/src/app/(app)/agents/[agentId]/checks/[checkId]/page.tsx`,
`snapshotConfigRows()` — `{ label: 'judge model', value: snap.judge_model_id }`,
and the simulator row joins the raw ids with `·`. The stored
`config_snapshot` genuinely only carries ids (`judge_model_id: 321`,
`user_simulator_model_id: 321`, `odyssey_simulator_model_id: 321`), so this is
not purely a display bug — the name is not in the evidence to begin with.

**Why it matters here specifically.** Model identity is a methodology fact. A
reproduction claim is meaningless without "judged by X, simulated by Y", and
right now that is only recoverable by separately querying the org catalogue —
which is also the one place a model can be *renamed or deleted*, so the id may
one day resolve to nothing at all.

**Shape of the ask, following the pattern already in the codebase.**
`harness_variant_snapshots` freeze the variant `name` alongside a nullable
`variant_id` precisely "so past evidence stays renderable". Do the same here:
freeze the model slug next to each model id in `config_snapshot`, and render
`claude-sonnet-4-5` with the id available as secondary detail. Display-time
lookup alone would still leave old checks unreadable once a model is retired.

---

## 12. Seeding advisories fire on the natural key spelling

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
