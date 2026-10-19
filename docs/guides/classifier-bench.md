# Classifier Bench — scoring the seven backend classifiers

The backend decides things from text and from events: which **tool groups** a
message needs, the **intent** of a message, the **class** of a chat turn, which
**skill** a message activates, which event **trigger** fires, and which note the
**recall rule** surfaces. Each of those is a classifier. Until this bench, none
of them had an accuracy number — only the tool-group router had a metamorphic
invariance suite.

The bench scores all seven **offline**, on hand-labelled fixtures, and compares
each one with the **majority-class baseline**: the accuracy a classifier that
ignores its input completely would get. A classifier that cannot beat that
baseline is not doing any work, and the bench fails.

The bench touches no network and no graph, and reads no clock in its
predictions: the same fixtures give the same figures on any machine.

## Run it

```bash
cargo test --test classifier_bench -- --nocapture
```

The `--nocapture` flag prints the table:

```
classifier    cases fr/en  accuracy  baseline   macro-F1   FR<->EN  pairs
tool_groups      32 16/16     100.0%     18.8%     1.000    100.0%     16
intent           32 16/16      81.2%     25.0%     0.827     75.0%     16
task_class       32 16/16      75.0%     25.0%     0.706     62.5%     16
skills           32 16/16     100.0%     25.0%     1.000    100.0%     16
triggers         32 16/16      56.2%     43.8%     0.639       n/a      0
bench wall time: 195ms
```

## What each column means

| Column | Meaning |
|---|---|
| `cases` | labelled cases in the fixture (`fr/en` splits them by language) |
| `accuracy` | share of cases where the classifier's label equals the gold label |
| `baseline` | accuracy of always predicting the most frequent gold label |
| `macro-F1` | macro-averaged F1 over every label (one-vs-rest) |
| `FR<->EN` | share of FR/EN pairs answered identically (`n/a` without pairs) |
| `pairs` | number of complete FR/EN pairs in the fixture |

The baseline matters more than the accuracy: `tool_groups` at 100 % against a
18.8 % baseline is a classifier doing real work; `triggers` at 56.2 % against
43.8 % barely beats "always answer the same thing" — that gap is exactly what
the follow-up work is meant to widen.

## The recall rule fixture

`recall_ranking` scores the integration rule of the spreading-activation engine
(`src/neurons/activation.rs`), not a text classifier. Each case declares a small
graph — seeds, weighted edges, node properties — and the adapter runs
`spread_and_rank`, the same pure function `activate_with_profile` drives hop by
hop. The label is the node that **should** win the propagation.

The fixture is built so the answer discriminates between the two candidate
rules:

- `N = 2` and `N = 3` are **controls**: a target reached by 2 or 3 moderate
  paths loses to a target reached by one stronger path under both rules;
- `N >= 4` is the **discriminant**: the convergent target outranks the strong
  single path only when contributions accumulate.

Read it as a **non-regression ratchet on a chosen rule**, not as a measure of
retrieval quality. The five text classifiers are judged on human labels
independent of the code; this one is judged against a spec written before the
code, and its job is to make the rule impossible to change by accident.

The fixture is language-neutral by construction (a graph is not French or
English), so it declares no `pair` ids and `FR<->EN` reads `n/a`. The `lang`
field only satisfies the bench shape.

## The ranking fixture

`recall_selection` scores the *selection* that follows propagation: given the
candidates and a slot count, which nodes survive the truncation, and in which
order. The adapter runs the production `select_top`
(`src/neurons/activation.rs`), the same function `activate_with_profile` calls
in Phase 3.

The fixture pins the outcomes that can silently drift:

- **order and truncation** — score descending, then node id ascending; more
  slots than candidates returns them all; zero slots returns `none`;
- **ties** — the tie cases list their candidates in non-ascending name order, so
  a score-only sort would answer differently.

The fixture pins those *outcomes*; the tie-break *determinism* is pinned by
`truncation_does_not_depend_on_the_input_order` in `activation.rs`, which fails
if the node-id tie-break is removed. The bench alone could not: its only
pass criterion is "better than the majority baseline", and a fixture whose tie
cases are a minority still clears that bar with the tie-break gone.

Like `recall_ranking` it is language-neutral and declares no `pair` ids.

## Where the figures are published

- **Shadow routing report** — `RoutingReport.classifiers`, served by
  `GET /api/chat/routing/report`. The field is **additive**: a report
  serialised before the field existed reads back as an empty list.
- **Heartbeat** — the `classifier_bench` check re-scores the fixtures every six
  hours and logs one line per classifier, with a warning for any classifier
  under its baseline. It is pure computation (in memory, under a second).

## Add a case to a fixture

Fixtures are versioned JSON under `tests/fixtures/classifiers/`, embedded in the
binary at build time (`include_str!`). A case is:

```json
{ "id": "tool_groups-33-fr", "lang": "fr", "pair": "33",
  "input": { "message": "cherche où le jeton de rafraîchissement est validé" },
  "label": "CodeExploration" }
```

- `id` must be unique in the fixture;
- `lang` is `fr` or `en` (the bench requires at least 10 French cases);
- `pair` ties a French case and an English case that mean the same thing: the
  share of pairs answered identically is the `FR<->EN` column. The twin case
  carries the same `pair` value and the other `lang`. Without any `pair`, the
  fixture simply has no language-agreement figure (as for `triggers`, which is
  driven by events, not by a French or English message);
- `label` is the hand-written gold label.

Then **refresh the manifest**, or the bench fails:

```bash
cd tests/fixtures/classifiers
shasum -a 256 *.json > MANIFEST.sha256
```

`the_fixtures_match_their_sha256_manifest` recomputes every hash and also
refuses a fixture on disk that is missing from the manifest: the two lists must
match exactly.

## Add a new classifier

1. Write the adapter in `src/evaluation/classifiers.rs`: implement
   `Classifier` (`name` + `predict`) calling the **production** code path — no
   parameter of its own, so the bench measures what the backend really does.
2. Register it in `classifiers::all()`.
3. Add its fixture under `tests/fixtures/classifiers/` and to `embedded_fixtures()`
   in `src/evaluation/mod.rs` (the `SOURCES` table).
4. Refresh the manifest (above). The bench already asserts that every fixture
   has at least 30 cases and at least 10 French ones, and that every embedded
   fixture names a classifier that exists.

The majority baseline, the accuracy, the macro-F1 and the language-agreement
metrics are shared and must not be re-implemented per classifier: they live in
`src/evaluation/mod.rs`, and `macro_f1` reuses `f1_score` from
`neural-routing-gnn`'s benchmark so the arithmetic is the one the neural crates
already use.
