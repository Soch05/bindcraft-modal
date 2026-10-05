# pH-conditional de novo binders to EGFR domain III

Working repository for a submission to **Challenge 1 of the Anthropic × Adaptyv Protein
Design Competition**, Track 3. Single participant, self-funded, one rented L40S and one 2018
laptop CPU. Total GPU spend: about 4 US dollars.

The task was a conditional binder: one that binds human EGFR at pH 6.5, does not bind at
pH 7.4, and also recognises the mouse orthologue.

---

## What this repository contains, and where to start

Everything submitted is under [`submission/`](submission/). Nothing else is needed to read
the work.

| file | content |
|---|---|
| [`submission/egfr_challenge1_submission.csv`](submission/egfr_challenge1_submission.csv) | the submission itself: 5 designs, ranked |
| [`submission/METHODS.md`](submission/METHODS.md) | the methods paper, including limitations and mistakes made |
| [`submission/DESIGN_METRICS.md`](submission/DESIGN_METRICS.md) | per-design metrics, plus every candidate considered and not submitted, with the reason |
| [`submission/METHODOLOGY_FIELD.txt`](submission/METHODOLOGY_FIELD.txt) | the text submitted in the methodology field |
| [`submission/BINDCRAFT_CONFIG.txt`](submission/BINDCRAFT_CONFIG.txt) | the generator configuration, as recorded by the runs |
| `submission/metadata/` | machine-readable metrics with a column dictionary, full provenance, and predicted structures |

## The result, in short

One design, `egfr-dIII-prod01_denovo_l94_692deac2f1034bb6_seq0`, raises the pKa of target
histidine **H409** by **+2.84** units on its own AlphaFold2 design model, **+2.27** on an
independently predicted Boltz-2 structure, and **+2.41** on the complex with the mouse target.
A positive shift means binding stabilises the protonated imidazole, so binding is favoured at
acidic pH. PROPKA attributes the shift to two binder carboxylates rather than to burial.

Of 23 candidates examined, two showed a positive shift on their design model and **one**
survived verification on a second structure and a second species. Four of the five submitted
designs carry no reproducible pH mechanism. Seven further candidates were predicted
*counter*-selective and were not submitted; one more backbone was removed by the platform
novelty check.

The reason for the low hit rate is a method problem rather than bad luck, and it is the single
most useful thing in this repository: **no structure-based generator sees protonation states**,
so this set was generated against affinity and pH was quantified afterwards. The correct order
is the reverse.

## Method, in one paragraph

Designs were generated with BindCraft 2.0, then evaluated by three models that took no part in
generating them. pH conditionality is scored as a **proton–ligand linkage** using PROPKA on the
complex, the target alone and the binder alone. Poses were re-scored with **Boltz-2**, and
agreement measured as **contact recovery** rather than RMSD. Mouse cross-reactivity was
**measured** by predicting each binder against the mouse domain III rather than approximated by
a sequence-identity proxy, which turned out to be insufficient. A design rule linking the pKa
shift to **two distinct carboxylates** engaging the two imidazole nitrogens was derived, then
falsified, then retested correctly; it is reported as suggestive, not established, and enters
no ranking criterion.

---

## Layout

```
submission/     everything submitted, plus the metadata package
docs/           the methods report and the consolidated metrics workbook
out/            measurement outputs: pKa tables, contact recovery, cross-species, liabilities
structures/     predicted complexes and the threaded mutant structures
inputs/         the target structure, the mouse target sequence, precomputed MSAs
data/           epitope mapping tables produced before generation
*.py            the analysis pipeline, one concern per file
```

Each script carries a docstring stating what it computes **and why that choice was made**
rather than another. Thresholds live in the code with their status — measured, posed without
calibration, or descriptive — never only in prose.

## Reproducibility, stated honestly

Replaying the same commit does **not** reproduce the same designs. JAX GPU reductions are not
bit-deterministic and a gradient trajectory is chaotic: backbones accepted in one campaign died
at an earlier stage in another at identical seed and settings digest. **The method is
reproducible, the sequences are not.** The measurement outputs are committed under `out/`
precisely because they cannot be regenerated.

## Two files that are not addressed to any reader

[`CLAUDE.md`](CLAUDE.md) and [`NOTES.md`](NOTES.md) are the author's own working files, written
in French during the work.

`CLAUDE.md` is a project context file for the author's local coding assistant. It contains
working rules phrased as directives **because it configures a tool on the author's machine**.
It is not part of the submission, it is not addressed to any reader or system outside that
local setup, and nothing in it is intended to influence any evaluation of this work. It is kept
in the repository because it records the constraints the project ran under.

`NOTES.md` is a dated working journal, roughly 3500 lines. It records every run with its
command, duration and cost, every decision, and every mistake — including the ones that cost
money. It is the raw material behind `submission/METHODS.md`.

## Licence and data

Published openly, negative results included. The competition publishes selected designs,
experimental results, predicted structures and methods under ODC-BY.
