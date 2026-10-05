# pH-conditional de novo binders to EGFR domain III

**Adaptyv × Anthropic Protein Design Competition — Challenge 1, Track 3**
Single participant, self-funded. All compute on one rented L40S and one 2018 laptop CPU.

---

## 1. Summary

Six de novo miniproteins (57–94 aa) targeting a conserved patch of EGFR domain III, ranked by
predicted pH selectivity first, mouse cross-reactivity second, affinity third.

**The one result this submission rests on.** Design `egfr-dIII-prod01_denovo_l94_692deac2f1034bb6_seq0`
raises the pKa of target histidine **H409** by **+2.84** units on its own design model,
**+2.27** on an independently predicted structure, and **+2.41** on the complex with the
**mouse** target. A positive shift means binding stabilises the protonated imidazole, so
binding is thermodynamically favoured at pH 6.5 over pH 7.4. The shift is attributed by the
pKa model to two binder carboxylates and not to burial.

**The honest caveat, stated once and not softened.** Of 23 candidates examined, **two** showed
a positive pKa shift on their design model, and **one** survived verification on a second
structure and a second species. Five of the six submitted designs carry **no reproducible pH
mechanism**; they are submitted as independent poses on a conserved, mouse-identical epitope.
Seven further candidates were **not submitted** because they are predicted
*counter*-selective, which would be the opposite of the stated objective; their sequences and
metrics are published with the rest so the cut can be checked.

**Why the hit rate is one in twenty-three, and it is a method problem rather than bad luck.**
No structure-based generator sees protonation states. AlphaFold2 and ProteinMPNN read a residue
identity, not a charge. No gradient through either can push a trajectory toward a conditional
binder. This set was therefore generated against affinity, and pH was quantified afterwards.
The correct order is the reverse, and §8 reports the design rule that this work extracted for
that purpose.

---

## 2. Target and epitope

| | |
|---|---|
| Human target | UniProt **P00533**, extracellular region, domain III |
| Mouse target | UniProt **Q01279**, domain III, UniProt 333–530 |
| Structure | PDB **6ARU** chain A, folded conformation, 3.20 Å |
| Target file | `inputs/6ARU_A_309-506.pdb` — 198 residues, chain A only, no heteroatoms |
| Domain boundaries | CATH-Gene3D `G3DSA:3.80.20.20` |
| PDB → UniProt offset | **+24**, read in `_struct_ref_seq` and independently recovered by scanning |
| Hotspots | **A318, A323, A406, A409** |
| Coldspot | **A359** |

Domain III is the epitope of cetuximab and panitumumab, which is precisely why the most
documented patch was **not** used: cetuximab does not recognise mouse EGFR, so the
best-characterised epitope is the one that endangers the cross-reactivity objective. The patch
was selected instead on **measured** human/mouse conservation across 96 candidate surface
patches.

Two measurements drove the final hotspot set, which spans **16.73 Å** between alpha carbons and
is **4/4 identical** in mouse:

- **A323 is an acidic anchor.** Its carboxylate carries 75.0 Å² of accessible surface, 0.55
  apolar fraction, pointing outward at 33°, in the 89th percentile of the patch survey. It
  supports the route where a binder histidine pairs with a target acid.
- **A409 is a conserved target histidine.** Its imidazole exposes 44.7 Å² on CE1 alone. It
  supports the inverse and much easier route: placing an acid on a binder surface is trivial
  compared with placing a histidine with the right pKa and geometry. H409 is `identical` in
  mouse, so this route also serves the cross-reactivity objective.

`A359` was set as a coldspot because it is a human histidine that becomes **arginine** in mouse
(`H359R`, confirmed in §7), and it carries 16 % of the apolar surface of its zone — a residue a
trajectory would otherwise be drawn to, and which would break cross-reactivity.

A candidate hotspot `A325` offering +111 Å² of apolar anchor was **rejected**: its nearest
glycosylation sequon sits at 7.3 Å, and that distance is measured to the sequon CB rather than
to the glycan tree, so it understates occlusion.

---

## 3. Generation

| | |
|---|---|
| Generator | **BindCraft 2.0**, repository `PacesaLab/BindCraft2` |
| Pinned commit | `a8d0f2002df373842b86a3c20c5a060c5cfdf980` |
| Recorded revision | `a8d0f2002df373842b86a3c20c5a060c5cfdf980-dirty` |
| Campaigns | `egfr-dIII-prod01` (3 designs), `egfr-dIII-prod02` (20 designs) |
| Binder lengths | 55–95 aa, sampled |
| `max_trajectories` | 150 |
| `kept_sequences` | 2 |
| Hardware | NVIDIA L40S, 2 concurrent workers |

The `-dirty` suffix is reported as recorded: the working tree inside the container differed
from the pinned commit, so that hash does not fully pin the code state.

**23 designs on 13 independent backbones.** A trajectory produces a backbone, identified by a
hash; ProteinMPNN then proposes sequences for that backbone and `kept_sequences=2` retains two.
Both are unmodified campaign outputs, each with its own BindCraft rank, metrics and predicted
structure — neither is derived from the other, and `seq0` is not more canonical than `seq1`.
What they share is the **backbone**, so two sequences with the same hash are **not independent
poses**. One design per backbone is therefore submitted.

**Designs are not bit-reproducible.** Replaying the same commit does not return the same
sequences: JAX GPU reductions are not bit-deterministic and a gradient trajectory is chaotic.
Three backbones accepted in one campaign died at an earlier stage in the next, at identical
seed and recipe hash. What is reproducible is the method.

---

## 4. Quantifying the pH mechanism

This is the central methodological contribution.

### The mechanism being engineered

Histidine is the only canonical residue whose pKa — about 6.0–6.5 free in solution, shiftable
by environment — falls between the two assay pH values. Asp and Glu sit near 4, Lys and Arg
above 10. Two routes exist:

- **Route 1** — a binder histidine against a target acid (`D323`): a salt bridge that exists
  only when the histidine is protonated, so binding is **gained** at pH 6.5.
- **Route 2** — a binder acid against the conserved target histidine (`H409`): far easier to
  design, and it serves cross-reactivity because H409 is identical in mouse.

Measured outcome: **Route 1 is dead across this set.** Two of three designs examined for it
carry no histidine at all, and the single interface histidine found sits 28.4 Å from D323.
Every mechanism reported below is Route 2.

### How the shift is computed

PROPKA 3 is run on three states extracted from the **same file**, so the comparison isolates
the presence of the partner rather than a conformational difference: the complex, the target
alone, and the binder alone. 141 runs in the first pass.

The reported quantity is the shift of the pKa of H409 between bound and unbound target. Its
thermodynamic meaning is the proton–ligand linkage: if binding raises the pKa of H409, binding
stabilises the protonated form, so binding is favoured at lower pH. A predicted selectivity
factor follows:

```
Ka(6.5)/Ka(7.4) = [(1+10^(pKa_bound−6.5))/(1+10^(pKa_bound−7.4))]
                ÷ [(1+10^(pKa_free−6.5))/(1+10^(pKa_free−7.4))]
```

**Numbering was verified on all 23 structures, not on one.** The mechanism rests on H409 in PDB
numbering; UniProt numbering would be offset by 24 and would silently address a different
residue. The conversion step refuses to proceed unless residue 409 of the target chain is a
histidine, and the full six-histidine fingerprint of the target — 334, 346, 359, 394, 409, 483
— was recovered on every structure.

### Result

On the design models, **2 of 23** designs raise the pKa of H409. The other 21 are at zero or
**negative** shift, meaning binding would be *disfavoured* at acidic pH.

For the leading design, the pKa model decomposes the shift of H409 from 6.50 to **9.11**:

```
HIS 409 A   9.11   86 %   −2.50 desolvation   +1.60 ASP 56 B   +1.60 GLU 73 B
                                              +1.39 ASP 56 B   +0.54 GLU 73 B  (coulombic)
```

The **desolvation term is negative**: burial alone would *lower* the pKa. The rise is
attributable entirely to the two binder carboxydates `ASP56` and `GLU73`, through side-chain
hydrogen bonding and through the coulombic term. This is not a burial artifact and not a
neighbouring target residue.

### An unbiased cross-check

Restricting attention to H409 assumes the answer. A second indicator was therefore computed:
the product, over **every** ionisable group in the complex, of each group's individual
contribution to pH selectivity — the full proton–ligand linkage, privileging no residue chosen
in advance.

| design | H409 alone | all groups | significantly shifted groups only |
|---|---|---|---|
| `692deac2f1034bb6_seq0` | **5.28** | **5.49** | **5.65** |
| `36dbfc4737a3e59b_seq1` | 2.60 | 2.39 | 2.44 |
| `987fe804e455bc58_seq0` | 0.88 | 0.33 | 0.39 |

The two indicators agree on the leading designs and their order. Where they disagree, the
unbiased view is **less** favourable, never more. No candidate is rescued by it, which is the
strongest available evidence that the hypothesis-driven epitope choice did not miss a better
option.

An unbiased scan over all shifted groups found only one other candidate route: `ASP344A`
reaches a bound pKa of 6.08 in one design, inside the useful window, for a factor near 1.3.
`ASP436A` shifts by up to 1.8 units in many designs but stays below pKa 5, hence deprotonated
at **both** pH values — a real effect with no consequence for the objective.

### Calibration

PROPKA is commonly wrong by about one pKa unit, more on large shifts. These values **rank**
designs; no absolute factor is claimed. For the same reason the ranking **discretises** the pH
criterion into tiers instead of sorting on the continuous value: ordering a design at −0.01
above one at −0.40 would be reading an order out of noise.

---

## 5. Orthogonal validation of the pose

BindCraft optimises designs by gradient descent **through AlphaFold2**. Its reported `i_pTM`
and `i_pAE` are therefore in-sample: the generator had access to the judge during every
trajectory. In machine-learning terms these are training scores, not test scores.

**Boltz-2 2.2.0** was used as an independent folding model — different architecture, separately
trained weights, no involvement in generating these designs. Its **affinity module was not
used**, being calibrated for small molecules rather than protein–protein interfaces.

| | |
|---|---|
| Samples | 3 diffusion samples, 3 recycling steps |
| Target MSA | ColabFold MMseqs2, mode `env`, **precomputed once and baked into the image** |
| Binder MSA | **empty**, single-sequence mode |
| Throughput | ~67 s per complex on an L40S |

The binder MSA is empty by design: these binders are de novo and have no natural homologues, so
a homology search over them returns noise, and sequence novelty is a requirement of the
challenge. Precomputing the target MSA removes every network call from the GPU container.

**Agreement is measured as contact recovery, not RMSD.** An interface RMSD forgives a rotation
of the binder that keeps it on the same patch while no residue faces the same partner any more,
and it penalises a rigid translation that preserves every contact pair. The question here is
whether the independent model places the same residues face to face, so residue-residue contact
pairs are counted — any heavy-atom pair within 5.0 Å — and the score is the fraction of the
design model's pairs recovered. It is asymmetric on purpose.

**Result: 23 of 23 designs recovered, contact recovery 0.696 to 0.969, folding-model iptm 0.848
to 0.957.** The design poses are not AlphaFold2 artifacts.

**A negative result about this project's own threshold.** The "confirmed pose" criterion set in
advance — recovery ≥ 0.50 and iptm ≥ 0.60 — **separates nothing** on this set, since all 23
candidates clear it comfortably. It therefore carries no ranking information. It is reported
rather than quietly dropped.

**The limit of this orthogonality.** Boltz-2 and AlphaFold2 are independent in architecture and
in weights, but both were trained on the PDB. Their agreement rules out an artifact specific to
AlphaFold2; it does not rule out a bias shared by both and inherited from the data. It is not
experimental validation.

---

## 6. Interface metrics of the submitted set

See §12 for the metric dictionary. Scales: `i_pTM` and `i_pAE` on [0,1], areas in Å².

### pH selectivity, objective 1

| rank | design | len | pH tier | ΔpKa design / folding / mouse | factor |
|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | 94 | **robust mechanism** | 2.84 / 2.27 / 2.41 | 5.28 |
| 2 | `l63_987fe804e455bc58_seq1` | 63 | **non-reproducible mechanism** | -0.29 / 0.97 / -0.7 | 0.848 |
| 3 | `l59_36dbfc4737a3e59b_seq1` | 59 | **non-reproducible mechanism** | 0.96 / -0.17 / -0.38 | 2.599 |
| 4 | `l61_cd272a8fd929c7ee_seq0` | 61 | **neutral** | -0.4 / -0.39 / -0.5 | 0.808 |
| 5 | `l58_fd5dae7987a2388d_seq1` | 58 | **neutral** | -0.07 / -0.12 / -0.13 | 0.956 |
| 6 | `l57_9526c9216eb7d6db_seq0` | 57 | **neutral** | 0.03 / -0.41 / -2.62 | 1.022 |

### Interface quality and self-consistency

| rank | i_pTM | i_pAE | BSA Å² | hotspot frac | ipSAE max / A→B | LIS | contact recov. | self-consist. |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.83 | 0.19 | 902.6 | 0.5 | 0.9071 / 0.8309 | 0.7108 | 0.897 | 0.5824 |
| 2 | 0.83 | 0.19 | 1049.0 | 0.5 | 0.9193 / 0.8126 | 0.7454 | 0.935 | 0.6171 |
| 3 | 0.77 | 0.24 | 953.8 | 0.5 | 0.9028 / 0.762 | 0.7323 | 0.857 | 0.5466 |
| 4 | 0.84 | 0.18 | 723.5 | 0.25 | 0.8878 / 0.8044 | 0.7311 | 0.96 | 0.6557 |
| 5 | 0.82 | 0.19 | 916.6 | 0.5 | 0.8771 / 0.739 | 0.7168 | 0.938 | 0.6207 |
| 6 | 0.76 | 0.27 | 722.5 | 0.25 | 0.8391 / 0.704 | 0.6765 | 0.917 | 0.5811 |

### Mouse cross-reactivity (objective 2), liabilities, descriptive

| rank | mouse iptm | Δ iptm | mouse epitope recov. | mouse identity proxy | liabilities (exposed) | ESM-2 PLL |
|---|---|---|---|---|---|---|
| 1 | 0.914 | -0.016 | **0.897** | 0.857 | 5 degradation, 1 patches | -2.0341 |
| 2 | 0.94 | -0.006 | **0.935** | 0.87 | 5 degradation, 1 patches | -2.259 |
| 3 | 0.928 | -0.01 | **0.839** | 0.81 | 1 degradation, 2 patches | -2.0516 |
| 4 | 0.933 | -0.008 | **0.92** | 0.8 | 2 degradation, 3 patches | -2.0997 |
| 5 | 0.938 | 0.002 | **0.812** | 0.778 | 0 degradation, 1 patches | -1.9588 |
| 6 | 0.851 | -0.069 | **0.479** | 0.778 | 6 degradation, 0 patches | -1.9272 |

Free cysteines: **0** in every design. Hard liabilities: **0**. N-glycosylation sequons are counted in the metadata but not scored, because expression is cell-free.

`ipTM_af`, `pDockQ` and `pDockQ2` are computed by the ipSAE reference implementation but are **not reported**: on Boltz-2 input `ipTM_af` reads 0.000 because the tool expects an AlphaFold JSON, and the two pDockQ columns are constant across all six designs. The reason for each exclusion is recorded in `metadata/design_metrics.csv`.

---

## 7. Mouse cross-reactivity, measured

The second objective requires one sequence to recognise both P00533 and Q01279. Until this step
it was approximated by a **sequence proxy** — the fraction of contacted target residues
identical in mouse — which says nothing about local mouse conformation.

### The mouse target, derived twice

The mouse domain III sequence was constructed by **two paths sharing no step**, and the
procedure refuses to emit the file if they disagree:

- **A** — alignment of the complete 1210-residue Q01279 sequence against the human domain III
  sequence read from the target PDB;
- **B** — the per-position mouse column of an alignment produced separately during epitope
  mapping, restricted to PDB 309–506.

They agree. This guard exists because selecting the wrong region would have produced a
plausible but false mouse target, which the folding model would have folded with high
confidence, making the "measured cross-reactivity" an artifact that nothing downstream would
flag.

| | |
|---|---|
| Mouse domain III | UniProt **333–530**, 198 residues |
| Identity | **173/198 = 87.4 %** |
| Indels in the window | **none** — so PDB numbering 309–506 applies to both species and the contact mapping is the identity |
| **H409** | **conserved** — Route 2 is transferable |
| Target histidines | human {334, 346, 359, 394, 409, 483}; mouse {334, 346, 394, 409, 480} |
| Divergent positions | S324T, N337Y, S340A, R353K, **H359R**, Q366R, D369E, E388D, R390W, S418G, K443R, S460P, G461N, I467M, S468N, G471A, N473K, S474D, T478V, G479N, Q480H, **H483N**, A484P, P488S, R503Q |

A separate 3502-sequence MSA was computed for the mouse target; reusing the human one would
inject the alignment of the wrong protein.

### Controlling for the predictor

The published pKa shifts came from design models. Comparing them with a mouse shift computed on
a Boltz-2 model would confound species with predictor. PROPKA was therefore re-run on the
**human Boltz-2** models as well, so the species comparison is Boltz against Boltz, and three
independent measurements exist per design.

### Results

| | |
|---|---|
| Δ iptm, mouse − human | **−0.098 to +0.007**, mean **−0.016** |
| Human epitope recovered on mouse | **0.478 to 0.935** |

**Binding confidence barely suffers from the change of species**, which is the measurement that
validates choosing the epitope on conservation rather than on the documented cetuximab patch.

But three designs change binding mode on the mouse target — `9526c9216eb7d6db` seq0 and seq1 at
0.479 and 0.478, and `f6d5f550a210fd48_seq0` at 0.604 — losing roughly half their epitope. **The
sequence proxy did not separate them**: their proxy values, 0.778 and 0.800, sit mid-pack. The
second objective is therefore ranked on the structural measurement, with the proxy kept only as
a tiebreaker.

### The predictor-dependence finding

| design | design model | folding model, human | folding model, mouse | verdict |
|---|---|---|---|---|
| `692deac2f1034bb6_seq0` | **+2.84** | **+2.27** | **+2.41** | robust |
| `36dbfc4737a3e59b_seq1` | **+0.96** | −0.17 | −0.38 | **not reproducible** |
| `987fe804e455bc58_seq1` | −0.29 | **+0.97** | −0.70 | not reproducible |
| `987fe804e455bc58_seq0` | −0.22 | **+1.04** | −0.11 | not reproducible |

Across the 23 designs the difference between the two predictors spans −1.13 to +1.26, median
−0.10, and the **mechanism verdict agrees on 20 of 23**. The three disagreements fall exactly on
the borderline cases, between −0.3 and +1.1 — that is, on the designs one would be tempted to
promote.

`36dbfc4737a3e59b_seq1` was consequently **demoted**: its mechanism was a property of one
predicted structure, not of the sequence. A mechanism now counts as **robust** only when the
shift is positive on all three measurements, and one design in this set qualifies.

---

## 8. A design rule: derived, falsified, retested

The criterion used to select this epitope was "a binder carboxylate within 4 Å of an imidazole
nitrogen of H409". **Eleven designs satisfy it. Two raise the pKa.** The criterion does not
predict the mechanism: three designs with near-ideal salt-bridge geometry — 3.21 Å at 119.5°,
3.17 Å at 120.7°, 3.36 Å — have zero or negative shifts.

**First hypothesis, refuted.** "Both imidazole nitrogens must be engaged." Nine designs engage
both within 5.0 Å, with shifts from +2.84 to −1.19. Necessary, not sufficient — no design with
zero engaged nitrogens has a positive shift.

**Second hypothesis.** What matters is that the two nitrogens are engaged by **two distinct
carboxylate residues**. A single carboxylate pivoting between them can stabilise only one
hydrogen bond at a time; two distinct residues can saturate the protonated imidazole from both
sides, which is the condition for displacing its acid–base equilibrium. The measured quantity
is the worst of the two distances in the best pair of distinct residues, one on ND1 and one on
NE2 — the bottleneck of the bidentate arrangement.

| test | geometry from | pKa from | positive bottlenecks | negative bottlenecks | outcome |
|---|---|---|---|---|---|
| 1 | design model | design model | 3.34, 3.35 | 4.54, 4.55, 6.40 | clean, **1.19 Å** margin |
| 2 | design model | **folding model** | 3.35, 4.54, 4.55 | **3.34**, 6.40 | **overlap 1.21 Å — fails** |
| 3 | **folding model** | **folding model** | 3.19, 3.88, 4.03 | 6.23, 6.74, 7.97 | clean, **2.20 Å** margin |

Test 1 was the original derivation and is the weakest of the three: geometry and pKa came from
the **same** structure, so the correlation was partly self-referential. Test 2 breaks it. Test 3
is the correct test — one predictor on both sides, so the rule is evaluated on data that did not
generate it — and it passes with nearly twice the original margin.

**What the failure of test 2 teaches.** The rule is **local to the structure**: the bottleneck
must be measured on the structure whose pKa is being evaluated, never transferred between
predictors, because two predictors place side chains differently and it is the carboxylate
position that makes the mechanism. This is consistent with `36dbfc4737a3e59b_seq1`, whose
bottleneck is 3.34 Å on its design model but falls into the 6.2–8.0 Å group on the folding
model: the carboxylates move and the mechanism goes with them.

**Bounded honestly.** The informative comparison is not 3 against 20 — seventeen designs have no
carboxylate within reach of H409 and form a degenerate class. It is **3 against 3** among the
six designs capable of forming a pair, where a clean split has a probability near **5 %** under
the null. Test 1 gives 2 against 3 among five, near 10 %. Two tests on two structure sets, both
passed, margins 1.19 and 2.20 Å: **the rule is supported, not established.**

It is falsifiable, it costs no GPU time to evaluate, and it is **actionable** — it prescribes
two distinct carboxylates near 3.4 Å from ND1 and NE2 rather than one at 2.5 Å, which is a
different design brief from the one that produced this set. **It enters no ranking criterion**
and is reported as a method result.

It also explains, after the fact, why the mutation campaign of §9 failed: each mutation added
**one** carboxylate where the rule calls for a coordinated pair.

---

## 9. An acid-mutation campaign that failed, and why

Twelve single substitutions toward Asp or Glu were grafted onto accepted backbones by
**multi-rotamer threading**: the mutation is placed on the parent's backbone, which stays
rigorously identical, so a paired analysis has exactly one variable. Three to four rotamers per
mutant were generated from a populated library with the backbone untouched, and rotamers whose
heavy atoms came within 2.2 Å of another were discarded.

Multi-rotamer rather than single-rotamer was not optional: the pKa of a buried carboxylate
depends almost entirely on desolvation and local coulombic terms, hence on the **exact**
side-chain position, so a single unrelaxed rotamer yields a pKa whose provenance — chemistry or
placement artifact — is unknowable.

**Geometric screening.** Of 12 mutants, **3** have at least one rotamer both within salt-bridge
range of H409 and free of overlap. `S28D` is **structurally impossible** on a rigid backbone:
all three rotamers clash.

**Why the position-selection proxy failed.** Positions were chosen on the **CB**-to-H409
distance. CB is fixed by the backbone: it says where the side chain departs, not where its tip
arrives. Beyond roughly **4.3 Å of CB** the carboxylate no longer reaches H409 — the four
mutations at CB ≥ 4.55 Å all end 4.9–9.2 Å away. The real threshold was near 4.3 Å and had
never been set.

**The self-defeating carboxylate.** An introduced acid must be **charged** at pH 6.5 to form the
bridge. Buried against a positive charge its own pKa rises; above about 5.0 it is partly
protonated already at pH 6.5 and the mechanism cancels itself. Five mutants are **red** on this
test across all retained rotamers. `S15D` — one of only three within reach — carries an `ASP15`
at pKa **8.67–8.77**: neutral at both pH values, hence unable to form the bridge it was
introduced for.

**Paired outcome: 1 improves, 4 neutral, 7 degrade.** The single improvement, `S44D` on
`a6d2f6834f22e574_seq1`, moves the shift from −2.85 to **+0.23** — a ΔΔ of **+3.08** with a
rotamer spread of only 0.06 and a healthy carboxylate. It **repairs** a counter-selective design
to neutrality; it does not create a switch.

**Structural cost, measured.** The 12 mutant sequences were predicted by the folding model —
the first time any structure model saw them, the threaded files being side-chain grafts rather
than predictions — and compared against their parent's contacts. **No mutation breaks the
interface**: the worst cost is **0.079** of contact recovery, against a 0.696–0.969 spread
between designs. The failure is therefore **chemical and electrostatic, not structural**: a
carboxylate out of reach, or one whose own pKa rises too far, is invisible in interface
geometry.

**No mutant is submitted.** A mutant enters only with a robust mechanism and a healthy
carboxylate, and none satisfies both.

---

## 10. Ranking

**Lexicographic, no weighted composite.** The challenge ranks pH selectivity first, mouse
cross-reactivity second, affinity third. A weighted score would require inventing weights
between those axes that could not be justified and that would hide the trade-offs.

Order of criteria:

1. **pH tier** — robust mechanism, then non-reproducible mechanism, then neutral, then
   counter-selective. Within the robust tier only, the selectivity factor separates designs,
   because there the differences exceed the pKa model's error.
2. **Mouse cross-reactivity** — the structural measurement of §7, with the sequence proxy as
   tiebreaker.
3. **Affinity proxies** — `i_pAE` then `i_pTM`.

A design carrying a measured pH mechanism was never placed below one without, and no design was
demoted because an orthogonal model scored it lower. The one demotion in this set rests on **pKa
measurements** on two further structures, not on a pose preference.

**Allocation: one design per backbone.** Two sequences sharing a backbone are not independent
poses. The alternative of submitting matched wild-type/mutant pairs was costed at 14 designs
over 7 backbones and **rejected**: the uniqueness filter is stated only as "do not submit the
same design twice", with no identity threshold, and a point mutant at 98–99 % identity to its
parent is a rejection risk rather than a strategy. Maximum pairwise identity in the submitted
set is **20.7 %**.

**Why six and not thirteen.** One design per backbone yields 13. Seven of those are predicted
**counter-selective** on the highest-ranked criterion — their binding would be disfavoured at
pH 6.5. Submitting them would contradict the instruction to submit the designs expected to
perform best, so they were not submitted, at the cost of seven independent backbones. Their
sequences and metrics are published with the rest. The quota of
20 is a ceiling and was never treated as a target.

**The scope of that cut.** It removes designs predicted **wrong** on objective 1, not designs
predicted **weak** on objective 2. Rank 6 is neutral on the pH criterion rather than
counter-selective, so the rule does not reach it; it is retained for the reason given in §10.1.

### 10.1 Rank 6 is a negative control, not a sixth candidate

`egfr-dIII-prod02_denovo_l57_9526c9216eb7d6db_seq0` is **submitted as a negative control for
the mouse cross-reactivity prediction: expected to bind human EGFR and not mouse.**

Three independent measurements converge on it as the weakest member of the submitted set, and
all three point the same way:

| measurement | value | rank within the submitted six |
|---|---|---|
| mouse epitope recovery | **0.479** | lowest |
| ipSAE, target-to-binder direction | **0.704** | lowest |
| pKa shift of H409 on the mouse complex | **−2.62** | lowest |

It is the only submitted design on which this pipeline makes a **directional and falsifiable**
prediction rather than expressing a hope. The other five are submitted because they are
expected to bind; this one is submitted because of what either outcome would establish:

- **If it binds human and not mouse**, the epitope analysis of §7 is validated
  experimentally — a contact-recovery collapse from 0.9 to 0.48 on a predicted mouse complex
  would have predicted a real loss of species cross-reactivity. For a methods contribution
  that is worth more than a sixth binder that works.
- **If it binds mouse anyway**, then a contact recovery of 0.479 does not mean what it was
  taken to mean here, and the structural cross-reactivity measurement that this work
  substituted for a sequence proxy needs recalibration.

Retaining it costs one slot out of a ceiling of 20 and buys a test. It is labelled as a control
in `metadata/design_metrics.csv` under `submission_note` so the intent travels with the design
rather than living only in this document.

---

## 11. Known weaknesses

Stated so that a reader need not infer them.

- **One design in twenty-three carries a verified pH mechanism.** Five of the six submitted
  carry none. The set was generated without any pH objective, for the reason given in §1.
- **The predicted factor is of order 5×**, while the challenge asks for no detectable binding at
  pH 7.4. That gap is not closed and nothing here claims to close it.
- **Nothing is experimental.** Every number comes from predicted structures and empirical
  models. Two predictors agreeing remain two predictors, both trained on the PDB.
- **The mouse structure is itself predicted.** Q01279 domain III has no experimental structure.
  The sequence was extracted and double-checked; its fold is inferred.
- **Multitarget design was never used.** Cross-reactivity was **evaluated**, never **optimised**.
- **Mutant structures are unrelaxed** side-chain grafts, so their absolute pKa values are coarse.
  The paired comparison, which shares a backbone, is the clean part.
- **The global linkage factor assumes independent sites**, which they are not when two groups
  touch, and it accumulates one model error per group.
- **Two thresholds set in this work turned out to be non-discriminating** — the confirmed-pose
  criterion of §5. They are reported, not removed.
- **A constant was nearly shipped as a metric.** The ipSAE reference implementation also emits
  `pDockQ`, `pDockQ2` and `ipTM_af`. On Boltz-2 input `pDockQ` returned **0.0183** and
  `pDockQ2` **0.0073** — identical to four decimals across all six designs — and `ipTM_af`
  returned **0.000**, because the tool reads interface pTM from an AlphaFold2 or AlphaFold3
  JSON that a Boltz-2 run does not produce. The Boltz branch simply does not populate what
  those scores need. The constancy was caught by reading the raw output table rather than
  trusting the parsed numbers, and the three columns are excluded with the reason recorded in
  `metadata/design_metrics.csv`. A fourth column, `n0res`, had initially been documented here
  as a count of interface residues; it is in fact the d0 normalisation count and equals the
  aligned chain length, 94 or 198 depending on direction. That description was wrong and has
  been corrected.
- **One GPU run produced data and discarded it.** `--write_full_pae` was enabled without
  extending the copy step that moves outputs from the container to persistent storage, which
  took only structure and confidence files. The PAE matrices were computed and destroyed with
  the container, at a cost of roughly half a dollar. The copy step now takes every output
  class and reports how many files it kept per complex, so the failure cannot recur silently.
- **Rotamer ranking was blind for four mutants**, where the tightest contact is carried by CB,
  an atom that does not move between rotamers. Detected by measuring carboxylate centroid
  spread; one mutant has three near-identical rotamers.
- **Self-consistency is partly circular.** BindCraft uses ProteinMPNN to generate sequences, so
  asking ProteinMPNN whether it agrees is asking the generator about itself. It is not fully
  circular, because BindCraft selects sequences after refolding on AlphaFold2 filters rather
  than at the ProteinMPNN optimum, which is why recovery sits near 0.6 rather than near 1.0.
- **Local convexity, hotspot co-facing and extended-conformation behaviour were never measured.**
  6ARU is the folded conformation and every surface calculation inherits that.

---

## 12. Package contents

| path | content |
|---|---|
| `egfr_challenge1_submission.csv` | the submission: 6 rows, ranked, columns `name` / `sequence` / `molecule_class`, the last set to `single_chain` |
| `METHODS.md` | this document |
| `metadata/design_metrics.csv` | one row per candidate, English keys. Sequences printed only for submitted designs |
| `metadata/design_metrics.json` | the same, nested |
| `metadata/metric_dictionary.csv` | every column: meaning, provenance, and whether it is measured, posed without calibration, or descriptive |
| `metadata/provenance.json` | generator, folding model, pKa model, inverse-folding model, MSA construction, hardware |
| `metadata/structures/` | per submitted design: the AlphaFold2 design model, and Boltz-2 folding models against the human and mouse targets |

Metrics **and sequences** are included for all 23 candidates, submitted or not, so the
selection funnel can be audited end to end: which designs were considered, which were cut, and
on which measurement. The repository linked with this submission is public and contains the
full analysis code, the per-run outputs and the dated working journal, including the two
mistakes recorded in §11.

**Models and tools.** BindCraft 2.0 (`a8d0f200…-dirty`) with AlphaFold2 and ProteinMPNN;
Boltz-2 2.2.0; PROPKA 3; ProteinMPNN `v_48_020`; ESM-2 `esm2_t33_650M_UR50D` (descriptive
only); ipSAE reference implementation from `DunbrackLab/IPSAE`; ColabFold MMseqs2 for target
MSAs; PyMOL open-source for rotamer threading; Biopython and SciPy for geometry and SASA.

ESM-2 pseudo log-likelihood is reported and **enters no ordering**: it measures resemblance to
natural proteins, while the challenge requires novelty, so ranking on it would favour the least
novel designs. On this set it spans −2.72 to −1.93 per residue with a median of −2.21 and flags
no outlier.
