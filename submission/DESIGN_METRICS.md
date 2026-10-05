# Per-design metrics

Companion table to `METHODS.md`. Every value here is also in `metadata/design_metrics.csv` inside the attached archive, together with a dictionary that states for each column what it means, where it came from, and whether it is measured, posed without calibration, or descriptive and excluded from ranking.

Scales: `i_pTM`, `i_pAE`, ipSAE, contact recovery and epitope recovery are on [0,1]. `i_pAE` is better when lower; everything else is better when higher. pKa shifts are in pKa units and are positive when binding favours the acidic pH.

---

## Submitted designs

### Objective 1 — pH selectivity

| rank | design | aa | tier | ΔpKa H409: design model / folding model / mouse | factor | salt bridge | bidentate bottleneck |
|---|---|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | 94 | **robust mechanism** | 2.84 / 2.27 / 2.41 | 5.28 | ASP56 2.52 Å at 128.4° | 3.35 |
| 2 | `l63_987fe804e455bc58_seq1` | 63 | **non-reproducible mechanism** | -0.29 / 0.97 / -0.7 | 0.848 | ASP52 3.21 Å at 119.5° | 4.54 |
| 3 | `l59_36dbfc4737a3e59b_seq1` | 59 | **non-reproducible mechanism** | 0.96 / -0.17 / -0.38 | 2.599 | ASP24 3.08 Å at 85.9° | 3.34 |
| 4 | `l61_cd272a8fd929c7ee_seq0` | 61 | **neutral** | -0.4 / -0.39 / -0.5 | 0.808 | GLU26 12.94 Å at 99.8° | no pair possible |
| 5 | `l58_fd5dae7987a2388d_seq1` | 58 | **neutral** | -0.07 / -0.12 / -0.13 | 0.956 | GLU24 3.31 Å at 159.9° | no pair possible |
| 6 | `l57_9526c9216eb7d6db_seq0` | 57 | **neutral** | 0.03 / -0.41 / -2.62 | 1.022 | ASP32 2.42 Å at 96.1° | no pair possible |

### Objective 2 — mouse cross-reactivity

| rank | design | mouse iptm | Δ iptm vs human | epitope recovered on mouse | mechanism conserved | sequence-identity proxy | divergent contacted residues |
|---|---|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | 0.914 | -0.016 | **0.897** | yes | 0.857 | S418,I467,S468 |
| 2 | `l63_987fe804e455bc58_seq1` | 0.94 | -0.006 | **0.935** | lost in mouse | 0.87 | R353,I467,S468 |
| 3 | `l59_36dbfc4737a3e59b_seq1` | 0.928 | -0.01 | **0.839** | not applicable | 0.81 | S324,S418,K443,S468 |
| 4 | `l61_cd272a8fd929c7ee_seq0` | 0.933 | -0.008 | **0.92** | not applicable | 0.8 | S418,I467,S468 |
| 5 | `l58_fd5dae7987a2388d_seq1` | 0.938 | 0.002 | **0.812** | not applicable | 0.778 | S324,S418,I467,S468 |
| 6 | `l57_9526c9216eb7d6db_seq0` | 0.851 | -0.069 | **0.479** | not applicable | 0.778 | R353,S418,I467,S468 |

### Objective 3 — interface quality, and independent checks

| rank | design | i_pTM | i_pAE | buried area Å² | hotspot frac | off-epitope | ipSAE sym / target→binder | LIS | contact recovery | ProteinMPNN recovery |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | 0.83 | 0.19 | 902.6 | 0.5 | 0.29 | 0.9071 / 0.8309 | 0.7108 | 0.897 | 0.5824 |
| 2 | `l63_987fe804e455bc58_seq1` | 0.83 | 0.19 | 1049.0 | 0.5 | 0.3 | 0.9193 / 0.8126 | 0.7454 | 0.935 | 0.6171 |
| 3 | `l59_36dbfc4737a3e59b_seq1` | 0.77 | 0.24 | 953.8 | 0.5 | 0.24 | 0.9028 / 0.762 | 0.7323 | 0.857 | 0.5466 |
| 4 | `l61_cd272a8fd929c7ee_seq0` | 0.84 | 0.18 | 723.5 | 0.25 | 0.47 | 0.8878 / 0.8044 | 0.7311 | 0.96 | 0.6557 |
| 5 | `l58_fd5dae7987a2388d_seq1` | 0.82 | 0.19 | 916.6 | 0.5 | 0.39 | 0.8771 / 0.739 | 0.7168 | 0.938 | 0.6207 |
| 6 | `l57_9526c9216eb7d6db_seq0` | 0.76 | 0.27 | 722.5 | 0.25 | 0.5 | 0.8391 / 0.704 | 0.6765 | 0.917 | 0.5811 |

### Developability and descriptive

| rank | design | net charge | free Cys | exposed deamidation / isomerisation / DP / oxidation | charge and hydrophobic patches | N-glyc sequons (not scored) | ESM-2 PLL |
|---|---|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | -7.0 | 0.0 | 0 / 1 / 0 / 4 | 0 neg, 1 pos, 0 hydrophobic | 0 | -2.0341 |
| 2 | `l63_987fe804e455bc58_seq1` | -1.0 | 0.0 | 0 / 0 / 1 / 4 | 1 neg, 0 pos, 0 hydrophobic | 2 | -2.259 |
| 3 | `l59_36dbfc4737a3e59b_seq1` | -3.0 | 0.0 | 0 / 0 / 0 / 1 | 0 neg, 2 pos, 0 hydrophobic | 0 | -2.0516 |
| 4 | `l61_cd272a8fd929c7ee_seq0` | -5.0 | 0.0 | 1 / 0 / 0 / 1 | 0 neg, 2 pos, 1 hydrophobic | 0 | -2.0997 |
| 5 | `l58_fd5dae7987a2388d_seq1` | -8.0 | 0.0 | 0 / 0 / 0 / 0 | 1 neg, 0 pos, 0 hydrophobic | 0 | -1.9588 |
| 6 | `l57_9526c9216eb7d6db_seq0` | -7.0 | 0.0 | 1 / 3 / 0 / 2 | 0 neg, 0 pos, 0 hydrophobic | 0 | -1.9272 |

### Note on rank 6

`egfr-dIII-prod02_denovo_l57_9526c9216eb7d6db_seq0`

> Submitted as a negative control for the mouse cross-reactivity prediction: expected to bind human EGFR and not mouse. This design recovers only 0.479 of its human epitope when predicted against the mouse target, the lowest of the submitted set, and it also carries the lowest ipSAE in the target-to-binder direction (0.704) and a mouse pKa shift of -2.62. It is the only submitted design on which this pipeline makes a directional, falsifiable prediction rather than a hope: if it binds human and not mouse, the epitope analysis is validated experimentally; if it binds mouse anyway, a contact recovery of 0.479 does not mean what it was taken to mean. Both outcomes are informative.

---

## Candidates considered and not submitted

17 further candidates were generated and analysed. Sequences and metrics are given so the selection funnel can be checked rather than taken on trust. Designs sharing a backbone hash are not independent poses, and only one design per backbone was ever eligible.

| design | backbone | tier | ΔpKa design / folding / mouse | mouse epitope recovery | i_pTM | reason not submitted |
|---|---|---|---|---|---|---|
| `l64_1e7ab6d8f00c9958_seq0` | `1e7ab6d8f0` | counter-selective | -2.25 / -2.3 / -2.12 | 0.831 | 0.85 | predicted counter-selective on the pH criterion |
| `l64_4a818d7951649b77_seq0` | `4a818d7951` | counter-selective | -1.89 / -2.16 / -2.78 | 0.889 | 0.75 | predicted counter-selective on the pH criterion |
| `l64_4a818d7951649b77_seq1` | `4a818d7951` | counter-selective | -1.85 / -2.47 / -2.52 | 0.785 | 0.73 | predicted counter-selective on the pH criterion |
| `l73_5b295c4d9e1ff73f_seq0` | `5b295c4d9e` | counter-selective | -1.73 / -1.74 / -2.18 | 0.804 | 0.8 | predicted counter-selective on the pH criterion |
| `l73_5b295c4d9e1ff73f_seq1` | `5b295c4d9e` | counter-selective | -1.85 / -2.53 / -1.72 | 0.761 | 0.81 | predicted counter-selective on the pH criterion |
| `l92_5c3ec1903e03c261_seq1` | `5c3ec1903e` | counter-selective | -0.86 / -1.82 / -1.84 | 0.929 | 0.82 | predicted counter-selective on the pH criterion |
| `l92_5c3ec1903e03c261_seq0` | `5c3ec1903e` | counter-selective | -2.36 / -2.96 / -2.95 | 0.911 | 0.81 | predicted counter-selective on the pH criterion |
| `l61_a6334a3a912c86f1_seq0` | `a6334a3a91` | counter-selective | -2.68 / -3.05 / -2.63 | 0.833 | 0.81 | predicted counter-selective on the pH criterion |
| `l61_a6334a3a912c86f1_seq1` | `a6334a3a91` | counter-selective | -2.75 / -2.47 / -2.5 | 0.806 | 0.81 | predicted counter-selective on the pH criterion |
| `l62_a6d2f6834f22e574_seq1` | `a6d2f6834f` | counter-selective | -2.85 / -3.07 / -2.83 | 0.921 | 0.8 | predicted counter-selective on the pH criterion |
| `l62_a6d2f6834f22e574_seq0` | `a6d2f6834f` | counter-selective | -2.72 / -3.57 / -3.16 | 0.812 | 0.81 | predicted counter-selective on the pH criterion |
| `l55_f6d5f550a210fd48_seq0` | `f6d5f550a2` | counter-selective | -2.89 / -2.56 / -2.86 | 0.604 | 0.81 | predicted counter-selective on the pH criterion |
| `l59_36dbfc4737a3e59b_seq0` | `36dbfc4737` | neutral | -1.19 / -0.41 / -0.33 | 0.897 | 0.76 | a design on the same backbone ranked higher |
| `l57_9526c9216eb7d6db_seq1` | `9526c9216e` | neutral | -0.23 / 0.45 / -2.48 | 0.478 | 0.71 | a design on the same backbone ranked higher |
| `l61_cd272a8fd929c7ee_seq1` | `cd272a8fd9` | neutral | -0.4 / -0.36 / -0.37 | 0.9 | 0.81 | a design on the same backbone ranked higher |
| `l58_fd5dae7987a2388d_seq0` | `fd5dae7987` | neutral | -0.01 / -0.11 / -2.65 | 0.806 | 0.82 | a design on the same backbone ranked higher |
| `l63_987fe804e455bc58_seq0` | `987fe804e4` | non-reproducible mechanism | -0.22 / 1.04 / -0.11 | 0.906 | 0.83 | a design on the same backbone ranked higher |

### Sequences of the candidates not submitted

| design | sequence |
|---|---|
| `l55_f6d5f550a210fd48_seq0` | `MIDVSKLSKEELWWLVIEIIAKYNDPEAQALFSNPETSKYSLEELQKEVEKILKK` |
| `l64_1e7ab6d8f00c9958_seq0` | `MELSAEEMADRIIEVARTGDLEEAAEVSKLSWEEVMKIGEETGRVDEVLAAWLIVGSIVKRGKL` |
| `l57_9526c9216eb7d6db_seq1` | `MEILEEEVNGIRVIVDDEGDGNISVYIFSERDIWSPVVPANGKSVKEILEEVKKKLE` |
| `l58_fd5dae7987a2388d_seq0` | `IEVDPNLPVHEQVFLILEKGTKAEFEALRAGNEKEAEEISKEAVEKIEEAFEKEENPE` |
| `l59_36dbfc4737a3e59b_seq0` | `SSAKEKEKKEEEEVLKLKADLSADVYYKRGWFKNEKEKEEFAEKLYKELKEEEEKKKKS` |
| `l61_a6334a3a912c86f1_seq0` | `DEEMKEFAIEWVRMSIEWGKKTGEVERALRIARLSAEQFGERSGSEELLEELYEVIKELTE` |
| `l61_a6334a3a912c86f1_seq1` | `DKEMEEFAEEWVRMSLEWGKKTGDVERAVRIARLSAEQFGERSGSKELLERLEKIIKELTE` |
| `l61_cd272a8fd929c7ee_seq1` | `SEEKKKLTEELIHEIVWDYLNFGSDETLEKYRKRIEEMKKEGKVSEKEYPELEELLEFLSS` |
| `l62_a6d2f6834f22e574_seq0` | `KPTPVEELSEDEWWELYIKATQWIHEEYKNEKNWEEAYEIMKASHSGRKELYVEKAQKALSK` |
| `l62_a6d2f6834f22e574_seq1` | `APTPVEDLSEDEWWELYIKATQWIHEEYKNEKHWEEMYEVMKASHSGRKEEYVAVAQKVLSK` |
| `l63_987fe804e455bc58_seq0` | `MRKPLTKEEIKKLLTSEEYNESGDPWDAFFKLQEKEKMTMEERKMAKDVVFDIQIKKEEEKMK` |
| `l64_4a818d7951649b77_seq0` | `MSEIEKKVLKAKEEYEKFLGGYDLSEEEKYMQAIIFWFPKDGKTISDEEFDKMVEIGIKLGKEE` |
| `l64_4a818d7951649b77_seq1` | `MSEIEKKVLEAIETYEKFLGGYNLSAEEKGMQAIIFWFPKDGKTISDEEFDEMVKVAIELGKKE` |
| `l73_5b295c4d9e1ff73f_seq0` | `MLSKEEKDEAEVKIITWFLSVSHEELEEFIKKHLPPEVQEKLKDPSTDMLELLIEHALKNPELLEAILKLYES` |
| `l73_5b295c4d9e1ff73f_seq1` | `MLSKEEKDEAEVKIITWFLSVSHEELEAFIKENLDEESQEELRDPSTDMLEYLIELALKNKELLEAILKKYNE` |
| `l92_5c3ec1903e03c261_seq0` | `MTKEELWNKIMFEMLWPLPAEERAKRISEIIKEAIEENEKELGKDNEEFKKELEEVLKELEEFLEWTKDSPDWNNPEMKKALEPLKEAIKKM` |
| `l92_5c3ec1903e03c261_seq1` | `MTGEELWNQLMFTMLWPLPPEEIAQKISEIIEKAIKEAKEKYGEDNEEFKKELEKVLEELKEFLEWTKDSPDWSHPAMEEAIKPLEKAIKEM` |
