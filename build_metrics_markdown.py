#!/usr/bin/env python3
"""Per-design metrics as readable Markdown, for a reviewer that reads text rather than CSV.

    uv run python build_metrics_markdown.py

WHY A MARKDOWN COPY OF DATA THAT ALREADY EXISTS AS CSV. The submission form accepts `.md` and
`.zip` attachments and states that reviewers read them when choosing which designs go to the
lab. A CSV inside a zip may or may not be opened; a Markdown table is read directly. The
machine-readable copies stay in the zip for anything that wants to parse them.

Output: submission/DESIGN_METRICS.md
"""

from __future__ import annotations

import csv
from pathlib import Path

SOURCE = Path("submission/metadata/design_metrics.csv")
OUT = Path("submission/DESIGN_METRICS.md")


def short(name: str) -> str:
    return name.split("_denovo_")[1] if "_denovo_" in name else name


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open(newline="")))
    sent = sorted((r for r in rows if r["submitted"] == "yes"),
                  key=lambda r: int(r["rank_submitted"]))
    held = [r for r in rows if r["submitted"] == "no"]

    out = ["# Per-design metrics",
           "",
           "Companion table to `METHODS.md`. Every value here is also in "
           "`metadata/design_metrics.csv` inside the attached archive, together with a "
           "dictionary that states for each column what it means, where it came from, and "
           "whether it is measured, posed without calibration, or descriptive and excluded "
           "from ranking.",
           "",
           "Scales: `i_pTM`, `i_pAE`, ipSAE, contact recovery and epitope recovery are on "
           "[0,1]. `i_pAE` is better when lower; everything else is better when higher. "
           "pKa shifts are in pKa units and are positive when binding favours the acidic pH.",
           "",
           "---",
           "",
           "## Submitted designs",
           "",
           "### Objective 1 — pH selectivity",
           "",
           "| rank | design | aa | tier | ΔpKa H409: design model / folding model / mouse "
           "| factor | salt bridge | bidentate bottleneck |",
           "|---|---|---|---|---|---|---|---|"]
    for r in sent:
        bridge = (f"{r['salt_bridge_residue']} {r['salt_bridge_distance_A']} Å "
                  f"at {r['salt_bridge_angle_deg']}°"
                  if r["salt_bridge_residue"] else "none")
        bident = r["bidentate_bottleneck_A"] or "no pair possible"
        out.append(f"| {r['rank_submitted']} | `{short(r['design_id'])}` | "
                   f"{r['length_aa']} | **{r['ph_mechanism_tier']}** | "
                   f"{r['dpka_h409_design_model']} / {r['dpka_h409_folding_model']} / "
                   f"{r['dpka_h409_mouse']} | {r['ph_selectivity_factor']} | {bridge} | "
                   f"{bident} |")

    out += ["", "### Objective 2 — mouse cross-reactivity", "",
            "| rank | design | mouse iptm | Δ iptm vs human | epitope recovered on mouse "
            "| mechanism conserved | sequence-identity proxy | divergent contacted residues |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sent:
        out.append(f"| {r['rank_submitted']} | `{short(r['design_id'])}` | "
                   f"{r['mouse_iptm']} | {r['mouse_delta_iptm']} | "
                   f"**{r['mouse_epitope_recovery']}** | "
                   f"{r['mouse_mechanism_conserved']} | "
                   f"{r['mouse_epitope_identity_fraction']} | "
                   f"{r['mouse_divergent_contacted_residues'] or 'none'} |")

    out += ["", "### Objective 3 — interface quality, and independent checks", "",
            "| rank | design | i_pTM | i_pAE | buried area Å² | hotspot frac | off-epitope "
            "| ipSAE sym / target→binder | LIS | contact recovery | ProteinMPNN recovery |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sent:
        out.append(f"| {r['rank_submitted']} | `{short(r['design_id'])}` | "
                   f"{r['design_model_iptm']} | {r['design_model_ipae']} | "
                   f"{r['interface_buried_area_A2']} | {r['hotspot_contact_fraction']} | "
                   f"{r['off_epitope_contact_fraction']} | {r['ipsae_symmetrised']} / "
                   f"{r['ipsae_target_to_binder']} | {r['lis']} | "
                   f"{r['contact_recovery_vs_design']} | {r['mpnn_sequence_recovery']} |")

    out += ["", "### Developability and descriptive", "",
            "| rank | design | net charge | free Cys | exposed deamidation / isomerisation / "
            "DP / oxidation | charge and hydrophobic patches | N-glyc sequons (not scored) "
            "| ESM-2 PLL |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sent:
        deg = (f"{r['liability_exposed_deamidation']} / "
               f"{r['liability_exposed_isomerisation']} / "
               f"{r['liability_dp_cleavage']} / {r['liability_exposed_oxidation']}")
        patch = (f"{r['liability_negative_patches']} neg, "
                 f"{r['liability_positive_patches']} pos, "
                 f"{r['liability_hydrophobic_patches']} hydrophobic")
        out.append(f"| {r['rank_submitted']} | `{short(r['design_id'])}` | "
                   f"{r['net_charge']} | {r['free_cysteines']} | {deg} | {patch} | "
                   f"{r['n_glyc_sequons_not_scored']} | {r['esm2_pseudo_log_likelihood']} |")

    note = next((r for r in sent if r["submission_note"]), None)
    if note:
        out += ["", "### Note on rank " + note["rank_submitted"], "",
                f"`{note['design_id']}`", "", "> " + note["submission_note"]]

    out += ["", "---", "",
            "## Candidates considered and not submitted",
            "",
            f"{len(held)} further candidates were generated and analysed. Sequences and "
            "metrics are given so the selection funnel can be checked rather than taken on "
            "trust. Designs sharing a backbone hash are not independent poses, and only one "
            "design per backbone was ever eligible.",
            "",
            "| design | backbone | tier | ΔpKa design / folding / mouse | mouse epitope "
            "recovery | i_pTM | reason not submitted |",
            "|---|---|---|---|---|---|---|"]
    order = {"counter-selective": 0, "neutral": 1, "non-reproducible mechanism": 2,
             "robust mechanism": 3}
    for r in sorted(held, key=lambda r: (order.get(r["ph_mechanism_tier"], 9),
                                         r["backbone_id"])):
        reason = ("predicted counter-selective on the pH criterion"
                  if r["ph_mechanism_tier"] == "counter-selective"
                  else "a design on the same backbone ranked higher")
        out.append(f"| `{short(r['design_id'])}` | `{r['backbone_id'][:10]}` | "
                   f"{r['ph_mechanism_tier']} | {r['dpka_h409_design_model']} / "
                   f"{r['dpka_h409_folding_model'] or '—'} / "
                   f"{r['dpka_h409_mouse'] or '—'} | "
                   f"{r['mouse_epitope_recovery'] or '—'} | {r['design_model_iptm']} | "
                   f"{reason} |")

    out += ["", "### Sequences of the candidates not submitted", "",
            "| design | sequence |", "|---|---|"]
    for r in sorted(held, key=lambda r: r["design_id"]):
        out.append(f"| `{short(r['design_id'])}` | `{r['sequence']}` |")
    out.append("")

    OUT.write_text("\n".join(out))
    print(f"-> {OUT}  ({OUT.stat().st_size / 1024:.1f} Ko, "
          f"{len(sent)} soumis, {len(held)} non soumis)")


if __name__ == "__main__":
    main()
