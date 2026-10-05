#!/usr/bin/env python3
"""Metadata package for submission: English-keyed metrics, dictionary, provenance, structures.

    uv run --with openpyxl python build_metadata_package.py

WHY THIS EXISTS. The challenge text invites participants to attach metrics, structure models
from both design and folding models, methodologies and provenance, and notes that for
Tracks 2 and 3 this additional information is given to Claude to help select designs. This
script assembles that material in a form a model can read without guessing: explicit units,
explicit scales, one row per design, and a dictionary that states for every column what it
means, where it came from, and whether it is measured, posed without calibration, or purely
descriptive.

WHY ENGLISH HERE AND FRENCH ELSEWHERE. The analysis scripts and the repository journal are in
French, which is the working language of the project. This package is read by the selection
workflow, so its keys and its dictionary are in English. The mapping between the two lives in
one place — the FIELDS table below — so the two conventions cannot drift apart.

WITHHELD SEQUENCES. All submitted data may be made public, and further challenge windows run
through 1 November. The package therefore reports metrics for every candidate, so the funnel
is auditable, but prints the amino-acid sequence only for designs that are actually
submitted. Withheld designs carry the literal value `withheld` in the sequence column. This
is a deliberate choice and it is stated in the dictionary rather than hidden.

NO EMBEDDED INSTRUCTIONS. Nothing in this package addresses a reader or asks anything of one.
The challenge text states that embedded instructions or prompt injection may be grounds for
disqualification. Every file here is descriptive.

Output: submission/metadata/
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

OUT_DIR = Path("submission/metadata")
STRUCTURE_DIR = OUT_DIR / "structures"

MASTER = Path("out/master_rank.csv")
SUBMISSION = Path("submission/egfr_challenge1_submission.csv")
RANKED = {
    "egfr-dIII-prod01": Path("out/egfr-dIII-prod01/3_Ranked/!_Ranked.csv"),
    "egfr-dIII-prod02": Path("out/egfr-dIII-prod02/3_Ranked/!_Ranked.csv"),
}
EXTRA = {
    "boltz": Path("out/boltz_contacts.csv"),
    "cross": Path("out/cross_species.csv"),
    "ipsae": Path("out/ipsae.csv"),
    "mpnn": Path("out/self_consistency.csv"),
    "liabilities": Path("out/sequence_liabilities.csv"),
    "geometry": Path("out/geometry_wt.csv"),
    "bidentate": Path("out/bidentate.csv"),
    "bidentate_boltz": Path("out/bidentate_boltz.csv"),
    "esm2": Path("out/esm2_pll.csv"),
}
CAMPAIGNS = {
    "egfr-dIII-prod01": Path("out/egfr-dIII-prod01/campaign_metadata.json"),
    "egfr-dIII-prod02": Path("out/egfr-dIII-prod02/campaign_metadata.json"),
}
AF2_STRUCTURES = Path("structures/wt")
BOLTZ_HUMAN = Path("out/rescore01")
BOLTZ_MOUSE = Path("out/mouse01")

WITHHELD = "withheld"

# NOTES DE SOUMISSION, par design. Elles decrivent l'intention derriere l'inclusion d'un
# design et n'adressent rien a personne.
SUBMISSION_NOTES = {
    "egfr-dIII-prod02_denovo_l57_9526c9216eb7d6db_seq0":
        "Submitted as a negative control for the mouse cross-reactivity prediction: "
        "expected to bind human EGFR and not mouse. This design recovers only 0.479 of its "
        "human epitope when predicted against the mouse target, the lowest of the submitted "
        "set, and it also carries the lowest ipSAE in the target-to-binder direction (0.704) "
        "and a mouse pKa shift of -2.62. It is the only submitted design on which this "
        "pipeline makes a directional, falsifiable prediction rather than a hope: if it binds "
        "human and not mouse, the epitope analysis is validated experimentally; if it binds "
        "mouse anyway, a contact recovery of 0.479 does not mean what it was taken to mean. "
        "Both outcomes are informative.",
}

# Les colonnes enumerees viennent des CSV francais du depot. Elles sont traduites ICI, en un
# seul endroit, pour que le paquet soit entierement en anglais sans dupliquer la logique.
VALUE_MAP = {
    "mecanisme robuste": "robust mechanism",
    "mecanisme non reproductible": "non-reproducible mechanism",
    "neutre": "neutral",
    "contre-selectif": "counter-selective",
    "robuste": "robust",
    "non reproductible": "non-reproducible",
    "absent": "absent",
    "confirmee": "confirmed",
    "contestee": "contested",
    "non mesuree": "not measured",
    "non mesure": "not measured",
    "non mesure (mutant)": "not measured (mutant)",
    "oui": "yes",
    "non": "no",
    "perdu chez la souris": "lost in mouse",
    "sans objet": "not applicable",
    "propre": "own",
    "parent": "parent",
    "aucune paire possible": "no pair possible",
}

# (english_key, source_table, source_column, meaning, provenance, status)
#
# `status` takes one of three values and nothing else:
#   measured                      computed from a structure or a sequence
#   posed, not calibrated         a threshold chosen without a reference set to calibrate it
#   descriptive, not ranked       reported, entering no ordering
FIELDS: list[tuple[str, str, str, str, str, str]] = [
    ("design_id", "master", "design_id",
     "unique identifier; encodes campaign, modality, binder length, backbone hash and "
     "ProteinMPNN sequence index",
     "BindCraft 2.0", "measured"),
    ("rank_submitted", "", "",
     "position in the submitted CSV, 1 is highest; empty when the design is not submitted",
     "this pipeline", "measured"),
    ("submitted", "", "",
     "whether this design appears in the submitted CSV",
     "this pipeline", "measured"),
    ("submission_note", "", "",
     "the intent behind including this design, where it differs from the default of "
     "submitting a design expected to bind. Empty when the design is submitted on its "
     "predicted merits alone",
     "this pipeline", "measured"),
    ("backbone_id", "master", "squelette",
     "BindCraft trajectory hash. Two designs sharing it share a backbone and are NOT "
     "independent poses",
     "BindCraft 2.0", "measured"),
    ("mpnn_index", "master", "index_mpnn",
     "ProteinMPNN sequence index on that backbone (seq0 or seq1). With kept_sequences=2 two "
     "sequences are kept per backbone; neither is derived from the other",
     "BindCraft 2.0", "measured"),
    ("campaign", "master", "run", "generation campaign", "BindCraft 2.0", "measured"),
    ("sequence", "master", "sequence",
     "binder amino-acid sequence; the literal value 'withheld' for designs that are not "
     "submitted",
     "BindCraft 2.0", "measured"),
    ("length_aa", "master", "longueur", "number of binder residues", "derived", "measured"),
    ("net_charge", "master", "charge_nette", "net binder charge",
     "BindCraft 2.0", "measured"),
    ("free_cysteines", "master", "cysteines",
     "number of cysteines; zero across the whole set, which matters because expression is "
     "cell-free and free cysteines are an expression liability",
     "BindCraft 2.0", "measured"),

    # --- design model, AlphaFold2 through BindCraft ---
    ("design_model_iptm", "master", "i_pTM_AF2",
     "AlphaFold2 interface pTM on scale [0,1]. IN-SAMPLE: BindCraft optimises designs by "
     "gradient descent through AlphaFold2, so this is a training score, not a test score",
     "BindCraft 2.0 / AlphaFold2", "measured"),
    ("design_model_ipae", "master", "i_pAE_AF2",
     "AlphaFold2 interface predicted aligned error on scale [0,1], lower is better. Also "
     "in-sample",
     "BindCraft 2.0 / AlphaFold2", "measured"),
    ("interface_buried_area_A2", "master", "Interface_BuriedArea",
     "buried surface area at the interface, square angstroms",
     "BindCraft 2.0", "measured"),
    ("hotspot_contact_fraction", "master", "Hotspot_Contact_Fraction",
     "fraction of contacts touching the requested hotspots A318, A323, A406, A409",
     "BindCraft 2.0", "measured"),
    ("off_epitope_contact_fraction", "master", "Off_Epitope",
     "fraction of contacts outside the intended epitope",
     "BindCraft 2.0", "measured"),

    # --- folding model, Boltz-2 ---
    ("folding_model_iptm", "master", "confiance_modele_orthogonal",
     "Boltz-2 interface pTM, mean over three diffusion samples. Boltz-2 never took part in "
     "generating these designs",
     "Boltz-2 2.2.0", "measured"),
    ("contact_recovery_vs_design", "master", "recuperation_contacts",
     "fraction of residue-residue contact pairs of the design model that are recovered in "
     "the folding model, best of three samples. Contact means any heavy-atom pair within "
     "5.0 angstroms. Asymmetric on purpose: the question is whether the design pose is "
     "recovered, not whether the two poses are identical",
     "Boltz-2 + geometry", "measured"),
    ("contact_recovery_min", "boltz", "recuperation_min",
     "the same quantity on the worst of the three samples", "Boltz-2 + geometry", "measured"),
    ("pose_verdict", "master", "pose",
     "whether contact recovery and folding-model iptm both clear their thresholds. These "
     "thresholds separate nothing on this set: all 23 candidates clear them, so this column "
     "carries no ranking information",
     "derived", "posed, not calibrated"),
    ("ipsae_symmetrised", "ipsae", "ipSAE_moyen",
     "interface predicted aligned error score, mean over samples, computed by the reference "
     "implementation of its authors at PAE cutoff 10 and distance cutoff 15 angstroms. "
     "Unlike iptm it is renormalised on interface residues, so a small binder against a "
     "large target is not diluted",
     "DunbrackLab/IPSAE on Boltz-2 output", "measured"),
    ("ipsae_target_to_binder", "ipsae", "ipSAE_A_vers_B_moyen",
     "the same score in the target-to-binder direction, which is the more demanding of the "
     "two for a small binder. Reported alongside the symmetrised value so the choice of the "
     "maximum is visible rather than implicit",
     "DunbrackLab/IPSAE on Boltz-2 output", "measured"),
    ("lis", "ipsae", "LIS_moyen", "local interaction score, mean over samples",
     "DunbrackLab/IPSAE on Boltz-2 output", "measured"),
    ("ipsae_columns_discarded", "ipsae", "colonnes_ecartees",
     "columns the reference implementation computes but which are not usable from a Boltz-2 "
     "input, with the reason for each. ipTM_af reads 0.000 because the tool expects an "
     "AlphaFold2 or AlphaFold3 JSON; pDockQ and pDockQ2 are constant across all designs "
     "because the Boltz branch does not supply what they need; n0res is the d0 "
     "normalisation count, equal to the aligned chain length, and is not a count of "
     "interface residues. Verified by reading the raw output rather than assumed",
     "this pipeline", "measured"),

    # --- self-consistency ---
    ("mpnn_sequence_recovery", "mpnn", "recuperation_moyenne",
     "fraction of positions at which ProteinMPNN, resampling the binder on its own design "
     "backbone with the target held fixed, reproduces the submitted residue; mean over 8 "
     "samples at temperature 0.1. PARTLY CIRCULAR, because BindCraft itself uses ProteinMPNN "
     "to generate sequences. It is not fully circular: BindCraft selects sequences after "
     "refolding on AlphaFold2 filters rather than on the ProteinMPNN optimum, which is why "
     "recovery sits near 0.6 instead of near 1.0",
     "ProteinMPNN v_48_020", "measured"),
    ("mpnn_sequence_recovery_min", "mpnn", "recuperation_min",
     "the same quantity on the worst of the samples", "ProteinMPNN v_48_020", "measured"),

    # --- objective 1: pH selectivity ---
    ("ph_mechanism_tier", "master", "palier_pH",
     "one of four tiers: robust mechanism, non-reproducible mechanism, neutral, "
     "counter-selective. A mechanism counts as robust only when the pKa shift is positive on "
     "all three independent measurements",
     "derived from three pKa computations", "measured"),
    ("ph_mechanism_measurements_positive", "master", "mesures_pH_positives",
     "how many of the available pKa-shift measurements are positive",
     "derived", "measured"),
    ("dpka_h409_design_model", "master", "dpKa_H409",
     "shift of the pKa of target histidine H409 between bound and unbound target, computed on "
     "the AlphaFold2 design model. Positive means binding stabilises the protonated form, so "
     "binding is favoured at acidic pH. Negative means the opposite of the objective",
     "PROPKA 3 on design model", "measured"),
    ("dpka_h409_folding_model", "master", "dpKa_humain_Boltz",
     "the same shift computed on the Boltz-2 model of the same complex. Tests whether the "
     "mechanism is a property of the sequence or of one predicted structure",
     "PROPKA 3 on folding model", "measured"),
    ("dpka_h409_mouse", "master", "dpKa_souris_Boltz",
     "the same shift on the complex with the mouse domain III",
     "PROPKA 3 on folding model", "measured"),
    ("ph_selectivity_factor", "master", "facteur_pH_predit",
     "predicted ratio of affinity at pH 6.5 over pH 7.4 arising from the H409 pKa shift "
     "alone; 1.0 means no selectivity. PROPKA is commonly wrong by about one pKa unit, so "
     "this ranks designs and is not claimed as a value",
     "derived from the pKa shift", "measured"),
    ("ph_selectivity_factor_all_groups", "master", "facteur_global_tous_groupes",
     "the same ratio taken over every ionisable group in the complex rather than H409 alone, "
     "which is the unbiased form of the proton-ligand linkage. It assumes independent sites",
     "derived from PROPKA 3", "measured"),
    ("salt_bridge_residue", "master", "pont_residu",
     "binder acidic residue closest to an imidazole nitrogen of H409",
     "geometry on design model", "measured"),
    ("salt_bridge_distance_A", "master", "pont_distance_A",
     "distance from that carboxylate to the nearest imidazole nitrogen, angstroms",
     "geometry on design model", "measured"),
    ("salt_bridge_angle_deg", "master", "pont_angle_deg",
     "angle at the accepting oxygen. An sp2 carboxylate points its lone pairs near 120 "
     "degrees, so values near 80 or 160 are poor geometries",
     "geometry on design model", "measured"),
    ("bidentate_bottleneck_A", "master", "bidente_goulot_A",
     "worst of the two distances in the best pair of TWO DISTINCT binder carboxylates, one on "
     "ND1 and one on NE2 of H409. Measured on the design model",
     "geometry on design model", "measured"),
    ("bidentate_bottleneck_folding_model_A", "bidentate_boltz", "bidente_goulot_A",
     "the same quantity measured on the Boltz-2 model. The two are reported separately "
     "because the rule they support does not survive transferring geometry between "
     "predictors",
     "geometry on folding model", "measured"),

    # --- objective 2: mouse cross-reactivity ---
    ("mouse_epitope_recovery", "master", "epitope_souris_retrouve",
     "fraction of the human design-model contact pairs recovered when the same binder is "
     "predicted against the mouse domain III. Structural measurement of cross-reactivity",
     "Boltz-2 + geometry", "measured"),
    ("mouse_iptm", "master", "iptm_souris",
     "Boltz-2 interface pTM against the mouse target", "Boltz-2 2.2.0", "measured"),
    ("mouse_delta_iptm", "master", "delta_iptm_souris",
     "mouse iptm minus human iptm; near zero means binding confidence does not suffer from "
     "the change of species",
     "Boltz-2 2.2.0", "measured"),
    ("mouse_mechanism_conserved", "master", "mecanisme_conserve_souris",
     "whether a pH mechanism seen on the human target persists on the mouse target",
     "derived", "measured"),
    ("mouse_epitope_identity_fraction", "master", "epitope_conservation_frac",
     "SEQUENCE PROXY: fraction of contacted target residues identical in mouse. Demonstrably "
     "insufficient on this set, since it does not separate the designs that change binding "
     "mode on the mouse target",
     "P00533 / Q01279 alignment", "measured"),
    ("mouse_divergent_contacted_residues", "master", "epitope_residus_divergents",
     "contacted target residues that are not identical in mouse",
     "P00533 / Q01279 alignment", "measured"),

    # --- liabilities ---
    ("liability_exposed_deamidation", "liabilities", "desamidation_exposee",
     "solvent-exposed deamidation motifs; NG strongest, then NS, NT, NN, NA, NH. Buried "
     "motifs are counted but not scored, because water does not reach them",
     "motif scan weighted by side-chain SASA", "measured"),
    ("liability_exposed_isomerisation", "liabilities", "isomerisation_exposee",
     "solvent-exposed isomerisation motifs; DG strongest, then DS, DT, DD",
     "motif scan weighted by side-chain SASA", "measured"),
    ("liability_dp_cleavage", "liabilities", "clivage_DP_expose",
     "exposed DP motifs, acid hydrolysis. The assay runs at pH 6.5, closer to the regime "
     "where this matters than a neutral assay would be",
     "motif scan weighted by side-chain SASA", "measured"),
    ("liability_exposed_oxidation", "liabilities", "oxydation_exposee",
     "exposed methionine and tryptophan", "motif scan weighted by side-chain SASA",
     "measured"),
    ("liability_negative_patches", "liabilities", "plaque_negative",
     "sliding windows of 5 residues with at least 4 exposed acidic residues",
     "sequence and SASA", "measured"),
    ("liability_positive_patches", "liabilities", "plaque_positive",
     "the same for basic residues", "sequence and SASA", "measured"),
    ("liability_hydrophobic_patches", "liabilities", "plaque_hydrophobe",
     "the same for exposed hydrophobic residues", "sequence and SASA", "measured"),
    ("liability_repeats", "liabilities", "repetition",
     "runs of four or more identical consecutive residues", "sequence", "measured"),
    ("n_glyc_sequons_not_scored", "liabilities", "sequons_N_glyc_non_retenus",
     "N-X-S/T sequons with X not proline. NOT scored as a liability here, because expression "
     "is cell-free and therefore carries no glycosylation machinery. Counted because the "
     "motif would become a liability in a eukaryotic system",
     "sequence", "descriptive, not ranked"),

    # --- descriptive ---
    ("esm2_pseudo_log_likelihood", "master", "ESM2_PLL",
     "ESM-2 pseudo log-likelihood per residue, masked marginals, esm2_t33_650M_UR50D. "
     "EXCLUDED FROM EVERY ORDERING by construction: it measures resemblance to natural "
     "proteins, while the challenge requires novelty, so ranking on it would favour the "
     "least novel designs. Read only as an outlier detector, and on this set it flags nothing",
     "ESM-2 650M", "descriptive, not ranked"),
]


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""))) if path.is_file() else []


def main() -> None:
    master = rows(MASTER)
    if not master:
        raise SystemExit(f"{MASTER} missing - run rank_designs.py first")
    submitted = {r["name"]: index for index, r in enumerate(rows(SUBMISSION), 1)}

    tables: dict[str, dict[str, dict]] = {}
    for name, path in EXTRA.items():
        key = "design" if name != "liabilities" and name != "esm2" else (
            "design_id" if name == "liabilities" else "design_id"
        )
        table = {}
        for record in rows(path):
            identifier = record.get("design") or record.get("design_id")
            if identifier:
                table[identifier] = record
        tables[name] = table
    tables["master"] = {r["design_id"]: r for r in master}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    designs = []
    for record in master:
        if record["type"] != "bindcraft":
            continue
        design = record["design_id"]
        rank = submitted.get(design)
        row = {}
        for key, table, column, *_ in FIELDS:
            if key == "rank_submitted":
                row[key] = rank or ""
                continue
            if key == "submitted":
                row[key] = "yes" if rank else "no"
                continue
            if key == "submission_note":
                row[key] = SUBMISSION_NOTES.get(design, "") if rank else ""
                continue
            if key == "sequence":
                row[key] = record["sequence"] if rank else WITHHELD
                continue
            source = tables.get(table, {}).get(design, {})
            value = source.get(column, "")
            row[key] = VALUE_MAP.get(str(value).strip(), value)
        designs.append(row)

    designs.sort(key=lambda r: (r["rank_submitted"] == "", r["rank_submitted"] or 0))

    with (OUT_DIR / "design_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f[0] for f in FIELDS])
        writer.writeheader()
        writer.writerows(designs)
    (OUT_DIR / "design_metrics.json").write_text(json.dumps(designs, indent=2))

    with (OUT_DIR / "metric_dictionary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column", "meaning", "provenance", "status",
                         "source_file_in_repository", "source_column"])
        for key, table, column, meaning, provenance, status in FIELDS:
            writer.writerow([key, meaning, provenance, status,
                             str(EXTRA.get(table, MASTER if table == "master" else "")),
                             column])

    # --- provenance, read from what the campaigns actually recorded ---
    provenance = {
        "challenge": "Adaptyv x Anthropic Protein Design Competition, Challenge 1, Track 3",
        "target": {
            "human": "UniProt P00533, extracellular region, domain III",
            "mouse": "UniProt Q01279, domain III, UniProt 333-530",
            "structure": "PDB 6ARU chain A, folded conformation, 3.20 A",
            "target_file": "inputs/6ARU_A_309-506.pdb, 198 residues, PDB numbering 309-506",
            "domain_boundaries": "CATH-Gene3D G3DSA:3.80.20.20",
            "human_mouse_identity_domain_III": "173/198 = 87.4 percent, no indels",
        },
        "generator": {},
        "folding_model": {
            "name": "Boltz-2", "version": "2.2.0",
            "diffusion_samples": 3, "recycling_steps": 3,
            "target_msa": "ColabFold MMseqs2, mode env, precomputed once and baked into the "
                          "image; 3725 sequences for the human target, 3502 for the mouse",
            "binder_msa": "empty, single-sequence mode, because the binders are de novo and "
                          "have no natural homologues",
            "affinity_module": "not used; it is calibrated for small molecules rather than "
                               "protein-protein interfaces",
            "note": "requires --no_kernels, because boltz 2.2.0 calls a cuEquivariance "
                    "kernel that pip install boltz does not provide, and otherwise exits "
                    "with status 0 having produced nothing",
        },
        "pka_model": {"name": "PROPKA", "version": "3",
                      "states": "complex, target alone, binder alone, all extracted from the "
                                "same file so the comparison isolates the partner"},
        "inverse_folding_model": {"name": "ProteinMPNN", "weights": "v_48_020",
                                  "temperature": 0.1, "samples_per_design": 8},
        "sequence_model": {"name": "ESM-2", "checkpoint": "esm2_t33_650M_UR50D",
                           "use": "descriptive only, excluded from every ordering"},
        "hardware": "NVIDIA L40S on Modal for the folding model; local CPU for pKa, "
                    "geometry, threading and liabilities",
        "reproducibility_caveat":
            "Replaying the same commit does not reproduce the same designs. JAX GPU "
            "reductions are not bit-deterministic and a gradient trajectory is chaotic. "
            "What is reproducible is the method, not the sequences.",
    }
    for name, path in CAMPAIGNS.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        settings = payload.get("settings", {})
        provenance["generator"][name] = {
            "name": payload.get("version"),
            "revision": payload.get("revision"),
            "revision_caveat":
                "the -dirty suffix means the working tree in the container differed from the "
                "pinned commit, so that hash does not fully pin the code state",
            "settings_digest": payload.get("settings_digest"),
            "targets": settings.get("targets"),
            "binder_lengths": settings.get("binder_lengths"),
            "max_trajectories": settings.get("max_trajectories"),
            "kept_sequences": settings.get("kept_sequences"),
            "number_of_final_designs": settings.get("number_of_final_designs"),
        }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # --- structures, only for submitted designs ---
    if STRUCTURE_DIR.exists():
        shutil.rmtree(STRUCTURE_DIR)
    copied = 0
    for design, rank in submitted.items():
        folder = STRUCTURE_DIR / f"{rank:02d}_{design}"
        folder.mkdir(parents=True, exist_ok=True)
        source = AF2_STRUCTURES / f"{design}.pdb"
        if source.is_file():
            shutil.copy(source, folder / "design_model_alphafold2.pdb")
            copied += 1
        for origin, label in ((BOLTZ_HUMAN, "folding_model_boltz2_human"),
                              (BOLTZ_MOUSE, "folding_model_boltz2_mouse")):
            directory = origin / design
            if not directory.is_dir():
                continue
            for index, structure in enumerate(sorted(directory.glob("*.cif"))):
                shutil.copy(structure, folder / f"{label}_sample{index}.cif")
                copied += 1

    print(f"-> {OUT_DIR}")
    print(f"   design_metrics.csv       {len(designs)} designs x {len(FIELDS)} columns")
    print(f"   design_metrics.json")
    print(f"   metric_dictionary.csv    {len(FIELDS)} columns documented")
    print(f"   provenance.json")
    print(f"   structures/              {copied} files for "
          f"{len(submitted)} submitted designs")
    print()
    filled = {
        key: sum(1 for d in designs if str(d[key]).strip() not in {"", "non mesure"})
        for key, *_ in FIELDS
    }
    missing = [key for key, count in filled.items() if count == 0]
    print(f"   columns with no data at all : {missing or 'none'}")
    partial = [f"{k} ({v}/{len(designs)})" for k, v in filled.items()
               if 0 < v < len(designs)]
    print(f"   partially filled           : {partial or 'none'}")


if __name__ == "__main__":
    main()
