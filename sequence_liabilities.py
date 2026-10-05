#!/usr/bin/env python3
"""Liabilités de séquence, pondérées par l'exposition réelle au solvant.

    uv run --with biopython python sequence_liabilities.py

POURQUOI. Le règlement nomme les « sequence liability scores » parmi les pièces encouragées.
Et Adaptyv exprime en acellulaire puis mesure par BLI/SPR : un design qui ne s'exprime pas,
s'agrège ou se dégrade produit zéro information, quelle que soit son affinité prédite.

POURQUOI PONDÉRER PAR LA SASA. Un motif `NG` enfoui dans le cœur hydrophobe n'est pas une
liabilité : l'eau n'y accède pas. Le même motif en surface l'est. Un simple balayage de
motifs produirait donc des faux positifs en masse. On calcule l'exposition relative de chaque
site signalé sur la structure prédite du binder SEUL — une liabilité de stockage concerne la
protéine libre, pas l'interface.

CE QUI COMPTE DANS CE CONTEXTE PRÉCIS, ET CE QUI NE COMPTE PAS :

- **cystéine libre** : liabilité dure, pontage non voulu et dimérisation.
- **désamidation** (`NG` fort, puis `NS`/`NT`/`NN`/`NA`/`NH`) et **isomérisation** (`DG` fort,
  puis `DS`/`DT`/`DD`) : dégradation en stockage.
- **clivage `DP`** : hydrolyse acide. L'essai est à **pH 6,5**, donc plus proche du régime où
  `DP` compte qu'un essai à pH neutre.
- **oxydation** de Met et Trp exposés : dégradation.
- **séquon de N-glycosylation** `N-X-[ST]`, X ≠ P : **PAS une liabilité ici.** L'expression
  est acellulaire, donc sans machinerie de glycosylation. Compté et rapporté pour information,
  parce qu'il redeviendrait une liabilité en système eucaryote, mais exclu du score.
- **plaques de charge et plaques hydrophobes exposées** : liaison non spécifique, agrégation.

AUCUN SCORE COMPOSITE PONDÉRÉ. On compte par catégorie plutôt que d'inventer un poids entre
« une désamidation exposée » et « une plaque hydrophobe ». Seul `n_liabilites_dures` agrège,
et il ne contient que du bloquant.

Sortie : out/sequence_liabilities.csv, out/sequence_liabilities_sites.csv
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

from Bio.PDB import PDBParser, ShrakeRupley

warnings.filterwarnings("ignore")

STRUCTURES = Path("structures/wt")
MASTER = Path("out/master_rank.csv")
OUT = Path("out/sequence_liabilities.csv")
OUT_SITES = Path("out/sequence_liabilities_sites.csv")

BINDER_CHAIN = "B"

# Au-dela de cette fraction de SASA de reference, une chaine laterale est tenue pour
# accessible. 0,20-0,25 est la convention usuelle ; on prend la borne basse, plus inclusive
# donc plus prudente pour un signalement.
EXPOSED_REL_SASA = 0.20

# SASA de chaine laterale isolee, tripeptide Gly-X-Gly etendu, Tien et al. 2013.
SIDECHAIN_REF_A2 = {
    "ALA": 67.0, "ARG": 196.0, "ASN": 113.0, "ASP": 106.0, "CYS": 104.0,
    "GLN": 144.0, "GLU": 138.0, "GLY": 1.0, "HIS": 151.0, "ILE": 140.0,
    "LEU": 137.0, "LYS": 167.0, "MET": 160.0, "PHE": 175.0, "PRO": 105.0,
    "SER": 80.0, "THR": 102.0, "TRP": 217.0, "TYR": 187.0, "VAL": 117.0,
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Le premier residu du motif est le site reactif, donc celui dont l'exposition decide.
MOTIFS = [
    ("desamidation", "forte", r"N[G]"),
    ("desamidation", "moderee", r"N[STNAH]"),
    ("isomerisation", "forte", r"D[G]"),
    ("isomerisation", "moderee", r"D[STD]"),
    ("clivage_DP", "forte", r"DP"),
]
OXIDISABLE = {"M": "moderee", "W": "forte"}

# 5 residus : l'echelle d'un tour et demi d'helice, assez court pour localiser une plaque,
# assez long pour ne pas declencher sur un doublet.
PATCH_WINDOW = 5
PATCH_MIN = 4
HYDROPHOBIC = set("AVILMFWY")
POSITIVE = set("KR")
NEGATIVE = set("DE")
REPEAT_MIN = 4


def exposure(design: str) -> dict[int, float] | None:
    path = STRUCTURES / f"{design}.pdb"
    if not path.is_file():
        return None
    structure = PDBParser(QUIET=True).get_structure(design, str(path))
    model = structure[0]
    for chain in [c.id for c in model]:
        if chain != BINDER_CHAIN:
            model.detach_child(chain)
    ShrakeRupley().compute(model, level="A")
    out = {}
    for index, residue in enumerate(
        (r for r in model[BINDER_CHAIN] if r.id[0] == " "), start=1
    ):
        reference = SIDECHAIN_REF_A2.get(residue.get_resname(), 1.0)
        total = sum(a.sasa for a in residue
                    if a.element != "H" and a.get_name() not in BACKBONE)
        out[index] = round(total / reference, 3)
    return out


def scan(design: str, sequence: str, sasa: dict[int, float]) -> list[dict]:
    sites = []

    def visible(position: int) -> tuple[float, bool]:
        value = sasa.get(position)
        if value is None:
            return (float("nan"), False)
        return (value, value >= EXPOSED_REL_SASA)

    def add(category: str, severity: str, motif: str, position: int,
            kept: bool | None = None) -> None:
        value, is_exposed = visible(position)
        sites.append({
            "design": design, "categorie": category, "gravite": severity,
            "motif": motif, "position": position,
            "rel_sasa": value, "expose": "oui" if is_exposed else "non",
            "retenu": "oui" if (is_exposed if kept is None else kept) else "non",
        })

    for category, severity, pattern in MOTIFS:
        for match in re.finditer(f"(?={pattern})", sequence):
            start = match.start()
            add(category, severity, sequence[start:start + 2], start + 1)

    for position, letter in enumerate(sequence, start=1):
        if letter in OXIDISABLE:
            add("oxydation", OXIDISABLE[letter], letter, position)
        if letter == "C":
            add("cysteine_libre", "dure", "C", position, kept=True)

    for match in re.finditer(r"(?=N[^P][ST])", sequence):
        start = match.start()
        add("sequon_N_glycosylation", "sans objet en acellulaire",
            sequence[start:start + 3], start + 1, kept=False)

    for start in range(len(sequence) - PATCH_WINDOW + 1):
        window = sequence[start:start + PATCH_WINDOW]
        shown = [p for p in range(start + 1, start + PATCH_WINDOW + 1) if visible(p)[1]]
        for group, label in ((POSITIVE, "plaque_positive"),
                             (NEGATIVE, "plaque_negative"),
                             (HYDROPHOBIC, "plaque_hydrophobe")):
            if sum(1 for p in shown if sequence[p - 1] in group) >= PATCH_MIN:
                sites.append({
                    "design": design, "categorie": label, "gravite": "moderee",
                    "motif": window, "position": start + 1, "rel_sasa": "",
                    "expose": "oui", "retenu": "oui",
                })

    for match in re.finditer(r"(.)\1{" + str(REPEAT_MIN - 1) + r",}", sequence):
        sites.append({
            "design": design, "categorie": "repetition", "gravite": "moderee",
            "motif": match.group(0), "position": match.start() + 1,
            "rel_sasa": "", "expose": "", "retenu": "oui",
        })
    return sites


def main() -> None:
    master = list(csv.DictReader(MASTER.open(newline="")))
    if not master:
        raise SystemExit(f"{MASTER} absent — lancer rank_designs.py d'abord")

    all_sites, summary = [], []
    for record in master:
        design = record["design_id"]
        sequence = record["sequence"].strip().upper()
        # Un mutant n'a pas de structure propre : on prend celle de son parent, dont il
        # partage le squelette, et la colonne `source_sasa` le dit.
        reference = record["parent"] or design
        sasa = exposure(reference)
        if sasa is None:
            print(f"  {design} : pas de structure, ignore")
            continue
        sites = scan(design, sequence, sasa)
        all_sites.extend(sites)
        kept = [s for s in sites if s["retenu"] == "oui"]

        def count(category: str) -> int:
            return sum(1 for s in kept if s["categorie"] == category)

        hard = count("cysteine_libre")
        summary.append({
            "design_id": design, "squelette": record["squelette"],
            "type": record["type"], "longueur": len(sequence),
            "source_sasa": "parent" if record["parent"] else "propre",
            "n_liabilites_dures": hard, "cysteine_libre": hard,
            "desamidation_exposee": count("desamidation"),
            "isomerisation_exposee": count("isomerisation"),
            "clivage_DP_expose": count("clivage_DP"),
            "oxydation_exposee": count("oxydation"),
            "plaque_positive": count("plaque_positive"),
            "plaque_negative": count("plaque_negative"),
            "plaque_hydrophobe": count("plaque_hydrophobe"),
            "repetition": count("repetition"),
            "n_sites_retenus": len(kept),
            "sequons_N_glyc_non_retenus": sum(
                1 for s in sites if s["categorie"] == "sequon_N_glycosylation"
            ),
            "charge_nette": record["charge_nette"],
        })

    for path, data in ((OUT, summary), (OUT_SITES, all_sites)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    print(f"-> {len(summary)} designs dans {OUT}")
    print(f"   {len(all_sites)} sites dans {OUT_SITES}")
    print()
    print(f"  liabilites DURES sur tout le vivier : "
          f"{sum(r['n_liabilites_dures'] for r in summary)}")
    print(f"  sequons N-glyc comptes mais NON retenus : "
          f"{sum(r['sequons_N_glyc_non_retenus'] for r in summary)}")
    print()
    columns = ["desamidation_exposee", "isomerisation_exposee", "clivage_DP_expose",
               "oxydation_exposee", "plaque_negative", "plaque_positive",
               "plaque_hydrophobe", "repetition"]
    labels = ["desam", "isom", "DP", "oxyd", "pl.neg", "pl.pos", "pl.hyd", "repet"]
    print(f"  {'design':<26}" + "".join(f"{l:>8}" for l in labels) + f"{'total':>7}")
    for record in sorted(summary, key=lambda r: -r["n_sites_retenus"]):
        if record["type"] != "bindcraft":
            continue
        print(f"  {record['design_id'].split('_denovo_')[-1][:24]:<26}"
              + "".join(f"{record[c]:>8}" for c in columns)
              + f"{record['n_sites_retenus']:>7}")


if __name__ == "__main__":
    main()
