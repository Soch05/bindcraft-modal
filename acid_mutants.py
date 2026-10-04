#!/usr/bin/env python3
"""Propose une substitution en acide par design, pour mettre un carboxylate à portée de H409.

    uv run --with openpyxl python acid_mutants.py out/egfr-dIII-prod01 out/egfr-dIII-prod02

Écrit une feuille `mutants_acide` dans le classeur de chaque campagne, et un CSV récapitulatif
`out/mutants_acide.csv` couvrant toutes les campagnes, avec la séquence d'origine et la
séquence mutée côte à côte.

MÉCANISME VISÉ — route 2 de CLAUDE.md §6 : un carboxylate du binder contre `H409`, qui est
une His conservée de la cible. À pH 6,5 `H409` est davantage protonée donc chargée, et le
pont salin est renforcé ; à pH 7,4 elle est neutre et il s'affaiblit. Gain de liaison à
pH 6,5, ce que demande l'objectif n°1.

CE QUE CE SCRIPT NE FAIT PAS, ET IL FAUT LE LIRE :

  - il ne prédit RIEN. Les `i_pTM`/`i_pAE` d'un design muté sont **inconnus**. Les colonnes
    de métriques de la feuille décrivent la séquence **d'origine** et ne s'appliquent pas au
    mutant. `bindcraft score` ne peut pas combler ce trou : il réaffiche les valeurs du
    tampon du fichier au lieu de les recalculer (vérifié dans `bindcraft/score.py`, qui
    construit `StructurePrediction(metrics={})` et l'imprime lui-même) ;
  - il ne vérifie pas l'enfouissement du résidu dans le cœur du binder. Un CB proche de
    `H409` fait face à la cible, ce qui est un indice, pas une preuve ;
  - il ne modélise pas le repliement du mutant. Une substitution peut déplacer la pose.

Un mutant est donc une **hypothèse à faire scorer**, jamais un design validé.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------------------
# Seuils, dans le code et non dans la prose (CLAUDE.md §7).
# ----------------------------------------------------------------------------------------

TARGET_HIS_ANCHOR = 409
HIS_CHARGED_ATOMS = ("ND1", "NE2")

# Portée d'une chaîne latérale depuis le CB : Asp ~2,5 Å, Glu ~3,9 Å. On choisit donc Asp
# quand le CB est déjà proche et Glu quand il faut aller chercher plus loin.
ASP_CB_MAX_A = 5.0
GLU_CB_MAX_A = 7.0

# Déjà un pont salin à cette distance : pas besoin de muter.
SALT_BRIDGE_A = 4.0

# Classement du coût d'une substitution vers un carboxylate. Petit polaire ou petit neutre
# est un échange modeste ; une charge inverse ou un résidu de cœur hydrophobe ne l'est pas.
CONSERVATIVE = {"SER", "THR", "ASN", "GLN", "ALA", "GLY"}
RISKY = {"LYS", "ARG", "HIS", "TYR"}
AVOID = {"PHE", "TRP", "LEU", "ILE", "VAL", "MET", "PRO", "CYS"}
ALREADY_ACID = {"ASP", "GLU"}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


@dataclass(frozen=True)
class Atom:
    chain: str
    resnum: int
    resname: str
    atom: str
    x: float
    y: float
    z: float


def read_cif(path: Path) -> tuple[list[Atom], dict[str, str]]:
    text = path.read_text()
    stamp = dict(re.findall(r"^_bindcraft\.(\S+)\s+'?([^'\n]+?)'?\s*$", text, re.M))
    columns: list[str] = []
    atoms: list[Atom] = []
    in_loop = False
    for line in text.splitlines():
        if line.startswith("_atom_site."):
            columns.append(line.strip().removeprefix("_atom_site."))
            in_loop = True
            continue
        if in_loop and line.startswith(("ATOM", "HETATM")):
            fields = line.split()
            if len(fields) < len(columns):
                continue
            row = dict(zip(columns, fields))
            try:
                atoms.append(
                    Atom(row["auth_asym_id"], int(row["auth_seq_id"]), row["auth_comp_id"],
                         row["auth_atom_id"], float(row["Cartn_x"]), float(row["Cartn_y"]),
                         float(row["Cartn_z"]))
                )
            except (KeyError, ValueError):
                continue
        elif in_loop and atoms and not line.startswith(("ATOM", "HETATM")):
            break
    return atoms, stamp


def substitution_tier(resname: str) -> tuple[int, str]:
    if resname in CONSERVATIVE:
        return 0, "conservative"
    if resname in RISKY:
        return 1, "risquee"
    if resname in AVOID:
        return 2, "a eviter"
    return 3, "inconnue"


def existing_salt_bridge(atoms: list[Atom], binder: str, target: str) -> float | None:
    """Distance minimale carboxylate du binder — imidazole de H409, si elle existe."""
    his = [a for a in atoms if a.chain == target and a.resnum == TARGET_HIS_ANCHOR
           and a.atom in HIS_CHARGED_ATOMS]
    acids = [a for a in atoms if a.chain == binder and a.resname in ALREADY_ACID
             and a.atom in {"OD1", "OD2", "OE1", "OE2"}]
    if not his or not acids:
        return None
    return min(math.dist((a.x, a.y, a.z), (h.x, h.y, h.z)) for a in acids for h in his)


def propose(path: Path) -> dict | None:
    atoms, stamp = read_cif(path)
    binder = stamp.get("binder_chains", "B").split(",")[0]
    target = stamp.get("target_chains", "A").split(",")[0]

    his = [a for a in atoms if a.chain == target and a.resnum == TARGET_HIS_ANCHOR
           and a.atom in HIS_CHARGED_ATOMS]
    if not his:
        return None

    # Séquence reconstruite depuis la structure, pour pouvoir vérifier l'index.
    residues = {a.resnum: a.resname for a in atoms if a.chain == binder}
    order = sorted(residues)
    sequence = "".join(THREE_TO_ONE.get(residues[n], "X") for n in order)
    first = order[0]

    candidates = []
    for atom in atoms:
        if atom.chain != binder or atom.atom != "CB":
            continue
        distance = min(math.dist((atom.x, atom.y, atom.z), (h.x, h.y, h.z)) for h in his)
        if distance > GLU_CB_MAX_A:
            continue
        tier, label = substitution_tier(atom.resname)
        candidates.append((tier, distance, atom.resnum, atom.resname, label))
    candidates.sort()

    bridge = existing_salt_bridge(atoms, binder, target)
    row = {
        "design": path.stem,
        "hash": stamp.get("campaign", "") and path.stem.split("_")[-2],
        "length": len(sequence),
        "pont_salin_existant_A": round(bridge, 2) if bridge is not None else "",
        "pH_deja_etabli": "oui" if bridge is not None and bridge <= SALT_BRIDGE_A else "non",
        "Binder_Sequence": sequence,
        "mutation": "",
        "Binder_Sequence_mutee": "",
        "residu_dorigine": "",
        "CB_vers_H409_A": "",
        "cout_substitution": "",
        "candidats_a_portee": len(candidates),
    }

    if row["pH_deja_etabli"] == "oui":
        row["mutation"] = "(aucune, pont deja present)"
        return row
    if not candidates:
        row["mutation"] = "(aucun residu a portee)"
        return row

    tier, distance, resnum, resname, label = candidates[0]
    if resname in ALREADY_ACID:
        row["mutation"] = "(deja un acide, mais hors portee)"
        return row

    index = resnum - first
    found = sequence[index]
    expected = THREE_TO_ONE.get(resname, "X")
    if found != expected:
        row["mutation"] = f"ERREUR d'index : attendu {expected}{resnum}, trouve {found}"
        return row

    new = "D" if distance <= ASP_CB_MAX_A else "E"
    row.update(
        mutation=f"{expected}{resnum}{new}",
        Binder_Sequence_mutee=sequence[:index] + new + sequence[index + 1:],
        residu_dorigine=f"{resname}{resnum}",
        CB_vers_H409_A=round(distance, 2),
        cout_substitution=label,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaigns", type=Path, nargs="+")
    args = parser.parse_args()

    all_rows: list[dict] = []
    for root in args.campaigns:
        ranked = root / "3_Ranked"
        if not ranked.is_dir():
            print(f"{root.name} : aucun 3_Ranked, ignore")
            continue
        print(f"{root.name}")
        rows = []
        for path in sorted(ranked.glob("*.cif")):
            row = propose(path)
            if row:
                row["run"] = root.name
                rows.append(row)
                print(f"  {path.stem[-28:]:<30} {row['mutation']:<34} "
                      f"CB {row['CB_vers_H409_A'] or '-'}  {row['cout_substitution']}")
        all_rows += rows
        print()

    out = Path("out/mutants_acide.csv")
    columns = ["run", "design", "length", "pH_deja_etabli", "pont_salin_existant_A",
               "mutation", "residu_dorigine", "CB_vers_H409_A", "cout_substitution",
               "candidats_a_portee", "Binder_Sequence", "Binder_Sequence_mutee"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"-> {out}  ({len(all_rows)} lignes)")

    proposed = [r for r in all_rows if r["Binder_Sequence_mutee"]]
    print(f"   mutations proposees : {len(proposed)} / {len(all_rows)}")
    print()
    print("⚠️ Les i_pTM/i_pAE des lignes mutees sont INCONNUS. Les colonnes de metriques")
    print("   decrivent la sequence d'origine. Un mutant est une hypothese a faire scorer.")


if __name__ == "__main__":
    main()
