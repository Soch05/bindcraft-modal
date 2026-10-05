#!/usr/bin/env python3
"""Récupération de contacts entre la pose AF2 et la pose Boltz-2.

    modal volume get bindcraft "boltz/rescore01" out/
    uv run --with biopython --with gemmi --with scipy python contact_recovery.py

POURQUOI LA RÉCUPÉRATION DE CONTACTS ET PAS LE RMSD. Un RMSD d'interface pardonne une
ROTATION du binder qui reste sur la même zone — il peut rester bas alors que plus aucun
résidu ne se fait face. À l'inverse il pénalise lourdement une TRANSLATION rigide qui
conserve toutes les paires en contact, ce qui est sans conséquence biologique. La question
posée ici est « le modèle indépendant met-il les mêmes résidus face à face ? », donc on
compte des paires de résidus en contact, pas des angströms.

Score : |paires AF2 ∩ paires Boltz| / |paires AF2|. Asymétrique à dessein — on demande si
la pose d'AF2 est RETROUVÉE, pas si les deux poses sont identiques.

PIÈGE DE NUMÉROTATION, VÉRIFIÉ ET NON SUPPOSÉ. Boltz renumérote chaque chaîne à partir de 1,
alors que la cible de BindCraft porte la numérotation PDB 309–506. L'offset est donc de 308,
mais on ne le prend pas pour acquis : on compare les séquences résidu par résidu et on refuse
de calculer si elles ne concordent pas.

Sortie : out/boltz_contacts.csv
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

warnings.filterwarnings("ignore")

AF2_DIR = Path("structures/wt")
BOLTZ_DIR = Path("out/rescore01")
MUTANT_DIR = Path("out/rescore_mut01")
OUT = Path("out/boltz_contacts.csv")
OUT_MUTANTS = Path("out/boltz_contacts_mutants.csv")

TARGET_CHAIN = "A"
BINDER_CHAIN = "B"

# Deux résidus sont en contact si un atome lourd de l'un est à moins de ça d'un atome lourd
# de l'autre. 5,0 A est la convention usuelle pour un contact inter-résidus ; plus serré on
# ne compterait que les liaisons, plus large on compterait la seconde couche.
CONTACT_CUTOFF_A = 5.0

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load(path: Path):
    parser = MMCIFParser(QUIET=True) if path.suffix in {".cif", ".mmcif"} \
        else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))[0]


def residues(model, chain: str) -> list:
    if chain not in [c.id for c in model]:
        return []
    return [r for r in model[chain] if r.id[0] == " "]


def sequence(model, chain: str) -> str:
    return "".join(THREE_TO_ONE.get(r.get_resname(), "X") for r in residues(model, chain))


def contacts(model, offset_target: int, offset_binder: int) -> set[tuple[int, int]]:
    """Paires (résidu cible, résidu binder) en contact, exprimées dans la numérotation AF2.

    IMPLÉMENTATION PAR ARBRE K-D, et pas par boucles imbriquées. La version naïve compare
    chaque paire de résidus puis chaque paire d'atomes : sur 198 x ~70 résidus et 69
    structures, elle a mis 3 h 30 de CPU la première nuit. Un arbre k-d sur tous les atomes
    lourds d'une chaîne, interrogé par rayon depuis ceux de l'autre, donne EXACTEMENT le même
    ensemble de paires en moins d'une seconde par structure. Le résultat a été comparé ligne
    à ligne avec celui de la version naïve avant de la remplacer.
    """
    from scipy.spatial import cKDTree

    target = residues(model, TARGET_CHAIN)
    binder = residues(model, BINDER_CHAIN)
    if not target or not binder:
        return set()

    def flatten(group, offset: int) -> tuple[np.ndarray, np.ndarray]:
        coords, labels = [], []
        for residue in group:
            for atom in residue:
                if atom.element == "H":
                    continue
                coords.append(atom.coord)
                labels.append(residue.id[1] + offset)
        return (np.asarray(coords, dtype=float), np.asarray(labels, dtype=int))

    target_coords, target_labels = flatten(target, offset_target)
    binder_coords, binder_labels = flatten(binder, offset_binder)
    if not len(target_coords) or not len(binder_coords):
        return set()

    tree = cKDTree(binder_coords)
    found = set()
    for index, neighbours in enumerate(
        tree.query_ball_point(target_coords, CONTACT_CUTOFF_A)
    ):
        if not neighbours:
            continue
        label = int(target_labels[index])
        for neighbour in neighbours:
            found.add((label, int(binder_labels[neighbour])))
    return found


def offsets(af2_model, boltz_model, allowed_mismatches: int = 0) -> tuple[int, int]:
    """Détermine et VÉRIFIE le décalage de numérotation entre les deux conventions.

    `allowed_mismatches` sert aux MUTANTS : leur binder diffère de celui du parent d'exactement
    un résidu, et la référence de comparaison est forcément la pose AF2 du parent puisque le
    mutant n'a pas de pose AF2. On tolère donc ce nombre exact de différences, pas plus — une
    tolérance illimitée laisserait passer une comparaison entre deux binders sans rapport.
    """
    af2_target = residues(af2_model, TARGET_CHAIN)
    boltz_target = residues(boltz_model, TARGET_CHAIN)
    if len(af2_target) != len(boltz_target):
        raise ValueError(
            f"cible de longueur differente : AF2 {len(af2_target)} vs "
            f"Boltz {len(boltz_target)} — comparaison refusee"
        )
    offset_target = af2_target[0].id[1] - boltz_target[0].id[1]
    if sequence(af2_model, TARGET_CHAIN) != sequence(boltz_model, TARGET_CHAIN):
        raise ValueError("sequences de cible differentes — comparaison refusee")

    af2_binder = residues(af2_model, BINDER_CHAIN)
    boltz_binder = residues(boltz_model, BINDER_CHAIN)
    first, second = sequence(af2_model, BINDER_CHAIN), sequence(boltz_model, BINDER_CHAIN)
    if len(first) != len(second):
        raise ValueError(
            f"binder de longueur differente : {len(first)} vs {len(second)}"
        )
    mismatches = sum(1 for a, b in zip(first, second) if a != b)
    if mismatches > allowed_mismatches:
        raise ValueError(
            f"{mismatches} difference(s) de binder, {allowed_mismatches} toleree(s) "
            f"— comparaison refusee"
        )
    offset_binder = af2_binder[0].id[1] - boltz_binder[0].id[1]
    return (offset_target, offset_binder)


def evaluate(folder: Path, af2_path: Path, label: str,
             allowed_mismatches: int = 0) -> dict | None:
    """Compare les échantillons Boltz d'un dossier à une pose AF2 de référence."""
    structures = sorted(folder.glob("*.cif"))
    if not structures or not af2_path.is_file():
        return None
    af2_model = load(af2_path)
    reference = contacts(af2_model, 0, 0)
    if not reference:
        return None

    per_sample = []
    for structure_path in structures:
        try:
            boltz_model = load(structure_path)
            offset_target, offset_binder = offsets(
                af2_model, boltz_model, allowed_mismatches
            )
        except (ValueError, KeyError) as problem:
            print(f"  {label} / {structure_path.name} : {problem}")
            continue
        predicted = contacts(boltz_model, offset_target, offset_binder)
        per_sample.append((structure_path.name,
                           round(len(reference & predicted) / len(reference), 3)))
    if not per_sample:
        return None

    confidences = [
        json.loads(path.read_text()) for path in sorted(folder.glob("confidence_*.json"))
    ]
    recoveries = [value for _, value in per_sample]

    def average(key: str) -> float | str:
        values = [c[key] for c in confidences if c.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else ""

    return {
        "design": label,
        "n_echantillons": len(per_sample),
        "paires_AF2": len(reference),
        "recuperation_contacts": round(max(recoveries), 3),
        "recuperation_min": round(min(recoveries), 3),
        "recuperation_moyenne": round(sum(recoveries) / len(recoveries), 3),
        "iptm_moyen": average("iptm"),
        "iptm_max": round(max(
            [c["iptm"] for c in confidences if c.get("iptm") is not None] or [0]
        ), 3),
        "ptm_moyen": average("ptm"),
        "iplddt_moyen": average("complex_iplddt"),
        "detail": " | ".join(f"{n}:{v}" for n, v in per_sample),
    }


def write(path: Path, results: list[dict]) -> None:
    if not results:
        print(f"  (rien a ecrire dans {path})")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"-> {len(results)} lignes dans {path}")


def main() -> None:
    if not BOLTZ_DIR.is_dir():
        raise SystemExit(
            f"{BOLTZ_DIR} absent. Rapatrier d'abord :\n"
            f"  modal volume get bindcraft 'boltz/rescore01' out/"
        )

    print("=== DESIGNS NATIFS : pose Boltz vs pose AF2 du meme design ===")
    natives = []
    for folder in sorted(p for p in BOLTZ_DIR.iterdir() if p.is_dir()):
        row = evaluate(folder, AF2_DIR / f"{folder.name}.pdb", folder.name)
        if row is None:
            continue
        natives.append(row)
        print(f"  {row['design'][-26:]:<28} paires AF2 {row['paires_AF2']:4d}  "
              f"recup max {row['recuperation_contacts']:5.3f}  "
              f"min {row['recuperation_min']:5.3f}  iptm {row['iptm_moyen']}")

    mutants = []
    if MUTANT_DIR.is_dir():
        print()
        print("=== MUTANTS : pose Boltz du mutant vs pose AF2 du PARENT ===")
        print("    (la reference est le parent : un mutant n'a pas de pose AF2)")
        for folder in sorted(p for p in MUTANT_DIR.iterdir() if p.is_dir()):
            parent = folder.name.split("__")[0]
            row = evaluate(folder, AF2_DIR / f"{parent}.pdb", folder.name,
                           allowed_mismatches=1)
            if row is None:
                continue
            row["parent"] = parent
            row["mutation"] = folder.name.split("__")[1]
            mutants.append(row)
            print(f"  {parent[-22:]:<24} {row['mutation']:<6} "
                  f"paires parent {row['paires_AF2']:4d}  "
                  f"recup max {row['recuperation_contacts']:5.3f}  "
                  f"min {row['recuperation_min']:5.3f}  iptm {row['iptm_moyen']}")
    else:
        print()
        print(f"{MUTANT_DIR} absent : coût structural des mutants non mesuré.")

    print()
    write(OUT, natives)
    write(OUT_MUTANTS, mutants)


if __name__ == "__main__":
    main()
