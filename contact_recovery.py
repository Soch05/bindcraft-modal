#!/usr/bin/env python3
"""Récupération de contacts entre la pose AF2 et la pose Boltz-2.

    modal volume get bindcraft "boltz/rescore01" out/
    uv run --with biopython --with gemmi python contact_recovery.py

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
OUT = Path("out/boltz_contacts.csv")

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
    """Paires (résidu cible, résidu binder) en contact, exprimées dans la numérotation AF2."""
    target = residues(model, TARGET_CHAIN)
    binder = residues(model, BINDER_CHAIN)
    if not target or not binder:
        return set()

    def heavy(residue) -> np.ndarray:
        return np.array([a.coord for a in residue if a.element != "H"])

    target_atoms = [(r.id[1] + offset_target, heavy(r)) for r in target]
    binder_atoms = [(r.id[1] + offset_binder, heavy(r)) for r in binder]

    # Pré-filtrage par centroïde : sans lui on compare 198 x 95 x atomes paires à chaque
    # fois, ce qui est lent pour rien sur une machine sans GPU.
    found = set()
    for target_number, target_coords in target_atoms:
        if not len(target_coords):
            continue
        centre = target_coords.mean(axis=0)
        for binder_number, binder_coords in binder_atoms:
            if not len(binder_coords):
                continue
            if np.linalg.norm(centre - binder_coords.mean(axis=0)) > 25.0:
                continue
            distances = np.linalg.norm(
                target_coords[:, None, :] - binder_coords[None, :, :], axis=-1
            )
            if distances.min() <= CONTACT_CUTOFF_A:
                found.add((target_number, binder_number))
    return found


def offsets(af2_model, boltz_model) -> tuple[int, int]:
    """Détermine et VÉRIFIE le décalage de numérotation entre les deux conventions."""
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
    if sequence(af2_model, BINDER_CHAIN) != sequence(boltz_model, BINDER_CHAIN):
        raise ValueError("sequences de binder differentes — comparaison refusee")
    offset_binder = af2_binder[0].id[1] - boltz_binder[0].id[1]
    return (offset_target, offset_binder)


def main() -> None:
    if not BOLTZ_DIR.is_dir():
        raise SystemExit(
            f"{BOLTZ_DIR} absent. Rapatrier d'abord :\n"
            f"  modal volume get bindcraft 'boltz/rescore01' out/"
        )

    results = []
    for folder in sorted(p for p in BOLTZ_DIR.rglob("*") if p.is_dir()):
        structures = sorted(folder.glob("*.cif"))
        if not structures:
            continue
        design = folder.name
        af2_path = AF2_DIR / f"{design}.pdb"
        if not af2_path.is_file():
            print(f"  {design} : pas de structure AF2 locale, ignore")
            continue
        af2_model = load(af2_path)
        reference = contacts(af2_model, 0, 0)
        if not reference:
            print(f"  {design} : aucun contact AF2, ignore")
            continue

        per_sample, confidences = [], []
        for structure_path in structures:
            try:
                boltz_model = load(structure_path)
                offset_target, offset_binder = offsets(af2_model, boltz_model)
            except (ValueError, KeyError) as problem:
                print(f"  {design} / {structure_path.name} : {problem}")
                continue
            predicted = contacts(boltz_model, offset_target, offset_binder)
            recovered = len(reference & predicted) / len(reference)
            per_sample.append((structure_path.name, round(recovered, 3), len(predicted)))

        for confidence_path in sorted(folder.glob("confidence_*.json")):
            payload = json.loads(confidence_path.read_text())
            confidences.append(payload)

        if not per_sample:
            continue
        recoveries = [r for _, r, _ in per_sample]
        iptms = [c.get("iptm") for c in confidences if c.get("iptm") is not None]
        ptms = [c.get("ptm") for c in confidences if c.get("ptm") is not None]
        plddts = [
            c.get("complex_iplddt") for c in confidences
            if c.get("complex_iplddt") is not None
        ]
        results.append({
            "design": design,
            "n_echantillons": len(per_sample),
            "paires_AF2": len(reference),
            "recuperation_contacts": round(max(recoveries), 3),
            "recuperation_min": round(min(recoveries), 3),
            "recuperation_moyenne": round(sum(recoveries) / len(recoveries), 3),
            "iptm_moyen": round(sum(iptms) / len(iptms), 3) if iptms else "",
            "iptm_max": round(max(iptms), 3) if iptms else "",
            "ptm_moyen": round(sum(ptms) / len(ptms), 3) if ptms else "",
            "iplddt_moyen": round(sum(plddts) / len(plddts), 3) if plddts else "",
            "detail": " | ".join(f"{n}:{r}" for n, r, _ in per_sample),
        })
        print(f"  {design[-26:]:<28} paires AF2 {len(reference):4d}  "
              f"recup max {max(recoveries):5.3f}  min {min(recoveries):5.3f}  "
              f"iptm {results[-1]['iptm_moyen']}")

    if not results:
        raise SystemExit("aucun resultat exploitable")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print()
    print(f"-> {len(results)} designs dans {OUT}")


if __name__ == "__main__":
    main()
