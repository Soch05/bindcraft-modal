#!/usr/bin/env python3
"""Teste une règle de conception : faut-il DEUX carboxylates distincts sur H409 ?

    uv run --with biopython --with gemmi python bidentate_rule.py

D'OÙ VIENT LA QUESTION. Sur les 23 designs, deux seulement font monter le pKa de H409. Le
critère géométrique qui avait servi à les sélectionner — « un pont salin à moins de 4 Å » —
ne les distingue pas : plusieurs designs à pont salin impeccable ont un ΔpKa nul ou négatif.
Il manque donc quelque chose au critère.

PREMIÈRE HYPOTHÈSE, RÉFUTÉE. « Les deux azotes de l'imidazole doivent être engagés. » Mesurée :
9 designs engagent les deux azotes et leur ΔpKa va de +2,84 à −1,19. Nécessaire, pas suffisant.

SECONDE HYPOTHÈSE, CELLE QU'ON TESTE ICI. Ce qui compte n'est pas que les deux azotes soient
approchés, mais qu'ils le soient par **deux résidus carboxylate DIFFÉRENTS**. Un seul
carboxylate qui pivote entre les deux azotes ne peut stabiliser qu'une seule liaison à la
fois ; deux résidus distincts peuvent saturer l'imidazole protoné des deux côtés, ce qui est
la condition pour déplacer son équilibre acido-basique.

La mesure : pour chaque design, on cherche la meilleure affectation de DEUX résidus distincts,
l'un à ND1 et l'autre à NE2, et on retient la PLUS MAUVAISE des deux distances. C'est le
goulot d'étranglement du mécanisme bidenté.

Sortie : out/bidentate.csv
"""

from __future__ import annotations

import csv
import itertools
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore")

STRUCTURES = Path("structures/wt")
BOLTZ_HUMAN = Path("out/rescore01")
PROPKA_H409 = Path("out/propka_h409.csv")
CROSS = Path("out/cross_species.csv")
OUT = Path("out/bidentate.csv")
OUT_BOLTZ = Path("out/bidentate_boltz.csv")

# Décalage de la numérotation Boltz (1..198) vers la numérotation PDB (309..506). Valable
# parce qu'il n'y a aucun indel dans la fenêtre — vérifié par mouse_target.py.
PDB_OFFSET = 308

TARGET_CHAIN, BINDER_CHAIN = "A", "B"
TARGET_HIS = 409
NITROGENS = ("ND1", "NE2")
ACID = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}

# Rayon de recherche des carboxylates candidats autour de chaque azote. Large exprès : on
# veut voir la distribution, pas appliquer un seuil choisi d'avance.
SEARCH_A = 8.0

# Plancher de bruit de PROPKA, comme ailleurs dans le pipeline.
PKA_NOISE = 0.5


def closest_oxygen(residue, coord: np.ndarray) -> tuple[float, str]:
    names = ACID[residue.get_resname()]
    return min(
        (float(np.linalg.norm(residue[o].coord - coord)), o)
        for o in names if o in residue
    )


def best_boltz(folder: Path) -> Path | None:
    """L'échantillon Boltz de plus haut iptm, renuméroté en PDB dans un fichier temporaire."""
    import json
    import tempfile

    import gemmi

    best, best_value = None, -1.0
    for confidence in folder.glob("confidence_*.json"):
        value = json.loads(confidence.read_text()).get("iptm")
        if value is None:
            continue
        stem = confidence.name.replace("confidence_", "").replace(".json", "")
        candidate = folder / f"{stem}.cif"
        if candidate.is_file() and value > best_value:
            best, best_value = candidate, value
    if best is None:
        return None
    structure = gemmi.read_structure(str(best))
    structure.setup_entities()
    for residue in structure[0][TARGET_CHAIN]:
        residue.seqid.num += PDB_OFFSET
    out = Path(tempfile.mkdtemp(prefix="bident_")) / f"{folder.name}.pdb"
    structure.write_pdb(str(out))
    return out


def analyse(design: str, path: Path | None = None) -> dict | None:
    path = path or STRUCTURES / f"{design}.pdb"
    if not path.is_file():
        return None
    model = PDBParser(QUIET=True).get_structure(design, str(path))[0]
    his = model[TARGET_CHAIN][(" ", TARGET_HIS, " ")]
    if his.get_resname() != "HIS":
        raise SystemExit(f"{design} : A{TARGET_HIS} n'est pas une HIS")
    nitrogens = {n: his[n].coord for n in NITROGENS if n in his}

    acids = [r for r in model[BINDER_CHAIN] if r.get_resname() in ACID]
    reach: dict[str, list[tuple[float, str, str]]] = {n: [] for n in nitrogens}
    for residue in acids:
        for name, coord in nitrogens.items():
            distance, oxygen = closest_oxygen(residue, coord)
            if distance <= SEARCH_A:
                reach[name].append(
                    (distance, f"{residue.get_resname()}{residue.id[1]}", oxygen)
                )

    # Mono-denté : le meilleur carboxylate, quel que soit l'azote.
    everything = [item for items in reach.values() for item in items]
    best_single = min(everything, default=(float("nan"), "", ""))

    # Bidenté par DEUX résidus distincts : on minimise la plus mauvaise des deux distances.
    best_pair, best_bottleneck = None, float("inf")
    if len(nitrogens) == 2:
        first, second = NITROGENS
        for a, b in itertools.product(reach.get(first, []), reach.get(second, [])):
            if a[1] == b[1]:
                continue
            bottleneck = max(a[0], b[0])
            if bottleneck < best_bottleneck:
                best_bottleneck, best_pair = bottleneck, (a, b)

    row = {
        "design": design,
        "squelette": design.split("_")[-2],
        "n_acides_binder": len(acids),
        "mono_residu": best_single[1],
        "mono_distance_A": round(best_single[0], 2) if best_single[1] else "",
        "bidente_possible": "oui" if best_pair else "non",
    }
    if best_pair:
        (d1, r1, o1), (d2, r2, o2) = best_pair
        row.update({
            "bidente_ND1": f"{r1}:{o1}", "bidente_ND1_A": round(d1, 2),
            "bidente_NE2": f"{r2}:{o2}", "bidente_NE2_A": round(d2, 2),
            "bidente_goulot_A": round(best_bottleneck, 2),
        })
    else:
        row.update({"bidente_ND1": "", "bidente_ND1_A": "", "bidente_NE2": "",
                    "bidente_NE2_A": "", "bidente_goulot_A": ""})
    return row


def separation(rows: list[dict], shift_key: str) -> str:
    """La règle sépare-t-elle les designs à mécanisme des autres, sur CE jeu de ΔpKa ?"""
    with_pair = [r for r in rows if isinstance(r.get("bidente_goulot_A"), float)]
    positive = [r["bidente_goulot_A"] for r in with_pair if r[shift_key] > PKA_NOISE]
    negative = [r["bidente_goulot_A"] for r in with_pair if r[shift_key] <= PKA_NOISE]
    orphan = sum(
        1 for r in rows
        if not isinstance(r.get("bidente_goulot_A"), float) and r[shift_key] > PKA_NOISE
    )
    print(f"  positifs avec paire : {sorted(round(v, 2) for v in positive)}")
    print(f"  negatifs avec paire : {sorted(round(v, 2) for v in negative)}")
    print(f"  positifs SANS paire bidentee possible : {orphan}")
    if not positive or not negative:
        return "indecidable (un des deux groupes est vide)"
    gap = min(negative) - max(positive)
    if gap > 0:
        verdict = f"SEPARATION NETTE, marge {gap:.2f} A"
    else:
        verdict = f"RECOUVREMENT de {-gap:.2f} A — la regle ne separe pas"
    print(f"  -> {verdict}")
    return verdict


def main() -> None:
    af2_shift = {
        r["design"]: float(r["dpKa_H409"])
        for r in csv.DictReader(PROPKA_H409.open(newline="")) if r["type"] == "WT"
    }
    boltz_shift = {}
    if CROSS.is_file():
        for record in csv.DictReader(CROSS.open(newline="")):
            value = record.get("dpKa_humain_Boltz")
            if value:
                boltz_shift[record["design"]] = float(value)

    # --- Test 1 : geometrie AF2 contre dpKa AF2 (la derivation d'origine) --------------
    print("=" * 84)
    print("TEST 1 — geometrie mesuree sur AF2, dpKa calcule sur AF2")
    print("  ATTENTION : les deux viennent de la MEME structure, donc la correlation est")
    print("  en partie auto-referentielle. C'est le test le plus faible des trois.")
    print("=" * 84)
    af2_rows = []
    for design, shift in af2_shift.items():
        row = analyse(design)
        if row is None:
            continue
        row["dpKa_H409"] = shift
        row["source_geometrie"] = "AF2"
        af2_rows.append(row)
    verdict_af2 = separation(af2_rows, "dpKa_H409")

    # --- Test 2 : geometrie AF2 contre dpKa Boltz (croise) ----------------------------
    print()
    print("=" * 84)
    print("TEST 2 — geometrie mesuree sur AF2, dpKa calcule sur Boltz (croise)")
    print("=" * 84)
    crossed = [
        {**r, "dpKa_Boltz": boltz_shift[r["design"]]}
        for r in af2_rows if r["design"] in boltz_shift
    ]
    verdict_cross = separation(crossed, "dpKa_Boltz") if crossed else "non calculable"

    # --- Test 3 : geometrie Boltz contre dpKa Boltz (le test propre) ------------------
    print()
    print("=" * 84)
    print("TEST 3 — geometrie mesuree sur Boltz, dpKa calcule sur Boltz")
    print("  C'est le test CORRECT : un seul predicteur des deux cotes, donc la regle est")
    print("  evaluee sur des donnees qui ne l'ont pas engendree.")
    print("=" * 84)
    boltz_rows = []
    if BOLTZ_HUMAN.is_dir():
        for design, shift in boltz_shift.items():
            folder = BOLTZ_HUMAN / design
            if not folder.is_dir():
                continue
            path = best_boltz(folder)
            if path is None:
                continue
            row = analyse(design, path)
            if row is None:
                continue
            row["dpKa_Boltz"] = shift
            row["source_geometrie"] = "Boltz"
            boltz_rows.append(row)
    verdict_boltz = separation(boltz_rows, "dpKa_Boltz") if boltz_rows else "non calculable"

    print()
    print("=" * 84)
    print("BILAN")
    print("=" * 84)
    print(f"  AF2 geometrie  / AF2 dpKa    : {verdict_af2}")
    print(f"  AF2 geometrie  / Boltz dpKa  : {verdict_cross}")
    print(f"  Boltz geometrie/ Boltz dpKa  : {verdict_boltz}")

    if af2_rows:
        with OUT.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(af2_rows[0]))
            writer.writeheader()
            writer.writerows(af2_rows)
        print(f"\n-> {len(af2_rows)} lignes dans {OUT}")
    if boltz_rows:
        with OUT_BOLTZ.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(boltz_rows[0]))
            writer.writeheader()
            writer.writerows(boltz_rows)
        print(f"-> {len(boltz_rows)} lignes dans {OUT_BOLTZ}")


if __name__ == "__main__":
    main()
