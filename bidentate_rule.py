#!/usr/bin/env python3
"""Teste une règle de conception : faut-il DEUX carboxylates distincts sur H409 ?

    uv run --with biopython python bidentate_rule.py

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
PROPKA_H409 = Path("out/propka_h409.csv")
OUT = Path("out/bidentate.csv")

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


def analyse(design: str) -> dict | None:
    path = STRUCTURES / f"{design}.pdb"
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


def main() -> None:
    shifts = {
        r["design"]: float(r["dpKa_H409"])
        for r in csv.DictReader(PROPKA_H409.open(newline="")) if r["type"] == "WT"
    }
    rows = []
    for design, shift in shifts.items():
        row = analyse(design)
        if row is None:
            continue
        row["dpKa_H409"] = shift
        row["mecanisme"] = "oui" if shift > PKA_NOISE else "non"
        rows.append(row)

    rows.sort(key=lambda r: -r["dpKa_H409"])
    print(f"{'design':<28}{'dpKa':>7}{'goulot bidente':>16}  paire")
    print("-" * 92)
    for row in rows:
        bottleneck = row["bidente_goulot_A"]
        print(f"{row['design'][-26:]:<28}{row['dpKa_H409']:>7.2f}"
              f"{str(bottleneck):>16}  "
              f"{row['bidente_ND1']}/{row['bidente_NE2']}")

    with_mechanism = [r for r in rows if r["mecanisme"] == "oui"]
    without = [r for r in rows if r["mecanisme"] == "non"]

    def bottlenecks(group: list[dict]) -> list[float]:
        return [r["bidente_goulot_A"] for r in group
                if isinstance(r["bidente_goulot_A"], float)]

    yes, no = bottlenecks(with_mechanism), bottlenecks(without)
    print()
    print(f"avec mecanisme (dpKa > {PKA_NOISE}) : {len(with_mechanism)} designs")
    if yes:
        print(f"   goulot bidente : {min(yes):.2f} a {max(yes):.2f} A")
    print(f"sans mecanisme : {len(without)} designs")
    if no:
        print(f"   goulot bidente : {min(no):.2f} a {max(no):.2f} A "
              f"({len(without) - len(no)} sans paire bidentee possible)")
    if yes and no:
        gap = min(no) - max(yes)
        print()
        if gap > 0:
            print(f"SEPARATION NETTE : aucun recouvrement. Le seuil se situe entre "
                  f"{max(yes):.2f} et {min(no):.2f} A ({gap:.2f} A de marge).")
        else:
            print(f"RECOUVREMENT de {-gap:.2f} A : la regle ne separe pas les deux groupes.")

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {len(rows)} lignes dans {OUT}")


if __name__ == "__main__":
    main()
