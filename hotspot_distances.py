#!/usr/bin/env python3
"""Distances CA entre residus hotspot candidats, 6ARU chaine A.

Le choix d'un jeu de hotspots repose sur ces distances : un jeu doit etre
atteignable par UN SEUL binder. Ce script les recalcule sans PyMOL, pour que la
decision soit rejouable depuis un commit.

Les valeurs ont ete recoupees avec PyMOL 3 en headless (`cmd.get_distance` sur
`chain A and resi N and name CA`). Tout ecart signalerait un probleme de lecture
de la numerotation auteur du mmCIF.

Verifie l'existence de chaque residu avant de mesurer : une selection vide
donnerait silencieusement une distance aberrante.

Usage : python hotspot_distances.py [resi ...]
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

import egfr_epitope_map as E

DEFAULT = [318, 323, 325, 357, 406, 409, 434, 463]


def main() -> None:
    wanted = [int(a) for a in sys.argv[1:]] or DEFAULT
    entry = E.load_uniprot(E.HUMAN_AC)
    residues, _, _, _ = E.load_chain(entry["sequence"]["value"])
    by = {r.id[1]: r for r in residues}

    missing = [n for n in wanted if n not in by or "CA" not in by[n]]
    if missing:
        raise SystemExit(f"residus absents ou sans CA : {missing}")
    ca = {n: np.asarray(by[n]["CA"].coord, dtype=float) for n in wanted}
    print(f"  {len(wanted)} residus, tous presents avec un CA")
    for n in wanted:
        print(f"    {E.AA3TO1[by[n].get_resname()]}{n}")

    print("\n  matrice des distances CA (A)")
    print("        " + "".join(f"{n:>7}" for n in wanted))
    for a in wanted:
        row = f"  {a:<6}"
        for b in wanted:
            row += f"{np.linalg.norm(ca[a] - ca[b]):>7.1f}"
        print(row)

    print("\n  etendue maximale de chaque sous-ensemble de taille >= 3")
    spans = []
    for k in range(3, len(wanted) + 1):
        for combo in combinations(wanted, k):
            span = max(
                float(np.linalg.norm(ca[a] - ca[b]))
                for a, b in combinations(combo, 2)
            )
            spans.append((span, combo))
    spans.sort()
    for span, combo in spans[:12]:
        print(f"    {span:>5.1f} A   {','.join(f'A{n}' for n in combo)}")


if __name__ == "__main__":
    main()
