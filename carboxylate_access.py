#!/usr/bin/env python3
"""Accessibilite des carboxylates des ancres acides du domaine III de l'EGFR.

Pourquoi ce script existe
-------------------------
`egfr_epitope_map.py` retient une ancre acide sur `rel_sasa`, qui est la SASA du
RESIDU ENTIER rapportee a son maximum theorique. Or ce qui doit etre accessible
pour former un pont salin avec une His de binder, ce n'est pas le residu : c'est
le CARBOXYLATE. Un Asp dont le CB est expose mais dont OD1/OD2 pointent vers
l'interieur passerait le seuil `exposed` tout en etant inatteignable.

Ce script separe les deux cas. Il ne modifie rien, il mesure.

Ce qu'il calcule
----------------
1. SASA du carboxylate seul (OD1+OD2 pour Asp, OE1+OE2 pour Glu), du CB, et du
   residu entier, pour tous les Asp/Glu observes.
2. La distribution de reference sur le domaine III : sans elle, aucun seuil n'est
   interpretable.
3. La CALIBRATION : les Asp/Glu de l'empreinte du Fab de cetuximab, c'est-a-dire
   des acides dont on sait qu'une proteine vient reellement les contacter. C'est
   le seul etalon disponible pour « accessible a un partenaire proteique ».
4. L'orientation : angle entre CB -> carboxylate et le vecteur sortant local,
   pour distinguer un carboxylate qui pointe vers le solvant d'un qui longe la
   surface.

Limites, a lire avant d'utiliser un chiffre
-------------------------------------------
- SASA calculee sur 6ARU, conformation REPLIEE, resolution 3,20 A.
- Un seul rotamere cristallographique : un carboxylate peut tourner en solution.
  La mesure dit « accessible dans cette structure », pas « accessible ».
- La sonde de 1,40 A est un rayon de molecule d'eau. SASA > 0 signifie donc
  qu'une eau peut s'approcher, pas qu'un cycle imidazole encombrant peut le faire.

Usage : python carboxylate_access.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

import egfr_epitope_map as E

CARBOXYL = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
LOCAL_RADIUS = 12.0  # rayon du voisinage servant a definir le vecteur sortant
SITES = ("G317", "N449", "K375", "C502")

# ------------------------------------------------------------------------- #
# Criteres d'ancre REELLE. Les trois sont POSES, non calibres : la calibration
# prevue a echoue, l'empreinte du cetuximab ne contenant qu'un seul Asp/Glu
# (E472), donc n = 1 et aucune distribution exploitable.
#
#   MIN_CARBOX_SASA  un cycle imidazole mesure ~4,5 x 4,0 A, soit une section de
#                    20-25 A2. Pour qu'un cycle approche la ou une seule molecule
#                    d'eau suffirait, le carboxylate doit presenter l'ordre de sa
#                    propre section. Argument geometrique, pas une mesure.
#   MIN_CARBOX_PART  le carboxylate d'un Glu est porte plus loin du squelette que
#                    celui d'un Asp et sort donc systematiquement plus. La
#                    fraction dit si c'est le groupe fonctionnel qui est expose
#                    ou seulement la tige.
#   MAX_OUTWARD_DEG  sans dimension, et c'est le vrai discriminant. Sous 60 deg le
#                    groupe pointe vers le solvant ; au-dela de 90 il longe la
#                    surface ou rentre.
#
# Limite commune : SASA calculee sur un rotamere cristallographique unique a
# 3,20 A. Un carboxylate peut tourner en solution.
# ------------------------------------------------------------------------- #
MIN_CARBOX_SASA = 25.0
MIN_CARBOX_PART = 0.40
MAX_OUTWARD_DEG = 60.0


def is_real_anchor(
    carbox: float | None, part: float | None, angle: float | None
) -> bool | None:
    """Verdict d'ancre reelle. None si une des trois mesures manque."""
    if carbox is None or part is None or angle is None:
        return None
    return (
        carbox >= MIN_CARBOX_SASA
        and part >= MIN_CARBOX_PART
        and angle <= MAX_OUTWARD_DEG
    )


def carboxylate_sasa(res) -> float | None:
    """SASA cumulee des deux oxygenes du carboxylate. None si atomes absents."""
    names = CARBOXYL.get(res.get_resname())
    if names is None:
        return None
    if not all(n in res for n in names):
        return None
    return float(sum(res[n].sasa for n in names))


def outward_angle(res, residues) -> float | None:
    """Angle en degres entre CB -> carboxylate et le vecteur sortant local.

    Vecteur sortant approxime par centroide local -> carboxylate : un angle faible
    signifie que le groupe pointe vers le solvant, un angle eleve qu'il longe la
    surface ou rentre.
    """
    names = CARBOXYL.get(res.get_resname())
    if names is None or "CB" not in res or not all(n in res for n in names):
        return None
    carb = np.mean([res[n].coord for n in names], axis=0)
    cb = res["CB"].coord
    near = [
        r["CA"].coord
        for r in residues
        if "CA" in r and float(np.linalg.norm(r["CA"].coord - carb)) <= LOCAL_RADIUS
    ]
    if len(near) < 4:
        return None
    outward = carb - np.mean(near, axis=0)
    v = carb - cb
    nv, no = np.linalg.norm(v), np.linalg.norm(outward)
    if nv == 0 or no == 0:
        return None
    cos = float(np.clip(np.dot(v, outward) / (nv * no), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def site_anchors() -> dict[str, list[int]]:
    """Ancres acides conservees de chaque site, lues dans les CSV publies."""
    pat = {r["centre"]: r for r in csv.DictReader(open(E.DATA / "egfr_patches.csv"))}
    res = {
        int(r["pdb_resnum"]): r
        for r in csv.DictReader(open(E.DATA / "egfr_residues.csv"))
    }
    out: dict[str, list[int]] = {}
    for site in SITES:
        if site not in pat:
            continue
        members = [int(x) for x in re.findall(r"\d+", pat[site]["member_resnums_full"])]
        out[site] = [
            n
            for n in members
            if n in res and res[n]["acidic"] == "True" and res[n]["status"] == "identical"
        ]
    return out


def describe(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label:<34} (aucune valeur)")
        return
    a = np.array(values)
    print(
        f"  {label:<34} n={len(a):>3}  min {a.min():>5.1f}  p25 {np.percentile(a,25):>5.1f}"
        f"  median {np.percentile(a,50):>5.1f}  p75 {np.percentile(a,75):>5.1f}"
        f"  max {a.max():>5.1f}"
    )


def main() -> None:
    entry = E.load_uniprot(E.HUMAN_AC)
    seq_h = entry["sequence"]["value"]
    # load_chain fait la preparation identique au pipeline et calcule la SASA
    # au niveau atomique : on reutilise, on ne reimplemente pas.
    residues, offset, foot, _ = E.load_chain(seq_h)
    dom3 = E.resolve_domain_iii({n + offset for n in foot}, offset)
    lo, hi = dom3[0] - offset, dom3[1] - offset

    rows = []
    for res in residues:
        carb = carboxylate_sasa(res)
        if carb is None:
            continue
        num = res.id[1]
        rows.append(
            {
                "num": num,
                "aa": E.AA3TO1[res.get_resname()],
                "carb": carb,
                "cb": float(res["CB"].sasa) if "CB" in res else 0.0,
                "res": float(sum(a.sasa for a in res)),
                "angle": outward_angle(res, residues),
                "in_dom3": lo <= num <= hi,
                "in_foot": num in foot,
            }
        )
    print(f"\n{len(rows)} Asp/Glu avec carboxylate complet sur la chaine A")

    print("\n" + "=" * 78)
    print("1. DISTRIBUTION DE REFERENCE — SASA du carboxylate (A2)")
    print("=" * 78)
    describe("chaine A entiere", [r["carb"] for r in rows])
    describe("domaine III", [r["carb"] for r in rows if r["in_dom3"]])
    describe("carboxylates a SASA nulle", [r["carb"] for r in rows if r["carb"] == 0])
    nz = [r for r in rows if r["carb"] > 0]
    print(
        f"  carboxylates strictement enfouis (SASA = 0) : "
        f"{len(rows) - len(nz)}/{len(rows)} ({(len(rows)-len(nz))/len(rows):.0%})"
    )

    print("\n" + "=" * 78)
    print("2. CALIBRATION — Asp/Glu de l'empreinte du Fab de cetuximab")
    print("=" * 78)
    foot_acids = [r for r in rows if r["in_foot"]]
    print(
        "  Ces acides sont contactes par une proteine reelle. Leur SASA de\n"
        "  carboxylate donne l'etalon « atteignable par un partenaire proteique »."
    )
    describe("acides de l'empreinte", [r["carb"] for r in foot_acids])
    print(f"  {'res':<6}{'carbox':>8}{'CB':>7}{'residu':>8}{'part':>7}{'angle':>7}")
    for r in sorted(foot_acids, key=lambda r: -r["carb"]):
        ang = f"{r['angle']:.0f}" if r["angle"] is not None else "-"
        part = r["carb"] / r["res"] if r["res"] else 0.0
        print(
            f"  {r['aa']}{r['num']:<5}{r['carb']:>8.1f}{r['cb']:>7.1f}"
            f"{r['res']:>8.1f}{part:>7.0%}{ang:>7}"
        )

    print("\n" + "=" * 78)
    print("3. ANCRES ACIDES CONSERVEES DES QUATRE SITES")
    print("=" * 78)
    by_num = {r["num"]: r for r in rows}
    for site, anchors in site_anchors().items():
        print(f"\n  site {site}")
        if not anchors:
            print("    aucune ancre acide conservee")
            continue
        print(f"    {'res':<6}{'carbox':>8}{'CB':>7}{'residu':>8}{'part':>7}{'angle':>7}")
        for n in sorted(anchors):
            r = by_num.get(n)
            if r is None:
                print(f"    {n:<6} carboxylate incomplet dans la structure")
                continue
            ang = f"{r['angle']:.0f}" if r["angle"] is not None else "-"
            part = r["carb"] / r["res"] if r["res"] else 0.0
            print(
                f"    {r['aa']}{r['num']:<5}{r['carb']:>8.1f}{r['cb']:>7.1f}"
                f"{r['res']:>8.1f}{part:>7.0%}{ang:>7}"
            )

    print("\n" + "=" * 78)
    print("4. OU SE SITUENT LES ANCRES DANS LA DISTRIBUTION")
    print("=" * 78)
    d3 = np.array([r["carb"] for r in rows if r["in_dom3"]])
    footv = np.array([r["carb"] for r in foot_acids])
    for site, anchors in site_anchors().items():
        for n in sorted(anchors):
            r = by_num.get(n)
            if r is None:
                continue
            pct_d3 = float((d3 < r["carb"]).mean() * 100)
            pct_ft = float((footv < r["carb"]).mean() * 100) if footv.size else float("nan")
            print(
                f"  {site:<6} {r['aa']}{r['num']:<5} carbox={r['carb']:>6.1f} A2  "
                f"centile domaine III = {pct_d3:>3.0f}  "
                f"centile empreinte = {pct_ft:>3.0f}"
            )


if __name__ == "__main__":
    main()
