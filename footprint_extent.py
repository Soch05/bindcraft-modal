#!/usr/bin/env python3
"""Etendue spatiale de l'empreinte du Fab de cetuximab sur 6ARU.

Objet : situer l'ordre de grandeur de PATCH_RADIUS contre une empreinte reelle.
LECTURE SEULE. Ne modifie aucune constante, ne relance pas le pipeline, n'ecrit
aucun CSV. Le seul effet de bord est l'impression du resultat.

Atomes d'ancrage identiques a ceux d'enumerate_patches : CB, ou CA quand il n'y a
pas de CB (glycines, chaines laterales incompletes).

Quatre grandeurs
---------------
1. diametre : distance maximale entre deux residus de l'empreinte
2. rayon de la plus petite sphere englobante, centre libre dans l'espace
3. couverture a PATCH_RADIUS : pour chacun des 24 residus pris comme centre,
   combien des 23 autres tombent dans le rayon
4. rayon minimal couvrant les 24 depuis un centre contraint a etre l'un d'eux

La sphere englobante est calculee exactement : avec 24 points, la sphere minimale
est determinee par 2, 3 ou 4 points de sa frontiere, et l'enumeration de tous ces
sous-ensembles est triviale a cette taille.

Usage : python footprint_extent.py
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

import egfr_epitope_map as E

EPS = 1e-6


def circumcentre(points: np.ndarray) -> np.ndarray | None:
    """Centre equidistant de 2, 3 ou 4 points. None si degenere."""
    p0 = points[0]
    rest = points[1:]
    if len(points) == 2:
        return (points[0] + points[1]) / 2.0
    # systeme lineaire : |c - p0|^2 = |c - pi|^2  =>  2(pi - p0).c = |pi|^2 - |p0|^2
    a = 2.0 * (rest - p0)
    b = np.sum(rest**2, axis=1) - np.sum(p0**2)
    if len(points) == 4:
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
    # 3 points : contraindre le centre au plan du triangle
    normal = np.cross(rest[0] - p0, rest[1] - p0)
    norm = np.linalg.norm(normal)
    if norm < EPS:
        return None
    a_full = np.vstack([a, normal])
    b_full = np.append(b, float(np.dot(normal, p0)))
    try:
        return np.linalg.solve(a_full, b_full)
    except np.linalg.LinAlgError:
        return None


def smallest_enclosing_sphere(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Sphere minimale englobante, exacte par enumeration des frontieres."""
    best_c, best_r = None, float("inf")
    n = len(pts)
    for k in (2, 3, 4):
        if n < k:
            continue
        for idx in combinations(range(n), k):
            c = circumcentre(pts[list(idx)])
            if c is None:
                continue
            r = float(np.max(np.linalg.norm(pts - c, axis=1)))
            if r < best_r:
                best_c, best_r = c, r
    return best_c, best_r


def main() -> None:
    entry = E.load_uniprot(E.HUMAN_AC)
    residues, offset, foot, _ = E.load_chain(entry["sequence"]["value"])

    anchors: dict[int, np.ndarray] = {}
    for res in residues:
        if res.id[1] in foot:
            try:
                anchors[res.id[1]] = np.asarray(E.anchor_atom(res).coord, dtype=float)
            except KeyError:
                print(f"  !! residu {res.id[1]} sans CB ni CA, exclu")
    nums = sorted(anchors)
    pts = np.array([anchors[n] for n in nums])
    print(f"\nempreinte : {len(nums)} residus avec atome d'ancrage, {min(nums)}-{max(nums)}")

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

    print("\n" + "=" * 70)
    print("1. DIAMETRE DE L'EMPREINTE")
    print("=" * 70)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    print(f"  distance maximale entre deux residus : {d[i, j]:.1f} A")
    print(f"  paire la plus eloignee               : {nums[i]} <-> {nums[j]}")

    print("\n" + "=" * 70)
    print("2. PLUS PETITE SPHERE ENGLOBANTE (centre libre)")
    print("=" * 70)
    _, r_free = smallest_enclosing_sphere(pts)
    print(f"  rayon : {r_free:.1f} A")
    print(f"  (soit un diametre de {2 * r_free:.1f} A)")

    print("\n" + "=" * 70)
    print(f"3. COUVERTURE A PATCH_RADIUS = {E.PATCH_RADIUS:.0f} A")
    print("=" * 70)
    print("  pour chaque residu pris comme centre, combien des 23 autres tombent dedans")
    counts = [(n, int((d[k] <= E.PATCH_RADIUS).sum()) - 1) for k, n in enumerate(nums)]
    vals = np.array([c for _, c in counts])
    other = len(nums) - 1
    print(
        f"  mediane {np.median(vals):.0f}/{other}  "
        f"min {vals.min()}/{other}  max {vals.max()}/{other}  "
        f"moyenne {vals.mean():.1f}/{other}"
    )
    print(f"  soit une fraction mediane de l'empreinte capturee : "
          f"{(np.median(vals) + 1) / len(nums):.0%} (centre inclus)")
    print("\n  detail, trie par couverture decroissante :")
    for n, c in sorted(counts, key=lambda t: -t[1]):
        bar = "#" * c
        print(f"    {n:<5} {c:>2}/{other}  {bar}")

    print("\n" + "=" * 70)
    print("4. RAYON MINIMAL COUVRANT LES 24, CENTRE CONTRAINT A UN RESIDU")
    print("=" * 70)
    maxima = d.max(axis=1)
    k = int(np.argmin(maxima))
    print(f"  meilleur centre : {nums[k]}")
    print(f"  rayon necessaire : {maxima[k]:.1f} A")
    print(f"  a comparer a PATCH_RADIUS = {E.PATCH_RADIUS:.0f} A "
          f"(facteur {maxima[k] / E.PATCH_RADIUS:.1f})")
    print("\n  les cinq meilleurs centres :")
    for kk in np.argsort(maxima)[:5]:
        print(f"    {nums[kk]:<5} rayon {maxima[kk]:.1f} A")

    print("\n" + "=" * 70)
    print("RESERVES")
    print("=" * 70)
    print(
        "  - Un Fab fait ~50 kDa sur deux chaines et couvre une zone plus large\n"
        "    qu'un minibinder de 83 aa : cette mesure SURESTIME le rayon pertinent.\n"
        "  - Ordre de grandeur, pas calibration. Ne distingue pas 9 de 11 de 13 A.\n"
        "  - PATCH_RADIUS reste POSE, NON CALIBRE. Aucune valeur n'est proposee ici."
    )


if __name__ == "__main__":
    main()
