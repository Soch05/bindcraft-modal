#!/usr/bin/env python3
"""
EGFR - carte d'epitope pour le Challenge 1 (Anthropic x Adaptyv 2026).

Construit 4 masques sur la surface du domaine III et les croise :
  1. conservation humain (P00533) / souris (Q01279)   -> objectif "cross-reactivite"
  2. exposition au solvant (SASA relative sur 6ARU chaine A, ectodomaine entier)
  3. ancrage acide : Asp/Glu conserves et exposes      -> partenaire du switch His
  4. exclusion des N-glycanes annotes dans UniProt

Sorties :
  data/egfr_residues.csv          une ligne par residu de la region cible
  data/6aru_chainA_conserv.pdb    B-factor = score de conservation (PyMOL: spectrum b)
  stdout                          controle cetuximab + patches candidats dedupliques

La numerotation PDB est deduite par alignement de la chaine A observee sur la
sequence UniProt : ne jamais supposer l'offset du peptide signal.

Delimitation de la region cible
-------------------------------
UniProt n'annote aucun "Receptor L-domain" sur P00533 : la seule feature de type
Domain est la kinase (712-979), et les L-domaines n'apparaissent que comme Repeat
"Approximate" (75-300, 390-600). La seconde, 390-600 UniProt = 366-576 mature, est
decalee d'une cinquantaine de residus par rapport au domaine III structural : elle
n'est pas utilisable comme borne, et elle reste ici purement informative.

La region est donc definie *structurellement*, depuis l'empreinte du Fab cetuximab
(chaines B et C de 6ARU) : les residus dont le CA est a moins de REGION_RADIUS du
centroide de cette empreinte. Aucune borne codee de memoire.

Conformation
------------
6ARU est en conformation repliee (tethered) : le bras de dimerisation du domaine II
contacte le domaine IV. Le script le verifie et l'affiche. La SASA etant calculee
sur la chaine A entiere, l'occlusion par les autres domaines dans cet etat est
prise en compte ; un patch retenu ici est expose dans l'etat repliee.

Usage:  python egfr_epitope_map.py
Deps :  pip install biopython numpy
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HUMAN_AC = "P00533"
MOUSE_AC = "Q01279"
PDB_ID = "6ARU"
PDB_CHAIN = "A"
FAB_CHAINS = ("B", "C")  # Fab cetuximab dans 6ARU

MIN_REL_SASA = 0.20  # seuil d'exposition d'un residu de surface
PATCH_RADIUS = 11.0  # rayon du patch autour de l'ancre acide (Angstroms)
GLYCAN_EXCLUSION = 12.0  # distance minimale a un site de N-glycosylation
CONTACT_CUTOFF = 4.5  # contact lourd Fab <-> cible
TETHER_MIN_SEQ_SEP = 150  # |delta seq| minimal pour un contact "longue portee"

# Deduplication : deux patches partageant plus que cette fraction de leurs membres
# decrivent le meme site. On compare les ensembles de membres et non la distance
# entre ancres : a PATCH_RADIUS = 11, deux ancres separees de 8 A partagent encore
# les deux tiers de leur patch (constate sur E320 / D323).
MAX_PATCH_OVERLAP = 0.5

# Rayon de la region cible autour du centroide de l'empreinte du Fab. 30 A est le
# plus petit rayon qui rende la selection contigue en sequence sur 6ARU : 311-511
# avec 1 discontinuite, contre 7 a 25 A. Critere structural, pas une constante
# choisie a la main. Toute modification va dans NOTES.md.
REGION_RADIUS = 30.0

# Proxy de designabilite : le design de novo reussit mieux sur les patches
# apolaires. Hypothese de travail, non calibree sur ce projet.
HYDROPHOBIC = set("LIVFMWY")

DATA = Path("data")
DATA.mkdir(exist_ok=True)

# Tien et al. 2013, valeurs theoriques d'ASA maximale
MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225,
    "G": 104, "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240,
    "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C",
}


# --------------------------------------------------------------------------- #
# Telechargements (avec cache disque)
# --------------------------------------------------------------------------- #
def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  telechargement {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_uniprot(accession: str) -> dict:
    path = fetch(
        f"https://rest.uniprot.org/uniprotkb/{accession}.json",
        DATA / f"{accession}.json",
    )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# 1. Alignement humain / souris
# --------------------------------------------------------------------------- #
def align_orthologs(seq_h: str, seq_m: str):
    """Alignement global des deux precurseurs entiers, puis mapping position a position.

    On aligne les sequences completes et on decoupe ensuite : decouper avant
    l'alignement revient a supposer les bornes du domaine des deux cotes.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.end_gap_score = 0.0  # gaps terminaux gratuits

    aln = aligner.align(seq_h, seq_m)[0]
    blosum = aligner.substitution_matrix

    mapping = {}  # position humaine 1-based -> (aa_souris, statut)
    for (h_start, h_end), (m_start, m_end) in zip(*aln.aligned):
        for off in range(h_end - h_start):
            h_pos = h_start + off + 1
            aa_h, aa_m = seq_h[h_start + off], seq_m[m_start + off]
            if aa_h == aa_m:
                status = "identical"
            elif aa_h in MAX_ASA and aa_m in MAX_ASA and blosum[aa_h, aa_m] > 0:
                status = "similar"
            else:
                status = "different"
            mapping[h_pos] = (aa_m, status)

    ident = sum(1 for v in mapping.values() if v[1] == "identical")
    print(f"  identite globale sur les residus alignes : {ident / len(mapping):.1%}")
    return mapping


# --------------------------------------------------------------------------- #
# 2. Annotations UniProt : domaine III et N-glycanes
# --------------------------------------------------------------------------- #
def uniprot_features(entry: dict, ftype: str) -> list[tuple[str, int, int]]:
    out = []
    for feat in entry.get("features", []):
        if feat["type"] != ftype:
            continue
        loc = feat["location"]
        start, end = loc["start"].get("value"), loc["end"].get("value")
        if start is None or end is None:
            continue
        out.append((feat.get("description", ""), start, end))
    return out


def annotated_repeat(entry: dict) -> tuple[int, int] | None:
    """Seconde Repeat "Approximate" d'UniProt, affichee pour comparaison seulement.

    Ce n'est PAS le domaine III : sur P00533 elle couvre 390-600 UniProt, soit
    366-576 mature, decale d'une cinquantaine de residus aux deux bouts. On la
    reporte en colonne pour pouvoir constater l'ecart, jamais pour filtrer.
    """
    for ftype in ("Domain", "Repeat"):
        feats = uniprot_features(entry, ftype)
        if feats:
            print(f"  features {ftype} (numerotation UniProt) :")
            for desc, s, e in feats:
                print(f"    {desc or '(sans description)':<28} {s:>4}-{e:<4}")
    repeats = uniprot_features(entry, "Repeat")
    return (repeats[1][1], repeats[1][2]) if len(repeats) >= 2 else None


# --------------------------------------------------------------------------- #
# 3. Structure : numerotation, SASA
# --------------------------------------------------------------------------- #
class ChainOnly(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "


def fab_footprint(model) -> dict[int, float]:
    """Residus de la cible en contact avec le Fab, sous CONTACT_CUTOFF.

    Appele AVANT de detacher le Fab. Double usage : controle positif du masque de
    conservation (une proteine qui se lie vraiment au domaine III), et systeme de
    coordonnees pour delimiter la region cible.
    """
    fab = [
        atom
        for cid in FAB_CHAINS
        for res in model[cid]
        if res.id[0] == " "
        for atom in res
        if atom.element != "H"
    ]
    if not fab:
        raise SystemExit(
            f"Chaines {FAB_CHAINS} absentes de {PDB_ID} : empreinte impossible."
        )
    search = NeighborSearch(fab)
    foot: dict[int, float] = {}
    for res in model[PDB_CHAIN]:
        if res.id[0] != " ":
            continue
        best = float("inf")
        for atom in res:
            if atom.element == "H":
                continue
            for other in search.search(atom.coord, CONTACT_CUTOFF):
                best = min(best, float(np.linalg.norm(atom.coord - other.coord)))
        if best < CONTACT_CUTOFF:
            foot[res.id[1]] = round(best, 2)
    if not foot:
        raise SystemExit("Empreinte vide : verifier les identifiants de chaines.")
    return foot


def check_tethered(residues) -> None:
    """Determine l'etat conformationnel par les contacts longue portee en sequence.

    En conformation repliee le bras de dimerisation du domaine II contacte le
    domaine IV. On ne cite aucun numero de residu : on cherche les paires en
    contact dont les positions sont eloignees de plus de TETHER_MIN_SEQ_SEP.
    """
    atoms = [a for r in residues for a in r if a.element != "H"]
    search = NeighborSearch(atoms)
    pairs = {
        (res.id[1], other.get_parent().id[1])
        for res in residues
        for atom in res
        if atom.element != "H"
        for other in search.search(atom.coord, 5.0)
        if other.get_parent().id[1] - res.id[1] > TETHER_MIN_SEQ_SEP
    }
    if not pairs:
        print("  aucun contact longue portee -> conformation etendue")
        return

    # Regrouper par segment contigu cote bas : un min-max global melangerait le
    # tether domaine II <-> domaine IV avec l'empilement domaine I <-> domaine II.
    clusters: list[tuple[list[int], list[int]]] = []
    for low in sorted({i for i, _ in pairs}):
        partners = sorted({j for i, j in pairs if i == low})
        if clusters and low - clusters[-1][0][-1] <= 10:
            clusters[-1][0].append(low)
            clusters[-1][1].extend(partners)
        else:
            clusters.append(([low], list(partners)))

    print(f"  {len(pairs)} contacts longue portee, {len(clusters)} segment(s) :")
    for lows, highs in clusters:
        print(f"    {lows[0]}-{lows[-1]} <-> {min(highs)}-{max(highs)}")
    print("  -> conformation repliee (tethered)")


def region_mask(residues, foot: dict[int, float]) -> set[int]:
    """Region cible : CA a moins de REGION_RADIUS du centroide de l'empreinte."""
    centre = np.mean(
        [r["CA"].coord for r in residues if r.id[1] in foot and "CA" in r], axis=0
    )
    sel = {
        r.id[1]
        for r in residues
        if "CA" in r
        and float(np.linalg.norm(r["CA"].coord - centre)) <= REGION_RADIUS
    }
    ordered = sorted(sel)
    gaps = sum(1 for a, b in zip(ordered, ordered[1:]) if b - a > 1)
    print(
        f"  region : {len(sel)} residus, {ordered[0]}-{ordered[-1]} "
        f"(num. PDB), {gaps} discontinuite(s) de sequence"
    )
    return sel


def load_chain(seq_h: str):
    path = fetch(
        f"https://files.rcsb.org/download/{PDB_ID}.cif", DATA / f"{PDB_ID}.cif"
    )
    structure = MMCIFParser(QUIET=True).get_structure(PDB_ID, path)
    model = structure[0]

    foot = fab_footprint(model)
    print(f"  empreinte du Fab : {len(foot)} residus sous {CONTACT_CUTOFF} A")

    # on ne garde que la chaine cible, sans heteroatomes ni eau
    for chain in list(model):
        if chain.id != PDB_CHAIN:
            model.detach_child(chain.id)
    chain = model[PDB_CHAIN]
    for res in list(chain):
        if res.id[0] != " " or res.get_resname() not in AA3TO1:
            chain.detach_child(res.id)

    residues = [r for r in chain]
    print(f"  {len(residues)} residus observes dans la chaine {PDB_CHAIN}")

    # offset PDB -> UniProt, deduit par comptage de correspondances
    best_offset, best_score = None, -1
    for offset in range(-60, 61):
        score = 0
        for res in residues:
            u = res.id[1] + offset
            if 1 <= u <= len(seq_h) and seq_h[u - 1] == AA3TO1[res.get_resname()]:
                score += 1
        if score > best_score:
            best_offset, best_score = offset, score
    frac = best_score / len(residues)
    print(f"  offset PDB -> UniProt = {best_offset:+d}  ({frac:.1%} de concordance)")
    if frac < 0.95:
        raise SystemExit("Offset peu fiable : inspecter la chaine a la main.")

    ShrakeRupley().compute(model, level="R")
    return residues, best_offset, foot


# --------------------------------------------------------------------------- #
# 4. Assemblage des masques
# --------------------------------------------------------------------------- #
def anchor_atom(res):
    return res["CB"] if "CB" in res else res["CA"]


def build_table(residues, offset, mapping, region, repeat2, foot, glyc_sites):
    glyc_coords = [
        anchor_atom(res).coord
        for res in residues
        if res.id[1] + offset in glyc_sites and "CA" in res
    ]
    if not glyc_coords:
        raise SystemExit(
            "Aucun site de N-glycosylation ne tombe sur un residu observe. Le masque "
            "glycane serait inoperant et laisserait passer tous les residus : arret "
            "plutot qu'un repli silencieux sur une distance infinie."
        )

    rows = []
    for res in residues:
        aa_h = AA3TO1[res.get_resname()]
        uni = res.id[1] + offset
        aa_m, status = mapping.get(uni, ("-", "gap"))
        rel_sasa = res.sasa / MAX_ASA[aa_h]
        d_glyc = min(
            float(np.linalg.norm(anchor_atom(res).coord - c)) for c in glyc_coords
        )
        rows.append(
            {
                "pdb_resnum": res.id[1],
                "uniprot_pos": uni,
                "aa_human": aa_h,
                "aa_mouse": aa_m,
                "status": status,
                "rel_sasa": round(float(rel_sasa), 3),
                "in_region": res.id[1] in region,
                # informatif seulement : l'annotation UniProt ne delimite pas le domaine III
                "uniprot_repeat2": bool(repeat2 and repeat2[0] <= uni <= repeat2[1]),
                "fab_contact": res.id[1] in foot,
                "acidic": aa_h in "DE",
                "hydrophobic": aa_h in HYDROPHOBIC,
                "dist_glycan": round(float(d_glyc), 1),
                "exposed": bool(rel_sasa >= MIN_REL_SASA),
            }
        )
    return rows


def rank_patches(rows, residues):
    by_pdb = {r.id[1]: r for r in residues}

    anchors = [
        row
        for row in rows
        if row["in_region"]
        and row["acidic"]
        and row["status"] == "identical"
        and row["exposed"]
        and row["dist_glycan"] > GLYCAN_EXCLUSION
    ]
    print(f"  {len(anchors)} ancres acides conservees et exposees dans la region")

    patches = []
    for anchor in anchors:
        centre = anchor_atom(by_pdb[anchor["pdb_resnum"]]).coord
        members = []
        for row in rows:
            if not (row["in_region"] and row["exposed"]):
                continue
            d = float(np.linalg.norm(anchor_atom(by_pdb[row["pdb_resnum"]]).coord - centre))
            if d <= PATCH_RADIUS:
                members.append((d, row))
        if len(members) < 6:
            continue
        members.sort(key=lambda t: t[0])
        n_ident = sum(1 for _, r in members if r["status"] == "identical")
        n_diff = sum(1 for _, r in members if r["status"] in ("different", "gap"))
        patches.append(
            {
                "anchor_num": anchor["pdb_resnum"],
                "anchor": f"{anchor['aa_human']}{anchor['pdb_resnum']}",
                "centre": centre,
                "member_set": frozenset(r["pdb_resnum"] for _, r in members),
                "n": len(members),
                "frac_ident": n_ident / len(members),
                "n_diff": n_diff,
                # proxy de designabilite, et mesure de la tension avec l'ancrage acide
                "n_hydro": sum(1 for _, r in members if r["hydrophobic"]),
                "n_acidic": sum(1 for _, r in members if r["acidic"]),
                "n_fab": sum(1 for _, r in members if r["fab_contact"]),
                "min_glyc": min(r["dist_glycan"] for _, r in members),
                "hotspots": [r["pdb_resnum"] for _, r in members[:4]],
                "members": [
                    f"{r['aa_human']}{r['pdb_resnum']}"
                    + ("" if r["status"] == "identical" else f"({r['aa_mouse']})")
                    for _, r in members
                ],
            }
        )

    patches.sort(key=lambda p: (-p["frac_ident"], -p["n"]))

    # Le seuil glycane doit porter sur le patch entier, pas sur la seule ancre :
    # une ancre a 20 A d'un glycane peut avoir des membres a 6 A, et c'est le patch
    # complet qui sera dans l'ombre du glycane a l'experience.
    shadowed = [p for p in patches if p["min_glyc"] <= GLYCAN_EXCLUSION]
    patches = [p for p in patches if p["min_glyc"] > GLYCAN_EXCLUSION]
    if shadowed:
        detail = ", ".join(f"{p['anchor']} ({p['min_glyc']:.0f} A)" for p in shadowed)
        print(
            f"  {len(shadowed)} patches ecartes, membres a moins de "
            f"{GLYCAN_EXCLUSION:.0f} A d'un N-glycane : {detail}"
        )

    # Deduplication gloutonne sur le recouvrement des membres.
    kept: list[dict] = []
    for patch in patches:
        if all(
            len(patch["member_set"] & k["member_set"])
            / min(len(patch["member_set"]), len(k["member_set"]))
            <= MAX_PATCH_OVERLAP
            for k in kept
        ):
            kept.append(patch)
    print(f"  {len(patches)} patches -> {len(kept)} sites distincts apres deduplication")
    return kept


def cetuximab_control(foot, offset, mapping) -> None:
    """Controle positif : conservation humain/souris sur l'empreinte du cetuximab.

    Attendu INVERSE de l'intuition. Le cetuximab ne reconnait pas l'EGFR murin :
    une fonction de score saine doit donc classer son epitope BAS. Si l'empreinte
    ressort fortement conservee, c'est le masque de conservation qui ne mesure rien
    (offset faux, alignement casse), pas une bonne nouvelle.
    """
    status = []
    for pdb_num in sorted(foot):
        aa_m, st = mapping.get(pdb_num + offset, ("-", "gap"))
        status.append((pdb_num, aa_m, st))
    n_ident = sum(1 for _, _, st in status if st == "identical")
    frac = n_ident / len(status)

    print(f"  empreinte : {len(status)} residus, {min(foot)}-{max(foot)} (num. PDB)")
    print(f"  identite humain/souris sur l'empreinte : {frac:.1%}")
    diverg = [f"{p}->{aa}" for p, aa, st in status if st != "identical"]
    print(f"  positions divergentes : {', '.join(diverg) if diverg else 'aucune'}")

    if frac >= 0.90:
        print(
            "  !! empreinte quasi totalement conservee. Le cetuximab ne reconnaissant\n"
            "     pas l'EGFR murin, verifier l'offset et l'alignement avant de se fier\n"
            "     au classement : le masque de conservation ne discrimine peut-etre rien."
        )
    else:
        print(
            "  -> l'empreinte porte des divergences humain/souris, coherent avec "
            "l'absence\n     de cross-reactivite du cetuximab. Masque de conservation "
            "operant."
        )


def write_bfactor_pdb(residues, rows):
    score = {
        "identical": 100.0,
        "similar": 50.0,
        "different": 0.0,
        "gap": 0.0,
    }
    lookup = {r["pdb_resnum"]: score[r["status"]] for r in rows}
    for res in residues:
        for atom in res:
            atom.bfactor = lookup.get(res.id[1], 0.0)
    io = PDBIO()
    io.set_structure(residues[0].get_parent().get_parent().get_parent())
    out = DATA / "6aru_chainA_conserv.pdb"
    io.save(str(out), ChainOnly(PDB_CHAIN))
    return out


# --------------------------------------------------------------------------- #
def main():
    print("[1/5] UniProt")
    h_entry, m_entry = load_uniprot(HUMAN_AC), load_uniprot(MOUSE_AC)
    seq_h = h_entry["sequence"]["value"]
    seq_m = m_entry["sequence"]["value"]
    print(f"  humain {len(seq_h)} aa | souris {len(seq_m)} aa")
    repeat2 = annotated_repeat(h_entry)
    print(
        f"  Repeat 2 (informative, PAS le domaine III) : "
        f"{repeat2[0]}-{repeat2[1]} UniProt" if repeat2 else "  aucune Repeat annotee"
    )
    # N-linked seulement, conformement au docstring. Les sites murins sont un
    # sous-ensemble strict des sites humains sur ces deux entrees (verifie), donc
    # masquer sur l'humain couvre aussi la souris.
    glyc_sites = {
        s
        for desc, s, _ in uniprot_features(h_entry, "Glycosylation")
        if desc.startswith("N-linked")
    }
    print(f"  {len(glyc_sites)} sites de N-glycosylation annotes (humain)")

    print("\n[2/5] Alignement humain / souris")
    mapping = align_orthologs(seq_h, seq_m)

    print(f"\n[3/5] Structure {PDB_ID} chaine {PDB_CHAIN}")
    residues, offset, foot = load_chain(seq_h)
    check_tethered(residues)

    print("\n[4/5] Controle cetuximab")
    cetuximab_control(foot, offset, mapping)
    region = region_mask(residues, foot)

    print("\n[5/5] Masques et patches")
    rows = build_table(residues, offset, mapping, region, repeat2, foot, glyc_sites)
    in_region = [r for r in rows if r["in_region"]]
    n_ident = sum(1 for r in in_region if r["status"] == "identical")
    print(f"  identite humain/souris sur la region : {n_ident / len(in_region):.1%}")

    csv_path = DATA / "egfr_residues.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(in_region)

    pdb_path = write_bfactor_pdb(residues, rows)
    patches = rank_patches(rows, residues)

    print("\n" + "=" * 78)
    print("SITES CANDIDATS (numerotation PDB, celle attendue par BindCraft)")
    print("=" * 78)
    for p in patches[:10]:
        print(
            f"\nancre {p['anchor']:<6} | {p['n']:>2} exposes | "
            f"{p['frac_ident']:.0%} ident. | {p['n_diff']} diverg. | "
            f"{p['n_hydro']} hydrophobes | {p['n_acidic']} acides | "
            f"{p['n_fab']} dans l'empreinte Fab | glycane {p['min_glyc']:.0f} A"
        )
        print(f"  hotspots suggeres : {','.join(f'{PDB_CHAIN}{r}' for r in p['hotspots'])}")
        print(f"  patch : {' '.join(p['members'])}")

    print(f"\nCSV  : {csv_path}")
    print(f"PDB  : {pdb_path}")
    print("PyMOL: load data/6aru_chainA_conserv.pdb; spectrum b, red_white_blue, all, 0, 100")
    print(
        "\nRappel : la cle de tri reste (-frac_ident, -n), volontairement inchangee.\n"
        "Les colonnes n_hydro / n_acidic sont la pour mesurer la tension\n"
        "designabilite vs ancrage acide avant de decider d'un score composite."
    )


if __name__ == "__main__":
    main()
