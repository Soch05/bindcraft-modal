#!/usr/bin/env python3
"""
EGFR - carte d'epitope pour le Challenge 1 (Anthropic x Adaptyv 2026).

Enumere les patches de surface du domaine III de l'EGFR humain et les decrit
sur les axes qui comptent pour le challenge : conservation humain/souris
(objectif cross-reactivite), contenu apolaire expose (designabilite de novo),
presence d'ancres acides conservees (partenaire du switch His/pH), proximite
des N-glycanes, et orientation de la face.

Ce script NE SELECTIONNE PAS de site. Il sort des colonnes et des
distributions ; les seuils de retenue sont decides ensuite, a la lecture.

Sorties :
  data/egfr_residues.csv          une ligne par residu du domaine III
  data/egfr_patches.csv           une ligne par patch, avant toute retenue
  data/6aru_chainA_conserv.pdb    B-factor = score de conservation (PyMOL: spectrum b)
  stdout                          bornes de domaine, controle cetuximab,
                                  distributions, classements

Bornes du domaine III
---------------------
Aucune borne n'est codee en dur. Trois sources sont interrogees et affichees :
  - CATH structural sur 6ARU via PDBe   (/pdbe/api/mappings/cath/)
  - Pfam PF01030 "Receptor L domain"    (InterPro, numerotation UniProt)
  - CATH-Gene3D G3DSA:3.80.20.20        (InterPro, numerotation UniProt)
Le domaine retenu est, dans la source choisie, celui qui contient le plus de
residus de l'empreinte du Fab cetuximab - l'epitope du cetuximab etant dans le
domaine III. La regle est geometrique, pas nominative.

Numerotation
------------
L'offset PDB -> UniProt est deduit par balayage ET lu dans _struct_ref_seq du
mmCIF. Les deux doivent concorder, sinon le script s'arrete. Les ecarts
sequence/structure declares dans _struct_ref_seq_dif sont signales, et bruyamment
si l'un d'eux tombe dans le domaine III.

Conformation
------------
6ARU est en conformation repliee (tethered). La SASA est calculee sur la chaine A
entiere : l'occlusion par les autres domaines dans cet etat est prise en compte,
et un patch retenu ici est expose dans l'etat replie. Les heteroatomes (dont les
NAG) sont detaches avant le calcul - 2 NAG seulement sont modelises sur la chaine
A pour 13 sequons annotes, l'occlusion glycanique n'est donc pas mesurable sur
cette structure et reste traitee par la distance au sequon.

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
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
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
PFAM_AC = "PF01030"  # Receptor L domain
GENE3D_SF = "3.80.20.20"  # Receptor L-domain, superfamille CATH

MIN_REL_SASA = 0.20  # seuil d'exposition d'un residu de surface
PATCH_RADIUS = 11.0  # rayon du patch autour du centre (Angstroms)
CONTACT_CUTOFF = 4.5  # contact lourd Fab <-> cible

# La distance au sequon N-linked le plus proche est une COLONNE BRUTE, en
# Angstroms. Ni filtre, ni penalite. Une penalite lineaire clamp(d/12, 0, 1)
# appliquee multiplicativement a la SASA apolaire a ete essayee puis retiree :
# elle annulait 22 patches sur 93 a min_glyc = 0 sans ligne de journal (le filtre
# dur, en pire), et sa pente imposait un taux de change entre Angstroms et SASA
# que personne n'avait choisi (1 A = 8,3 % de la SASA apolaire). Les distances se
# lisent a la main sur la liste courte.

# Taille minimale d'un patch pour figurer au classement. La distribution complete
# des tailles est sortie avant application, ce seuil ayant ete choisi pour un
# regime d'enumeration qui n'existe plus.
MIN_PATCH_MEMBERS = 6

# Deduplication : AUCUN seuil n'est fixe. Le script sort la distribution des
# recouvrements et un balayage seuil -> nombre de sites. Le choix se fait a la
# lecture, pas ici.
DEDUP_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Proxy de designabilite conserve a titre de comparaison avec la SASA apolaire,
# qui le remplace comme cle de tri. Un comptage de residus rend invisibles les
# tiges aliphatiques des Lys/Arg, qui contribuent reellement a la surface.
HYDROPHOBIC = set("LIVFMWY")

APOLAR_ELEMENTS = {"C", "S"}
POLAR_ELEMENTS = {"N", "O"}

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
# Telechargements (cache disque, meme motif pour toutes les sources)
# --------------------------------------------------------------------------- #
def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  telechargement {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def fetch_json(url: str, name: str) -> dict | None:
    """Telecharge et met en cache un JSON. None si la source ne repond pas."""
    dest = DATA / name
    try:
        return json.loads(fetch(url, dest).read_text())
    except Exception as exc:  # source indisponible : on continue sans elle
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        print(f"  !! {name} indisponible ({exc})")
        return None


def load_uniprot(accession: str) -> dict:
    return json.loads(
        fetch(
            f"https://rest.uniprot.org/uniprotkb/{accession}.json",
            DATA / f"{accession}.json",
        ).read_text()
    )


# --------------------------------------------------------------------------- #
# 1. Bornes de domaine : trois sources, aucune borne codee en dur
# --------------------------------------------------------------------------- #
def interpro_domains(db: str, query: str, name: str) -> list[tuple[int, int]]:
    """Segments d'une base membre InterPro sur P00533, en numerotation UniProt."""
    url = f"https://www.ebi.ac.uk/interpro/api/entry/{db}/{query}/protein/uniprot/{HUMAN_AC}/?page_size=100"
    data = fetch_json(url, name)
    if data is None:
        return []
    results = data.get("results") or [data]
    segs = []
    for res in results:
        for prot in res.get("proteins", []):
            for loc in prot.get("entry_protein_locations", []):
                for frag in loc["fragments"]:
                    segs.append((frag["start"], frag["end"]))
    return sorted(set(segs))


def gene3d_superfamily_domains() -> list[tuple[int, int]]:
    """Segments CATH-Gene3D de la superfamille Receptor L-domain uniquement."""
    data = fetch_json(
        "https://www.ebi.ac.uk/interpro/api/entry/cathgene3d/protein/uniprot/"
        f"{HUMAN_AC}/?page_size=100",
        f"cathgene3d_{HUMAN_AC}.json",
    )
    if data is None:
        return []
    segs = []
    for res in data.get("results", []):
        acc = res["metadata"]["accession"]
        if not acc.endswith(GENE3D_SF):
            continue
        for prot in res.get("proteins", []):
            for loc in prot.get("entry_protein_locations", []):
                for frag in loc["fragments"]:
                    segs.append((frag["start"], frag["end"]))
    return sorted(set(segs))


def pdbe_cath_domains() -> list[tuple[int, int]]:
    """Domaines CATH sur la chaine cible de 6ARU, en numerotation auteur (PDB)."""
    data = fetch_json(
        f"https://www.ebi.ac.uk/pdbe/api/mappings/cath/{PDB_ID.lower()}",
        f"pdbe_cath_{PDB_ID}.json",
    )
    if data is None:
        return []
    segs = []
    for body in data.values():
        for entries in body.values():
            for node in entries.values():
                for m in node.get("mappings", []):
                    if m.get("chain_id") == PDB_CHAIN:
                        segs.append(
                            (
                                m["start"]["author_residue_number"],
                                m["end"]["author_residue_number"],
                            )
                        )
    return sorted(set(segs))


def pick_domain(
    segs: list[tuple[int, int]], foot_uni: set[int]
) -> tuple[int, int] | None:
    """Segment contenant le plus de residus de l'empreinte du Fab.

    L'epitope du cetuximab est dans le domaine III (etabli, cf. NOTES.md). La
    regle est donc geometrique : on ne nomme ni n'indexe aucun domaine, on prend
    celui que l'empreinte designe. Si aucun segment n'en contient, on ne choisit
    pas.
    """
    scored = [(sum(1 for u in foot_uni if s <= u <= e), (s, e)) for s, e in segs]
    scored = [x for x in scored if x[0] > 0]
    return max(scored)[1] if scored else None


def resolve_domain_iii(foot_uni: set[int], offset: int) -> tuple[int, int]:
    """Interroge les trois sources, affiche tout, retient CATH-Gene3D.

    Motif du choix, consigne dans NOTES.md :
      - PDBe CATH ne classe pas la chaine cible de 6ARU (seulement le Fab) ;
      - Pfam PF01030 declare lui-meme omettre les ~50 premiers residus du
        domaine, ses bornes sont tronquees par construction ;
      - CATH-Gene3D est la meme classification structurale projetee sur la
        sequence, et couvre le domaine entier.
    """
    cath_pdb = pdbe_cath_domains()
    if cath_pdb:
        print(f"  PDBe CATH / chaine {PDB_CHAIN} : {len(cath_pdb)} domaine(s)")
        for s, e in cath_pdb:
            print(f"    {s}-{e} (num. PDB)")
    else:
        print(
            f"  PDBe CATH : aucun domaine sur la chaine {PDB_CHAIN} de {PDB_ID} "
            "(la classification ne couvre que les chaines du Fab)"
        )

    pfam = interpro_domains("pfam", PFAM_AC, f"interpro_{PFAM_AC}_{HUMAN_AC}.json")
    pfam_pick = pick_domain(pfam, foot_uni)
    print(f"  Pfam {PFAM_AC} : {len(pfam)} segment(s) (num. UniProt)")
    for s, e in pfam:
        tag = "  <- contient l'empreinte" if (s, e) == pfam_pick else ""
        print(f"    {s}-{e}{tag}")

    g3d = gene3d_superfamily_domains()
    g3d_pick = pick_domain(g3d, foot_uni)
    print(f"  CATH-Gene3D G3DSA:{GENE3D_SF} : {len(g3d)} segment(s) (num. UniProt)")
    for s, e in g3d:
        tag = "  <- contient l'empreinte" if (s, e) == g3d_pick else ""
        print(f"    {s}-{e}{tag}")

    if g3d_pick is None:
        raise SystemExit(
            "CATH-Gene3D n'a renvoye aucun segment contenant l'empreinte du Fab : "
            "source retenue indisponible, arret plutot qu'un repli silencieux."
        )
    if pfam_pick:
        print(
            f"  ecart Pfam vs Gene3D : {pfam_pick[0] - g3d_pick[0]:+d} au N-term, "
            f"{pfam_pick[1] - g3d_pick[1]:+d} au C-term "
            "(Pfam declare omettre ~50 residus en tete du domaine)"
        )
    print(
        f"  RETENU : CATH-Gene3D {g3d_pick[0]}-{g3d_pick[1]} UniProt = "
        f"{g3d_pick[0] - offset}-{g3d_pick[1] - offset} PDB"
    )
    return g3d_pick


def other_l_domain(foot_uni: set[int]) -> tuple[int, int] | None:
    """Le second segment Receptor L-domain : le domaine I, reference de la face.

    Designe par exclusion de celui que l'empreinte selectionne, pas par un indice.
    """
    segs = gene3d_superfamily_domains()
    pick = pick_domain(segs, foot_uni)
    others = [s for s in segs if s != pick]
    return others[0] if others else None


# --------------------------------------------------------------------------- #
# 2. Alignement humain / souris
# --------------------------------------------------------------------------- #
def align_orthologs(seq_h: str, seq_m: str) -> dict[int, tuple[str, str]]:
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

    mapping: dict[int, tuple[str, str]] = {}
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


def disulfide_cys(entry: dict) -> set[int]:
    """Positions UniProt des cysteines engagees dans un pont disulfure annote.

    Source : features "Disulfide bond" de l'entree UniProt, deja en cache. Sert a
    defalquer la SASA apolaire qui provient de soufres deja engages.
    """
    out: set[int] = set()
    for _, start, end in uniprot_features(entry, "Disulfide bond"):
        out.update((start, end))
    return out


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


# --------------------------------------------------------------------------- #
# 3. Structure : empreinte, numerotation, SASA
# --------------------------------------------------------------------------- #
class ChainOnly(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "


def fab_footprint(model) -> set[int]:
    """Residus de la cible a moins de CONTACT_CUTOFF d'un atome lourd du Fab.

    Appele AVANT de detacher le Fab. Double usage : controle positif du masque de
    conservation (une proteine qui se lie vraiment au domaine III), et designation
    du domaine III parmi les segments renvoyes par les bases de domaines.
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
    foot = {
        res.id[1]
        for res in model[PDB_CHAIN]
        if res.id[0] == " "
        for atom in res
        if atom.element != "H" and search.search(atom.coord, CONTACT_CUTOFF)
    }
    if not foot:
        raise SystemExit("Empreinte vide : verifier les identifiants de chaines.")
    return foot


def deposited_offset(path: Path) -> tuple[int, list[tuple[int, str, str, str]]]:
    """Offset PDB -> UniProt declare par le deposant, et ecarts sequence/structure.

    Lu dans _struct_ref_seq / _struct_ref_seq_dif du mmCIF : une autorite externe,
    la ou le balayage ne produit qu'une inference.
    """
    d = MMCIF2Dict(str(path))

    def col(key: str) -> list[str]:
        v = d.get(key, [])
        return v if isinstance(v, list) else [v]

    rows = list(
        zip(
            col("_struct_ref_seq.pdbx_strand_id"),
            col("_struct_ref_seq.pdbx_db_accession"),
            col("_struct_ref_seq.pdbx_auth_seq_align_beg"),
            col("_struct_ref_seq.db_align_beg"),
        )
    )
    hits = [
        int(db_beg) - int(auth_beg)
        for strand, acc, auth_beg, db_beg in rows
        if strand == PDB_CHAIN and acc == HUMAN_AC
    ]
    if len(set(hits)) != 1:
        raise SystemExit(
            f"_struct_ref_seq : {len(set(hits))} offsets distincts pour la chaine "
            f"{PDB_CHAIN} / {HUMAN_AC} ({sorted(set(hits))}). La correspondance "
            "n'est pas un decalage unique, le modele du script ne tient plus."
        )

    difs = [
        (int(auth), mon, db_mon, det)
        for strand, auth, mon, db_mon, det in zip(
            col("_struct_ref_seq_dif.pdbx_pdb_strand_id"),
            col("_struct_ref_seq_dif.pdbx_auth_seq_num"),
            col("_struct_ref_seq_dif.mon_id"),
            col("_struct_ref_seq_dif.db_mon_id"),
            col("_struct_ref_seq_dif.details"),
        )
        if strand == PDB_CHAIN and auth not in ("?", ".")
    ]
    return hits[0], difs


def scan_offset(residues, seq_h: str) -> int:
    """Offset PDB -> UniProt par maximisation des correspondances d'identite."""
    best_offset, best_score = 0, -1
    for offset in range(-60, 61):
        score = sum(
            1
            for res in residues
            if 1 <= res.id[1] + offset <= len(seq_h)
            and seq_h[res.id[1] + offset - 1] == AA3TO1[res.get_resname()]
        )
        if score > best_score:
            best_offset, best_score = offset, score
    frac = best_score / len(residues)
    print(f"  offset par balayage       = {best_offset:+d}  ({frac:.1%} de concordance)")
    if frac < 0.95:
        raise SystemExit("Offset peu fiable : inspecter la chaine a la main.")
    return best_offset


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

    residues = list(chain)
    print(f"  {len(residues)} residus observes dans la chaine {PDB_CHAIN}")

    offset = scan_offset(residues, seq_h)
    dep_offset, difs = deposited_offset(path)
    print(f"  offset depose (_struct_ref_seq) = {dep_offset:+d}")
    if offset != dep_offset:
        raise SystemExit(
            f"DESACCORD de numerotation : balayage {offset:+d} vs _struct_ref_seq "
            f"{dep_offset:+d}. Deux routes independantes divergent, aucune n'est "
            "fiable sans inspection manuelle."
        )
    print("  -> les deux routes concordent")

    # SASA au niveau atomique : necessaire pour la ventilation apolaire/polaire
    ShrakeRupley().compute(model, level="A")
    return residues, offset, foot, difs


def report_difs(difs, offset: int, dom3_pdb: tuple[int, int]) -> None:
    """Ecarts sequence/structure. Bruyant si l'un tombe dans le domaine III."""
    if not difs:
        print("  _struct_ref_seq_dif : aucun ecart declare")
        return
    inside = [d for d in difs if dom3_pdb[0] <= d[0] <= dom3_pdb[1]]
    print(f"  _struct_ref_seq_dif : {len(difs)} ecart(s) declare(s) sur la chaine")
    for auth, mon, db_mon, det in difs:
        where = "DANS LE DOMAINE III" if dom3_pdb[0] <= auth <= dom3_pdb[1] else "hors domaine III"
        print(
            f"    auth {auth:<5} cristal={mon:<4} UniProt={db_mon:<4} "
            f"{det:<16} {where}"
        )
    if inside:
        print(
            "\n  !!!!! " + "=" * 66 + "\n"
            f"  !!!!! {len(inside)} ecart(s) sequence/structure tombent dans le domaine III.\n"
            "  !!!!! aa_human est lu dans la STRUCTURE, le statut humain/souris est\n"
            "  !!!!! derive de la sequence UniProt : ces positions sont incoherentes\n"
            "  !!!!! dans la table. A trancher avant d'utiliser un patch qui les touche.\n"
            "  !!!!! " + "=" * 66
        )
    else:
        print("  -> aucun ecart dans le domaine III, la table est coherente")


# --------------------------------------------------------------------------- #
# 4. Table par residu
# --------------------------------------------------------------------------- #
def anchor_atom(res):
    return res["CB"] if "CB" in res else res["CA"]


def split_sasa(res) -> tuple[float, float]:
    """SASA du residu ventilee en (apolaire, polaire) selon l'element.

    C et S apolaires, N et O polaires. Un comptage de residus hydrophobes rend
    invisibles les tiges aliphatiques des Lys/Arg, qui exposent du carbone.
    """
    apolar = sum(a.sasa for a in res if a.element in APOLAR_ELEMENTS)
    polar = sum(a.sasa for a in res if a.element in POLAR_ELEMENTS)
    return float(apolar), float(polar)


def face_axis(residues, offset: int, dom3: tuple[int, int], dom1: tuple[int, int] | None):
    """Axe oriente du centroide du domaine III vers celui du domaine I.

    Le site de liaison de l'EGF est menage entre les domaines I et III : la face
    du domaine III tournee vers le domaine I est la face ligand.
    """
    if dom1 is None:
        return None, None
    c3 = [
        anchor_atom(r).coord
        for r in residues
        if dom3[0] <= r.id[1] + offset <= dom3[1]
    ]
    c1 = [
        anchor_atom(r).coord
        for r in residues
        if dom1[0] <= r.id[1] + offset <= dom1[1]
    ]
    if not c3 or not c1:
        return None, None
    centre3 = np.mean(c3, axis=0)
    axis = np.mean(c1, axis=0) - centre3
    return centre3, axis / float(np.linalg.norm(axis))


def build_table(
    residues, offset, mapping, dom3, foot, glyc_sites, ss_cys, centre3, axis
):
    glyc_coords = [
        anchor_atom(res).coord
        for res in residues
        if res.id[1] + offset in glyc_sites and "CA" in res
    ]
    if not glyc_coords:
        raise SystemExit(
            "Aucun site de N-glycosylation ne tombe sur un residu observe. La "
            "colonne de distance glycane serait inoperante : arret plutot qu'un "
            "repli silencieux sur une distance infinie."
        )

    rows = []
    for res in residues:
        aa_h = AA3TO1[res.get_resname()]
        uni = res.id[1] + offset
        aa_m, status = mapping.get(uni, ("-", "gap"))
        apolar, polar = split_sasa(res)
        total = apolar + polar
        rel_sasa = total / MAX_ASA[aa_h]
        coord = anchor_atom(res).coord
        d_glyc = min(float(np.linalg.norm(coord - c)) for c in glyc_coords)
        if axis is None:
            face = "?"
        else:
            face = "ligand" if float(np.dot(coord - centre3, axis)) > 0 else "externe"
        rows.append(
            {
                "pdb_resnum": res.id[1],
                "uniprot_pos": uni,
                "aa_human": aa_h,
                "aa_mouse": aa_m,
                "status": status,
                "sasa_total": round(total, 1),
                "sasa_apolar": round(apolar, 1),
                "sasa_polar": round(polar, 1),
                "apolar_frac": round(apolar / total, 3) if total > 0 else 0.0,
                "rel_sasa": round(float(rel_sasa), 3),
                "exposed": bool(rel_sasa >= MIN_REL_SASA),
                "in_domain3": dom3[0] <= uni <= dom3[1],
                "face": face,
                "fab_contact": res.id[1] in foot,
                "acidic": aa_h in "DE",
                "hydrophobic": aa_h in HYDROPHOBIC,
                "cys_bridged": bool(aa_h == "C" and uni in ss_cys),
                "dist_glycan": round(float(d_glyc), 1),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 5. Enumeration des patches
# --------------------------------------------------------------------------- #
def enumerate_patches(rows: list[dict], residues) -> list[dict]:
    """Un patch par residu expose du domaine III, pris comme centre.

    L'ancrage acide n'est plus la source des centres : il devient une colonne
    comptee parmi les membres. Centrer sur Asp/Glu faisait du contenu apolaire
    nul une tautologie de l'echantillonnage, pas une mesure de la surface.
    """
    coords = {r.id[1]: anchor_atom(r).coord for r in residues}
    surface = [r for r in rows if r["in_domain3"] and r["exposed"]]
    print(f"  {len(surface)} residus exposes dans le domaine III -> autant de centres")

    pos = np.array([coords[r["pdb_resnum"]] for r in surface])
    patches = []
    for i, centre in enumerate(surface):
        d = np.linalg.norm(pos - pos[i], axis=1)
        members = [surface[j] for j in np.argsort(d) if d[j] <= PATCH_RADIUS]
        n = len(members)
        apolar = sum(m["sasa_apolar"] for m in members)
        polar = sum(m["sasa_polar"] for m in members)
        min_glyc = min(m["dist_glycan"] for m in members)
        patches.append(
            {
                "centre": f"{centre['aa_human']}{centre['pdb_resnum']}",
                "centre_num": centre["pdb_resnum"],
                "centre_uniprot": centre["uniprot_pos"],
                "n": n,
                "sasa_apolar": round(apolar, 1),
                # la SASA apolaire absolue correle a la taille du patch : la
                # densite par membre est sortie en parallele pour le verifier
                "sasa_apolar_per_res": round(apolar / n, 1),
                "sasa_polar": round(polar, 1),
                "apolar_frac": round(apolar / (apolar + polar), 3) if apolar + polar else 0.0,
                "frac_ident": round(
                    sum(1 for m in members if m["status"] == "identical") / n, 3
                ),
                "n_diff": sum(1 for m in members if m["status"] in ("different", "gap")),
                "n_acidic": sum(1 for m in members if m["acidic"]),
                "n_acidic_cons": sum(
                    1 for m in members if m["acidic"] and m["status"] == "identical"
                ),
                "n_hydro": sum(1 for m in members if m["hydrophobic"]),
                # un soufre de cysteine pontee compte comme apolaire mais n'offre
                # pas ce qu'offre une leucine exposee : SASA apolaire attribuable
                # aux cysteines pontees, sortie a part pour pouvoir la defalquer
                "n_cys_ponte": sum(1 for m in members if m["cys_bridged"]),
                "sasa_apolar_cys_ponte": round(
                    sum(m["sasa_apolar"] for m in members if m["cys_bridged"]), 1
                ),
                "n_fab": sum(1 for m in members if m["fab_contact"]),
                "frac_ligand_face": round(
                    sum(1 for m in members if m["face"] == "ligand") / n, 3
                ),
                "min_glyc": round(min_glyc, 1),
                "member_set": frozenset(m["pdb_resnum"] for m in members),
                "members": [
                    f"{m['aa_human']}{m['pdb_resnum']}"
                    + ("" if m["status"] == "identical" else f"({m['aa_mouse']})")
                    for m in members
                ],
            }
        )
    return patches


def size_distribution(patches: list[dict]) -> None:
    sizes = sorted(p["n"] for p in patches)
    arr = np.array(sizes)
    print(f"\n  distribution des tailles de patch (avant toute retenue, n={len(sizes)}) :")
    print(
        f"    min {arr.min()}  p10 {np.percentile(arr, 10):.0f}  "
        f"median {np.percentile(arr, 50):.0f}  p90 {np.percentile(arr, 90):.0f}  "
        f"max {arr.max()}  moyenne {arr.mean():.1f}"
    )
    hist: dict[int, int] = {}
    for s in sizes:
        hist[s] = hist.get(s, 0) + 1
    line = "  ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
    print(f"    taille:effectif  {line}")
    below = sum(1 for s in sizes if s < MIN_PATCH_MEMBERS)
    print(
        f"    sous le minimum actuel de {MIN_PATCH_MEMBERS} membres : {below} "
        f"({below / len(sizes):.0%})"
    )


def overlap_distribution(patches: list[dict]) -> None:
    """Recouvrement de chaque patch avec l'union de tous ceux mieux classes.

    Mesure independante de tout seuil : elle ne presuppose aucune acceptation.
    """
    taken: set[int] = set()
    fracs = []
    for p in patches:
        fracs.append(len(p["member_set"] & taken) / len(p["member_set"]))
        taken |= p["member_set"]
    arr = np.array(fracs[1:])  # le premier patch a 0 par construction
    if arr.size == 0:
        return
    print("\n  recouvrement avec l'union des patches mieux classes :")
    print(
        "    "
        + "  ".join(
            f"p{q}:{np.percentile(arr, q):.2f}" for q in (10, 25, 50, 75, 90, 100)
        )
    )


def dedup_sweep(patches: list[dict]) -> None:
    """Balayage seuil -> nombre de sites distincts. AUCUN seuil n'est applique."""
    print("\n  balayage du seuil de deduplication (aucun seuil fixe ici) :")
    print(f"    {'seuil':<8}{'sites':<8}{'centres retenus (5 premiers)'}")
    for thr in DEDUP_SWEEP:
        kept: list[dict] = []
        taken: set[int] = set()
        for p in patches:
            if len(p["member_set"] & taken) / len(p["member_set"]) <= thr:
                kept.append(p)
                taken |= p["member_set"]
        head = ", ".join(p["centre"] for p in kept[:5])
        print(f"    {thr:<8.1f}{len(kept):<8}{head}")


# --------------------------------------------------------------------------- #
# 6. Controle et sorties
# --------------------------------------------------------------------------- #
def cetuximab_control(foot: set[int], offset: int, mapping) -> None:
    """Controle positif : conservation humain/souris sur l'empreinte du cetuximab.

    Attendu INVERSE de l'intuition. Le cetuximab ne reconnait pas l'EGFR murin :
    une fonction de score saine doit donc classer son epitope BAS. Si l'empreinte
    ressort fortement conservee, c'est le masque de conservation qui ne mesure rien
    (offset faux, alignement casse), pas une bonne nouvelle.
    """
    status = [(n, *mapping.get(n + offset, ("-", "gap"))) for n in sorted(foot)]
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


def write_bfactor_pdb(residues, rows) -> Path:
    score = {"identical": 100.0, "similar": 50.0, "different": 0.0, "gap": 0.0}
    lookup = {r["pdb_resnum"]: score[r["status"]] for r in rows}
    for res in residues:
        for atom in res:
            atom.bfactor = lookup.get(res.id[1], 0.0)
    io = PDBIO()
    io.set_structure(residues[0].get_parent().get_parent().get_parent())
    out = DATA / "6aru_chainA_conserv.pdb"
    io.save(str(out), ChainOnly(PDB_CHAIN))
    return out


def write_csv(path: Path, rows: list[dict], drop: tuple[str, ...] = ()) -> Path:
    clean = [{k: v for k, v in r.items() if k not in drop} for r in rows]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(clean[0].keys()))
        writer.writeheader()
        writer.writerows(clean)
    return path


RANK_HDR = (
    f"{'rang':>5} {'centre':<8}{'n':>3}{'apolA2':>8}{'ap/res':>8}{'polA2':>7}"
    f"{'apfr':>6}{'ident':>7}{'acid':>5}{'acidC':>6}{'hyd':>4}{'cysSS':>6}"
    f"{'glyc':>6}{'face':>6}{'fab':>4}"
)
RULE = "-" * 90


def fmt_patch(p: dict, rank: int) -> str:
    return (
        f"{rank:>5} {p['centre']:<8}{p['n']:>3}{p['sasa_apolar']:>8.0f}"
        f"{p['sasa_apolar_per_res']:>8.1f}{p['sasa_polar']:>7.0f}"
        f"{p['apolar_frac']:>6.2f}{p['frac_ident']:>7.2f}{p['n_acidic']:>5}"
        f"{p['n_acidic_cons']:>6}{p['n_hydro']:>4}{p['n_cys_ponte']:>6}"
        f"{p['min_glyc']:>6.1f}{p['frac_ligand_face']:>6.2f}{p['n_fab']:>4}"
    )


def show_ranking(patches: list[dict], key: str, title: str, n: int = 10) -> list[dict]:
    ordered = sorted(patches, key=lambda p: -p[key])
    print(f"\n{title}")
    print(RULE)
    print(RANK_HDR)
    for i, p in enumerate(ordered[:n], 1):
        print(fmt_patch(p, i))
    return ordered


def pearson(xs, ys) -> float:
    x, y = np.array(xs, dtype=float), np.array(ys, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def footprint_report(ranked: list[dict], foot: set[int]) -> None:
    """A. Ou tombent les patches touchant l'empreinte du Fab cetuximab.

    Seul point de calibration disponible : un site ou une proteine se lie
    reellement. Il donne une echelle aux valeurs de SASA apolaire.
    """
    print("\n[A] EMPREINTE CETUXIMAB — calibration")
    print(RULE)
    rank = {p["centre"]: i for i, p in enumerate(ranked, 1)}
    hit = [p for p in ranked if p["n_fab"] > 0]
    if not hit:
        print("  aucun patch ne touche l'empreinte")
        return
    print(
        f"  {len(hit)} patches sur {len(ranked)} contiennent au moins un des "
        f"{len(foot)} residus de l'empreinte"
    )
    apol_all = [p["sasa_apolar"] for p in ranked]
    apol_hit = [p["sasa_apolar"] for p in hit]
    print(
        f"  SASA apolaire : mediane tous patches {np.median(apol_all):.0f} A2  |  "
        f"mediane patches d'empreinte {np.median(apol_hit):.0f} A2"
    )
    best = max(hit, key=lambda p: p["n_fab"])
    print(
        f"  recouvrement maximal : {best['centre']} avec {best['n_fab']} residus "
        f"d'empreinte, rang {rank[best['centre']]}"
    )
    print(RANK_HDR)
    for p in sorted(hit, key=lambda p: -p["n_fab"])[:12]:
        print(fmt_patch(p, rank[p["centre"]]))


def density_report(ranked_abs: list[dict], patches: list[dict]) -> None:
    """B. SASA apolaire par membre : le tri absolu classe-t-il la taille ?"""
    print("\n[B] DENSITE APOLAIRE PAR MEMBRE — effet de taille")
    print(RULE)
    n = [p["n"] for p in patches]
    print(
        f"  correlation n / SASA apolaire absolue   : {pearson(n, [p['sasa_apolar'] for p in patches]):+.3f}"
    )
    print(
        f"  correlation n / SASA apolaire par membre: {pearson(n, [p['sasa_apolar_per_res'] for p in patches]):+.3f}"
    )
    ranked_den = sorted(patches, key=lambda p: -p["sasa_apolar_per_res"])
    top_abs = {p["centre"] for p in ranked_abs[:10]}
    top_den = {p["centre"] for p in ranked_den[:10]}
    print(
        f"  taille mediane du top-10 absolu : {np.median([p['n'] for p in ranked_abs[:10]]):.1f}"
        f"  |  du top-10 par membre : {np.median([p['n'] for p in ranked_den[:10]]):.1f}"
    )
    print(f"  intersection des deux top-10 : {len(top_abs & top_den)}/10")
    if top_abs & top_den:
        print(f"    communs  : {', '.join(sorted(top_abs & top_den))}")
    if top_den - top_abs:
        print(f"    propres a la densite : {', '.join(sorted(top_den - top_abs))}")
    rank_abs = {p["centre"]: i for i, p in enumerate(ranked_abs, 1)}
    print(RANK_HDR)
    for i, p in enumerate(ranked_den[:10], 1):
        print(fmt_patch(p, i) + f"   (rang absolu {rank_abs[p['centre']]})")


def cysteine_report(ranked: list[dict], n: int = 5) -> None:
    """C. Le haut du classement tire-t-il sa SASA apolaire de soufres pontes ?"""
    print("\n[C] CYSTEINES PONTEES — qualite de la SASA apolaire")
    print(RULE)
    n_with = sum(1 for p in ranked if p["n_cys_ponte"] > 0)
    print(f"  {n_with} patches sur {len(ranked)} contiennent >= 1 cysteine pontee")
    print(
        f"  {'centre':<8}{'n':>3}{'cysSS':>7}{'apolA2':>9}{'dont cysSS':>12}"
        f"{'part':>7}{'apol. corrigee':>16}{'rang corrige':>14}"
    )
    corrected = sorted(
        ranked, key=lambda p: -(p["sasa_apolar"] - p["sasa_apolar_cys_ponte"])
    )
    rank_corr = {p["centre"]: i for i, p in enumerate(corrected, 1)}
    for p in ranked[:n]:
        net = p["sasa_apolar"] - p["sasa_apolar_cys_ponte"]
        part = p["sasa_apolar_cys_ponte"] / p["sasa_apolar"] if p["sasa_apolar"] else 0.0
        print(
            f"  {p['centre']:<8}{p['n']:>3}{p['n_cys_ponte']:>7}{p['sasa_apolar']:>9.0f}"
            f"{p['sasa_apolar_cys_ponte']:>12.0f}{part:>7.1%}{net:>16.0f}"
            f"{rank_corr[p['centre']]:>14}"
        )


# --------------------------------------------------------------------------- #
def main() -> None:
    print("[1/6] UniProt")
    h_entry, m_entry = load_uniprot(HUMAN_AC), load_uniprot(MOUSE_AC)
    seq_h = h_entry["sequence"]["value"]
    seq_m = m_entry["sequence"]["value"]
    print(f"  humain {len(seq_h)} aa | souris {len(seq_m)} aa")
    glyc_sites = {
        s
        for desc, s, _ in uniprot_features(h_entry, "Glycosylation")
        if desc.startswith("N-linked")
    }
    print(f"  {len(glyc_sites)} sites de N-glycosylation annotes (humain)")
    ss_cys = disulfide_cys(h_entry)
    print(f"  {len(ss_cys)} cysteines engagees dans un pont disulfure annote")

    print("\n[2/6] Alignement humain / souris")
    mapping = align_orthologs(seq_h, seq_m)

    print(f"\n[3/6] Structure {PDB_ID} chaine {PDB_CHAIN}")
    residues, offset, foot, difs = load_chain(seq_h)

    print("\n[4/6] Bornes du domaine III")
    foot_uni = {n + offset for n in foot}
    dom3_uni = resolve_domain_iii(foot_uni, offset)
    dom3_pdb = (dom3_uni[0] - offset, dom3_uni[1] - offset)
    dom1_uni = other_l_domain(foot_uni)
    if dom1_uni:
        print(
            f"  domaine I (reference de face) : {dom1_uni[0]}-{dom1_uni[1]} UniProt = "
            f"{dom1_uni[0] - offset}-{dom1_uni[1] - offset} PDB"
        )
    report_difs(difs, offset, dom3_pdb)
    in_dom3_glyc = sorted(g for g in glyc_sites if dom3_uni[0] <= g <= dom3_uni[1])
    print(
        f"  sequons N-linked dans le domaine III : {len(in_dom3_glyc)} "
        f"({', '.join(map(str, in_dom3_glyc))} UniProt)"
    )

    print("\n[5/6] Controle cetuximab")
    cetuximab_control(foot, offset, mapping)

    print("\n[6/6] Table et patches")
    centre3, axis = face_axis(residues, offset, dom3_uni, dom1_uni)
    rows = build_table(
        residues, offset, mapping, dom3_uni, foot, glyc_sites, ss_cys, centre3, axis
    )
    in_dom3 = [r for r in rows if r["in_domain3"]]
    n_ident = sum(1 for r in in_dom3 if r["status"] == "identical")
    print(f"  {len(in_dom3)} residus dans le domaine III, {n_ident / len(in_dom3):.1%} identiques")
    n_lig = sum(1 for r in in_dom3 if r["face"] == "ligand")
    print(f"  face ligand : {n_lig} | face externe : {len(in_dom3) - n_lig}")

    patches_all = enumerate_patches(rows, residues)
    size_distribution(patches_all)

    patches = [p for p in patches_all if p["n"] >= MIN_PATCH_MEMBERS]
    patches.sort(key=lambda p: -p["sasa_apolar"])
    print(f"\n  {len(patches)} patches a {MIN_PATCH_MEMBERS} membres ou plus, tries sur la SASA apolaire")
    overlap_distribution(patches)
    dedup_sweep(patches)

    res_csv = write_csv(DATA / "egfr_residues.csv", in_dom3)
    pat_csv = write_csv(DATA / "egfr_patches.csv", patches, drop=("member_set", "members"))
    pdb_path = write_bfactor_pdb(residues, rows)

    ranked = show_ranking(
        patches,
        "sasa_apolar",
        "CLASSEMENT UNIQUE — SASA APOLAIRE ABSOLUE (A2). min_glyc est brute, non ponderee.",
        n=15,
    )
    footprint_report(ranked, foot)
    density_report(ranked, patches)
    cysteine_report(ranked)

    top = sorted(patches, key=lambda p: -p["sasa_apolar"])[:3]
    print("\ncomposition des 3 premiers (SASA apolaire) :")
    for p in top:
        print(f"  {p['centre']:<7} {' '.join(p['members'])}")

    print(f"\nCSV residus : {res_csv}")
    print(f"CSV patches : {pat_csv}")
    print(f"PDB         : {pdb_path}")
    print("PyMOL: load data/6aru_chainA_conserv.pdb; spectrum b, red_white_blue, all, 0, 100")
    print(
        "\nAucun seuil de retenue n'est applique : ni deduplication, ni filtre "
        "glycane,\nni critere de conservation. Les colonnes et distributions "
        "ci-dessus sont la\nmatiere a decision, pas une selection."
    )


if __name__ == "__main__":
    main()
