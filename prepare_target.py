#!/usr/bin/env python3
"""Prepare le PDB cible pour BindCraft depuis 6ARU.

Extrait la chaine A, polymere seul (ni heteroatomes, ni NAG, ni eau), et la
tronque a un intervalle de numerotation PDB. La numerotation d'origine est
conservee : les hotspots cites ailleurs restent valides sans retraduction.

Bornes par defaut : celles du domaine III resolues par egfr_epitope_map.py
depuis CATH-Gene3D (G3DSA:3.80.20.20), soit 333-530 UniProt = 309-506 PDB.
Elles ne sont pas codees en dur ici, elles sont demandees au meme resolveur.

Troncature justifiee par mesure, pas par commodite : la SASA des 20 residus du
site de reference est identique a 0,0 A2 pres entre la chaine A entiere (609
residus) et le domaine III isole (198). Seuls 9 residus de tout le domaine
gagnent plus de 20 A2, aucun adjacent au site. L'occlusion qui compte pour un
binder a ce site est locale ; les domaines I, II et IV n'y contribuent pas.

Usage:
    python prepare_target.py                      # domaine III, bornes resolues
    python prepare_target.py --first 309 --last 506 --out inputs/x.pdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, Select

import egfr_epitope_map as E


class Window(Select):
    """Chaine cible, residus polymere standard, dans une fenetre de numerotation."""

    def __init__(self, chain_id: str, first: int, last: int):
        self.chain_id = chain_id
        self.first = first
        self.last = last

    def accept_chain(self, chain) -> bool:
        return chain.id == self.chain_id

    def accept_residue(self, residue) -> bool:
        return (
            residue.id[0] == " "
            and residue.get_resname() in E.AA3TO1
            and self.first <= residue.id[1] <= self.last
        )

    def accept_atom(self, atom) -> bool:
        return atom.element != "H" and atom.get_altloc() in (" ", "A")


def resolve_domain_window() -> tuple[int, int]:
    """Bornes du domaine III en numerotation PDB, via le resolveur du projet."""
    entry = E.load_uniprot(E.HUMAN_AC)
    seq_h = entry["sequence"]["value"]
    residues, offset, foot, _ = E.load_chain(seq_h)
    dom3 = E.resolve_domain_iii({n + offset for n in foot}, offset)
    return dom3[0] - offset, dom3[1] - offset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=int, default=None, help="premier residu (num. PDB)")
    ap.add_argument("--last", type=int, default=None, help="dernier residu (num. PDB)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.first is None or args.last is None:
        first, last = resolve_domain_window()
        print(f"bornes resolues depuis CATH-Gene3D : {first}-{last} (num. PDB)")
    else:
        first, last = args.first, args.last
        print(f"bornes fournies : {first}-{last} (num. PDB)")

    out = args.out or Path("inputs") / f"6ARU_A_{first}-{last}.pdb"
    out.parent.mkdir(exist_ok=True)

    path = E.fetch(
        f"https://files.rcsb.org/download/{E.PDB_ID}.cif",
        E.DATA / f"{E.PDB_ID}.cif",
    )
    structure = MMCIFParser(QUIET=True).get_structure(E.PDB_ID, path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out), Window(E.PDB_CHAIN, first, last))

    kept = [
        line for line in out.read_text().splitlines() if line.startswith("ATOM")
    ]
    nres = len({line[22:27] for line in kept})
    print(f"ecrit {out} : {nres} residus, {len(kept)} atomes")
    if nres == 0:
        raise SystemExit("aucun residu retenu : verifier les bornes et la chaine.")


if __name__ == "__main__":
    main()
