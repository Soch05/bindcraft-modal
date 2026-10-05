#!/usr/bin/env python3
"""Récupère UNE fois la MSA du domaine III d'EGFR, et la met en cache sur disque.

    uv run --with requests python fetch_msa.py

POURQUOI PRÉCALCULER. Boltz-2 sait appeler un serveur MSA distant à chaque complexe
(`--use_msa_server`). Avec 20 à 60 complexes à prédire, c'est autant d'appels réseau
séquentiels depuis le conteneur GPU — donc du temps GPU payé à attendre le réseau, et la
cause de blocage la plus probable de toute la phase. La cible est la MÊME dans tous les
complexes : sa MSA se calcule une fois, se stocke, et se réinjecte en fichier.

POURQUOI PAS DE MSA POUR LE BINDER. Les binders sont de novo : ils n'ont aucun homologue
naturel. Une recherche d'homologues sur eux ne ramènerait que du bruit, et la nouveauté de
séquence est précisément ce que le règlement EXIGE. Le binder passe donc en MSA vide
(`msa: empty` côté Boltz), ce qui est le mode « single sequence ».

Sortie : inputs/egfr_dIII.a3m
"""

from __future__ import annotations

import io
import sys
import tarfile
import time
from pathlib import Path

import requests

TARGET_PDB = Path("inputs/6ARU_A_309-506.pdb")
OUT_A3M = Path("inputs/egfr_dIII.a3m")

API = "https://api.colabfold.com"
# `env` = UniRef30 + les bases environnementales. C'est le mode par défaut de ColabFold pour
# un monomère, et celui sur lequel Boltz-2 a été évalué.
MODE = "env"
QUERY_NAME = "egfr_dIII"
POLL_SECONDS = 10
MAX_WAIT_SECONDS = 1800
HEADERS = {"User-Agent": "adaptyv-challenge1-egfr (contact via depot public)"}


def target_sequence() -> str:
    import gemmi

    structure = gemmi.read_structure(str(TARGET_PDB))
    structure.setup_entities()
    return gemmi.one_letter_code([r.name for r in structure[0]["A"]]).upper()


def submit(sequence: str) -> str:
    response = requests.post(
        f"{API}/ticket/msa",
        data={"q": f">egfr_dIII\n{sequence}\n", "mode": MODE},
        headers=HEADERS, timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "ERROR":
        raise SystemExit(f"le serveur a refuse la requete : {payload}")
    return payload["id"]


def wait(ticket: str) -> None:
    started = time.time()
    seen = ""
    while True:
        payload = requests.get(f"{API}/ticket/{ticket}", headers=HEADERS, timeout=120).json()
        status = payload.get("status", "?")
        if status != seen:
            print(f"  [{time.time() - started:6.0f}s] {status}")
            seen = status
        if status == "COMPLETE":
            return
        if status in {"ERROR", "UNKNOWN", "RATELIMIT", "MAINTENANCE"}:
            raise SystemExit(f"ticket {ticket} en etat {status} — abandon")
        if time.time() - started > MAX_WAIT_SECONDS:
            raise SystemExit(f"depassement de {MAX_WAIT_SECONDS}s, ticket {ticket}")
        time.sleep(POLL_SECONDS)


def download(ticket: str, sequence: str) -> str:
    """Fusionne les .a3m de l'archive en UN SEUL alignement valide.

    L'archive de ColabFold contient plusieurs .a3m (uniref, bases environnementales), et
    CHACUN commence par sa propre copie de la séquence requête. Les concaténer produit un
    fichier où l'en-tête de requête apparaît plusieurs fois — ce n'est pas un alignement,
    c'est deux alignements collés, et un parseur a3m qui suppose « première séquence =
    requête, toutes les autres alignées sur elle » n'en fait rien de bon.

    HONNÊTETÉ SUR LA CAUSE : cette fusion a d'abord été écrite en croyant qu'elle expliquait
    l'échec de Boltz-2 du 5 octobre. C'ÉTAIT FAUX. Le diagnostic a montré un
    `ModuleNotFoundError: cuequivariance_torch` levé au fond du pairformer, donc APRÈS que la
    MSA ait été lue sans problème. L'a3m concaténé était malformé en principe, pas la cause
    mesurée. La fusion est conservée parce qu'elle est correcte, pas parce qu'elle a réparé
    quoi que ce soit.

    On garde donc UNE requête en tête, puis uniquement les entrées non-requête de chaque
    bloc.
    """
    blob = requests.get(f"{API}/result/download/{ticket}", headers=HEADERS, timeout=600).content
    merged = [f">{QUERY_NAME}", sequence]
    seen = {sequence}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            if not member.name.endswith(".a3m"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            text = handle.read().decode("utf-8", errors="replace")
            lines = [l.rstrip("\n") for l in text.splitlines() if l.strip()]
            added = 0
            for index in range(0, len(lines) - 1, 2):
                header, body = lines[index], lines[index + 1]
                if not header.startswith(">"):
                    continue
                # On saute la copie de la requête propre à ce bloc.
                if header[1:].split()[0] == QUERY_NAME or body in seen:
                    continue
                merged.extend([header, body])
                seen.add(body)
                added += 1
            print(f"  {member.name:<40} {added:6d} sequences retenues")
    if len(merged) <= 2:
        raise SystemExit("aucune sequence homologue dans l'archive")
    return "\n".join(merged) + "\n"


def main() -> None:
    if OUT_A3M.is_file() and OUT_A3M.stat().st_size > 0:
        existing = OUT_A3M.read_text()
        print(f"deja en cache : {OUT_A3M} ({existing.count('>')} sequences) — rien a faire")
        return
    sequence = target_sequence()
    print(f"cible : {len(sequence)} residus")
    ticket = submit(sequence)
    print(f"ticket {ticket}, mode {MODE}")
    wait(ticket)
    text = download(ticket, sequence)

    # Garde-fou : une seule requete en tete, et c'est bien notre cible.
    lines = text.splitlines()
    queries = [i for i, l in enumerate(lines) if l.startswith(f">{QUERY_NAME}")]
    if queries != [0] or lines[1] != sequence:
        raise SystemExit(
            f"a3m mal forme : en-tetes de requete aux lignes {queries}, attendu [0]"
        )
    OUT_A3M.write_text(text)
    print(f"-> {OUT_A3M} : {text.count('>')} sequences, {OUT_A3M.stat().st_size / 1e6:.2f} Mo")


if __name__ == "__main__":
    sys.exit(main())
