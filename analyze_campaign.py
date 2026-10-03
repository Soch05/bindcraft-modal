#!/usr/bin/env python3
"""Lit les sorties d'une campagne BindCraft 2.0 et produit le funnel, les distributions,
le chronométrage et le coût.

Rien de ce script ne tourne sur GPU : il lit des CSV rapatriés depuis le Volume Modal.

    modal volume get bindcraft <run_name> out/<run_name>
    uv run analyze_campaign.py out/<run_name>

Structure attendue, établie au run `egfr-dIII-cal01` du 3 octobre :

    <run>/1_Trajectories/!_Trajectories.csv    une ligne par trajectoire
    <run>/2_Refolded/!_Refolded.csv            candidats ProteinMPNN repliés
    <run>/3_Ranked/!_Ranked.csv                designs acceptés, réécrit à chaque acceptation
    <run>/summary.csv                          écrit à la fin de la campagne
    <run>/campaign_metadata.json               réglages résolus — la source autoritative

⚠️ Ce n'est PAS la structure de BindCraft 1 (`Accepted/Ranked/`, `failure_csv.csv`,
`final_design_stats.csv`). Tout script écrit pour celle-là est caduc.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Tarif L40S mesuré, cf. NOTES.md. Surchargeable en CLI pour une autre carte.
DEFAULT_USD_PER_SECOND = 0.000542

# Les métriques qu'on regarde par défaut, et le sens du seuil. Les seuils eux-mêmes sont
# lus dans `campaign_metadata.json` et jamais codés ici : ils changent avec la modalité.
WATCHED = (
    "i_pTM",
    "i_pAE",
    "pTM",
    "pLDDT",
    "Unbound_Binder_pLDDT",
    "Interface_Residues",
    "Interface_BuriedArea",
    "Hotspot_Contact_Fraction",
    "Coldspot_Contact_Fraction",
    "Off_Epitope_Contact_Fraction",
    "Surface_Hydrophobicity",
    "Binder_Free_Cysteines",
)


@dataclass(frozen=True)
class Timing:
    """La colonne `Timing`, au format `worker=0;start=<epoch>;design=<s>;compiled=<0|1>`."""

    worker: int
    start: float
    design: float
    compiled: bool

    @classmethod
    def parse(cls, raw: str) -> Timing | None:
        if not raw:
            return None
        parts = dict(piece.split("=", 1) for piece in raw.split(";") if "=" in piece)
        try:
            return cls(
                worker=int(parts["worker"]),
                start=float(parts["start"]),
                design=float(parts["design"]),
                compiled=parts.get("compiled") == "1",
            )
        except (KeyError, ValueError):
            return None


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numbers(rows: list[dict[str, str]], column: str) -> list[float]:
    out = []
    for row in rows:
        raw = (row.get(column) or "").strip()
        if raw and raw.lower() != "nan":
            try:
                out.append(float(raw))
            except ValueError:
                pass
    return out


def resolved_thresholds(metadata: dict) -> dict[str, tuple[float, bool]]:
    """Seuils réellement en vigueur : `{metrique: (seuil, plus_haut_est_mieux)}`.

    Lus dans `campaign_metadata.json`, pas dans `settings/core/default.json` — le preset de
    modalité écrase le noyau. C'est comme ça qu'on a découvert que `Unbound_Binder_pLDDT`
    vaut 0,70 et non 0,80.
    """
    filters = metadata.get("filters") or metadata.get("settings", {}).get("filters") or {}
    out: dict[str, tuple[float, bool]] = {}
    for name, entry in filters.items():
        if not isinstance(entry, dict):
            continue
        threshold = entry.get("threshold")
        if threshold in (None, "Infinity", "-Infinity"):
            continue
        try:
            out[name] = (float(threshold), bool(entry.get("higher")))
        except (TypeError, ValueError):
            continue
    return out


def print_funnel(trajectories: list[dict], refolded: list[dict], ranked: list[dict]) -> None:
    print("=" * 78)
    print("FUNNEL")
    print("=" * 78)
    stopped = Counter((row.get("terminated") or "").strip() or "va au bout" for row in trajectories)
    print(f"  trajectoires lancées            {len(trajectories)}")
    for stage, count in sorted(stopped.items(), key=lambda item: -item[1]):
        share = 100 * count / len(trajectories) if trajectories else 0
        label = "mortes au stade " + stage if stage != "va au bout" else "allées au bout"
        print(f"    {label:<30} {count:>4}  ({share:4.0f} %)")
    print(f"  candidats ProteinMPNN repliés   {len(refolded)}")
    print(f"  designs acceptés                {len(ranked)}")
    if refolded:
        print(f"  taux d'acceptation par candidat {100 * len(ranked) / len(refolded):.1f} %")
    if trajectories:
        print(f"  designs par trajectoire         {len(ranked) / len(trajectories):.2f}")


def print_timing(trajectories: list[dict], usd_per_second: float) -> None:
    timings = [t for t in (Timing.parse(row.get("Timing", "")) for row in trajectories) if t]
    if not timings:
        print("\n(aucune colonne Timing exploitable)")
        return
    print()
    print("=" * 78)
    print("CHRONOMÉTRAGE ET COÛT")
    print("=" * 78)
    by_worker = Counter(t.worker for t in timings)
    print(f"  workers observés                {sorted(by_worker)}  {dict(by_worker)}")
    compiled = sum(t.compiled for t in timings)
    print(f"  trajectoires ayant compilé      {compiled} / {len(timings)}")

    design = [t.design for t in timings]
    print(f"  temps `design` : min {min(design):6.1f}s  médian {st.median(design):6.1f}s  max {max(design):6.1f}s")

    full = [t.design for t, row in zip(timings, trajectories) if not (row.get("terminated") or "").strip()]
    early = [t.design for t, row in zip(timings, trajectories) if (row.get("terminated") or "").strip()]
    for label, values in (("complètes", full), ("mortes tôt", early)):
        if values:
            print(f"    {label:<12} n={len(values):<4} médian {st.median(values):6.1f}s")

    # Le mur : du premier démarrage à la fin de la dernière. C'est ce qui est facturé.
    span = max(t.start + t.design for t in timings) - min(t.start for t in timings)
    gpu_seconds = span  # une seule carte, les workers la partagent
    print(f"  mur, premier départ → dernière fin  {span:7.1f}s = {span / 3600:5.2f} h")
    print(f"  coût GPU                            ${gpu_seconds * usd_per_second:6.3f}")
    if len(timings) > 1:
        print(f"  coût par trajectoire                ${gpu_seconds * usd_per_second / len(timings):6.3f}")

    # Somme du travail contre le mur : au-dessus de 1, la concurrence a servi.
    speedup = sum(design) / span if span else 0
    print(f"  somme des `design` / mur            {speedup:5.2f}×  (>1 = la concurrence rapporte)")


def print_distributions(rows: list[dict], label: str, thresholds: dict) -> None:
    if not rows:
        return
    print()
    print("=" * 78)
    print(f"DISTRIBUTIONS — {label} (n={len(rows)})")
    print("=" * 78)
    print(f"  {'métrique':<30}{'min':>8}{'médian':>9}{'max':>8}   seuil / taux de passage")
    for name in WATCHED:
        values = numbers(rows, name)
        if not values:
            continue
        line = f"  {name:<30}{min(values):>8.2f}{st.median(values):>9.3f}{max(values):>8.2f}"
        if name in thresholds:
            threshold, higher = thresholds[name]
            passing = sum(v >= threshold for v in values) if higher else sum(v <= threshold for v in values)
            sign = ">=" if higher else "<="
            line += f"   {sign} {threshold:<6g} {passing:>3}/{len(values)} passent"
        else:
            line += "   (non posé)"
        print(line)


def print_generalisation(trajectories: list[dict], refolded: list[dict]) -> None:
    """Écart entre l'auto-évaluation du gradient et la validation tenue à l'écart.

    BindCraft impose des modèles de design et de validation disjoints. L'écart mesure donc
    de combien l'auto-évaluation d'AF2 surestime l'interface — c'est le chiffre qui motive
    un re-scoring orthogonal, et il est directement versable au dossier de méthodes.
    """
    pairs = []
    by_design = {row.get("design"): row for row in trajectories if row.get("design")}
    grouped: dict[str, list[dict]] = {}
    for row in refolded:
        parent = (row.get("design") or "").split("_candidate")[0].split("_seq")[0]
        grouped.setdefault(parent, []).append(row)
    for name, candidates in grouped.items():
        trajectory = by_design.get(name)
        if not trajectory:
            continue
        for metric in ("i_pTM", "i_pAE"):
            gradient = numbers([trajectory], metric)
            validated = numbers(candidates, metric)
            if gradient and validated:
                pairs.append((metric, gradient[0], st.mean(validated), len(validated)))
    if not pairs:
        return
    print()
    print("=" * 78)
    print("ÉCART DE GÉNÉRALISATION — gradient (modèles de design) vs validation tenue à l'écart")
    print("=" * 78)
    for metric in ("i_pTM", "i_pAE"):
        rows = [p for p in pairs if p[0] == metric]
        if not rows:
            continue
        gaps = [abs(gradient - validated) for _, gradient, validated, _ in rows]
        print(f"  {metric:<8} sur {len(rows)} squelettes : écart médian {st.median(gaps):.3f}, max {max(gaps):.3f}")
        for _, gradient, validated, n in rows[:5]:
            print(f"    gradient {gradient:.2f}  ->  validation {validated:.3f} (n={n})")


def print_ph_material(rows: list[dict], label: str) -> None:
    """Matière pour l'objectif n°1, au niveau séquence seulement.

    ⚠️ Compter les His d'une séquence ne dit RIEN sur leur appariement géométrique avec
    `D323`, ni un Asp avec `H409`. Ça demande les structures. Ces chiffres bornent le
    possible, ils ne mesurent pas le switch.
    """
    sequences = [(row.get("Binder_Sequence") or "").strip() for row in rows]
    sequences = [s for s in sequences if s]
    if not sequences:
        return
    print()
    print("=" * 78)
    print(f"MATIÈRE pH — {label} (niveau séquence uniquement)")
    print("=" * 78)
    his = [s.count("H") for s in sequences]
    acid = [s.count("D") + s.count("E") for s in sequences]
    cys = [s.count("C") for s in sequences]
    print(f"  His par binder        min {min(his)}  médian {st.median(his):.1f}  max {max(his)}")
    print(f"  Asp+Glu par binder   min {min(acid)}  médian {st.median(acid):.1f}  max {max(acid)}")
    print(f"  séquences sans His    {sum(h == 0 for h in his)} / {len(sequences)}")
    print(f"  cystéines (doit être 0) max {max(cys)}")
    print("  ⚠️ ne dit rien de l'appariement His–D323 ni Asp–H409 : il faut les structures.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign", type=Path, help="dossier de campagne rapatrié")
    parser.add_argument("--usd-per-second", type=float, default=DEFAULT_USD_PER_SECOND)
    args = parser.parse_args()

    root: Path = args.campaign
    if not root.is_dir():
        raise SystemExit(f"{root} n'est pas un dossier")

    trajectories = read_table(root / "1_Trajectories" / "!_Trajectories.csv")
    refolded = read_table(root / "2_Refolded" / "!_Refolded.csv")
    ranked = read_table(root / "3_Ranked" / "!_Ranked.csv")
    metadata_path = root / "campaign_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    thresholds = resolved_thresholds(metadata)

    print(f"campagne : {root}")
    if not trajectories:
        raise SystemExit("aucune trajectoire lue — le dossier est-il bien rapatrié ?")
    print(f"seuils résolus lus : {len(thresholds)} filtres restrictifs")

    print_funnel(trajectories, refolded, ranked)
    print_timing(trajectories, args.usd_per_second)
    print_distributions(refolded, "candidats repliés (validation)", thresholds)
    print_distributions(ranked, "designs acceptés", thresholds)
    print_generalisation(trajectories, refolded)
    print_ph_material(refolded or trajectories, "candidats repliés" if refolded else "trajectoires")


if __name__ == "__main__":
    main()
