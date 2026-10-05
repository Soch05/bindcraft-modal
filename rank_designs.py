#!/usr/bin/env python3
"""Assemble le classement du vivier et l'analyse appariée WT vs mutant.

    uv run python rank_designs.py

CLASSEMENT LEXICOGRAPHIQUE, PAS DE SCORE COMPOSITE. Le règlement classe les objectifs dans
cet ordre : sélectivité pH, puis cross-réactivité souris, puis affinité. Un score pondéré
exigerait d'inventer des poids entre ces trois axes, qu'on ne saurait pas justifier et qui
masqueraient les arbitrages. On trie donc par groupes, puis à l'intérieur de chaque groupe
sur le critère de rang supérieur.

ESM-2 N'ENTRE DANS AUCUN ORDRE. La pseudo-vraisemblance ESM-2 mesure la ressemblance aux
protéines naturelles. Nos binders sont de novo et le règlement EXIGE qu'ils ne ressemblent
pas au naturel ; classer dessus favoriserait les designs les moins nouveaux, c'est-à-dire
exactement ce que le règlement pénalise. La colonne est descriptive.

Sorties : out/master_rank.csv, out/paires_wt_mutant.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RANKED = {
    "egfr-dIII-prod01": Path("out/egfr-dIII-prod01/3_Ranked/!_Ranked.csv"),
    "egfr-dIII-prod02": Path("out/egfr-dIII-prod02/3_Ranked/!_Ranked.csv"),
}
GEOMETRY_WT = Path("out/geometry_wt.csv")
GEOMETRY_MUT = Path("out/geometry_mutants.csv")
PROPKA_H409 = Path("out/propka_h409.csv")
PROPKA_ACIDS = Path("out/propka_acids.csv")
PROPKA_ALL = Path("out/propka_all_groups.csv")
MUTANTS = Path("out/mutants_acide.csv")
THREADED = Path("structures/threaded/threaded_index.csv")
RESIDUES = Path("data/egfr_residues.csv")
BOLTZ_CONTACTS = Path("out/boltz_contacts.csv")
BIDENTATE = Path("out/bidentate.csv")
BOLTZ_MUTANTS = Path("out/boltz_contacts_mutants.csv")
ESM2 = Path("out/esm2_pll.csv")
CROSS = Path("out/cross_species.csv")

OUT_RANK = Path("out/master_rank.csv")
OUT_PAIRS = Path("out/paires_wt_mutant.csv")

PH_ACID = 6.5
PH_NEUTRAL = 7.4

# --- Seuils ADOSSÉS À UNE RÉFÉRENCE ---------------------------------------------------
# Pont salin entre atomes chargés, Barlow & Thornton 1983.
SALT_BRIDGE_A = 4.0
# PROPKA se trompe couramment d'une unité de pKa. On prend 0,5 comme plancher de bruit :
# un décalage plus petit que ça ne se distingue pas de l'erreur du modèle.
PKA_NOISE = 0.5
# Un carboxylate dont le pKa dépasse ce seuil est partiellement protoné dès pH 6,5, donc
# perd sa charge là où le mécanisme en a besoin.
CARBOXYLATE_SELF_DEFEAT_PKA = 5.0

# --- Seuils POSÉS, NON CALIBRÉS -------------------------------------------------------
# Aucun jeu de référence ne permet de calibrer « pose confirmée » sur ce challenge : il
# faudrait des structures expérimentales de binders de novo contre cette cible. Ces deux
# valeurs sont des conventions, et tout design qu'elles déclassent est signalé comme tel
# plutôt que silencieusement rétrogradé.
CONTACT_RECOVERY_CONFIRMED = 0.50
ORTHOGONAL_IPTM_CONFIRMED = 0.60


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""))) if path.is_file() else []


def number(value: str | None, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ph_fraction(pka: float) -> float:
    return (1 + 10 ** (pka - PH_ACID)) / (1 + 10 ** (pka - PH_NEUTRAL))


def selectivity(pka_bound: float, pka_free: float) -> float:
    return ph_fraction(pka_bound) / ph_fraction(pka_free)


# ---------------------------------------------------------------------------------------
# Facteur de sélectivité GLOBAL — la version non biaisée
# ---------------------------------------------------------------------------------------


def global_linkage() -> dict[str, dict]:
    """Produit, sur TOUS les groupes ionisables, de leur contribution à la sélectivité pH.

    POURQUOI C'EST PLUS HONNÊTE QUE H409 SEUL. La dépendance au pH de l'affinité n'est pas
    portée par un résidu choisi d'avance : thermodynamiquement, c'est la somme des
    contributions de CHAQUE groupe dont le pKa se décale entre l'état libre et l'état lié
    (relation de couplage proton-ligand). Ne regarder que H409 revient à supposer la réponse.
    On calcule donc les deux et on les compare.

    LIMITE À ÉNONCER : le produit suppose les sites INDÉPENDANTS, ce qu'ils ne sont pas
    quand deux groupes se touchent, et il cumule ~80 erreurs de PROPKA par complexe. On
    rapporte donc aussi la version restreinte aux groupes qui bougent de plus que le bruit,
    qui est moins sensible au cumul.
    """
    tables: dict[str, dict] = {}
    for record in rows(PROPKA_ALL):
        tables.setdefault(record["cle"], {})[record["groupe"]] = (
            number(record["pKa"]), record["chaine"]
        )

    out: dict[str, dict] = {}
    for key, bound in tables.items():
        kind, stem, state = key.split("|")
        if state != "complexe":
            continue
        design = stem.split("__")[0]
        target_free = tables.get(f"WT|{design}|cible", {})
        binder_free = tables.get(f"{kind}|{stem}|binder", {})
        if not target_free or not binder_free:
            continue

        total, restricted, movers = 1.0, 1.0, 0
        for group, (pka, chain) in bound.items():
            reference = target_free if chain == "A" else binder_free
            if group not in reference:
                continue
            free = reference[group][0]
            ratio = selectivity(pka, free)
            total *= ratio
            if abs(pka - free) > PKA_NOISE:
                restricted *= ratio
                movers += 1
        out[stem] = {
            "facteur_global": round(total, 3),
            "facteur_groupes_mobiles": round(restricted, 3),
            "n_groupes_mobiles": movers,
        }
    return out


# ---------------------------------------------------------------------------------------
# Conservation de l'épitope — objectif n°2
# ---------------------------------------------------------------------------------------


def conservation_map() -> dict[int, str]:
    return {int(r["pdb_resnum"]): r["status"] for r in rows(RESIDUES)}


def epitope_conservation(contacted: str, status: dict[int, str]) -> tuple[float, int, str]:
    """Fraction des résidus de cible contactés qui sont IDENTIQUES chez la souris.

    C'est le proxy de l'objectif n°2 : la même séquence doit reconnaître P00533 et Q01279.
    Un binder qui s'appuie sur des résidus divergents ne sera pas cross-réactif, quelle que
    soit son affinité sur l'humain.
    """
    residues = [r.strip() for r in contacted.split(",") if r.strip()]
    found, divergent = [], []
    for item in residues:
        digits = "".join(c for c in item if c.isdigit())
        if not digits:
            continue
        state = status.get(int(digits))
        if state is None:
            continue
        found.append(state)
        if state != "identical":
            divergent.append(item)
    if not found:
        return (float("nan"), 0, "")
    identical = sum(1 for s in found if s == "identical")
    return (round(identical / len(found), 3), len(found), ",".join(divergent))


# ---------------------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------------------


def build() -> tuple[list[dict], list[dict]]:
    status = conservation_map()
    linkage = global_linkage()

    af2: dict[str, dict] = {}
    for run, path in RANKED.items():
        for record in rows(path):
            record["run"] = run
            af2[record["design"]] = record

    geometry = {r["design"]: r for r in rows(GEOMETRY_WT)}
    h409 = rows(PROPKA_H409)
    wt_h409 = {r["design"]: r for r in h409 if r["type"] == "WT"}
    mut_h409: dict[tuple[str, str], list[dict]] = {}
    for record in (r for r in h409 if r["type"] == "mutant"):
        mut_h409.setdefault((record["design"], record["mutation"]), []).append(record)

    acids: dict[tuple[str, str], list[dict]] = {}
    for record in rows(PROPKA_ACIDS):
        acids.setdefault((record["design"], record["mutation"]), []).append(record)

    reach: dict[tuple[str, str], list[dict]] = {}
    for record in rows(GEOMETRY_MUT):
        reach.setdefault((record["design"], record["mutation"]), []).append(record)

    orthogonal = {r["design"]: r for r in rows(BOLTZ_CONTACTS)}
    bidentate = {r["design"]: r for r in rows(BIDENTATE)}
    # Les mutants sont indexés `<parent>__<mutation>`, comme les dossiers Boltz.
    orthogonal_mutants = {r["design"]: r for r in rows(BOLTZ_MUTANTS)}
    esm2 = {r["design_id"]: r for r in rows(ESM2)}
    cross = {r["design"]: r for r in rows(CROSS)}
    mutant_rows = {r["design"]: r for r in rows(MUTANTS)}

    entries: list[dict] = []

    def common(design: str) -> dict:
        record = af2[design]
        fraction, counted, divergent = epitope_conservation(
            record["Interface_Target_Residues"], status
        )
        return {
            "design_id": design,
            "squelette": design.split("_")[-2],
            "run": record["run"],
            "longueur": int(number(record["length"], 0)),
            "sequence": record["Binder_Sequence"].strip().upper(),
            "charge_nette": number(record["Binder_Net_Charge"]),
            "cysteines": number(record["Binder_Cysteines"]),
            "i_pTM_AF2": number(record["i_pTM"]),
            "i_pAE_AF2": number(record["i_pAE"]),
            "Interface_BuriedArea": number(record["Interface_BuriedArea"]),
            "Hotspot_Contact_Fraction": number(record["Hotspot_Contact_Fraction"]),
            "Off_Epitope": number(record["Off_Epitope_Contact_Fraction"]),
            "epitope_conservation_frac": fraction,
            "epitope_residus_comptes": counted,
            "epitope_residus_divergents": divergent,
            "residus_cible_contactes": record["Interface_Target_Residues"],
        }

    # --- les 23 designs natifs ---------------------------------------------------------
    for design in af2:
        entry = common(design)
        geo = geometry.get(design, {})
        pka = wt_h409.get(design, {})
        distance = number(geo.get("distance_A"))
        shift = number(pka.get("dpKa_H409"))

        entry.update({
            "type": "frere" if design.endswith("_seq1") else "natif",
            "parent": "",
            "mutations": "",
            "pont_salin_WT_verifie": geo.get("pont", ""),
            "pont_residu": geo.get("residu_acide", ""),
            "pont_distance_A": distance,
            "pont_angle_deg": number(geo.get("angle_deg")),
            "pont_alerte": geo.get("alerte", ""),
            "pKa_H409_lie": number(pka.get("pKa_H409_lie")),
            "pKa_H409_libre": number(pka.get("pKa_H409_libre")),
            "dpKa_H409": shift,
            "dpKa_etendue_rotameres": "",
            "facteur_pH_predit": number(pka.get("facteur_pH_predit")),
            "pKa_DE_min": "", "pKa_DE_max": "", "pKa_DE_etendue": "",
            "facteur_min_rotamere": "", "facteur_max_rotamere": "",
            "verdict_carboxylate": "sans objet",
            "ddpKa_vs_parent": "",
            "rotameres_retenus": "",
        })
        entry["route_pH"] = (
            "route2" if shift == shift and shift > PKA_NOISE and distance <= SALT_BRIDGE_A
            else "aucune"
        )
        entry["mecanisme_pH"] = (
            "robuste" if shift == shift and shift > PKA_NOISE else "absent"
        )
        entries.append(entry)

    # --- les 12 mutants ----------------------------------------------------------------
    for (design, mutation), acid_rows in acids.items():
        entry = common(design)
        parent_record = mutant_rows.get(design, {})
        sequence = parent_record.get("Binder_Sequence_mutee", "").strip().upper()
        if not sequence:
            continue

        geometry_rows = reach.get((design, mutation), [])
        by_rotamer = {int(r["rotamer"]): r for r in geometry_rows}

        # Rotamères retenus : ceux sans recouvrement. Si moins de deux survivent, on garde
        # tout en le signalant — un verdict sur un seul rotamere n'est pas un verdict.
        clean = [r for r in acid_rows if r["clash_threading"] == "non"]
        retained = clean if len(clean) >= 2 else acid_rows
        note = "tous (moins de 2 sans clash)" if len(clean) < 2 else "sans clash"
        if all(r["clash_threading"] == "oui" for r in acid_rows):
            note = "IMPOSSIBLE : tous les rotameres clashent"

        pkas = [number(r["pKa_DE"]) for r in retained]
        above = [p for p in pkas if p > CARBOXYLATE_SELF_DEFEAT_PKA]
        if len(above) == len(pkas):
            verdict = "ROUGE"
        elif not above:
            verdict = "VERT"
        else:
            verdict = "INDETERMINE"

        keep = {int(r["rotamer"]) for r in retained}
        shifts = [
            number(r["dpKa_H409"]) for r in mut_h409.get((design, mutation), [])
            if int(r["rotamer"]) in keep
        ]
        spread = (max(shifts) - min(shifts)) if len(shifts) > 1 else 0.0
        best = max(shifts) if shifts else float("nan")
        mean = sum(shifts) / len(shifts) if shifts else float("nan")
        # Le facteur est une fonction NON LINÉAIRE du pKa : en faire la moyenne sur les
        # rotamères donnerait un nombre incohérent avec le ΔpKa moyen rapporté juste à
        # côté (inégalité de Jensen). On le recalcule donc DEPUIS le ΔpKa moyen, et on
        # rapporte séparément son étendue sur les rotamères.
        free_values = [
            number(r["pKa_H409_libre"]) for r in mut_h409.get((design, mutation), [])
            if int(r["rotamer"]) in keep
        ]
        pka_free = free_values[0] if free_values else float("nan")
        factors = [
            number(r["facteur_pH_predit"]) for r in mut_h409.get((design, mutation), [])
            if int(r["rotamer"]) in keep
        ]
        factor_from_mean = (
            selectivity(pka_free + mean, pka_free)
            if mean == mean and pka_free == pka_free else float("nan")
        )

        in_reach = [r for r in geometry_rows if r["a_portee"] == "oui"
                    and r["clash_threading"] == "non"]

        if spread > PKA_NOISE:
            mechanism = "INDETERMINE"
        elif mean == mean and mean > PKA_NOISE:
            mechanism = "robuste"
        else:
            mechanism = "absent"

        parent_shift = number(wt_h409.get(design, {}).get("dpKa_H409"))
        entry.update({
            "type": "mutant",
            "parent": design,
            "mutations": mutation,
            "sequence": sequence,
            "longueur": len(sequence),
            "pont_salin_WT_verifie": "",
            "pont_residu": f"{acid_rows[0]['residu']}",
            "pont_distance_A": min(
                (number(r["distance_H409_A"]) for r in geometry_rows), default=float("nan")
            ),
            "pont_angle_deg": "",
            "pont_alerte": f"{len(in_reach)}/{len(geometry_rows)} rotameres a portee "
                           f"et sans clash",
            "pKa_H409_lie": "", "pKa_H409_libre": "",
            "dpKa_H409": round(mean, 2) if mean == mean else "",
            "dpKa_etendue_rotameres": round(spread, 2),
            "facteur_pH_predit": round(factor_from_mean, 3)
            if factor_from_mean == factor_from_mean else "",
            "facteur_min_rotamere": round(min(factors), 3) if factors else "",
            "facteur_max_rotamere": round(max(factors), 3) if factors else "",
            "pKa_DE_min": round(min(pkas), 2) if pkas else "",
            "pKa_DE_max": round(max(pkas), 2) if pkas else "",
            "pKa_DE_etendue": round(max(pkas) - min(pkas), 2) if len(pkas) > 1 else 0.0,
            "verdict_carboxylate": verdict,
            "ddpKa_vs_parent": round(mean - parent_shift, 2)
            if mean == mean and parent_shift == parent_shift else "",
            "rotameres_retenus": note,
            "mecanisme_pH": mechanism,
            "route_pH": "route2 (tentee)" if in_reach else "aucune (hors portee)",
        })
        entry["dpKa_H409_meilleur_rotamere"] = round(best, 2) if best == best else ""
        entries.append(entry)

    # --- orthogonal + facteur global ---------------------------------------------------
    for entry in entries:
        stem = entry["design_id"] if entry["type"] != "mutant" else \
            f"{entry['parent']}__{entry['mutations']}__rot1"
        link = linkage.get(stem, {})
        entry["facteur_global_tous_groupes"] = link.get("facteur_global", "")
        entry["facteur_global_groupes_mobiles"] = link.get("facteur_groupes_mobiles", "")
        entry["n_groupes_mobiles"] = link.get("n_groupes_mobiles", "")

        # Règle bidentée : mesurée sur le squelette, donc valable pour le natif comme
        # pour ses mutants, qui partagent ce squelette.
        source = entry["parent"] if entry["type"] == "mutant" else entry["design_id"]
        bident = bidentate.get(source, {})
        entry["bidente_goulot_A"] = bident.get("bidente_goulot_A", "")
        entry["bidente_paire"] = (
            f"{bident.get('bidente_ND1', '')}/{bident.get('bidente_NE2', '')}"
            if bident.get("bidente_goulot_A") else "aucune paire possible"
        )

        if entry["type"] == "mutant":
            ortho = orthogonal_mutants.get(
                f"{entry['parent']}__{entry['mutations']}"
            )
        else:
            ortho = orthogonal.get(entry["design_id"])
        if ortho:
            recovery = number(ortho.get("recuperation_contacts"))
            iptm = number(ortho.get("iptm_moyen"))
            entry["recuperation_contacts"] = recovery
            entry["confiance_modele_orthogonal"] = iptm
            entry["orthogonal_detail"] = ortho.get("detail", "")
            entry["pose"] = (
                "confirmee" if recovery >= CONTACT_RECOVERY_CONFIRMED
                and iptm >= ORTHOGONAL_IPTM_CONFIRMED else "contestee"
            )
        else:
            entry["recuperation_contacts"] = "NON MESUREE"
            entry["confiance_modele_orthogonal"] = "NON MESUREE"
            entry["orthogonal_detail"] = ""
            entry["pose"] = "non mesuree"
        # --- cross-especes et robustesse au predicteur ---
        # Un mutant n'a pas ete predit contre la souris : il herite des valeurs de son
        # parent pour la PARTIE CIBLE (meme epitope, meme H409), et ses propres colonnes
        # restent vides pour tout ce qui depend de sa sequence.
        reference = entry["parent"] if entry["type"] == "mutant" else entry["design_id"]
        species = cross.get(reference, {})
        inherited = entry["type"] == "mutant"
        entry["dpKa_humain_Boltz"] = (
            "" if inherited else species.get("dpKa_humain_Boltz", "")
        )
        entry["dpKa_souris_Boltz"] = (
            "" if inherited else species.get("dpKa_souris_Boltz", "")
        )
        entry["iptm_souris"] = "" if inherited else species.get("iptm_souris", "")
        entry["delta_iptm_souris"] = "" if inherited else species.get("delta_iptm", "")
        entry["epitope_souris_retrouve"] = (
            "" if inherited else species.get("recuperation_epitope_souris", "")
        )
        entry["mecanisme_conserve_souris"] = (
            "non mesure (mutant)" if inherited
            else species.get("mecanisme_conserve", "non mesure")
        )

        # ESM-2 : DESCRIPTIF. N'entre dans aucun tri — voir le docstring du module.
        key = (f"{entry['design_id']}__{entry['mutations']}"
               if entry["type"] == "mutant" else entry["design_id"])
        entry["ESM2_PLL"] = esm2.get(key, {}).get("ESM2_PLL_par_residu", "non mesure")

    def mechanism_robustness(entry: dict) -> tuple[str, int, int]:
        """Le mécanisme pH tient-il sur PLUSIEURS structures et sur LES DEUX espèces ?

        POURQUOI CE DURCISSEMENT. Le ΔpKa publié jusqu'ici venait d'une seule structure AF2.
        Trois mesures indépendantes existent maintenant : AF2 humain, Boltz humain, Boltz
        souris. Elles ne concordent pas toujours, et les désaccords tombent EXACTEMENT sur
        les cas limites — `36dbfc4737a3e59b_seq1` passe de +0,96 sur AF2 à −0,17 sur Boltz.
        Un mécanisme qui ne survit pas au changement de structure n'est pas un mécanisme,
        c'est une propriété de la structure.

        Et comme l'objectif n°2 exige la même séquence sur P00533 ET Q01279, un mécanisme pH
        perdu chez la souris ne sert pas le challenge.
        """
        values = [
            entry.get(key) for key in
            ("dpKa_H409", "dpKa_humain_Boltz", "dpKa_souris_Boltz")
        ]
        numbers = [number(v) for v in values]
        numbers = [v for v in numbers if v == v]
        if not numbers:
            return ("non mesure", 0, 0)
        positive = sum(1 for v in numbers if v > PKA_NOISE)
        negative = sum(1 for v in numbers if v < -PKA_NOISE)
        if positive == len(numbers) and len(numbers) >= 3:
            return ("robuste", positive, len(numbers))
        if positive:
            return ("non reproductible", positive, len(numbers))
        if negative == len(numbers):
            return ("contre-selectif", positive, len(numbers))
        return ("neutre", positive, len(numbers))

    # --- robustesse du mécanisme pH, calculée AVANT les groupes qui en dépendent ------
    for entry in entries:
        verdict, positive, total = mechanism_robustness(entry)
        entry["mecanisme_robustesse"] = verdict
        entry["mesures_pH_positives"] = f"{positive}/{total}" if total else ""

    # --- groupes et tri ----------------------------------------------------------------
    for entry in entries:
        robust = entry["mecanisme_robustesse"] == "robuste"
        confirmed = entry["pose"] == "confirmee"
        if robust and confirmed:
            group = 1
        elif robust:
            group = 2
        elif confirmed:
            group = 3
        else:
            group = 4
        # RÈGLE MUTANT : seul un ROUGE, une impossibilité structurale ou une pose cassée
        # rétrograde. Un INDETERMINE reste à son rang, avec la mention — le threading ne
        # tranche pas, et on refuse de perdre un candidat sur un artefact de rotamère.
        if entry["type"] == "mutant" and (
            entry["verdict_carboxylate"] == "ROUGE"
            or entry["rotameres_retenus"].startswith("IMPOSSIBLE")
        ):
            group = 4
        entry["groupe"] = group

    def ph_tier(entry: dict) -> int:
        """Trois paliers de pH, et non une valeur continue.

        POURQUOI DISCRÉTISER. PROPKA se trompe couramment d'une unité de pKa. Classer deux
        designs à ΔpKa −0,01 et −0,40 l'un au-dessus de l'autre serait lire un ordre dans
        du bruit. On distingue donc seulement ce que l'outil peut distinguer : mécanisme
        présent, neutre, ou contre-sélectif. L'ordre FIN à l'intérieur d'un palier passe au
        critère de rang suivant, c'est-à-dire la cross-réactivité souris.
        """
        verdict = entry["mecanisme_robustesse"]
        if verdict == "robuste":
            return 0
        if verdict == "non reproductible":
            return 1
        if verdict in {"neutre", "non mesure"}:
            return 2
        return 3

    def sort_key(entry: dict) -> tuple:
        factor = entry["facteur_pH_predit"]
        factor = factor if isinstance(factor, float) and factor == factor else 0.0
        tier = ph_tier(entry)
        # Le facteur ne départage QUE dans le palier à mécanisme robuste, où les écarts
        # dépassent largement l'erreur de PROPKA.
        inside = -factor if tier == 0 else 0.0

        # OBJECTIF N°2 : on classe sur la MESURE et non sur le proxy. L'épitope retrouvé
        # chez la souris par un modèle indépendant est une évidence strictement plus forte
        # que la fraction de résidus identiques dans l'alignement. Le proxy reste en
        # départage pour les mutants, qui n'ont pas été prédits contre la souris.
        measured = number(entry.get("epitope_souris_retrouve"))
        proxy = entry["epitope_conservation_frac"]
        proxy = proxy if proxy == proxy else 0.0
        mouse = -measured if measured == measured else 0.0
        return (
            entry["groupe"],
            tier,
            inside,
            mouse,
            -proxy,
            entry["i_pAE_AF2"],
            -entry["i_pTM_AF2"],
        )

    entries.sort(key=sort_key)
    for rank, entry in enumerate(entries, start=1):
        entry["rang_global"] = rank
        entry["palier_pH"] = (
            "mecanisme robuste", "mecanisme non reproductible",
            "neutre", "contre-selectif",
        )[ph_tier(entry)]

    # --- phase 5 : analyse appariée ----------------------------------------------------
    by_id = {e["design_id"]: e for e in entries if e["type"] != "mutant"}
    pairs = []
    for entry in entries:
        if entry["type"] != "mutant":
            continue
        parent = by_id.get(entry["parent"])
        if parent is None:
            continue
        mutant_factor = entry["facteur_pH_predit"]
        parent_factor = parent["facteur_pH_predit"]
        gain = (
            round(mutant_factor / parent_factor, 3)
            if isinstance(mutant_factor, float) and isinstance(parent_factor, float)
            and parent_factor else ""
        )
        delta = entry["ddpKa_vs_parent"]
        spread = entry["dpKa_etendue_rotameres"]

        if entry["rotameres_retenus"].startswith("IMPOSSIBLE"):
            verdict = "degrade (structurellement impossible)"
        elif entry["verdict_carboxylate"] == "ROUGE":
            verdict = "degrade (carboxylate auto-destructeur)"
        elif entry["route_pH"].startswith("aucune"):
            verdict = "neutre (hors portee de H409)"
        elif isinstance(delta, float) and isinstance(spread, float) and spread > abs(delta):
            verdict = "non concluant (etendue rotamere > effet)"
        elif isinstance(delta, float) and delta > PKA_NOISE:
            verdict = "ameliore"
        elif isinstance(delta, float) and delta < -PKA_NOISE:
            verdict = "degrade"
        else:
            verdict = "neutre"

        pairs.append({
            "squelette": entry["squelette"],
            "parent": entry["parent"],
            "mutation": entry["mutations"],
            "dpKa_parent": parent["dpKa_H409"],
            "dpKa_mutant": entry["dpKa_H409"],
            "ddpKa": delta,
            "ddpKa_etendue_rotameres": spread,
            "facteur_parent": parent_factor,
            "facteur_mutant": mutant_factor,
            "gain_facteur": gain,
            "pKa_DE_min": entry["pKa_DE_min"],
            "pKa_DE_max": entry["pKa_DE_max"],
            "verdict_carboxylate": entry["verdict_carboxylate"],
            "rotameres_retenus": entry["rotameres_retenus"],
            "portee_H409": entry["pont_alerte"],
            "charge_parent": parent["charge_nette"],
            "charge_mutant": entry["charge_nette"],
            "recuperation_parent": parent["recuperation_contacts"],
            "recuperation_mutant": entry["recuperation_contacts"],
            "cout_structural": (
                round(entry["recuperation_contacts"] - parent["recuperation_contacts"], 3)
                if isinstance(entry["recuperation_contacts"], float)
                and isinstance(parent["recuperation_contacts"], float)
                else "non mesure (paire incomplete)"
            ),
            "verdict": verdict,
        })
    return entries, pairs


def main() -> None:
    entries, pairs = build()
    columns = [
        "rang_global", "groupe", "design_id", "squelette", "run", "type", "parent",
        "mutations", "sequence", "longueur", "charge_nette", "cysteines",
        "mecanisme_pH", "mecanisme_robustesse", "mesures_pH_positives", "palier_pH",
        "route_pH", "facteur_pH_predit", "dpKa_H409", "dpKa_humain_Boltz",
        "dpKa_souris_Boltz", "mecanisme_conserve_souris", "iptm_souris",
        "delta_iptm_souris", "epitope_souris_retrouve",
        "dpKa_etendue_rotameres", "facteur_min_rotamere", "facteur_max_rotamere",
        "pKa_H409_lie", "pKa_H409_libre",
        "pont_salin_WT_verifie", "pont_residu", "pont_distance_A", "pont_angle_deg",
        "bidente_goulot_A", "bidente_paire",
        "pont_alerte", "pKa_DE_min", "pKa_DE_max", "pKa_DE_etendue",
        "verdict_carboxylate", "rotameres_retenus", "ddpKa_vs_parent",
        "facteur_global_tous_groupes", "facteur_global_groupes_mobiles",
        "n_groupes_mobiles", "epitope_conservation_frac", "epitope_residus_comptes",
        "epitope_residus_divergents", "i_pTM_AF2", "i_pAE_AF2",
        "Interface_BuriedArea", "Hotspot_Contact_Fraction", "Off_Epitope",
        "pose", "recuperation_contacts", "confiance_modele_orthogonal",
        "orthogonal_detail", "ESM2_PLL", "residus_cible_contactes",
    ]
    with OUT_RANK.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    with OUT_PAIRS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)

    print(f"-> {len(entries)} candidats dans {OUT_RANK}")
    print(f"   {len(pairs)} paires dans {OUT_PAIRS}")
    print()
    for group in (1, 2, 3, 4):
        members = [e for e in entries if e["groupe"] == group]
        print(f"  groupe {group} : {len(members)} candidats")
    print()
    print("  TETE DE CLASSEMENT")
    for entry in entries[:8]:
        print(f"   {entry['rang_global']:3d}  G{entry['groupe']}  "
              f"{entry['design_id'][-26:]:<28} {entry['type']:<7} "
              f"facteur {entry['facteur_pH_predit']!s:<7} "
              f"dpKa {entry['dpKa_H409']!s:<7} "
              f"conserv {entry['epitope_conservation_frac']!s:<6} "
              f"i_pTM {entry['i_pTM_AF2']}")
    print()
    print("  VERDICTS APPARIES")
    tally: dict[str, int] = {}
    for pair in pairs:
        key = pair["verdict"].split(" (")[0]
        tally[key] = tally.get(key, 0) + 1
    for key, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"   {key:<16} {count}")


if __name__ == "__main__":
    main()
