# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""BindCraft 2.0 sur Modal — image, validation de build, et lancement de campagne.

Remplace l'entrypoint `modal_bindcraft.py` écrit pour BindCraft 1 au commit c0a48d5
(supprimé le 3 octobre, récupérable dans l'historique git).

Dépôt amont : https://github.com/PacesaLab/BindCraft2 — ce n'est PAS un tag de
martinpacesa/BindCraft, c'est un dépôt distinct.

L'image est portée de `containers/Dockerfile` de l'amont. Trois différences avec
BindCraft 1, toutes lues dans la source au commit épinglé :

  - PyRosetta, DSSP et DAlphaBall ont disparu. Zéro occurrence dans le dépôt. Plus de
    contrainte de licence académique, plus de `chmod` de binaires, et plus de pin
    `numpy<2.0` (il n'existait que pour PyRosetta).
  - Il n'y a aucune notion de TIMEOUT dans BindCraft 2.0. Le budget se pilote par
    `max_trajectories` et par le `timeout` de la fonction Modal, pas par une variable
    d'environnement interne.
  - Les poids ProteinMPNN sont livrés dans le paquet (77 Mo). Seuls les 5,3 Go d'AlphaFold
    sont téléchargés, et ils sont *bakés dans l'image* — choix reproduit de BindCraft 1.

Deux entrypoints, à nommer explicitement : `modal run` sans entrypoint explicite
n'exécute rien et sort en 0.

    modal run modal_bindcraft2.py::selfcheck
    modal run --detach modal_bindcraft2.py::design --run-name <nom> --max-trajectories <n>

Parallélisation : il n'y a PAS de sharding ici, contrairement à BindCraft 1. BindCraft 2.0
répartit lui-même des workers concurrents sur l'allocation visible — voir `WORKERS` et
`GPU_COUNT`. Deux variables d'environnement suffisent :

    WORKERS=1 modal run ...              # série, pour mesurer un temps par trajectoire
    GPU_COUNT=4 modal run --detach ...   # 4 cartes dans un conteneur, une seule campagne
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from modal import App, Image, Volume

# ----------------------------------------------------------------------------------------
# Épinglages. Tout chiffre ici doit être justifié dans NOTES.md.
# ----------------------------------------------------------------------------------------

# HEAD de PacesaLab/BindCraft2 au 3 octobre, "Update pdl1_cyclic_peptide.json".
BC2_COMMIT = "a8d0f2002df373842b86a3c20c5a060c5cfdf980"

# L40S : compute capability 8.9, au-dessus du plancher 7.5 de CUDA 13. 46 Go vérifiés.
GPU = os.environ.get("GPU", "L40S")

# Nombre de cartes dans UN SEUL conteneur. BindCraft 2.0 a `auto_multi_gpu` à true et
# répartit lui-même ses workers sur toute l'allocation visible, donc monter ce chiffre
# parallélise sans sharder.
#
# ⚠️ NE PAS revenir au sharding de BindCraft 1 (`shard-000`, `shard-001`…). Sur 2.0 chaque
# shard serait une campagne indépendante chassant son propre `number_of_final_designs` :
# N shards = N × les designs et N × le coût. Et le barreau de l'échelle de désespoir est
# « lu sur les tables de la campagne, donc chaque worker et une campagne reprise sont sur
# le même » — des shards séparés le compteraient chacun de leur côté.
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))

# `workers_per_gpu` : workers de design concurrents PAR CARTE.
#
# `auto` résout à 7 puis se fait plafonner par la mémoire, dans
# `bindcraft/design_workers.py` :
#     mem_par_worker = 2.0 × (3.4 + 38000 × n_residus² / 1e9)
#     workers        = (libre_Go − 4) // mem_par_worker
# Calculé le 3 octobre pour notre cible (198 résidus) sur une L40S de 46 Go :
#     binder 55 → 253 résidus → 11,66 Go/worker → 3 workers
#     binder 95 → 293 résidus → 13,32 Go/worker → 3 workers
# On ne tombe à 1 worker que vers 430 résidus au total.
#
# ⚠️ Conséquence pour une MESURE : avec 3 workers, trois trajectoires partent en même
# temps. Le temps mural vaut alors une trajectoire plus la compilation, et on ne peut PAS
# en déduire un temps par trajectoire. Mettre WORKERS=1 pour calibrer, laisser `auto` pour
# produire.
WORKERS = os.environ.get("WORKERS", "auto")

# Installation éditable à cette racine : `settings/` et `scaffolds/` vivent à la racine du
# dépôt et non dans le paquet, et le runtime les trouve relativement à lui. Une install
# classique ne copierait que le paquet et les orphelinerait.
BC2_ROOT = "/opt/bindcraft"

# `model_weights.SHIPPED_WEIGHTS` = <dir du paquet>/weights, soit ceci en install éditable.
# `alphafold_parameters()` y cherche `alphafold/` avant de télécharger : baker là rend le
# téléchargement inutile au runtime. Vérifié par `selfcheck`.
SHIPPED_WEIGHTS = f"{BC2_ROOT}/bindcraft/weights"

OUTPUTS = "/outputs"

# Modal attend "L40S" pour une carte, "L40S:4" pour quatre dans le même conteneur.
GPU_SPEC = GPU if GPU_COUNT == 1 else f"{GPU}:{GPU_COUNT}"

# ----------------------------------------------------------------------------------------
# La cible et les biais de ciblage. Numérotation PDB, celle du fichier cible.
# ----------------------------------------------------------------------------------------

TARGET_PDB = "inputs/6ARU_A_309-506.pdb"
TARGET_NAME = "hEGFR_dIII"
TARGET_CHAINS = "A"

# Jeu arrêté le 3 octobre, motifs dans NOTES.md : étendue CA 16,73 Å, identité humain/souris
# 4/4, D323 porte la route pH 1 (carboxylate 75,0 Å²), H409 la route pH 2 (His conservée).
HOTSPOTS = "A318,A323,A406,A409"

# A359 est un His en numérotation PDB — vérifié par lecture directe du fichier cible, pas
# de mémoire. Il diverge en Arg chez la souris et porte 16 % de la surface apolaire de sa
# zone : le contacter mettrait en danger l'objectif n°2. BindCraft 1 ne pouvait que
# l'éviter par choix de fenêtre et espérer.
#
# A359 est sûr parce qu'il est à 13,50 Å CA du hotspot le plus proche (A323), donc hors du
# rayon de 8,0 Å des pertes coldspot de BindCraft 2.0 (`bindcraft/loss.py`, cutoff=8.0).
#
# ⚠️ A325 est volontairement ABSENT, contre ce que NOTES.md du 3 octobre en faisait un
# « candidat coldspot au même titre que H359 ». Mesuré le 3 octobre : A325 est à
# **6,36 Å CA de A323**, sous le rayon de 8,0 Å. Le déclarer coldspot ferait repousser le
# binder hors de D323, qui porte la route pH n°1 — soit sacrifier l'objectif le mieux
# classé pour écarter un risque glycanique sur un résidu qu'on a déjà retiré des hotspots.
# Ne pas le rajouter sans avoir d'abord mesuré le séquon dans PyMOL.
#
# ⚠️ Incomplet : CLAUDE.md §8 action 4 demande aussi d'écarter les résidus à moins de ~10 Å
# d'un séquon. Cette distance n'a jamais été mesurée depuis les hotspots retenus. Ne pas
# inventer la liste — la mesurer, puis l'ajouter ici, et vérifier chaque candidat contre le
# rayon de 8,0 Å comme ci-dessus.
COLDSPOTS = "A359"

# Catégorie minibinders d'Adaptyv : 40–100 inclus. 55–95 laisse une marge aux deux bornes.
BINDER_LENGTHS = [55, 95]

# ----------------------------------------------------------------------------------------
# Image. Portée de containers/Dockerfile de l'amont.
# ----------------------------------------------------------------------------------------

# jax[cuda13] garde ses bibliothèques sous site-packages/nvidia/*/lib, où le loader ne
# regarde pas. Sans ce fichier, jax avertit une fois puis tourne sur le CPU, cent fois plus
# lentement, et rien ensuite ne le signale.
_LDCONF = (
    "import nvidia, pathlib; "
    "print(chr(10).join(sorted(str(p) "
    "for r in nvidia.__path__ for p in pathlib.Path(r).glob('*/lib'))))"
)

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "ca-certificates")
    .env({"PYTHONUNBUFFERED": "1", "PIP_NO_CACHE_DIR": "1"})
    .run_commands(
        f"git clone https://github.com/PacesaLab/BindCraft2.git {BC2_ROOT}",
        f"cd {BC2_ROOT} && git checkout {BC2_COMMIT}",
    )
    .run_commands(f"python -m pip install -e '{BC2_ROOT}[cuda13]'")
    .run_commands(
        f'python -c "{_LDCONF}" > /etc/ld.so.conf.d/bindcraft-cuda.conf',
        "ldconfig",
        # Échoue au build plutôt qu'à la minute 40 d'une campagne si les wheels ne
        # s'alignent pas.
        "ldconfig -p | grep -q libcupti",
    )
    # 5,3 Go, une fois, dans la couche d'image. Placé après l'install et avant la
    # vérification : une invalidation de cache en amont ne doit pas le refaire pour rien.
    .run_commands(
        f"BINDCRAFT_WEIGHTS={SHIPPED_WEIGHTS} bindcraft fetch-weights",
        f"test -d {SHIPPED_WEIGHTS}/alphafold",
    )
    .run_commands(
        'python -c "import jax, bindcraft.proteinmpnn; print(\'jax\', jax.__version__)"',
        "bindcraft --help > /dev/null",
        # pip signale un succès pour un environnement auquel il manque un module. Ceci
        # nomme chaque module et chaque checkpoint.
        "python -m bindcraft.selfcheck cuda13 --shipped-only",
    )
    .add_local_dir("inputs", remote_path="/root/inputs")
)

app = App("bindcraft2")
volume = Volume.from_name("bindcraft", create_if_missing=True)


# ----------------------------------------------------------------------------------------
# Réglages de campagne
# ----------------------------------------------------------------------------------------


def campaign_settings(run_name: str, max_trajectories: int, n_designs: int) -> dict:
    """Construit les réglages de la campagne.

    Les seuils laissés de côté viennent de `settings/core/default.json` de l'amont, lu au
    commit épinglé. ⚠️ Ils sont sur [0,1], pas sur 0–100 ni en Å : `i_pAE` 0,35,
    `i_pTM` 0,70, `Unbound_Binder_pLDDT` 0,80, `Interface_Residues` 7.
    """
    return {
        "campaign_name": run_name,
        "project_folder": f"{OUTPUTS}/{run_name}",
        "modality": "binder",
        "binder_lengths": BINDER_LENGTHS,
        "number_of_final_designs": n_designs,
        # Le seul garde de budget interne à BindCraft 2.0 : il n'y a pas de TIMEOUT.
        "max_trajectories": max_trajectories,
        # Écrit explicitement pour que `campaign_metadata.json` garde la trace de la
        # concurrence sous laquelle la mesure a été prise. `auto` donne 3 ici (cf. WORKERS).
        "workers_per_gpu": WORKERS,
        # Adaptyv exprime en acellulaire : pas de cystéines libres. Ce réglage les interdit
        # à la source plutôt que de les filtrer après coup.
        "aa_bias": {"C": 0},
        "targets": [
            {
                "name": TARGET_NAME,
                "target_path": f"/root/{TARGET_PDB}",
                "chains": TARGET_CHAINS,
                "hotspots": HOTSPOTS,
                "coldspots": COLDSPOTS,
                "weight": 1.0,
            }
        ],
    }


def _run(command: list[str], cwd: str | None = None) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


# ----------------------------------------------------------------------------------------
# Entrypoints
# ----------------------------------------------------------------------------------------


@app.function(image=image, gpu=GPU, volumes={OUTPUTS: volume}, timeout=600)
def selfcheck() -> None:
    """Valide le build avant toute dépense. ~1 min de GPU.

    Un build raté coûte le run entier, et un exit code 0 ne prouve pas qu'un GPU a été vu :
    jax tombe sur le CPU en silence. Cette fonction échoue si ce n'est pas un GPU CUDA.
    """
    import jax

    devices = jax.devices()
    print(f"jax {jax.__version__} | backend {jax.default_backend()} | devices {devices}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"jax tourne sur {jax.default_backend()}, pas sur gpu : {devices}")

    from bindcraft.model_weights import SHIPPED_WEIGHTS as runtime_weights
    from bindcraft.model_weights import alphafold_parameters, missing_model_weights

    # L'emplacement baké doit être celui que le runtime interroge, sinon la campagne
    # retéléchargerait 5,3 Go à chaque conteneur.
    assert str(runtime_weights) == SHIPPED_WEIGHTS, (runtime_weights, SHIPPED_WEIGHTS)
    params = alphafold_parameters(download=False)
    if not params:
        raise RuntimeError(f"poids AF2 introuvables sous {SHIPPED_WEIGHTS}")
    print(f"poids AF2 : {params}")

    problems = missing_model_weights(params, str(runtime_weights / "proteinmpnn" / "weights_neutral"))
    if problems:
        raise RuntimeError("checkpoints manquants :\n  " + "\n  ".join(problems))
    print("checkpoints : les 7 modèles AF2 et les 3 variantes ProteinMPNN sont complets")

    # La cible est-elle bien montée et lisible ?
    target = Path("/root") / TARGET_PDB
    n_ca = sum(1 for line in target.read_text().splitlines() if line.startswith("ATOM") and line[12:16].strip() == "CA")
    print(f"cible {target} : {n_ca} résidus")

    print("\nBUILD VALIDE")


@app.function(image=image, gpu=GPU_SPEC, volumes={OUTPUTS: volume}, timeout=86400)
def design(run_name: str, max_trajectories: int, n_designs: int) -> None:
    """Lance une campagne. `resume` est à true par défaut dans BindCraft 2.0, donc un
    rappel sur le même `run_name` reprend là où le précédent s'est arrêté."""
    settings = campaign_settings(run_name, max_trajectories, n_designs)
    folder = Path(settings["project_folder"])
    folder.mkdir(parents=True, exist_ok=True)

    # Les réglages résolus sont écrits à côté des sorties : la campagne reste auditable
    # sans relire ce fichier source.
    config = folder / "settings.json"
    config.write_text(json.dumps(settings, indent=2))
    print(json.dumps(settings, indent=2), flush=True)
    volume.commit()

    try:
        _run(["bindcraft", "design", str(config)], cwd=BC2_ROOT)
    finally:
        volume.commit()
        print(f"volume commité : {folder}", flush=True)


@app.local_entrypoint()
def main(run_name: str, max_trajectories: int = 10, n_designs: int = 10) -> None:
    print(
        f"GPU {GPU_SPEC} | workers_per_gpu {WORKERS} | run {run_name} | "
        f"max_trajectories {max_trajectories} | n_designs {n_designs}"
    )
    if WORKERS == "auto" and max_trajectories <= 4:
        print(
            "\n⚠️  workers_per_gpu=auto donne 3 workers sur une L40S pour cette cible, donc "
            f"ces {max_trajectories} trajectoires partiront en parallèle. Le temps mural ne "
            "donnera PAS un temps par trajectoire.\n"
            "    Pour calibrer : WORKERS=1 modal run ...\n"
            "    Pour produire : laisser auto et monter max_trajectories.\n"
        )
    design.remote(run_name=run_name, max_trajectories=max_trajectories, n_designs=n_designs)
