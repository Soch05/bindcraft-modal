# NOTES — BindCraft sur Modal

## Environnement local (23/09)

- MacBook Air 2018, Intel x86_64, macOS (Darwin 23.6).
- venv : `uv venv --python 3.12` → CPython 3.12.13, dans `.venv/`.
- `modal` : **1.5.5**.
- **Accroc :** `uv pip install modal` échoue seul. `modal` tire `cbor2>=6`, qui compile
  une extension Rust et n'a pas de wheel pour macOS x86_64 → `error: can't find Rust
  compiler`. Corrigé en épinglant **`cbor2==5.9.0`** (wheel disponible). Le pin est dans
  `pyproject.toml` — ne pas l'enlever, sinon l'install casse à nouveau.

## Cible de démo

`inputs/PDL1.pdb` — 923 lignes, 921 atomes, **115 résidus, chaîne A unique**.
Avec un binder de 50-130 résidus, le complexe plafonne à ~245 résidus, très en dessous
du seuil de ~550 sur 32 Go (règle 7). La mémoire GPU n'est donc pas limitante sur cette
cible, et le choix du GPU se fait sur le seul critère vitesse/prix.

## Décisions d'architecture

**Base :** `modal_bindcraft.py` de `hgbrian/biomodals` (1157 lignes), copié puis adapté
par edits chirurgicaux — 5 zones touchées, la logique BindCraft intacte.
BindCraft épinglé au commit `c0a48d595d4976694aa979438712ac94c16620bb`.

**Poids AF2 : laissés dans l'Image, contrairement à l'architecture cible de CLAUDE.md.**
Décision prise sciemment. La couche `aria2c` qui télécharge les 5,3 Go est placée *avant*
`set_up_pyrosetta()` et avant le `uv_pip_install` final de `jax[cuda]`/`numpy`/
`matplotlib`. Donc ajuster les pins qu'on est le plus susceptible de toucher (règles 2 et 8)
**ne réinvalide pas** la couche des poids — le risque que CLAUDE.md veut éviter est déjà
neutralisé par l'ordre des couches. Déplacer vers un Volume aurait coûté une fonction de
download one-shot plus le patch des chemins que BindCraft attend en dur
(`/root/bindcraft/params`), pour économiser un téléchargement de ~1 min en datacenter.
À revoir si on finit par modifier `apt_install` ou le premier pin `numpy<2.0`.

**Volume pour les sorties : obligatoire, pas optionnel.** La référence *retourne* les
fichiers par valeur et les écrit en local depuis le `local_entrypoint`. Ce design est
**incompatible avec `--detach`** (règle 4) : le client se déconnecte, la valeur de retour
ne parvient à personne, et `modal volume get` n'a rien à lire. Adaptation :

- Volume `bindcraft` monté sur `/outputs`, `design_path = /outputs/<run_name>/`.
  BindCraft écrit donc directement dans le Volume — un run tué ou crashé laisse ses
  résultats partiels, ce qui est précisément ce qu'on veut pour diagnostiquer.
- `volume.commit()` dans un `finally`, via un wrapper fin autour de la fonction d'origine
  (renommée `_bindcraft`) pour éviter de réindenter 1000 lignes. La doc Modal ne promet la
  visibilité hors conteneur qu'après un commit explicite ; on ne s'appuie pas sur un
  commit implicite non documenté.
- Le retour ne contient plus qu'un inventaire (`n_files` + liste), pas les octets.

## Versions épinglées dans l'image

Héritées de la référence, non modifiées à ce stade :

| Composant | Pin |
|---|---|
| python (image) | 3.11 |
| numpy | `<2.0`, posé **avant** tout le reste et re-posé après PyRosetta |
| jax | `jax[cuda]<0.7.0` — évite la suppression de `wraps` en 0.7.0 |
| matplotlib | `==3.8.1` — BindCraft issue #4 |
| BindCraft | commit `c0a48d5` |
| ColabDesign | `git+main` (⚠️ **flottant**, contraire à la règle 2 — à épingler si un échec apparaît ici) |
| AF2 params | `alphafold_params_2022-12-06.tar` |

## Garde-fous déjà présents dans la référence

Vérifié en lisant le code, pas supposé :

- `TIMEOUT` est en **minutes**, défaut 300 → `timeout=18000s` = 5 h. Généreux, pas trop court.
  Le commentaire du code note qu'un timeout trop haut rend le provisionnement GPU plus dur.
- `max_trajectories` existe déjà comme paramètre → plafond dur disponible.
- `enable_rejection_check` + `acceptance_rate` : le script s'arrête de lui-même si le taux
  d'acceptation s'effondre. Pas de risque de boucle qui brûle les crédits toute la nuit.

## Coûts attendus

Du docstring de la référence, pour **3** designs sur PDL1 :

| GPU | Coût | Durée |
|---|---|---|
| A10G | $2 | 1,5 h |
| A100 | $3 | 1 h |
| H100 | $4 | 40 min |

À `--number-of-final-designs 1`, compter ~⅓ : **~$1, ~20-30 min**. Sous le seuil de la
règle 9. **GPU retenu : L40S** (48 Go, défaut de la référence) — surdimensionné pour cette
cible, mais généralement plus rapide à provisionner qu'un A100.

## Smoke test `check_gpu` — PASSÉ (23/09)

Workspace Modal : `soch05`.

```
GPU=L40S .venv/bin/modal run modal_bindcraft.py::check_gpu
```

Sortie réelle, premier build (~10 min) :

```
NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0
NVIDIA L40S    3MiB / 46068MiB
jax 0.6.2 | jaxlib 0.6.2 | numpy 1.26.4
jax.devices() -> [CudaDevice(id=0)]
Built image im-7noEgKlToD1tPPeGN48xzI
```

**Versions effectivement résolues** (à noter : `jax[cuda]<0.7.0` donne **0.6.2**, et
`numpy<2.0` donne **1.26.4**. Ce sont les versions à réépingler à l'identique si un
run futur casse) :

| Composant | Résolu |
|---|---|
| jax / jaxlib | 0.6.2 |
| numpy | 1.26.4 |
| driver NVIDIA | 580.95.05, CUDA 13.0 |
| GPU | L40S, 46 Go |

Poids AF2 présents dans l'image : 16 fichiers — les 5 modèles en 3 variantes
(`params_model_N.npz`, `_ptm`, `_multimer_v3`) + LICENSE.
Les deux JSON de settings par défaut présents et non modifiés.

**Cache de build vérifié :** 2ᵉ exécution de la même commande = **12 s** de bout en bout,
aucune ligne `Built image`. L'image et les 5,3 Go de poids viennent du cache. Le choix de
laisser les poids dans l'Image est donc validé en pratique.

## Journal des runs

| Date | Commande | GPU | Durée | Coût | Trajectoires | Acceptés |
|---|---|---|---|---|---|---|
| 23/09 | `check_gpu` (1er build) | L40S | ~10 min | ~0 (GPU qq s) | — | — |
| 23/09 | `check_gpu` (cache) | L40S | 12 s | ~0 | — | — |
| — | _run PDL1 : pas encore lancé_ | — | — | — | — | — |

## Critères d'acceptation — état

- [x] `jax.devices()` → GPU CUDA visible : `[CudaDevice(id=0)]` sur L40S
- [x] Image se construit sans erreur, 2e build caché (10 min → 12 s)
- [x] Poids AF2 ne se retéléchargent pas (cachés avec la couche de l'Image)
- [ ] Run détaché sur `PDL1.pdb` sans exception
- [ ] `modal volume get` ramène les sorties
- [ ] `Trajectory/`, `MPNN/`, `Accepted/`, `final_design_stats.csv`, `failure_csv.csv` existent
- [ ] On sait lire `failure_csv.csv`
- [ ] Durée et coût du run notés ici

## Risque identifié sur le run réel

`--number-of-final-designs 1` est un critère d'arrêt sur le **résultat**, pas sur l'effort :
le pipeline boucle jusqu'à ce qu'un design passe tous les filtres. L'estimation de ~$1 /
30 min suppose qu'un design passe vite. Si aucun ne passe, le run continue jusqu'à
l'arrêt par `enable_rejection_check` ou jusqu'au timeout de **5 h** — ce qui sortirait
largement du seuil « quelques dollars » de la règle 9.

D'où l'usage de `--max-trajectories` comme plafond dur sur le premier run. Le test valide
la plomberie, et CLAUDE.md dit explicitement que zéro design accepté ne l'invalide pas.
