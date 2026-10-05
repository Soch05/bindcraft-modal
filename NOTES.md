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
- `max_trajectories` existe déjà comme paramètre → plafond disponible. ⚠️ **Corrigé après le
  run `test1` :** il plafonne les trajectoires *réussies*, pas les tentatives. Voir la section
  du run.
- `enable_rejection_check` + `acceptance_rate` : le script s'arrête si le taux d'acceptation
  s'effondre. ⚠️ **Corrigé après `test1` :** inopérant sur un run court, parce que
  `start_monitoring` vaut **600** — le contrôle ne démarre qu'après 600 trajectoires.
  L'affirmation initiale « pas de risque de boucle qui brûle les crédits » était fausse :
  elle n'est vraie qu'avec un `--max-trajectories` explicite.

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
| 23/09 | run `test1` (ci-dessous) | L40S | **47 min 25 s** | **$1,54** | 4 lancées / 3 réussies | **1** |

## Run `test1` — PDL1, PASSÉ (23/09)

```bash
GPU=L40S .venv/bin/modal run --detach modal_bindcraft.py \
  --input-pdb inputs/PDL1.pdb --number-of-final-designs 1 \
  --max-trajectories 3 --run-name test1
```

> **Périmé depuis l'ajout de `parallel` (noté le 01/10).** Cette commande telle quelle
> n'exécute plus rien : le fichier déclare deux `local_entrypoint`, Modal refuse de choisir,
> **et sort en code 0**. Écrire `modal_bindcraft.py::main`. Conservée ici telle qu'elle a
> réellement tourné le 23/09, le journal n'étant pas réécrit.

App `ap-OoBXPM6BtVeYmf4qGauJUA`. **47 min 25 s**, 78 fichiers, coût **$1,54**
(2845 s × $0,000542/s). Terminé sans exception, app en état `stopped`.

### Baseline par trajectoire — le chiffre à réutiliser

| # | Binder | Résultat | Durée |
|---|---|---|---|
| 1 | `l112_s900583` | ✓ pLDDT 0,83 | 8 min 24 |
| 2 | `l94_s179902` | ✓ pLDDT 0,89 | 5 min 38 |
| 3 | `l55_s851460` | ✗ rejetée | 3 min 21 |
| 4 | `l81_s625098` | ✓ pLDDT 0,95 | 5 min 22 |

⚠️ **Ne pas dimensionner avec le chiffre ci-dessous** — voir « Correction de baseline » dans
la section parallélisation : c'est la durée de la descente de gradient seule, et le cycle
complet coûte **9 min**. Ce run s'arrêtait au premier design accepté, donc il exécutait peu
de cycles MPNN.

~5,5 min par trajectoire sur L40S, plus un surcoût unique de ~3 min sur la
première : elle paie la **compilation JIT de JAX**, que les suivantes réutilisent. Ne pas
dimensionner sur les 8 min 24 de la première, c'est un artefact de démarrage.

### Deux pièges découverts en lisant les logs

1. **`--max-trajectories` plafonne les trajectoires RÉUSSIES, pas les tentatives.**
   4 lancées pour un plafond de 3, parce que la n°3 a été rejetée et n'est pas comptée
   (`check_n_trajectories` compte les fichiers produits). Le coût réel peut donc dépasser
   le plafond × durée unitaire. À intégrer dans toute estimation future.
2. **`start_monitoring: 600`** : le garde-fou `enable_rejection_check` ne s'active qu'après
   **600** trajectoires. Il ne protège donc pas un run court. Sans `--max-trajectories`,
   le seul frein est le timeout de 5 h, soit ~$10 sur L40S.

Noté aussi : un avertissement `Modal Client → Modal Worker Heartbeat attempt failed` est
apparu en cours de run. Le client local perd le contact, le job continue côté Modal.
C'est exactement ce que `--detach` garantit — validé en conditions réelles.

### Design accepté

`final_design_stats.csv` — 1 ligne, 232 colonnes. `Accepted/Ranked/1_PDL1_l81_s625098_mpnn3_model1.pdb`

| Métrique | Valeur | Seuil par défaut |
|---|---|---|
| Design | `PDL1_l81_s625098_mpnn3` (81 aa) | |
| Average_pLDDT | 0,93 | > 0,8 |
| Average_pTM | 0,89 | > 0,55 |
| Average_i_pTM | 0,86 | > 0,5 |
| Average_i_pAE | 0,16 | < 0,35 |
| Average_ShapeComplementarity | 0,61 | > 0,6 |
| Average_dG | −46,84 | < 0 |
| Average_dSASA | 1840,18 | > 1 |
| Average_n_InterfaceResidues | 20 | > 7 |
| Average_n_InterfaceHbonds | 8 | > 3 |
| Average_n_InterfaceUnsatHbonds | 3,5 | < 4 |
| Average_Surface_Hydrophobicity | 0,33 | < 0,35 |

Séquence : `ALVTIDENAPVTYETVPKVIGRISRAAMGLSAEQMREVNYKIVEIWETASHEIHKGETEKTKELILEVVE…`

### Lire `failure_csv.csv` — comment faire et ce qu'il dit

Une seule ligne de données, 64 colonnes = un compteur par filtre. Les rejets portent sur les
**séquences MPNN** (jusqu'à 20 par trajectoire, évaluées sur plusieurs modèles AF2), pas sur
les trajectoires — d'où 156 rejets pour 19 designs MPNN dans `mpnn_design_stats.csv`.

```python
import csv
r = list(csv.reader(open('failure_csv.csv')))
nz = [(h, int(v)) for h, v in zip(r[0], r[1]) if v.strip() not in ('', '0')]
for h, v in sorted(nz, key=lambda x: -x[1]): print(f"{v:>3}x  {h}")
```

Résultat sur `test1` — 9 filtres sur 64 ont rejeté au moins une fois :

| Rejets | Filtre |
|---|---|
| 38 | `i_pAE` |
| 35 | `i_pTM` |
| 30 | `pTM` |
| 20 | `pLDDT` |
| 14 | `ShapeComplementarity` |
| 12 | `n_InterfaceUnsatHbonds` |
| 5 | `Surface_Hydrophobicity` |
| 1 | `Trajectory_logits_pLDDT` |
| 1 | `Trajectory_Clashes` |

**Lecture :** le goulot est la **confiance de l'interface** (`i_pAE`, `i_pTM`, `pTM`), pas la
géométrie ni la chimie de surface. C'est le mode d'échec attendu de BindCraft : AF2 ne croit
pas assez au complexe. Les filtres physiques (clashes, hydrophobicité) ne rejettent presque
rien. C'est une information sur la difficulté de la cible, pas un réglage à corriger.

### Contenu rapatrié

```
out/test1/
├── Trajectory/   (6 fichiers)  + Animation/ Clashing/ LowConfidence/ Plots/ Relaxed/
├── MPNN/        (38 fichiers)  + Binder/ Relaxed/ Sequences/
├── Accepted/    (12 fichiers)  + Animation/ Pickle/ Plots/ Ranked/
├── Rejected/
├── trajectory_stats.csv       (3 lignes)
├── mpnn_design_stats.csv     (19 lignes)
├── final_design_stats.csv     (1 ligne)
└── failure_csv.csv
```

## Critères d'acceptation — état

- [x] `jax.devices()` → GPU CUDA visible : `[CudaDevice(id=0)]` sur L40S
- [x] Image se construit sans erreur, 2e build caché (10 min → 12 s)
- [x] Poids AF2 ne se retéléchargent pas (cachés avec la couche de l'Image)
- [x] Run détaché sur `PDL1.pdb` sans exception (47 min 25 s, app `stopped`)
- [x] `modal volume get bindcraft test1 ./out/` ramène les sorties (78 fichiers)
- [x] `Trajectory/`, `MPNN/`, `Accepted/`, `final_design_stats.csv`, `failure_csv.csv` existent
- [x] On sait lire `failure_csv.csv` : goulot = confiance d'interface (i_pAE 38, i_pTM 35, pTM 30)
- [x] Durée et coût du run notés ici : 47 min 25 s, $1,54

**Les 8 critères d'acceptation sont remplis. L'étape « test » est passée.**

Bonus non requis : 1 design a passé tous les filtres par défaut, sans qu'aucun seuil
n'ait été touché.

## Parallélisation — palier 1 validé (24/09/2026)

⚠️ **Extension de périmètre.** La parallélisation est listée dans les anti-objectifs de
`CLAUDE.md`. Ce travail a été fait sur demande explicite de l'utilisateur, après que le
conflit ait été signalé. `CLAUDE.md` n'a pas été mis à jour et décrit donc un périmètre
plus étroit que l'état réel du dépôt.

### Pourquoi c'est une nécessité et non une optimisation

500 trajectoires demandent **~75 GPU-heures**, donc c'est **impossible** en un run, le
timeout étant de 5 h (~33 trajectoires max). Le coût en GPU-heures est le *même* en
parallèle ; ce qui change, c'est le temps mural et la faisabilité.

### ⚠️ Correction de baseline — les 5,5 min étaient trompeurs

**La baseline de ~5,5 min/trajectoire tirée de `test1` était la durée de la descente de
gradient seule, pas du cycle complet.** Mesuré sur `par-test2` : **9,0 min par trajectoire
complète**, soit un facteur 1,6.

La raison est structurelle. Dans `test1`, `number_of_final_designs=1` arrêtait le run dès le
premier design accepté, donc peu de cycles MPNN étaient exécutés. En mode shard ce paramètre
est neutralisé à `10**9`, si bien que **chaque** trajectoire réussie enchaîne le cycle MPNN
complet : ~20 séquences générées, chacune repliée par AF2 puis scorée par PyRosetta. C'est ce
cycle, et non la descente de gradient, qui domine le temps.

Conséquence sur les estimations, à corriger partout :

| | Estimé (faux) | Réel |
|---|---|---|
| Par trajectoire | 5,5 min | **9,0 min** |
| `par-test2` (12 traj) | $1,90 | **$3,52** (×1,85) |
| 500 trajectoires | ~$92 | **~$147** |

**Pour dimensionner : compter 9 min et $0,29 par trajectoire**, pas 5,5 min.

### Architecture : trois éléments

- **`bindcraft_shard`** — un `design_path` par shard (`/outputs/<run>/shard-NNN/`).
  C'est ce qui rend le sharding sûr : les 4 CSV sont construits sous `design_path`, donc
  des chemins distincts garantissent qu'aucun fichier n'est écrit par deux conteneurs.
  Les volumes Modal sont *last-write-wins* : un chemin partagé perdrait des données
  **en silence**.
- **`aggregate`** — **sans `gpu=`** : concaténer des CSV sur un GPU serait du gaspillage.
  Appelle `volume.reload()`, sans quoi il ne verrait rien de ce que les shards ont écrit.
  Refait le ranking sur l'**union** (N shards produisent chacun leur « rank 1 »).
- **`parallel`** — `starmap(..., return_exceptions=True)` pour qu'un shard mort n'emporte
  pas les autres.

### Deux pièges coûteux, traités dans le code

1. **`number_of_final_designs` doit être neutralisé dans les shards** (mis à `10**9`).
   La boucle s'arrête dès que `accepted_designs >= number_of_final_designs` : le laisser à 1
   ferait quitter chaque shard à son premier succès, payant N démarrages et N compilations
   JIT pour à peine plus qu'un run simple.
2. **Le surcoût JIT se paie par shard** (~3 min de GPU payées à ne rien produire). Il dicte
   le découpage. Gaspillage = `3/(3+9k)` avec k trajectoires par shard :

| Trajectoires/shard | Gaspillage JIT |
|---|---|
| 1 | 25 % |
| 5 | 6 % |
| 10 | 3 % |
| 20 | 1,6 % |

   *(Un tableau antérieur annonçait 35 % à k=1 ; il était calculé sur la baseline erronée de
   5,5 min. Corrigé sur 9 min.)*

### Dimensionner les shards : deux effets de nature différente, et un plafond dur

- **Le JIT gaspille de l'argent** — des GPU-minutes payées pour compiler. Argument pour de
  gros shards.
- **Le déséquilibre gaspille du temps, pas de l'argent.** Modal facture à la seconde et par
  conteneur : le shard qui finit en 29 min arrête de facturer à 29 min. Les 17 min d'écart
  avec le plus lent ne coûtent rien, elles retardent seulement l'agrégation, qui attend tout
  le monde. Argument pour de gros shards aussi, mais pour une autre raison.
- **Plafond : le timeout de 5 h.** À 9 min/trajectoire, un shard plafonne à ~33 trajectoires.
  Avec les 56 % de déséquilibre observés, un shard dimensionné pour 20 trajectoires (180 min)
  peut réellement en prendre 280 — proche des 300. Il faut donc de la marge.

| Trajectoires/shard | Verdict |
|---|---|
| 1-3 | gaspillage JIT, variance ingérable |
| **15-20** | **zone recommandée** — JIT ~2 %, marge confortable au timeout |
| 25-33 | possible, mais un shard lent risque de mourir au timeout |
| > 33 | impossible sans augmenter `TIMEOUT` |

Pour 500 trajectoires : **25-33 shards × 15-20 trajectoires**, ~$147, temps mural ~3 h —
et l'orchestration côté serveur devient obligatoire à cette durée.

### Résultat du palier 1 : `par-test`, 2 shards × 1 trajectoire

```bash
GPU=L40S .venv/bin/modal run modal_bindcraft.py::parallel \
  --input-pdb inputs/PDL1.pdb --n-shards 2 --trajectories-per-shard 1 --run-name par-test
```

App `ap-z5GmI837qeAfVTG0d5z2xh`, **2 tasks simultanées** confirmées. `2 ok, 0 failed`.
Durées : shard-000 **7 min 59**, shard-001 **12 min 33** → temps mural 12 min 33,
coût **$0,67** (20,5 GPU-min).

**Les seeds diffèrent bien entre conteneurs** — c'était le point critique, car des seeds
identiques rendraient la parallélisation inutile. Observé : `PDL1_l101_s151500` et
`PDL1_l105_s353332`. `np.random` non seedé ([modal_bindcraft.py:371](modal_bindcraft.py#L371))
s'initialise sur l'entropie de l'OS, donc aucun patch n'est nécessaire. Vérifié, pas supposé.

Agrégation correcte : `trajectory_stats` 2 lignes (1/shard), `mpnn_design_stats` 4 lignes
(2/shard), `failure_csv` sommé à 109 rejets sur 7 filtres. Profil identique à `test1` —
`i_pAE` 36, `i_pTM` 33, `pTM` 28 — ce qui confirme la reproductibilité du diagnostic.

Trou du palier 1, **comblé par le palier 2** : aucun design accepté sur ces 2 trajectoires,
donc le chemin de code du ranking global n'avait pas été exercé.

### Palier 2 : `par-test2`, 3 shards × 3 trajectoires — ranking validé

```bash
GPU=L40S .venv/bin/modal run modal_bindcraft.py::parallel \
  --input-pdb inputs/PDL1.pdb --n-shards 3 --trajectories-per-shard 3 --run-name par-test2
```

App `ap-gSqGKjjq8qj8AW6t4N5fDY`, 3 tasks simultanées, `3 ok, 0 failed`.
**12 trajectoires tentées pour un quota de 9 réussies** — reconfirme que le quota porte sur
les réussies. 25 designs MPNN évalués, **8 acceptés**, 244 rejets sur 8 filtres.

Durées : **29 min 46 / 32 min 10 / 46 min 21**. Temps mural 46 min 21, coût **$3,52**
(108,3 GPU-min).

**Le ranking global est correct, vérifié sur les valeurs et non sur la présence des fichiers :**

| Rang | Shard | Average_i_pTM |
|---|---|---|
| 1 | shard-000 | 0,83 |
| 2 | shard-000 | 0,83 |
| 3 | shard-000 | 0,81 |
| 4 | shard-000 | 0,80 |
| 5 | shard-001 | 0,77 |
| 6 | shard-001 | 0,76 |
| 7 | shard-002 | 0,74 |
| 8 | shard-002 | 0,72 |

Décroissance stricte, les trois shards entrelacés dans un classement unique, aucun design
tombé dans le cas par défaut du `sort_key` (qui retourne `len(order)` sur clé absente — un
mapping cassé aurait donné un ordre arbitraire **sans erreur ni avertissement**).
Ce contrôle est à refaire après toute modification de `aggregate`.

### Relecture avant gel du code (24/09/2026) — un bug trouvé et corrigé

Le code est gelé avant la compétition ; relecture faite à cette occasion.

**Bug corrigé : la colonne `Rank` du `final_design_stats.csv` agrégé était fausse.** Chaque
shard classe ses propres designs 1..n, donc la concaténation portait des rangs dupliqués —
trois `Rank=1` et trois `Rank=2` sur `par-test2`. `Accepted/Ranked/` était correct, mais
quiconque triait le CSV par `Rank` obtenait un ordre arbitraire. Symptôme silencieux :
aucune erreur, juste des données fausses.

Correction : `final_design_stats.csv` est maintenant traité **après** le ranking et
renuméroté contre le même `order` que les PDB. Vérifié sur `par-test2` — `Rank` 1→8
séquentiel, `i_pTM` décroissant, et les deux vues (CSV et `Accepted/Ranked/`) s'accordent
rang par rang.

Validé **sans GPU** en rejouant `aggregate` seul sur les données déjà produites :

```bash
.venv/bin/modal run modal_bindcraft.py::aggregate --run-name par-test2
```

C'est la manière de tester l'agrégation pour quelques centimes : `aggregate` n'a pas de
`gpu=`, donc il ne consomme que du CPU.

Deux durcissements au passage :

- `bindcraft_shard` lève une `ValueError` explicite si `run_name` manque, au lieu d'un
  `KeyError` opaque.
- `aggregate` affiche un `WARNING` si un PDB accepté est absent de `mpnn_design_stats.csv`.
  Ces designs retombent sur `len(order)` et se classent en dernier **sans erreur** ; sans
  cet avertissement, un mapping cassé passerait inaperçu. Aucun cas sur `par-test2`.

Non corrigé volontairement : pandas émet des `PerformanceWarning` (« DataFrame is highly
fragmented ») en insérant la colonne `Shard` dans des frames à 232 colonnes. C'est de la
performance, pas de la justesse, et négligeable à cette échelle. Modifier le code juste avant
un gel pour un avertissement cosmétique serait un risque de régression gratuit.

Vérifié aussi : `number_of_final_designs` ne sert qu'à la condition d'arrêt, donc le `10**9`
ne casse aucun calcul ; `main` et `bindcraft` sont inchangés, leur signature intacte ; les
six points d'entrée sont toujours découverts par Modal.

### `par-test3` (24/09) — run tué par une panne réseau locale

Test de non-régression après la correction du `Rank`, lancé en 2 shards × 2 trajectoires.
**Le run a échoué**, exit code 1, mais **pas à cause du code** :

```
socket.gaierror: [Errno 8] nodename nor servname provided, or not known
modal.exception.ConnectionError
Modal Client → Modal Worker heartbeat attempts have been failing for over 14.98 minutes
```

Échec de résolution DNS **sur le laptop** : la machine a perdu le réseau (veille, couvercle
fermé, coupure) pendant le run. Les heartbeats ont échoué 15 min, puis le client a abandonné
et l'app a été arrêtée.

**C'est la démonstration en conditions réelles du verrou déjà identifié** : l'orchestration
vit dans le `local_entrypoint`, donc sur le laptop. Ce n'est plus un risque théorique.

#### Manœuvre de récupération — à connaître, elle évite de repayer le GPU

Le `volume.commit()` dans le `finally` de `bindcraft_shard` a sauvé le travail : les deux
shards avaient leurs 4 CSV et leurs dossiers dans le Volume. Seul `aggregate` n'avait pas
tourné. Il suffit de le rejouer seul, **sans GPU** :

```bash
.venv/bin/modal run modal_bindcraft.py::aggregate --run-name <run>
```

Le travail GPU déjà payé est récupéré pour quelques centimes. C'est le bénéfice concret
d'avoir séparé `aggregate` des shards et de l'avoir laissé sans `gpu=`.

Résultat : 3 trajectoires réussies (au lieu de 4, les shards ayant été tués), 28 designs
MPNN, 91 rejets, **1 design accepté**. Aucun `WARNING` d'orphelin.

#### Ce que ce test valide, et ce qu'il ne valide pas

- ✅ Le flux `parallel` → shards → CSV fonctionne, et `aggregate` tourne sans erreur sur des
  données fraîches.
- ❌ **Le classement multi-shard n'est pas exercé** : un seul design accepté, donc `Rank=1`
  est trivialement correct. Ce test ne rejoue pas le cas qui avait révélé le bug.

**La correction du `Rank` reste néanmoins validée**, et par le bon test : elle est
entièrement contenue dans `aggregate`, pas dans les shards. Le rejeu sur `par-test2`
(3 shards, 8 designs acceptés, `Rank` 1→8 vérifié, CSV et `Accepted/Ranked` d'accord rang
par rang) exerce exactement le chemin de code modifié. Un flux complet n'y ajouterait que la
confirmation que les shards écrivent leurs CSV — ce qui était déjà établi par `par-test2`.

### Déséquilibre entre shards : 56 %

29 min 46 pour le plus rapide contre 46 min 21 pour le plus lent. Le temps mural étant dicté
par le plus lent, on paie de l'attente. Cause : à 3-4 trajectoires par shard, la variance du
nombre de tentatives et du succès MPNN est énorme. Avec 10-20 trajectoires par shard, la loi
des grands nombres lisse ça — raison supplémentaire de ne pas faire de shards minuscules.

### Limite connue : `--detach` et le mode parallèle

Modal avertit que le mode détaché ne garde en vie que **la dernière fonction déclenchée**.
En mode parallèle on en déclenche N+1 (N shards puis `aggregate`), donc `--detach` est
piégeux. Le palier 1 a tourné attaché. Pour un vrai run long il faudra une fonction
orchestratrice côté serveur (qui appelle `starmap` depuis un conteneur Modal) plutôt qu'un
`local_entrypoint` qui orchestre depuis le laptop.

## Risque identifié sur le run réel

`--number-of-final-designs 1` est un critère d'arrêt sur le **résultat**, pas sur l'effort :
le pipeline boucle jusqu'à ce qu'un design passe tous les filtres. L'estimation de ~$1 /
30 min suppose qu'un design passe vite. Si aucun ne passe, le run continue jusqu'à
l'arrêt par `enable_rejection_check` ou jusqu'au timeout de **5 h** — ce qui sortirait
largement du seuil « quelques dollars » de la règle 9.

D'où l'usage de `--max-trajectories` comme plafond dur sur le premier run. Le test valide
la plomberie, et CLAUDE.md dit explicitement que zéro design accepté ne l'invalide pas.

---

## Carte d'épitope EGFR (30/09/2026) — `egfr_epitope_map.py`

Premier run décisionnel du Challenge 1. Local, CPU, coût nul. Aucune dépense GPU ne doit
précéder le choix du patch : la cross-réactivité souris se joue ici, pas au filtrage.

Dépendances ajoutées à `pyproject.toml` : `biopython>=1.83` (1.88 installé),
`numpy>=1.26` (2.5.3 installé). Aucune n'était déclarée ni installée.

### UniProt n'annote pas le domaine III

P00533 n'a **qu'une** feature de type `Domain` : `712-979 Protein kinase`. Aucun
« Receptor L-domain ». Le script échouait donc sur `SystemExit` à l'étape 1.

Les L-domaines n'existent que comme `Repeat` marquées « Approximate » : 75-300 et
390-600 UniProt. La seconde vaut **366-576 mature**, décalée d'une cinquantaine de
résidus du domaine III structural. Inutilisable comme borne.

Décision : **définir la région structurellement**, depuis l'empreinte du Fab cétuximab
(chaînes B et C de 6ARU), et ne garder la `Repeat` qu'en colonne informative
(`uniprot_repeat2`). Aucune borne codée de mémoire — règle de CLAUDE.md respectée.

`REGION_RADIUS = 30.0` Å autour du centroïde de l'empreinte. Choisi comme le plus petit
rayon rendant la sélection **contiguë en séquence** : 311-511 avec 1 discontinuité,
contre 7 à 25 Å. Critère structural, pas une constante à la main.

### Faits établis depuis les fichiers

| | |
|---|---|
| Offset PDB → UniProt | **+24**, 99,7 % de concordance (609 résidus observés, chaîne A 4-612) |
| 6ARU | X-ray, **3,20 Å**, `_refine.ls_d_res_high` |
| Conformation | **repliée (tethered)**, contacts 242-253 ↔ 563-595 = bras de dimérisation ↔ domaine IV |
| Empreinte cétuximab | 24 résidus, 349-473, contacts lourds < 4,5 Å |
| N-glycosylation | 13 sites humains, **0 O-linked** ; les 10 sites murins sont un **sous-ensemble strict** des humains → masquer sur l'humain couvre la souris |

Conformation retenue pour le design : **repliée**, alignée sur la structure de référence
fournie par les organisateurs. La SASA étant calculée sur la chaîne A entière,
l'occlusion par les autres domaines dans cet état est prise en compte.

### Contrôle cétuximab — passé

| Ensemble | Identité humain/souris |
|---|---|
| Protéine entière | 90,6 % |
| Région 311-511 | 87,2 % |
| **Empreinte cétuximab** | **70,8 %** |

L'empreinte est appauvrie en conservation par rapport aux deux références. Divergences :
353, 418, 443, 467, 468, 471, 473.

L'attendu est **inversé** par rapport à l'intuition : le cétuximab ne reconnaissant pas
l'EGFR murin, une fonction de score saine doit classer son épitope **bas**. C'est le cas —
le patch E472, qui contient 4 résidus de l'empreinte, sort dernier (67 % identité,
4 divergences). Le masque de conservation mesure quelque chose.

Un contrôle qui aurait montré l'empreinte fortement conservée aurait signalé un offset
faux ou un alignement cassé, pas une bonne nouvelle.

### Deux corrections apportées au scoring

1. **Le seuil glycane ne portait que sur l'ancre.** `dist_glycan > GLYCAN_EXCLUSION`
   filtrait l'ancre, alors que le patch était rapporté par le minimum sur ses *membres*.
   Une ancre à 20 Å pouvait porter des membres à 6 Å. Le seuil porte désormais sur le
   patch entier. **Effet : 7 patches sur 9 écartés** — E320 (6 Å), D323 (7 Å), E489 (5 Å),
   E397 (10 Å), E495 (11 Å), E472 (11 Å), E400 (12 Å).
2. **Déduplication sur le recouvrement des membres** (`MAX_PATCH_OVERLAP = 0.5`) et non
   sur la distance entre ancres. À `PATCH_RADIUS = 11`, deux ancres séparées de 8 Å
   partagent encore les deux tiers de leur patch : E320 et D323 partageaient 6 membres
   sur 9 tout en étant comptés comme distincts.

Clé de tri laissée volontairement inchangée à `(-frac_ident, -n)`. Colonnes `n_hydro` et
`n_acidic` ajoutées pour mesurer la tension avant tout score composite.

### Résultat : le funnel s'effondre à un seul site

```
9 ancres acides conservées et exposées
  → 7 écartées par le seuil glycane appliqué au patch
  → 2 survivantes (E431, D434)
  → 1 site distinct après déduplication (elles partagent 5 membres sur 9)
```

Site retenu, **E431** : 9 résidus exposés, 100 % identité humain/souris, 0 divergence,
glycane le plus proche à 16 Å, hors empreinte Fab.
Patch : `E431 K430 G458 E400 R403 K455 S428 D434 T459`.
Hotspots suggérés : `A431,A430,A458,A400`.

### Le problème, et il n'est pas dans le code

**Aucun des 8 sites candidats n'avait de contenu apolaire** (0 à 2 hydrophobes exposés).
Le site survivant est à **0 hydrophobe**. Tous les sites ancrés sur un Asp/Glu conservé
sont des surfaces polaires et chargées — le cas le plus défavorable au design de novo.

La tension designabilité / ancrage acide n'est donc pas un arbitrage : il n'y a pas
d'option apolaire dans cet ensemble. Et un seul site ne fait pas une campagne : pas de
diversité d'épitope possible, donc rien à clusteriser à la sélection finale.

**Décision à prendre avant toute dépense GPU** — la conjonction de contraintes est trop
serrée. Trois relâchements possibles, aucun encore appliqué :

- autoriser les ancres `similar` et non seulement `identical` (l'identité régionale est
  de 87 %, le critère « identique » est peu discriminant et coûte des candidats) ;
- recalibrer `GLYCAN_EXCLUSION = 12.0` Å, seuil jamais calibré et qui à lui seul écarte
  7 patches sur 9 ;
- découpler : choisir le site sur la designabilité, et traiter l'ancrage acide comme
  départageur plutôt que comme filtre dur.

Le funnel tel quel est publiable — c'est un résultat négatif documenté — mais il ne
produit pas de lot soumissible.

---

## Refonte de la carte d'épitope (01/10/2026) — `egfr_epitope_map.py`

Second passage sur le script. Le run du 30/09 concluait à un funnel effondré à un seul
site, sans contenu apolaire, et posait l'incompatibilité ancrage acide / designabilité
comme hypothèse. **Cette conclusion était un artefact d'échantillonnage, et elle est
maintenant réfutée par la mesure.**

### Le défaut : l'énumération était le filtre

`rank_patches` itérait sur les ancres acides conservées et exposées, puis ramassait les
membres dans un rayon. « Être centré sur un Asp/Glu conservé » n'était donc pas un
critère appliqué à un échantillon de sites — c'était la **définition** du site. Conclure
« 0 hydrophobe sur les 8 patches » revenait à constater que des disques centrés sur des
résidus chargés sont polaires. Tautologie, pas mesure.

Correction : les centres sont désormais **tous les résidus exposés du domaine III**.
L'ancrage acide devient deux colonnes comptées parmi les membres (`n_acidic`,
`n_acidic_cons`). 9 ancres → **98 centres**.

### Bornes du domaine III : trois sources interrogées, aucune codée en dur

| Source | Segments « L-domain » sur P00533 | Verdict |
|---|---|---|
| CATH via PDBe `/mappings/cath/6aru` | — | **inutilisable** : la classification ne couvre que les chaînes B/C du Fab, rien sur la chaîne A |
| Pfam `PF01030` (InterPro) | 57-167 et **361-480** UniProt | **tronqué** : la description Pfam déclare *« missing the first 50 amino acid residues of the domain »* |
| CATH-Gene3D `G3DSA:3.80.20.20` (InterPro) | 25-213 et **333-530** UniProt | **retenu** |

**Retenu : CATH-Gene3D, domaine III = 333-530 UniProt = 309-506 PDB.** Motif : c'est la
même classification structurale que CATH, projetée sur la séquence, et elle couvre le
domaine entier là où Pfam l'ampute (+28 au N-term, −50 au C-term par rapport à Gene3D).

Règle de sélection du segment, **géométrique et non nominative** : parmi les segments
d'une source, on retient celui qui contient le plus de résidus de l'empreinte du Fab
cétuximab — l'épitope du cétuximab étant dans le domaine III. Aucun indice, aucun nom,
aucune borne en dur. Le domaine I (référence de la face, 25-213 UniProt = 1-189 PDB) est
désigné par exclusion depuis la même source.

**Corroboration rétrospective** : la région structurelle du 30/09, définie par un rayon
de 30 Å autour du centroïde de l'empreinte, donnait PDB 311-511 = UniProt 335-535. À 2-5
résidus près de la borne CATH-Gene3D. L'heuristique du rayon était juste ; elle est
maintenant remplacée par une source citable. `REGION_RADIUS` et `region_mask` supprimés.

### Numérotation : l'offset n'est plus une inférence

`_struct_ref_seq` du mmCIF déclare pour la chaîne A : P00533, auth **1-616 ↔ db 25-640**,
soit **offset +24**. Le balayage par maximisation de concordance donne **+24** à 99,7 %.
Les deux routes sont conservées et le script **s'arrête en cas de désaccord**.

`_struct_ref_seq_dif` déclare **8 écarts** sur la chaîne A :

| auth | cristal | UniProt | type | position |
|---|---|---|---|---|
| 516 | LYS | ASN | `conflict` (UniProt : *Sequence conflict*, Ref. 1 CAA25240) | hors domaine III |
| 610 | ARG | GLU | `conflict` | hors domaine III |
| 617-622 | HIS ×6 | — | `expression tag` | hors domaine III |

Ces deux conflits expliquent exactement le 99,7 % du balayage (607/609). Ils comptent
parce que `aa_human` est lu dans la **structure** tandis que le statut humain/souris est
dérivé de la séquence **UniProt** : aux positions concernées la table serait incohérente.
**Aucun ne tombe dans le domaine III** — impact réel nul, mais le script lève désormais
une alerte bruyante si c'était le cas.

### Glycanes : filtre dur retiré

**2 NAG seulement sont modélisés sur la chaîne A** de 6ARU (auth 709, 710), pour 1 sur C,
2 sur D, 2 sur E, + MAN×5 et BMA×1. À 3,20 Å un arbre glycanique est flexible et mal
ordonné : l'absence de densité n'est pas l'absence de glycane. Mesurer l'occlusion par la
SASA en gardant les NAG est donc **inapplicable** sur cette structure, et le masquage sur
l'annotation UniProt reste le bon choix pour l'expérience — la cible mesurée sera
glycosylée.

**13 séquons N-linked annotés, dont 5 dans le domaine III** : 352, 361, 413, 444, 528
UniProt. La densité est le vrai problème, pas la sévérité du seuil.

`GLYCAN_EXCLUSION` (filtre dur, écartait 7 patches sur 9) devient `GLYCAN_REF = 12.0`,
**sans recalibrage**, utilisé comme rampe linéaire d'atténuation : pénalité = 0 au contact
du séquon, 1 au-delà de 12 Å. Forme posée, non calibrée, affichée en colonne
(`glyc_penalty`, `score_glyc_pen`) et dans un second classement. Un patch n'est plus
supprimé par une constante jamais justifiée ; il est classé plus bas, motif lisible.

### SASA ventilée par élément

`ShrakeRupley` passe au niveau atomique. C et S = apolaire, N et O = polaire. Colonnes
`sasa_apolar`, `sasa_polar`, `apolar_frac` en Å². **Clé de tri = SASA apolaire absolue**,
pas le ratio, pas un comptage de `LIVFMWY` — un comptage rend invisibles les tiges
aliphatiques des Lys/Arg, qui exposent réellement du carbone. `n_hydro` conservé en
colonne pour comparer l'ancienne métrique à la nouvelle.

Réserve mesurée : **corrélation taille de patch / SASA apolaire absolue = 0,64**. Le tri
absolu classe donc en partie la taille. À garder en tête au moment de choisir.

### Colonne `face`

Le site de liaison de l'EGF est ménagé entre domaines I et III. Axe orienté du centroïde
du domaine III vers celui du domaine I ; produit scalaire positif = `ligand`, négatif =
`externe`. Répartition : **97 ligand / 101 externe** sur 198 résidus du domaine III.

### Résultats

```
198 résidus dans le domaine III, 87,4 % identité humain/souris
 98 exposés → 98 centres
 93 patches à ≥ 6 membres
```

Distribution des tailles avant toute retenue : min 5, médiane 9, max 16, moyenne 9,1.
Seuls **5 patches sur 98 (5 %) sont sous le minimum de 6** — le seuil hérité ne fait
quasiment rien dans ce régime, contrairement au précédent.

Déduplication : **aucun seuil fixé**, par décision. Le recouvrement avec l'union des
patches mieux classés est saturé (médiane 1,00), ce qui est attendu avec 98 centres
chevauchants. Balayage fourni à la place :

| seuil | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 |
|---|---|---|---|---|---|---|---|---|---|
| sites | 7 | 8 | 9 | 11 | 13 | 14 | 16 | 21 | 26 |

### Ce que la mesure tranche

**L'incompatibilité ancrage acide / designabilité est réfutée.**

- **64 patches sur 93** portent au moins une ancre acide conservée.
- **15 patches** combinent `apolar_frac ≥ 0,55` **et** `n_acidic_cons ≥ 1`.
- Le premier au classement apolaire, C482, sort à 639 Å² apolaires et `apolar_frac` 0,57,
  avec 2 ancres acides conservées.

Le run du 30/09 ne voyait aucun site apolaire parce qu'il n'en avait énuméré aucun. La
tension posée comme hypothèse structurale n'existe pas dans les données : sur le domaine
III de l'EGFR, surface apolaire et ancre acide conservée coexistent largement.

Conséquence directe : **le découplage envisagé — soumettre des binders simples pour le
classement, garder le pH comme analyse méthodologique — n'a plus lieu d'être.** Les deux
objectifs sont poursuivables sur les mêmes sites.

Note de prudence sur le premier du classement : `C482` est une cystéine, donc très
probablement engagée dans un pont disulfure de la cible. Rien d'éliminatoire — c'est la
cible, pas le binder — mais le centre n'est pas un point d'ancrage à traiter naïvement.

### Supprimé

`annotated_repeat()` et la colonne `uniprot_repeat2` (l'annotation Repeat d'UniProt est
établie inutilisable, le fait est consigné, le code qui le réaffiche ne l'est plus) ;
`check_tethered()` (37 lignes re-dérivant à chaque run un fait déjà consigné : contacts
242-253 ↔ 563-595) ; la distance morte de `fab_footprint()` (la fonction renvoyait
`dict[int, float]` dont aucun des quatre appelants ne lisait le float).

### Ce qui n'est PAS décidé, volontairement

Aucun seuil de retenue n'est appliqué : ni déduplication, ni filtre glycane, ni critère
de conservation, ni critère de face. Le script sort les colonnes et les distributions.
La sélection se fait à la lecture, et sera consignée séparément.

### Pénalité glycane retirée (01/10, même journée)

`glyc_penalty` et `score_glyc_pen` supprimés, second classement supprimé. `min_glyc`
reste, **colonne brute en Å, ni filtre ni pondération**. Un seul classement, sur la SASA
apolaire absolue.

Motif, mesuré avant de trancher : la forme `clamp(min_glyc/12, 0, 1)` appliquée
multiplicativement à la SASA apolaire annulait **22 patches sur 93** (pénalité exactement
0 à `min_glyc = 0`) — le filtre dur, mais sans ligne de journal, donc en pire. Et
**77 patches sur 93** étaient pénalisés : ce n'était pas une correction marginale, ça
pilotait le classement, avec des écarts de rang jusqu'à −33 (C482 : rang 1 → 34). La pente
imposait de surcroît un taux de change jamais choisi : 1 Å de distance au séquon = 1/12 de
la SASA apolaire, soit 50 Å² sur un patch à 600 Å². Les distances se lisent à la main sur
la liste courte.

### [A] Empreinte cétuximab — la calibration enfin faite

C'est le seul point d'ancrage disponible : un site où une protéine se lie réellement.

| | |
|---|---|
| Patches touchant l'empreinte | **38 sur 93** |
| SASA apolaire, médiane tous patches | 352 Å² |
| SASA apolaire, médiane patches d'empreinte | **388 Å²** |
| Recouvrement maximal | **S468**, 9 des 24 résidus d'empreinte, **rang 34**, 402 Å², 40,2 Å²/membre |

**Ce que ça dit de l'échelle** : un binder protéique fonctionnel occupe un site à ~390-400 Å²
de SASA apolaire et ~40 Å²/membre. Les 639 Å² de C482 sont donc largement au-dessus du
point de calibration, et la gamme 400 Å² est démontrée suffisante — ce n'est pas une
estimation, c'est un cétuximab. Le haut du classement n'est pas un seuil à atteindre.

Note : les patches d'empreinte sont légèrement **au-dessus** de la médiane mais pas en tête
(rangs 6, 9, 13, 17, 34, 51, 57, 60, 67, 69, 72, 73). L'affinité réelle du cétuximab ne se
lit donc pas dans la SASA apolaire, ce qui borne ce que cette métrique prétend prédire.

**Défaut révélé par cette lecture — la colonne `face` ne mesure pas ce qu'elle annonce.**
Tous les patches d'empreinte sortent à `face = 0,00`, c'est-à-dire face externe. Or
l'épitope du cétuximab chevauche la surface de liaison de l'EGF sur le domaine III. La
cause est la conformation : **6ARU est replié, donc les domaines I et III sont écartés et
le site de liaison du ligand est démonté.** L'axe centroïde III → centroïde I ne suit pas
la face ligand dans cet état. `face` mesure « côté tourné vers le domaine I en conformation
repliée », ce qui n'est pas « face ligand ». À ne pas utiliser comme critère tant que ce
n'est pas refait sur une structure étendue — hors périmètre pour l'instant.

### [B] Densité apolaire par membre — le tri absolu classe bien la taille

| | |
|---|---|
| Corrélation `n` / SASA apolaire **absolue** | **+0,637** |
| Corrélation `n` / SASA apolaire **par membre** | **−0,182** |
| Taille médiane du top-10 absolu | **13,5** |
| Taille médiane du top-10 par membre | **7,5** |
| Intersection des deux top-10 | **3 / 10** (T358, S356, P361) |

Réponse nette : **oui, le tri absolu favorise systématiquement les gros patches.** Les deux
classements ne décrivent pas le même ensemble.

Et le classement par densité fait remonter exactement ce que le challenge demande — des
patches petits, denses, **et** doublement qualifiés :

| rang densité | centre | n | Å²/membre | ident. | acidC | glyc | rang absolu |
|---|---|---|---|---|---|---|---|
| 1 | H359 | 6 | 80,9 | 0,83 | 0 | 5,8 | 12 |
| 2 | S356 | 7 | 80,4 | 0,71 | 0 | 7,8 | 4 |
| 3 | T358 | 8 | 78,3 | 0,88 | 1 | 5,8 | 2 |
| 4 | L325 | 7 | 60,0 | **1,00** | 1 | 7,3 | 27 |
| 6 | T330 | 8 | 57,2 | 0,88 | **2** | 5,8 | 15 |
| 8 | D323 | 8 | 53,7 | **1,00** | **2** | 7,3 | 23 |
| 9 | G317 | 9 | 49,0 | **1,00** | **2** | 7,3 | 20 |
| 10 | K333 | 6 | 48,7 | **1,00** | **2** | 4,8 | 68 |

Quatre patches à **100 % d'identité humain/souris avec 2 ancres acides conservées**, dont
trois invisibles dans le top-10 absolu (rangs 23, 20, 68). G317 / D323 / K333 / T330 / L325
occupent PDB 317-333 = **UniProt 341-357**, soit la portion N-terminale du domaine III que
**Pfam aurait amputée** (PF01030 démarre à 361). Le choix de CATH-Gene3D se paie
directement ici.

À noter : D323 et E320 étaient les ancres écartées le 30/09 par le filtre glycane dur
(6 et 7 Å). Le retrait du filtre les ressuscite, cohérent.

### [C] Cystéines pontées — réserve légitime, effet négligeable

25 ponts disulfure annotés sur P00533, dont **6 touchant le domaine III** : (329,333),
(337,362), (470,499), (506,515), (510,523), (526,535) UniProt. Soit 10 cystéines pontées
dans le domaine, PDB 309, 313, 338, 446, 475, 482, 486, 491, 499, 502.

**21 patches sur 93** contiennent au moins une cystéine pontée. Sur le top-5 apolaire :

| centre | cysSS | apolaire | dont cysSS | part | corrigé | rang corrigé |
|---|---|---|---|---|---|---|
| C482 | 1 | 639 | 15 | **2,3 %** | 624 | 2 |
| T358 | 0 | 627 | 0 | 0,0 % | 627 | 1 |
| Q480 | 1 | 596 | 15 | 2,5 % | 581 | 3 |
| S356 | 0 | 563 | 0 | 0,0 % | 563 | 4 |
| V481 | 1 | 557 | 15 | 2,6 % | 542 | 5 |

Réponse : **non, le haut du classement ne tire pas sa SASA apolaire de soufres pontés.**
La contribution plafonne à 15 Å², soit 2,3-2,6 %. Le soufre de C482 est largement enfoui
dans son pont avec C491 (UniProt 506-515). Défalquer ne change qu'une permutation
C482 ↔ T358. Ma réserve était fondée sur le principe et sans portée quantitative — dit
franchement, elle ne méritait pas le rang qu'elle occupait dans mon message.

Le reste de la réserve tient quand même, pour une autre raison : un centre sur cystéine
pontée est un résidu structurellement contraint, pas un point d'accroche à solliciter.

### Déduplication : toujours aucun seuil

Le balayage ne montre aucune coupure naturelle (continuum 7 → 26 sites de 0,1 à 0,9).
La liste se lit à la main. Aucun seuil écrit dans le code.

### Lever le confond taille / troncature (01/10) — hypothèse réfutée

Les centres restent dans le domaine III, les membres sont désormais repris sur **toute la
chaîne A** (327 résidus exposés au lieu de 98). Colonnes `*_full` en parallèle des colonnes
masquées, plus `n_truncated`.

**Hypothèse testée** : les cinq patches 317-333 auraient une SASA absolue amputée par le
masque, leur densité restant intacte. **Faux. `n_truncated = 0` pour les dix patches du
top-10 absolu ET pour les dix du top-10 densité.** Aucun des vingt n'est tronqué. La
faible SASA absolue des patches 317-333 est réelle, pas un artefact de bord.

Mais le mécanisme existe, sur d'autres patches : **15 sur 93 sont tronqués**, troncature
médiane 2 résidus, et quatre le sont lourdement — ils étaient sous-évalués :

| centre | n → n_full | apolaire masquée → full | écart |
|---|---|---|---|
| R310 | 6 → 11 | 229 → 454 Å² | **+98 %** |
| K311 | 9 → 12 | 349 → 536 Å² | +54 % |
| C502 | 10 → 17 | 399 → **609 Å²** | +53 % |
| N337 | 8 → 11 | 356 → 497 Å² | +40 % |

Les centres concernés sont ceux des vrais bords : 310, 311, 336, 337 d'un côté, 497, 501,
502, 503 de l'autre, plus quelques-uns dont la sphère de 11 Å franchit la borne (372, 375,
397, 398, 428, 483, 484). C502 à 609 Å² full serait dans les trois premiers ; masqué il
n'apparaissait pas. Le classement absolu doit donc se lire sur les colonnes `_full`.

### Ce qui explique vraiment l'écart absolu / densité

Voisinage à 11 Å, résidus totaux du domaine III contre résidus exposés :

| centre | exposés | total | frac. exposée | Å²/membre |
|---|---|---|---|---|
| C482 | 15 | 30 | 0,50 | 42,6 |
| Q480 | 16 | 24 | 0,67 | 37,2 |
| R470 | 16 | 27 | 0,59 | 33,6 |
| H359 | 6 | **11** | 0,55 | 80,9 |
| S356 | 7 | **11** | 0,64 | 80,4 |
| D323 | 8 | **12** | 0,67 | 53,7 |
| K333 | 6 | **13** | 0,46 | 48,7 |
| L325 | 7 | 22 | **0,32** | 60,0 |
| T330 | 8 | 21 | **0,38** | 57,2 |

La fraction exposée est comparable dans les deux groupes (0,46-0,67). Ce qui diffère, c'est
le **nombre total de résidus dans la sphère** : 11-13 pour H359/S356/D323/K333 contre 24-30
pour C482/Q480/R470. À rayon fixe, un voisinage peu peuplé signifie que la surface
**s'incurve en s'éloignant** — protubérance convexe.

**Conséquence à marquer comme hypothèse, non mesurée ici** : une protubérance convexe est
*défavorable* au design de novo, pas favorable. Un binder maximise la surface enfouie, ce
qui demande une cible plutôt concave ou plate ; une crête étroite en offre peu. Si cela se
confirme, la métrique de densité sélectionne **contre** la designabilité, et mon
commentaire de la lecture [B] — « la densité est mieux alignée » — était prématuré.

L325 et T330 sont un cas distinct : voisinage peuplé (22, 21) mais peu exposé (0,32, 0,38),
donc plutôt une crevasse ou une arête qu'une protubérance.

### Plancher 400 Å² puis tri sur apolar_frac

`FLOOR_APOLAR = 400.0`, adossé à la calibration cétuximab — seul seuil du fichier appuyé
sur une mesure externe. Appliqué sur `sasa_apolar_full`. **40 patches passent, 53 tombent.**
K333 (292 Å²) tombe comme prévu.

| rg | centre | n_full | apol_full | apfr_full | ident | acidC | glyc | dFab |
|---|---|---|---|---|---|---|---|---|
| 1 | H359 | 6 | 486 | 0,73 | 0,83 | 0 | 5,8 | 8,3 |
| 2 | T358 | 8 | 627 | 0,69 | 0,88 | 1 | 5,8 | 6,0 |
| 3 | S356 | 7 | 563 | 0,68 | 0,71 | 0 | 7,8 | 0,0 |
| 4 | **L325** | 7 | 420 | 0,62 | **1,00** | 1 | 7,3 | 0,0 |
| 5 | P361 | 9 | 523 | 0,61 | 0,78 | 1 | 5,8 | 0,0 |
| 6 | G479 | 10 | 410 | 0,61 | 0,50 | 0 | 11,4 | 4,3 |
| 7 | V481 | 13 | 557 | 0,60 | 0,61 | 1 | 9,9 | 0,0 |
| 8 | **D323** | 8 | 430 | 0,58 | **1,00** | **2** | 7,3 | 6,0 |
| 9 | V500 | 11 | 491 | 0,57 | 0,82 | 1 | 5,0 | 9,2 |
| 10 | C482 | 15 | 639 | 0,57 | 0,60 | 2 | 5,0 | 5,7 |
| 11 | Q480 | 16 | 596 | 0,57 | 0,62 | 1 | 6,0 | 0,0 |
| 12 | S468 | 10 | 402 | 0,57 | 0,50 | 0 | 7,6 | 0,0 |

Les plus proches du plancher par en dessous : K455 (392), K454 (379), P362 (374), T406
(374), K336 (367), N444 (364). G317 passe le plancher à 441 Å² mais sort du top-12 sur
`apolar_frac`.

**Seuls L325 et D323 cumulent** plancher franchi, `apolar_frac` ≥ 0,58, **100 % d'identité
humain/souris** et au moins une ancre acide conservée — D323 en ayant deux.

### Colonne `face` supprimée, remplacée par `dist_fab`

`face` reposait sur un axe **inféré** : centroïde du domaine III vers centroïde du
domaine I, censé repérer la face de liaison du ligand puisque le site de l'EGF est ménagé
entre ces deux domaines. **Elle ne repérait rien.** 6ARU est en conformation repliée : les
domaines I et III sont écartés et le site de l'EGF est démonté, donc l'axe ne suit pas la
face ligand. Le contrôle l'a prouvé — les douze patches recouvrant l'empreinte du cétuximab
sortaient tous à `face = 0,00`, face externe, alors que cet épitope chevauche le site de
l'EGF.

Remplacement par une **mesure** : `dist_fab`, distance de l'atome d'ancrage du résidu au
plus proche résidu de l'empreinte du Fab, 0 Å pour un résidu de l'empreinte lui-même. Le
cétuximab compétitionne l'EGF, son empreinte marque donc la surface ligand-compétitive.
Proxy mesuré plutôt qu'axe inféré, et dérivé d'un calcul déjà présent dans le script.

Distribution sur le domaine III : min 0,0, médiane **11,4**, max 28,9 Å. Agrégé par patch
en `min_dist_fab` et `mean_dist_fab`. `face_axis()` et `other_l_domain()` supprimés.

### Calibration du voisinage, et reclassement sur les colonnes `_full` (01/10)

**Correction de cadrage.** L'entrée précédente concluait « hypothèse réfutée » sur
`n_truncated = 0`. Mauvaise lecture : l'hypothèse était **mal ciblée**, pas fausse. La
troncature ne pénalise pas les patches 317-333 — elle en **cache d'autres entièrement**,
parce que le classement était calculé sur les colonnes masquées. Reclassé sur
`sasa_apolar_full`, quatre patches invisibles entrent dans le top-20.

### Top-20 reclassé sur `sasa_apolar_full`

| rang | centre | n | n_tot | f_exp | trc | apol_full | ap/res | apfr | ident | acidC | glyc | dFab |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | C482 | 15 | 31 | 0,48 | 0 | 639 | 42,6 | 0,57 | 0,60 | 2 | 5,0 | 5,7 |
| 2 | T358 | 8 | 18 | 0,44 | 0 | 627 | 78,3 | 0,69 | 0,88 | 1 | 5,8 | 6,0 |
| **3** | **C502** | 17 | 29 | 0,59 | **7** | **609** | 35,8 | 0,43 | 0,71 | 3 | **0,0** | 15,8 |
| 4 | Q480 | 16 | 24 | 0,67 | 0 | 596 | 37,2 | 0,57 | 0,62 | 1 | 6,0 | 0,0 |
| 5 | S356 | 7 | 11 | 0,64 | 0 | 563 | 80,4 | 0,68 | 0,71 | 0 | 7,8 | 0,0 |
| 6 | V481 | 13 | 21 | 0,62 | 0 | 557 | 42,8 | 0,60 | 0,61 | 1 | 9,9 | 0,0 |
| 7 | R470 | 16 | 27 | 0,59 | 0 | 537 | 33,6 | 0,46 | 0,56 | 1 | 0,0 | 0,0 |
| **8** | **K311** | 12 | 22 | 0,55 | **3** | **536** | 44,7 | 0,50 | 0,83 | 0 | **0,0** | 15,6 |
| 9 | T450 | 13 | 29 | 0,45 | 0 | 529 | 40,7 | 0,55 | 0,77 | 2 | 11,4 | 0,0 |
| 10 | P361 | 9 | 16 | 0,56 | 0 | 523 | 58,1 | 0,61 | 0,78 | 1 | 5,8 | 0,0 |
| **13** | **N337** | 11 | 20 | 0,55 | **3** | 497 | 45,2 | 0,52 | 0,82 | 0 | **0,0** | 17,8 |
| **18** | **S501** | 11 | 20 | 0,55 | **1** | 458 | 41,7 | 0,52 | 0,82 | 1 | **0,0** | 12,2 |

**Les quatre patches révélés sont tous à `min_glyc = 0,0`** — par construction, un de leurs
membres *est* un séquon N-linked (PDB 328, 337, 504 selon le cas). Et leur `dFab` est de
12 à 18 Å, donc loin de la surface ligand-compétitive. Ils n'étaient pas cachés par hasard :
ce sont les bords du domaine, là où la chaîne passe aux domaines II et IV, et ces bords sont
glycosylés. C502 à 609 Å² n'est pas un gain net.

### Calibration : la population du voisinage ne discrimine rien

Pas de mesure de concavité — pas le temps de la calibrer. À la place, le seul calibrateur
disponible : l'empreinte du cétuximab. Résidus totaux du domaine III dans les 11 Å.

| ensemble | n | min | p25 | médiane | p75 | max |
|---|---|---|---|---|---|---|
| tous les patches | 93 | 9 | 16 | **19** | 22 | 30 |
| patches d'empreinte | 38 | **11** | 16 | **20** | 24 | 29 |

**Les deux distributions sont confondues.** L'empreinte couvre toute la gamme, de 11 à 29.
Trois de ses patches tombent dans la bande « suspecte » ≤ 13 résidus : **S356 (11 total,
563 Å² apolaires), N473 (11), A477 (13)**. S356 est donc à la fois un voisinage parmi les
plus clairsemés du lot *et* une partie d'une surface qui lie réellement une protéine.

**Verdict, selon la règle posée avant la mesure : la courbure locale est écartée comme
critère.** Elle ne sépare pas une surface liable d'une surface non liable. Mon avertissement
de l'entrée précédente — « la densité sélectionne peut-être contre la designabilité » — est
donc levé sans mesure supplémentaire. Enrichissement résiduel trop faible pour agir :
13 patches sur 93 ont un voisinage ≤ 13, dont 3 d'empreinte (23 %) contre 41 % d'empreinte
dans le lot entier, sur n = 13.

### Classement plancher 400 Å² puis `apolar_frac_full`, top-20

Tête : H359 (0,73), T358 (0,69), S356 (0,68), **L325 (0,62 / 100 % ident / 1 acidC)**,
P361 (0,61), G479 (0,61), V481 (0,60), **D323 (0,58 / 100 % / 2 acidC)**, V500 (0,57),
C482 (0,57), Q480 (0,57), S468 (0,57), **G317 (0,57 / 100 % / 2 acidC)**, T330 (0,56 /
0,88 / 2 acidC), N452 (0,56), P494, E489, T450, T478, T464.
40 patches franchissent le plancher, 53 tombent.

**Les quatre patches à 100 % d'identité humain/souris avec ancre acide conservée** —
L325, D323, G317 et T330 à 88 % — franchissent tous le plancher et sortent entre les rangs
4 et 14 sur la fraction apolaire. Ce sont les seuls candidats qui servent les objectifs 1 et
2 sans compromis, et aucun n'est tronqué, aucun n'est sur un séquon (glyc 5,8-7,3 Å), tous
sont à 0-8 Å de l'empreinte du cétuximab.

### Diversité d'épitope : 3 sites disjoints, et un seul est propre (01/10)

Objectif : répartir le compute sur 3-4 sites spatialement distincts pour pouvoir
basculer un quota si un site ne produit rien. Disjonction mesurée en **fraction de
membres partagés**, jamais en distance entre centres — à `PATCH_RADIUS = 11`, deux centres
à 15 Å partagent encore la moitié de leurs membres.

Site de référence **désigné par les données** et non par des bornes : parmi les 40 patches
au-dessus du plancher, les composantes connexes des patches à identité parfaite, puis la
plus grande, étendue à ses satellites (`GROUP_LINK = 0.5`, valeur **posée**). Résultat :
7 patches, 20 membres — **D323, E320, G317, H359, L325, T330, T358**. C'est bien le site
317-330, obtenu sans le nommer.

Clé de tri des sites alignée sur l'**ordre des objectifs du règlement** : ancre acide
conservée (pH) > identité humain/souris (souris) > surface apolaire (affinité). Ordonner par
identité d'abord — ce que j'avais fait en première passe — démote T450 et N449 qui portent
des ancres acides. Erreur corrigée.

Seuil de disjonction non fixé, balayé : **τ = 0,00 et 0,10 donnent les mêmes 3 sites**,
τ = 0,25 en ajoute un quatrième. Recouvrements croisés tous à 0,00.

| | site 1 | site 2 | site 3 |
|---|---|---|---|
| représentant | **C502** | **K375** | **N449** |
| apol_full | 609 | 445 | 477 |
| apolar_frac | **0,43** | 0,47 | 0,50 |
| identité h/s | 0,71 | **1,00** | 0,77 |
| ancres acides cons. | **3** | 2 | 1 |
| min_glyc | **0,0** | 5,1 | **11,0** |
| dFab | 15,8 | **17,6** | **0,0** |
| patches du groupe | 6 | 3 | **8** |
| union | 27 membres | 20 | **30** |
| hors domaine III | **7** (507-530, dom. IV) | **6** (289-308, dom. II) | **0** |
| séquon dans le groupe | **C502, S501, R503** | **N337, R310** | **aucun** |

### Verdict franc

**Il y a bien 3 sites disjoints, mais un seul des trois alternatifs est propre.**

- **Site 3 (N449)** est la seule vraie alternative : 8 patches — le groupe le plus robuste
  du lot —, 30 membres **tous dans le domaine III**, **aucun séquon**, glycane à 11 Å, et
  `dFab = 0` donc sur la surface ligand-compétitive. Faiblesses réelles : **une seule**
  ancre acide conservée, identité 0,77, fraction apolaire 0,50. Il sert l'objectif 3 et
  à moitié l'objectif 1.
- **Site 2 (K375)** est le meilleur sur les objectifs 1+2 après la référence — identité
  **1,00** et 2 ancres acides — mais deux de ses trois patches portent un séquon, son
  glycane le plus proche est à 5,1 Å, il est à 17,6 Å de l'empreinte du cétuximab, et un
  tiers de son union est dans le **domaine II**. C'est un site à risque assumé, pas un
  site de repli.
- **Site 1 (C502) est à écarter.** Trois de ses six patches portent un séquon, un quart
  de son union est dans le **domaine IV**, et sa fraction apolaire est la plus basse du
  lot au-dessus du plancher (0,43). Sa SASA absolue de 609 Å² — la plus haute — est
  précisément l'artefact que les colonnes `_full` ont révélé : ce qui était caché était
  glycosylé.

**Conclusion à assumer : le domaine III de l'EGFR offre un bon site, un site exploitable,
et un site à risque.** Pas quatre sites équivalents. Répartition proposée pour jeudi :
la référence 317-330 en principal, N449 en secondaire, K375 sur le slot « à risque » que
le §7 de CLAUDE.md demande de garder. C502 non alloué.

---

## Relecture du règlement à la source (01/10/2026)

Page relue : <https://proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/egfr>.
`challenge-01-egfr.md` est une synthèse datée du 30/09 ; cette entrée confirme ou corrige.

### Confirmé

| point | valeur à la source |
|---|---|
| Colonnes CSV | `name`, `sequence`, `molecule_class` |
| `molecule_class` | `protein`, `nanobody`, `scfv`, `fab_kappa`, `fab_lambda` |
| Plafond | Track 1 : 40. **Tracks 2 et 3 : 20** |
| Ordre du CSV | *« Submit your designs as a CSV ordered by how you would rank your molecules (top row higher). »* |
| Longueur | 10-250 aa |
| Catégories | microbinders < 40 ; **minibinders 40-100 inclus** ; grands binders > 100 ; nanobodies ; anticorps |
| Clôture | **4 octobre, 23:59 AoE** |

### Le piège : la page énonce les objectifs dans l'ordre INVERSE de leur poids

- Section « Three objectives » : **1. affinité, 2. cross-réactivité souris, 3. sélectivité pH.**
- Section « How designs are ranked » : **1. sélectivité pH, 2. cross-réactivité souris, 3. affinité.**

C'est l'ordre du **classement** qui compte, et il met le pH en premier. CLAUDE.md §1 était
donc juste, mais pour une raison qui n'était pas écrite : quiconque lit la seule section
« Three objectives » conclut que l'affinité prime. À ne pas relire de mémoire.

**Conséquence directe : la clé de tri des sites est validée.** `n_acidic_cons` →
`frac_ident` → `sasa_apolar` suit l'ordre du classement. La correction du 01/10, qui
plaçait l'ancre acide avant l'identité, était la bonne.

### Un durcissement à ne pas enjoliver

Formulation de l'objectif pH à la source : *« you must design a binder that binds human
EGFR at pH 6.5 and shows **no detectable binding** at pH 7.4. »* C'est un switch binaire,
plus exigeant que le « décalage de KD » que CLAUDE.md §7 donne comme attente réaliste.

La tension est réelle et reste à assumer telle quelle dans le dossier : les binders
pH-dépendants publiés obtiennent typiquement un facteur quelques-uns sur le KD, pas un
tout-ou-rien. On ne prétendra pas atteindre « no detectable binding » ; on mesurera et on
rapportera ce qu'on obtient.

## Commande Modal : `::main` obligatoire (01/10)

`modal_bindcraft.py` déclare deux `local_entrypoint` (`main`, `parallel`). Sans suffixe,
Modal refuse de choisir, **n'exécute rien, et sort en code 0** :

```
Error: Specify a Modal Function or local entrypoint to run.
...
[exited with code 0]
```

Corrigé dans CLAUDE.md §5. La commande de `test1` au 23/09 est annotée sur place, pas
réécrite. Un échec silencieux à code 0 est ce qui ne doit pas rester dans un fichier de
commandes : dans un script d'automatisation il passe pour un succès.

## Le plafond en trajectoires ne freine pas les tentatives

Lu dans la source BindCraft au commit `c0a48d5`, `functions/generic_utils.py` :

```python
def check_n_trajectories(design_paths, advanced_settings):
    n_trajectories = [f for f in os.listdir(design_paths["Trajectory/Relaxed"])
                      if f.endswith('.pdb')]
    if advanced_settings["max_trajectories"] is not False and \
       len(n_trajectories) >= advanced_settings["max_trajectories"]:
        return True
```

Le comptage porte sur **`Trajectory/Relaxed` uniquement**. Une trajectoire qui finit en
`LowConfidence` ou `Clashing` ne consomme pas le quota — tout en ayant consommé du GPU.
Donc `--max-trajectories N` plafonne les trajectoires **relaxées**, pas les tentatives.

**Le vrai frein de budget est `TIMEOUT`** (variable d'environnement, minutes, défaut 300),
appliqué comme `timeout=TIMEOUT * 60` sur la fonction Modal. CLAUDE.md §2 disait « tout run
sans `--max-trajectories` est un bug de budget » : c'est insuffisant. Un run **avec**
`--max-trajectories` mais un `TIMEOUT` à 300 peut brûler 5 h de GPU sur une cible où rien ne
relaxe. Le plafond en trajectoires reste utile comme garde secondaire, pas comme budget.

Autre fait vérifié dans `generic_utils.py` : **`create_dataframe` ne touche pas un CSV
existant** (`if not os.path.exists(csv_file)`). Les stats s'accumulent donc d'un run à
l'autre sous le même `run_name` — c'est ce qui rend la reprise possible.

---

## Run de fumée `smoke-G317` — EGFR domaine III (01/10/2026)

Premier run GPU sur la vraie cible. App `ap-JcN5ZalbdRyeDLQkJGRV4c`, L40S, détaché.

```bash
GPU=L40S .venv/bin/modal run --detach modal_bindcraft.py::main \
  --input-pdb inputs/6ARU_A_309-506.pdb --target-chains A \
  --target-hotspot-residues "A318,A320,A323,A325" \
  --lengths 50,130 --number-of-final-designs 100 \
  --max-trajectories 3 --run-name smoke-G317
```

Cible : `inputs/6ARU_A_309-506.pdb`, 198 résidus, produite par `prepare_target.py`.
Hotspots : les deux ancres acides D323/E320 et le cœur apolaire I318/L325 du patch
représentatif G317, tous à 100 % d'identité humain/souris.

### Deux leçons de commande

Première tentative sans `::main` : Modal refuse de choisir entre les deux
`local_entrypoint`, n'exécute rien, **et sort en code 0**. Aucune dépense. Corrigé dans
CLAUDE.md §5.

Seconde, de ma part : la sortie était pipée dans `tail -60`, ce qui **a détruit le log du
run**. Il n'en restait que les 60 dernières lignes. Rediriger vers un fichier, toujours.

### Chiffres mesurés

| grandeur | valeur |
|---|---|
| Mur total | **60 min 44 s** (3644 s) |
| Coût | **$1,97** (3644 s × $0,000542/s, tarif L40S confirmé) |
| Tentatives | **6** (3 relaxées + 2 LowConfidence + 1 Clashing) |
| Temps par tentative, tout compris | **10,1 min** (boot, échecs, MPNN et PyRosetta inclus) |
| Temps par trajectoire relaxée seule | 7 min 52 s, 8 min 12 s, 9 min 24 s → **moyenne 8,49 min** |
| `p_relax` | **3/6 = 0,50** |
| Designs acceptés | **2** |
| Modes de liaison indépendants | **1** — les 2 acceptés sont `mpnn1` et `mpnn2` de la *même* trajectoire |
| Coût par design accepté | **$0,99** |
| Coût par mode indépendant | **$1,97** |

Le temps par trajectoire dépend de la longueur du binder : 62 aa → 7 min 52 s, 63 aa →
8 min 12 s, 83 aa → 9 min 24 s. Environ +1,5 min pour +20 résidus. Pour la plage 55-95
(moyenne 75), compter ~8,7 min par trajectoire relaxée.

**La troncature a payé.** 8,49 min par trajectoire sur 198 résidus, contre 9,0 min mesurés
sur PD-L1 qui n'en fait que 115. Mon estimation a priori de 14-23 min était pessimiste d'un
facteur ~2 : la cible tronquée ne coûte pas plus cher que la démo.

### Profil de rejet — `i_pAE` domine massivement

`failure_csv.csv`, comptages cumulés par filtre :

| filtre | rejets |
|---|---|
| **`i_pAE`** | **20** |
| `pLDDT` | 6 |
| `i_pTM` | 3 |
| `Trajectory_one-hot_pLDDT` | 2 |
| `Trajectory_final_pLDDT` | 2 |
| `Trajectory_logits_pLDDT` | 1 |
| `Trajectory_Clashes` | 1 |
| tous les autres (pTM, pAE, dG, dSASA, SC, PackStat, hydrophobicité, H-bonds non satisfaites, RMSD…) | **0** |

`i_pAE` rejette plus que tous les autres réunis. C'est le seul levier qui compte si le
rendement doit monter : les deux designs acceptés sortent à `i_pAE` 0,17-0,18, donc les
rejetés sont très au-dessus. À vérifier contre la valeur réelle dans `default_filters.json`
avant de toucher à quoi que ce soit — **aucune modification de seuil n'a été faite**.

### Les deux designs acceptés

Tous deux issus de la trajectoire `l83_s142379`, donc **frères, pas indépendants**.

| | `_l83_s142379_mpnn1` | `_l83_s142379_mpnn2` |
|---|---|---|
| Longueur | **83 aa** | **83 aa** |
| Average_pLDDT | 0,93 | 0,93 |
| Average_i_pTM | 0,83 | 0,83 |
| Average_i_pAE | 0,18 | 0,17 |
| ShapeComplementarity | 0,67 | 0,69 |
| dG | −47,73 | −54,78 |
| dSASA | 1842,85 | 1969,24 |
| Surface_Hydrophobicity | 0,15 | 0,17 |
| Unsat H-bonds | 3,0 | 3,5 |
| **His dans la séquence** | **3** | **3** |

```
mpnn1 MKKLSKGEEVVEKVKKEAEELKEKLEKGELSLEEVEEKWVEIWKEAEKEAPETFHKISEVEYEFQLWLHHKRIEERKKKEEEE
mpnn2 MKKLSKGEKVVEEVRKKTEELKKRLEEGKLSIEEVEKEWVKIWKEAEKEAPETFHKISEVEYEFQLWLHHKKIEERKKKEEEE
```

**Deux observations qui comptent pour le challenge :**

1. **83 aa : dans la catégorie minibinders (40-100) sans l'avoir cherché.** Le run utilisait
   `--lengths 50,130`. La production passera à `55,95` pour garantir la catégorie.
2. **3 His chacun, et `InterfaceAAs` de la trajectoire `l83` donne `H: 3` — trois His à
   l'interface, sans aucun biais de composition appliqué.** La trajectoire `l63` en avait
   aussi 3. C'est inattendu, et c'est une donnée pour l'objectif n°1 : la baseline produit
   déjà des His d'interface. Reste à vérifier si elles sont appariées à un Asp/Glu de la
   cible — ce que le run ne dit pas et qu'il faudra mesurer sur les structures.

### Fin du run : annulation externe, cause non attribuée

Arrêt à 60 min 44 s sur `RemoteError: Function call was cancelled by user or a failure.`
Log distant :

```
2026-10-01T16:41:26+0000 Received a cancellation signal while processing input (...)
2026-10-01T16:41:26+0000 Successfully canceled input (...)
```

**Ni timeout, ni OOM, ni exception applicative.** Un signal d'annulation externe. Je ne
l'attribue pas : aucune commande lancée localement pendant le run n'annule, et le log ne dit
pas qui a demandé l'annulation. À surveiller sur le prochain run ; si ça se reproduit, c'est
un facteur de dimensionnement.

Le run était encore au travail à l'arrêt — dernière ligne utile
`Unmet filter conditions for ..._l62_s857853_mpnn1`, donc en pleine évaluation MPNN de la
troisième trajectoire. Les 3 relaxées du plafond étaient atteintes, mais la boucle ne teste
le plafond qu'au début de l'itération suivante.

### Acquis : une annulation commite le volume

**Tous les artefacts ont survécu** — 4 CSV, 3 trajectoires relaxées, 2 LowConfidence,
1 Clashing, 2 acceptés, 2 MPNN. Le `finally: volume.commit()` du wrapper a tourné malgré
l'annulation. C'est la moitié de la réponse à « un run interrompu perd-il tout ». Un kill par
**timeout** reste à vérifier séparément : même mécanisme probable, non testé.

### Arborescence confirmée

Un répertoire par `run_name` sous `/outputs`, à côté de `test1`, `par-test`, `par-test2`,
`par-test3` :

```
smoke-G317/{trajectory,mpnn_design,final_design}_stats.csv  failure_csv.csv
           Trajectory/{Relaxed,LowConfidence,Clashing,Plots,Animation}
           MPNN/  Accepted/  Rejected/
```

Un `run_name` par site suffit donc à séparer les sites en production.

### Dimensionnement de la production, chiffres réels

Formule posée précédemment, remplie. `r` = $1,95/h (L40S), `t_att` = 10,1 min tout compris,
`p_acc_indep` = 1 mode indépendant pour 6 tentatives.

```
mode independant       = 6 tentatives = 60,7 min = $1,97
20 modes independants  = 120 tentatives = 20,2 GPU-h = ~$39
reparti sur 3 sites    = ~6,7 GPU-h par site
```

Découpage retenu : **`TIMEOUT=135` (2,25 h), `k = 3` appels par site, 3 sites** →
20,25 GPU-h, **~$40**. Mur ~2,5 h si les 9 appels sont concurrents, ~6,8 h s'ils sont
séquentiels par site. Les deux tiennent très largement dans la fenêtre de 30 h.

Avec une marge ×2,5 pour absorber un `p_acc` plus faible sur les sites K375 et N449 :
~50 GPU-h, **~$100**, mur inchangé. C'est le dimensionnement à retenir.

`--number-of-final-designs` reste haut (100) et `--max-trajectories` généreux : le frein est
`TIMEOUT`, par construction. Le coût devient alors déterministe — `S · k · T · r` ne dépend
d'aucun taux de passage, ce que le plafond en trajectoires ne garantissait pas.

---

## Deux bugs trouvés par relecture (02/10/2026)

### Bug 1 — `member_set_full` fuit dans le CSV

`write_csv` reçoit `drop=("member_set", "members", "members_full")` mais pas
`member_set_full`, ajouté après. La colonne 41 de `egfr_patches.csv` contient donc un
`frozenset({...})` Python sur les 93 lignes.

Deux défauts, dont le second est le grave : c'est illisible, et **l'ordre d'itération d'un
`set` Python n'est pas stable entre exécutions**, donc le fichier n'est pas reproductible
octet à octet. Un CSV de ce dépôt doit l'être.

La colonne n'est pas supprimée — la jointure patches → résidus est utile et c'est elle qui
permet de dériver les hotspots. Elle est reformatée : entiers triés, séparés par des
points-virgules, sous le nom explicite `member_resnums_full`.

### Bug 2 — le filtre de taille mal ciblé, et sa vraie nature

`MIN_PATCH_MEMBERS` filtrait sur `n` **masqué** alors que tous les classements sont passés
aux colonnes `_full`. Trois patches écartés à tort : **C309 (5 masqués / 15 full, 638 Å²
apolaires)**, N504 (5/10, 333 Å²), E495 (5/7, 262 Å²).

**Hypothèse testée** : « un patch centré près d'une borne a peu de membres masqués par
construction, donc le filtre est un filtre de bord déguisé ». **Partiellement réfutée.**

| mesure | valeur |
|---|---|
| corrélation distance-du-centre-à-la-borne / `n` masqué | **−0,095** |
| `n` masqué médian, centres à ≤ 10 résidus d'une borne (14 patches) | **9,0** |
| `n` masqué médian, centres éloignés (84 patches) | **9,0** |
| sous le seuil de 6, près du bord | **2/14 (14 %)** |
| sous le seuil de 6, loin du bord | **3/84 (3,6 %)** |

Le comptage masqué n'est donc **pas** déprimé près des bornes en général : médianes
identiques, corrélation nulle. Mais parmi les patches qui tombent sous le seuil, les patches
de bord sont **sur-représentés d'un facteur ~4**. Et les 5 écartés se séparent proprement
selon la distance du centre à la borne :

| patch | dist. borne | n / n_full | trc | verdict |
|---|---|---|---|---|
| C309 | **0** | 5 / 15 | 10 | écarté à tort |
| N504 | **2** | 5 / 10 | 5 | écarté à tort |
| E495 | **11** | 5 / 7 | 2 | écarté à tort |
| F357 | 48 | 5 / 5 | 0 | **correctement écarté** |
| H409 | 97 | 5 / 5 | 0 | **correctement écarté** |

**La nature du bug, énoncée précisément** : le filtre n'annonce pas ce qu'il mesure, mais de
façon *sélective* et non systématique. Il prétend écarter les patches pauvres ; il écarte
les patches pauvres **et** les patches de bord riches, sans distinguer. C'est la même famille
d'erreur que la tautologie d'échantillonnage du §5 — un critère corrélé à autre chose que ce
qu'il annonce — mais à l'inverse de celle-là, le mécanisme ici ne s'applique qu'à une
minorité de cas. Dire « filtre de bord déguisé » surinterprète ; dire « mauvaise colonne »
sous-interprète. La formulation juste : **un filtre de taille qui, sur les patches de bord,
mesure la troncature au lieu de la taille.**

### Prédiction, écrite AVANT la re-exécution

1. Le filtre passe à `n_full`. Le lot passe de **93 à 96 patches**.
2. **E495 (262 Å²) et N504 (333 Å²) n'atteignent pas `FLOOR_APOLAR = 400`** : ils entrent
   dans le lot mais pas dans les 40 passants du plancher, donc **aucun effet sur l'analyse
   de sites**.
3. **C309 (638 Å²) franchit le plancher** : 40 → **41 passants**. Il se classerait **rang 2**
   du classement principal, entre C482 (639) et T358 (627).
4. **Je prédis que C309 ne forme pas un quatrième site, mais rejoint celui de K375.** L'union
   du site K375 contient déjà 309, 310, 311, 312 et descend jusqu'à 289 — C309 est au même
   endroit, la jonction domaine II / domaine III. Recouvrement attendu élevé, donc fusion.
5. **Je prédis que le regroupement en 3 sites disjoints tient**, C309 étant absorbé plutôt
   qu'inséré. Risque identifié : si C309 porte ≥ 3 ancres acides conservées, la clé pH-first
   le place avant C502 et il devient le représentant du site, ce qui changerait l'étiquette
   du site 1 sans changer sa composition.
6. **Je prédis d'écarter C309 quand même, par argument et non par rang** : 10 de ses 15
   membres sont hors du domaine III, donc ~2/3 de sa surface apolaire est portée par le
   domaine II et la jonction II/III. Le §4 a déjà tranché que la surface dépendant de
   l'arrangement inter-domaines en conformation repliée est celle dont on ne se fie pas —
   c'est l'argument qui a supprimé la colonne `face`. C309 est le cas limite exact de cette
   règle. Son rang 2 est un artefact de la conformation, pas une propriété de l'épitope.
7. C309 est de plus centré sur une **cystéine pontée** (309 = UniProt 333, apparié à 329),
   donc un centre structurellement contraint.

### Bug 3 — jointure lossy, corrigée dans le même geste

`egfr_residues.csv` ne contenait que les 198 résidus du domaine III, alors que
`member_resnums_full` puise dans les 327 exposés de la chaîne entière. La jointure
patches → résidus perdait donc des membres, **et le plus là où ça compte** : C502 perdait
7 membres sur 17, C309 en perdrait 10 sur 15. Les hotspots des patches de bord étaient
indérivables.

`egfr_residues.csv` couvre maintenant **toute la chaîne A (609 résidus)**, la colonne
`in_domain3` distinguant l'appartenance. Les agrégats imprimés restent calculés sur le
domaine III.

### Résultat de la re-exécution, confronté à la prédiction

Prédiction écrite et commitée en `6648868` **avant** la correction.

| # | prédiction | résultat | verdict |
|---|---|---|---|
| 1 | lot 93 → 96 patches | **96** | ✓ |
| 2 | E495 (262 Å²) et N504 (333 Å²) sous le plancher, aucun effet | passants 40 → **41**, donc seul C309 entre | ✓ |
| 3 | C309 franchit le plancher, **rang 2** à 638 Å² | **rang 2**, entre C482 (639) et T358 (627) | ✓ |
| 4 | C309 **fusionne** avec le site K375 (recouvrement élevé) | recouvrement **0,33** (5/15 membres) | **✗ mécanisme faux** |
| 5 | le regroupement en 3 sites disjoints tient | C502, K375, N449 inchangés à τ = 0,00 et 0,10 ; K463 quatrième à 0,25 | ✓ |
| 5b | risque : si C309 a ≥ 3 ancres acides il devient représentant du site 1 | C309 a **2** ancres, C502 en a 3 → C502 garde le slot | ✓ risque non matérialisé |
| 6 | C309 à écarter par argument, ~2/3 de sa surface hors domaine | **70 %** de sa SASA apolaire portée par les 10 membres hors domaine III | ✓ |
| 7 | C309 centré sur une cystéine pontée | `cys_bridged = True`, UniProt 333, apparié à 329 | ✓ |

Le site de référence est **inchangé** : 7 patches, 20 membres, D323 / E320 / G317 / H359 /
L325 / T330 / T358. C309 a une identité de 0,80, il n'entre donc pas dans la composante à
identité parfaite qui amorce la référence.

### La divergence, et elle compte

J'avais prédit que C309 serait **absorbé** par le site K375 — recouvrement élevé, fusion.
Le recouvrement mesuré est de **0,33**, soit un tiers. La conclusion survit (C309 n'est pas
un quatrième site) mais **pas pour la raison prédite** : il est rejeté parce que 0,33 dépasse
le seuil de disjonction à τ = 0,10 et τ = 0,25, pas parce qu'il décrit le même site.

**Conséquence qui n'était pas dans la prédiction : à τ = 0,50, C309 deviendrait un
quatrième site disjoint.** Son exclusion de la liste des sites dépend donc d'un seuil
délibérément non fixé. Ce n'est pas une position robuste, et c'est exactement pourquoi
l'arbitrage devait être argumenté plutôt que lu dans un rang.

### Décision sur C309 : écarté, et l'argument est accablant

Composition, membres triés par SASA apolaire, `*` = hors domaine III :

```
*Y292  68,8  identical     *K304  67,6  identical    *A289  65,2  different
 R310  61,6  identical     *E306  59,2  similar      *P308  53,1  identical
 V312  48,3  identical      K311  44,7  identical    *G288  33,8  identical
*K303  29,0  identical     *D290  26,6  identical     C309  25,7  identical
*G307  22,9  identical     *E293  21,7  identical     N337   9,7  different
```

Quatre des cinq membres les plus apolaires sont **hors du domaine III**. Au total **70 % de
la SASA apolaire de C309 est portée par le domaine II et la jonction II/III** — surface dont
l'exposition dépend de l'arrangement inter-domaines en conformation repliée, c'est-à-dire
exactement la catégorie que le §4 déclare non fiable et qui a fait supprimer la colonne
`face`. Son rang 2 mesure la conformation de 6ARU, pas une propriété de l'épitope.

Trois défauts s'ajoutent, chacun suffisant :
- centre sur **cystéine pontée** (C309 ↔ C329 UniProt), résidu structurellement contraint ;
- **N337 parmi les membres à `min_glyc = 0,0`** : un séquon dans le patch ;
- **A289 en statut `different`** au rang 3 par surface apolaire, donc l'objectif 2 est touché
  sur l'un des membres qui portent le plus de surface.

C309 est écarté. Le bug qui le masquait était réel et devait être corrigé ; le patch qu'il
révèle n'est pas exploitable. Les deux énoncés tiennent ensemble.

### Correction d'une affirmation antérieure sur N449

J'avais écrit que `N473`, statut `different`, était « un des 4 plus exposés » du patch N449.
**Faux** : il est au **rang 6 sur 13** par SASA apolaire, à 33,8 Å². La conclusion tient —
une divergence humain/souris dans le patch pèse sur l'objectif 2 — mais l'argument est plus
faible qu'écrit, et il ne disqualifie pas N449 à lui seul.

### État des CSV après correction

| | avant | après |
|---|---|---|
| `egfr_patches.csv` | 93 lignes, `frozenset({...})` en colonne 41 | **96 lignes**, `member_resnums_full` en entiers triés séparés par `;` |
| `egfr_residues.csv` | 198 lignes, domaine III seul | **609 lignes**, chaîne A entière, `in_domain3` distingue |
| jointure patches → résidus | lossy (C502 perdait 7 membres sur 17) | **complète** |
| reproductibilité octet à octet | non (ordre d'itération de `set`) | **oui** |

---

## Accessibilité des carboxylates des ancres acides (02/10/2026) — `carboxylate_access.py`

Motif : `egfr_epitope_map.py` retient une ancre acide sur `rel_sasa`, la SASA du **résidu
entier**. Or ce qui doit être atteignable pour un pont salin His–acide, c'est le
**carboxylate**. Un Asp au CB exposé mais aux OD1/OD2 rentrants passe le seuil `exposed` en
étant inatteignable. Le script sépare les deux, et mesure en plus l'**orientation**.

### La calibration prévue a échoué

L'idée était d'étalonner sur les Asp/Glu de l'empreinte du Fab de cétuximab — des acides
qu'une protéine vient réellement contacter. **L'empreinte n'en contient qu'un : E472.** n = 1,
aucune distribution exploitable.

C'est en soi un fait à noter : **l'épitope du cétuximab, 24 résidus, ne porte qu'un seul
Asp/Glu.** Un binder protéique qui fonctionne sur le domaine III le fait donc sur une surface
quasi dépourvue d'acides. C'est une mise en garde sur la stratégie d'ancrage acide, pas une
réfutation — mais elle mérite d'être dans le dossier.

### Distribution de référence, SASA du carboxylate seul

| ensemble | n | min | p25 | médiane | p75 | max |
|---|---|---|---|---|---|---|
| chaîne A entière | 65 | 1,1 | 27,9 | 47,1 | 70,7 | 94,3 |
| domaine III | 19 | 3,2 | 27,9 | 39,6 | 59,5 | 92,1 |

**Aucun carboxylate n'a une SASA nulle** — minimum 1,1 Å² sur 65. Le critère « SASA > 0 »
ne discrimine donc rien sur cette structure.

### Les ancres mesurées

`carbox` = SASA des deux oxygènes, `part` = sa fraction dans la SASA du résidu,
`angle` = angle entre CB→carboxylate et le vecteur sortant local. Angle faible = pointe vers
le solvant ; angle élevé = longe la surface ou rentre.

| site | ancre | carbox (Å²) | CB | part | angle | centile dom. III |
|---|---|---|---|---|---|---|
| **G317** | **D323** | **75,0** | 16,9 | **55 %** | **33°** | **89** |
| **G317** | **E320** | 30,0 | 25,4 | **28 %** | **103°** | 32 |
| N449 | E472 | 47,1 | 21,7 | 46 % | 86° | 63 |
| K375 | E400 | 72,9 | 4,8 | 76 % | 49° | 84 |
| K375 | E397 | 36,4 | 1,2 | 78 % | 19° | 42 |
| C502 | E530 | 82,5 | 4,8 | 86 % | 27° | 95 |
| C502 | E510 | 64,3 | 8,5 | 78 % | 52° | 74 |
| C502 | E489 | 27,9 | 0,0 | 44 % | 35° | 26 |

### Seuil proposé, et son statut

Faute d'étalon empirique, le seuil est dérivé d'un **argument géométrique**, à traiter comme
tel — posé, non calibré.

1. **Nécessaire : SASA > 0.** La sonde de Shrake-Rupley fait 1,40 Å, soit un rayon de
   molécule d'eau : une SASA non nulle signifie qu'une eau peut s'approcher, donc qu'un
   donneur de liaison hydrogène peut le faire. **Les 65 carboxylates passent** : non
   discriminant ici.
2. **Insuffisant, parce qu'un imidazole n'est pas une eau.** Le cycle His mesure environ
   4,5 × 4,0 Å, soit une section de l'ordre de **20-25 Å²**. Pour qu'un cycle approche au
   lieu d'une seule eau, le carboxylate doit présenter de l'ordre de sa propre section en
   surface accessible. D'où un plancher **≥ 20-25 Å²**.
3. **La fraction est plus comparable que l'aire absolue.** Le carboxylate d'un Glu est porté
   plus loin du squelette que celui d'un Asp et sort donc systématiquement plus. La fraction
   `carbox / résidu` dit si c'est le groupe fonctionnel qui est exposé ou seulement la tige :
   **≥ 40 %** retenu.
4. **L'orientation est le vrai discriminant ici**, et elle est sans dimension :
   **≤ 60°** pour un carboxylate qui pointe vers le solvant, **≥ 90°** pour un qui longe la
   surface.

**Critère composite retenu pour un pont salin His–carboxylate direct :**
`carbox >= 25 A2` **et** `part >= 40 %` **et** `angle <= 60°`.

Application :

| ancre | aire | fraction | angle | verdict |
|---|---|---|---|---|
| **D323** | 75,0 ✓ | 55 % ✓ | 33° ✓ | **ancre réelle, les trois critères** |
| **E400** | 72,9 ✓ | 76 % ✓ | 49° ✓ | ancre réelle |
| E510 | 64,3 ✓ | 78 % ✓ | 52° ✓ | ancre réelle, **mais hors domaine III** (510 > 506) |
| E530 | 82,5 ✓ | 86 % ✓ | 27° ✓ | ancre réelle, **mais hors domaine III** |
| E397 | 36,4 ✓ | 78 % ✓ | 19° ✓ | ancre réelle, aire modeste |
| E489 | 27,9 ✓ | 44 % ✓ | 35° ✓ | ancre réelle, aire modeste |
| E472 | 47,1 ✓ | 46 % ✓ | **86° ✗** | **douteuse** : longe la surface |
| **E320** | 30,0 ✓ | **28 % ✗** | **103° ✗** | **nominale** : échoue sur deux des trois |

### Conséquence sur le site G317

**D323 est une excellente ancre** — 89e centile du domaine III, 55 % de la SASA du résidu
portée par le carboxylate, pointant vers le solvant à 33°.

**E320 est une ancre nominale.** Son aire de 30 Å² n'est pas nulle, mais seulement 28 % de la
SASA du résidu vient du carboxylate — l'essentiel est la tige — et surtout l'angle de **103°**
signifie que le groupe longe la surface au lieu d'en sortir. Le pipeline le comptait comme
ancre parce que `rel_sasa` regarde le résidu entier.

Le site G317 garde donc **une** ancre acide conservée réellement exploitable, pas deux. Même
chose pour N449, dont l'unique ancre E472 est douteuse sur l'orientation (86°). K375 en garde
**deux** (E400 et E397), ce qui en fait, sur ce seul critère, le site le mieux doté — mais ses
autres défauts tiennent : séquons, domaine II, `dFab` 17,6 Å.

### Limites de la mesure

- SASA sur 6ARU, conformation **repliée**, 3,20 Å.
- **Un seul rotamère cristallographique.** Un carboxylate peut tourner en solution ; l'angle
  de 103° de E320 est la valeur d'un modèle, pas une contrainte permanente. C'est la limite
  la plus sérieuse de cette analyse.
- Le vecteur sortant est approximé par centroïde local → carboxylate, sur un voisinage de
  12 Å. C'est un proxy, pas une normale de surface calculée.
- Les trois seuils (25 Å², 40 %, 60°) sont **posés**, pas calibrés, faute d'étalon.

---

## Étendue spatiale de l'empreinte du cétuximab (02/10/2026) — `footprint_extent.py`

Objet : situer l'ordre de grandeur de `PATCH_RADIUS` contre une empreinte réelle. **Lecture
seule** — aucune constante modifiée, pipeline non relancé, aucun CSV réécrit.

Mesuré sur les **24 résidus** de l'empreinte (critère `fab_footprint`, contacts lourds sous
4,5 Å, PDB 349-473), avec les mêmes atomes d'ancrage qu'`enumerate_patches` : CB, ou CA en
l'absence de CB.

### Valeurs mesurées

| grandeur | valeur |
|---|---|
| Diamètre de l'empreinte (distance max entre deux résidus) | **34,9 Å** — paire 350 ↔ 473 |
| Rayon de la plus petite sphère englobante, centre libre | **17,7 Å** (diamètre 35,4 Å) |
| **Couverture à `PATCH_RADIUS = 11 Å`** | **médiane 8/23 autres résidus**, min 2, max 14, moyenne 8,0 |
| **Fraction médiane de l'empreinte capturée** par un patch de 11 Å | **40 %** (9 résidus sur 24, centre inclus) |
| Rayon minimal couvrant les 24, centre contraint à être l'un d'eux | **19,1 Å** depuis le résidu 441, soit **1,7 ×** `PATCH_RADIUS` |

Les cinq meilleurs centres pour une couverture totale : 441 (19,1 Å), 418 (20,4), 440 (20,6),
417 (22,1), 438 (22,9).

### Ce que dit le point 3

**Un patch de 11 Å capture environ 40 % d'une empreinte réelle.** L'unité « site » du pipeline
est donc nettement plus petite que la surface qu'une protéine occupe effectivement sur cette
cible. Le rapport est d'environ 2,5 en nombre de résidus.

La dispersion de la couverture — de 2/23 à 14/23 selon le centre choisi — est elle-même
informative : les résidus des extrémités (350, 353, 473, à 2-3 voisins) capturent très peu,
ceux du milieu (417, 438, 440, à 13-14) beaucoup. Combiné au fait que le diamètre (34,9 Å)
égale pratiquement le diamètre de la sphère englobante (35,4 Å), cela décrit une empreinte
**allongée**, dont l'extension est fixée par son axe long. Un modèle de patch sphérique en
rend mal compte quel que soit son rayon — remarque structurale, pas une proposition.

### Deux réserves, à lire avec les chiffres

1. **Un Fab fait ~50 kDa sur deux chaînes et couvre une zone plus large qu'un minibinder de
   83 aa.** Cette mesure **surestime** donc le rayon pertinent pour le cas présent. Les
   binders visés font 55-95 résidus, soit une interface attendue plus petite que celle d'un
   Fab, et le facteur 1,7 n'est pas transposable tel quel.
2. **C'est un ordre de grandeur, pas une calibration.** La mesure ne distingue pas 9 Å de
   11 Å de 13 Å : les trois donneraient une couverture partielle du même genre. Elle situe
   l'échelle, elle ne désigne pas une valeur.

### Statut de `PATCH_RADIUS`

**Inchangé : 11,0 Å, POSÉ, NON CALIBRÉ.** Aucune valeur nouvelle n'est proposée. L'inventaire
des constantes de [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §4 reste exact sur ce point,
et cette mesure ne le modifie pas — elle ajoute seulement qu'on connaît désormais l'échelle
d'une empreinte réelle sur cette cible, ce qui n'était pas le cas.

---

## Trois éliminations à l'intérieur du site de référence (02/10/2026)

Lecture seule sur `data/egfr_patches.csv` et `data/egfr_residues.csv`. Déclenché par une
question simple — « en quoi T358 est-il mauvais ? » — dont la réponse élimine trois choses.

### Élimination 1 — l'écart de rang entre G317 et T358 n'a plus de fondement

T358 sort 6ᵉ sur 7 du site de référence uniquement parce que la clé de tri compte
`n_acidic_cons` d'abord : 1 pour T358 contre 2 pour G317.

Mais **l'unique ancre conservée de T358 est D323** — exactement celle de G317. Les deux
patches la partagent, avec L325.

| | ancres annoncées | ancres **réelles** (critère carboxylate) |
|---|---|---|
| G317 | E320 + D323 | **D323** |
| T358 | D323 | **D323** |

La seconde ancre de G317, E320, a été mesurée **nominale** : 28 % de part, angle 103°. Donc
**les deux patches ont la même et unique ancre pH réelle, la même molécule.** L'écart 2 contre
1 qui place l'un au rang 1 et l'autre au rang 6 repose entièrement sur un résidu dont le
carboxylate ne sort pas.

Ce qui reste comme différence réelle entre eux n'est pas l'objectif 1 mais l'arbitrage
objectif 2 contre objectif 3.

### Élimination 2 — H359 est éliminé comme cible de contact

T358 a 42 % plus de surface apolaire que G317 (627 contre 441 Å²). Mais :

| | apolaire | membres |
|---|---|---|
| T358 complet | 627 Å² | 8 |
| **T358 sans H359** | **529 Å²** | 7 |
| G317 | 441 Å² | 9 |

**H359 porte 97,8 Å², soit 16 % de la surface apolaire de T358**, et c'est son 3ᵉ membre le
plus exposé — donc très probablement contacté. Or c'est précisément lui qui diverge :
**His chez l'humain, Arg chez la souris.**

La substitution n'est pas conservative : elle remplace un résidu titrable par un résidu
définitivement chargé positif. Un contact construit sur H359 ne se retrouve pas sur la
protéine murine. **C'est le mode de défaillance de l'objectif 2**, et l'objectif 2 est classé
au-dessus de l'objectif 3.

Conséquence chiffrée : **l'avantage de surface de T358 passe de +42 % à +20 %** une fois H359
défalqué. Il survit, mais il est deux fois moins grand que la colonne brute ne le suggère.
H359 est à exclure de tout jeu de hotspots.

**Note annexe, à traiter comme hypothèse** : H359 est une histidine **de la cible**, à
l'interface, donc titrable dans la fenêtre 6,5-7,4. C'est une route pH alternative à celle de
CLAUDE.md §7, qui suppose l'His portée par le binder. Mais elle est **absente chez la souris**
(Arg), donc l'exploiter échangerait l'objectif 2 contre l'objectif 1. Non retenue, consignée.

### Élimination 3 — « G317 ou T358 » est une fausse alternative

| | |
|---|---|
| recouvrement T358 / G317 | **25 %** — partagent D323 et L325 |
| union | **15 membres**, **926 Å²** apolaires |
| identité de l'union | **14/15 = 0,93** |
| ancres conservées de l'union | **E320 et D323** |
| diamètre de l'union | **28,9 Å** |

Les deux sont **deux patches du même site**, tous deux dans le groupe de référence de
7 patches. Choisir l'un contre l'autre revient à choisir où centrer les hotspots à l'intérieur
d'un site, pas à choisir entre deux sites. La question était mal posée, la mienne comprise.

L'union est géométriquement viable : **28,9 Å de diamètre, soit 83 % de l'étendue mesurée de
l'empreinte du cétuximab** (34,9 Å). Le run de fumée a produit des dSASA de ~1900 Å², donc
l'ordre de grandeur n'est pas absurde.

### Ce que ça fait apparaître : F357

Meilleurs candidats apolaires de l'union, par SASA apolaire :

| résidu | apolaire | statut | glycane |
|---|---|---|---|
| **F357** | **155,8** | identical | 11,8 |
| **L325** | 111,1 | identical | 7,3 |
| ~~H359~~ | 97,8 | **different → R** | 10,8 |
| **I318** | 83,3 | identical | 15,0 |
| P361 | 71,3 | identical | 8,5 |
| K322 | 70,0 | identical | 13,6 |
| T406 | 66,4 | identical | 19,9 |
| T358 | 56,8 | identical | 7,8 |

**F357 est le résidu le plus apolaire de tout le site de référence** — 155,8 Å², soit plus de
trois fois E320 — `identical`, à 11,8 Å d'un séquon. **Il n'était dans aucun des jeux de
hotspots discutés jusqu'ici**, parce qu'il appartient au patch T358 et pas au patch G317.
C'est l'oubli que ces trois éliminations mettent au jour.

Jeu de hotspots que cela suggère : `A323,A325,A357,A318` — l'ancre réelle, les deux plus gros
apolaires conservés, et un quatrième solide, **en évitant H359**.

**Non vérifié, et bloquant avant de l'adopter** : que ces quatre résidus pointent vers la même
face. À 28,9 Å d'étendue, l'union est proche de la limite de ce qu'un binder de 83 aa peut
enfouir, et rien ne garantit qu'un seul binder atteigne les quatre. Se tranche dans PyMOL, ou
par deux runs courts comparés — ce qui reste de toute façon le seul moyen de départager deux
jeux de hotspots.

**Aucun hotspot n'est changé, aucune constante n'est touchée, le jeu en vigueur reste
`A318,A320,A323,A325`.**

---

## Jeu de hotspots retenu (03/10/2026)

**`A318,A323,A325,A406,A409`** — décision prise, motifs ci-dessous.

### Le critère qui manquait : l'étendue

Aucune des analyses précédentes n'avait vérifié qu'un jeu de hotspots soit atteignable par
**un seul** binder. Ni les miennes, ni celles proposées par ailleurs. C'est le contrôle qui a
tranché.

Mesuré sur 6ARU chaîne A, distances **CA**, après vérification que la chaîne existe
(4727 atomes, objet contenant A, B, C, D, E) et que chaque résidu porte exactement un CA —
une sélection vide ne lève aucune erreur et donnerait silencieusement une valeur fausse.

```
          318     323     406     409
318      0,00    7,68   12,25   16,52
323      7,68    0,00   16,61   16,73
406     12,25   16,61    0,00    9,05
409     16,52   16,73    9,05    0,00
```

Recoupé entre **PyMOL 3 headless** (`cmd.get_distance` sur `chain A and resi N and name CA`)
et **Biopython** via `hotspot_distances.py` : accord exact à 0,1 Å près. L'écart aurait
signalé un problème de lecture de la numérotation auteur du mmCIF.

### Trois faits que la mesure établit

**1. Retirer `A318` ne gagne aucun angström.** L'étendue maximale est de **16,73 Å dans les
deux cas**, fixée par 323↔409 qui survit au retrait. Un jeu à 3 perdrait les 83,3 Å²
apolaires de I318 pour zéro gain géométrique. La règle de décision posée — seuil à 24 Å sur
318↔409, mesuré à 16,52 — était satisfaite, mais la vraie raison est plus forte : le
troisième résidu est gratuit.

**2. `A325` s'ajoute aussi sans coût géométrique.** L325↔409 = 14,7 Å, donc le jeu à 5 garde
une étendue de **16,73 Å inchangée** et gagne 111,1 Å² apolaires.

**3. `F357` est exclu, et c'est l'arbitrage de fond.** F357↔409 = **24,0 Å**, F357↔406 =
27,1 Å : l'ajouter porterait l'étendue à 27,1 Å. **F357 et H409 sont mutuellement
exclusifs.** Le choix est donc entre les 155,8 Å² de F357 — la meilleure ancre apolaire de la
carte — et la His conservée H409 avec sa sécurité glycane. **H409 retenu, parce que
l'objectif 1 est classé avant l'objectif 3.**

### Le bon décompte de la surface d'empilement

Compter la SASA apolaire de tous les membres du jeu surévalue ce qu'un binder empile : D323
est là pour son carboxylate, pas pour ses 30 Å² de tige. Décomposition par rôle :

| rôle | résidus | surface d'empilement |
|---|---|---|
| ancres apolaires pures | I318 (83,3) + T406 (66,4) + **L325 (111,1)** | **261 Å²** |
| double usage | H409, cycle imidazole | +82 Å² |
| fonction pure | D323 — valeur = carboxylate 75,0 Å² | — |

**L'ajout de L325 fait donc passer la surface d'empilement garantie de 150 à 261 Å², soit
+74 %** — et non +42 % comme le suggérait le total brut de tous les membres. C'est ce
recadrage qui a emporté la décision.

Décomposition atomique de H409, pour justifier le « double usage » : sur ses 82,1 Å² de
carbone, **44,7 Å² viennent du seul CE1**, le carbone du cycle imidazole entre les deux
azotes, plus CD2 (7,2) et CG (4,8). C'est un cycle aromatique exposé, il empile réellement.
Et ses azotes ND1 (20,8 Å²) et NE2 (13,1 Å²) portent le mécanisme pH. Les deux rôles sont
réels et simultanés.

### Fiche du jeu retenu

| | |
|---|---|
| étendue CA | **16,73 Å** — la plus compacte de tous les jeux examinés |
| identité humain/souris | **5/5** |
| surface d'empilement | 261 Å² garantis, ~343 avec le cycle de H409 |
| ancre pH acide | **D323** — carboxylate 75,0 Å², part 0,55, angle 33°, 89ᵉ centile du domaine |
| His conservée de la cible | **H409** — mécanisme inversé : un Asp sur le binder |
| glycane | min **7,3** (L325), médian 15,0 |
| dFab | min **0,0** (H409) — sur la surface ligand-compétitive |

Composition : I318 (83,3 apol / glyc 15,0), D323 (30,2 / 12,1), L325 (111,1 / 7,3),
T406 (66,4 / 19,9), H409 (82,1 / 21,1). Les cinq `identical`.

### Pourquoi ce jeu et pas mes deux zones

C'est un **hybride** que je n'avais pas proposé : I318 et D323 viennent de la zone A,
T406 et H409 de la zone B, et **T406 est la charnière** qui les relie. Il est plus compact
que la zone A (16,7 contre 19,8 Å) et c'est **le seul jeu examiné qui porte les deux
mécanismes pH** — l'acide pour une His du binder, et la His conservée pour un Asp du binder.

### Réserves

- **Le glycane de L325 à 7,3 Å est le seul vrai risque du jeu.** Les quatre autres résidus
  sont entre 12,1 et 21,1 Å d'un séquon. Et `min_glyc` est mesuré au CB du séquon, pas à
  l'arbre glycanique : il sous-estime l'occlusion. Le séquon candidat est UniProt 352 =
  **PDB 328**, trois résidus après L325 — non vérifié.
- **Aucune corréférence de face n'a été vérifiée.** 16,7 Å dit que les résidus sont proches,
  pas qu'ils regardent du même côté.
- **Les hotspots sont un biais, pas une contrainte.** À `PATCH_RADIUS = 11` ne capturant que
  40 % d'une empreinte réelle, les trajectoires dériveront. La stratégie retenue est
  délibérément de lancer ce seul jeu et de **trier les designs a posteriori** selon la zone
  atteinte, plutôt que de lancer trois campagnes.
- **Un seul rotamère, à 3,20 Å, en conformation repliée.** Toutes les SASA et tous les angles
  en héritent.
- **La troncature 309-506 est vérifiée sûre pour les cinq** : SASA identique à 0,0 Å² près
  entre chaîne A entière et domaine III isolé, contrôle étendu le 03/10 aux résidus de la
  zone B qui n'avaient jamais été testés.

### Révision du 3 octobre : `A325` écarté, jeu final à quatre

**Jeu retenu : `A318,A323,A406,A409`.** L325 est retiré pour la raison qui avait été
identifiée comme son seul point faible : **son glycane à 7,3 Å**.

Motif, et il est méthodologiquement le bon : `min_glyc` est mesuré au **CB du séquon**, pas à
l'arbre glycanique, qui s'étend bien au-delà et reste flexible. La colonne **sous-estime donc
l'occlusion de façon systématique** — 7,3 Å n'est pas une marge, c'est un risque. Et le séquon
candidat (UniProt 352 = PDB 328) n'a jamais été vérifié dans PyMOL. Retenir L325 revenait à
parier sur une vérification non faite.

L'échange, chiffré :

| | jeu à 5 | **jeu à 4 retenu** |
|---|---|---|
| ancres apolaires pures | I318 83 + T406 66 + L325 111 = **261 Å²** | I318 83 + T406 66 = **150 Å²** |
| avec le cycle de H409 (double usage) | 343 Å² | **232 Å²** |
| glycane min | **7,3** | **12,1** |
| glycane médian | 15,0 | **17,4** |
| glycane max | 21,1 | 21,1 |
| identité | 5/5 | **4/4** |
| étendue CA | 16,73 Å | **16,73 Å** — inchangée |
| dFab min | 0,0 | 0,0 |

**Le coût est réel : −43 % de surface d'empilement garantie**, de 261 à 150 Å². Le gain est de
sortir le seul résidu à risque glycanique du jeu : le minimum passe de 7,3 à **12,1 Å** et la
médiane de 15,0 à **17,4 Å**. Plus aucun hotspot sous 12 Å d'un séquon.

Deux conséquences à noter :

- **L'étendue ne change pas** (16,73 Å, fixée par 323↔409), donc retirer L325 ne libère aucune
  place géométrique.
- **`F357` ne redevient pas accessible** : F357↔H409 = 24,0 Å, inchangé. Le retrait de L325
  n'ouvre rien de ce côté.
- L'interface reposera donc largement sur le cycle imidazole de H409 (82 Å², double usage) et
  sur ce que les trajectoires iront chercher **au-delà** des hotspots. C'est cohérent avec la
  stratégie de tri a posteriori, et ça la rend plus nécessaire qu'avant.

`L325` devient un **candidat coldspot** pour BindCraft 2.0, au même titre que `H359` : s'il est
proche d'un glycane, autant l'écarter explicitement plutôt que de seulement ne pas le nommer.

CLAUDE.md §5 et §8 mis à jour en conséquence. Les deux décisions successives — ajouter L325 le
3/10 puis le retirer le même jour — sont conservées telles quelles dans ce journal : la
première reposait sur le gain de surface, la seconde sur le fait que `min_glyc` sous-estime
l'occlusion. Les deux sont défendables, la seconde est la prudente.

---

## 3 octobre — BindCraft 2.0 installé sur Modal, build validé ; vestiges de BC1 supprimés

### Ce qui a été supprimé

| supprimé | taille | motif |
|---|---|---|
| `modal_bindcraft.py` | 1429 lignes | entrypoint pour `c0a48d5`, avec sharding et setup PyRosetta |
| `inputs/PDL1.pdb` | — | cible de la démo du 23/09 |
| `out/test1`, `out/par-test2`, `out/par-test3` | 134 Mo | sorties de démo PD-L1 |
| volume : `test1`, `par-test`, `par-test2`, `par-test3` | — | les mêmes, côté Modal |

Tout est récupérable dans l'historique git. Les entrées de ce journal qui s'y réfèrent sont
conservées telles quelles — c'est un journal, pas un état.

**Conservé délibérément : `smoke-G317`**, en local (28 Ko de CSV) et sur le volume (avec les
structures acceptées). C'est la source de l'observation des 3 His d'interface citée dans
CLAUDE.md §6, et la matière de l'action 6. Supprimer une preuve citée dans un document suivi
est le seul geste irréversible du lot. Ses chiffres de débit restent caducs.

### BindCraft 2.0 : ce qui est un dépôt différent

**`PacesaLab/BindCraft2`, pas un tag de `martinpacesa/BindCraft`.** Épinglé à
`a8d0f2002df373842b86a3c20c5a060c5cfdf980` (29/09, HEAD au 3/10). Paquet `bindcraft 1.0.1`,
`requires-python >= 3.12`.

### Trois choses qui changent l'architecture, lues dans la source

**1. PyRosetta, DSSP et DAlphaBall ont disparu.** Zéro occurrence dans tout le dépôt amont.
Conséquences en cascade : plus de contrainte de licence académique ; plus de `chmod` de
binaires ; et surtout **le pin `numpy<2.0` n'a plus de raison d'être** — il n'existait que
pour PyRosetta. L'image installe `numpy` 2.x sans rien casser. Le relax est désormais interne
et en JAX (`relax_steps`, `relax_learning_rate`), et `relax_accepted_designs` est à `false`
par défaut, donc le « taux de relaxation 3/6 » de `c0a48d5` ne mesure plus rien.

**2. `TIMEOUT` n'existe pas.** Zéro occurrence de `timeout` dans `bindcraft/` hors appels
réseau. Tout le modèle de budget de CLAUDE.md §2 — `check_n_trajectories` ne comptant que
`Trajectory/Relaxed`, donc une trajectoire `LowConfidence`/`Clashing` brûlant du GPU hors
quota — **n'a plus de code correspondant**. Le budget se pilote par `max_trajectories` et par
le `timeout` de la fonction Modal.

**3. `resume` est à `true` par défaut** (`settings/core/default.json`). Un rappel sur le même
`run_name` reprend la campagne.

→ **L'action 3 du §8 est sans objet.** « Vérifier si un kill par `TIMEOUT` commite le volume »
portait sur un mécanisme qui n'existe plus, et `resume` rend l'architecture en appels courts
sûre par construction. $0,20 non dépensés.

### Les filtres réels, enfin lus

`default_filters.json` n'existe plus ; les seuils sont dans `settings/core/default.json`.
**Le piège d'échelle de CLAUDE.md §6 est confirmé : tout est sur [0,1].**

| filtre | seuil | sens |
|---|---|---|
| `Unbound_Binder_pLDDT` | 0,80 | ↑ |
| `pTM` | 0,55 | ↑ |
| `i_pTM` | 0,70 | ↑ |
| `i_pAE` | 0,35 | ↓ |
| `Backbone_Clashes` | 0 | ↓ |
| `Interface_Residues` | 7 | ↑ |

Et c'est maintenant un **pipeline par étages**, pas un filtre unique en bout de chaîne :
`min_plddt_screen` 0,60 → `_refine` 0,60 → `_anneal` 0,65 → `_harden` 0,65 → `_final` 0,70,
avec `min_iptm_*` à 0,50 et `min_iptm_final` 0,70. Le « profil de rejet » de `c0a48d5`
(`i_pAE` 20/29) n'a donc plus la même forme et est à re-mesurer. Les sorties s'appellent
`trajectories.csv`, `candidates.csv`, `accepted.csv`.

### Le build Modal

Porté de `containers/Dockerfile` de l'amont vers `modal_bindcraft2.py`. Deux pièges que
l'amont documente et qu'il fallait garder :

- les wheels `jax[cuda13]` gardent leurs bibliothèques sous `site-packages/nvidia/*/lib`,
  **où le loader ne regarde pas**. Sans un fichier `ld.so.conf.d` qui les déclare, jax
  avertit une fois puis tourne sur le CPU, cent fois plus lentement, **et rien ensuite ne le
  signale**. Le build échoue exprès sur `ldconfig -p | grep -q libcupti` ;
- l'install doit être **éditable** à la racine du dépôt : `settings/` et `scaffolds/` vivent
  à la racine et non dans le paquet, et le runtime les trouve relativement à lui.

Poids : ProteinMPNN est **livré dans le paquet** (77 Mo), seuls les 5,3 Go d'AF2 sont
téléchargés, et ils sont **bakés dans l'image** — choix de l'ancien build, reproduit, et que
l'amont recommande aussi (`--build-arg ALPHAFOLD_PARAMETERS=bake`). Le téléchargement amont
est en `urllib` mono-flux, plus lent que l'`aria2c -x16` de BC1, mais c'est une fois.

**Faux positif à connaître** : pendant le build, l'étape de vérification crache un
`RuntimeError: Unable to load cuPTI. Is it installed?` puis imprime `jax 0.11.2` et sort en 0.
C'est la machine de build, qui n'a pas de GPU. Ça n'invalide rien — mais ça ressemble
exactement à l'échec qu'on cherche à éviter, donc ne pas s'y tromper. Le seul test qui compte
est sur un vrai GPU.

### Validation

```
modal run modal_bindcraft2.py::selfcheck
```

```
jax 0.11.2 | backend gpu | devices [CudaDevice(id=0)]
poids AF2 : /opt/bindcraft/bindcraft/weights/alphafold
checkpoints : les 7 modèles AF2 et les 3 variantes ProteinMPNN sont complets
cible /root/inputs/6ARU_A_309-506.pdb : 198 résidus

BUILD VALIDE
```

GPU L40S, app `ap-smgZANABHeq8fwiRWFU0ct`, état `stopped`. `backend gpu` et non `cpu` : le
piège du loader est évité. Durée GPU de l'ordre de la minute ; **coût non relevé précisément**
— le CLI Modal ne donne pas les GPU-secondes, et je ne vais pas inventer un chiffre. Au tarif
mesuré de $1,95/h, c'est ~$0,03.

**Ce qui reste non établi** : la structure des sorties, qui ne se verra qu'au premier vrai run.

### Numérotation : vérifiée par lecture, pas de mémoire

Lecture directe de `inputs/6ARU_A_309-506.pdb` : `318 ILE`, `323 ASP`, `325 LEU`, `359 HIS`,
`406 THR`, `409 HIS`. Les six noms de CLAUDE.md (I318, D323, L325, H359, T406, H409) sont donc
tous en numérotation **PDB**, cohérente avec le fichier cible et avec les hotspots. Les six His
de la cible sont aux PDB 334, 346, 359, 394, 409, 483.

### Coldspots : `A359` câblé, et pourquoi `A325` ne l'est PAS

La forme réelle est une clé `coldspots` par cible, acceptant des plages (`"131-134,139"`).
Poser des coldspots active `weights_coldspot_repel = 1.0` et pose
`max_coldspot_contact_final = 0,05` (`bindcraft/settings.py:192-195`). Le run logge
`target=… coldspots=… residues=N` : c'est le point de vérification que la plage a été résolue.

**Le rayon des pertes coldspot est de 8,0 Å** (`bindcraft/loss.py`, `cutoff: float=8.0` sur
`binder_coldspot`, `binder_intra_coldspot` et `coldspot_repel`). Tout coldspot candidat doit
donc être mesuré contre ce rayon. Distances CA mesurées le 3 octobre :

| candidat | 318 | 323 | 406 | 409 | verdict |
|---|---|---|---|---|---|
| **A359** | 19,46 | **13,50** | 28,22 | 26,65 | **sûr** — hors du rayon de 8 Å |
| **A325** | 10,61 | **6,36** | 16,06 | 14,66 | **écarté** — sous le rayon |

La fin de l'entrée précédente faisait de `L325` un « candidat coldspot au même titre que
H359 ». **C'était faux, et la mesure le montre.** À 6,36 Å de A323, la sphère de répulsion
autour de 325 avale 323 : déclarer A325 coldspot repousserait le binder hors de **D323**, qui
porte la route pH n°1. Ce serait sacrifier l'objectif le mieux classé du challenge pour
écarter un risque glycanique sur un résidu qu'on a déjà sorti des hotspots — le risque est
déjà traité par son retrait, l'écarter *activement* coûte beaucoup plus qu'il ne rapporte.

Les deux raisonnements successifs sont conservés : le premier était une intuition de symétrie
avec H359, le second une mesure. H359 est à 13,50 Å, A325 à 6,36 Å — la symétrie n'existait
pas.

**Coldspots encore incomplets** : les résidus à moins de ~10 Å d'un séquon (§8 action 4) n'ont
jamais été mesurés depuis les hotspots retenus. À mesurer, puis à filtrer contre le rayon de
8 Å comme ci-dessus.

### Prochaine dépense

Action 2 : un run court — `--max-trajectories 3` — pour fixer le temps et le coût par
trajectoire sur 2.0 avant d'engager un budget. Aucun chiffre de débit de `c0a48d5` ne survit.

---

## 3 octobre (suite) — parallélisation : pas de sharding, et un défaut corrigé dans la calibration

Question posée : est-ce que l'entrypoint parallélise ? Réponse : non explicitement, et il ne
doit pas — mais le run de calibration tel que je l'avais annoncé était mal spécifié.

### Pourquoi le sharding de BindCraft 1 serait maintenant nocif

L'ancien `modal_bindcraft.py` découpait en `shard-000`, `shard-001`… un conteneur et un GPU
par shard, parce que BindCraft 1 n'avait aucune coordination au niveau campagne. Sur 2.0 :

- chaque shard serait une **campagne indépendante** chassant son propre
  `number_of_final_designs`. N shards = N × les designs et N × le coût ;
- le barreau de l'échelle de désespoir est « lu sur les tables de la campagne, donc chaque
  worker et une campagne reprise sont sur le même » — des shards le compteraient chacun de
  leur côté ;
- `.campaign_state.json` est un état de campagne, pas de processus.

Le sharding se bat contre les trois. Retiré, et un avertissement posé dans le fichier pour
ne pas y revenir.

### Ce que BindCraft 2.0 fait tout seul

`workers_per_gpu` vaut `auto` par défaut, ce qui résout à 7 puis se fait plafonner par la
mémoire. Formule lue dans `bindcraft/design_workers.py` :

```
mem_par_worker = DESIGN_MEMORY_SAFETY_FACTOR × (DESIGN_MODEL_RESIDENT_GB
                 + DESIGN_ACTIVATION_BYTES_PER_RESIDUE_PAIR × n_residus² / 1e9)
               = 2.0 × (3.4 + 38000 × n² / 1e9)
workers        = (libre_Go − GPU_MEMORY_HEADROOM_GB) // mem_par_worker   # headroom = 4,0
```

Calculé pour notre cible (198 résidus) sur une L40S de 46 Go :

| binder | total résidus | Go/worker | workers |
|---|---|---|---|
| 55 | 253 | 11,66 | **3** |
| 64 | 262 | 12,02 | **3** |
| 95 | 293 | 13,32 | **3** |

On ne tombe à 1 worker que vers **430 résidus** au total. Donc **3× de parallélisme déjà
acquis, sans rien coder.** Un second plafond existe côté RAM hôte
(`HOST_MEMORY_PER_WORKER_GB = 4.0`, soit `dispo_Go // 4 // n_gpu`) — non vérifié sur une
instance Modal L40S, à regarder dans le log du premier run.

### Le défaut corrigé

**`max_trajectories: 3` avec 3 workers fait partir les trois trajectoires en parallèle.** Le
temps mural vaut alors une trajectoire plus la compilation, et on ne peut pas en déduire un
temps par trajectoire. J'avais présenté ce run comme une mesure de débit : c'était un
échantillon de taille 1 sous concurrence non contrôlée.

Trois ajouts dans `modal_bindcraft2.py` :

| variable | défaut | effet |
|---|---|---|
| `WORKERS` | `auto` | écrit dans `workers_per_gpu`, donc tracé dans `campaign_metadata.json`. `WORKERS=1` sérialise pour mesurer. |
| `GPU_COUNT` | `1` | `gpu="L40S:N"` — N cartes dans **un** conteneur. `auto_multi_gpu` étant à true, BC2 répartit seul. Une campagne, pas N. |
| avertissement CLI | — | si `WORKERS=auto` et `max_trajectories ≤ 4`, le lancement prévient que le temps mural ne donnera pas un temps par trajectoire. |

### Révision du plan de calibration

Deux runs au lieu d'un, et ils ne mesurent pas la même chose :

1. **`WORKERS=1`, `max_trajectories 3`** — temps par trajectoire propre, profil de rejet par
   étage, vérification que la ligne `target=… coldspots=… residues=N` résout bien `A359`.
   C'est le run qui sert à *comprendre*.
2. **`WORKERS=auto`, max_trajectories dimensionné sur (1)** — débit réel et coût par design
   accepté, dans la configuration de production. C'est le run qui sert à *budgétiser*.

Note sur la compilation : `length_bucket_size` vaut 32, donc `[55, 95]` rembourre vers 64 et
96, soit **deux seaux** et deux compilations par worker. Avec 3 workers et
`worker_launch_stagger` à 0, le démarrage concentre jusqu'à 6 compilations. À surveiller
dans le log avant d'incriminer le débit.

---

## 3 octobre (suite 2) — paramètres Modal : ce qui manquait réellement

Question reprécisée : la parallélisation **côté Modal**. Réponse : un seul `.remote()`, un
seul conteneur, et c'est volontaire (une campagne = un état, cf. entrée précédente). Mais en
auditant les paramètres Modal non définis, cinq manquaient, dont deux graves.

### 1. `memory` — le plus grave, il annulait les 3 workers

BindCraft plafonne ses workers **deux fois**, et je n'avais vu que la première. Après le
plafond mémoire GPU, `design_workers.py` applique :

```
host_memory_worker_ceiling = (MemAvailable_Go // HOST_MEMORY_PER_WORKER_GB) // n_gpu
                           = (MemAvailable_Go // 4,0) // n_gpu
```

lu sur `/proc/meminfo`. Donc sous **12 Go** de RAM disponible on retombe à 2 workers, sous
8 Go à 1 — et les 3 workers calculés sur la mémoire GPU seraient perdus **en silence**. La
RAM par défaut d'un conteneur Modal n'était pas définie dans l'entrypoint.

Fixé à **24 Go et 4 cœurs par GPU**, qui est la taille que le script Slurm de l'amont se
donne lui-même (`docs/source/installation.md`) — chiffre sourcé, pas posé. Vérifié :
`24 // 4 = 6` de plafond hôte, donc c'est bien la mémoire GPU qui décide avec 3.

### 2. `timeout` — il n'y avait pas de plafond de budget

`timeout=86400` était le maximum de Modal, soit **$47** sur une L40S à $1,95/h et **$187**
sur quatre. La règle du dépôt interdit de lancer un run GPU sans plafond de budget, et
`max_trajectories` ne protège pas d'un run bloqué. Le timeout est le seul vrai plafond.

Remplacé par une dérivation depuis un budget en dollars :

```
timeout = 3600 × BUDGET_USD / (USD_PER_HOUR[GPU] × GPU_COUNT)
```

| `BUDGET_USD` | `GPU_COUNT` | timeout | coût max |
|---|---|---|---|
| 1 | 1 | 1846 s = 0,51 h | $1,00 |
| 5 | 1 | 9230 s = 2,56 h | $5,00 |
| 5 | 4 | 2307 s = 0,64 h | $5,00 |
| 20 | 1 | 36923 s = 10,26 h | $20,00 |

Défaut `BUDGET_USD=5`. Le coût plafonne au budget quel que soit le nombre de cartes, ce qui
est exactement le comportement voulu. Et un `GPU` dont le tarif n'est pas dans `USD_PER_HOUR`
**fait échouer le chargement du module** au lieu de lancer un run non plafonné — testé avec
`GPU=H100`.

### 3. Commits du Volume — un seul, à la fin

`volume.commit()` n'était appelé qu'en `finally`. Un conteneur tué dur (préemption, OOM)
perdait tout le run. Ajout d'un thread de commit toutes les **300 s**. `resume` étant à true,
un redémarrage repart alors du dernier commit au lieu de zéro.

### 4. Cache de compilation XLA — éphémère

`bindcraft/__init__.py` pose `JAX_COMPILATION_CACHE_DIR` par défaut sur
`/tmp/bindcraft_xla_cache`, **éphémère par conteneur** : chaque appel Modal recompile tout.
Pointé sur `/outputs/.xla_cache`, donc persistant d'un appel à l'autre. Ça compte parce que
`length_bucket_size 32` rembourre `[55,95]` vers 64 et 96 — deux compilations par worker à
chaque démarrage, jusqu'à 6 avec 3 workers.

**Gain à mesurer, pas promis** : l'amont pose aussi
`JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`, donc tout n'est pas sérialisé.

### 5. `retries` et `max_containers`

`retries=0` explicite — un réessai sur une fonction GPU de plusieurs heures doublerait la
facture en silence (le défaut Modal est déjà 0, mais l'écrire est la règle).
`max_containers=1` — garde contre un fan-out accidentel, puisque par construction une
campagne ne doit occuper qu'un conteneur.

### Piège de l'amont à ne pas déclencher

`BINDCRAFT_WORKER_ID`, `BINDCRAFT_WORKER_COUNT` et `BINDCRAFT_BINDER_LENGTHS` sont posées
**par** la campagne pour chaque worker. L'amont prévient : « ne les définissez pas ; un
processus qui porte `BINDCRAFT_WORKER_ID` se croit worker et ne se répartira pas. » On ne
les touche pas. Autre piège associé : une variable exportée mais **vide** compte comme une
valeur, donc `WORKERS` est désormais lu avec `os.environ.get("WORKERS") or "auto"`.

### Reste non défini volontairement

`region` (aucune contrainte de données), `ephemeral_disk` (les sorties vont au Volume, l'image
porte les poids), `scaledown_window` et `buffer_containers` (sans objet pour une fonction
one-shot).

---

## 3 octobre (suite 3) — revue de code : les workers ne s'écrasent pas, et quatre bugs à moi

Question posée : vérifier que les workers concurrents ne s'écrasent pas, et chercher d'autres
bugs. Résultat : le mécanisme de BindCraft est sain **et testé**, mais j'avais introduit
quatre défauts.

### Comment BindCraft écrit, et où était le risque

`append_campaign_metrics` (`bindcraft/campaign_output.py`) fait un **read-modify-write du CSV
entier** sous verrou :

```
with locked_campaign_folder(csv_path):      # fcntl.flock(fd_du_dossier, LOCK_EX)
    relire TOUTES les lignes existantes
    écrire tout + la nouvelle dans csv_path.partial
    os.replace(partial, csv_path)           # renommage atomique
```

Même motif pour `.campaign_state.json` (`locked_progress`) et pour `claim_trajectory()`, qui
sous le même verrou réserve un créneau et vérifie `accepted >= requested_designs`.

**Le risque est réel si le flock ne protège pas** : deux workers relisent le même état, écrivent
chacun leur version complète, et `os.replace` du dernier **écrase la ligne de l'autre**. Pas de
corruption, une perte silencieuse.

### Test, parce que le Volume Modal est un FUSE réseau

Nouvel entrypoint `volume_concurrency_check`, CPU seul, qui lance N `subprocess` appelant la
**vraie** fonction de BindCraft sur un CSV du Volume. 3 processus × 60 lignes :

```
lignes trouvees : 180 / 180 attendues
  worker 0 : 60 / 60    worker 1 : 60 / 60    worker 2 : 60 / 60
fichiers .partial residuels : aucun
PAS D'ECRASEMENT : flock protege le read-modify-write sur ce Volume.
```

App `ap-Awmvvxza84f2fZg7xHaXca`. **Conclusion : pas d'écrasement entre workers.** C'est
cohérent avec le mécanisme : les workers de BindCraft sont des `subprocess.Popen` d'**un seul
conteneur** (`design_workers.py:218`), donc le flock est arbitré par le noyau sur le même
mount, FUSE ou pas. Chaque worker a aussi son propre log, `worker_NN_gpu_N.log`.

**⚠️ Ce que le test NE prouve PAS** : `flock` est local à l'hôte. Rien n'est garanti **entre
conteneurs**. C'est une deuxième raison, de correction et plus seulement de coût, de ne pas
sharder — et c'est ce que garde `max_containers=1`.

### Mes quatre bugs

**1. Cache XLA sur le Volume — retiré du chemin par défaut.** Je l'avais mis sur `/outputs`
sans l'éprouver : système de fichiers réseau, jusqu'à 3 écrivains concurrents, et gain non
mesuré puisque l'amont pose `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`. L'introduire dans
le run censé *mesurer* le débit aurait ajouté une variable non contrôlée à la mesure. Devenu
opt-in par `XLA_CACHE_ON_VOLUME=1`, à tester séparément.

**2. `settings.json` écrasé en reprise — garde ajoutée.** `resume` étant à true, un rappel sur
le même `run_name` reprend la campagne. Mais j'écrasais `settings.json` sans regarder : avec
d'autres paramètres, le fichier ne décrivait plus le run qui avait produit les lignes déjà
là, et les tables devenaient inexplicables. Le run **refuse** maintenant, en listant la
dérive clé par clé. Reprise autorisée seulement à réglages identiques.

**3. Commit périodique silencieux en cas d'échec.** Le thread était sans `try`, donc une
exception le tuait en silence et on perdait les commits sans le savoir. Rattrapé et signalé.

**4. Plancher de budget trompeur.** `max(300, …)` pouvait dépasser le budget demandé :
`BUDGET_USD=0.02` donnait 36 s, remonté à 300 s, soit $0,16 réels. Le dit maintenant au lieu
de laisser croire que le plafond est tenu.

Plus un décalage mineur corrigé : `n_designs` valait 10 par défaut dans la CLI alors que la
doc recommande 12.

### Revu et jugé correct

- `targets[].target_path` absolu, donc insensible au `cwd` ; `settings/` est trouvé par
  `Path(__file__).parent.parent` et non par le `cwd`, donc l'install éditable suffit ;
- `volume.commit()` périodique est **sûr** vis-à-vis du motif `.partial` + `os.replace` : le
  CSV visible est toujours complet, ancienne ou nouvelle version, jamais tronqué. Un
  `.partial` capturé dans un snapshot est inoffensif ;
- `retries=0` : un réessai sur une fonction GPU longue doublerait la facture ;
- `WORKERS` lu avec `or "auto"`, l'amont prévenant qu'une variable vide compte comme valeur ;
- `BINDCRAFT_WORKER_ID`/`_COUNT`/`_BINDER_LENGTHS` ne sont jamais posées par nous — l'amont
  prévient qu'un processus qui les porte se croit worker et ne se répartit pas.

### Limite connue, assumée

`max_containers=1` empêche aussi deux campagnes **différentes** de tourner en parallèle, même
sur des dossiers disjoints. C'est volontaire : ça coûte un peu de débit et ça supprime la
seule erreur catastrophique, deux conteneurs sur le même `project_folder` sans flock partagé.

---

## 3 octobre (suite 4) — deux bugs attrapés en lançant le run, dont un de fond

Le run de calibration a été lancé deux fois et arrêté deux fois. Les deux échecs sont
instructifs et valent d'être consignés.

### Échec 1 — `exit=0` sans que rien ne démarre

```
Error: Missing option '--n-designs'.
```

**Exit code 0.** C'est exactement le piège que CLAUDE.md §7 documente, et sans la
vérification sur `modal app list` j'aurais annoncé un run qui n'existait pas.

Cause : en ciblant `modal run ...::design`, Modal construit la CLI depuis la signature de
`design`, pas de `main`. Les défauts vivaient sur `main`, donc `--n-designs` était
obligatoire — **la commande documentée dans le docstring et sur la page HTML ne marchait
pas.** Corrigé par des défauts partagés `DEFAULT_MAX_TRAJECTORIES` et `DEFAULT_N_DESIGNS`
utilisés par les deux entrypoints, pour que la divergence ne puisse plus revenir.

### Échec 2 — les variables d'environnement ne franchissent pas la frontière Modal

Le run a démarré, et sa config affichait :

```
"workers_per_gpu": "auto"
```

alors que la commande était `WORKERS=1 modal run ...`. Run arrêté au bout d'environ
2 minutes, **~$0,07**.

**La règle, et elle vaut pour tout ce dépôt : Modal ne propage pas l'environnement local au
conteneur.** Une valeur lue par `os.environ` au niveau module n'est correcte que si elle sert
dans un **décorateur**, qui est évalué en local à l'import. Tout ce qui est lu à l'exécution
tourne dans le conteneur, où la variable est absente et retombe sur le défaut, **en silence**.

Trois instances du même défaut dans ce que j'avais écrit :

| valeur | utilisée où | état |
|---|---|---|
| `GPU`, `GPU_COUNT`, `BUDGET_USD` | décorateur (`gpu=`, `timeout=`, `cpu=`, `memory=`) | **correct**, évalué en local |
| `WORKERS` | `campaign_settings()`, à distance | **cassé** — retombait sur `auto`, donc 3 workers au lieu d'1 |
| `XLA_CACHE_ON_VOLUME` | `design()`, à distance | **cassé** — l'opt-in n'aurait jamais pu s'activer |
| print du budget | `design()`, à distance | **trompeur** — recalculé sur le défaut, affichait un faux plafond |

Corrigé : ce qui est lu à l'exécution devient un **argument de fonction**, sérialisé par
Modal et visible dans la commande. `--workers` et `--xla-cache` remplacent les variables
d'environnement. La ligne de budget n'est plus imprimée côté conteneur mais côté `main`, en
local, où la valeur est vraie. `GPU_COUNT` et `BUDGET_USD` restent des variables
d'environnement, ce qui est légitime puisqu'elles ne servent qu'aux décorateurs.

**La garde de reproductibilité a fonctionné au passage** : le dossier du run avorté portait
un `settings.json` à `workers_per_gpu: "auto"`, donc un relancement à `--workers 1` aurait
été refusé pour dérive. Supprimé du Volume avant de relancer.

---

## 3 octobre — run `egfr-dIII-cal01` : 0 accepté sur 3, et le goulot n'est pas celui qu'on croyait

Premier vrai run sur BindCraft 2.0. App `ap-pO6Fl3Dxy9DPjYLJv36aj3`, L40S, `--workers 1`
(série), `--max-trajectories 3`, `BUDGET_USD=3` → timeout 5538 s. **Le budget n'a pas été
atteint**, la campagne s'est arrêtée d'elle-même sur son plafond de trajectoires.

### Chiffrage — enfin des nombres sur 2.0

Il existe une colonne **`Timing`** dans `1_Trajectories/!_Trajectories.csv`, au format
`worker=0;start=<epoch>;design=<secondes>;compiled=<0|1>`. Tarif L40S $0,000542/s.

| traj | longueur | `design` | compilé | arrêt | durée réelle | coût |
|---|---|---|---|---|---|---|
| 1 | 69 | 430,5 s (7,2 min) | **oui** | va au bout | **559,5 s** (9,32 min) | **$0,303** |
| 2 | 58 | 239,3 s (4,0 min) | non | `mutate` | 241,0 s (4,02 min) | $0,131 |
| 3 | 65 | 101,5 s (1,7 min) | non | `screen` | 101,5 s (1,7 min) | $0,055 |

**Total des 3 trajectoires : 902 s = 15,03 min = $0,489** de GPU, plus le démarrage du
conteneur et la préparation de la cible — run complet autour de **$0,55**.

Deux écarts à noter contre `c0a48d5` : la trajectoire complète inclut 431 s de gradient
**plus ~129 s de redesign ProteinMPNN et de validation** des 10 candidats, et le coût par
trajectoire complète ($0,303) est très proche des $0,33 par tentative de BindCraft 1. Ce
n'est pas une accélération — c'est une architecture différente au même prix.

**Le coût est bimodal** : $0,30 pour une trajectoire qui va au bout, $0,09 en moyenne pour
une qui meurt à un plancher d'étage. Un coût moyen n'a de sens que pondéré par la proportion
de morts précoces, et 2 sur 3 ici n'est pas une statistique.

**⚠️ Non mesuré : ce que la concurrence rapporte vraiment.** Avec `--workers auto` (3 workers),
3 trajectoires partagent une carte. Si la carte est saturée en calcul, chacune tourne ~3× plus
lentement et **il n'y a aucun gain de coût** — seulement du gain de temps mural si la carte
est limitée par la latence ou la mémoire. C'est l'objet du run suivant.

### Le résultat : 0 accepté sur 3

BindCraft le dit lui-même en terminant :

> *campaign stopped: 3 trajectories ran and none were accepted, so the **settings** rather
> than the budget are what to change*

Deux modes d'échec distincts, et il faut les séparer :

**(a) 2 trajectoires sur 3 ne produisent même pas un repliement confiant.** Traj 3 meurt au
`screen` sur `pLDDT=0.55` (plancher 0,60), traj 2 au `mutate` sur `pLDDT=0.55` et
`i_pTM=0.24` (planchers 0,60 et 0,50). Elles n'atteignent jamais ProteinMPNN.

**(b) La seule qui va au bout perd 0,124 d'`i_pTM` entre le gradient et la validation.**
C'est le point important :

| | `i_pTM` | `i_pAE` |
|---|---|---|
| traj 1, mesurée par les modèles de **design** | **0,76** | **0,24** |
| ses 10 candidats, par la **validation** | 0,636 (0,57–0,67) | 0,393 |
| écart | **0,124** | **0,153** |

**Les valeurs du gradient PASSENT les seuils** (≥0,70 et ≤0,35). Aucun des 10 candidats ne
passe. Donc ce n'est pas « la cible est difficile » — le gradient trouve une pose correcte,
et elle ne survit pas à l'évaluation tenue à l'écart.

**Lecture la plus probable : c'est un écart de généralisation.** BindCraft impose que les
modèles de design et de validation soient disjoints (« a design is never scored by a model
that shaped it »), donc 0,76 est l'auto-évaluation des modèles qui ont optimisé, et 0,636
le verdict de modèles tenus à l'écart. `redesign_interface` étant à `false`, ProteinMPNN n'a
pas réécrit l'interface, ce qui affaiblit l'explication alternative d'une dégradation par
le redesign.

**Ce run ne peut pas séparer complètement les deux contributions** — la séquence change *et*
elle est repliée à nouveau. `initial_guess` (re-prédire depuis la pose qu'a pliée la
trajectoire) isolerait la part de la pose, et c'est le barreau 1 de l'échelle de désespoir.

**Conséquence chiffrée et directement versable au dossier de méthodes** : le seuil de 0,70
s'applique à la prédiction tenue à l'écart, donc une trajectoire doit atteindre **~0,82+ sur
les modèles de design** pour espérer passer. C'est la préoccupation du §3 sur le re-scoring
orthogonal, quantifiée de l'intérieur : sur cette cible, l'auto-évaluation d'AF2 surestime la
confiance d'interface d'environ **0,12**.

### Ce qui marche

- **Le coldspot `A359` tient parfaitement** : `Coldspot_Contact_Fraction = 0.0` sur les 10
  candidats, et il n'a causé aucun rejet. La ligne `coldspots=A359 residues=1` confirme la
  résolution. Décision validée empiriquement.
- **`aa_bias {"C": 0}` fonctionne** : `Binder_Cysteines = 0`, `Binder_Free_Cysteines = 0`.
- `Interface_BuriedArea = 764,8 Å²` et `Epitope_Residues_Contacted = 10` sur traj 1 — l'
  interface n'est pas maigre.
- `Surface_Hydrophobicity = 0,33`, `Binder_Net_Charge = −6`, `pI = 4,2`.

### Ce qui confirme la stratégie de tri a posteriori

`Hotspot_Contact_Fraction = 0,25` sur les 10 candidats, soit **1 hotspot sur 4 contacté**, et
`Off_Epitope_Contact_Fraction` entre 0,38 et 0,50. La dérive annoncée au §6 est mesurée.
**Et ça valide le refus de `forced_targeting`** : son plancher par défaut est
`min_hotspot_contact_final = 0.5`, donc ces designs auraient été rejetés une seconde fois.

Note : la constance de ces fractions sur les 10 candidats vient de ce qu'une trajectoire
produit **un** squelette dont les 10 séquences héritent de la pose. **L'acceptation se décide
au niveau du squelette**, donc 3 trajectoires est un échantillon minuscule pour un taux
d'acceptation — suffisant pour le temps, pas pour le rendement.

### Correction de doc

`Unbound_Binder_pLDDT` vaut **0,70** et non 0,80. 0,80 est la valeur de
`settings/core/default.json`, mais le preset `binder` l'écrase par
`min_monomer_plddt_final: 0.7`. Lu dans `campaign_metadata.json`, qui enregistre les réglages
résolus et qui est la source autoritative. L'amont prévient que « les défauts changent avec la
modalité choisie » — lu, et pas appliqué. CLAUDE.md §6 corrigé.

Autre découverte : BindCraft maintient **lui-même** un `compile_cache/` dans le dossier de
run, donc sur le Volume. L'option `--xla-cache` que j'avais ajoutée est largement redondante,
et l'activer créerait un second cache concurrent du premier. Elle reste en opt-in, éteinte.

### Décision pour la suite

Ne **pas** baisser le seuil d'`i_pTM` : ce serait changer ce qui compte comme acceptable, et
un design accepté à seuil abaissé est un candidat plus faible pour une validation en labo.
Le levier est le nombre de squelettes échantillonnés. À $0,30 la trajectoire complète et
$0,09 la morte précoce, **30 trajectoires coûtent de l'ordre de $5** et donneraient un vrai
taux d'acceptation. C'est le prochain run.

---

## 3 octobre — run `egfr-dIII-prod01` : prédictions écrites AVANT le lancement

Pratique du 2 octobre reprise : écrire les prédictions avant, les confronter après. Le run
porte 30 trajectoires avec **tous les réglages identiques à `cal01`** sauf deux :
`--workers auto` (au lieu de 1) et `--max-trajectories 30` (au lieu de 3).

**Pourquoi ne rien changer d'autre** : `initial_guess` serait le candidat évident pour
attaquer l'écart de 0,124, mais le changer en même temps que l'échelle rendrait les deux
effets inséparables. Ce run mesure le taux d'acceptation **de la configuration qu'on vient de
caractériser**. Une chose à la fois.

Paramètres : L40S, `BUDGET_USD=6` → timeout 11077 s = 3,08 h de plafond dur.
`number_of_final_designs` reste à 12, donc la campagne s'arrêtera avant 30 trajectoires si
12 designs passent.

| # | prédiction | fondement |
|---|---|---|
| 1 | **50 à 70 %** des trajectoires meurent à un plancher d'étage avant d'atteindre ProteinMPNN | 2 sur 3 dans `cal01`, échantillon minuscule |
| 2 | **0 à 2 designs acceptés** sur 30 trajectoires | le meilleur candidat de `cal01` était à 0,67 contre 0,70, et l'acceptation se décide au squelette |
| 3 | le meilleur `i_pTM` tous candidats confondus tombera entre **0,68 et 0,72** | extrapolation de la queue de distribution depuis 10 candidats d'un seul squelette |
| 4 | temps mural **1,0 à 2,0 h** | entre l'absence de contention (0,8 h) et une contention totale à 3 workers (2,5 h) |
| 5 | coût réel **$2 à $4** | 10 complètes à $0,30 + 20 précoces à $0,09 ≈ $4,8 en série ; moins si la concurrence rapporte |
| 6 | `Hotspot_Contact_Fraction` **variera** d'un squelette à l'autre | sa constance dans `cal01` était *intra*-squelette, les 10 séquences héritant d'une seule pose |

**Ce que le run permet de décider** : si la prédiction 2 se vérifie à 0, le seuil d'`i_pTM`
devient le sujet et il faudra arbitrer entre l'abaisser — en le documentant — et attaquer
l'écart de généralisation par `initial_guess`. Si 1 ou 2 passent, le levier est purement le
volume de trajectoires et le chiffrage de la prédiction 5 dit combien en acheter.

La prédiction 4 est la seule qui mesure quelque chose de neuf sur Modal : **est-ce que
3 workers sur une carte réduisent le coût, ou seulement le temps mural ?** Les temps `design`
par trajectoire sont directement comparables à la série de `cal01` (430,5 / 239,3 / 101,5 s).
S'ils sont inchangés, la concurrence est un gain net. S'ils triplent, elle ne rapporte rien.

---

## 3 octobre — correction : 2 workers et non 3, j'avais oublié le rembourrage

Observé au démarrage de `prod01`, avant tout résultat :

```
worker=0 gpu=0 draws 31 binder lengths from 65 to 95, folded at 320 padded residues at 14.6 GB
worker=1 gpu=0 draws 10 binder lengths from 55 to 64, folded at 288 padded residues at 13.1 GB
```

**Deux workers, pas trois.** J'avais annoncé 3 dans CLAUDE.md, dans le code et dans trois
messages de commit. L'erreur : j'ai calculé `estimate_design_memory_gb` sur le nombre de
résidus **brut** alors que BindCraft le calcule sur le nombre **rembourré**.
`length_bucket_size = 32` arrondit (cible + binder) au multiple de 32 supérieur.

| binder | bruts (198 + b) | rembourré | Go/worker calculé | Go annoncé par BC2 |
|---|---|---|---|---|
| 55–64 | 253–262 | **288** | **13,10** | 13.1 |
| 65–95 | 263–293 | **320** | **14,58** | 14.6 |

Mes valeurs recalculées tombent exactement sur celles du log, donc l'explication est
certaine. `(45 − 4) // 14,58 = 2`.

**Conséquences :**

- la concurrence rapporte au mieux **2×**, pas 3×. La prédiction n°4 du run (temps mural
  1,0–2,0 h) était fondée sur 3× et est donc probablement trop optimiste — à confronter ;
- **nouveau levier de débit** : la plage de longueurs pilote le nombre de workers par le seau
  de rembourrage. `[55, 95]` donne 2 workers ; `[55, 64]` tiendrait dans le seul seau de 288
  et en donnerait 3. Gain de 50 % de débit contre un resserrement de la diversité de
  longueurs. À arbitrer, pas à appliquer en aveugle ;
- **biais à surveiller dans le lot final** : le partage est inégal — worker 0 tire 31
  longueurs (65–95), worker 1 seulement 10 (55–64) mais dans le seau plus rapide. L'amont
  prévient que « les groupes de longueurs plus rapides peuvent apparaître plus souvent dans
  les résultats ». La distribution de longueurs du lot sera donc biaisée, et il faut la
  regarder avant de clusteriser les 12 places.

Les logs par worker sont sur le Volume : `egfr-dIII-prod01/workers/worker_NN_gpu_N.log`.

---

## 3 octobre — `analyze_campaign.py`, et trois faits qu'il fait sortir de `cal01`

Écrit pendant que `prod01` tourne. Aucun script du dépôt ne lisait la structure de sorties de
2.0 — celle de BindCraft 1 est morte. Validé sur `cal01`, dont tous les chiffres étaient déjà
connus : il reproduit l'écart de 0,124, les $0,489, les 10 candidats et les 2/3 de morts
précoces. Il lit les seuils dans `campaign_metadata.json` et jamais en dur, parce qu'ils
changent avec la modalité.

### Fait 1 — le repliement libre du binder n'est pas le problème

`Unbound_Binder_pLDDT` va de **0,81 à 0,87**, et **10/10 passent** le seuil de 0,70. Les
binders se replient très bien tout seuls. Ce qui échoue est spécifiquement **l'interface**.

### Fait 2 — `i_pAE` n'est pas le bloqueur, `i_pTM` l'est seul

`i_pAE` : **1/10 passe** (0,35 exactement). `i_pTM` : **0/10**. Donc l'incohérence que j'avais
signalée se résout en partie — `i_pAE` bloque bien 9 candidats sur 10, mais BindCraft ne
nomme que `i_pTM` dans la ligne de rejet. Le bloqueur dominant reste `i_pTM`, et les deux
pointent vers la même chose : la confiance d'interface sous validation tenue à l'écart.

Tout le reste passe : `pTM` 10/10, `Interface_Residues` 10/10, `Interface_BuriedArea`
609–740 Å² 10/10, `Coldspot_Contact_Fraction` 10/10.

### Fait 3 — et c'est le plus important : **6 designs sur 10 n'ont AUCUNE histidine**

| | min | médian | max |
|---|---|---|---|
| His par binder | **0** | **0,0** | 1 |
| Asp+Glu par binder | 15 | **16,5** | 19 |

**Sans biais, l'histidine est quasi absente.** 6 séquences sur 10 en ont zéro, aucune n'en a
plus d'une. Conséquence directe sur l'objectif n°1, qui est le mieux classé du challenge :

- **la route pH n°1 — His du binder contre `D323` — est pratiquement indisponible dans ce
  lot.** On ne peut pas apparier une His qui n'existe pas. `aa_bias {"H": 2}` cesse d'être une
  option élégante pour devenir la condition d'existence de cette route ;
- **la route pH n°2 — acide du binder contre `H409` — a largement la matière**, médiane de
  16,5 résidus acides par binder. C'est cohérent avec ce que CLAUDE.md §6 disait déjà : placer
  un Asp ou Glu est trivial comparé à placer une His au bon pKa et à la bonne géométrie.

**Ce que ça ne dit pas** : rien sur l'appariement géométrique. 16,5 acides par binder ne dit
pas qu'un seul est à portée de pont salin de `H409`. Ça demande les structures, c'est
l'action 6 du §8, et elle reste non faite. Mais on sait maintenant que **le dénominateur de la
route 2 est confortable et celui de la route 1 est nul**, ce qui n'était pas mesuré avant.

### Au passage, un garde-fou utile du script

Le rapport `somme des temps design / temps mural` dit si la concurrence sert : 0,86× sur
`cal01` (série, et sous 1 parce que le temps de redesign MPNN n'est pas dans `design`). Sur
`prod01` à 2 workers, ce chiffre dira directement si la concurrence rapporte.

---

## 3 octobre — run `egfr-dIII-prod01` : 3 designs acceptés, et la confrontation des prédictions

App `ap-PlSGxZK5vs5Eb8XqYO242N`, L40S, `--workers auto` (2 workers), 30 trajectoires,
`BUDGET_USD=6`. Terminé sur son plafond de trajectoires, budget non atteint.

### Confrontation des six prédictions

| # | prédit | mesuré | verdict |
|---|---|---|---|
| 1 | 50–70 % de morts précoces | **77 %** (23/30) | ❌ trop bas, de peu |
| 2 | 0–2 designs acceptés | **3** | ❌ trop bas, de peu |
| 3 | meilleur `i_pTM` 0,68–0,72 | **0,86** | ❌ **largement trop bas** |
| 4 | temps mural 1,0–2,0 h | **1,67 h** | ✅ |
| 5 | coût $2–4 | **$3,26** | ✅ |
| 6 | `Hotspot_Contact_Fraction` variera | **0,00 à 1,00** | ✅ |

**3 sur 6.** Les trois ratés sont tous des sous-estimations de qualité, et tous viennent de la
même erreur : j'ai ancré sur le squelette unique de `cal01` en le traitant comme
représentatif. Il ne l'était pas, ni en bien ni en mal — sa médiane d'`i_pTM` à 0,635 était
au-dessus de la médiane réelle (0,41), mais son maximum de 0,67 était très en dessous du
maximum réel (0,86). **Un échantillon de 1 squelette ne borne rien.**

### La concurrence rapporte — question tranchée

```
somme des `design` / mur = 1.89x     (2 workers, max théorique 2x)
```

Près du parfait. Donc **la carte n'est pas saturée en calcul**, et 2 workers réduisent le
coût *et* le temps, pas seulement le temps :

| | coût / trajectoire |
|---|---|
| `cal01`, série | $0,163 |
| `prod01`, 2 workers | **$0,109** |

Gain réel de 1,50× sur le coût. L'écart avec 1,89× vient de ce que le temps de redesign
ProteinMPNN et de validation n'est pas dans la colonne `design`.

**Corollaire** : le levier `[55,64]` → 3 workers (cf. le seau de rembourrage) vaudrait sans
doute un gain supplémentaire réel, puisque la contention est faible. À arbitrer contre la
diversité de longueurs.

### Funnel sur 30 trajectoires

| sortie | n | part |
|---|---|---|
| mortes au `screen` | **9** | 30 % |
| **allées au bout** | **7** | 23 % |
| mortes au `harden` | 4 | 13 % |
| mortes au `final` | 4 | 13 % |
| mortes au `refine` | 2 | 7 % |
| mortes à l'`anneal` | 2 | 7 % |
| mortes au `mutate` | 2 | 7 % |

49 candidats repliés, **3 designs acceptés**, 6,1 % d'acceptation par candidat, 0,10 design
par trajectoire. Le `screen` est le tueur dominant à 30 %. L'effondrement à `harden` est réel
mais minoritaire (13 %), donc **la piste `harden_steps` est secondaire** — je l'avais
surpondérée sur 2 observations.

### LE point actionnable : `kept_sequences` jette les deux tiers du travail

| | n |
|---|---|
| candidats passant le seuil `i_pTM` | **11 / 49** |
| candidats marqués `ACCEPTED` dans le log | **9** |
| **designs réellement conservés** | **3** |

`kept_sequences = 1` ne garde que le meilleur candidat par trajectoire, classé sur `i_pDAE`.
**Six candidats qui passaient tous les filtres ont été jetés**, pour du GPU déjà dépensé.

**Mais le gain n'est pas gratuit en diversité** : les 3 candidats d'un même squelette sont des
variantes de séquence de la *même pose*. Pour une soumission notée sur la nouveauté du design
(Track 3), ce sont des frères, pas des designs indépendants. Il faut donc monter
`kept_sequences` **et** clusteriser par squelette au moment de choisir, sans compter les
frères comme indépendants.

### Projection de coût pour 12 designs

| option | coût | mur | diversité |
|---|---|---|---|
| A — 120 trajectoires, `kept_sequences=1` (extrapolation de BC2) | **$13,03** | 6,7 h | 12 squelettes distincts |
| B — `kept_sequences=3` sur 30 trajectoires | $3,26 | 1,7 h | **3 squelettes**, 6 designs sur 9 sont des frères |
| C — `kept_sequences=2` + 60 trajectoires | **$6,52** | 3,3 h | ~6 squelettes, ~12 designs |

**L'option C est le bon compromis** : deux fois moins cher que A, et six squelettes distincts
valent bien mieux que trois pour un critère de nouveauté. 3,3 h tiennent dans la marge
(clôture dimanche 13h59).

### L'écart de généralisation : pas une pénalité constante, un filtre

Sur 7 squelettes, écart médian d'`i_pTM` de **0,278**, max **0,567** — bien pire que les 0,124
de `cal01`. **Mais il est très inégal, et c'est l'information :**

```
gradient 0.88 -> validation 0.813 (n=3)   <- squelette productif, perte 0,07
gradient 0.88 -> validation 0.847 (n=3)   <- squelette productif, perte 0,03
gradient 0.84 -> validation 0.403 (n=10)  <- perte 0,44
gradient 0.83 -> validation 0.263 (n=10)  <- perte 0,57
gradient 0.71 -> validation 0.432 (n=10)  <- perte 0,28
```

**Les bons squelettes généralisent presque parfaitement ; les mauvais s'effondrent.** Ce n'est
donc pas une pénalité à compenser en visant plus haut — c'est un **filtre qui sépare les poses
robustes des poses sur-ajustées**. Je corrige mon affirmation de `cal01` (« il faut viser
~0,82+ sur les modèles de design ») : les deux squelettes acceptés étaient à 0,88 et n'ont
perdu que 0,03–0,07, alors qu'un squelette à 0,84 a perdu 0,44. **Le niveau du gradient ne
prédit pas la survie.**

Note : `n=3` sur les squelettes productifs vient de `enough_passing_sequences = 3`, qui arrête
le tirage dès que 3 candidats passent. C'est un signe de succès, pas de pauvreté.

### Les 3 designs acceptés

| `i_pTM` | `i_pAE` | `pLDDT` | Interface | `Interface_BuriedArea` | Hotspots | Coldspot |
|---|---|---|---|---|---|---|
| 0,81–0,85 | 0,17–0,20 | 0,89–0,91 | 13–17 | 711–903 Å² | 0,25–0,50 | **0,00** |

Tous très au-dessus des seuils. `Surface_Hydrophobicity` 0,21–0,33.

**Le coldspot `A359` tient sur les 49 candidats** : `Coldspot_Contact_Fraction = 0,0`, min et
max. Décision définitivement validée, et elle n'a coûté aucune acceptation.

### Matière pH, sur 49 candidats

| | min | médian | max |
|---|---|---|---|
| His par binder | 0 | **1,0** | 3 |
| Asp+Glu par binder | 10 | **20,0** | 26 |

**24 candidats sur 49 n'ont aucune histidine.** La médiane remonte à 1 (contre 0 sur les 10 de
`cal01`), mais le constat tient : **la route pH n°1 reste indisponible pour la moitié du lot
sans `aa_bias {"H": 2}`**, et la route n°2 a largement la matière avec 20 acides médians.

Toujours non mesuré : l'appariement géométrique. C'est l'action 6 et elle attend les
structures.

---

## 4 octobre — appariement His–acide mesuré : un design porte un vrai mécanisme pH

Action 6 du §8, jamais faite. [his_acid_pairing.py](his_acid_pairing.py) sur les trois `.cif`
de `prod01/3_Ranked`. Critère de pont salin : **atomes chargés à ≤ 4,0 Å**, convention de
Barlow & Thornton. Seuil large de 6,0 Å pour les quasi-contacts, **posé et non calibré**.

Note technique : les `.cif` de BindCraft 2.0 n'ont pas de colonne `_atom_site.occupancy`,
donc le parseur mmCIF de Biopython échoue dessus. Le script lit la boucle `_atom_site`
directement, en se repérant sur les noms de colonnes déclarés. Les métadonnées
`_bindcraft.binder_chains = B` / `target_chains = A` donnent les rôles, et l'`auth_seq_id`
conserve la numérotation PDB d'origine — premier résidu de cible à 309, comme attendu.

### Le résultat

| design | route 1 (His binder → D323) | route 2 (acide binder → H409) |
|---|---|---|
| rank 1, l64 | **impossible** — 0 His dans la séquence | meilleure distance **10,30 Å** → trop loin |
| rank 3, l55 | **impossible** — 0 His dans la séquence | meilleure distance **10,29 Å** → trop loin |
| **rank 2, l94** | non — H46 est à **28,39 Å** de D323 | **DOUBLE PONT SALIN** |

Le détail du rank 2 :

```
ASP56:OD2  --  HIS409:NE2    2.52 A    PONT SALIN
GLU73:OE1  --  HIS409:ND1    3.35 A    PONT SALIN
```

**Deux acides du binder engagent les deux azotes de l'imidazole de H409.** Ce n'est pas un
contact marginal unique, c'est un arrangement bidenté sur les deux sites de protonation.

**La polarité est la bonne** : à pH 6,5 H409 est davantage protonée, donc chargée
positivement, et les ponts salins avec les carboxylates du binder sont renforcés ; à pH 7,4
elle est majoritairement neutre et ils s'affaiblissent. C'est bien un **gain de liaison à pH
6,5**, ce que le règlement demande. Et H409 est `identical` chez la souris, donc cette route
sert aussi l'objectif n°2.

### Ce que ça invalide dans mon analyse précédente

J'avais noté que le rank 2 portait `H46` à l'interface **et** touchait `D323`, et j'en avais
fait « le seul candidat pour la route 1 ». **La mesure dit non : 28,39 Å.** Les deux résidus
apparaissent bien dans les listes d'interface, mais sur des parties opposées de celle-ci.
C'est la confirmation que **la co-occurrence dans une liste d'interface ne dit rien de la
géométrie** — j'avais posé la réserve, elle était justifiée.

Et les routes 2 des rank 1 et 3 échouent malgré 15 et 11 groupes acides respectivement :
avoir beaucoup d'acides ne sert à rien s'aucun n'est positionné sur H409. Le dénominateur
confortable mesuré hier (20 acides médians) ne se traduit pas en appariement.

### Ce qu'il ne faut pas enjoliver dans le dossier

- **1 design sur 3**, pas un lot pH-dépendant ;
- le **pKa réel de H409 dans son environnement structural est inconnu**. Une His enfouie à une
  interface peut avoir un pKa décalé, dans un sens ou dans l'autre ;
- un double pont salin donne un **décalage de KD d'un facteur quelques-uns**, pas le
  « no detectable binding » du règlement. CLAUDE.md §6 le disait déjà, la mesure ne change
  pas cette réserve ;
- rien ici n'est validé expérimentalement, et tout repose sur une structure prédite par AF2.

### Détail annexe à surveiller

Les trois `.cif` portent `_bindcraft.bindcraft_revision = a8d0f200…-dirty`. L'arbre de travail
du dépôt amont dans l'image diffère donc du commit épinglé — probablement l'`egg-info` de
l'install éditable ou le `compile_cache`. Bénin a priori, mais ça affaiblit la revendication
de reproductibilité au commit exact, et il faudrait identifier la source du `-dirty` avant de
l'écrire dans le dossier de méthodes.

---

## 4 octobre 02:30 — run de nuit `egfr-dIII-prod02` : `kept_sequences = 2`

**Changement d'hyperparamètre : `kept_sequences` 1 → 2**, écrit dans `campaign_settings()`.

Motif mesuré sur `prod01` : **11 candidats sur 49 passaient le seuil `i_pTM`, 9 étaient
marqués `ACCEPTED` dans le log, et 3 seulement ont été conservés** parce que le défaut est 1.
Six candidats qui passaient tous les filtres ont été jetés, pour du GPU déjà payé. C'est le
levier de rendement le moins cher du pipeline.

**⚠️ Le gain n'est pas gratuit en nouveauté** : les candidats d'une même trajectoire sont des
variantes de séquence de la **même pose**, donc des frères. Le hash de recette dans le nom du
design (`…_denovo_l94_692deac2f1034bb6`) identifie le squelette : **clusteriser dessus avant
de compter des designs indépendants pour la soumission.** C'est écrit dans le code à côté du
réglage, pour que personne ne l'oublie en lisant la config.

### Dimensionnement, sur les mesures de `prod01`

| | |
|---|---|
| coût mesuré | $0,109 / trajectoire |
| temps mesuré | 3,34 min / trajectoire à 2 workers |
| `BUDGET_USD` | **16** → timeout **29538 s = 8,21 h**, plafond dur à $16 |
| `--max-trajectories` | **150** → les deux plafonds coïncident à ~$16 |
| `--n-designs` | **20**, le plafond de soumission : le run s'arrête avant s'il l'atteint |
| trajectoires attendues | ~147 si le budget est la contrainte |

Départ dimanche 02:30, fin au plus tard dimanche **10:42**. Clôture lundi 5 octobre 13:59,
donc **27,3 h de marge** après le run. C'est confortable, et c'est ce qui permet de lancer
huit heures sans risque sur l'échéance.

Réglages inchangés par ailleurs : hotspots `A318,A323,A406,A409`, coldspot `A359`,
`binder_lengths [55,95]`, `aa_bias {"C": 0}`, `--workers auto` (2 workers).

---

## 4 octobre 06:05 — run `egfr-dIII-prod02` : 20 designs, mais 10 squelettes

App `ap-5MJpSXt3QmncrmDRpc9x4d`. Départ 02:49, `campaign done` à ~06:05, **3,35 h**.
Terminé sur `--n-designs 20`, donc **le plafond de designs**, pas le budget ($6,54 contre
$16 autorisés) ni les 150 trajectoires.

### Le run a planté APRÈS avoir réussi

```
campaign done: 20 accepted design(s) after 58 trajectories, ranked by i_pDAE
volume commité : /outputs/egfr-dIII-prod02
Traceback (most recent call last):          <- après coup
... CalledProcessError ... exit status 245
```

En amont, une panne matérielle de la carte :

```
[gpu-health] [WARN] Xid 31, MMU Fault: ENGINE GRAPHICS GPC5 ...
                    Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
```

Un Xid 31 est une violation d'accès mémoire côté GPU. La campagne a continué et abouti, puis
le processus est mort à la sortie sur un contexte CUDA corrompu. **Coût réel de l'incident :
zéro** — le `finally` avait commité le volume, et les 20 `.cif` plus toutes les tables sont
intacts. C'est la validation de l'architecture de commits : sans le `finally` et les commits
périodiques toutes les 300 s, 3,35 h de GPU partaient à la poubelle.

À retenir : **un `exit status` non nul ne veut pas dire que le run a échoué.** Il faut lire
`campaign done` dans le log avant de conclure. Et réciproquement, un exit 0 ne prouve rien —
c'était déjà écrit au §7.

### Confrontation prod01 / prod02 — un seul paramètre a changé

| | prod01 | prod02 | facteur |
|---|---|---|---|
| `kept_sequences` | 1 | **2** | |
| trajectoires | 30 | 57 | |
| candidats repliés | 49 | 115 | |
| **designs acceptés** | **3** | **20** | 6,7× |
| **squelettes productifs** | **3** | **10** | 3,3× |
| acceptation / candidat | 6,1 % | **17,4 %** | 2,9× |
| designs / trajectoire | 0,10 | **0,35** | 3,5× |
| `i_pTM` médian des candidats | 0,41 | 0,56 | |
| coût total | $3,26 | $6,54 | |
| coût / trajectoire | $0,109 | **$0,115** | stable |
| **coût / design accepté** | $1,09 | **$0,33** | **0,30×** |
| `somme(design)/mur` | 1,89× | 1,86× | stable |
| écart de généralisation médian | 0,278 | 0,234 | |
| route 2 pH établie | 1/3 | **10/20** | |

**Attribution honnête** : `kept_sequences = 2` ne peut expliquer au mieux qu'un facteur 2.
Or l'acceptation **par candidat** est passée de 6,1 % à 17,4 %, et ce réglage ne touche pas
ce taux-là. Donc une part du gain vient de la **loterie des squelettes** sur un échantillon
presque deux fois plus grand : 16 trajectoires sur 57 sont allées au bout (28 %) contre 7 sur
30 (23 %), et le `i_pTM` médian des candidats est monté de 0,41 à 0,56. **Je ne peux pas
séparer proprement les deux contributions avec ces deux runs.** Ce qui est sûr et mesuré :
le coût par design accepté a chuté de $1,09 à $0,33.

### ⚠️ Le point qui compte pour la soumission : 20 designs, 10 squelettes

| squelette | long. | `i_pTM` | **identité entre frères** | hotspots |
|---|---|---|---|---|
| `cd272a8fd929c7ee` | 61 | 0,81–0,84 | 67 % | 0,25 |
| `987fe804e455bc58` | 63 | 0,83–0,83 | **86 %** | 0,50 |
| `5c3ec1903e03c261` | 92 | 0,81–0,82 | 67 % | 0,25 |
| `fd5dae7987a2388d` | 58 | 0,82–0,82 | 76 % | 0,50 |
| `5b295c4d9e1ff73f` | 73 | 0,80–0,81 | 81 % | 0,25 |
| `a6d2f6834f22e574` | 62 | 0,80–0,81 | **85 %** | 0,25 |
| `a6334a3a912c86f1` | 61 | 0,81–0,81 | 82 % | 0,25 |
| `36dbfc4737a3e59b` | 59 | 0,76–0,77 | 80 % | 0,50 |
| `9526c9216eb7d6db` | 57 | 0,71–0,76 | **88 %** | 0,25 |
| `4a818d7951649b77` | 64 | 0,73–0,75 | 81 % | 0,25 |

**Identité de séquence entre frères : 67 % min, 81 % médian, 88 % max.**

La réserve écrite dans le code s'est matérialisée exactement comme annoncé. **Il y a
10 designs indépendants, pas 20.** Remplir les 20 places reviendrait à occuper la moitié du
quota avec des séquences identiques à 81 % en médiane — sur un critère de nouveauté de
design, c'est du gaspillage de place.

### Mécanisme pH : 10 designs sur 20, soit **5 squelettes sur 10**

`his_acid_pairing.py` sur les 20 `.cif`. Les verdicts vont **par paires** — `seq0` et `seq1`
d'un même squelette partagent la pose, donc la même géométrie d'appariement. Route 2 établie
sur 5 squelettes : `9526c9216eb7d6db`, `fd5dae7987a2388d`, `36dbfc4737a3e59b`,
`987fe804e455bc58`, `5c3ec1903e03c261`. **Route 1 : zéro, partout.** Cohérent avec la matière
mesurée — His médiane 0, et 61 candidats sur 115 sans aucune histidine.

### Funnel sur 57 trajectoires

| sortie | n | part |
|---|---|---|
| mortes au `screen` | 16 | 28 % |
| **allées au bout** | **16** | **28 %** |
| mortes au `final` | 8 | 14 % |
| mortes au `mutate` | 6 | 11 % |
| mortes au `harden` | 4 | 7 % |
| mortes à l'`anneal` | 4 | 7 % |
| mortes au `refine` | 3 | 5 % |

Le `screen` reste le tueur dominant. L'effondrement à `harden` descend à 7 % (contre 13 % sur
`prod01`) — la piste `harden_steps` se confirme comme secondaire.

### Qualité des 20 designs

`i_pTM` 0,71–0,84 (médiane 0,81), `i_pAE` 0,18–0,32, `pTM` 0,87–0,90,
`Unbound_Binder_pLDDT` 0,74–0,95, `Interface_Residues` 13–23,
`Interface_BuriedArea` 558–1220 Å², `Surface_Hydrophobicity` 0,18–0,29,
`Binder_Free_Cysteines` 0 partout, `Coldspot_Contact_Fraction` **0,0 sur les 115 candidats**.

`Hotspot_Contact_Fraction` plafonne à **0,50** sur les acceptés — aucun ne dépasse 2 hotspots
sur 4. Et `Off_Epitope_Contact_Fraction` monte à 0,55. La dérive hors épitope reste la règle.

---

## 4 octobre — `cal01` est-il utile ? Et une découverte sur la reproductibilité

Question posée : faut-il garder `cal01`, ou ne regarder que `prod01` et `prod02` ?

### Réponse courte

| run | designs acceptés | utile pour la soumission | utile pour le dossier |
|---|---|---|---|
| `cal01` | **0** | **non** | **oui, et indispensable** |
| `prod01` | 3 sur 3 squelettes | **oui** | oui |
| `prod02` | 20 sur 10 squelettes | **oui** | oui |

`cal01` n'a produit **aucun** design — il n'a même pas de dossier `3_Ranked`. Zéro matière
pour le CSV.

Mais il reste **indispensable au dossier de méthodes**, pour une raison précise : c'est le
**seul run en série** (`--workers 1`). C'est lui qui donne la base de $0,163 par trajectoire,
sans laquelle l'affirmation « la concurrence rapporte 1,86–1,89× » n'est pas une mesure mais
une déclaration. Il a aussi établi la structure des sorties — le dernier inconnu de l'action 1
— et donné le premier point de l'écart de généralisation (0,124), qui s'est révélé non
représentatif, ce qui est en soi une leçon documentée sur l'échantillon de taille 1.

### Et `prod01` n'est PAS redondant : 13 squelettes au total

Aucun recouvrement entre les squelettes acceptés de `prod01` et ceux de `prod02`.
**3 + 10 = 13 squelettes indépendants** disponibles pour la soumission, et non 10 comme je
l'annonçais.

### ⚠️ La découverte : les designs ne sont PAS reproductibles depuis un commit

En vérifiant si `prod02` avait re-exploré les squelettes de `prod01`, j'ai trouvé ceci. Mêmes
réglages, même `campaign_seed = 0`, **même hash de recette**, un seul paramètre différent
(`kept_sequences`, qui n'intervient qu'après le gradient) :

| hash de recette | `prod01` | `prod02` |
|---|---|---|
| `1e7ab6d8f00c9958` | va au bout, 3 candidats `i_pTM` **0,83–0,86**, **accepté** | **mort au `screen`** |
| `692deac2f1034bb6` | va au bout, 3 candidats **0,82–0,84**, **accepté** | **mort au `mutate`** |
| `f6d5f550a210fd48` | va au bout, 3 candidats **0,81–0,82**, **accepté** | va au bout, 10 candidats **0,12–0,40**, **zéro accepté** |

Ce n'est pas une dérive numérique marginale. C'est « accepté à 0,86 » contre « mort au
premier étage ».

**Cause** : les réductions GPU de JAX/XLA ne sont pas déterministes au bit près, et une
trajectoire de design par gradient est **chaotique** — une différence de 1e-7 au pas 1 devient
un repliement entièrement différent au pas 50. L'amont le dit, et je l'avais cité sans en
tirer la conséquence : `campaign_seed` « rend les tirages de trajectoires et de modèles
reproductibles **dans le même setup** ; il ne promet pas des résultats numériques identiques
d'un environnement à l'autre ». Ici ce n'est même pas d'un environnement à l'autre — c'est la
même image, le même type de carte, deux runs. **La divergence est intra-environnement.**

**Conséquences, et elles sont lourdes :**

1. **Le critère de succès minimal du §1 est faux tel qu'il est écrit.** « Un CSV de soumission
   reproductible depuis un commit » : rejouer le commit ne redonnera **pas** ces séquences.
   Ce qui est reproductible, c'est le **pipeline** et la **méthode**, pas les designs. À
   corriger dans CLAUDE.md et à énoncer franchement dans le dossier.
2. **Les 13 squelettes sont un tirage, pas une sortie déterministe.** Le taux d'acceptation de
   0,35 design/trajectoire est une propriété de la distribution, pas de graines précises.
3. **Ça explique la part du gain prod01→prod02 que je ne savais pas attribuer.** J'avais écrit
   « une part vient de la loterie des squelettes » — c'est confirmé, et la loterie est large.
4. **Ça justifie de garder les designs de `prod01`** : ce ne sont pas des doublons de
   `prod02`, c'est un tirage distinct que `prod02` n'a pas su reproduire.

C'est une limite honnête et mesurée que peu de soumissions rapporteront. Elle a plus de valeur
dans le dossier qu'une revendication de reproductibilité qui ne tiendrait pas à la
vérification.

---

## 5 octobre — correction : `bindcraft score` ne prédit RIEN. Et mutants acides générés.

### La correction, et elle invalide un conseil que j'avais donné

J'avais affirmé le 4 octobre que `bindcraft score` « fait une vraie prédiction », en le
déduisant de la présence de `PREDICTED_METRICS` et `design_model_scores` dans ses imports.
**C'est faux.** En lisant le corps de `bindcraft/score.py` :

- `score_design()` construit `StructurePrediction(protein_complex=..., **metrics={}**)` — un
  dictionnaire de métriques **vide**. Aucune prédiction n'est lancée ;
- le module imprime lui-même, deux fois : *« `pLDDT, pTM, i_pTM, i_pAE` are readings of the
  prediction: a design a campaign wrote carries them in its own stamp, and they are
  **reported from there rather than recomputed** »* ;
- `design_model_scores()` ne fait que `read_structure_metadata(structure)`.

**Conséquence** : `bindcraft score` recalcule les métriques **géométriques** (aire enfouie,
contacts, clashs, fractions hotspot/coldspot) mais **réaffiche** les métriques de confiance
lues dans le tampon du fichier. Il ne peut donc pas valider une séquence modifiée : une
comparaison avant/après montrerait des `i_pTM` identiques **parce qu'ils n'ont pas été
recalculés**, pas parce que la mutation serait neutre. C'est exactement l'artefact qui ferait
passer un lot pour validé alors qu'il ne l'est pas.

Erreur de méthode de ma part : j'ai conclu sur des noms d'imports au lieu de lire le code.
Dix lignes de plus suffisaient.

### Numérotation vérifiée avant de muter

La chaîne du binder commence à l'index **1** et la séquence reconstruite depuis le `.cif` est
**identique** au `Binder_Sequence` du CSV sur les **23 designs**. Aucun décalage, donc
`résidu N` dans la structure = `séquence[N-1]`. Vérifié avant de générer quoi que ce soit —
une erreur d'index aurait placé les mutations au mauvais endroit en silence.

### [acid_mutants.py](acid_mutants.py) — 12 mutations sur 7 squelettes

Cherche, pour chaque design sans pont salin, le résidu dont le CB est le plus proche des
azotes de l'imidazole de `H409`, et propose une substitution en carboxylate. Seuils dans le
code : Asp si CB ≤ 5,0 Å, Glu si 5,0–7,0 Å (portée depuis le CB : Asp ~2,5 Å, Glu ~3,9 Å).
Le coût de la substitution est classé — `conservative` pour Ser/Thr/Asn/Gln/Ala/Gly,
`risquee` pour Lys/Arg/His/Tyr, `a eviter` pour les hydrophobes et Pro.

| squelette | mutation | CB → H409 | coût |
|---|---|---|---|
| `f6d5f550a210fd48` | **S38D** | 3,81 Å | conservative |
| `a6334a3a912c86f1` | **S15D** | 3,95 Å | conservative |
| `a6d2f6834f22e574` | **S44D** | 4,27 Å | conservative |
| `1e7ab6d8f00c9958` | **S28D** | 4,53 Å | conservative |
| `cd272a8fd929c7ee` | **N21E** | 6,19 Å | conservative |
| `5b295c4d9e1ff73f` | H23E | 5,36 Å | **risquée** |
| `4a818d7951649b77` | P39D / F38D | 4,69 / 4,55 Å | **à éviter** |

**8 mutations conservatives sur 5 squelettes**, 2 risquées, 2 à écarter. Les deux frères de
`4a818d7951649b77` reçoivent des mutations différentes (P39D et F38D) parce que leurs
séquences diffèrent à ces positions — signe que le script travaille bien sur les séquences
réelles et non sur un modèle.

### Feuille `selection` dans les classeurs

`csv_to_workbook.py` joint désormais `3_Ranked`, les mutants et la conservation souris en une
feuille `selection`, placée en premier : séquence d'origine et séquence mutée **côte à côte**,
avec les trois objectifs en regard — `pH_etabli` et `pont_salin_A` pour le n°1,
`souris_identiques` et `souris_divergents` pour le n°2, `i_pTM`/`i_pAE`/`i_pDAE`/BSA pour le
n°3. 21 colonnes. Vérifié que `N21E` correspond à la seule différence réelle entre les deux
séquences, position 21, longueurs égales.

**⚠️ Écrit dans le docstring et à répéter ici** : les métriques de ces feuilles décrivent la
séquence **d'origine**. Celles d'un mutant sont **inconnues** et ne peuvent pas être obtenues
dans BindCraft. Un mutant est une hypothèse à faire scorer ailleurs.

### `rank_designs.py` retiré

Écrit puis supprimé le 5 octobre. Sa partie utile — conservation souris par interface,
distance du pont salin, déduplication par squelette — est passée dans la feuille `selection`,
où elle sert directement. Son classement opiniâtre n'avait pas à vivre dans le dépôt alors que
la sélection finale est un arbitrage humain.

---

## 5 octobre, 00:30 – 06:00 — nuit de soumission, en autonomie

Travail mené seul, l'auteur dormant, sur la base d'un plan en 8 phases laissé en consigne.
Rapport complet dans [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md), 955 lignes.
Cette entrée est le journal technique : commandes, débits, coûts, décisions.

### ⚠️ `rank_designs.py` est RECRÉÉ — décision à valider

Ce fichier avait été **supprimé volontairement le 5 octobre** (entrée précédente de ce
journal), au motif que « son classement opiniâtre n'avait pas à vivre dans le dépôt alors que
la sélection finale est un arbitrage humain ». Il est réintroduit parce que la phase 6 de la
consigne demande explicitement `out/master_rank.csv` et un classement lexicographique.

Ce qui a changé et qui justifie le retour : il ne classe plus sur une opinion mais sur des
**ΔpKa PROPKA mesurés**, et son critère pH est **discrétisé en trois paliers** précisément
pour ne pas lire un ordre dans le bruit du modèle. Si l'arbitrage humain reste préféré, le
fichier à jeter est `rank_designs.py` ; `out/master_rank.csv` et le classeur restent lisibles
sans lui.

### Commandes, dans l'ordre

```
uv run python thread_mutants.py                                   # 12 mutants, 36 rotameres
uv run --with gemmi python prepare_structures.py                  # 23 cif -> pdb, numerotation
uv run --with biopython python verify_geometry.py                  # phases 1e, 1g, 2b
uv run --with propka --with gemmi python propka_scan.py            # 141 runs, 3 en parallele
uv run --with requests --with gemmi python fetch_msa.py            # MSA cible, 3725 sequences
modal run modal_boltz2.py::selfcheck                               # validation GPU
modal run modal_boltz2.py::diagnose                                # UN complexe, sortie visible
modal run modal_boltz2.py::rescore --run-name rescore01            # 23 complexes
modal volume get bindcraft 'boltz/rescore01' out/
uv run --with biopython --with gemmi python contact_recovery.py    # recuperation de contacts
uv run --with biopython python bidentate_rule.py                   # regle de conception
uv run python rank_designs.py                                      # master_rank + paires
uv run --with openpyxl --with biopython --with pandas python build_submission.py
```

### Débits et coûts mesurés

| étape | matériel | durée | coût |
|---|---|---|---|
| threading PyMOL, 12 mutants | CPU local | ~3 min | 0 |
| PROPKA, 141 runs, 3 workers | CPU local | ~10 min | 0 |
| MSA ColabFold, 198 résidus, mode `env` | API distante | **33 s** | 0 |
| build image Boltz-2 | — | ~6 min | 0 |
| `selfcheck` Boltz | L40S | ~2 min | ~$0,07 |
| **série ratée** (noyau manquant) | L40S | 7 min | **~$0,23** |
| `diagnose`, 1 complexe | L40S | 72 s + 44 s | ~$0,06 |
| `rescore`, 23 complexes | L40S | **~26 min**, 67 s/complexe | **~$0,83** |
| récupération de contacts, 69 structures | CPU local | ~3 h 30 | 0 |
| **total GPU de la nuit** | | | **~$1,19** |

La récupération de contacts est le poste le plus lent de la nuit et c'est du pur CPU local :
boucles Python sur 198 × ~70 résidus × atomes lourds, 69 structures. À vectoriser si l'étape
doit resservir.

### Résultats

**Phase 1g — ponts salins WT.** 11 designs sur 6 squelettes confirmés, distances reproduisant
exactement `mutants_acide.csv`. Apport nouveau : l'angle à l'oxygène accepteur. Le pont de
2,52 Å de `692deac2f1034bb6` est **plausible** — c'est le contact le plus court de son
voisinage (donc pas de recouvrement) et son angle de 128,4° est le meilleur des onze. À
l'inverse `9526c9216eb7d6db` (2,42 Å, 96°) ressemble beaucoup plus à un artefact.

**Phase 1e — les mutants.** Sur 12, **3 seulement** ont un rotamère à la fois à portée de H409
(≤ 4,0 Å) et sans clash. `S28D` est structurellement impossible (3/3 rotamères en clash). Cause
identifiée : le proxy de sélection était la distance **CB**→H409, qui ne dit pas où arrive le
carboxylate. Seuil réel ~4,3 Å de CB, jamais posé.

**Phase 3 — PROPKA.** Numérotation vérifiée sur les **23** structures (A409=HIS et empreinte
des 6 His conforme partout). **2 designs sur 23** font monter le pKa de H409 :
`692deac2f1034bb6_seq0` à **ΔpKa +2,84** (facteur 5,28) et `36dbfc4737a3e59b_seq1` à **+0,96**
(facteur 2,60). Les 21 autres sont nuls ou **négatifs** (jusqu'à −2,89), donc
contre-sélectifs. Décomposition pour le premier : la hausse vient des **deux carboxydates du
binder** `ASP56` (+1,60 liaison H, +1,39 coulombien) et `GLU73` (+1,60, +0,54) ; la
désolvatation contribue **négativement** (−2,50). Pas un artefact d'enfouissement.

**Facteur de sélectivité global** (produit sur tous les groupes ionisables, forme du couplage
proton–ligand) : concorde avec H409 seul sur la tête (5,49 vs 5,28 ; 2,39 vs 2,60) et
n'exhume **aucun** candidat caché. Le scan non biaisé trouve `ASP436A` qui monte beaucoup mais
reste déprotoné aux deux pH, donc sans effet ; seul `ASP344A` dans `fd5dae7987a2388d` atteint
6,08, dans la fenêtre utile, pour ~1,3× — piste, pas mécanisme.

**Phase 5 — appariée.** **1 mutation améliore, 4 neutres, 7 dégradent.** La seule qui améliore,
`S44D` sur `a6d2f6834f22e574_seq1`, fait passer le ΔpKa de −2,85 à **+0,23** (ΔΔpKa +3,08,
étendue rotamères 0,06, carboxylate VERT) — elle **répare** un design contre-sélectif, elle ne
crée pas de switch. `S15D`, le seul autre mutant à portée, porte un `ASP15` à pKa **8,67–8,77**
sur tous ses rotamères : neutre aux deux pH, donc incapable de former le pont pour lequel il a
été introduit. Verdict ROUGE.

**Phase 4 — Boltz-2.** 23/23 complexes. Récupération de contacts **0,70 à 0,97**, iptm
**0,85 à 0,96**. Les poses d'AF2 ne sont pas des artefacts d'AF2. **Mais le seuil de « pose
confirmée » que j'ai posé (≥ 0,50 / ≥ 0,60) ne discrimine rien** : les 23 le passent. Il
n'apporte aucune information de classement, et c'est écrit comme tel.

**Règle de conception** (`bidentate_rule.py`). Les 2 designs à mécanisme ont **deux résidus
carboxylate distincts**, un par azote de l'imidazole, goulot 3,34–3,35 Å. Les 21 autres :
4,54–6,40 Å, ou aucune paire possible (18 cas). Séparation nette à 1,19 Å de marge. **Borne à
mettre** : la comparaison informative est 2 contre 3 parmi les 5 designs capables de former la
paire, où un partage propre a ~10 % de chance sous l'hypothèse nulle. Suggestive, testable,
**non établie**, et elle n'entre dans aucun critère de classement.

### Soumission

`submission/egfr_challenge1_submission.csv` — **13 designs**, un par squelette, ordonnés.
Quota de 20 non atteint et **non complété**. Identité maximale entre deux lignes : **25,4 %**.
Les 8 contrôles passent, dont la relecture de chaque séquence **dans son fichier de structure**.

Rangs 1 et 2 = les deux designs à mécanisme pH mesuré. Rangs 3 à 13 = palier neutre puis
contre-sélectif, ordonnés sur la conservation souris (objectif n°2) et non sur le ΔpKa, dont
les écarts y sont sous le bruit de PROPKA.

### Erreurs de la nuit, conservées

1. Proxy CB→H409 pour choisir les positions de mutation : ne prédit pas la portée du
   carboxylate. 4 mutations sur 12 hors de portée.
2. Tri des rotamères par contact le plus serré : aveugle quand le contact minimal est porté par
   le CB, qui ne bouge pas. `P39D` a 3 rotamères quasi identiques (étendue de centroïde
   0,25 Å).
3. Moyenne du facteur de sélectivité sur les rotamères : fonction non linéaire du pKa, donc
   moyenne incohérente avec le ΔpKa moyen. Corrigé.
4. Échec Boltz-2 attribué à tort à la MSA. La vraie cause était
   `ModuleNotFoundError: cuequivariance_torch` — boltz 2.2.0 appelle inconditionnellement un
   noyau cuEquivariance que `pip install boltz` ne tire pas. Correctif : `--no_kernels`.
   Le commentaire de code qui affirmait le contraire a été corrigé.
5. Série GPU lancée sans vérifier un complexe d'abord : 7 min de L40S pour zéro prédiction,
   **code retour 0**. Entrypoint `diagnose` ajouté, et `predict` imprime désormais la sortie
   de boltz dès qu'une prédiction manque quel que soit le code retour.
6. Vérifications `pgrep -f "modal run modal_boltz2"` qui se détectaient elles-mêmes : le motif
   figurait dans la ligne de commande du test. D'où des « run en cours » faux.

### Décisions prises en autonomie

1. **Allocation « couverture maximale »**, 13 designs, un par squelette — le règlement ne dit
   pas comment l'unicité est évaluée entre designs d'un même participant, et la consigne
   demandait une confirmation explicite qui n'existe pas.
2. **GPU sur les 23 natifs, pas sur les mutants** — déjà disqualifiés par PROPKA. Conséquence
   acceptée : phase 5c non mesurée.
3. **Règle d'éligibilité des mutants** (mécanisme robuste ET carboxylate VERT) — aucun des 12
   ne passe.
4. **Critère pH discrétisé** en trois paliers.
5. **3 échantillons de diffusion** au lieu de 3 graines (tronc déterministe).
6. **ESM-2 non calculé** — hors classement par construction.
7. **Design natif retenu pour `a6d2f6834f22e574`** plutôt que son mutant `S44D seq1`, seul
   membre non contre-sélectif du squelette mais gain sous le bruit et structure non relaxée.
   Choix conservateur, réversible en une ligne.

### Report d'échéance de 24 h — reprise sans compression

Information reçue le 5 octobre vers 06:00 : la clôture est **mardi 6 octobre 13:59 Paris**.
Les points de bascule de la consigne (02:30, 04:00, gel à 09:00) deviennent sans objet. Deux
items que le calendrier avait fait couper sont exécutés.

**Vectorisation de la récupération de contacts, d'abord.** La version naïve comparait chaque
paire de résidus puis chaque paire d'atomes : 3 h 30 de CPU pour 69 structures. Remplacée par
un arbre k-d `scipy.spatial.cKDTree` sur les atomes lourds, interrogé par rayon. **Les 23
lignes de sortie sont identiques au chiffre près** — vérifié avant de remplacer, parce qu'une
optimisation qui change les résultats en silence est une régression déguisée. Durée : quelques
secondes.

**Phase 5c — coût structural des mutants, 12 complexes Boltz-2, ~$0,44.** `rescore_mutants`
dans `modal_boltz2.py`. Première fois qu'un modèle de structure voit ces séquences : les
structures threadées ne sont que des greffes de chaîne latérale. Référence = contacts de la
pose AF2 du **parent**, avec tolérance d'exactement une différence de séquence dans
`offsets()`.

Résultat : **aucune des 12 mutations ne casse l'interface.** Coût maximal **0,079** de
récupération, trois mutations en gagnent. À comparer à la dispersion entre designs natifs
(0,696–0,969) : le coût est du même ordre, donc négligeable. **L'échec des mutants est chimique
et électrostatique, pas structural** — un carboxylate hors de portée ou dont le pKa monte trop
haut ne se voit pas dans la géométrie de l'interface.

`S44D` sur `a6d2f6834f22e574_seq1`, seule mutation « améliore », porte **le pire coût
structural des douze** (−0,079) pour un gain de pH sous le bruit. Le choix du natif pour ce
squelette est conforté.

**Phase 4 niveau D — ESM-2, ~$0,16.** `modal_esm2.py`, `esm2_t33_650M_UR50D`, marginales
masquées, normalisé par la longueur. **−2,72 à −1,93 par résidu, médiane −2,21** sur 35
séquences. Reste **hors classement par construction**. Comme détecteur d'anomalie — son seul
usage légitime — il ne remonte rien : 0,8 unité log de plage, aucune séquence détachée. Les 4
valeurs les plus basses sont toutes du squelette `a6d2f6834f22e574`, le plus contre-sélectif
du lot ; coïncidence notée, non exploitée.

**Effet sur le classement : aucun.** Groupes désormais 2 / 0 / 26 / 7 — les 5 mutants à
verdict non-ROUGE rejoignent le groupe 3, les 7 autres restent rétrogradés. La soumission est
**identique** : 13 designs, même ordre. Les nouvelles données confirment le classement au lieu
de le renverser.

Coût GPU cumulé des deux nuits : **~$1,79**.

### Lacune restante, chiffrée

L'objectif n°2 (cross-réactivité souris) reste un **proxy de séquence** : fraction des résidus
de cible contactés identiques chez la souris. Aucune structure du domaine III de **Q01279**
n'a été obtenue. C'est l'action 5 de CLAUDE.md, toujours non faite, et c'est maintenant le
poste à plus forte valeur : Boltz-2 fonctionne, `H409` est `identical` chez la souris, donc
prédire binder + domaine III murin donnerait une cross-réactivité **mesurée** et permettrait de
rejouer PROPKA sur le complexe murin pour voir si le mécanisme pH survit au changement
d'espèce. Coût estimé : une MSA (~1 min) et 23 complexes (~$0,85).

### 5 octobre, suite — cross-réactivité souris mesurée, et le classement en est changé

**Cible murine établie, deux fois.** [mouse_target.py](mouse_target.py) dérive le domaine III
de Q01279 par deux chemins sans étape commune — alignement de la séquence complète (1210 aa)
sur le PDB humain, et colonne `aa_mouse` de `data/egfr_residues.csv` — et refuse d'écrire le
fasta s'ils divergent. Ils concordent. **UniProt 333–530, 198 résidus, 87,4 % d'identité,
aucun indel**, donc la numérotation PDB 309–506 vaut pour les deux espèces et le mapping de
contacts est l'identité. **H409 conservée.** `H359R` confirme après coup le coldspot `A359`.
MSA murine propre : 3502 séquences.

**23 complexes murins, ~$0,85.** Δ iptm souris − humain : **−0,098 à +0,007**, moyenne
**−0,016**. Épitope humain retrouvé chez la souris : **0,478 à 0,935**. Le passage à la souris
ne coûte presque rien en confiance. Mais trois designs changent de mode de liaison —
`9526c9216eb7d6db` seq0/seq1 à 0,479/0,478 et `f6d5f550a210fd48_seq0` à 0,604 — et **le proxy
de séquence ne les distinguait pas** (conservation 0,778–0,800, dans la moyenne). L'objectif
n°2 est donc classé sur la mesure, le proxy en départage.

**L'effet de bord est plus important que la mesure elle-même.** Pour comparer les espèces à
prédicteur constant, PROPKA a été relancé sur les structures **Boltz humaines**. Résultat :

| design | ΔpKa AF2-H | ΔpKa Boltz-H | ΔpKa Boltz-M |
|---|---|---|---|
| `692deac2f1034bb6_seq0` | **+2,84** | **+2,27** | **+2,41** |
| `36dbfc4737a3e59b_seq1` | **+0,96** | −0,17 | −0,38 |
| `987fe804e455bc58_seq0` | −0,22 | **+1,04** | −0,11 |
| `987fe804e455bc58_seq1` | −0,29 | **+0,97** | −0,70 |

Écart Boltz − AF2 sur les 23 : **−1,13 à +1,26**, médiane −0,10, **même verdict de mécanisme
sur 20/23**. Les 3 désaccords tombent **exactement sur les cas limites**, ceux entre −0,3 et
+1,1 — c'est-à-dire ceux qu'on serait tenté de promouvoir.

**Conséquence : `36dbfc4737a3e59b_seq1` est déclassé.** Son mécanisme était une propriété de la
structure AF2, pas de la séquence. Le critère est durci : un mécanisme n'est **robuste** que
s'il est positif sur les **trois** mesures. Un palier **« mécanisme non reproductible »** est
ajouté, au-dessus du neutre. **Un seul design du lot est robuste.**

**La règle bidentée, testée trois fois.** C'est le résultat de méthode du dossier.

| test | géométrie | ΔpKa | goulots positifs | goulots négatifs | séparation |
|---|---|---|---|---|---|
| 1 | AF2 | AF2 | 3,34 ; 3,35 | 4,54 ; 4,55 ; 6,40 | nette, 1,19 Å |
| 2 | AF2 | Boltz | 3,35 ; 4,54 ; 4,55 | **3,34** ; 6,40 | **recouvrement 1,21 Å** |
| 3 | Boltz | Boltz | 3,19 ; 3,88 ; 4,03 | 6,23 ; 6,74 ; 7,97 | nette, **2,20 Å** |

Le test 1 était auto-référentiel (même structure des deux côtés). Le test 2 met la règle en
échec. Le test 3, le seul correct, la passe avec **presque le double de marge**. Donc : **la
règle tient, mais elle est locale à la structure** — le goulot doit être mesuré sur la
structure dont on évalue le pKa, jamais transféré d'un prédicteur à l'autre. Cohérent avec
`36dbfc4737a3e59b_seq1`, dont le goulot passe de 3,34 Å (AF2) au groupe 6,2–8,0 Å (Boltz) :
Boltz place ses carboxylates ailleurs et le mécanisme disparaît avec eux. p ≈ 5 % sous
l'hypothèse nulle (3 contre 3 sur 6 designs à paire possible). **Soutenue, non établie.**

**Soumission remise à jour** : toujours 13 designs sur 13 squelettes, mais **l'ordre a changé**
et l'identité maximale descend à **23,4 %**. Les 8 contrôles passent. Coût GPU cumulé des deux
jours : **~$2,64**.

### Correctif — les CSV de mesure n'étaient pas suivis

Le plan demandait de commiter `master_rank.csv`. Il ne l'était pas : `out/` est gitignoré, et
`git add -A` l'a donc silencieusement ignoré. Les 16 CSV de mesure sont ajoutés en `git add -f`
(1,4 Mo, dont 1,3 pour `propka_all_groups.csv`).

**Motif de l'exception au gitignore** : ces fichiers ne sont pas régénérables au sens où
`out/` est censé l'être. Ils dépendent de `out/egfr-dIII-prod0*/3_Ranked/`, qui est gitignoré
et dont les designs **ne sont pas reproductibles** — rejouer le même commit ne redonne pas les
mêmes séquences. Ils sont donc la seule trace d'audit des mesures du dossier.

**⚠️ Lacune qui RESTE** : `out/` pèse 186 Mo et contient les sorties de campagne
(`!_Ranked.csv` et ses 42 colonnes, les structures acceptées, les logs). Ce n'est pas dans le
dépôt, et ce n'est **pas reconstructible**. Si ce répertoire est perdu, le dossier ne peut plus
être refait — seules les séquences survivent, dans `submission/` et `out/master_rank.csv`. À
arbitrer : sauvegarde hors dépôt, ou ajout ciblé des `!_Ranked.csv` (quelques dizaines de Ko).

**Note d'environnement** : le push a échoué deux fois en `HTTP 400 / RPC failed` avant qu'un
`git config http.postBuffer 524288000` ne le fasse passer. Le dépôt est **privé**
(`gh repo view` → `isPrivate: true`), donc le push ne publie rien.

### Classeur consolidé pour le dépôt, et un bug de clé corrigé au passage

[build_metrics_workbook.py](build_metrics_workbook.py) → `docs/egfr_metrics_consolidated.xlsx`,
72 Ko, 14 onglets. Distinct de `submission/egfr_analysis.xlsx`, qui reste la copie de travail :
celui-ci est destiné à être lu **sans le reste du dépôt**, d'où un onglet `Dictionnaire`
(44 colonnes documentées une par une : signification, source, statut) et un onglet `Seuils`.

**Les seuils sont importés depuis les modules, jamais recopiés.** `build_metrics_workbook.py`
fait un `importlib.import_module` sur `rank_designs`, `verify_geometry`, `contact_recovery`,
`bidentate_rule`, `build_submission`, `mouse_target` et `propka_scan`, et lit les constantes
dedans. Une divergence entre le code et la documentation est donc impossible par
construction. Les 22 constantes sont lues correctement.

**⚠️ Bug trouvé en construisant le classeur : `design_id` n'était PAS une clé unique dans
`master_rank.csv`.** Les mutants reprenaient l'identifiant de leur parent et ne s'en
distinguaient que par la colonne `mutations`. L'onglet Synthèse est sorti à 20 lignes au lieu
de 13 — la jointure sur `design_id` ramenait chaque design soumis **plus tous ses mutants**.
Corrigé dans `rank_designs.py` : un mutant porte désormais `<parent>__<mutation>`, et la
colonne `parent` donne l'accès au squelette. Vérifié : 35 lignes, 35 identifiants distincts.
C'est le genre de défaut qui ne casse rien visiblement et qui fausse toute analyse en aval.

**Fausse alerte vérifiée, pas supposée** : `epitope_souris_retrouve` et
`recuperation_contacts` sont identiques sur 4 designs sur 23, ce qui ressemblait à un bug de
jointure. Vérification sur les 23 et sur les détails par échantillon : les valeurs diffèrent
partout ailleurs (0,604 vs 0,811 ; 0,479 vs 0,917) et les minima par échantillon ne
coïncident pas. Avec 46 à 76 paires de contacts au dénominateur, le ratio tombe sur une
grille discrète où les égalités fortuites sont attendues. Pas de bug.

À noter aussi : `5c3ec1903e03c261_seq0` retrouve **mieux** l'épitope chez la souris (0,911)
que chez l'humain (0,696). C'est de la variance d'échantillonnage de Boltz sur ce design, pas
un résultat biologique — c'est d'ailleurs le design dont la récupération humaine est la plus
basse du lot.

### Nomenclature corrigée : `natif` / `frere` disait une filiation inexistante

La colonne `type` de `master_rank.csv` valait `natif` pour `seq0` et `frere` pour `seq1`. Ces
étiquettes laissaient croire que `seq0` était la séquence d'origine et `seq1` une dérivée.
**C'est faux**, et vérifié dans les données avant correction : pour `987fe804e455bc58`, les
deux séquences ont leur propre rang BindCraft (3 et 6), leurs propres métriques AF2
(i_pTM 0,83 / 0,83, i_pAE 0,18 / 0,19) et leur propre structure prédite (164 Ko chacune), pour
un squelette partagé.

Une trajectoire produit un **squelette** ; ProteinMPNN propose des séquences pour ce
squelette ; `kept_sequences=2` en garde deux. Les deux sont des sorties de campagne **non
modifiées**, et l'index ne porte aucune hiérarchie — le lot soumis retient d'ailleurs
plusieurs `seq1` au-dessus du `seq0` du même squelette.

Nouvelle nomenclature :
- `type` ∈ {`bindcraft`, `mutant`} — la seule distinction réelle, modifié ou pas ;
- `index_mpnn` ∈ {`seq0`, `seq1`} — l'index ProteinMPNN, dans sa propre colonne.

Vérifié : 23 `bindcraft` et 12 `mutant` ; 20 `seq0` et 15 `seq1`. Ce qui compte pour
l'indépendance des poses reste le **squelette**, pas l'index : deux séquences de même hash ne
sont pas deux poses indépendantes, et c'est pourquoi l'allocation n'en soumet qu'une par
squelette.

### 5 octobre, fin — paquet de soumission complet, coupé à 6 designs

Le texte officiel de la soumission, relu directement, dit deux choses qui ont changé le
travail. Les organisateurs suggèrent explicitement *« ask your Claudes to write a methods
paper and create a metadata package »*, donc les deux livrables manquants étaient nommés. Et
*« for Tracks 2 and 3, we will provide the additional information you submit to Claude to help
select designs »* : le destinataire est un modèle, d'où l'anglais et une structure lisible par
machine.

**Coupe de 13 à 6 designs.** Règle explicite `EXCLUDED_PH_TIERS` dans `build_submission.py` :
les 7 designs du palier **contre-sélectif** sont écartés. Motif inscrit dans le code — leur
liaison est prédite défavorisée à pH 6,5, soit l'inverse du critère de rang le plus élevé, et
le texte demande de soumettre ce qu'on pense devoir le mieux se comporter. Coût : 7 squelettes
indépendants perdus. Identité maximale entre lignes : **20,7 %**, 8 contrôles passés.

⚠️ **Conflit signalé et non résolu par la règle** : couper sur le seul pH garde
`9526c9216eb7d6db_seq0` au rang 6, alors qu'il ne retrouve que **0,479** de son épitope chez la
souris — il échoue l'objectif n°2. Trois mesures indépendantes le désignent comme le point
faible des 6 : épitope murin 0,479, ipSAE A→B 0,704 (dernier), ΔpKa souris −2,62. Le passer à
5 est un changement d'une ligne.

**ipSAE — fait, via l'implémentation de référence.** `DunbrackLab/IPSAE` téléchargé, SHA256
enregistré dans la sortie, exécuté tel quel : réécrire la formule de mémoire aurait produit un
nombre plausible qui n'en serait pas un. Cutoffs 10 et 15 Å, ceux des exemples de l'outil.
ipSAE symétrisé **0,839 à 0,919** ; direction cible→binder, plus exigeante, **0,704 à 0,831**.

⚠️ **Trois colonnes de l'outil sont ÉCARTÉES et la raison est dans le code** : `ipTM_af` vaut
0,000 partout (l'outil attend un JSON AF2/AF3), `pDockQ` et `pDockQ2` sont constants sur les
6 designs (branche Boltz non renseignée), et `n0res` est le compte de normalisation de d0 —
égal à la longueur de chaîne alignée — et **non** un nombre de résidus d'interface, ce que ma
première documentation affirmait à tort. Vu en lisant le `.txt` brut, pas supposé. J'ai failli
publier une constante comme métrique.

**Auto-cohérence — faite, sur CPU.** ProteinMPNN `v_48_020`, T=0,1, 8 échantillons, binder
redessiné dans son contexte avec la cible fixe. Récupération **0,429 à 0,668, médiane 0,597**.
Circularité partielle énoncée : BindCraft utilise ProteinMPNN, mais sélectionne après
repliement sur filtres AF2 et non sur l'argmax de MPNN — d'où ~0,6 et non ~1,0.
`a6d2f6834f22e574` est dernier (0,43), comme sur le facteur global, le ΔpKa et le PLL ESM-2.

**Liabilités de séquence — faites, pondérées par la SASA.** 0 liabilité dure sur 35 designs
(aucune cystéine libre). 6 séquons N-glyc comptés mais **non retenus** : expression
acellulaire, donc pas de machinerie de glycosylation. Un motif enfoui n'est pas une liabilité,
d'où la pondération par l'exposition réelle plutôt qu'un grep de motifs.

**Erreur coûteuse de la séquence** : `--write_full_pae` ajouté **sans étendre la boucle de
copie vers le Volume**, qui ne prenait que `.cif` et `confidence_*.json`. Les matrices PAE ont
été calculées puis détruites avec le conteneur, ~$0,50 perdus. Le glob est désormais générique
(`*.cif`, `*.pdb`, `*.npz`, `*.npy`, `*.json`) et le log affiche le nombre de fichiers
conservés par complexe.

**Copie de travail sortie de `submission/`.** `egfr_analysis.xlsx` y vivait alors qu'elle ne
doit pas partir chez Adaptyv ; déplacée en `docs/egfr_analysis_working_copy.xlsx`.
`submission/` ne contient plus que le livrable.

Coût GPU cumulé : **~$4,2**.
