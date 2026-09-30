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
