# CLAUDE.md — BindCraft sur Modal : étape « test »

## Objectif unique de cette étape

Faire tourner la **démo PD-L1 de BindCraft de bout en bout sur Modal**, récupérer les
fichiers de sortie en local, et savoir combien ça coûte et combien de temps ça prend.

Rien d'autre. Ce n'est pas l'étape où l'on conçoit des binders pour la compétition,
pas l'étape où l'on ajuste les losses, pas l'étape où l'on optimise le débit.
Le livrable est une commande qui marche et une facture connue.

Si tu te surprends à proposer d'améliorer la qualité des designs, tu es hors périmètre :
dis-le et reviens à l'objectif.

## Contexte

Préparation à la compétition de protein design Adaptyv Bio x Anthropic. BindCraft
(Pacesa et al.) génère des binders *de novo* par rétropropagation à travers
AlphaFold2-Multimer, puis redesign ProteinMPNN et scoring PyRosetta.

Usage académique — la licence PyRosetta est couverte. Ne pas configurer ce projet
pour un usage commercial.

## Contraintes machine — non négociables

- Machine locale : **MacBook Air 2018, Intel x86_64. Aucun GPU NVIDIA, aucun CUDA.**
- BindCraft ne tournera **jamais** en local. Ne propose ni `install_bindcraft.sh`,
  ni conda/mamba, ni `jax[cuda]`, ni PyRosetta sur cette machine. Toute tentative
  est une perte de temps pure.
- Le local sert à trois choses : écrire du Python, lancer `modal run`, inspecter
  les résultats téléchargés.
- Environnement local : venv Python 3.12 géré par `uv`. Seule dépendance nécessaire
  au projet : `modal`.

## Architecture cible

Trois primitives Modal, et il faut comprendre pourquoi chacune est là :

| Primitive | Rôle | Pourquoi |
|---|---|---|
| `modal.Image` | Déclare l'environnement (CUDA, JAX, PyRosetta, BindCraft) | L'install de BindCraft est le vrai obstacle. L'Image la rend reproductible et cachée. |
| `modal.Volume` | Poids AF2 (5,3 Go) + sorties | Les poids se téléchargent une fois, pas à chaque run. Les résultats survivent au conteneur. |
| `--detach` | Le job continue après déconnexion | On ferme le laptop, le run continue. |

Les poids AF2 vont dans le **Volume**, jamais dans l'Image : un rebuild d'image ne doit
pas coûter 5,3 Go de téléchargement.

## Règles de travail

1. **Ne réinvente pas l'image.** Pars du `modal_bindcraft.py` du dépôt
   `hgbrian/biomodals`, qui résout déjà le problème. Lis-le, comprends-le, adapte-le.
   Écrire une image from scratch est un piège à trois jours.
2. **Épingle toutes les versions.** Jamais de version flottante sur `jax`, `jaxlib`,
   `cuda`, `numpy`. La majorité des échecs BindCraft viennent d'incompatibilités
   JAX/CUDA. Une version flottante rend le bug non reproductible.
3. **Build avant run.** Valide que l'image se construit et que
   `jax.devices()` voit bien un GPU, dans une fonction triviale, *avant* de lancer
   la moindre trajectoire. Un build raté à la minute 40 d'un run coûte le run entier.
4. **Toujours `--detach`** sur les runs réels.
5. **Ne modifie aucun JSON de settings par défaut** à ce stade. `default_filters.json`
   et `default_4stage_multimer.json` restent intacts. Les seuils ont été calibrés
   contre des résultats expérimentaux ; les toucher pour « faire passer des designs »
   casse ce lien. Si un filtre bloque, c'est une information, pas un obstacle.
6. **Le test tourne avec `--number-of-final-designs 1`.** On veut une preuve que le
   pipeline boucle, pas un résultat scientifique.
7. **Cible tronquée.** La mémoire GPU croît de façon quadratique avec le nombre de
   résidus (représentation de paires d'AF2), et la rétropropagation stocke ses
   activations. Ordre de grandeur : ~550 résidus sur 32 Go, ~950 sur 80 Go.
   Le `PDL1.pdb` d'exemple est déjà tronqué — ne le rallonge pas.
8. **Pas de tâtonnement sur les versions.** Devant une erreur : lis le message,
   identifie le composant, vérifie sa version, corrige une chose. Changer trois
   pins d'un coup rend le diagnostic impossible.
9. **Demande avant de dépenser.** Tout run dont tu estimes le coût au-dessus de
   quelques dollars, ou la durée au-dessus d'une heure : annonce l'estimation et
   attends l'accord.

## Critères d'acceptation

L'étape test est passée quand **tous** ces points sont vrais :

- [ ] `modal run` sur une fonction triviale confirme `jax.devices()` -> GPU CUDA visible
- [ ] L'image se construit sans erreur et le build est caché (2e build quasi instantané)
- [ ] Les poids AF2 sont dans le Volume et ne se retéléchargent pas au run suivant
- [ ] Un run détaché sur `PDL1.pdb` se termine sans exception
- [ ] `modal volume get` ramène les sorties en local
- [ ] Les fichiers attendus existent : `Trajectory/`, `MPNN/`, `Accepted/`,
      `final_design_stats.csv`, `failure_csv.csv`
- [ ] On sait lire `failure_csv.csv` : quel filtre rejette, et combien de fois
- [ ] Durée et coût du run sont notés dans `NOTES.md`

Zéro design accepté n'invalide **pas** le test. Sur cible difficile il faut souvent
quelques centaines à quelques milliers de trajectoires. Ce qu'on valide ici est la
plomberie, pas le rendement.

## Anti-objectifs

- Installer quoi que ce soit de BindCraft en local
- Toucher aux seuils de filtres ou aux poids de loss
- Paralléliser, optimiser le débit, gérer plusieurs cibles
- Passer à BindCraft2 ou à une autre cible avant que la démo PD-L1 ne tourne
- Utiliser Colab en secours : le notebook tourne environ 10x plus lentement qu'une
  install locale à hardware équivalent, et les sessions meurent

## Structure du dépôt

```
.
├── CLAUDE.md              # ce fichier
├── NOTES.md               # journal : ce qui a marché, versions, durées, coûts
├── modal_bindcraft.py     # point d'entrée Modal
├── inputs/
│   └── PDL1.pdb           # cible de démo, tronquée
└── out/                   # résultats rapatriés du Volume (gitignored)
```

`NOTES.md` est à tenir à jour à chaque run : commande exacte, GPU, durée, coût,
nombre de trajectoires, nombre d'acceptés. C'est ce qui servira de baseline quand
on passera à la vraie cible.

## Commandes de référence

```bash
# cible de démo
curl -LO https://raw.githubusercontent.com/martinpacesa/BindCraft/refs/heads/main/example/PDL1.pdb

# run de test, détaché
GPU=A100 uv run --with modal modal run --detach modal_bindcraft.py \
  --input-pdb inputs/PDL1.pdb --number-of-final-designs 1

# rapatrier les sorties
modal volume get bindcraft <run_name> ./out/
```

## Sécurité

- Le token Modal ne va ni dans le dépôt, ni dans un fichier suivi par git.
  `modal token set` uniquement.
- `out/` et tout fichier de résultat sont dans `.gitignore`.

## Sorties à comprendre avant de passer à la suite

Ces trois fichiers sont ce qui compte, et il faut savoir les lire avant l'étape suivante :

- `final_design_stats.csv` — une ligne par design accepté, toutes les métriques.
  Les colonnes sont préfixées `1_`, `2_` … `Average_` : ce sont les cinq modèles AF2.
  Seuls les modèles 1, 2 et la moyenne sont filtrés par défaut.
- `failure_csv.csv` — combien de designs chaque filtre a rejetés. **Premier réflexe
  de diagnostic**, avant de toucher à quoi que ce soit.
- `Trajectory/` — les PDB intermédiaires. Utile pour vérifier visuellement qu'on
  génère des protéines et pas du spaghetti.

## Ton comportement attendu

- Direct et technique. Pas de préambule, pas de reformulation de la demande.
- Si une approche est mauvaise, dis-le avant de l'implémenter, pas après.
- Ne prétends jamais qu'un run a réussi sans avoir lu la sortie réelle.
- Devant une incertitude sur une version ou une API Modal : vérifie la doc,
  ne devine pas.
