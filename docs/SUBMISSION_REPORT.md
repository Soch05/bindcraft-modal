# Dossier de soumission — Challenge 1, binder EGFR conditionnel

Track 3, participant solo. Rédigé **au fur et à mesure** pendant la nuit du 4 au 5 octobre
2026, en autonomie, l'auteur étant endormi. Les décisions prises sans arbitrage humain sont
marquées **[autonomie]** à l'endroit où elles sont prises.

Horodatage d'ouverture du fichier : **5 octobre 2026, 01:20 Paris**.

⚠️ **Clôture repoussée de 24 h**, information reçue le 5 octobre vers 06:00 : l'échéance est
**mardi 6 octobre 13:59 Paris** et non lundi 5. Les sections ci-dessous écrites entre 00:30 et
06:00 l'ont été sous la contrainte de l'échéance du 5, avec des points de bascule horaires
(02:30, 04:00, gel à 09:00) qui sont devenus sans objet. Elles sont conservées telles quelles,
et le travail repris ensuite sans compression est marqué **[après report]**.

---

## Phase 0 — Règlement et format

### Source

La page du challenge a **changé de domaine** : `design.adaptyvbio.com/challenges/1` renvoie un
301 vers `proteinbase.com/challenges/1`, qui répond **404**. La page de compétition
`proteinbase.com/competitions/anthropic-adaptyv-2026` existe mais **ne contient aucune
spécification technique** de format : ni colonnes, ni séparateur, ni quota, ni règle
d'unicité. Elle ne dit que ceci, qui est pertinent pour la suite : *« Selection will not rely
on a single in silico metric »*, et que le workflow de sélection est fondé sur Claude et sera
publié après la compétition.

Le format retenu vient donc de [challenge-01-egfr.md](../challenge-01-egfr.md), relevé à la
source le 1er octobre, recoupé avec la description publique de la compétition.

### Ce qui est établi

| point | valeur | source |
|---|---|---|
| colonnes minimales | `name`, `sequence`, `molecule_class` | règlement §3 |
| `molecule_class` pour nous | `protein` | règlement §3 |
| ordre | **CSV ordonné par classement**, meilleur en première ligne | règlement §3 |
| quota Track 3 | **20 max** — un plafond, pas une cible | règlement §3 |
| longueur | 10–250 aa, chaîne unique | règlement §3 |
| catégorie | minibinders (40–100 aa) — nos 23 designs y sont tous | règlement §3 |
| critères de sélection | qualité prédite, nouveauté du design, **nouveauté de la méthode** | règlement §5 |
| ordre de priorité des objectifs | **pH → cross-réactivité souris → affinité** | section « How designs are ranked » |

Séparateur, encodage et nom de fichier ne sont **pas spécifiés** par la source. Retenu :
virgule, UTF-8 sans BOM, `egfr_challenge1_submission.csv`.

### §0f — le filtre d'unicité : NON TRANCHÉ

C'était la question critique, parce qu'elle décide l'allocation de la phase 6. Réponse
honnête : **le règlement ne la tranche pas.**

Le texte complet du filtre d'unicité est : *« Unicité : ne pas soumettre deux fois le même
design. »* Aucun seuil d'identité de séquence. Aucune indication automatique vs humain. Le
filtre de nouveauté voisin parle bien de *« diversité de séquence et de structure suffisante »*
mais **par rapport aux protéines connues**, pas entre les designs d'un même participant.

Un WT et son mutant ponctuel (98–99 % d'identité) ne sont pas littéralement « le même
design », mais c'est une **interprétation**, pas une confirmation explicite.

**[autonomie] Conséquence appliquée** : la consigne était de n'ajouter frères ou mutants que si
l'unicité était explicitement confirmée comme non fondée sur l'identité de séquence. Elle ne
l'est pas. L'allocation retenue est donc **« couverture maximale »** — le meilleur candidat par
squelette, un seul par squelette. L'allocation « paires appariées » est chiffrée en phase 6
pour information, mais **non retenue**.

Cela ne rend pas l'analyse appariée inutile : sous couverture maximale, elle sert à décider,
**pour chaque squelette**, si c'est le WT ou son mutant qui est le meilleur représentant. C'est
même son usage le plus défendable, puisque le choix est interne au squelette.

---

## Phase 1 — Inventaire du vivier

### a) Les 23 designs, les 13 squelettes

Confirmé : **23 designs sur 13 squelettes indépendants**, et **zéro recouvrement** entre les
deux runs.

| run | designs | squelettes | structure |
|---|---|---|---|
| `egfr-dIII-prod01` | 3 | 3 | 1 design par squelette |
| `egfr-dIII-prod02` | 20 | 10 | **2 séquences par squelette** (`kept_sequences=2`) |
| **total** | **23** | **13** | — |

**Ce que « deux séquences par squelette » veut dire, et ce que ça ne veut pas dire.** Une
trajectoire BindCraft produit un **squelette**, identifié par un hash. ProteinMPNN propose
ensuite des séquences pour ce squelette, et `kept_sequences=2` en conserve deux : `seq0` et
`seq1`. **Les deux sont des sorties de campagne non modifiées** — chacune a son propre rang
BindCraft, ses propres métriques AF2 et sa propre structure prédite. Aucune n'est dérivée de
l'autre, et `seq0` n'est pas « l'originale » : l'index ne porte aucune hiérarchie. Le lot
soumis retient d'ailleurs plusieurs `seq1` **au-dessus** du `seq0` du même squelette.

Ce qui les relie est le **squelette partagé** : deux séquences de même hash ne sont donc
**pas des poses indépendantes**, et c'est la raison pour laquelle l'allocation retenue n'en
soumet qu'une par squelette.

**Le seul type réellement modifié est `mutant`** : une substitution ponctuelle greffée sur le
squelette d'un design BindCraft, jamais passée par les filtres de la campagne. La colonne
`type` de `master_rank.csv` ne distingue donc que `bindcraft` et `mutant`, et l'index
ProteinMPNN vit dans sa propre colonne `index_mpnn`.

⚠️ **Correction du 5 octobre** : cette colonne valait auparavant `natif` pour `seq0` et
`frere` pour `seq1`, ce qui suggérait une filiation inexistante. Renommée.

Les hash de squelette de `prod01` (`f6d5f550a210fd48`, `1e7ab6d8f00c9958`,
`692deac2f1034bb6`) n'apparaissent dans aucune ligne de `prod02`. Les 3 designs de `prod01`
comptent donc comme 3 squelettes pleins.

Tous les 23 sont dans la catégorie **minibinders** (40–100 aa) : longueurs 55 à 94.

### b, c, d) Les mutants acides

`out/mutants_acide.csv` contient **23 lignes, une par design**, dont **12 portent une
substitution** (colonne `Binder_Sequence_mutee` non vide). Les 11 autres sont les designs qui
portaient déjà un pont salin et pour lesquels aucune mutation n'a été proposée.

| squelette | design | mutation | résidu d'origine | CB→H409 | coût annoncé |
|---|---|---|---|---|---|
| `f6d5f550a210fd48` | prod01 l55 seq0 | S38D | SER38 | 3,81 Å | conservative |
| `1e7ab6d8f00c9958` | prod01 l64 seq0 | S28D | SER28 | 4,53 Å | conservative |
| `a6334a3a912c86f1` | prod02 l61 seq0 | S15D | SER15 | 3,95 Å | conservative |
| `a6334a3a912c86f1` | prod02 l61 seq1 | S15D | SER15 | 3,96 Å | conservative |
| `cd272a8fd929c7ee` | prod02 l61 seq0 | N21E | ASN21 | 6,19 Å | conservative |
| `cd272a8fd929c7ee` | prod02 l61 seq1 | N21E | ASN21 | 6,28 Å | conservative |
| `a6d2f6834f22e574` | prod02 l62 seq0 | S44D | SER44 | 4,27 Å | conservative |
| `a6d2f6834f22e574` | prod02 l62 seq1 | S44D | SER44 | 4,58 Å | conservative |
| `4a818d7951649b77` | prod02 l64 seq0 | P39D | PRO39 | 4,69 Å | à éviter |
| `4a818d7951649b77` | prod02 l64 seq1 | F38D | PHE38 | 4,55 Å | à éviter |
| `5b295c4d9e1ff73f` | prod02 l73 seq0 | H23E | HIS23 | 5,36 Å | risquée |
| `5b295c4d9e1ff73f` | prod02 l73 seq1 | H23E | HIS23 | 5,32 Å | risquée |

Chaque mutant a un parent identifiable, et le parent existe comme structure. **Aucun mutant
orphelin.** **Aucun doublon exact**, et aucune séquence mutée identique à son parent : les 12
substitutions modifient bien un résidu.

Un point à signaler qui n'était pas marqué dans le CSV : **`H23E` détruit la seule histidine
du binder.** Le fichier la classe « risquée » sans dire pourquoi ; la raison est que remplacer
une His par un Glu supprime la possibilité d'une Route 1 sur ce squelette pour tenter une
Route 2. C'est un échange, pas un ajout.

### e) Vérification géométrique des mutations — le résultat qui change tout

Mesuré par [verify_geometry.py](../verify_geometry.py) sur les structures threadées, pour
chaque rotamère : distance du carboxylate introduit aux azotes de l'imidazole de H409, angle
à l'oxygène accepteur, et contacts lourds trop courts.

**Sur 12 mutants, 3 seulement ont au moins un rotamère à la fois à portée de pont salin
(≤ 4,0 Å) et sans recouvrement** : `S15D` sur les deux frères de `a6334a3a912c86f1`, et
`S44D` sur `a6d2f6834f22e574_seq1`.

| mutation | design | rotamères à portée et sans clash | distance la plus courte |
|---|---|---|---|
| S15D | `a6334a3a912c86f1_seq0` | 1/3 | 3,96 Å |
| S15D | `a6334a3a912c86f1_seq1` | 1/3 | 3,94 Å |
| S44D | `a6d2f6834f22e574_seq1` | 2/3 | 2,48 Å |
| S38D | `f6d5f550a210fd48_seq0` | 0/3 | 3,60 Å mais en clash |
| S28D | `1e7ab6d8f00c9958_seq0` | 0/3 | **3 rotamères sur 3 en clash** |
| S44D | `a6d2f6834f22e574_seq0` | 0/3 | 1,96 Å, recouvrement |
| N21E | `cd272a8fd929c7ee` ×2 | 0/3 | 7,0–9,2 Å, hors de portée |
| P39D | `4a818d7951649b77_seq0` | 0/3 | 6,56 Å, hors de portée |
| F38D | `4a818d7951649b77_seq1` | 0/3 | 4,91 Å, hors de portée |
| H23E | `5b295c4d9e1ff73f` ×2 | 0/3 | 5,24–6,92 Å, hors de portée |

**`S28D` est structurellement impossible** sur squelette rigide : ses trois rotamères
clashent (2,08–2,16 Å). C'est une information, pas un échec de mesure.

**Pourquoi le proxy de sélection a échoué.** `acid_mutants.py` a choisi les positions sur la
distance **CB→H409**. Or le CB est fixé par le squelette : il dit où part la chaîne latérale,
pas où arrive son extrémité. Au-delà d'environ **4,3 Å de CB**, le carboxylate ne rejoint plus
H409 — les quatre mutations à CB ≥ 4,55 Å (`N21E`, `F38D`, `P39D`, `H23E`) finissent toutes à
4,9–9,2 Å. Les trois qui portent (`S15D` 3,95 Å, `S44D` 4,27/4,58 Å) sont celles de CB le plus
court. Le seuil implicite était donc ~4,3 Å, et il n'avait pas été posé.

### f) Compte des candidats

| | |
|---|---|
| designs BindCraft | 23 |
| mutants threadés | 12 |
| **total au classement** | **35** |
| quota Track 3 | 20 (plafond) |

### g) Re-vérification des ponts salins WT — jamais faite avant

Les **11 designs** marqués « pH déjà établi », sur **6 squelettes**, portent bien un
carboxylate de binder à moins de 4,0 Å d'un azote de l'imidazole de H409. Mes distances
**reproduisent exactement** celles de `mutants_acide.csv`, ce qui est une vérification croisée
de la mesure antérieure.

L'apport nouveau est l'**angle** à l'oxygène accepteur et l'**enfouissement**, que la distance
seule ne donnait pas. Pour un carboxylate sp2, les doublets libres pointent vers ~120° : un
angle proche de 120° est une géométrie de liaison hydrogène correcte, un angle de 80° ou de
160° ne l'est pas.

| squelette | résidu | distance | angle | enfoui par l'interface | verdict géométrique |
|---|---|---|---|---|---|
| `692deac2f1034bb6` seq0 | ASP56 | **2,52 Å** | **128,4°** | 30,0 Å² | court mais **angle quasi idéal** |
| `9526c9216eb7d6db` seq0 | ASP32 | 2,42 Å | 96,1° | 34,3 Å² | court **et** angle médiocre |
| `9526c9216eb7d6db` seq1 | ASP32 | 2,46 Å | 99,7° | 32,1 Å² | court **et** angle médiocre |
| `fd5dae7987a2388d` seq0 | GLU24 | 3,36 Å | 147,1° | 26,8 Å² | distance propre, angle trop ouvert |
| `fd5dae7987a2388d` seq1 | GLU24 | 3,31 Å | 159,9° | 21,4 Å² | angle quasi colinéaire |
| `36dbfc4737a3e59b` seq0 | ASP24 | 3,22 Å | 79,1° | 37,5 Å² | angle médiocre |
| `36dbfc4737a3e59b` seq1 | ASP24 | 3,08 Å | 85,9° | 43,8 Å² | angle médiocre |
| `987fe804e455bc58` seq0 | ASP52 | 3,26 Å | 116,8° | 18,2 Å² | **géométrie saine** |
| `987fe804e455bc58` seq1 | ASP52 | 3,21 Å | 119,5° | 16,1 Å² | **géométrie saine** |
| `5c3ec1903e03c261` seq0 | ASP72 | 3,31 Å | 109,1° | 28,0 Å² | géométrie correcte |
| `5c3ec1903e03c261` seq1 | ASP72 | 3,17 Å | 120,7° | 31,2 Å² | **géométrie saine** |

#### Le cas `692deac2f1034bb6` : le pont à 2,52 Å est-il un artefact ?

C'était la question posée, et la réponse est **non, pas au sens d'une impossibilité
physique** — avec une réserve.

- **2,52 Å entre OD2 et NE2 est court** : un pont salin O···N se situe normalement entre
  2,6 et 3,2 Å. C'est en dessous.
- **Mais c'est le contact le plus court de tout le voisinage.** Aucun autre atome lourd
  non lié n'est plus près de la chaîne latérale d'ASP56 que ce contact-là. Il n'y a donc
  **pas de recouvrement stérique** : la chaîne latérale n'est pas écrasée dans la cible, elle
  est serrée contre son partenaire et rien d'autre.
- **L'angle est de 128,4°**, le plus proche de l'idéal des onze ponts mesurés. Un artefact de
  prédiction produit typiquement un contact court *et* une géométrie tordue ; ici la
  géométrie est bonne.
- À comparer à `9526c9216eb7d6db`, qui est **plus court encore** (2,42 Å) **avec un angle de
  96°** : celui-là ressemble beaucoup plus à un artefact.

**Verdict** : géométriquement plausible comme liaison hydrogène forte à caractère partagé.
La réserve est qu'une structure prédite ne permet pas de distinguer une liaison hydrogène
forte d'un contact légèrement sur-optimisé, et que ce contact n'a pas été relaxé.

---

## Phase 2 — Threading multi-rotamère

### a) Outil retenu : PyMOL open-source

Choisi pour trois raisons. Il est **déjà installé** sur la machine
(`/usr/local/bin/pymol`, mode headless `pymol -cq`) donc aucune minute n'a été dépensée en
installation. Son assistant de mutagenèse **énumère les rotamères d'une bibliothèque
peuplée** et les expose comme états d'un objet, ce qui donne exactement la sortie voulue.
Et il **ne touche pas au squelette** : seule la chaîne latérale est posée, ce qui est la
condition de l'analyse appariée.

PDBFixer/OpenMM aurait relaxé, donc bougé le squelette — c'est précisément ce qu'on ne veut
pas ici. Une géométrie idéale posée à la main n'aurait pas donné de bibliothèque de rotamères.

Détail d'implémentation à garder : l'appel est `pymol -cq fichier.py` et **non** `pymol -d`.
L'option `-d` exécute ligne par ligne au niveau toplevel, donc aucun bloc multi-ligne ne
passe.

### b) Contrôle de clash

Seuil : aucun atome lourd de la chaîne latérale greffée à moins de **2,2 Å** d'un autre atome
lourd non lié. Mesure faite par nos propres fonctions et non par la contrainte interne de
PyMOL, qui n'est pas accessible par l'API.

**36 structures écrites, 13 en clash** (3 à 4 rotamères par mutant, les 3 moins encombrés
retenus). Pour `S28D`, **les 3 rotamères clashent** : le mutant est déclaré structuralement
impossible sur squelette rigide.

### c) Nommage

`structures/threaded/<design>__<mutation>__rot<N>.pdb`, index complet dans
`structures/threaded/threaded_index.csv` avec distance de contact minimale, partenaire et
drapeau de clash par rotamère.

### d) Ordre de traitement

Par paire : parent puis ses mutants, paire suivante. Le run a été mené à son terme, donc les
12 paires sont complètes.

### Un défaut de méthode trouvé en vérifiant, et corrigé

Le threading classe les rotamères sur le **contact le plus serré**, en gardant les trois moins
encombrés. Pour `N21E`, `H23E`, `F38D` et `P39D`, ce contact minimal est **identique sur les
trois rotamères** (par exemple 3,16 Å pour les deux `H23E`). La raison est que le contact
minimal y est porté par le **CB**, atome dont la position est fixée par le squelette et qui ne
bouge donc pas d'un rotamère à l'autre. **Le tri est aveugle pour ces mutants.**

Conséquence vérifiée et non supposée : l'étendue du **centroïde du carboxylate** entre
rotamères a été mesurée pour chaque mutant.

| mutant | étendue du centroïde | rotamères réellement distincts |
|---|---|---|
| `P39D` sur `4a818d7951649b77_seq0` | **0,25 Å** | **NON** |
| tous les autres | 2,52 à 3,68 Å | oui |

Donc un seul mutant, `P39D`, a trois rotamères quasi identiques — son verdict multi-rotamère
n'en est pas un, et c'est noté comme tel. Pour les onze autres, les rotamères explorent bien
des positions différentes et le verdict en trois états a un sens.

### e) Limite à retenir

Squelettes **non relaxés**, chaînes latérales posées par rotamère idéal. Les pKa **absolus**
qui en sortent sont donc grossiers. Pour la **comparaison appariée** parent contre mutant,
c'est au contraire l'approche la plus propre, puisque le squelette est partagé et que la
mutation est la seule variable.

---

## Phase 3 — PROPKA

### a) Version et invocation

`propka3` installé en dépendance jetable via `uv run --with propka --with gemmi`. 141 runs,
3 en parallèle, par [propka_scan.py](../propka_scan.py).

### b) Vérification de la numérotation — faite sur les 23 structures, pas sur une seule

C'était le piège à désamorcer : le mécanisme repose sur H409 en numérotation **PDB**, et une
numérotation UniProt (décalage +24) aurait fait lire un autre résidu sans que rien n'ait l'air
anormal.

[prepare_structures.py](../prepare_structures.py) **refuse de convertir** un fichier dont le
résidu 409 de la chaîne cible n'est pas une histidine. Résultat sur les 23 :

```
409=HIS   His cible 334,346,359,394,409,483   empreinte oui     (23 fois sur 23)
```

L'empreinte des **six** histidines de la cible est exacte partout, pas seulement le résidu
409. Un glissement de numérotation ne pourrait pas reproduire cet ensemble.

Au passage, une raison technique de cette étape : les `.cif` de BindCraft 2.0 n'ont pas de
colonne `_atom_site.occupancy`, que le `MMCIFParser` de biopython tient pour obligatoire — il
lève un `KeyError`. La conversion en PDB par gemmi règle ça et sert aussi PROPKA, qui lit le
PDB.

### c) H409 lié contre libre — le résultat central

Chaque complexe a été calculé, puis **la cible seule extraite du même fichier**, pour que la
différence ne porte que sur la présence du binder.

**Sur 23 designs, 2 font monter le pKa de H409** — sur les structures AF2. Un seul des deux
survivra à la vérification sur une seconde structure et chez la souris ; voir la section
cross-réactivité.

| design | pKa H409 lié | pKa H409 libre | ΔpKa | facteur pH prédit |
|---|---|---|---|---|
| `692deac2f1034bb6_seq0` | **9,11** | 6,27 | **+2,84** | **5,28** |
| `36dbfc4737a3e59b_seq1` | 7,27 | 6,31 | **+0,96** | **2,60** |
| `9526c9216eb7d6db_seq0` | 6,33 | 6,30 | +0,03 | 1,02 |
| les 20 autres | — | — | **−0,01 à −2,89** | 0,66 à 0,99 |

**Comment lire un ΔpKa négatif.** Si lier le binder *abaisse* le pKa de H409, la liaison
déstabilise la forme protonée, donc elle est **défavorisée** à pH acide : c'est l'inverse de
l'objectif. Vingt designs sur vingt-trois sont dans ce cas ou dans le bruit.

#### Décomposition, pour `692deac2f1034bb6_seq0`

PROPKA attribue explicitement la hausse :

```
HIS 409 A   9.11   86 %   -2.50 (désolvatation)   +1.60 ASP 56 B   +1.60 GLU 73 B
                                                  +1.39 ASP 56 B   +0.54 GLU 73 B  (coulombien)
```

Le terme de **désolvatation contribue négativement** (−2,50) : l'enfouissement, à lui seul,
ferait *baisser* le pKa. La hausse vient donc entièrement des **deux carboxylates du
binder**, `ASP56` et `GLU73`, par liaison hydrogène de chaîne latérale (+1,60 chacun) et par
terme coulombien (+1,39 et +0,54). **Ce n'est pas un artefact d'enfouissement, et ce n'est
pas un voisin de la cible.** C'est le mécanisme Route 2 visé, décomposé.

#### Le fait mécanistique le plus utile du projet

Les deux carboxylates engagent **les deux azotes** de l'imidazole : `ASP56–NE2` et
`GLU73–ND1`. Or plusieurs designs portent **un** pont salin de géométrie impeccable
(`987fe804e455bc58` à 3,21 Å et 119,5° ; `5c3ec1903e03c261` à 3,17 Å et 120,7° ;
`fd5dae7987a2388d` à 3,36 Å) et obtiennent un **ΔpKa nul ou négatif**.

**Un pont salin unique, même géométriquement parfait, ne décale pas le pKa de H409.** C'est
une conclusion tirée de 23 cas sur la même cible et le même résidu, et elle contredit le
critère géométrique qui avait servi à sélectionner les designs.

La formulation « il faut que les deux azotes soient engagés » a été testée puis **réfutée** :
neuf designs engagent les deux azotes avec des ΔpKa allant de +2,84 à −1,19. La condition qui
sépare réellement les deux gagnants est plus fine — **deux résidus carboxylate distincts**, un
par azote — et elle est traitée dans sa propre section, avec la borne qu'il faut mettre sur sa
solidité statistique.

### d) Le piège du carboxylate auto-destructeur — verdict en trois états

Seuil : un carboxylate de pKa > **5,0** est partiellement protoné dès pH 6,5, donc perd sa
charge là où le mécanisme en a besoin.

| mutant | pKa du D/E par rotamère retenu | min | max | verdict |
|---|---|---|---|---|
| `S15D` `a6334a3a912c86f1_seq0` | 8,67 / 8,77 / 5,53 | 5,53 | 8,77 | **ROUGE** |
| `S15D` `a6334a3a912c86f1_seq1` | 8,70 / 8,65 / 5,54 | 5,54 | 8,70 | **ROUGE** |
| `N21E` `cd272a8fd929c7ee_seq0` | 5,17 / 5,16 / 5,81 | 5,16 | 5,81 | **ROUGE** |
| `N21E` `cd272a8fd929c7ee_seq1` | 5,19 / 5,22 / 5,35 | 5,19 | 5,35 | **ROUGE** |
| `F38D` `4a818d7951649b77_seq1` | 6,25 / 6,17 | 6,17 | 6,25 | **ROUGE** |
| `S44D` `a6d2f6834f22e574_seq1` | 4,07 / 3,49 | 3,49 | 4,07 | **VERT** |
| `P39D` `4a818d7951649b77_seq0` | 3,24 / 3,65 / 3,55 | 3,24 | 3,65 | **VERT** |
| `S38D` `f6d5f550a210fd48_seq0` | 5,12 / 2,87 / 4,78 | 2,87 | 5,12 | INDÉTERMINÉ |
| `S28D` `1e7ab6d8f00c9958_seq0` | 3,64 / 5,37 / 3,52 | 3,52 | 5,37 | INDÉTERMINÉ |
| `S44D` `a6d2f6834f22e574_seq0` | 3,77 / 7,47 / 9,91 | 3,77 | 9,91 | INDÉTERMINÉ |
| `H23E` `5b295c4d9e1ff73f_seq0` | 4,67 / 5,09 / 4,98 | 4,67 | 5,09 | INDÉTERMINÉ |
| `H23E` `5b295c4d9e1ff73f_seq1` | 4,85 / 5,18 / 5,00 | 4,85 | 5,18 | INDÉTERMINÉ |

**Le cas `S15D` est le plus parlant.** C'est le seul mutant, avec `S44D`, dont un rotamère
soit à la fois à portée de H409 et sans clash. Et son `ASP15` sort à **8,67–8,77** : il serait
**neutre aux deux pH**, donc incapable de former le pont salin pour lequel il a été introduit.
La mutation est auto-annulante, de façon robuste sur tous ses rotamères.

#### Même logique appliquée au ΔpKa de H409

| mutant | ΔpKa par rotamère retenu | étendue | verdict mécanisme |
|---|---|---|---|
| `S44D` `a6d2f6834f22e574_seq1` | +0,20 / +0,26 | **0,06** | stable, mais sous le bruit |
| `N21E` ×2 | −0,36 à −0,40 | 0,01–0,04 | stable **négatif** |
| `P39D`, `F38D` | −1,54 / −1,80 | 0,00 | stable **négatif** |
| `H23E` ×2 | −1,42 à −1,72 | 0,02–0,14 | stable **négatif** |
| `S38D`, `S28D`, `S15D` ×2, `S44D` seq0 | de −3,04 à +0,73 | **2,16 à 3,71** | **INDÉTERMINÉ** |

**Aucun mutant n'obtient un ΔpKa positif robuste.** Les cinq dont l'étendue dépasse 2 unités
sont indéterminés, et aucun n'est rétrogradé pour autant — mais aucun ne peut non plus être
présenté comme porteur d'un mécanisme.

### e) Scan non biaisé — recherche d'une « Route 3 »

Tous les groupes ionisables dont le pKa bouge de plus de 0,5 unité entre lié et libre, sur les
23 complexes, côté cible **et** côté binder (le binder seul a donc été calculé aussi, sans
quoi ses groupes n'auraient aucun état libre de référence). 180 décalages relevés.

Ce que le scan trouve, et pourquoi ça ne donne pas de meilleure route :

| groupe | occurrences | pKa lié typique | utilisable entre 6,5 et 7,4 ? |
|---|---|---|---|
| `HIS409A` | 15 | 3,3 à 9,1 | **oui** — c'est la route conçue |
| `ASP436A` | 17 | 4,6 à 5,0 | **non** — déprotoné aux deux pH |
| `ASP344A` | 15 | jusqu'à 6,08 | **marginalement** — voir ci-dessous |
| `HIS346A` | 18 | 3,9 | non — protoné à aucun des deux pH utiles |
| `LYS465A`, `ARG353A` | 11, 10 | > 10 | non |
| `TYR` du binder | plusieurs | 13–14,5 | non |

**Un groupe ne crée de sélectivité entre 6,5 et 7,4 que si son pKa tombe dans ou près de cette
fenêtre.** `ASP436A` monte de 1,3 à 1,8 unité dans beaucoup de designs, ce qui est le plus gros
effet du scan après H409 — mais il passe de ~3,2 à ~4,8, donc reste déprotoné aux deux pH :
l'effet est réel et **sans conséquence** sur l'objectif.

Le seul candidat Route 3 est **`ASP344A` dans `fd5dae7987a2388d`**, qui atteint un pKa lié de
**6,08** (libre 4,40). C'est dans la fenêtre. Son facteur propre est de l'ordre de **1,3×**,
donc très inférieur aux 5,28× de H409 dans `692deac2f1034bb6`. À noter comme piste, pas
comme mécanisme.

#### Vérification croisée : le facteur de sélectivité **global**

Pour ne pas supposer la réponse, un second indicateur a été calculé : le produit, sur **tous**
les groupes ionisables du complexe, de leur contribution individuelle à la sélectivité pH.
C'est la forme thermodynamique du couplage proton–ligand, et elle ne privilégie aucun résidu
choisi d'avance.

| design | facteur H409 seul | facteur global, tous groupes | facteur global, groupes mobiles |
|---|---|---|---|
| `692deac2f1034bb6_seq0` | **5,28** | **5,49** | **5,65** |
| `36dbfc4737a3e59b_seq1` | **2,60** | **2,39** | **2,44** |
| `9526c9216eb7d6db_seq0` | 1,02 | 1,00 | 1,01 |
| `987fe804e455bc58_seq0` | 0,88 | 0,33 | 0,39 |
| `a6d2f6834f22e574_seq0` | 0,69 | 0,30 | 0,30 |

Les deux indicateurs **concordent sur la tête du classement** et sur son ordre. Là où ils
divergent (`987fe804e455bc58`, `a6d2f6834f22e574`), c'est **en défaveur** du design : vu
globalement, il est plus contre-sélectif que H409 seul ne le disait. **Aucun candidat n'est
sauvé par la vue non biaisée**, et le mécanisme conçu explique l'essentiel de la sélectivité
prédite des deux premiers. C'est le meilleur argument disponible pour dire que le choix
d'épitope dirigé par hypothèse n'a pas raté une meilleure option.

Limite du facteur global, à énoncer : le produit suppose les sites **indépendants**, ce qu'ils
ne sont pas quand deux groupes se touchent, et il cumule une erreur de PROPKA par groupe — d'où
la version restreinte aux groupes qui bougent de plus que le bruit, rapportée à côté.

### f) Calibration — ce que ces chiffres valent

PROPKA se trompe couramment d'une unité de pKa, davantage sur les gros décalages. **Les
valeurs servent à CLASSER, pas à annoncer un facteur.** Aucun facteur absolu ne figure dans la
soumission, et le ΔpKa de +2,84 de `692deac2f1034bb6` doit être lu comme « le plus grand
décalage du lot, dans le bon sens », pas comme « 2,84 unités ».

C'est pour la même raison que le classement **discrétise** le critère pH en trois paliers
(mécanisme / neutre / contre-sélectif) au lieu de trier sur la valeur continue : classer un
design à ΔpKa −0,01 au-dessus d'un autre à −0,40 serait lire un ordre dans du bruit.

---

## Phase 4 — Re-scoring orthogonal

### Niveau atteint : **A — Boltz-2 sur Modal**, 23/23 designs BindCraft, puis 12 mutants [après report]

Les niveaux B, C et D n'ont pas eu à servir. Le basculement de 02:30 n'a pas été déclenché :
le niveau A a produit son premier résultat à 01:49, soit 41 minutes avant la limite.

### Pourquoi un second prédicteur

BindCraft optimise ses designs **par descente de gradient à travers AlphaFold2**. Les `i_pTM`
et `i_pAE` qu'il rapporte sont donc des scores **in-sample** : le générateur a eu accès au juge
pendant chaque trajectoire. L'analogie est directe — c'est une erreur d'apprentissage, pas une
erreur de test. Demander à un modèle d'architecture et de poids indépendants s'il place le
binder au même endroit est le seul moyen de savoir si la pose est réelle.

### Configuration

| | |
|---|---|
| modèle | Boltz-2 **2.2.0**, épinglé |
| GPU | L40S, torch 2.14.1+cu130, capability 8.9 |
| échantillons | **3 échantillons de diffusion** par complexe, 3 étapes de recyclage |
| MSA cible | **3725 séquences**, ColabFold MMseqs2 mode `env`, précalculée une fois et **embarquée dans l'image** |
| MSA binder | **vide** (`msa: empty`), mode séquence seule |
| module d'affinité | **non utilisé** — calibré pour les petites molécules, pas les interfaces protéine–protéine |
| débit | ~67 s par complexe, 23 complexes |
| coût | **~$0,83** de L40S pour la série, plus ~$0,30 de diagnostic et d'échec |

Aucun appel réseau n'est fait depuis le conteneur GPU : la MSA est un fichier dans l'image.
C'était la cause de blocage annoncée comme la plus probable, et elle a été écartée par
construction.

#### Les trois échantillons de diffusion, et non trois graines

Écart assumé au plan. Le tronc de Boltz-2 (MSA et représentation de paires) est déterministe à
graine fixée : le relancer trois fois coûterait trois fois le calcul pour un tronc identique.
Ce qui varie entre poses, et ce qu'on veut échantillonner, est l'étape de diffusion.

### Le diagnostic, et ce qu'il a coûté

Le premier lancement a brûlé **sept minutes de L40S en produisant zéro prédiction**, avec un
**code retour 0** et six secondes par complexe. La cause, trouvée en relançant sur un seul
complexe avec la sortie affichée :

```
File "boltz/model/layers/triangular_mult.py", line 22, in kernel_triangular_mult
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
ModuleNotFoundError: No module named 'cuequivariance_torch'
```

Boltz 2.2.0 appelle **inconditionnellement** un noyau cuEquivariance pour la mise à jour
multiplicative triangulaire du pairformer, et `pip install boltz` **ne tire pas** ce paquet.
Correctif : `--no_kernels`, qui retombe sur l'implémentation PyTorch de référence — plus
lente, mais c'est la même fonction, et elle n'ajoute pas une dépendance à faire correspondre à
la version de CUDA.

Deux conséquences tirées, et inscrites dans le code :

1. **Un code retour 0 ne prouve pas qu'une prédiction a eu lieu.** `predict` imprime désormais
   la sortie de boltz dès qu'une prédiction manque, quel que soit le code retour.
2. **Un entrypoint `diagnose`** a été ajouté, qui traite un seul complexe en affichant tout. Il
   est à lancer avant toute série. Le dépôt avait déjà la règle « valider le build avant de
   brûler un run » ; elle avait été appliquée au GPU (`selfcheck`) mais pas au pipeline.

J'ai par ailleurs d'abord attribué l'échec à une MSA malformée, et **c'était faux** : l'erreur
survient au fond du pairformer, donc après lecture de l'alignement. La correction de la MSA
(fusion des deux `.a3m` en un alignement unique au lieu de deux collés bout à bout) est juste
en soi mais n'a rien réparé. Le commentaire du code qui le prétendait a été corrigé.

### Résultats — tous les designs survivent au retrait de leur échafaudage

Récupération de contacts = fraction des paires de résidus en contact de la pose AF2 retrouvées
dans la pose Boltz-2. Contact = au moins un couple d'atomes lourds à 5,0 Å ou moins.

| rang | design | récup. max | récup. min | iptm moyen | paires AF2 |
|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | **0.897** | 0.862 | 0.93 | 58 |
| 2 | `l59_36dbfc4737a3e59b_seq1` | **0.857** | 0.804 | 0.938 | 56 |
| 3 | `l63_987fe804e455bc58_seq0` | **0.969** | 0.953 | 0.948 | 64 |
| 4 | `l63_987fe804e455bc58_seq1` | **0.935** | 0.919 | 0.946 | 62 |
| 5 | `l61_cd272a8fd929c7ee_seq1` | **0.92** | 0.9 | 0.938 | 50 |
| 6 | `l61_cd272a8fd929c7ee_seq0` | **0.96** | 0.86 | 0.941 | 50 |
| 7 | `l58_fd5dae7987a2388d_seq0` | **0.887** | 0.855 | 0.937 | 62 |
| 8 | `l58_fd5dae7987a2388d_seq1` | **0.938** | 0.906 | 0.936 | 64 |
| 9 | `l57_9526c9216eb7d6db_seq0` | **0.917** | 0.896 | 0.92 | 48 |
| 10 | `l57_9526c9216eb7d6db_seq1` | **0.891** | 0.87 | 0.913 | 46 |
| 11 | `l64_1e7ab6d8f00c9958_seq0` | **0.864** | 0.831 | 0.932 | 59 |
| 12 | `l59_36dbfc4737a3e59b_seq0` | **0.879** | 0.845 | 0.925 | 58 |
| 13 | `l92_5c3ec1903e03c261_seq0` | **0.696** | 0.571 | 0.866 | 56 |
| 14 | `l62_a6d2f6834f22e574_seq0` | **0.899** | 0.87 | 0.949 | 69 |
| 15 | `l55_f6d5f550a210fd48_seq0` | **0.811** | 0.642 | 0.848 | 53 |
| 16 | `l62_a6d2f6834f22e574_seq1` | **0.921** | 0.829 | 0.935 | 76 |
| 17 | `l73_5b295c4d9e1ff73f_seq1` | **0.891** | 0.848 | 0.93 | 46 |
| 18 | `l73_5b295c4d9e1ff73f_seq0` | **0.87** | 0.783 | 0.932 | 46 |
| 19 | `l61_a6334a3a912c86f1_seq0` | **0.909** | 0.864 | 0.948 | 66 |
| 20 | `l92_5c3ec1903e03c261_seq1` | **0.929** | 0.875 | 0.91 | 56 |
| 21 | `l61_a6334a3a912c86f1_seq1` | **0.851** | 0.806 | 0.957 | 67 |
| 22 | `l64_4a818d7951649b77_seq1` | **0.892** | 0.815 | 0.909 | 65 |
| 23 | `l64_4a818d7951649b77_seq0` | **0.905** | 0.857 | 0.921 | 63 |

### Lecture honnête de ces chiffres

**Le bon résultat** : les 23 designs ont une récupération de contacts entre **0,70 et 0,97** et
un `iptm` Boltz entre **0,85 et 0,96**. Un modèle qui n'a jamais servi à les concevoir replace
le binder sur le même épitope, avec les mêmes résidus face à face, dans **tous** les cas. Les
poses d'AF2 ne sont pas des artefacts d'AF2. C'était la lacune la plus béante du pipeline et
elle est comblée.

**Le résultat gênant, à dire quand même** : mon seuil de « pose confirmée » (récupération
≥ 0,50 **et** iptm ≥ 0,60) **ne discrimine rien**. Les 23 designs le passent largement. Ce
seuil, posé sans calibration, n'apporte donc aucune information de classement — il déplace
simplement tout le monde du groupe 4 vers les groupes 1 et 3. La structure en groupes est
devenue informative sur l'axe pH et **vide** sur l'axe pose.

Les deux plus faibles, qui mériteraient un regard : `5c3ec1903e03c261_seq0` (récupération max
0,696, min 0,571 sur trois échantillons) et `f6d5f550a210fd48_seq0` (0,811 / 0,642). Ce sont
les deux seuls dont un échantillon descend sous 0,65.

**La limite d'orthogonalité, à ne pas enjoliver** : Boltz-2 et AlphaFold2 sont indépendants
**par l'architecture et par les poids**, mais tous deux entraînés sur la PDB. L'accord entre
eux écarte l'hypothèse d'un artefact propre à AF2 ; il n'écarte pas un biais partagé par les
deux modèles, hérité des données. Ce n'est pas une validation expérimentale, et ce n'est pas
un substitut.

### Ordre d'exécution et complétude des paires

L'ordre par paire prévu au plan n'a pas été appliqué, parce que la décision avait été prise de
**ne pas re-scorer les mutants** : la phase 3 les avait déjà tous disqualifiés. Les 23 designs
BindCraft ont donc été traités en une série. Conséquence assumée et rapportée : **les 12 paires
WT–mutant sont incomplètes du côté structural**, et la colonne de coût structural de la phase 5
porte « non mesuré », jamais une estimation.

---

## Une règle de conception, posée puis testée trois fois

C'est le contenu le plus transférable du dossier. Il a été **dérivé, puis mis en échec, puis
retesté correctement**, et les trois étapes sont rapportées parce que la deuxième est ce qui
rend la troisième crédible.

### Le constat de départ

Le critère utilisé jusqu'ici était « un carboxylate du binder à moins de 4 Å d'un azote de
l'imidazole de H409 ». **Onze designs le remplissent. Deux seulement font monter le pKa de
H409** (sur structures AF2). Le critère ne prédit donc pas le mécanisme : trois designs à
géométrie de pont salin quasi idéale (`987fe804e455bc58` à 3,21 Å et 119,5° ;
`5c3ec1903e03c261` à 3,17 Å et 120,7° ; `fd5dae7987a2388d` à 3,36 Å) ont un ΔpKa nul ou
négatif.

### Première hypothèse, réfutée par la mesure

« Les deux azotes de l'imidazole doivent être engagés. » Mesuré : **neuf** designs engagent les
deux azotes à moins de 5,0 Å, et leur ΔpKa va de **+2,84 à −1,19**. En revanche, aucun des dix
designs à zéro azote engagé n'a de ΔpKa positif. L'engagement des deux azotes est donc
**nécessaire, pas suffisant**.

### Seconde hypothèse

Ce qui compte n'est pas que les deux azotes soient approchés, mais qu'ils le soient par **deux
résidus carboxylate différents**. Un seul carboxylate qui pivote entre les deux azotes ne peut
stabiliser qu'une liaison à la fois ; deux résidus distincts peuvent saturer l'imidazole
protoné des deux côtés, ce qui est la condition pour déplacer son équilibre acido-basique.

Mesure, par [bidentate_rule.py](../bidentate_rule.py) : pour chaque design, la meilleure
affectation de deux résidus **distincts**, l'un à ND1 et l'autre à NE2, en retenant la **plus
mauvaise** des deux distances — le goulot d'étranglement du mécanisme bidenté.

### Les trois tests

| test | géométrie mesurée sur | ΔpKa calculé sur | goulots positifs | goulots négatifs | verdict |
|---|---|---|---|---|---|
| **1** | AF2 | AF2 | 3,34 ; 3,35 | 4,54 ; 4,55 ; 6,40 | séparation nette, **1,19 Å** |
| **2** | AF2 | **Boltz** | 3,35 ; 4,54 ; 4,55 | **3,34** ; 6,40 | **recouvrement 1,21 Å** |
| **3** | **Boltz** | **Boltz** | 3,19 ; 3,88 ; 4,03 | 6,23 ; 6,74 ; 7,97 | séparation nette, **2,20 Å** |

**Le test 1 est le plus faible des trois** et c'était la dérivation d'origine : géométrie et
ΔpKa venaient de la **même** structure, donc la corrélation était en partie auto-référentielle.

**Le test 2 la met en échec.** Transporter la géométrie mesurée sur AF2 vers les ΔpKa calculés
sur Boltz détruit la séparation. À ce stade, la règle semblait morte.

**Le test 3 est le test correct, et la règle le passe mieux que l'original.** Un seul
prédicteur des deux côtés, donc la règle est évaluée sur des données qui ne l'ont pas
engendrée : la séparation est **nette avec 2,20 Å de marge**, soit presque deux fois celle du
test 1.

### Ce que l'échec du test 2 apprend

La règle est **locale à la structure**. Le goulot bidenté doit être mesuré sur la structure
dont on évalue le pKa, pas transféré d'un prédicteur à l'autre — deux prédicteurs placent les
chaînes latérales différemment, et c'est la position du carboxylate qui fait le mécanisme.

Ce n'est pas un détail d'implémentation : c'est cohérent avec le cas de
`36dbfc4737a3e59b_seq1`, dont le goulot vaut 3,34 Å sur AF2 (ΔpKa +0,96) mais qui passe dans
le groupe 6,23–7,97 Å sur la structure Boltz (ΔpKa −0,17). **Boltz place ses carboxylates
ailleurs, et le mécanisme disparaît avec eux.** La géométrie et le pKa restent cohérents entre
eux dans chaque prédicteur ; c'est la pose qui diffère.

### Ce que la règle vaut, borné honnêtement

La comparaison informative n'est pas « 3 contre 20 » : dix-sept designs n'ont aucune paire
bidentée possible et forment une catégorie dégénérée. La vraie comparaison est **3 contre 3**
parmi les six designs capables de former une paire (test 3), où un partage propre a une
probabilité de **1/20 = 5 %** sous l'hypothèse nulle. Le test 1 donne 2 contre 3 sur cinq
designs, soit **10 %**.

Deux tests sur deux jeux de structures différents, chacun à p ≈ 0,05–0,10, tous deux passés,
avec des marges de 1,19 et 2,20 Å. **La règle est soutenue, pas établie.** Elle a trois
propriétés qui la rendent utile malgré ça : elle est **falsifiable** — et elle a effectivement
été mise en échec une fois, ce qui a appris quelque chose ; elle se calcule sur une structure
prédite **sans coût GPU** ; et elle est **actionnable**, puisqu'elle dit de placer deux
carboxylates distincts à ~3,4 Å de ND1 et de NE2 plutôt qu'un seul à 2,5 Å. C'est une consigne
de conception différente de celle qui a produit ce lot.

**Elle n'entre dans aucun critère de classement.** Elle est rapportée comme résultat de
méthode.

Elle explique aussi, après coup, pourquoi la campagne de mutants a échoué : chaque mutation
n'ajoutait **qu'un** carboxylate, là où la règle demande une paire coordonnée. Même `S44D`, la
seule mutation saine et à portée, ne produit qu'un ΔpKa de +0,23.

---

## Phase 5 — Analyse appariée WT vs mutant

Squelette identique, une seule variable, contrôle interne. C'est la mesure la plus propre du
projet, et son résultat est négatif.

| squelette | mutation | ΔpKa parent | ΔpKa mutant | ΔΔpKa | étendue rotamères | carboxylate | verdict |
|---|---|---|---|---|---|---|---|
| `a6d2f6834f22e574` seq1 | S44D | −2,85 | **+0,23** | **+3,08** | **0,06** | VERT | **améliore** |
| `f6d5f550a210fd48` seq0 | S38D | −2,89 | −2,25 | +0,64 | 2,16 | INDÉTERMINÉ | neutre (hors portée) |
| `4a818d7951649b77` seq0 | P39D | −1,89 | −1,54 | +0,35 | 0,00 | VERT | neutre (hors portée) |
| `5b295c4d9e1ff73f` seq0 | H23E | −1,73 | −1,51 | +0,22 | 0,14 | INDÉTERMINÉ | neutre (hors portée) |
| `5b295c4d9e1ff73f` seq1 | H23E | −1,85 | −1,71 | +0,14 | 0,02 | INDÉTERMINÉ | neutre (hors portée) |
| `1e7ab6d8f00c9958` seq0 | S28D | −2,25 | −0,75 | +1,50 | 2,33 | INDÉTERMINÉ | dégrade — structurellement impossible |
| `a6d2f6834f22e574` seq0 | S44D | −2,72 | −1,72 | +1,00 | 3,71 | INDÉTERMINÉ | dégrade — structurellement impossible |
| `a6334a3a912c86f1` seq0 | S15D | −2,75 | −1,65 | +1,10 | 3,46 | **ROUGE** | dégrade — carboxylate auto-destructeur |
| `a6334a3a912c86f1` seq1 | S15D | −2,68 | −2,74 | −0,06 | 0,01 | **ROUGE** | dégrade — carboxylate auto-destructeur |
| `cd272a8fd929c7ee` seq0 | N21E | −0,40 | −0,36 | +0,04 | 0,01 | **ROUGE** | dégrade — carboxylate auto-destructeur |
| `cd272a8fd929c7ee` seq1 | N21E | −0,40 | −0,37 | +0,03 | 0,04 | **ROUGE** | dégrade — carboxylate auto-destructeur |
| `4a818d7951649b77` seq1 | F38D | −1,85 | −1,80 | +0,05 | 0,00 | **ROUGE** | dégrade — carboxylate auto-destructeur |

### Combien de mutations améliorent réellement : **une sur douze**

Et il faut la lire précisément. `S44D` sur `a6d2f6834f22e574_seq1` fait passer le ΔpKa de
**−2,85 à +0,23**, soit un ΔΔpKa de **+3,08** avec une étendue sur rotamères de **0,06** —
donc un effet très supérieur à sa propre incertitude de rotamère, et un carboxylate **VERT**
(pKa 3,49–4,07, bien chargé à pH 6,5), avec 2 rotamères sur 3 à portée de H409.

**Mais son ΔpKa absolu reste +0,23, sous le plancher de bruit de 0,5.** La mutation **répare**
un design fortement contre-sélectif en le ramenant à la neutralité. Elle ne crée pas de switch
pH. Le gain de facteur va de 0,695 à 1,19.

### d) Charge nette et répulsion

Chaque substitution vers D ou E abaisse la charge nette du binder d'une unité. Les binders du
lot sont **déjà fortement négatifs** (charge nette de −12 à +2 ; médiane autour de −5), et
l'épitope visé porte lui-même `D323`. Ajouter du carboxylate sur une surface déjà acide, face
à une cible acide, crée de la **répulsion aux deux pH** — ce qui est cohérent avec le fait que
7 des 12 mutations dégradent.

### c) Coût structural — MESURÉ [après report]

Sous l'échéance du 5 octobre, cette colonne portait « non mesuré » : le GPU avait été dépensé
sur les 23 designs BindCraft seulement, les mutants ayant déjà été disqualifiés par PROPKA. Le
report de 24 h a rendu la dépense justifiable, et **les 12 paires sont désormais complètes.**

Les 12 séquences mutées ont été prédites par Boltz-2 — c'est la **première fois qu'un modèle
de structure voit ces séquences**, les structures threadées n'étant que des greffes de chaîne
latérale sur le squelette du parent. La référence de comparaison est nécessairement la pose
AF2 du **parent**, puisqu'un mutant n'a pas de pose AF2. La vérification de séquence tolère
donc exactement **une** différence, pas plus.

| squelette | mutation | récup. parent | récup. mutant | **coût** | verdict pH |
|---|---|---|---|---|---|
| `l62_a6d2f6834f22e574_seq1` | S44D | 0.921 | 0.842 | **-0.079** | ameliore |
| `l61_a6334a3a912c86f1_seq0` | S15D | 0.909 | 0.833 | **-0.076** | degrade |
| `l55_f6d5f550a210fd48_seq0` | S38D | 0.811 | 0.736 | **-0.075** | neutre |
| `l61_a6334a3a912c86f1_seq1` | S15D | 0.851 | 0.791 | **-0.06** | degrade |
| `l64_4a818d7951649b77_seq1` | F38D | 0.892 | 0.846 | **-0.046** | degrade |
| `l61_cd272a8fd929c7ee_seq0` | N21E | 0.96 | 0.92 | **-0.04** | degrade |
| `l73_5b295c4d9e1ff73f_seq1` | H23E | 0.891 | 0.87 | **-0.021** | neutre |
| `l64_4a818d7951649b77_seq0` | P39D | 0.905 | 0.905 | **0.0** | neutre |
| `l62_a6d2f6834f22e574_seq0` | S44D | 0.899 | 0.913 | **0.014** | degrade |
| `l64_1e7ab6d8f00c9958_seq0` | S28D | 0.864 | 0.881 | **0.017** | degrade |
| `l61_cd272a8fd929c7ee_seq1` | N21E | 0.92 | 0.94 | **0.02** | degrade |
| `l73_5b295c4d9e1ff73f_seq0` | H23E | 0.87 | 0.913 | **0.043** | neutre |

**Aucune des 12 mutations ne casse l'interface.** Le coût maximal est de **0,079** de
récupération de contacts, et trois mutations en gagnent. Pour mémoire, les designs BindCraft
eux-mêmes couvrent la plage 0,696 à 0,969 : le coût des mutations est donc **du même ordre que
la dispersion naturelle entre designs**, c'est-à-dire négligeable.

**C'est le résultat qui qualifie l'échec de la campagne de mutants.** Les substitutions sont
structurellement tolérées — elles ne délivrent simplement pas le mécanisme pH. **L'échec est
chimique et électrostatique, pas structural.** Un carboxylate hors de portée (4 cas), un
carboxylate dont le pKa propre monte trop haut pour rester chargé à pH 6,5 (5 cas ROUGE) : ni
l'un ni l'autre ne se voit dans la géométrie de l'interface.

Détail qui tranche le seul arbitrage ouvert du lot : `S44D` sur `a6d2f6834f22e574_seq1`, la
seule mutation au verdict « améliore », porte **le coût structural le plus élevé des douze**
(−0,079). Son gain de pH est sous le plancher de bruit et son coût de pose est le pire du
groupe. Le choix du design BindCraft pour ce squelette (rang 9) s'en trouve conforté plutôt que
simplement conservateur.

### Ce que la campagne de mutants apprend

Elle a échoué, et elle a échoué pour une raison identifiable : **le critère de sélection des
positions était la distance CB→H409, qui ne prédit pas où arrive le carboxylate.** Sur douze
positions choisies, quatre plaçaient le carboxylate hors de portée, une était
structurellement impossible, et sur les trois réellement à portée, deux avaient un pKa propre
trop élevé pour rester chargées à pH 6,5. Le seul succès partiel est celui dont le
carboxylate est sain **et** à portée — ce qui suggère que les deux conditions sont
nécessaires et qu'aucune n'avait été vérifiée avant de générer les séquences.

---

## Phase 6 — Classement et allocation

### a) `out/master_rank.csv`

**35 lignes** : les 23 designs BindCraft, plus les 12 mutants threadés. Aucun candidat
n'est écarté du fichier — les rejetés font partie du funnel publié, et l'onglet « Écartés et
motifs » du classeur dit pour chacun pourquoi il n'est pas dans la soumission.

Colonnes, dans l'ordre du plan, avec quelques ajouts dont la mesure a justifié l'existence :
`palier_pH` (la discrétisation du critère pH), `facteur_global_tous_groupes` et
`facteur_global_groupes_mobiles` (le contrôle non biaisé), `pont_angle_deg`,
`epitope_conservation_frac` et `epitope_residus_divergents` (le proxy de l'objectif n°2),
`facteur_min_rotamere` / `facteur_max_rotamere`.

### b) Récupération de contacts, et non RMSD

Définie comme la fraction des **paires de résidus en contact** de la pose AF2 qui sont
retrouvées dans la pose du modèle orthogonal. Contact = au moins un couple d'atomes lourds à
**5,0 Å** ou moins. Asymétrique à dessein : la question est « la pose d'AF2 est-elle
retrouvée », pas « les deux poses sont-elles identiques ».

Le RMSD a été écarté pour la raison énoncée au plan et qui est juste : il pardonne une
rotation du binder qui reste sur la même zone alors que plus aucun résidu ne se fait face, et
il pénalise une translation rigide qui conserve toutes les paires.

Un piège de numérotation est traité explicitement dans
[contact_recovery.py](../contact_recovery.py) : Boltz renumérote chaque chaîne à partir de 1,
alors que la cible porte la numérotation PDB 309–506. Le décalage n'est pas supposé, il est
**déduit puis vérifié** en comparant les séquences résidu par résidu, et le calcul est refusé
si elles ne concordent pas.

### c) Classement lexicographique

Quatre groupes, selon la consigne :

| groupe | définition | effectif |
|---|---|---|
| 1 | mécanisme pH **robuste** et pose confirmée | **1** |
| 2 | mécanisme pH robuste, pose contestée ou non mesurée | **0** |
| 3 | pas de mécanisme pH robuste, pose confirmée | **27** |
| 4 | le reste | **7** |

[après report] Le groupe 1 ne contient plus **qu'un** design. Avant la mesure inter-espèces
ils étaient deux ; `36dbfc4737a3e59b_seq1` en est sorti parce que son mécanisme ne se
reproduit ni sur une autre structure ni chez la souris. Le groupe 4 ne contient que les
**7 mutants rétrogradés par la règle** — carboxylate ROUGE (5) ou impossibilité structurale
(2). Le groupe 2 est vide parce que le seul design à mécanisme robuste a **aussi** une pose
confirmée.

**Le critère pH a quatre paliers depuis la mesure inter-espèces**, et non trois : un mécanisme
n'est *robuste* que s'il est positif sur les trois mesures (AF2 humain, Boltz humain, Boltz
souris). Le palier **« mécanisme non reproductible »** recueille les designs positifs sur au
moins une mesure mais pas toutes, et se classe au-dessus du neutre.

**Mais cette structure est moins informative qu'elle en a l'air.** Les 23 designs BindCraft
passent le seuil de pose confirmée, et largement. La répartition 2 / 21 reflète donc
uniquement l'axe pH : sur l'axe pose, le critère ne sépare personne. Voir phase 4.

**À l'intérieur d'un groupe**, l'ordre est : palier pH, puis — et seulement dans le palier à
mécanisme — le facteur de sélectivité, puis la conservation de l'épitope (objectif n°2), puis
`i_pAE` et `i_pTM` (objectif n°3).

**Pourquoi le facteur ne départage pas dans les paliers « neutre » et « contre-sélectif ».**
PROPKA se trompe d'environ une unité de pKa. Entre un design à ΔpKa −0,01 et un autre à
−0,40, il n'y a rien à lire. Les classer sur cette différence serait inventer un ordre dans du
bruit. C'est pourquoi, dans le lot soumis, le rang 3 (facteur 0,875) passe devant les rangs 5
et 6 (facteurs 0,993 et 1,022) : tous trois sont dans le palier « neutre », et ce qui les
sépare est la conservation de l'épitope — 0,870 contre 0,789 et 0,778. C'est voulu, et c'est
l'ordre de priorité du règlement appliqué littéralement.

**Anti-dégât respecté** : aucun design à mécanisme pH n'est passé sous un design sans
mécanisme, et aucun n'a été rétrogradé au motif qu'un modèle orthogonal l'aimait moins. Le
seul déclassement du lot, celui de `36dbfc4737a3e59b_seq1`, repose sur **une mesure de pH**
— son ΔpKa sur deux autres structures — et non sur une préférence de pose.

**Règle mutant appliquée** : seuls les ROUGE et les impossibilités structurales descendent au
groupe 4. Les cinq mutants INDÉTERMINÉS gardent leur rang avec la mention, comme demandé — le
threading ne tranche pas, et un candidat ne doit pas être perdu sur un artefact de rotamère.

### d) Les deux allocations, chiffrées

| allocation | designs | squelettes couverts | retenue ? |
|---|---|---|---|
| **couverture maximale** | **13** | **13 / 13** | **oui** |
| paires appariées | 14 | 7 | non |

**[autonomie] Couverture maximale retenue**, pour la raison de la phase 0f : le règlement ne
dit pas comment l'unicité est évaluée entre les designs d'un même participant. Soumettre un WT
et son mutant ponctuel, à 98–99 % d'identité, serait un risque de rejet des deux plutôt qu'une
stratégie. L'allocation retenue couvre **les 13 squelettes**, et l'identité maximale entre
deux lignes soumises tombe à **25,4 %** — très loin de tout seuil d'unicité plausible.

### Règle d'éligibilité des mutants, et son effet

Posée dans [build_submission.py](../build_submission.py) : un mutant n'entre que si son
mécanisme est **robuste** et son carboxylate **VERT**. **Aucun des 12 ne remplit les deux
conditions.** Le plus proche, `S44D` sur `a6d2f6834f22e574_seq1`, a bien un carboxylate VERT
et un gain apparié net (+3,08 unités), mais son ΔpKa absolu de +0,23 est sous le plancher de
bruit, donc son mécanisme est classé « absent ».

Conséquence : pour le squelette `a6d2f6834f22e574`, c'est le design BindCraft `seq0` qui est
soumis (rang 9), alors que le mutant `S44D seq1` est le seul membre du squelette qui ne soit
pas contre-sélectif. **C'est le seul arbitrage du lot où un autre choix serait défendable**,
et il est signalé comme tel : échanger le rang 9 contre la séquence de `S44D seq1` troquerait
un design filtré par BindCraft contre une greffe non relaxée jamais passée par un prédicteur
de structure, pour gagner un ΔpKa sous le bruit. Le choix conservateur a été fait ; il est
réversible en une ligne.

---

## Phase 7 — Fichiers de sortie et contrôles

### a) `submission/egfr_challenge1_submission.csv`

**13 lignes**, ordonnées par classement, meilleur design en première ligne. Trois colonnes
`name, sequence, molecule_class`, séparateur virgule, UTF-8 sans BOM, `molecule_class` =
`protein` partout. **Le quota de 20 n'est pas atteint et n'a pas été complété** : il n'existe
que 13 squelettes indépendants, et le lot n'est pas gonflé avec des frères.

| rang | design | palier pH | ΔpKa AF2 / Bz-H / Bz-M | épitope souris | i_pTM | aa |
|---|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | mecanisme robuste | 2.84 / 2.27 / 2.41 | 0.897 | 0.83 | 94 |
| 2 | `l63_987fe804e455bc58_seq1` | mecanisme non reproductible | -0.29 / 0.97 / -0.7 | 0.935 | 0.83 | 63 |
| 3 | `l59_36dbfc4737a3e59b_seq1` | mecanisme non reproductible | 0.96 / -0.17 / -0.38 | 0.839 | 0.77 | 59 |
| 4 | `l61_cd272a8fd929c7ee_seq0` | neutre | -0.4 / -0.39 / -0.5 | 0.92 | 0.84 | 61 |
| 5 | `l58_fd5dae7987a2388d_seq1` | neutre | -0.07 / -0.12 / -0.13 | 0.812 | 0.82 | 58 |
| 6 | `l57_9526c9216eb7d6db_seq0` | neutre | 0.03 / -0.41 / -2.62 | 0.479 | 0.76 | 57 |
| 7 | `l92_5c3ec1903e03c261_seq1` | contre-selectif | -0.86 / -1.82 / -1.84 | 0.929 | 0.82 | 92 |
| 8 | `l62_a6d2f6834f22e574_seq1` | contre-selectif | -2.85 / -3.07 / -2.83 | 0.921 | 0.8 | 62 |
| 9 | `l64_4a818d7951649b77_seq0` | contre-selectif | -1.89 / -2.16 / -2.78 | 0.889 | 0.75 | 64 |
| 10 | `l61_a6334a3a912c86f1_seq0` | contre-selectif | -2.68 / -3.05 / -2.63 | 0.833 | 0.81 | 61 |
| 11 | `l64_1e7ab6d8f00c9958_seq0` | contre-selectif | -2.25 / -2.3 / -2.12 | 0.831 | 0.85 | 64 |
| 12 | `l73_5b295c4d9e1ff73f_seq0` | contre-selectif | -1.73 / -1.74 / -2.18 | 0.804 | 0.8 | 73 |
| 13 | `l55_f6d5f550a210fd48_seq0` | contre-selectif | -2.89 / -2.56 / -2.86 | 0.604 | 0.81 | 55 |

**Un seul design porte un mécanisme pH robuste.** Les rangs 2 et 3 le portent de façon non
reproductible, les rangs 4 à 6 sont neutres, les rangs 7 à 13 contre-sélectifs. L'ordre du CSV
transmet cette information au sélecteur, et le dossier ne prétend pas autre chose.

Les `name` reprennent les identifiants de design du dépôt
(`egfr-dIII-prod02_denovo_l63_987fe804e455bc58_seq1`), ce qui relie chaque ligne à sa
trajectoire, ses métriques et ses structures. Aucun champ ne contient de texte adressé à un
lecteur.

### b) `submission/egfr_analysis.xlsx`

Copie de travail, **ne part pas à Adaptyv**. Onglets : Soumission, Classement complet, Paires
WT-mutant, PROPKA pKa des DE, PROPKA scan non biaisé, Géométrie ponts WT, Géométrie mutants,
Orthogonal détail, Méthode et limites, Contrôles, Écartés et motifs. En-têtes figés, filtres
automatiques, largeurs ajustées, et mise en forme conditionnelle sur ΔpKa (vert au-dessus de
+0,5, rouge sous −0,5), sur la récupération de contacts et sur le verdict carboxylate
(vert / rouge / orange pour INDÉTERMINÉ).

### c) Contrôles — tous passés

| contrôle | résultat | détail |
|---|---|---|
| nombre de lignes ≤ quota | **OK** | 13 pour un plafond de 20, non forcé à l'égalité |
| aucun doublon de séquence | **OK** | 0 doublon |
| identité de séquence maximale entre deux lignes | **OK** | **23,4 %**, entre `692deac2f1034bb6_seq0` et `4a818d7951649b77_seq0` |
| longueurs dans les bornes 10–250 | **OK** | observé 55–94 aa |
| aucune cystéine | **OK** | 0 séquence sur 13 — vérifié caractère par caractère, pas supposé |
| aucun caractère non standard, espace ou retour de ligne | **OK** | aucun |
| relecture pandas, encodage UTF-8 | **OK** | 13 lignes, colonnes `['name', 'sequence', 'molecule_class']` |
| chaque séquence correspond à sa structure | **OK** | les 13 séquences relues dans la chaîne B du PDB concordent |

Le dernier contrôle est le plus utile des huit : il relit la séquence du binder **dans le
fichier de structure** et la compare à celle du CSV, ce qui exclut une désynchronisation entre
la séquence soumise et la structure sur laquelle tous les pKa ont été calculés.

---

## Cross-réactivité souris et robustesse au prédicteur [après report]

C'est l'ajout le plus conséquent du temps supplémentaire, et **il a changé le classement**.

### Ce qui manquait

L'objectif n°2 du challenge — la même séquence doit reconnaître P00533 **et** Q01279 — n'était
approché que par un **proxy de séquence** : la fraction des résidus de cible contactés
identiques chez la souris. Ce proxy ignore la conformation locale murine et ne dit rien du
mécanisme pH chez la souris. Aucune structure du domaine III murin n'avait jamais été obtenue
(action 5 de CLAUDE.md, restée ouverte depuis le 2 octobre).

### La cible murine, dérivée deux fois

[mouse_target.py](../mouse_target.py) construit le domaine III de Q01279 par **deux chemins
qui ne partagent aucune étape**, et refuse d'écrire le fichier s'ils divergent :

- **A** — alignement de la séquence Q01279 complète (1210 aa, cache UniProt) sur la séquence
  du domaine III humain lue dans le PDB cible ;
- **B** — la colonne `aa_mouse` de `data/egfr_residues.csv`, issue d'un alignement produit
  séparément par `egfr_epitope_map.py`, restreinte aux PDB 309–506.

**Les deux concordent.** Pourquoi ce garde-fou : se tromper de région aurait donné une cible
murine plausible mais fausse, que Boltz-2 aurait repliée avec une confiance élevée, et la
« cross-réactivité mesurée » aurait été un artefact que rien en aval n'aurait signalé.

| | |
|---|---|
| domaine III murin | **UniProt 333–530**, 198 résidus |
| identité humain/souris | **173/198 = 87,4 %** |
| indels dans la fenêtre | **aucun** — donc la numérotation PDB 309–506 s'applique aux deux espèces et le mapping de contacts est l'identité |
| **H409** | **conservée** — le mécanisme Route 2 est transposable |
| His de la cible | humain {334, 346, 359, 394, 409, 483} ; souris {334, 346, 394, 409, 480} |
| 25 positions divergentes | S324T, N337Y, S340A, R353K, **H359R**, Q366R, D369E, E388D, R390W, S418G, K443R, S460P, G461N, I467M, S468N, G471A, N473K, S474D, T478V, G479N, Q480H, **H483N**, A484P, P488S, R503Q |

`H359R` confirme après coup le coldspot `A359` : c'est bien une His humaine qui devient une Arg
chez la souris. MSA murine propre récupérée séparément (3502 séquences) — réutiliser celle de
l'humain aurait injecté l'alignement de la mauvaise protéine.

### Le contrôle qui rend la comparaison lisible

Le ΔpKa publié jusqu'ici venait des structures **AF2/BindCraft**. Le comparer à un ΔpKa murin
issu de **Boltz-2** aurait mélangé l'effet d'espèce et l'effet de prédicteur. PROPKA a donc
été relancé sur les structures **Boltz humaines** aussi, et la comparaison d'espèce se fait
Boltz contre Boltz. Trois mesures indépendantes par design en résultent.

| rang | design | ΔpKa AF2-H | ΔpKa Boltz-H | ΔpKa Boltz-M | iptm H → M | épitope souris | verdict |
|---|---|---|---|---|---|---|---|
| 1 | `l94_692deac2f1034bb6_seq0` | 2.84 | 2.27 | 2.41 | 0.93 → 0.914 | 0.897 | mecanisme robuste |
| 2 | `l63_987fe804e455bc58_seq1` | -0.29 | 0.97 | -0.7 | 0.946 → 0.94 | 0.935 | mecanisme non reproductible |
| 3 | `l63_987fe804e455bc58_seq0` | -0.22 | 1.04 | -0.11 | 0.948 → 0.937 | 0.906 | mecanisme non reproductible |
| 4 | `l59_36dbfc4737a3e59b_seq1` | 0.96 | -0.17 | -0.38 | 0.938 → 0.928 | 0.839 | mecanisme non reproductible |
| 5 | `l61_cd272a8fd929c7ee_seq0` | -0.4 | -0.39 | -0.5 | 0.941 → 0.933 | 0.92 | neutre |
| 6 | `l61_cd272a8fd929c7ee_seq1` | -0.4 | -0.36 | -0.37 | 0.938 → 0.942 | 0.9 | neutre |
| 7 | `l59_36dbfc4737a3e59b_seq0` | -1.19 | -0.41 | -0.33 | 0.925 → 0.928 | 0.897 | neutre |
| 8 | `l58_fd5dae7987a2388d_seq1` | -0.07 | -0.12 | -0.13 | 0.936 → 0.938 | 0.812 | neutre |
| 9 | `l58_fd5dae7987a2388d_seq0` | -0.01 | -0.11 | -2.65 | 0.937 → 0.944 | 0.806 | neutre |
| 10 | `l57_9526c9216eb7d6db_seq0` | 0.03 | -0.41 | -2.62 | 0.92 → 0.851 | 0.479 | neutre |
| 11 | `l57_9526c9216eb7d6db_seq1` | -0.23 | 0.45 | -2.48 | 0.913 → 0.815 | 0.478 | neutre |
| 13 | `l92_5c3ec1903e03c261_seq1` | -0.86 | -1.82 | -1.84 | 0.91 → 0.879 | 0.929 | contre-selectif |
| 14 | `l62_a6d2f6834f22e574_seq1` | -2.85 | -3.07 | -2.83 | 0.935 → 0.921 | 0.921 | contre-selectif |
| 15 | `l92_5c3ec1903e03c261_seq0` | -2.36 | -2.96 | -2.95 | 0.866 → 0.856 | 0.911 | contre-selectif |
| 16 | `l64_4a818d7951649b77_seq0` | -1.89 | -2.16 | -2.78 | 0.921 → 0.911 | 0.889 | contre-selectif |
| 17 | `l61_a6334a3a912c86f1_seq0` | -2.68 | -3.05 | -2.63 | 0.948 → 0.939 | 0.833 | contre-selectif |
| 18 | `l64_1e7ab6d8f00c9958_seq0` | -2.25 | -2.3 | -2.12 | 0.932 → 0.932 | 0.831 | contre-selectif |
| 19 | `l62_a6d2f6834f22e574_seq0` | -2.72 | -3.57 | -3.16 | 0.949 → 0.93 | 0.812 | contre-selectif |
| 20 | `l61_a6334a3a912c86f1_seq1` | -2.75 | -2.47 | -2.5 | 0.957 → 0.948 | 0.806 | contre-selectif |
| 21 | `l73_5b295c4d9e1ff73f_seq0` | -1.73 | -1.74 | -2.18 | 0.932 → 0.924 | 0.804 | contre-selectif |
| 22 | `l64_4a818d7951649b77_seq1` | -1.85 | -2.47 | -2.52 | 0.909 → 0.873 | 0.785 | contre-selectif |
| 23 | `l73_5b295c4d9e1ff73f_seq1` | -1.85 | -2.53 | -1.72 | 0.93 → 0.926 | 0.761 | contre-selectif |
| 24 | `l55_f6d5f550a210fd48_seq0` | -2.89 | -2.56 | -2.86 | 0.848 → 0.836 | 0.604 | contre-selectif |

### Trois résultats, dont deux inattendus

**1. Le design de rang 1 tient sur les trois axes.** `692deac2f1034bb6_seq0` : ΔpKa de
**+2,84** (AF2 humain), **+2,27** (Boltz humain), **+2,41** (Boltz souris). Il survit donc à
un changement de **prédicteur de structure** et à un changement d'**espèce**, avec un `iptm`
murin de 0,914 et 89,7 % de son épitope humain retrouvé chez la souris. C'est le seul design
du lot dans ce cas, et c'est le meilleur résultat du projet.

**2. L'ancien rang 2 ne résiste pas, et il a été déclassé.** `36dbfc4737a3e59b_seq1` passait
pour le second porteur de mécanisme avec un ΔpKa de **+0,96**. Sur la structure Boltz il tombe
à **−0,17**, et chez la souris à **−0,38**. **Son mécanisme était une propriété de la structure
AF2, pas de la séquence.** C'est précisément ce qu'un contrôle orthogonal doit attraper, et ça
justifie à lui seul le coût de l'étape.

**3. Deux designs gagnent un mécanisme que l'AF2 ne voyait pas — et le perdent chez la
souris.** `987fe804e455bc58` seq0 et seq1 passent de −0,22 / −0,29 sur AF2 à **+1,04** et
**+0,97** sur Boltz, puis à −0,11 et −0,70 chez la souris. Le mécanisme n'est donc ni
reproductible entre prédicteurs, ni conservé entre espèces.

### Accord des prédicteurs, chiffré

Sur les 23 designs, l'écart ΔpKa(Boltz) − ΔpKa(AF2) va de **−1,13 à +1,26**, moyenne −0,10,
médiane −0,10. Le **verdict** de mécanisme (seuil 0,5) concorde sur **20 designs sur 23**.

Autrement dit : le ΔpKa est globalement reproductible, mais les trois désaccords tombent
**exactement sur les cas limites** — ceux dont le ΔpKa est compris entre −0,3 et +1,1. Or ce
sont précisément les designs qu'on serait tenté de promouvoir. L'incertitude de PROPKA, environ
une unité de pKa, suffit à faire basculer un design d'un palier à l'autre, et c'est la raison
pour laquelle le critère pH a été durci.

### Le durcissement du critère, et son effet

Dans [rank_designs.py](../rank_designs.py), un mécanisme n'est **robuste** que s'il est positif
sur **les trois mesures** : AF2 humain, Boltz humain, Boltz souris. Un palier intermédiaire,
**« mécanisme non reproductible »**, recueille les designs positifs sur au moins une mesure mais
pas sur toutes. Il se classe au-dessus du neutre — un design positif sur une structure reste un
meilleur pari pH qu'un design positif sur aucune — mais la mention voyage avec lui.

| palier | effectif (designs BindCraft) |
|---|---|
| mécanisme **robuste** | **1** |
| mécanisme **non reproductible** | 3 |
| neutre | 4 |
| contre-sélectif | 15 |

### La cross-réactivité elle-même : bonne, mais pas uniforme

| | |
|---|---|
| Δ iptm souris − humain | **−0,098 à +0,007**, moyenne **−0,016** |
| épitope humain retrouvé chez la souris | **0,478 à 0,935** |

**Le passage à la souris ne coûte pratiquement rien en confiance de liaison** — à 87,4 %
d'identité sur le domaine, c'est attendu, et c'est le résultat qui valide le choix d'épitope
fait sur la conservation mesurée plutôt que sur l'épitope documenté du cétuximab.

Mais trois designs changent de mode de liaison chez la souris :
`9526c9216eb7d6db` seq0 (**0,479**) et seq1 (**0,478**), et `f6d5f550a210fd48_seq0` (**0,604**).
Ils perdent la moitié de leur épitope. Le proxy de séquence ne les distinguait **pas** : leur
conservation d'épitope valait 0,778 et 0,800, dans la moyenne du lot. **C'est la démonstration
directe que le proxy était insuffisant**, et c'est pourquoi l'objectif n°2 est désormais classé
sur la mesure, le proxy ne servant plus qu'en départage pour les mutants, non prédits contre la
souris.

### Ce que ça ne dit pas

Boltz-2 prédit un complexe avec une cible murine dont la **structure n'a jamais été
déterminée expérimentalement** — elle est elle-même prédite, à partir d'une séquence
correctement extraite. L'absence d'indel et 87,4 % d'identité rendent le repliement murin très
probablement superposable à l'humain, mais « très probablement » n'est pas « mesuré ». Et deux
prédicteurs d'accord restent deux prédicteurs, tous deux entraînés sur la PDB.

---

## Limites et ce qui n'a pas été mesuré

Cette section existe pour qu'un tiers puisse juger la soumission sans avoir à deviner ce qui
n'a pas été fait.

### Ce qui est adossé à une mesure externe

Les bornes du domaine III (CATH-Gene3D `G3DSA:3.80.20.20`), l'offset PDB→UniProt de +24
(`_struct_ref_seq` et balayage, concordants), les 13 séquons et 25 ponts disulfure (UniProt),
la sûreté de la troncature 309–506 (écart de SASA de 0,0 Å² sur les hotspots), le critère de
pont salin de 4,0 Å entre atomes chargés (Barlow & Thornton 1983), les SASA de référence de
chaîne latérale (Tien et al. 2013), l'identité humain/souris résidu par résidu (alignement
P00533 / Q01279).

### Seuils posés, non calibrés

| seuil | valeur | statut |
|---|---|---|
| plancher de bruit de pKa | 0,5 unité | convention adossée à l'erreur typique de PROPKA, non mesurée sur ce système |
| carboxylate auto-destructeur | pKa > 5,0 | posé par le plan de travail |
| clash entre atomes lourds | 2,2 Å | posé par le plan de travail |
| contact trop court sur structure prédite | 2,60 Å | posé — borne basse de la plage usuelle d'un pont salin O···N |
| rotamères distincts | 0,30 Å d'étendue de centroïde | **posé par moi**, sans référence |
| contact inter-résidus | 5,0 Å | convention usuelle |
| pose confirmée | récupération ≥ 0,50 **et** iptm ≥ 0,60 | **posés par moi, non calibrés** — aucun jeu de référence de binders de novo validés expérimentalement contre cette cible n'existe pour les calibrer |
| règle bidentée | goulot ≤ ~3,4 Å (AF2) / ~4,0 Å (Boltz) | **dérivée a posteriori**, puis testée sur un second jeu de structures : passe avec 2,20 Å de marge, échoue si la géométrie est transportée d'un prédicteur à l'autre. Soutenue, non établie |
| mécanisme pH **robuste** | ΔpKa > 0,5 sur **les trois** mesures | durcissement introduit après la mesure inter-espèces, parce que le critère sur une seule structure promouvait un design qui ne se reproduit pas |

Le seuil de pose confirmée s'est révélé **non discriminant** : les 23 designs BindCraft le
passent. Il ne porte donc aucune information de classement, ce qui est dit en phase 4 plutôt
que masqué par une répartition en groupes d'apparence informative.

La règle bidentée est la plus fragile des trois, et elle n'entre dans **aucun** critère de
classement — elle est rapportée comme résultat de méthode, pas utilisée comme filtre.

### Non mesuré du tout

- **Le design multicible n'a jamais servi.** C'était la fonctionnalité qui avait motivé le
  passage à BindCraft 2.0, et aucune campagne n'a été lancée contre les deux espèces
  simultanément. La cross-réactivité a été **évaluée** [après report] mais jamais
  **optimisée** : les designs n'ont été conçus que contre l'humain. Qu'ils reconnaissent la
  souris est un constat, pas un résultat de conception.
- **La structure murine est elle-même prédite.** Le domaine III de Q01279 n'a pas de structure
  expérimentale ; la séquence a été correctement extraite et vérifiée deux fois, mais son
  repliement est inféré. L'absence d'indel et 87,4 % d'identité le rendent très probablement
  superposable à l'humain — « très probablement » n'est pas « mesuré ».
- **La conformation étendue.** 6ARU est replié. Toute SASA et toute géométrie calculées ici en
  héritent.
- **La corréférence de face des hotspots.** Les 16,73 Å d'étendue CA disent que les quatre
  hotspots sont proches, pas qu'ils regardent du même côté.
- **La pseudo-vraisemblance ESM-2 est désormais mesurée [après report], et elle ne sert à
  rien** — ce qui était prévisible et se dit quand même. `esm2_t33_650M_UR50D`, marginales
  masquées, normalisée par la longueur : **−2,72 à −1,93 par résidu, médiane −2,21** sur les
  35 séquences. Elle reste **hors classement par construction**, puisqu'elle mesure la
  ressemblance aux protéines naturelles alors que le règlement exige la nouveauté ; classer
  dessus favoriserait les designs les moins nouveaux. Lue comme **détecteur d'anomalie**, son
  seul usage légitime, elle ne remonte rien : la plage ne couvre que 0,8 unité log et aucune
  séquence ne se détache. Les quatre valeurs les plus basses appartiennent toutes au squelette
  `a6d2f6834f22e574` (designs BindCraft et mutants), qui est par ailleurs le plus contre-sélectif du lot —
  coïncidence notée, **pas exploitée**. La mention d'un usage par Adaptyv au round 2 n'a **pas**
  été vérifiée.
- **Le coût structural des mutations.** Le GPU est allé aux 23 designs BindCraft, pas aux
  mutants, ceux-ci ayant déjà été disqualifiés par PROPKA. Les paires sont donc incomplètes
  du côté structural, par décision assumée, et la colonne porte « non mesuré ».
- **Rien d'expérimental.** Tout ce dossier repose sur des structures prédites et des modèles
  empiriques. Deux prédicteurs d'accord restent deux prédicteurs, tous deux entraînés sur la
  PDB.

### Limites de méthode à ne pas enjoliver

- **Les structures des mutants ne sont pas relaxées.** Ce sont des greffes de chaîne latérale
  sur squelette rigide. Les pKa **absolus** qui en sortent sont grossiers. La comparaison
  **appariée**, qui partage le squelette, est en revanche la plus propre possible — c'est
  pour ça que le résultat rapporté est un ΔΔpKa et non un pKa.
- **PROPKA n'est pas une simulation.** C'est un modèle empirique sur descripteurs
  géométriques, calibré sur des pKa mesurés de protéines naturelles, appliqué ici à des
  interfaces de novo **prédites**. Deux sources d'erreur se composent : celle de la structure
  et celle du modèle de pKa.
- **Le facteur de sélectivité global suppose les sites indépendants.** Ils ne le sont pas
  quand deux groupes se touchent, et le produit cumule une erreur par groupe. C'est pourquoi
  la version restreinte aux groupes qui bougent de plus que le bruit est rapportée à côté.
- **La règle bidentée a été mise en échec une fois.** Transporter la géométrie d'un
  prédicteur vers les pKa d'un autre détruit sa séparation. Elle n'est valable que mesurée sur
  la structure dont on évalue le pKa. L'échec est rapporté parce qu'il borne l'usage de la
  règle.
- **Le switch demandé n'est pas celui qui est atteint.** Le règlement demande *« no detectable
  binding at pH 7.4 »*, un basculement binaire. Ce qui est mesuré ici est un **décalage de
  pKa** sur un seul résidu, qui prédit un rapport d'affinité d'un facteur de l'ordre de
  quelques unités. Un seul design du lot porte ce mécanisme de façon marquée. L'écart entre
  « facteur ~5 » et « plus de liaison détectable » n'est pas comblé, et rien dans ce dossier
  ne prétend le combler.
- **Les designs ne sont pas reproductibles à l'identique.** Rejouer le même commit ne redonne
  pas les mêmes séquences : les réductions GPU de JAX ne sont pas déterministes au bit près et
  une trajectoire de gradient est chaotique. Ce qui est reproductible est la **méthode**, pas
  les séquences.

### Erreurs commises cette nuit, et conservées au journal

1. **Le proxy CB→H409 pour choisir les positions de mutation.** Il ne prédit pas où arrive le
   carboxylate. Quatre mutations sur douze ont placé le groupe acide hors de portée. Le seuil
   réel est d'environ 4,3 Å de CB et il n'avait pas été posé.
2. **Le tri des rotamères par contact le plus serré.** Aveugle quand le contact minimal est
   porté par le CB, qui ne bouge pas d'un rotamère à l'autre. Détecté en mesurant l'étendue du
   centroïde du carboxylate ; un mutant (`P39D`) a trois rotamères quasi identiques.
3. **La moyenne du facteur de sélectivité sur les rotamères.** Le facteur est une fonction non
   linéaire du pKa : en faire la moyenne donnait un nombre incohérent avec le ΔpKa moyen
   rapporté à côté. Corrigé en recalculant le facteur depuis le ΔpKa moyen.
4. **Un diagnostic attribué à la mauvaise cause.** L'échec initial de Boltz-2 a d'abord été
   imputé à une MSA malformée. C'était faux : le diagnostic a montré un
   `ModuleNotFoundError: cuequivariance_torch`. La correction de la MSA est juste en soi mais
   n'a rien réparé, et le commentaire qui le prétendait a été corrigé dans le code.
5. **Un run GPU lancé sans vérifier un seul complexe d'abord.** Sept minutes de L40S à
   produire zéro prédiction avec un code retour 0. Un entrypoint `diagnose` a été ajouté pour
   que ça n'arrive qu'une fois, et `predict` imprime désormais la sortie de boltz dès qu'une
   prédiction manque, **quel que soit le code retour**.
6. **Des vérifications `pgrep` qui se détectaient elles-mêmes.** `pgrep -f "modal run
   modal_boltz2"` correspond à la ligne de commande du test lui-même, d'où plusieurs
   « run en cours » faux. Un test d'état qui ne peut pas renvoyer « terminé » n'est pas un test.
7. **Une règle de conception annoncée avant d'avoir été testée hors de ses propres données.**
   La séparation nette du premier test reposait sur une géométrie et un pKa issus de la même
   structure. Il a fallu un second jeu de structures pour savoir ce que la règle valait — et
   un test intermédiaire pour découvrir qu'elle est locale à la structure.

---

## Décisions prises en autonomie

L'auteur dormait. Ces choix n'ont pas été arbitrés par un humain.

| # | décision | raison |
|---|---|---|
| 1 | **Allocation « couverture maximale »**, 13 designs, un par squelette | Le règlement ne dit pas comment l'unicité est évaluée entre designs d'un même participant (phase 0f). La consigne était de n'ajouter frères ou mutants qu'en cas de confirmation explicite. Elle n'existe pas. |
| 2 | **GPU dépensé sur les 23 designs BindCraft, pas sur les mutants** | La phase 3 avait déjà disqualifié les 12 mutants. Dépenser du GPU sur eux aurait acheté une colonne pour des candidats inéligibles. Conséquence acceptée : phase 5c non mesurée. |
| 3 | **Règle d'éligibilité des mutants** (mécanisme robuste **et** carboxylate VERT) | Un mutant sans mécanisme ne gagne rien sur l'objectif n°1 et perd le filtrage BindCraft et la validation orthogonale de son parent. Aucun des 12 ne passe. |
| 4 | **Critère pH discrétisé en trois paliers** | Classer sur des écarts de ΔpKa inférieurs à l'erreur de PROPKA serait lire un ordre dans du bruit. |
| 5 | **Trois échantillons de diffusion au lieu de trois graines** | Le tronc de Boltz-2 est déterministe à graine fixée ; le relancer trois fois coûterait trois fois le calcul pour un tronc identique. Ce qui varie entre poses est l'étape de diffusion. |
| 6 | **ESM-2 non calculé** | Hors classement par construction. Le temps est allé au re-scoring orthogonal, qui entre dans le classement. |
| 7 | **Design BindCraft retenu pour `a6d2f6834f22e574`** plutôt que son mutant `S44D seq1` | Le mutant est le seul membre non contre-sélectif du squelette, mais son gain est sous le bruit. [après report] Son coût structural mesuré est de plus **le pire des douze** (−0,079), ce qui transforme un choix conservateur en choix étayé. |
| 8 | [après report] **Durcissement du critère pH** : mécanisme robuste = positif sur AF2 humain, Boltz humain **et** Boltz souris | Le critère sur une seule structure classait `36dbfc4737a3e59b_seq1` au rang 2 ; deux mesures indépendantes le contredisent. Un mécanisme qui ne survit pas au changement de structure est une propriété de la structure. |
| 9 | [après report] **Objectif n°2 classé sur la mesure**, le proxy de séquence passant en départage | L'épitope retrouvé chez la souris par un modèle indépendant est une évidence strictement plus forte. Et le proxy était démontrablement insuffisant : il ne distinguait pas les trois designs qui perdent la moitié de leur épitope chez la souris. |

---
