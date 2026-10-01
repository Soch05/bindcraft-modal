# Lire `egfr_patches.csv` pour choisir un site

Ce fichier dit ce que chaque colonne mesure, dans quelle unité, et **ce qu'elle ne capture
pas**. Il se termine par une procédure en cinq étapes qui aboutit à 3-4 sites.

Produit par [`egfr_epitope_map.py`](../egfr_epitope_map.py) ; le journal des décisions est
dans [`NOTES.md`](../NOTES.md), l'architecture dans [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

Une ligne = un patch = un résidu exposé du domaine III pris comme centre, plus tous les
résidus exposés dont l'atome d'ancrage (CB, ou CA si pas de CB) est à ≤ 11 Å de celui du
centre. 93 lignes. Les patches se recouvrent massivement : **ce ne sont pas 93 sites.**

---

## Les deux familles de colonnes

Chaque grandeur existe en deux versions.

| | membres retenus | à utiliser pour |
|---|---|---|
| nom nu (`sasa_apolar`) | exposés **et** dans le domaine III | rien, sauf comparer à l'historique |
| suffixe `_full` | exposés, **toute la chaîne A** | **tout classement** |

Le centre reste dans le domaine III dans les deux cas. La différence ne porte que sur les
membres. `n_truncated` = nombre de membres que la version masquée excluait.

**Pourquoi `_full` est la bonne version** : la surface d'une protéine ne s'arrête pas à une
borne de domaine. Masquer les membres ampute les patches de bord sans rien dire. Quatre
patches étaient ainsi sous-évalués :

| patch | apolaire masquée → `_full` | écart |
|---|---|---|
| R310 | 229 → 454 Å² | +98 % |
| K311 | 349 → 536 Å² | +54 % |
| C502 | 399 → 609 Å² | +53 % |
| N337 | 356 → 497 Å² | +40 % |

**Avertissement sur ces quatre** : ils portent **tous** un séquon de N-glycosylation parmi
leurs membres (`min_glyc = 0,0`), et leur `dFab` est de 12 à 18 Å. Ce que le masque cachait
était glycosylé et hors de la surface ligand-compétitive. Lire `_full` ne suffit pas, il
faut lire `min_glyc` avec.

---

## Colonne par colonne

### `centre`, `centre_num`, `centre_uniprot`
Le résidu central, en acide aminé + numéro PDB, puis numéro PDB seul, puis numéro UniProt.
**Offset PDB → UniProt = +24**, lu dans `_struct_ref_seq` du mmCIF *et* retrouvé par
balayage ; le script s'arrête si les deux divergent.
*Ne capture pas* : rien d'autre que l'étiquette. Le centre n'est pas un point de contact
privilégié, c'est une graine d'énumération.

### `n` / `n_full` — résidus exposés dans la sphère
Comptage, sans unité. Médiane 9 (masqué).
*Ne capture pas* la surface réelle : deux patches à `n = 8` peuvent différer d'un facteur
deux en SASA.

### `n_total` / `n_total_full` — résidus **totaux** dans la sphère
Comptage. Dénominateur de `frac_exposed`. Renseigne la densité locale de la structure.
*Statut* : **écarté comme critère de sélection.** Voir les limites établies ci-dessous.

### `frac_exposed` / `_full` — `n / n_total`
Fraction, 0 à 1.
*Statut* : **écarté comme critère**, même raison.

### `n_truncated` — membres exclus par le masque de domaine
Comptage. 0 pour la grande majorité ; 15 patches sur 93 sont tronqués, médiane 2 résidus.
*À quoi ça sert* : repérer les patches de bord, qui sont aussi ceux à vérifier sur
`min_glyc` et sur l'appartenance au domaine.

### `sasa_apolar` / `_full` — **Å²**, surface accessible apolaire
Somme, sur les membres, de la SASA des atomes de carbone et de soufre. Shrake-Rupley,
sonde 1,40 Å, 100 points par atome (valeurs par défaut de Biopython).
**C'est la clé de tri principale.**
*Ne capture pas l'affinité.* Les patches recouvrant l'empreinte du cétuximab s'étalent du
rang 6 au rang 73 : la SASA apolaire borne la **designabilité**, pas la performance.
*Ne capture pas la qualité du carbone exposé* : un carbone de tige de lysine et un carbone
de leucine comptent pareil. C'est volontaire — un comptage de résidus hydrophobes rendrait
les tiges aliphatiques invisibles alors qu'elles contribuent réellement — mais ça reste une
équivalence posée.

### `sasa_polar` / `_full` — **Å²**
Même calcul sur l'azote et l'oxygène. L'hydrogène est absent de la structure (rayons X à
3,20 Å) donc hors du calcul dans tous les cas.

### `sasa_apolar_per_res` / `_full` — **Å² par membre**
Densité apolaire. Sert à séparer l'effet de taille du tri absolu.
*Ne capture pas* la contiguïté : la même densité peut venir d'une crête étroite ou d'une
plage large.

### `apolar_frac` / `_full` — apolaire / (apolaire + polaire)
Fraction, 0 à 1. Gamme observée au-dessus du plancher : 0,43 à 0,73.
*Ne capture pas* la taille. Un patch à `apolar_frac` 0,73 et 486 Å² et un patch à 0,43 et
609 Å² ne sont pas comparables sur cette seule colonne — d'où le plancher.

### `frac_ident` / `_full` — identité humain / souris sur les membres
Fraction, 0 à 1. **Objectif n°2 du challenge.** Alignement global BLOSUM62 des deux
précurseurs entiers P00533 / Q01279, gaps −11 / −1, gaps terminaux gratuits.
*Ne capture pas* la conséquence structurale d'une divergence : une substitution
conservative en bordure de patch pèse autant qu'une substitution de charge en son centre.
*Référence d'échelle* : 90,6 % sur la protéine entière, 87,4 % sur le domaine III, 70,8 %
sur l'empreinte du cétuximab — qui ne reconnaît pas l'EGFR murin.

### `n_diff` / `_full`
Comptage des membres `different` ou `gap`. Complément brut de `frac_ident`.

### `n_acidic` / `_full` — Asp + Glu parmi les membres
Comptage.

### `n_acidic_cons` / `_full` — Asp/Glu **conservés** (statut `identical`)
Comptage. **Objectif n°1 du challenge** : partenaire du pont salin His–acide qui n'existe
que sous forme protonée, donc à pH 6,5 et pas à 7,4.
*Ne capture pas* la faisabilité du pont : il faut encore que la géométrie permette à une
His du binder d'atteindre l'acide. Cette colonne dit qu'un partenaire existe, pas qu'il est
atteignable.
*Ne capture pas les autres mécanismes pH* : His–His, His–Tyr, carbonyle de squelette,
empilement His–aromatique, décalage de pKa par champ anionique local. Asp/Glu est le plus
*designable*, pas le seul.

### `n_hydro` / `_full` — comptage de `LIVFMWY` exposés
Comptage. Conservé **uniquement** pour comparer à l'ancienne métrique.
*Ne capture pas* les tiges aliphatiques. Remplacé par `sasa_apolar` comme clé de tri.

### `n_cys_ponte` / `_full`, `sasa_apolar_cys_ponte` / `_full`
Comptage, puis Å². Cystéines engagées dans un pont disulfure annoté dans UniProt (25 ponts
sur P00533, 6 touchant le domaine III).
*Portée quantitative : faible.* Sur le top-5 apolaire, la contribution plafonne à 15 Å²,
soit **2,3 à 2,6 %** ; défalquer ne produit qu'une permutation. Mais un centre **sur** une
cystéine pontée est un résidu structurellement contraint, pas un point d'accroche à
solliciter — un argument structural, pas de surface.

### `n_fab` / `_full` — membres dans l'empreinte du Fab cétuximab
Comptage, contacts lourds < 4,5 Å dans 6ARU. 38 patches sur 93 en contiennent au moins un.
**C'est le calibrateur du fichier** : le seul endroit où on sait qu'une protéine se lie.

### `min_glyc` / `_full` — **Å**, distance au séquon N-linked le plus proche
Minimum sur les membres de la distance de l'atome d'ancrage au résidu portant un séquon
annoté (13 sur P00533, dont 5 dans le domaine III : UniProt 352, 361, 413, 444, 528).
`0,0` signifie qu'un membre **est** un séquon.
*Mesure au CB du séquon, pas à l'arbre glycanique.* Un N-glycane réel s'étend bien au-delà
et reste flexible : cette colonne **sous-estime l'occlusion de façon systématique**. Aucune
valeur seuil n'est justifiable dessus, c'est pourquoi c'est une colonne brute et non un
filtre. Mesurer l'occlusion par la SASA serait mieux mais est inapplicable ici : **2 NAG
seulement sont modélisés sur la chaîne A** pour 13 séquons, l'absence de densité à 3,20 Å
n'étant pas l'absence de glycane.

### `min_dist_fab`, `mean_dist_fab` / `_full` — **Å**, distance à l'empreinte
`0,0` pour un patch qui contient un résidu d'empreinte. Médiane 11,4 Å sur le domaine III,
max 28,9.
**Remplace la colonne `face`, supprimée.** `face` déduisait la face de liaison du ligand
d'un axe centroïde domaine III → centroïde domaine I. 6ARU étant en conformation **repliée**,
les deux domaines sont écartés et le site de l'EGF est démonté : les 12 patches recouvrant
l'empreinte du cétuximab sortaient tous en « face externe ». La colonne était fausse.
`dist_fab` est une mesure : le cétuximab compétitionne l'EGF, son empreinte marque donc la
surface ligand-compétitive.
*Ne capture pas* l'orientation du résidu, seulement sa proximité. Et l'empreinte d'un Fab
n'est pas l'empreinte de l'EGF, c'est un proxy.

---

## Limites établies, à citer avant toute conclusion

1. **Corrélation +0,637 entre `n` et `sasa_apolar`.** Le tri absolu classe en partie la
   taille du patch. Taille médiane du top-10 absolu : 13,5 ; du top-10 par densité : 7,5 ;
   intersection 3/10. Les deux classements ne décrivent pas le même ensemble.
2. **Colonnes masquées contre `_full`.** Classer sur les colonnes masquées cache R310,
   K311, C502, N337. Mais les quatre révélés portent tous un séquon et sont à 12-18 Å de
   l'empreinte : ne jamais lire `_full` sans `min_glyc` ni `n_truncated`.
3. **`min_glyc` est mesuré au CB du séquon, pas à l'arbre glycanique.** Sous-estimation
   systématique de l'occlusion. Pas de seuil défendable sur cette colonne.
4. **Soufres pontés comptés comme apolaires.** Contribution ≤ 2,6 % sur le top-5 : sans
   portée quantitative. Mais un centre sur cystéine pontée est contraint structurellement.
5. **`face` supprimée**, cassée par la conformation repliée de 6ARU, remplacée par
   `dist_fab`. Ne pas réintroduire un critère d'orientation sans structure étendue.
6. **`n_total` et `frac_exposed` écartés comme critère.** Calibrés contre l'empreinte du
   cétuximab, les deux distributions sont confondues : min 11 vs 9, médiane 20 vs 19, max
   29 vs 30. Et S356 est à la fois un des voisinages les plus clairsemés du lot (11 résidus
   totaux) **et** une partie de la surface que lie le cétuximab. La population du voisinage
   ne sépare pas une surface liable d'une surface non liable.
7. **La SASA apolaire borne la designabilité, pas l'affinité.** Les patches d'empreinte
   s'étalent du rang 6 au rang 73.
8. **La convexité n'est pas mesurée.** Hypothèse ouverte : une protubérance convexe offre
   moins de surface à enfouir qu'une surface plate ou concave. `n_total` était un proxy, il
   s'est révélé non discriminant (point 6), et aucune mesure de courbure n'a été faite.
9. **Calibration cétuximab : ~400 Å² absolus, ~40 Å²/membre.** C'est une **référence
   d'échelle, pas un seuil de suffisance.** Un Fab enfouit bien plus de surface qu'un
   mini-binder de 50-130 résidus ; que le cétuximab fonctionne à 400 Å² ne garantit pas
   qu'un petit binder y suffise. Le plancher écarte ce qui est manifestement trop petit,
   il ne valide pas ce qui le franchit.

---

## Procédure de lecture en cinq étapes

### 1. Ne lire que les colonnes `_full`
Classer sur `sasa_apolar_full`. Garder `n_truncated` sous les yeux : une valeur non nulle
signale un patch de bord, à vérifier à l'étape 3.

### 2. Appliquer le plancher de 400 Å² sur `sasa_apolar_full`
40 patches sur 93 le franchissent. Ce qui tombe est trop petit pour l'échelle du seul
binder protéique dont on sait qu'il fonctionne sur cette cible. Les plus proches par en
dessous (K455 392, K454 379, P362 374, T406 374) ne sont pas disqualifiés pour autant — le
plancher n'est pas une frontière physique.

### 3. Éliminer sur `min_glyc` et sur l'appartenance au domaine
Écarter tout patch à `min_glyc = 0,0` : un de ses membres est un séquon, il sera dans
l'ombre d'un glycane que ni le modèle ni la structure ne portent. Vérifier ensuite que
l'union du groupe ne déborde pas sur un autre domaine — un site à cheval sur le domaine II
ou IV n'est pas un site du domaine III.

### 4. Ordonner sur les objectifs, dans l'ordre du règlement
`n_acidic_cons_full` décroissant (objectif 1, pH), puis `frac_ident_full` décroissant
(objectif 2, souris), puis `sasa_apolar_full` (objectif 3, affinité). **Pas l'inverse** :
ordonner par identité d'abord démote les patches qui portent les ancres acides.
Lire `dist_fab` en appui : proche de 0 = sur la surface ligand-compétitive.

### 5. Regrouper par recouvrement de membres, jamais par distance entre centres
Deux centres à 15 Å partagent encore la moitié de leurs membres à `PATCH_RADIUS = 11`.
Utiliser `member_set_full` : au-delà de 0,5 de membres partagés, deux patches décrivent le
même site. Puis sélectionner les représentants mutuellement disjoints.

**Résultat de cette procédure au commit courant** : un site de référence de 7 patches
(D323, E320, G317, H359, L325, T330, T358), puis trois sites disjoints — N449 (8 patches,
30 membres tous dans le domaine III, aucun séquon, `dFab` 0), K375 (identité 1,00 et 2
ancres acides, mais 2 séquons dans son groupe et un tiers de son union en domaine II), et
C502 (à écarter : 3 séquons, un quart de l'union en domaine IV, `apolar_frac` 0,43).

Le détail et l'arbitrage sont dans [`NOTES.md`](../NOTES.md).
