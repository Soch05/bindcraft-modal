# Challenge 1 — EGFR : brief du règlement

Condensé de la page officielle du challenge et de sa FAQ.

- Source : https://proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/egfr
- Page générale : https://proteinbase.com/competitions/anthropic-adaptyv-2026
- Consulté le : 30 septembre 2026
- Statut : le règlement peut être mis à jour en cours de compétition. Revérifier
  la page avant soumission. La page propose un bouton « Copy Markdown » pour
  récupérer la FAQ intégrale ; ce fichier en est une synthèse, pas un substitut.

---

## 1. Ce qui est demandé

Concevoir un binder conditionnel contre l'EGFR humain. Trois objectifs, évalués
ensemble, dans cet ordre de classement :

1. **Sélectivité pH** — liaison à pH 6,5, pas de liaison détectable à pH 7,4. Le
   pH 6,5 reproduit le microenvironnement tumoral ; c'est ce qui donne la
   sélectivité thérapeutique.
2. **Cross-réactivité souris** — la même séquence doit reconnaître l'EGFR humain
   et l'EGFR murin, pour permettre le passage en modèle préclinique sans
   redévelopper un binder dédié.
3. **Affinité** — contre la région extracellulaire humaine.

Les organisateurs précisent que la difficulté relative est prise en compte : un
binder faible mais clairement pH-dépendant peut être jugé plus marquant qu'un
binder de forte affinité non conditionnel. Ils recommandent aussi de viser un
**épitope fonctionnel**, la pertinence thérapeutique étant un objectif affiché de
la compétition.

## 2. Cible

| | |
|---|---|
| Humain | UniProt P00533, région extracellulaire, résidus 25–645 (621 aa) |
| Souris | UniProt Q01279 |
| Structure de référence | PDB 6ARU, chaîne A (ectodomaine + Fab cétuximab mutant) |
| Épitope recommandé | domaine III, également ciblé par le cétuximab et le panitumumab |

Les mesures sont faites sur la région extracellulaire complète, humaine et
murine, et aux deux pH pour l'humain.

## 3. Contraintes de soumission

| | |
|---|---|
| Longueur | 10–250 aa pour une protéine à chaîne unique |
| Formats acceptés | protéine de novo, nanobody, scFv, Fab |
| Nombre de designs | 40 max en Track 1 ; **20 max en Tracks 2 et 3** |
| Format | CSV **ordonné par classement**, meilleur design en première ligne |
| Colonnes minimales | `name`, `sequence`, `molecule_class` |
| `molecule_class` | **`single_chain`**, `nanobody`, `scfv`, `fab_kappa`, `fab_lambda` — ⚠️ corrige le 6 octobre : ce tableau disait `protein`, valeur que le widget d'upload n'accepte pas. Champ par ailleurs **optionnel**, reglable par design apres l'upload. Le formulaire accepte CSV **ou FASTA**, et fournit un modele a telecharger. |
| Fab | une seule séquence `VH:VL` ; le type de chaîne légère passe par `molecule_class` |

Un nanobody ou un anticorps est reconnu comme tel s'il est annoté par un outil de
type ANARCI. Les scFv peuvent être reformatés avec le linker préféré des
organisateurs.

**Stratification des résultats** en cinq catégories, qui servent aussi à définir
des vainqueurs par catégorie :

- microbinders : < 40 aa
- minibinders : 40–100 aa
- grands binders : > 100 aa
- nanobodies
- anticorps (scFv ou Fab)

## 4. Filtres durs

Appliqués à toutes les soumissions avant toute sélection :

- Respect des contraintes de longueur et de format ci-dessus.
- Unicité : ne pas soumettre deux fois le même design.
- **De novo et zero-shot**, défini comme : n'avoir utilisé aucun binder existant
  comme point de départ, et présenter une diversité de séquence **et** de
  structure suffisante par rapport aux protéines connues (diversité des CDR dans
  le cas des nanobodies et anticorps). Reprendre un binder existant et le
  modifier est explicitement interdit ; les designs doivent être produits à
  partir de zéro.

Ce que le règlement autorise en revanche : utiliser des binders connus pour
**calibrer ou évaluer** des métriques de filtrage, ou pour **entraîner ou
fine-tuner** un modèle. La frontière est l'usage comme graine.

Lecture recommandée par les organisateurs sur ce point : le billet d'Adaptyv sur
la nouveauté, https://www.adaptyvbio.com/blog/novelty

## 5. Mécanique de sélection — Track 3

Environ **1500 designs criblés par challenge**, répartis à 50 % pour le Track 1,
25 % pour le Track 2 et **25 % pour le Track 3**, soit environ 375 places.

Pour les Tracks 2 et 3, toutes les soumissions sont mises en commun avec les
informations fournies, puis passées à Claude avec un prompt de sélection rédigé à
l'avance par Anthropic et Adaptyv. Les critères annoncés sont la **qualité
prédite du design**, sa **nouveauté**, et la **nouveauté de la méthode**. Le
prompt n'est pas publié avant la fin de la compétition, délibérément, pour éviter
la sur-optimisation d'une métrique unique.

**Conséquence pratique : le dossier de méthodes compte autant que les
séquences.** Les organisateurs encouragent à joindre métriques (ipTM, ipSAE,
auto-cohérence, métriques physiques, scores de liabilité de séquence), structures
prédites, et description détaillée des modèles et stratégies employés. Un dépôt
public lié à la soumission est explicitement prévu.

Ils précisent que les méthodes nouvelles et non encore validées
expérimentalement seront représentées équitablement dans la sélection — décrire
une approche inhabituelle en détail est donc un avantage, pas un risque.

**Toute instruction embarquée ou tentative d'injection de prompt peut valoir
disqualification.**

## 6. Calendrier

| | |
|---|---|
| Fenêtre Challenge 1 | 28 septembre – 4 octobre 2026 |
| Clôture | 4 octobre, 23:59 AoE (UTC−12) = **dimanche 5 octobre, 13h59 à Paris** |
| Challenges suivants | 5–11, 12–18, 19–25 octobre, 26 octobre – 1er novembre |
| Collection des designs sélectionnés | environ une semaine après le dernier challenge |
| Validation expérimentale | environ un mois après la clôture de chaque fenêtre ; premières données début novembre |
| Publication des résultats | 15 décembre 2026 |

## 7. Publication et propriété

Les participants conservent la propriété de leurs designs. Les séquences
validées, les résultats expérimentaux, les structures prédites et les méthodes
sont publiés ouvertement sur Proteinbase sous licence ODC-BY, **résultats
négatifs inclus**. Les designs non retenus mais présents dans une Collection
publique relèvent de la même licence.

Soumission anonyme possible via un compte Proteinbase anonyme, avec
dé-anonymisation ultérieure possible.

Conséquence : écrire chaque fichier du dépôt en supposant qu'un tiers le lira.

## 8. Outils

Aucune restriction sur les outils, à condition de respecter les licences — les
outils sous licence commerciale, Rosetta par exemple, supposent de détenir la
licence au préalable. L'usage de Claude est facultatif en Track 3.

Ressources pointées par les organisateurs :

- Blog Adaptyv, dont le billet sur la nouveauté — https://www.adaptyvbio.com/blog
- Analyse post-compétition EGFR — https://www.biorxiv.org/content/10.1101/2025.04.17.648362v2
- Proteinbase, corpus de données de design, compétitions passées comprises
- Billet Anthropic sur le design de binders de novo, avec rapport technique et
  jeu de données expérimentales publié sur Hugging Face
- Modèles open-source optimisés pour l'inférence :
  https://github.com/anthropics/uplifting-biomolecular-modeling — recommandés
  explicitement pour maximiser le compute GPU disponible
- Canal Slack Proteinbase pour les questions

---

## 9. Notes de lecture

Interprétation personnelle, pas du règlement.

- Le classement place la sélectivité pH en premier. Une campagne qui optimise
  l'affinité produit donc un résultat correct sur le critère le moins bien
  pondéré. L'ordre des objectifs doit structurer le pipeline, pas seulement la
  sélection finale.
- La cross-réactivité souris se joue au choix de l'épitope, en amont, pas au
  filtrage. Le cétuximab ne reconnaît pas l'EGFR murin : l'épitope le plus
  documenté du domaine III est précisément celui à traiter avec méfiance.
- Le règlement ne dit rien des glycanes. La cible expérimentale sera glycosylée,
  un modèle AF2 ne l'est pas. Les sites de N-glycosylation, **des deux espèces**,
  sont à exclure du patch.
- « 20 designs maximum » n'est pas un objectif à remplir. Vingt designs médiocres
  dans un pool commun valent moins que huit designs défendables accompagnés d'une
  méthode claire, puisque la sélection porte aussi sur la qualité prédite.
