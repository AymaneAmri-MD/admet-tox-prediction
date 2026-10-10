# Prédiction ADMET & toxicité par Machine Learning

Pipeline de prédiction de la **toxicité** de molécules à partir de leur structure, sur le jeu **Tox21** (MoleculeNet : 12 tests biologiques, récepteurs nucléaires et réponses au stress cellulaire).

Ce projet prolonge [`drug-design-morphine-atropine`](https://github.com/AymaneAmri-MD/drug-design-morphine-atropine) : après avoir généré et docké des analogues, il faut savoir s'ils sont plausibles du point de vue de la toxicité.

## Objectifs
- Standardiser les molécules (RDKit) et construire un **scaffold split** rigoureux
- Comparer des modèles de base (Random Forest, XGBoost) puis un GNN
- Expliquer les prédictions (SHAP) : quelles sous-structures sont associées à la toxicité ?
- Appliquer le modèle aux analogues issus du projet de drug design

## Structure
```
admet-tox-prediction/
├── notebooks/
│   ├── 01_data_tox21.ipynb        # chargement, nettoyage, scaffold split, descripteurs
│   └── 02_baseline_models.ipynb   # RF / XGBoost, ROC-AUC par tâche
├── src/chem_utils.py              # standardisation, split, features
├── environment.yml
└── results/                       # figures et tableaux de métriques
```

## Utilisation
**Colab** : ouvrir les notebooks et décommenter la cellule d'installation.
**En local** :
```bash
conda env create -f environment.yml
conda activate admet-tox
jupyter lab
```

## Résultats
*À compléter après l'exécution du notebook 02 (tableau ROC-AUC + figure).*

## Limites
Tox21 couvre un espace chimique restreint et les labels sont issus de tests in vitro : ce modèle est un **outil de criblage exploratoire**, pas un substitut à une évaluation toxicologique.

## Références
- Wu et al., *MoleculeNet: a benchmark for molecular machine learning*, Chemical Science, 2018
- Données Tox21 : https://tripod.nih.gov/tox21/

## Auteur
Aymane Amri Amine — [@AymaneAmri-MD](https://github.com/AymaneAmri-MD)
