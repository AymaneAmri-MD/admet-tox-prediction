"""Fonctions utilitaires : standardisation, scaffold split, features."""
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def standardize_smiles(smiles):
    """Nettoie une molécule : plus grand fragment, neutralisation, SMILES canonique."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol)


def scaffold_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1):
    """Split par scaffold : aucune structure de base partagée entre train/valid/test."""
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[scaffold_of(smi)].append(i)
    ordered = sorted(groups.values(), key=lambda s: (len(s), s[0]), reverse=True)
    n = len(smiles_list)
    train, valid, test = [], [], []
    for s in ordered:
        if len(train) + len(s) <= frac_train * n:
            train += s
        elif len(valid) + len(s) <= frac_valid * n:
            valid += s
        else:
            test += s
    return np.array(train), np.array(valid), np.array(test)


def morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)


def physchem(smiles):
    """Descripteurs physico-chimiques liés à l'ADME (Lipinski, TPSA, etc.)."""
    mol = Chem.MolFromSmiles(smiles)
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }
