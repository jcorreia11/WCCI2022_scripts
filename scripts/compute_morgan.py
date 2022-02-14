from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm


def encode_mol(mol):
    try:
        mol = Chem.MolFromSmiles(mol)
    except :
        pass
    return _featurize(mol)


def featurize(smiles, n_jobs):
    features = []
    pool = Pool(processes=n_jobs)
    for x in tqdm(pool.imap(encode_mol, smiles), total=len(smiles)):
        features.append(x)
    columns = ['mrg_'+ str(i+1) for i in range(1024)]
    return pd.DataFrame(features, columns=columns)

def _featurize(mol):
    try :
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                            2,
                                                            nBits=1024)
    except:
        fp = np.empty(1024, dtype=float)
        fp[:] = np.NaN

    fp = np.asarray(fp, dtype=np.float)
    return fp

def create_morgan_datasets():
    df_meta = pd.read_csv('data/iterations_dataset_metacyc_valid.csv', sep=',', header=0)
    df_train = pd.read_csv('data/train_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    df_valid = pd.read_csv('data/valid_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    df_test = pd.read_csv('data/test_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)

    x_meta = featurize(df_meta.smiles.values, 75)
    x_train = featurize(df_train.smiles.values, 75)
    x_valid = featurize(df_valid.smiles.values, 75)
    x_test = featurize(df_test.smiles.values, 75)

    df_meta_morgan = pd.concat([df_meta, x_meta], axis=1)
    df_train_morgan = pd.concat([df_train, x_train], axis=1)
    df_valid_morgan = pd.concat([df_valid, x_valid], axis=1)
    df_test_morgan = pd.concat([df_test, x_test], axis=1)

    df_meta_morgan.to_csv('data/iterations_dataset_morgan_metacyc.csv', index=False)
    df_train_morgan.to_csv('data/train_iterations_dataset_morgan_metainter_retrorules_random.csv', index=False)
    df_valid_morgan.to_csv('data/valid_iterations_dataset_morgan_metainter_retrorules_random.csv', index=False)
    df_test_morgan.to_csv('data/test_iterations_dataset_morgan_metainter_retrorules_random.csv', index=False)

def create_morgan_dataset_old_version():
    df = pd.read_csv('data/test_iterations_dataset_old.csv', sep=',', header=0, index_col=0)
    X = featurize(df.smiles.values, 4)
    df = pd.concat([df, X], axis=1)
    df.to_csv('data/test_iteration_dataset_morgan_old.csv', index=False)

if __name__ == '__main__':
    create_morgan_dataset_old_version()