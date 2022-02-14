from rdkit import Chem
import pandas as pd
from sklearn.model_selection import train_test_split


def valid_smiles(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        m = None
    if m is not None:
        return True
    else:
        return False

def split_data():
    df = pd.read_csv('data/iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    print('Dataset shape: ', df.shape)
    df_meta = pd.read_csv('data/iterations_dataset_metacyc.csv', sep=',', header=0)
    print('Metacyc dataset shape: ', df_meta.shape)

    df_meta['valid'] = df_meta['smiles'].map(lambda smiles: valid_smiles(smiles))
    df_meta = df_meta[df_meta['valid'] == True].drop(columns=['valid'])
    print('Metacyc dataset shape (only valid smiles): ', df_meta.shape)
    print('Number of compounds on iteration 1 (metacyc): ', df_meta[df_meta['iteration'] == 1].shape[0])
    print('Number of compounds on iteration 2 (metacyc): ', df_meta[df_meta['iteration'] == 2].shape[0])
    print('Number of compounds on iteration 3 (metacyc): ', df_meta[df_meta['iteration'] == 3].shape[0])
    print('Number of compounds on iteration 4 (metacyc): ', df_meta[df_meta['iteration'] == 4].shape[0])
    print('Number of compounds on iteration 5 (metacyc): ', df_meta[df_meta['iteration'] == 5].shape[0])

    df_meta.to_csv('data/iterations_dataset_metacyc_valid.csv', index=False)

    df['valid'] = df['smiles'].map(lambda smiles: valid_smiles(smiles))
    df = df[df['valid'] == True].drop(columns=['valid'])

    print('Dataset shape (only valid smiles): ', df.shape)
    print('Number of compounds on iteration 1: ', df[df['iteration'] == 1].shape[0])
    print('Number of compounds on iteration 2: ', df[df['iteration'] == 2].shape[0])
    print('Number of compounds on iteration 3: ', df[df['iteration'] == 3].shape[0])
    print('Number of compounds on iteration 4: ', df[df['iteration'] == 4].shape[0])
    print('Number of compounds on iteration 5: ', df[df['iteration'] == 5].shape[0])

    X = df.smiles.values
    y = df.iteration.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25, random_state=52)

    train_dataset = pd.DataFrame()
    train_dataset['smiles'] = X_train
    train_dataset['iteration'] = Y_train
    valid_dataset = pd.DataFrame()
    valid_dataset['smiles'] = X_valid
    valid_dataset['iteration'] = Y_valid
    test_dataset = pd.DataFrame()
    test_dataset['smiles'] = X_test
    test_dataset['iteration'] = Y_test

    train_dataset.to_csv('data/train_iterations_dataset_metainter_retrorules_random.csv', index=False)
    valid_dataset.to_csv('data/valid_iterations_dataset_metainter_retrorules_random.csv', index=False)
    test_dataset.to_csv('data/test_iterations_dataset_metainter_retrorules_random.csv', index=False)


if __name__ == '__main__':
    split_data()