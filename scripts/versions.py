
def versions():
    import tensorflow as tf
    print('Tensorflow version: ', tf.__version__)
    import rdkit
    print('RDKit version: ', rdkit.__version__)
    import sklearn as sk
    print('scikit-learn version: ', sk.__version__)
    import keras
    print('Keras version: ', keras.__version__)



if __name__ == '__main__':
    versions()