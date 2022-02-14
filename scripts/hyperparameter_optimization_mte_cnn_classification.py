import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential, Input, regularizers
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise, Reshape, Conv1D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

import os
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())

gpu = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def baseline_cnn_model(stddev = 0.05,
                       filters = 8,
                       kernel_size = 32,
                       units = 256,
                       dropout = 0.3):
    # create model
    model = Sequential()
    model.add(GaussianNoise(stddev=stddev, input_shape=(512,)))
    model.add(Reshape((512, 1)))
    model.add(Conv1D(filters, (kernel_size), activation='relu', padding='same'))
    model.add(Conv1D(filters, (kernel_size), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model



def run_hyperparameter_optimization():
    #LOAD DATA
    df_train = pd.read_csv('data/train_emb_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    X_train = df_train.drop(columns=['smiles', 'iteration'])
    Y_train = df_train.iteration.values
    df_valid = pd.read_csv('data/valid_emb_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    X_valid = df_valid.drop(columns=['smiles', 'iteration'])
    Y_valid = df_valid.iteration.values

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y_train)

    Y_train = encoder.transform(Y_train)
    Y_train = to_categorical(Y_train)
    Y_valid = encoder.transform(Y_valid)
    Y_valid = to_categorical(Y_valid)

    X_train = np.concatenate([X_train, X_valid])
    Y_train = np.concatenate([Y_train, Y_valid])


    model = KerasClassifier(build_fn=baseline_cnn_model, epochs=150, batch_size=512, verbose=1)

    param_grid = {'stddev' : [0.01, 0.05],
                  'filters' : [4, 8, 16],
                  'kernel_size' : [32, 64, 128],
                  'units' : [512, 256, 128],
                  'dropout' : [0, 0.3, 0.5]}

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, n_jobs=1, cv=3)
    grid_result = grid.fit(X_train, Y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__=='__main__':
    run_hyperparameter_optimization()
