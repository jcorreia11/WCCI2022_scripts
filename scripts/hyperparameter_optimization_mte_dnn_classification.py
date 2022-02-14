import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential, Input, regularizers
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
import os
import tensorflow as tf

print(tf.__version__)
print(tf.test.is_built_with_cuda())

gpu = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def baseline_dnn_model(n_hidden_layers = 4,
                       units_hidden_layers = 512,
                       dropout_1 = 0.2,
                       dropout_hidden_layers = 0.5,
                       l1 = 0.01,
                       l2 = 0.01):
    # create model
    model = Sequential()
    model.add(Input(shape=(512,)))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_1))

    for i in range(n_hidden_layers):
        model.add(Dense(units_hidden_layers, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_hidden_layers))
        units_hidden_layers = units_hidden_layers/2

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


    model = KerasClassifier(build_fn=baseline_dnn_model, epochs=250, batch_size=512, verbose=1)

    param_grid = {'n_hidden_layers' : [2, 4, 6],
                  'units_hidden_layers' : [1024, 512, 256],
                  'dropout_1' : [0, 0.2, 0.5],
                  'dropout_hidden_layers' : [0, 0.3, 0.4],
                  'l1' : [0, 0.001, 0.01],
                  'l2' : [0, 0.001, 0.01]}

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=15, n_jobs=1, cv=5)
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
