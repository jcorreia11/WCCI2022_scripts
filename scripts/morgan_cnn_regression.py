import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise, Reshape, Conv1D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import os
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())

gpu = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def baseline_cnn_model(stddev = 0.05,
                       filters = 16,
                       kernel_size = 64,
                       units = 256,
                       dropout = 0.5):
    # create model
    model = Sequential()
    model.add(GaussianNoise(stddev=stddev, input_shape=(1024,)))
    model.add(Reshape((1024, 1)))
    model.add(Conv1D(filters, (kernel_size), activation='relu', padding='same'))
    model.add(Conv1D(filters, (kernel_size), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['mean_absolute_error'])
    return model


def train_morgan_cnn_regression(output_path = 'outputs/regression_cnn_morgan_metainter_retrorules/',
                                    reduce_lr_args=None,
                                    early_stopping_args=None,
                                    csv_logger = False,
                                    epochs = 150,
                                    batch_size = 512,
                                    verbose = 1):

    if early_stopping_args is None:
        early_stopping_args = {'monitor': 'val_loss',
                               'patience': 15,
                               'restore_best_weights': True,
                               'verbose': 1}
    if reduce_lr_args is None:
        reduce_lr_args = {'monitor': 'val_loss',
                          'factor': 0.25,
                          'patience': 10,
                          'min_lr': 0.00001,
                          'verbose': 1}

    # time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    time = 'final_model/'
    if not os.path.exists(output_path+time):
        os.makedirs(output_path+time)

    folder = output_path+time

    df_train = pd.read_csv('data/train_iterations_dataset_morgan_metainter_retrorules_random.csv', sep=',', header=0)
    X_train = df_train.drop(columns=['smiles', 'iteration'])
    Y_train = df_train.iteration.values
    df_valid = pd.read_csv('data/valid_iterations_dataset_morgan_metainter_retrorules_random.csv', sep=',', header=0)
    X_valid = df_valid.drop(columns=['smiles', 'iteration'])
    Y_valid = df_valid.iteration.values
    df_test = pd.read_csv('data/test_iterations_dataset_morgan_metainter_retrorules_random.csv', sep=',', header=0)
    X_test = df_test.drop(columns=['smiles', 'iteration'])
    Y_test = df_test.iteration.values
    df_meta = pd.read_csv('data/iterations_dataset_morgan_metacyc.csv', sep=',', header=0)
    for sm in df_meta.smiles.values:
        if sm in df_train.smiles.values:
            df_meta = df_meta[df_meta['smiles']!=sm]
    X_meta = df_meta.drop(columns=['smiles', 'iteration'])
    Y_meta = df_meta.iteration.values
    df_old = pd.read_csv('data/test_iteration_dataset_morgan_old.csv', sep=',', header=0)
    X_old = df_old.drop(columns=['smiles', 'iteration'])
    Y_old = df_old.iteration.values

    model = baseline_cnn_model


    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=reduce_lr_args['monitor'],
                                                     factor=reduce_lr_args['factor'],
                                                     patience=reduce_lr_args['patience'],
                                                     min_lr=reduce_lr_args['min_lr'],
                                                     verbose=reduce_lr_args['verbose'])
    ea = tf.keras.callbacks.EarlyStopping(monitor=early_stopping_args['monitor'],
                                          patience=early_stopping_args['patience'],
                                          restore_best_weights=early_stopping_args['restore_best_weights'],
                                          verbose=early_stopping_args['verbose'])

    callbacks = [reduce_lr, ea]
    if csv_logger:
        csv_log = tf.keras.callbacks.CSVLogger(folder + '/training_cnn.log')
        callbacks.append(csv_log)

    estimator = KerasRegressor(build_fn=model,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=verbose,
                                callbacks=callbacks)

    hist = estimator.fit(X_train.values, Y_train, validation_data=(X_valid, Y_valid))

    print(estimator.model.summary())

    estimator.model.save(folder + '/trained_model_cnn_morgan.h5')

    estimator.model.save_weights(folder + '/trained_model_cnn_morgan_weights.h5')

    print('\nTraining mean_absolute_error: ', np.max(hist.history['mean_absolute_error']))
    print('Training loss: ', np.min(hist.history['loss']))
    print('\nValidation mean_absolute_error: ', np.max(hist.history['val_mean_absolute_error']))
    print('Validation loss: ', np.min(hist.history['val_loss']))

    y_pred = estimator.predict(X_test)
    test_mae = mean_absolute_error(Y_test, y_pred)
    print('\nTest mean_absolute_error: ', test_mae)
    test_mse = mean_squared_error(Y_test, y_pred)
    print('\nTest mean_squared_error: ', test_mse)
    test_r2 = r2_score(Y_test, y_pred)
    print('\nTest r2_score: ', test_r2)

    # METACYC
    y_pred_meta = estimator.predict(X_meta)
    test_mae_meta = mean_absolute_error(Y_meta, y_pred_meta)
    print('\nTest mean_absolute_error (Metacyc Dataset): ', test_mae_meta)
    test_mse_meta = mean_squared_error(Y_meta, y_pred_meta)
    print('\nTest mean_squared_error (Metacyc Dataset): ', test_mse_meta)
    test_r2_meta = r2_score(Y_meta, y_pred_meta)
    print('\nTest r2_score (Metacyc Dataset): ', test_r2_meta)

    # OLD TEST SET
    y_pred_old = estimator.predict(X_old)
    test_mae_old = mean_absolute_error(Y_old, y_pred_old)
    print('\nTest mean_absolute_error (Old Dataset): ', test_mae_old)
    test_mse_old = mean_squared_error(Y_old, y_pred_old)
    print('\nTest mean_squared_error (Old Dataset): ', test_mse_old)
    test_r2_old = r2_score(Y_old, y_pred_old)
    print('\nTest r2_score (Old Dataset): ', test_r2_old)

    # summarize history for accuracy
    plt.plot(hist.history['mean_absolute_error'])
    plt.plot(hist.history['val_mean_absolute_error'])
    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(folder + '/mean_absolute_error_history_cnn_regression.png')
    plt.close()

    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig(folder + '/loss_history_cnn_regression.png')
    plt.close()



if __name__ == '__main__':
    train_morgan_cnn_regression()