import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise, Reshape, Conv1D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


import os
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())

gpu = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def baseline_cnn_model(stddev = 0.01,
                       filters = 8,
                       kernel_size = 32,
                       units = 512,
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


def train_mte_cnn_classification(output_path = 'outputs/multiclass_cnn_mte_metainter_retrorules/',
                                    reduce_lr_args=None,
                                    early_stopping_args=None,
                                    csv_logger = False,
                                    epochs = 250,
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

    df_train = pd.read_csv('data/train_emb_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    X_train = df_train.drop(columns=['smiles', 'iteration'])
    Y_train = df_train.iteration.values
    df_valid = pd.read_csv('data/valid_emb_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    X_valid = df_valid.drop(columns=['smiles', 'iteration'])
    Y_valid = df_valid.iteration.values
    df_test = pd.read_csv('data/test_emb_iterations_dataset_metainter_retrorules_random.csv', sep=',', header=0)
    X_test = df_test.drop(columns=['smiles', 'iteration'])
    Y_test = df_test.iteration.values
    df_meta = pd.read_csv('data/iterations_dataset_emb_metacyc_valid.csv', sep=',', header=0)
    for sm in df_meta.smiles.values:
        if sm in df_train.smiles.values:
            df_meta = df_meta[df_meta['smiles'] != sm]
    X_meta = df_meta.drop(columns=['smiles', 'iteration'])
    Y_meta = df_meta.iteration.values
    df_old = pd.read_csv('data/test_emb_iterations_dataset_old.csv', sep=',', header=0)
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

    estimator = KerasClassifier(build_fn=model,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=verbose,
                                callbacks=callbacks)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    np.save(folder + '/label_encoder_classes.npy', encoder.classes_)

    Y_train = encoder.transform(Y_train)
    Y_train = to_categorical(Y_train)
    Y_valid = encoder.transform(Y_valid)
    Y_valid = to_categorical(Y_valid)
    Y_test = encoder.transform(Y_test)
    Y_test = to_categorical(Y_test)
    Y_meta = encoder.transform(Y_meta)
    Y_meta = to_categorical(Y_meta)
    Y_old = encoder.transform(Y_old)
    Y_old = to_categorical(Y_old)

    hist = estimator.fit(X_train.values, Y_train, validation_data=(X_valid, Y_valid))

    print(estimator.model.summary())

    estimator.model.save(folder + '/trained_model_cnn_mte.h5')

    estimator.model.save_weights(folder + '/trained_model_cnn_mte_weights.h5')

    print('\nTraining accuracy: ', np.max(hist.history['accuracy']))
    print('Training loss: ', np.min(hist.history['loss']))
    print('\nValidation accuracy: ', np.max(hist.history['val_accuracy']))
    print('Validation loss: ', np.min(hist.history['val_loss']))

    print('\nTest accuracy: ', estimator.score(X_test, Y_test))

    print('\nTest accuracy (metacyc dataset): ', estimator.score(X_meta, Y_meta))

    print('\nTest accuracy (old test dataset): ', estimator.score(X_old, Y_old))

    y_pred = estimator.predict(X_test)
    y_pred_meta = estimator.predict(X_meta)
    y_pred_old = estimator.predict(X_old)

    Y_test = np.argmax(Y_test, axis=1)
    Y_test_meta = np.argmax(Y_meta, axis=1)
    Y_test_old = np.argmax(Y_old, axis=1)

    print('\nConfusion Matrix:')
    print(confusion_matrix(Y_test, y_pred))

    print('\nConfusion Matrix (metacyc dataset):')
    print(confusion_matrix(Y_test_meta, y_pred_meta))

    print('\nConfusion Matrix (old test dataset):')
    print(confusion_matrix(Y_test_old, y_pred_old))

    print('\nClassification Report:')
    print(classification_report(Y_test, y_pred, target_names=['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5']))

    print('\nClassification Report (metacyc dataset):')
    print(classification_report(Y_test_meta, y_pred_meta, target_names=['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5']))

    print('\nClassification Report (old test dataset):')
    print(classification_report(Y_test_old, y_pred_old, target_names=['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5']))

    # summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(folder + '/accuracy_history_cnn_mte.png')
    plt.close()

    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig(folder + '/loss_history_cnn_mte.png')
    plt.close()



if __name__ == '__main__':
    train_mte_cnn_classification()