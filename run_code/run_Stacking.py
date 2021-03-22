# Wafer map pattern classification using CNN

import pickle
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.vgg16 import VGG16

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

DIM = 64
BATCH_SIZE = 32
MAX_EPOCH = 1000
TRAIN_SIZE_LIST = [500, 5000, 50000, 162946]
LEARNING_RATE = 1e-4

early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

with open('../data/X_CNN_64.pickle', 'rb') as f:
    X_resize = pickle.load(f)
with open('../data/y.pickle', 'rb') as f:
    y = pickle.load(f)
    
# Stack wafer maps as 3 channels to correspond with RGB channels.
X_resize = (X_resize - 0.5) * 2
X_resize_stacked = np.repeat(X_resize, 3, -1)
y_onehot = tf.keras.utils.to_categorical(y)

REP_ID = 0
RAN_NUM = 27407 + REP_ID
print('Replication:', REP_ID)
for TRAIN_SIZE_ID in range(4):
    TRAIN_SIZE = TRAIN_SIZE_LIST[TRAIN_SIZE_ID]

    y_trnval, y_tst =  train_test_split(y_onehot, test_size=10000, random_state=RAN_NUM)
    if TRAIN_SIZE == 162946:
            pass
    else:    
        y_trnval, _ = train_test_split(y_trnval, train_size=TRAIN_SIZE, random_state=RAN_NUM)

    filename_MFE = '../data/WMPC_'+'MFE_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'
    filename_CNN = '../data/WMPC_'+'CNN_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'

    with open(filename_MFE + 'softmax.pickle', 'rb') as f:
        y_trnval_hat_mfe, y_tst_hat_mfe = pickle.load(f)
    with open(filename_CNN + 'softmax.pickle', 'rb') as f:
        y_trnval_hat_cnn, y_tst_hat_cnn = pickle.load(f)
    X_trnval_concat = np.concatenate([y_trnval_hat_mfe, y_trnval_hat_cnn], axis=1)
    X_tst_concat = np.concatenate([y_tst_hat_mfe, y_tst_hat_cnn], axis=1)

    labels = np.unique(np.argmax(y_trnval, 1))

    
    model = FNN()
    log = model.fit(X_trnval_concat, y_trnval, validation_split=0.2, 
                    epochs=MAX_EPOCH, batch_size=BATCH_SIZE,
                    callbacks=[early_stopping], verbose=0)

    y_trnval_hat = model.predict(X_trnval_concat)  
    y_tst_hat = model.predict(X_tst_concat)
    macro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='macro')
    micro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='micro')
    cm = confusion_matrix(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1))

    filename = '../result/WMPC_'+'Stacking_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'
    with open(filename+'f1_score.pickle', 'wb') as f:
        pickle.dump([macro, micro, cm], f)
    with open(filename+'softmax.pickle', 'wb') as f:
        pickle.dump([y_trnval_hat,y_trnval], f)
    with open(filename+'coef_.pickle', 'wb') as f:
        pickle.dump(model.coef_, f)