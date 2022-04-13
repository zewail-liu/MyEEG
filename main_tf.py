import datetime

import numpy as np
import tensorflow as tf
from keras import callbacks

from b_data.curry_data_1DCNN import get_curry0408, get_curry0408_41DCNN
from c_module.EEGTCNet import EEGTCNet
from sklearn.metrics import accuracy_score
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

Adam = tf.optimizers.Adam
to_categorical = tf.keras.utils.to_categorical
data_path = r'D:\00-data\220408\SMOTE\\'
F1 = 8
KE = 32
KT = 3
L = 3
FT = 15
pe = 0.2
pt = 0.3
classes = 4
channels = 22
crossValidation = False
batch_size = 8
epochs = 750
lr = 0.001

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = get_curry0408()

    model = EEGTCNet(nb_classes=2, Chans=8, Samples=4096, layers=L, kernel_s=KT, filt=FT, dropout=pt, activation='elu',
                     F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=batch_size, epochs=500, verbose=1,
              validation_data=(test_data, test_label))

    print('---')
    testLoss, testAcc = model.evaluate(test_data, test_label)
    print('\nAccuracy:', testAcc)
    print('\nLoss: ', testLoss)
    print('---')
