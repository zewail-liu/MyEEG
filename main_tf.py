import datetime

import numpy as np
import tensorflow as tf
from keras import callbacks

from c_module.EEGTCNet import EEGTCNet
from sklearn.metrics import accuracy_score
from keras.losses import categorical_crossentropy

Adam = tf.optimizers.Adam
to_categorical = tf.keras.utils.to_categorical
data_path = r'D:\00-data\220408\\'
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
batch_size = 64
epochs = 750
lr = 0.001

if __name__ == '__main__':
    data = np.load(data_path + 'data.npy')
    label = np.load(data_path + 'labels.npy')
    X_train, X_test = data[:160], data[160:]
    y_train, y_test = label[:160], label[160:]

    model = EEGTCNet(nb_classes=2, Chans=8, Samples=4096, layers=L, kernel_s=KT, filt=FT, dropout=pt, activation='elu',
                     F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=700, verbose=1)

    y_pred = model.predict(X_test).argmax(axis=-1)
    labels = y_test.argmax(axis=-1)
    accuracy_of_test = accuracy_score(labels, y_pred)
    print(accuracy_of_test)
