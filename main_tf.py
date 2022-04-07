import datetime

import numpy as np
import tensorflow as tf
from keras import callbacks

from c_module.EEGTCNet import EEGTCNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.losses import categorical_crossentropy
Adam = tf.optimizers.Adam
to_categorical = tf.keras.utils.to_categorical
data_path = r'D:\00-data\BCI_IV_2a/'
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
    data = np.load('data_Ezone.npy')
    data = data[:, np.newaxis, :, :]
    label = np.load('label.npy')
    X_train, X_test = data[:260], data[260:]
    y_train, y_test = label[:260], label[260:]
    y_train_onehot, y_test_onehot = to_categorical(y_train), to_categorical(y_test)
    for j in range(12):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    model = EEGTCNet(nb_classes=4, Chans=12, Samples=1125, layers=L, kernel_s=KT, filt=FT, dropout=pt, activation='elu',
                     F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    model.fit(X_train, y_train_onehot, batch_size=batch_size, epochs=1200, verbose=1)

    y_pred = model.predict(X_test).argmax(axis=-1)
    labels = y_test_onehot.argmax(axis=-1)
    accuracy_of_test = accuracy_score(labels, y_pred)
    print(accuracy_of_test)
