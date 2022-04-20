import numpy as np
import tensorflow as tf

from b_data.bci_iv_2a import get_bci_iv_2a
from b_data.curry_data import get_curry0413
from c_module.EEGTCNet import EEGTCNet
from keras.losses import categorical_crossentropy

Adam = tf.optimizers.Adam
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
    train_data, test_data, train_label, test_label = get_curry0413()
    # x = np.empty((0, 1, 2, train_data.shape[3]))
    # x_test = np.empty((0, 1, 2, test_data.shape[3]))
    # for i in range(0, 8, 2):
    #     x = np.append(x, train_data[:, :, i:i + 2, :], axis=0)
    #     x_test = np.append(x_test, test_data[:, :, i:i + 2, :], axis=0)
    # train_label = np.concatenate((train_label, train_label, train_label, train_label), axis=0)
    # test_label = np.concatenate((test_label, test_label, test_label, test_label), axis=0)

    model = EEGTCNet(nb_classes=4, Chans=6, Samples=4096, layers=L, kernel_s=KT, filt=FT, dropout=pt, activation='elu',
                     F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=batch_size, epochs=750, verbose=1,
              validation_data=(test_data, test_label))
