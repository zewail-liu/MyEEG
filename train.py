import numpy as np
import tensorflow as tf

from c_module.models import EEGNet
from c_module.models import HopefullNet, SimpleCNN
from c_module.models import EEGTCNet

from b_data.bci_iv_2a import get_bci_iv_2a
from b_data.curry_data import get_curry0413
from b_data.physionet import get_physionet

from keras.losses import categorical_crossentropy

Adam = tf.optimizers.Adam


def train_with1DCNN():
    train_data, test_data, train_label, test_label = get_curry0413()
    train_data = train_data.squeeze().swapaxes(1, 2)
    test_data = test_data.squeeze().swapaxes(1, 2)
    x = np.empty((0, train_data.shape[1], 2))
    x_test = np.empty((0, test_data.shape[1], 2))
    for i in range(0, 8, 2):
        x = np.append(x, train_data[:, :, i:i + 2], axis=0)
        x_test = np.append(x_test, test_data[:, :, i:i + 2], axis=0)
    train_data, test_data = x, x_test
    train_label = np.concatenate((train_label, train_label, train_label, train_label), axis=0)
    test_label = np.concatenate((test_label, test_label, test_label, test_label), axis=0)

    model = HopefullNet(inp_shape=(4096, 2), class_num=4)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=10, epochs=500, verbose=1, validation_data=(test_data, test_label),
              )


def train_with_simpleCNN():
    train_data, test_data, train_label, test_label = get_physionet(1)

    model = SimpleCNN()
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=10, epochs=500, verbose=1, validation_data=(test_data, test_label),
              )


def train_withEEGNet():
    train_data, test_data, train_label, test_label = get_bci_iv_2a(1)
    train_data = train_data[:, :, :, np.newaxis]
    test_data = test_data[:, :, :, np.newaxis]

    model = EEGNet(nb_classes=4, Chans=12, Samples=1125)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=32, epochs=750, verbose=1,
              validation_data=(test_data, test_label))


def train_withEEGTCNet():
    F1 = 8
    KE = 32
    KT = 3
    L = 3
    FT = 15
    pe = 0.2
    pt = 0.3
    batch_size = 8
    train_data, test_data, train_label, test_label = get_bci_iv_2a(1)
    train_data = train_data[:, np.newaxis, :, :]
    test_data = test_data[:, np.newaxis, :, :]

    model = EEGTCNet(nb_classes=4, Chans=12, Samples=1125, layers=L, kernel_s=KT, filt=FT, dropout=pt, activation='elu',
                     F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=batch_size, epochs=750, verbose=1,
              validation_data=(test_data, test_label))


if __name__ == '__main__':
    # train_with1DCNN()
    # train_with_simpleCNN()
    # train_withEEGNet()
    train_withEEGTCNet()
    pass
