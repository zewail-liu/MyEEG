import datetime

import numpy as np
import tensorflow as tf

from b_data.curry_data import get_curry0413
from b_data.physionet import get_physionet
from c_module.models import HopefullNet
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

Adam = tf.optimizers.Adam

if __name__ == '__main__':
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
