import datetime

import numpy as np
import tensorflow as tf

from b_data.curry_data import get_curry0413
from b_data.physionet import get_physionet
from c_module.models import HopefullNet, SimpleCNN
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

Adam = tf.optimizers.Adam

if __name__ == '__main__':
    train_data, test_data, train_label, test_label = get_physionet(1)

    model = SimpleCNN()
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=10, epochs=500, verbose=1, validation_data=(test_data, test_label),
              )
