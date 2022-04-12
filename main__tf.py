import datetime

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

from b_data.curry_data_1DCNN import get_curry0408_withbaseSMOTE, get_curry0408_41DCNN
from c_module.models import HopefullNet
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

Adam = tf.optimizers.Adam

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = get_curry0408_41DCNN()

    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    # del data, label
    model = HopefullNet(inp_shape=(4096, 2), class_num=2)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=64, epochs=500, verbose=1, validation_data=(test_data, test_label),
              )
