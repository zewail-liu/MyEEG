"""
A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import tensorflow as tf


class HopefullNet(tf.keras.Model):
    """
    Original HopeFullNet
    """

    def __init__(self, inp_shape=(4096, 2), class_num=5):
        super(HopefullNet, self).__init__()
        self.inp_shape = inp_shape

        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0.5

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="same",
                                            input_shape=self.inp_shape)
        self.batch_n_1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding="valid")
        self.batch_n_2 = tf.keras.layers.BatchNormalization()
        self.spatial_drop_1 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding="valid")
        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(296, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense2 = tf.keras.layers.Dense(148, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense3 = tf.keras.layers.Dense(74, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)
        self.out = tf.keras.layers.Dense(class_num, activation='softmax')

    def call(self, input_tensor):
        print("input_tensor shape:", input_tensor.shape)
        conv1 = self.conv1(input_tensor)
        print("conv1 shape:", conv1.shape)
        batch_n_1 = self.batch_n_1(conv1)
        conv2 = self.conv2(batch_n_1)
        batch_n_2 = self.batch_n_2(conv2)
        spatial_drop_1 = self.spatial_drop_1(batch_n_2)
        conv3 = self.conv3(spatial_drop_1)
        avg_pool1 = self.avg_pool1(conv3)
        conv4 = self.conv4(avg_pool1)
        spatial_drop_2 = self.spatial_drop_2(conv4)
        flat = self.flat(spatial_drop_2)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        return self.out(dropout2)


class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=25, kernel_size=(11, 1), activation='relu', padding="valid", )
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.conv2 = tf.keras.layers.Conv2D(filters=25, kernel_size=(1, 2), activation='relu', padding="valid", )
        self.bn = tf.keras.layers.BatchNormalization()
        self.mp = tf.keras.layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1))
        self.conv3 = tf.keras.layers.Conv2D(filters=50, kernel_size=(11, 1), activation='relu', padding="valid", )
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.mp2 = tf.keras.layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1))
        self.conv4 = tf.keras.layers.Conv2D(filters=100, kernel_size=(11, 1), activation='relu', padding="valid", )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.mp3 = tf.keras.layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1))
        self.conv5 = tf.keras.layers.Conv2D(filters=200, kernel_size=(11, 1), activation='relu', padding="valid", )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.mp4 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, input_tensor):
        input_tensor = tf.expand_dims(input_tensor, axis=3)
        # print("input_tensor shape:", input_tensor.shape)
        conv1 = self.conv1(input_tensor)
        # print("conv1 shape:", conv1.shape)
        dropout1 = self.dropout1(conv1)
        conv2 = self.conv2(dropout1)
        # print("conv2 shape:", conv2.shape)
        bn = self.bn(conv2)
        mp = self.mp(bn)
        conv3 = self.conv3(mp)
        # print("conv3 shape:", conv3.shape)
        dropout2 = self.dropout2(conv3)
        mp2 = self.mp2(dropout2)
        conv4 = self.conv4(mp2)
        # print("conv4 shape:", conv4.shape)
        bn2 = self.bn2(conv4)
        mp3 = self.mp3(bn2)
        conv5 = self.conv5(mp3)
        # print("conv5 shape:", conv5.shape)
        bn3 = self.bn3(conv5)
        mp4 = self.mp4(bn3)
        flat = self.flat(mp4)
        dense1 = self.dense1(flat)
        # print("dense1 shape:", dense1.shape)
        return dense1


if __name__ == '__main__':
    # path = "YOUR MODEL PATH"
    # model = tf.keras.models.load_model(path, custom_objects={"CustomModel": HopefullNet})
    # model1 = HopefullNet()
    model = SimpleCNN()
    input_shape = (None, 640, 2)
    # model1.build(input_shape)
    model.build(input_shape)
