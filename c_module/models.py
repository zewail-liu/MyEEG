import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Add, Lambda, DepthwiseConv2D, Input, Permute
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K


class HopefullNet(tf.keras.Model):
    """
    Original HopeFullNet
    10.1088/1741-2552/ac4430
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
    """
    no code from 10.3389/fnhum.2020.00338
    try to use the same architecture as in the paper
    """

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


def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25,
           dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.

    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def EEGTCNet(nb_classes, Chans=64, Samples=128, layers=3, kernel_s=10, filt=10, dropout=0, activation='relu', F1=4, D=2,
             kernLength=64, dropout_eeg=0.1):
    def EEGNet(input_layer, F1=4, kernLength=64, D=2, Chans=22, dropout=0.1):
        F2 = F1 * D
        block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
        block1 = BatchNormalization(axis=-1)(block1)
        block2 = DepthwiseConv2D((1, Chans), use_bias=False,
                                 depth_multiplier=D,
                                 data_format='channels_last',
                                 depthwise_constraint=max_norm(1.))(block1)
        block2 = BatchNormalization(axis=-1)(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
        block2 = Dropout(dropout)(block2)
        block3 = SeparableConv2D(F2, (16, 1),
                                 data_format='channels_last',
                                 use_bias=False, padding='same')(block2)
        block3 = BatchNormalization(axis=-1)(block3)
        block3 = Activation('elu')(block3)
        block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)
        block3 = Dropout(dropout)(block3)
        return block3

    def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, activation='relu'):

        # Residual Block
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(input_layer)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)

        # conv1d added or ori data from last layer
        if input_dimension != filters:
            conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
            added = Add()([block, conv])
        else:
            added = Add()([block, input_layer])
        out = Activation(activation)(added)

        # nums of Residual
        for i in range(depth - 1):
            block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                           padding='causal', kernel_initializer='he_uniform')(out)
            block = BatchNormalization()(block)
            block = Activation(activation)(block)
            block = Dropout(dropout)(block)
            block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                           padding='causal', kernel_initializer='he_uniform')(block)
            block = BatchNormalization()(block)
            block = Activation(activation)(block)
            block = Dropout(dropout)(block)
            added = Add()([block, out])
            out = Activation(activation)(added)

        return out

    # start the model
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = .25
    numFilters = F1
    F2 = numFilters * D

    EEGNet_sep = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                     dropout=dropout, activation=activation)
    out = Lambda(lambda x: x[:, -1, :])(outs)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(regRate))(out)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


if __name__ == '__main__':
    model = SimpleCNN()
    input_shape = (None, 640, 2)
    # model1.build(input_shape)
    model.build(input_shape)
