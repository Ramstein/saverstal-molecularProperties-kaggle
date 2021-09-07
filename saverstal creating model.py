from __future__ import print_function

import numpy as np
import warnings

from keras.layers.convolutional import Conv2DTranspose, MaxPooling2D
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.layers import Dropout, Activation, Flatten, Conv2D

from keras import layers
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import ZeroPadding2D, AveragePooling2D, BatchNormalization

from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def Build_ResNet50(input_shape=None, classes=1000):
    img_input = Input(shape=input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')

    x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='f')

    x = conv_block(x, 3, [256, 256, 1024], stage=6, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='f')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='g')

    x = conv_block(x, 3, [256, 256, 1024], stage=7, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='f')
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='g')
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='h')

    x = conv_block(x, 3, [256, 256, 1024], stage=8, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=8, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=8, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=8, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=8, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=8, block='f')
    x = identity_block(x, 3, [256, 256, 1024], stage=8, block='g')

    x = conv_block(x, 3, [256, 256, 1024], stage=9, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=9, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=9, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=9, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=9, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=9, block='f')

    x = conv_block(x, 3, [256, 256, 1024], stage=10, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=10, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=10, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=10, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=10, block='e')

    x = conv_block(x, 3, [256, 256, 1024], stage=11, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=11, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=11, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=11, block='d')

    x = conv_block(x, 3, [512, 512, 2048], stage=12, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=12, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=12, block='c')

    #     x = AveragePooling2D((7, 7), padding='same', name='avg_pool')(x)

    #     x = GlobalMaxPooling2D()(x)
    outputs = Conv2D(classes, (2, 2), padding='same', activation='sigmoid')(x)

    model = Model(inputs=[img_input], outputs=[outputs], name='resnet50')

    return model