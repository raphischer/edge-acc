import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)

    return x

def load_segmentation_model(model_name, image_shape):
    inputs = Input(shape = image_shape)
    if model_name == 'MobileNetV2':
        encoder = MobileNetV2(include_top = False, weights = "imagenet", input_tensor = inputs, alpha = 1.0)
        """ Encoder """
        s1 = encoder.get_layer("input_1").output                ## (256 x 256)
        s2 = encoder.get_layer("block_1_expand_relu").output    ## (128 x 128)
        s3 = encoder.get_layer("block_3_expand_relu").output    ## (64 x 64)
        s4 = encoder.get_layer("block_6_expand_relu").output    ## (32 x 32)

        """ Bridge """
        b1 = encoder.get_layer("block_13_expand_relu").output   ## (16 x 16)

        """ Decoder """
        d1 = decoder_block(b1, s4, 256)                         ## (32 x 32)
        d2 = decoder_block(d1, s3, 128)                         ## (64 x 64)
        d3 = decoder_block(d2, s2, 64)                         ## (128 x 128)
        d4 = decoder_block(d3, s1, 32)                          ## (256 x 256)
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        model = Model(inputs, outputs, name="MobileNetV2_U-Net")
        return model

if __name__ == '__main__':
    model = load_segmentation_model('MobileNetV2',(256,256,3))
    model.summary()