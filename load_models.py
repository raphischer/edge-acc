import inspect
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


KERAS_BUILTINS = [e for e in tf.keras.applications.__dict__.values() if inspect.ismodule(e) and hasattr(e, 'preprocess_input')]
KERAS_MODELS = {n: e for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}
KERAS_PREPR = {n: mod.preprocess_input for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}
KERAS_MODELS['MobileNetV3Large'] = tf.keras.applications.MobileNetV3Large # manually adding MobileNetV3 since for some reason it is not callable from the respective submodule
KERAS_MODELS['MobileNetV3Small'] = tf.keras.applications.MobileNetV3Small
KERAS_PREPR['MobileNetV3Large'] = tf.keras.applications.mobilenet_v3.preprocess_input
KERAS_PREPR['MobileNetV3Small'] = tf.keras.applications.mobilenet_v3.preprocess_input


MODELS = {**KERAS_MODELS}
BUILTIN_PREPR = {**KERAS_PREPR}

# input sizes for specific models (default size would be 224)
INCEPTION_INPUT = {
    mname: (299, 299) for mname in MODELS if 'ception' in mname
}
EFFICIENT_INPUT = {
    'EfficientNetB1': (240, 240),
    'EfficientNetB2': (260, 260),
    'EfficientNetB3': (300, 300),
    'EfficientNetB4': (380, 380),
    'EfficientNetB5': (456, 456),
    'EfficientNetB6': (528, 528),
    'EfficientNetB7': (600, 600)
}
NASNET_INPUT = {
    'NASNetLarge': (331, 331)
}
MODEL_CUSTOM_INPUT = {**INCEPTION_INPUT, **EFFICIENT_INPUT, **NASNET_INPUT}


def prepare_model(model_name, metrics=None, weights='imagenet'):
    if metrics is None:
        metrics = ['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']

    # lookup model
    try:
        model = MODELS[model_name]
    except (TypeError, KeyError) as e:
        avail = ', '.join(n for n, _ in MODELS.items())
        raise RuntimeError(f'Error when loading {model_name}! \n{e}\nAvailable models:\n{avail}')

    model = model(weights=weights)
    model.compile(metrics=metrics)
    return model


def load_preprocessing(model_name):
    # prepares function to process one image or batch, based on input size
    if model_name == 'efficientnet-edgetpu-S_quant':
        model_name = 'EfficientNetB0'
    model_prepr = BUILTIN_PREPR[model_name]
    input_size = MODEL_CUSTOM_INPUT.get(model_name, (224, 224)) # default input size is 224, but some models have specific sizes
    proc_func = lambda img, label : simple_prepr(img, label, model_prepr, input_size)
    return proc_func


def simple_prepr(input, label, model_prepr, input_size):
    i = tf.cast(input, tf.float32) # cast to float
    i = tf.image.resize_with_crop_or_pad(i, input_size[0], input_size[1]) # resize for model input size
    i = model_prepr(i) # call model specific preprocessing
    return (i, label)
