import tensorflow as tf
import tensorflow_hub as hub
import os
from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils
import tensorflow.keras 

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048

# Download the model from Tensorflow Hub.
keras_layer = hub.KerasLayer(
  'https://tfhub.dev/google/mosaic/mobilenetmultiavgseg/2',
  signature='serving_default',
  output_key='logits')
model = tf.keras.Sequential([keras_layer])
model.build([None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

