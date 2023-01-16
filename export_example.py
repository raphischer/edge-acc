import tensorflow as tf
import tensorflow_datasets as tfds
import os
from load_data import load_data
from load_models import load_preprocessing

images_dir = os.path.join('m', 'saved_models')
models_dir='models'
saved_models_dir = os.path.join(models_dir, 'saved_models')
tflite_models_dir = os.path.join(models_dir, 'tflite_models')




#builder = tfds.builder_from_directory('mnt_data/unpacked') 




model_name = 'MobileNetV3Small'

path_to_model = os.path.join(saved_models_dir,model_name)
loaded = tf.saved_model.load(path_to_model)
converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)

tflite_model = converter.convert()

# Save the model.
with open(os.path.join(tflite_models_dir,model_name+'.tflite'), 'wb') as f:
  f.write(tflite_model)




