
import argparse
import time
import numpy as np
import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds
import os
from PIL import Image
import json
from pathlib import Path
import traceback
from helper_scripts.util import PatchedJSONEncoder
#import tflite_runtime.interpreter as tflite #Needed on EdgeTPU for delegations

from helper_scripts.load_models import prepare_model, load_preprocessing
from helper_scripts.load_data import load_data
from helper_scripts.monitoring import Monitoring

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

# Dict that translates codes to label names of imagenet
def create_labeldict(path_to_file):
  labelnames = load_labels('/home/staay/Git/imagenet-on-the-edge/imagenet_labels_map.txt')
  labelnamedict = {}
  for labelname in labelnames:
    number = labelname.split()[0]
    name = labelname.split()[2]
    labelnamedict[number] = name
  return labelnamedict

#returns input shape that the model wants
def get_input_shape(model_name,nbatches,batchsize):
  preprocess = load_preprocessing(model_name)
  ds_prep, _prep = load_data(preprocess=preprocess, n_batches=nbatches, batch_size=batchsize) #tf.data.Dataset (tensorflow.python.data.ops.dataset_ops.TakeDataset) , tfds.core.DatasetInfo
  ds_prep_numpy = tfds.as_numpy(ds_prep)
  for image, labelno in ds_prep_numpy:
    return image.shape
  
result_text = ""  
for model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet152', 'ResNet50', 'ResNet101V2', 'ResNet152V2', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'MobileNetV3Large', 'MobileNetV3Small']:


  labelnamedict = create_labeldict('/home/staay/Git/imagenet-on-the-edge/imagenet_labels_map.txt')
  image_directory = '/home/staay/Git/imagenet-on-the-edge/image_data/50'
  labelsfile = os.path.join(image_directory,model_name,model_name+'labels.txt')
  labels = load_labels('/home/staay/Git/imagenet-on-the-edge/mnt_data/unpacked/imagenet2012_subset/1pct/5.0.0/label.labels.txt')
  nbatches = 10
  batchsize = 32


  #load interpreter from tflite
  try:
    interpreter = tf.lite.Interpreter(
          model_path='/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/models/tflite_models/'+model_name+'.tflite')
    interpreter.resize_tensor_input(input_index = 0,tensor_size=get_input_shape(model_name , nbatches=1, batchsize=1))

    interpreter.allocate_tensors() # Allocate Memory
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #print(get_input_shape(model_name , nbatches=1, batchsize=1))
    #print(output_details[0]['quantization'])
    print(input_details)

    # get evaluation data
    #preprocess = load_preprocessing(model_name)
    #ds, _ = load_data(preprocess=preprocess, n_batches=nbatches, batch_size=batchsize)

    
    #ds = tf.keras.utils.image_dataset_from_directory(os.path.join('/home/staay/Git/imagenet-on-the-edge/image_data/50/',model_name), image_size =(input_details[0]['shape'][1],input_details[0]['shape'][2]), shuffle = False)
    x = []
    y = []
    #Get Images from File System
    images_dir = os.path.join('/home/staay/Git/imagenet-on-the-edge/image_data/50/',model_name)
    for class_number in os.listdir(images_dir):
      print(class_number)
      for saved_array in os.listdir(os.path.join(images_dir,class_number)): #DirNames
        image_file = os.path.join(images_dir,class_number,saved_array) #FileNames
        x.append(np.load(image_file))
        y.append(int(class_number))
      

    model = prepare_model(model_name) #keras.engine.functional.Functional' / tf.keras.applications.MobileNetV3Small
    

    normal_accuracy = tf.keras.metrics.Accuracy()
    tflite_accuracy = tf.keras.metrics.Accuracy()
    

    input_scale, input_zero_point = input_details[0]['quantization']
    

    for i in range(0,32):
      input_data = x[i][None,:,:,:]

      #quantize input image
      input_data = (input_data / input_scale) + input_zero_point
      input_data = np.around(input_data) 
      input_data = input_data.astype(np.int8)
      
      #print(input_data.shape)
      #print()
      interpreter.set_tensor(input_details[0]['index'], input_data)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      tf_results = np.squeeze(output_data)
      highest_pred_loc = np.argmax(tf_results)


      predictions = model.predict(x[i][None,:,:,:])
      

      normal_accuracy.update_state([predictions.argmax(axis=-1)[0]],[int(y[i])])
      tflite_accuracy.update_state(highest_pred_loc,[int(y[i])])
    result_text += model_name 
    result_text += '\n'
    result_text += str(normal_accuracy.result().numpy())
    result_text += '\n'
    result_text += str(tflite_accuracy.result().numpy())
    result_text += '\n'
    print(result_text)
  except Exception as e:
    print('COULD NOT COMPARE '+model_name)
    print(traceback.format_exc())



      
    
with open("accuracyresults.txt", "w") as text_file:
    text_file.write(result_text)
