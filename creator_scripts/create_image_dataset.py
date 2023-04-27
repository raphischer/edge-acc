import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import pathlib
from PIL import Image
from helper_scripts.load_data import load_data
from helper_scripts.load_models import load_preprocessing
from tifffile import imsave
import matplotlib.image



def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def create_labeldict(filename):
  labelnames = load_labels(filename)
  labelnamedict = {}
  for labelname in labelnames:
    number = labelname.split()[0]
    name = labelname.split()[2]
    labelnamedict[number] = name
  return labelnamedict

def create_image_dataset(model_name, size):
    image_directory = os.path.join(os.getcwd(),'mnt_data/staay/image_data')
    final_directory = os.path.join(image_directory,str(size),model_name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    # load data and model
    preprocess = load_preprocessing(model_name)
    ds, _ = load_data(preprocess=preprocess, n_batches=size, batch_size=1) #tf.data.Dataset (tensorflow.python.data.ops.dataset_ops.TakeDataset) , tfds.core.DatasetInfo
    y = np.concatenate([y for x, y in ds], axis=0)
    x = np.concatenate([x for x, y in ds], axis=0)
    print(model_name)

    for i in range(0,size):
        #print(x[i])
        if not os.path.exists(os.path.join(final_directory,str(y[i]))):
           os.makedirs(os.path.join(final_directory,str(y[i])))
        
        np.save(os.path.join(final_directory,str(y[i]),str(i)), x[i])
        #matplotlib.image.imsave(os.path.join(final_directory,str(y[i]),str(i)+".png"), x[i])

        #imsave(os.path.join(final_directory,str(y[i]),str(i)+".tiff"),x[i])

        #im = Image.fromarray((x[i]*255).astype(np.uint8)).convert('RGB')
        #im.save(os.path.join(final_directory,str(y[i]),str(i)+".jpeg"))
       

       
   


def create_image_dataset_no_processing(size):
    image_directory = '/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/image_data'
    final_directory = os.path.join(image_directory,str(size),'no_preprocessing')
    labelstext = ''
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    # load data and model
    ds, _ = load_data( n_batches=size, batch_size=1) #tf.data.Dataset (tensorflow.python.data.ops.dataset_ops.TakeDataset) , tfds.core.DatasetInfo
    y = np.concatenate([y for x, y in ds], axis=0)
    imagelist = []
    for x, y in ds:
       imagelist.append(x.numpy())

    for i in range(0,size):
        #print(x[i].shape)
        im = Image.fromarray((x[i] * 1)).convert('RGB')
        im.save(os.path.join(final_directory,str(i)+".jpeg"))
        if i == 0:
            labelstext = labelstext + str(y[i])
        else:
            labelstext = labelstext + '\n' +str(y[i])

       
   
    pathlib.Path(final_directory+'/no_preprocessing/+'+'labels.txt').write_text(labelstext)

  

for model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet152', 'ResNet50', 'ResNet101V2', 'ResNet152V2', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'MobileNetV3Large', 'MobileNetV3Small']:
    create_image_dataset(model_name, 320)
#create_image_dataset_no_processing(10)