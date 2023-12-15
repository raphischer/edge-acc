import os 
import numpy as np
from PIL import Image


def createDataset(model_name,count = 3200):
    if model_name == None:
        model_name = 'None'
    imageDataDir =  os.path.join(os.getcwd(),'mnt_data/staay/imagenet_data/',model_name)
    labelsFilePath = os.getcwd()+'/mnt_data/unpacked/imagenet2012_subset/1pct/5.0.0/label.labels.txt'
    with open(labelsFilePath) as labelsFile:
        labelsArray = labelsFile.readlines()
    if not os.path.exists(imageDataDir):
        os.makedirs(imageDataDir)
    from helper_scripts.load_models import load_preprocessing
    from helper_scripts.load_data import load_data
    if model_name != 'None':
        preprocess = load_preprocessing(model_name)
    else:
        preprocess = None
    ds, _ = load_data(preprocess= preprocess, n_batches= count, batch_size=1)
    xlist = []
    y = np.concatenate([y for x, y in ds], axis=0)
    for x, _y in ds:
        xlist.append(x)

    for i in range(0,count):
        #print(f'len: {len(y)}, type: {type(y)}, value: {y}')
        labeldir = os.path.join(imageDataDir,str(labelsArray[y[i]]))
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)
       #print(xlist[i].shape)
        np.save(os.path.join(labeldir,str(i)+'.npy'),np.asarray(xlist[i]))
   
 

#MODELS = ['MobileNetV3Small','MobileNetV3Large','NASNetLarge','ResNet152','ResNet152V2', 'VGG19']
MODELS = ['EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetV2S','EfficientNetV2M']
for model in MODELS:
    createDataset(model, count = 12000)
