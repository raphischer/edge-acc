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
   
 

MODELS = [None, 'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetMobile', 'ResNet101', 'ResNet50', 'ResNet101V2',  'ResNet50V2', 'VGG16', 'Xception']
for model in MODELS:
    createDataset(model, count = 12000)
GOOGLEMODELS = ['efficientnet-edgetpu-L_quant_edgetpu','efficientnet-edgetpu-M_quant_edgetpu', 'efficientnet-edgetpu-S_quant_edgetpu','inception_v1_224_quant_edgetpu.tflite','inception_v2_224_quant_edgetpu.tflite','inception_v3_299_quant_edgetpu.tflite','inception_v4_299_quant_edgetpu.tflite','mobilenet_v1_0.5_160_quant_edgetpu.tflite','google_edgetpu_models/mobilenet_v1_0.25_128_quant_edgetpu','mobilenet_v1_0.75_192_quant_edgetpu','mobilenet_v1_1.0_224_quant_edgetpu','mobilenet_v2_1.0_224_quant_edgetpu','tf2_mobilenet_v1_1.0_224_ptq_edgetpu','tf2_mobilenet_v2_1.0_224_ptq_edgetpu','tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu','tfhub_tf2_resnet_50_imagenet_ptq_edgetpu']
