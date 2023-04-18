import os 
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from PIL import Image
from pycoral_inference import loadData

def createGoogleDataset(model_name, directory):
    imageDataDir =  os.path.join(os.getcwd(),'mnt_data/staay/imagenet_data','None')
    modelDir = os.path.join(directory,model_name+'.tflite')
    x, listOfLabels = loadData(imageDataDir)
    interpreter = edgetpu.make_interpreter(modelDir)
    interpreter.allocate_tensors()

  # Model must be uint8 quantized
    if common.input_details(interpreter, 'dtype') != np.uint8:
        raise ValueError('Only support uint8 input type.')

    size = common.input_size(interpreter)
    image = np.resize(size, Image.ANTIALIAS)
    
    

    for i in range(0,len(listOfLabels)):
        labeldir = os.path.join(imageDataDir,str(listOfLabels[i]))
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)
        print('BEFORE')
        print(x[i].shape)
        img = np.asarray(x[i]).resize(size, Image.ANTIALIAS)
        print('AFTER')
        print(img.shape)
        np.save(os.path.join(labeldir,str(i)+'.npy'),img)
   



GOOGLEMODELS = ['efficientnet-edgetpu-L_quant_edgetpu','efficientnet-edgetpu-M_quant_edgetpu', 'efficientnet-edgetpu-S_quant_edgetpu','inception_v1_224_quant_edgetpu.tflite','inception_v2_224_quant_edgetpu.tflite','inception_v3_299_quant_edgetpu.tflite','inception_v4_299_quant_edgetpu.tflite','mobilenet_v1_0.5_160_quant_edgetpu.tflite','google_edgetpu_models/mobilenet_v1_0.25_128_quant_edgetpu','mobilenet_v1_0.75_192_quant_edgetpu','mobilenet_v1_1.0_224_quant_edgetpu','mobilenet_v2_1.0_224_quant_edgetpu','tf2_mobilenet_v1_1.0_224_ptq_edgetpu','tf2_mobilenet_v2_1.0_224_ptq_edgetpu','tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu','tfhub_tf2_resnet_50_imagenet_ptq_edgetpu']

for model in GOOGLEMODELS:
    createGoogleDataset(model,'/Users/lstaay/Documents/imagenet-on-the-edge/mnt_data/staay/models/google_edgetpu_models')