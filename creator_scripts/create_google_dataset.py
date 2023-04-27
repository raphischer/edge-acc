import os 
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from skimage import img_as_ubyte

from PIL import Image
import random 

def loadData(dataDir,imageCount):
  # Load Images from Numpy Files in DataDir

  listOfImages = []
  listOfLabels = []
  for root, dirs, files in os.walk(dataDir):
    for dir in dirs:
      for root2, dirs2, files2 in os.walk(os.path.join(dataDir,dir)):
        for file2 in files2:
            listOfLabels.append([str(dir)])
            listOfImages.append(np.load(os.path.join(os.path.join(dataDir,dir,file2))))
  randomIndices = random.sample(range(0, len(listOfImages)), imageCount)
  drawImages = [listOfImages[i]  for i in randomIndices]
  drawLabels = [listOfLabels[i]  for i in randomIndices]
  return drawImages, drawLabels

def createGoogleDataset(model_name, directory, imageCount = 12000):
    sourceimageDataDir =  os.path.join(os.getcwd(),'mnt_data/staay/imagenet_data','None')
    targetimageDataDir =  os.path.join(os.getcwd(),'mnt_data/staay/imagenet_data', model_name)

    modelDir = os.path.join(directory,model_name+'.tflite')
    x, listOfLabels = loadData(sourceimageDataDir,imageCount)
    interpreter = edgetpu.make_interpreter(modelDir)
    interpreter.allocate_tensors()

  # Model must be uint8 quantized
    if common.input_details(interpreter, 'dtype') != np.uint8:
        raise ValueError('Only support uint8 input type.')

    size = common.input_size(interpreter)
    print(size)
    for i in range(len(x)):
        print('X-Shape ',x[i].shape)
        im = Image.fromarray(img_as_ubyte(x[i]))
        x[i] = np.asarray(im.resize(size, Image.ANTIALIAS)).astype('uint8')
       
    

    for i in range(0,len(listOfLabels)):
        labeldir = os.path.join(targetimageDataDir,str(listOfLabels[i]))
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)
        np.save(os.path.join(labeldir,str(i)+'.npy'),x[i])
   



GOOGLEMODELS = ['efficientnet-edgetpu-L_quant_edgetpu','efficientnet-edgetpu-M_quant_edgetpu', 'efficientnet-edgetpu-S_quant_edgetpu','inception_v1_224_quant_edgetpu.tflite','inception_v2_224_quant_edgetpu.tflite','inception_v3_299_quant_edgetpu.tflite','inception_v4_299_quant_edgetpu.tflite','mobilenet_v1_0.5_160_quant_edgetpu.tflite','google_edgetpu_models/mobilenet_v1_0.25_128_quant_edgetpu','mobilenet_v1_0.75_192_quant_edgetpu','mobilenet_v1_1.0_224_quant_edgetpu','mobilenet_v2_1.0_224_quant_edgetpu','tf2_mobilenet_v1_1.0_224_ptq_edgetpu','tf2_mobilenet_v2_1.0_224_ptq_edgetpu','tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu','tfhub_tf2_resnet_50_imagenet_ptq_edgetpu']

for model in GOOGLEMODELS:
    createGoogleDataset(model,os.path.join(os.getcwd(),'mnt_data/staay/models/google_edgetpu_models'),imageCount = 32)
