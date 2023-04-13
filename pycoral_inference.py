
import argparse
import time
import numpy as np
import random
from PIL import Image
import os
import re
import json
from util import PatchedJSONEncoder
import pathlib
from codecarbon import OfflineEmissionsTracker
from util import create_output_dir
random.seed(21)

from threading import Thread
import cv2

def showImage(img):

    def show(img):
        cv2.imshow(str(dt.datetime.today()), img)
        cv2.waitKey()

    Thread(target=show, args=[img]).start()


def getImagenetLabelDict():
  cifar_imagenet_mapping_path = os.getcwd()+'/cifar_to_imagenet_classes.txt'
  with open(cifar_imagenet_mapping_path, 'r') as cifar_imagenet_map:
          label_lines = cifar_imagenet_map.readlines()
          mappingDict = {}
          currentValueList = []
          currentKey  = label_lines[0][:-1]
          for line in label_lines[1:]:
              if not line.startswith('-'):
                      mappingDict[currentKey] = currentValueList
                      currentKey = line[:-1]
                      currentValueList = []
              elif line.startswith('-'):
                  currentValueList.append(re.findall('n\d+:',line)[0][:-1])
          mappingDict[currentKey] = currentValueList
  return mappingDict

def calcAccuracy(x,y,imageCount):
  # Without TF because of compatibility issues
  correct = 0
  for i in range(0,imageCount):
    if x[i] in y[i]:
      correct = correct + 1
  
  return correct/imageCount


def edgetpu_inference(model_name,x,targetDir):
  #print('START EDGETPU')
  from pycoral.utils import edgetpu
  from pycoral.utils import dataset
  from pycoral.adapters import common
  from pycoral.adapters import classify
  interpreter = edgetpu.make_interpreter(os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_edgetpu.tflite'))
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_dtype = input_details[0]['dtype']
  output_size = output_details[0]['shape'][1]
  input_scale, input_zero_point = input_details[0]['quantization']
  x_quant = []
  for img in x:
    x_to_quant = (img / input_scale) + input_zero_point
    x_to_quant = np.around(x_to_quant)
    x_to_quant= x_to_quant.astype(input_dtype)
    x_quant.append(x_to_quant)
  classification_result = np.empty((imageCount,output_size),dtype=input_dtype)
  emissions_tracker = OfflineEmissionsTracker( log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  emissions_tracker.start()
  tflite_start_time = time.time()
  for i in range(0,imageCount):
          input_data = x_quant[i]#[None,:,:,:]
          interpreter.set_tensor(input_details[0]['index'], input_data)
          interpreter.invoke()            
          classification_result[i] = interpreter.get_tensor(output_details[0]['index'])
  tflite_end_time = time.time()
  emissions_tracker.stop()
  highest_pred_list = []
  workingDir = os.getcwd()
  labelsFilePath =  workingDir +'/mnt_data/unpacked/imagenet2012_subset/1pct/5.0.0/label.labels.txt'
  with open(labelsFilePath) as labelsFile:
      labelsArray = labelsFile.readlines()
  for i in  range(0,imageCount):
    highest_pred_list.append(labelsArray[np.argmax(classification_result[i])])
  return (tflite_end_time - tflite_start_time)*1000, highest_pred_list


def tflite_inference(model_name,x,targetDir):
  import tflite_runtime.interpreter as tflite 
  interpreter = tflite.Interpreter(
        model_path=os.path.join(os.getcwd(),'/mnt_data/staay/models/tflite_models/'+model_name+'.tflite'))
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  input_dtype = input_details[0]['dtype']
  output_details = interpreter.get_output_details()
  output_size = output_details[0]['shape'][1]
  input_scale, input_zero_point = input_details[0]['quantization']
  x_quant = (x / input_scale) + input_zero_point
  x_quant = np.around(x_quant) 
  x_quant = x_quant.astype(input_dtype)
  classification_result = np.empty((imageCount,output_size),dtype=input_dtype)
  emissions_tracker = OfflineEmissionsTracker( log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  emissions_tracker.start()
  tflite_start_time = time.time()
  for i in range(0,imageCount):
          print(i)
          input_data = x_quant[i][None,:,:,:]
          interpreter.set_tensor(input_details[0]['index'], input_data)
          interpreter.invoke()            
          classification_result[i] = interpreter.get_tensor(output_details[0]['index'])
  tflite_end_time = time.time()
  emissions_tracker.stop()
  highest_pred_list = []
  for i in  range(0,imageCount):
    highest_pred_list.append(np.argmax(classification_result[i]))
  return (tflite_end_time - tflite_start_time)*1000, highest_pred_list


def tf_inference(model_name,x,targetDir):
  from load_models import prepare_model
  model = prepare_model(model_name) 
  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  x_to_predict = np.stack( x, axis=0 ).squeeze()
  emissions_tracker.start()
  tflite_start_time = time.time()
  prediction = model.predict(x_to_predict, batch_size=32)
  tflite_end_time = time.time()
  emissions_tracker.stop()
  final_predictions = []
  workingDir = os.getcwd()
  labelsFilePath =  workingDir +'/mnt_data/unpacked/imagenet2012_subset/1pct/5.0.0/label.labels.txt'
  with open(labelsFilePath) as labelsFile:
      labelsArray = labelsFile.readlines()
  for i in range(0, imageCount):
    final_predictions.append(labelsArray[np.argmax(prediction[i])])
  return (tflite_end_time - tflite_start_time)*1000, final_predictions


def loadData(dataDir,imageCount, dataset):
  # Load Images from Numpy Files in DataDir
  if dataset == 'cifar10':
    labelMappingDict = getImagenetLabelDict()
  listOfImages = []
  listOfLabels = []
  for root, dirs, files in os.walk(dataDir):
    for dir in dirs:
      for root2, dirs2, files2 in os.walk(os.path.join(dataDir,dir)):
        for file2 in files2:
          if dataset == 'imagenet':
            listOfLabels.append([str(dir)])
          elif dataset == 'cifar10':
            listOfLabels.append(labelMappingDict[str(dir)])
          listOfImages.append(np.load(os.path.join(os.path.join(dataDir,dir,file2))))
  randomIndices = random.sample(range(0, len(listOfImages)), imageCount)
  drawImages = [listOfImages[i]  for i in randomIndices]
  drawLabels = [listOfLabels[i]  for i in randomIndices]
  return drawImages, drawLabels

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mn','--modelname', default='ResNet50', help='Model to view')
  parser.add_argument('-b',"--backend", default="tflite_edgetpu", type=str, choices=["tflite_edgetpu","tflite","tf_gpu","tf_cpu"], help="machine learning software to use")
  parser.add_argument('-ic','--imageCount', default = 320, help="Size of validation dataset")
  parser.add_argument('-md', '--monitoringdir' , default = os.path.join(os.getcwd(),'mnt_data/staay/eval3') )
  parser.add_argument('-d',"--dataset", default="imagenet", type=str, choices=["imagenet","cifar10","cifar100"], help="dataset to use")

  args = parser.parse_args()


  model_name = args.modelname
  imageCount = int(args.imageCount)
  assert imageCount % 32 == 0, f"pick imagecount that is a multiple of 32, got {imageCount}"
  monitoringDir = args.monitoringdir
  targetDir = create_output_dir(dir = monitoringDir,prefix = 'infer', config =args.__dict__)
  #targetDir = os.path.join(monitoringDir,model_name, args.backend) #Where to save the monitoring summary
  if args.dataset == 'imagenet':
    dataDir = os.path.join(os.getcwd(),'mnt_data/staay/imagenet_data', model_name)
  elif args.dataset == 'cifar10':
    dataDir = os.path.join(os.getcwd(),'mnt_data/staay/cifar10_data', model_name) # model_name
  if not os.path.exists(targetDir):
    os.makedirs(targetDir)

  
  
  x, listOfLabels = loadData(dataDir,imageCount,dataset = args.dataset)

  if args.backend == 'tflite_edgetpu':
    duration, highest_pred_list = edgetpu_inference(model_name,x,targetDir)     
    accuracy = calcAccuracy(highest_pred_list,listOfLabels,imageCount)
  elif args.backend == 'tflite':
    duration, highest_pred_list = tflite_inference(model_name,x,targetDir)     
    accuracy = calcAccuracy(highest_pred_list,listOfLabels,imageCount)
  elif args.backend == 'tf_gpu' or args.backend == 'tf_cpu':
    if args.backend == 'tf_cpu':
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    duration, highest_pred_list = tf_inference(model_name, x, targetDir)

    #highest_pred_list[1] = listOfLabels[1][1]
    accuracy = calcAccuracy(highest_pred_list,listOfLabels,imageCount)
  
  
  print('PREDICTED LABEL '+highest_pred_list[1])
  print('TRUE LABELS '+ str(listOfLabels[1]))
  results = {
                    'duration_ms':duration,
                    'dataset':'ImageNet',
                    'avg_duration_ms': duration/imageCount,
                    'output_dir': 'mnt_data/staay/eval3',
                    'datadir': dataDir,
                    'model': model_name,
                    'backend': args.backend,
                    'accuracy': accuracy,
                    'validation_size': imageCount,
                    'batch_size': 32 if args.backend == 'tf_gpu' or args.backend == 'tf_cpu' else 1
                }
  print(results)

  with open(os.path.join(targetDir, 'validation_results.json'), 'w') as rf:
      json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


