
import argparse
import time
import numpy as np
import random
from PIL import Image
import os
import re
import json
from helper_scripts.util import PatchedJSONEncoder
import pathlib
from codecarbon import OfflineEmissionsTracker
from helper_scripts.util import create_output_dir
random.seed(21)
from pynvml import nvmlInit

from threading import Thread


def calcAccuracy(highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10,y,imageCount):
  # Without TF because of compatibility issues
  highest_pred_list_1_stripped = [s.strip('\n') for s in highest_pred_list_1]
  highest_pred_list_3_stripped = [[e.strip('\n') for e in s ]for s in highest_pred_list_3]
  highest_pred_list_5_stripped = [[e.strip('\n') for e in s ] for s in highest_pred_list_5]
  highest_pred_list_10_stripped = [[e.strip('\n') for e in s ] for s in highest_pred_list_10]
  correct_1 = 0
  correct_3 = 0
  correct_5 = 0
  correct_10 = 0

  for i in range(0,imageCount):
    if highest_pred_list_1_stripped[i] in y[i]:
      correct_1 = correct_1 + 1
    for pred in highest_pred_list_3_stripped[i]:
       if pred in y[i]:
          correct_3 = correct_3 + 1 
    for pred in highest_pred_list_5_stripped[i]:
       if pred in y[i]:
          correct_5 = correct_5 +1
    for pred in highest_pred_list_10_stripped[i]:
       if pred in y[i]:
          correct_10 = correct_10 +1
  return correct_1/imageCount, correct_3/imageCount, correct_5/imageCount, correct_10/imageCount

def edgetpu_inference(model_name,x,targetDir,modDir):
  #print('START EDGETPU')
  from pycoral.utils import edgetpu
  from pycoral.utils import dataset
  from pycoral.adapters import common
  from pycoral.adapters import classify
  interpreter = edgetpu.make_interpreter(os.path.join(modDir, 'edgetpu_models', model_name + '_edgetpu.tflite'))
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
          print(classification_result[i].shape)
  tflite_end_time = time.time()
  emissions_tracker.stop()
  final_predictions_1 = []
  final_predictions_3 = []
  final_predictions_5 = []
  final_predictions_10 = []

  workingDir = os.getcwd()
  labelsFilePath =  workingDir +'/mnt_data/unpacked/imagenet2012_subset/1pct/5.0.0/label.labels.txt'
  with open(labelsFilePath) as labelsFile:
      labelsArray = labelsFile.readlines()
  for i in  range(0,imageCount):
    final_predictions_1.append(labelsArray[np.argmax(classification_result[i])])
    ind3 = np.argpartition(classification_result[i], -3)[-3:]
    ind5 = np.argpartition(classification_result[i], -5)[-5:]
    ind10 = np.argpartition(classification_result[i], -10)[-10:]

    
    final_predictions_3.append([labelsArray[x] for x in ind3])
    final_predictions_5.append([labelsArray[x] for x in ind5])
    final_predictions_10.append([labelsArray[x] for x in ind10])

  return (tflite_end_time - tflite_start_time)*1000, final_predictions_1, final_predictions_3, final_predictions_5, final_predictions_10

def ncs2_inference(model_name,x,targetDir,modDir):
  #Make sure to source l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64/setupvars.sh
  print('START NCS2')
  print(x[0].shape)
  from openvino.runtime import Core
  ie = Core()

  classification_model_xml = os.path.join(modDir, 'openVINO', model_name , 'saved_model.xml')

  model = ie.read_model(model=classification_model_xml)
  compiled_model = ie.compile_model(model=model, device_name="MYRIAD")

  input_layer = compiled_model.input(0)
  output_layer = compiled_model.output(0)
  
  classification_result = np.empty((imageCount,output_layer.shape[1]),dtype=np.float32)
  emissions_tracker = OfflineEmissionsTracker( log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  emissions_tracker.start()
  tflite_start_time = time.time()
  for i in range(0,imageCount):
          input_data = x[i]#[None,:,:,:]
          classification_result[i] = compiled_model(input_data)[output_layer]
  tflite_end_time = time.time()
  emissions_tracker.stop()
  final_predictions_1 = []
  final_predictions_3 = []
  final_predictions_5 = []
  final_predictions_10 = []

  labelsFilePath =  'label.labels.txt'
  with open(labelsFilePath) as labelsFile:
      labelsArray = labelsFile.readlines()
  for i in  range(0,imageCount):
    final_predictions_1.append(labelsArray[np.argmax(classification_result[i])])
    ind3 = np.argpartition(classification_result[i], -3)[-3:]
    ind5 = np.argpartition(classification_result[i], -5)[-5:]
    ind10 = np.argpartition(classification_result[i], -10)[-10:]

    
    final_predictions_3.append([labelsArray[x] for x in ind3])
    final_predictions_5.append([labelsArray[x] for x in ind5])
    final_predictions_10.append([labelsArray[x] for x in ind10])

  return (tflite_end_time - tflite_start_time)*1000, final_predictions_1, final_predictions_3, final_predictions_5, final_predictions_10

def tf_inference(model_name,x,targetDir):
  from helper_scripts.load_models import prepare_model
  model = prepare_model(model_name)
  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  x_to_predict = np.stack( x, axis=0 ).squeeze()
  emissions_tracker.start()
  tflite_start_time = time.time()
  prediction = model.predict(x_to_predict, batch_size=32)
  tflite_end_time = time.time()
  emissions_tracker.stop()
  final_predictions_1 = []
  final_predictions_3 = []
  final_predictions_5 = []
  final_predictions_10 = []
  workingDir = os.getcwd()
  labelsFilePath =  workingDir +'/mnt_data/unpacked/imagenet2012_subset/1pct/5.0.0/label.labels.txt'  
  with open(labelsFilePath) as labelsFile:
      labelsArray = labelsFile.readlines()
  for i in range(0, imageCount):
    
    final_predictions_1.append(labelsArray[np.argmax(prediction[i])])

    ind3 = np.argpartition(prediction[i], -3)[-3:]
    ind5 = np.argpartition(prediction[i], -5)[-5:]
    ind10 = np.argpartition(prediction[i], -10)[-10:]

    final_predictions_3.append([labelsArray[x] for x in ind3])
    final_predictions_5.append([labelsArray[x] for x in ind5])
    final_predictions_10.append([labelsArray[x] for x in ind10])

  return (tflite_end_time - tflite_start_time) * 1000, final_predictions_1, final_predictions_3, final_predictions_5, final_predictions_10

def loadData(dataDir,imageCount, dataset = 'imagenet'):
  # Load Images from Numpy Files in DataDir
  listOfImages = []
  listOfLabels = []
  for root, dirs, files in os.walk(dataDir):
    for dir in dirs:
      for root2, dirs2, files2 in os.walk(os.path.join(dataDir,dir)):
        for file2 in files2:
          if len(listOfLabels)<imageCount :
            listOfLabels.append([str(dir)])
            listOfImages.append(np.load(os.path.join(os.path.join(dataDir,dir,file2))))
  randomIndices = random.sample(range(0, len(listOfImages)), min(imageCount, len(listOfImages)) )
  drawImages = [listOfImages[i]  for i in randomIndices]
  drawLabels = [listOfLabels[i]  for i in randomIndices]
  return drawImages, drawLabels

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mn','--modelname', default='ResNet50', help='Model to view')
  parser.add_argument('-b',"--backend", default="tf_cpu", type=str, choices=["tflite_edgetpu","tf_gpu","tf_cpu","NCS2"], help="machine learning software to use") # "all" currently not working due to tfds/tf/pycoral Interpreter wrapper bug
  parser.add_argument('-ic','--imageCount', default = 320, help="Size of validation dataset")
  parser.add_argument('-md', '--monitoringdir' , default = os.path.join(os.getcwd(),'eval1') )
  parser.add_argument('-dd', '--datadir' , default = 'mnt_data/staay/imagenet_data' )
  parser.add_argument('-modd', '--modeldir' , default = 'mnt_data/staay/models/' )


  args = parser.parse_args()
  highest_pred_list_1 =  highest_pred_list_3 = highest_pred_list_5 = highest_pred_list_10 = []
  duration = 0
  dataDir = os.path.join(args.datadir, args.modelname)
 

  backend = args.backend
  failed_GPU_run = False
  model_name = args.modelname
  imageCount = int(args.imageCount)
  assert imageCount % 32 == 0, f"pick imagecount that is a multiple of 32, got {imageCount}"
  monitoringDir = args.monitoringdir
  targetDir = create_output_dir(dir = monitoringDir,prefix = 'infer_classification', config =args.__dict__)
  #targetDir = os.path.join(monitoringDir,model_name, args.backend) #Where to save the monitoring summary
  
  if not os.path.exists(targetDir):
    os.makedirs(targetDir)

  
  
  x, listOfLabels = loadData(dataDir,imageCount,dataset = "imagenet")
  if backend == 'tflite_edgetpu':
    duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = edgetpu_inference(model_name,x,targetDir, args.modeldir)    
  elif backend == 'NCS2':
    duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = ncs2_inference(model_name,x,targetDir, args.modeldir)    
  elif backend == 'tf_gpu' :
    try: 
      nvmlInit() # Will throw an exception if there is no corresponding NVML Shared Library
      duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = tf_inference(model_name, x, targetDir)
    except:
      failed_GPU_run = True # Dont save this run
      os.remove(targetDir+'/config.json')
      os.remove(targetDir+'/execution_platform.json')
      os.remove(targetDir+'/requirements.txt')
      os.rmdir(targetDir)
      print('NO GPU Detected')
  elif backend == 'tf_cpu':
    duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = tf_inference(model_name, x, targetDir)
  if not failed_GPU_run:
    accuracy_k1, accuracy_k3 ,accuracy_k5 ,accuracy_k10 = calcAccuracy(highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10,listOfLabels,imageCount)

  
    results = {
                      'duration_ms':duration,
                      'dataset':'imagenet',
                      'avg_duration_ms': duration/imageCount,
                      'output_dir': monitoringDir,
                      'datadir': dataDir,
                      'model': model_name,
                      'backend': backend,
                      'accuracy_k1': accuracy_k1,
                      'accuracy_k3': accuracy_k3,
                      'accuracy_k5': accuracy_k5,
                      'accuracy_k10': accuracy_k10,
                      'validation_size': imageCount,
                      'batch_size': 32 if backend == 'tf_gpu' or backend == 'tf_cpu' else 1,
                      'task': 'classification'
                  }
    print(results)

    
    with open(os.path.join(targetDir, 'validation_results.json'), 'w') as rf:
        json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


      
        
