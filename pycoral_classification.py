
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
import traceback
import sys
random.seed(21)

from tqdm import tqdm
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


def edgetpu_inference(model_name, x, targetDir, modDir):
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
  x_quant = np.around((x / input_scale) + input_zero_point).astype(input_dtype)
  classification_result = np.empty((imageCount,output_size),dtype=input_dtype)
  emissions_tracker = OfflineEmissionsTracker( log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  try:
    emissions_tracker.start()
    tflite_start_time = time.time()
    for i in tqdm(range(0,imageCount), 'TPU inference'):
      input_data = x_quant[i]#[None,:,:,:]
      interpreter.set_tensor(input_details[0]['index'], input_data)
      interpreter.invoke()            
      classification_result[i] = interpreter.get_tensor(output_details[0]['index'])
    tflite_end_time = time.time()
    emissions_tracker.stop()
  except Exception:
    emissions_tracker.stop()
    print(traceback.format_exc())
    
  final_predictions_1 = []
  final_predictions_3 = []
  final_predictions_5 = []
  final_predictions_10 = []

  workingDir = os.getcwd()
  labelsFilePath =  'label.labels.txt'
  with open(labelsFilePath) as labelsFile:
      labelsArray = labelsFile.readlines()
  for i in range(0,imageCount):
    final_predictions_1.append(labelsArray[np.argmax(classification_result[i])])
    ind3 = np.argpartition(classification_result[i], -3)[-3:]
    ind5 = np.argpartition(classification_result[i], -5)[-5:]
    ind10 = np.argpartition(classification_result[i], -10)[-10:]

    final_predictions_3.append([labelsArray[x] for x in ind3])
    final_predictions_5.append([labelsArray[x] for x in ind5])
    final_predictions_10.append([labelsArray[x] for x in ind10])
  return (tflite_end_time - tflite_start_time)*1000, final_predictions_1, final_predictions_3, final_predictions_5, final_predictions_10


def ncs2_inference(model_name, x, targetDir, modDir):
  #Make sure to source l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64/setupvars.sh
  print('START NCS2')
  from openvino.runtime import Core
  ie = Core()

  classification_model_xml = os.path.join(modDir, 'openVINO', model_name , 'saved_model.xml')

  model = ie.read_model(model=classification_model_xml)
  compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
  output_layer = compiled_model.output(0)
  classification_result = np.empty((imageCount,output_layer.shape[1]),dtype=np.float32)
  
  emissions_tracker = OfflineEmissionsTracker( log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  try:
    emissions_tracker.start()
    tflite_start_time = time.time()
    for i in tqdm( range(0, imageCount), 'NCS inference' ):
      classification_result[i] = compiled_model(x[i])[output_layer]
    tflite_end_time = time.time()
    emissions_tracker.stop()
  except Exception:
    emissions_tracker.stop()
    print(traceback.format_exc())

    pass
  final_predictions_1 = []
  final_predictions_3 = []
  final_predictions_5 = []
  final_predictions_10 = []

  labelsFilePath =  'label.labels.txt'
  with open(labelsFilePath) as labelsFile:
      labelsArray = labelsFile.readlines()
  for i in range(0, imageCount):
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
  x_to_predict = x.squeeze()
  try:
    emissions_tracker.start()
    tflite_start_time = time.time()
    prediction = model.predict(x_to_predict, batch_size=64)
    tflite_end_time = time.time()
    emissions_tracker.stop()
  except:
    emissions_tracker.stop()
    print(traceback.format_exc())
    print('couldnt compute')

    pass
  final_predictions_1 = []
  final_predictions_3 = []
  final_predictions_5 = []
  final_predictions_10 = []
  labelsFilePath =  'label.labels.txt'
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


def loadData(dataDir, imageCount):
  with open('prep_map.json', 'r') as jf:
    # multiple models use the SAME preprocessing - we only store one of them on disc to save some disc space!
    prep_map = json.load(jf)
    prep_name = prep_map[os.path.basename(dataDir)]
  npy_path = 'classification_data'
  npy_images = os.path.join(npy_path, f'images_{imageCount}_{prep_name}.npy')
  npy_labels = os.path.join(npy_path, f'labels_{imageCount}_{prep_name}.npy')
  if os.path.isfile(npy_images):
    # directly load images from local dir
    images = np.load(npy_images)
    labels = np.load(npy_labels)
  else:
    selection_file = f'classification_image_selection_{imageCount}.json'
    if os.path.isfile(selection_file):
      # use predefined image selection
      with open(selection_file, 'r') as jf:
        selection = json.load(jf)
    else:
      # check available data and sample instances
      selection, full_paths = {}, []
      for _, dirs, _ in os.walk(dataDir):
        for dir in dirs:
          full_paths = full_paths + [os.path.join(dir, fname) for fname in os.listdir(os.path.join(dataDir,dir)) if '.npy' in fname]
      # sample some data and write image selection file
      random_indices = random.sample(range(0, len(full_paths)), imageCount)
      for idx in random_indices:
        label, fname = os.path.dirname(full_paths[idx]), os.path.basename(full_paths[idx])
        if label not in selection:
          selection[label] = []
        selection[label].append(fname)
      with open(selection_file, 'w') as jf:
        json.dump(selection, jf)
    # load the data
    listOfImages, listOfLabels = [], []
    for label, files in tqdm(selection.items(), 'loading data'):
      listOfLabels = listOfLabels + [[label]] * len(files)
      for fname in files:
        try:
          listOfImages.append(np.load(os.path.join(dataDir, label, fname)))
        except FileNotFoundError:
          fname_corr = fname.split('/')[0] + '\n/' + fname.split('/')[1]
          listOfImages.append(np.load(os.path.join(dataDir, label, fname_corr)))
    images, labels = np.array(listOfImages), np.array(listOfLabels)
    if not os.path.isdir(npy_path):
       os.makedirs(npy_path)
    np.save(npy_images, images)
    np.save(npy_labels, labels)
  return images, labels

           
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
 
  x, listOfLabels = loadData(dataDir,imageCount)
  if backend == 'tflite_edgetpu':
    duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = edgetpu_inference(model_name,x,targetDir, args.modeldir)    
    print('couldnt compute '+model_name+' with TPU.')
  elif backend == 'NCS2':
    duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = ncs2_inference(model_name,x,targetDir, args.modeldir)    
    print('couldnt compute '+model_name+' with NCS.')
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
                      'batch_size': 64 if backend == 'tf_gpu' or backend == 'tf_cpu' else 1,
                      'task': 'classification'
                  }
    print(results)

    
    with open(os.path.join(targetDir, 'validation_results.json'), 'w') as rf:
        json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


      
        
