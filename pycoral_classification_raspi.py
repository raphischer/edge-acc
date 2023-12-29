
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
from tqdm import tqdm


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

def edgetpu_inference(model_name,x,targetDir,modDir,imageCount):
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
          #print(classification_result[i].shape)
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

def ncs2_inference(model_name,x,targetDir,modDir,imageCount):
  #Make sure to source l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64/setupvars.sh
  #source /opt/intel/openvino_2022.3.1/setupvars.sh
  #source l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64/setupvars.sh
  from openvino.runtime import Core
  import numpy as np


  ie = Core()

  devices = ie.available_devices

  for device in devices:
      device_name = ie.get_property(device, "FULL_DEVICE_NAME")
      print(f"{device}: {device_name}")
  print('START NCS2')
  print(x[0].shape)
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

def tf_inference(model_name,x,targetDir, imageCount, batch_size = 16):
  from helper_scripts.load_models import prepare_model
  model = prepare_model(model_name)
  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  x_to_predict = np.stack( x, axis=0 ).squeeze()
  emissions_tracker.start()
  tflite_start_time = time.time()
  prediction = model.predict(x_to_predict, batch_size )
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
  print('labelsArray:')
  print(len(labelsArray))
  print(prediction.shape)
  print(imageCount)
  for i in range(0, imageCount):
    #print(i)
    final_predictions_1.append(labelsArray[np.argmax(prediction[i])])
    #indices of best n predictions
    ind3 = np.argpartition(prediction[i], -3)[-3:]
    ind5 = np.argpartition(prediction[i], -5)[-5:]
    ind10 = np.argpartition(prediction[i], -10)[-10:]

    final_predictions_3.append([labelsArray[x] for x in ind3])
    final_predictions_5.append([labelsArray[x] for x in ind5])
    final_predictions_10.append([labelsArray[x] for x in ind10])

  return (tflite_end_time - tflite_start_time) * 1000, final_predictions_1, final_predictions_3, final_predictions_5, final_predictions_10

def tflite_inference(model_name,x,targetDir,modDir,imageCount):
  import tensorflow as tf

  interpreter = tf.lite.Interpreter(model_path = os.path.join(modDir, 'tflite_models', model_name + '.tflite'))
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
          #print(classification_result[i].shape)
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

  
def loadData(dataDir,imageCount,current_batch_size, macro_batch_size, iteration):
  print('currentbatch '+str(current_batch_size))
  selection_file = f'classification_image_selection_{imageCount}.json'
  if os.path.isfile(selection_file):
    with open(selection_file, 'r') as jf:
      selection = json.load(jf)
  else:
    # check available data and sample instances
    selection, full_paths = {}, []
    for _, dirs, _ in os.walk(dataDir):
      for dir in dirs:
        full_paths = full_paths + [os.path.join(dir, fname) for fname in os.listdir(os.path.join(dataDir,dir)) if '.npy' in fname]
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
  filecount = -1
  #load data based on Iteration! saves RAM
  relevant_selection_range = range(iteration * macro_batch_size,iteration*macro_batch_size + current_batch_size)
  relevant_selection_slice = slice(iteration * macro_batch_size,iteration*macro_batch_size + current_batch_size)
  for label, files in tqdm(selection.items(), 'loading data'):
    listOfLabels = listOfLabels + [[label]] * len(files)
    for fname in files:
      filecount += 1
      if filecount in relevant_selection_range:
        listOfImages.append(np.load(os.path.join(dataDir, label, fname)))
  return listOfImages,listOfLabels[relevant_selection_slice]


           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mn','--modelname', default='ResNet50', help='Model to view')
  parser.add_argument('-b',"--backend", default="tf_cpu", type=str, choices=["tflite_edgetpu","tf_gpu","tf_cpu","NCS2","tflite"], help="machine learning software to use") # "all" currently not working due to tfds/tf/pycoral Interpreter wrapper bug
  parser.add_argument('-ic','--imageCount', default = 32, help="Size of validation dataset")
  parser.add_argument('-md', '--monitoringdir' , default = os.path.join(os.getcwd(),'eval_test_raspi') )
  parser.add_argument('-modata', '--modelDataDir' , default = 'mnt_data/staay/' )
  parser.add_argument('-mbs','--macroBatchSize', default = 192, help="Size of batches for loaded data")
  parser.add_argument('-bz','--batchSize', default = 1, help="Size of batches for inference")

   
  args = parser.parse_args()
  macro_batch_size = args.macroBatchSize
  batch_size = args.batchSize
  if batch_size > 1 and args.backend not in ['tf_cpu','tf_gpu']:
     print('Batch size will be automatically set to 1 because of the backend you chose.')
                                             
  iterations = round(int(args.imageCount) / macro_batch_size )
  rest = round(int(args.imageCount) % macro_batch_size )
  print('ITERATIONS '+str(iterations))
  print('REST '+str(rest))

  highest_pred_list_1 =  highest_pred_list_3 = highest_pred_list_5 = highest_pred_list_10 = []
  duration = 0
  dataDir = os.path.join(args.modelDataDir,'imagenet_data', args.modelname)
  modelDir = os.path.join(args.modelDataDir, 'models')

  backend = args.backend
  failed_GPU_run = False
  model_name = args.modelname
  imageCount = int(args.imageCount)
  assert imageCount % 32 == 0, f"pick imagecount that is a multiple of 32, got {imageCount}"
  monitoringDir = args.monitoringdir
  targetDir = create_output_dir(dir = monitoringDir,prefix = 'infer_classification', config =args.__dict__)
  
  if not os.path.exists(targetDir):
    os.makedirs(targetDir)
  print(reversed(range(iterations)))
  for it in reversed(range(iterations+1)):
    if it == iterations:
      imageCount = rest
    else: 
      imageCount = macro_batch_size
#dataDir,imageCount,current_batch_size, macro_batch_size, iteration
    x, listOfLabels = loadData(dataDir,int(args.imageCount),imageCount,macro_batch_size,it) # Original Daten Laden 
    print(len(x))
    print(len(listOfLabels))
    print(imageCount)
    if backend == 'tflite_edgetpu':
      duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = edgetpu_inference(model_name,x,targetDir, modelDir,imageCount)    
    elif backend == 'NCS2':
      duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = ncs2_inference(model_name,x,targetDir, modelDir,imageCount)    
    elif backend == 'tf_gpu':
      try: 
        nvmlInit() # Will throw an exception if there is no corresponding NVML Shared Library
        duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = tf_inference(model_name, x, targetDir,imageCount)
      except:
        failed_GPU_run = True # Dont save this run
        os.remove(targetDir+'/config.json')
        os.remove(targetDir+'/execution_platform.json')
        os.remove(targetDir+'/requirements.txt')
        os.rmdir(targetDir)
        print('NO GPU Detected')
    elif backend == 'tf_cpu':
      duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = tf_inference(model_name, x, targetDir,imageCount,batch_size)
    elif backend == 'tflite' :
      duration, highest_pred_list_1,  highest_pred_list_3, highest_pred_list_5, highest_pred_list_10 = tflite_inference(model_name, x, targetDir, modelDir,imageCount)

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
                        'batch_size': batch_size if backend == 'tf_gpu' or backend == 'tf_cpu' else 1,
                        'task': 'classification'
                    }
      print(results)
    with open(os.path.join(targetDir, 'validation_results'+str(it)+'.json'), 'w') as rf:
        json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
    
    
  


      
        
