
import argparse
import time
import numpy as np
from PIL import Image
import os
import json
from util import PatchedJSONEncoder
import pathlib
from codecarbon import OfflineEmissionsTracker
from util import create_output_dir

def calcAccuracy(x,y,imageCount):
  # Without TF because of compatibility issues
  #print('CALCULATE ACCURACY')
  correct = 0
  for i in range(0,imageCount):
    if x[i]==y[i]:
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
  x_quant = (x / input_scale) + input_zero_point
  x_quant = np.around(x_quant) 
  x_quant = x_quant.astype(input_dtype)
  classification_result = np.empty((imageCount,output_size),dtype=input_dtype)
  emissions_tracker = OfflineEmissionsTracker(measure_power_secs=9, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  emissions_tracker.start()
  tflite_start_time = time.time()
  for i in range(0,imageCount):
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
  emissions_tracker = OfflineEmissionsTracker(measure_power_secs=9, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
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
  emissions_tracker = OfflineEmissionsTracker(measure_power_secs=9, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  emissions_tracker.start()
  tflite_start_time = time.time()
  prediction = model.predict(x, batch_size=32)
  tflite_end_time = time.time()
  emissions_tracker.stop()
  final_predictions = []
  for i in range(0, imageCount):
    final_predictions.append(np.argmax(prediction[i]))
  return (tflite_end_time - tflite_start_time)*1000, final_predictions


def loadData(dataDir):
  # Load Images from Numpy Files in DataDir
  listOfImages = []
  listOfLabels = []
  for root, dirs, files in os.walk(dataDir):
    for dir in dirs:
      for root2, dirs2, files2 in os.walk(os.path.join(dataDir,dir)):
        for file2 in files2:
          listOfLabels.append(int(dir))
          listOfImages.append(np.load(os.path.join(os.path.join(dataDir,dir,file2))))
  x = np.stack( listOfImages, axis=0 )
  return x, listOfLabels

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mn','--modelname', default='ResNet50', help='Model to view')
  parser.add_argument('-b',"--backend", default="tflite", type=str, choices=["tflite_edgetpu","tflite","tf_gpu","tf_cpu"], help="machine learning software to use")
  parser.add_argument('-ic','--imageCount', default = 320, help="Size of validation dataset")
  args = parser.parse_args()


  model_name = args.modelname
  imageCount = int(args.imageCount)
  assert imageCount % 32 == 0, f"pick imagecount that is a multiple of 32, got {imageCount}"
  monitoringDir = os.path.join(os.getcwd(),'mnt_data/staay/eval')
  targetDir = create_output_dir(dir = monitoringDir,prefix = 'infer', config =args.__dict__)
  #targetDir = os.path.join(monitoringDir,model_name, args.backend) #Where to save the monitoring summary
  dataDir = os.path.join(os.getcwd(),'mnt_data/staay/image_data', model_name)
  if not os.path.exists(targetDir):
    os.makedirs(targetDir)

  
  
  x, listOfLabels = loadData(dataDir)

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
    accuracy = calcAccuracy(highest_pred_list,listOfLabels,imageCount)
    


  results = {
                    'duration_ms':duration,
                    'dataset':'ImageNet',
                    'avg_duration_ms': duration/imageCount,
                    'output_dir': 'mnt_data/staay/eval',
                    'datadir': 'mnt_data/staay/image_data/'+model_name,
                    'model': model_name,
                    'backend': args.backend,
                    'accuracy': accuracy,
                    'validation_size': imageCount,
                    'batch_size': 32 if args.backend == 'tf_gpu' or args.backend == 'tf_cpu' else 1
                }
  print(results)

  with open(os.path.join(targetDir, 'validation_results.json'), 'w') as rf:
      json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


