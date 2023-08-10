
import argparse
import time
import numpy as np
import random
from PIL import Image
import os
import json
from helper_scripts.util import PatchedJSONEncoder
from codecarbon import OfflineEmissionsTracker
from helper_scripts.util import create_output_dir
from ultralytics import YOLO
random.seed(21)

def edgetpu_inference_pycoral(model_name, images, targetDir):
  from pycoral.utils import edgetpu
  from pycoral.utils import dataset
  from pycoral.adapters import common
  from pycoral.adapters import classify


  interpreter = edgetpu.make_interpreter(
    os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_full_integer_quant_edgetpu.tflite')
    )
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
  segmentation_result = np.empty((imageCount,output_size),dtype=input_dtype)


  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  emissions_tracker.start()
  tflite_start_time = time.time()
  for img in x_quant:
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.int8)[None,:,:,:])
    interpreter.invoke()            
    result = interpreter.get_tensor(output_details[0]['index'])
    #segmentation_result.append(segmentation_result)

  tflite_end_time = time.time()
  emissions_tracker.stop()
 
  return (tflite_end_time - tflite_start_time)*1000, segmentation_result

def edgetpu_inference_tflite(model_name, images, targetDir):
  import tflite_runtime.interpreter as tflite
  from tflite_runtime.interpreter import load_delegate


  interpreter = tflite.Interpreter(
    #os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_full_integer_quant_edgetpu_normal.tflite'),
    os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_full_integer_quant_edgetpu.tflite'),

    experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  segmentation_result = []
  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  emissions_tracker.start()
  tflite_start_time = time.time()
  for img in images:
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.int8)[None,:,:,:])
    interpreter.invoke()            
    result = interpreter.get_tensor(output_details[0]['index'])
    #segmentation_result.append(segmentation_result)

  tflite_end_time = time.time()
  emissions_tracker.stop()
 
  return (tflite_end_time - tflite_start_time)*1000, segmentation_result

def calcAccuracy(model_name, backend):
  print('START VALIDATION')
  from ultralytics import YOLO
  if backend == 'tflite_edgetpu':
    model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_full_integer_quant_edgetpu.tflite'))
    print(model.val())
  else:
    model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/saved_models/')+model_name)
    #print(model.val())
  print('FINISHED VALIDATION')

  return 'None'

def edgetpu_inference(model_name, dataset, targetDir):
  
  model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_full_integer_quant_edgetpu.tflite'), task = 'segment')
  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  if dataset == "COCO":
    print('START COCO INFERENCE')
    emissions_tracker.start()
    metrics = model.val('mnt_data/staay/coco.yaml')
    emissions_tracker.stop()
    print('INFERENCE FINISHED')
  else:
    print('START COCO128 INFERENCE')
    emissions_tracker.start()
    metrics = model.val('mnt_data/staay/coco128-seg.yaml')
    emissions_tracker.stop()
    print('INFERENCE FINISHED')

  return  metrics.speed['inference'], metrics.results_dict['metrics/precision(B)']

def tf_inference_cpu(model_name, dataset, targetDir):
  os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
  model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/saved_models/'+model_name+'_saved_model'), task = 'segment')
  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  if dataset == "COCO":
    print('START COCO INFERENCE')
    emissions_tracker.start()
    metrics = model.val('mnt_data/staay/coco.yaml',  device='cpu')
    emissions_tracker.stop()
    print('INFERENCE FINISHED')
  else:
    print('START COCO128 INFERENCE')
    emissions_tracker.start()
    metrics = model.val('mnt_data/staay/coco128-seg.yaml',  device='cpu')
    emissions_tracker.stop()
    print('INFERENCE FINISHED')

  return  metrics.speed['inference'], metrics.results_dict['metrics/precision(B)']

def tf_inference_gpu(model_name, dataset, targetDir):
  os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/saved_models/'+model_name+'_saved_model'), task = 'segment')
  emissions_tracker = OfflineEmissionsTracker(log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
  try:
    if dataset == "COCO":
      print('START COCO INFERENCE')
      emissions_tracker.start()
      metrics = model.val('mnt_data/staay/coco.yaml', device=0)
      emissions_tracker.stop()
      print('INFERENCE FINISHED')
    else:
      print('START COCO128 INFERENCE')
      emissions_tracker.start()
      metrics = model.val('mnt_data/staay/coco128-seg.yaml', device=0)
      emissions_tracker.stop()
      print('INFERENCE FINISHED')

    backend_change = False
  except: # NO GPU DETECTED -> RUN ON CPU
    if dataset == "COCO":
      print('START COCO INFERENCE')
      emissions_tracker.start()
      metrics = model.val('mnt_data/staay/coco.yaml', device='cpu')
      emissions_tracker.stop()
      print('INFERENCE FINISHED')
    else:
      print('START COCO128 INFERENCE')
      emissions_tracker.start()
      metrics = model.val('mnt_data/staay/coco128-seg.yaml', device='cpu')
      emissions_tracker.stop()
      print('INFERENCE FINISHED')
    backend_change = True




  return  metrics.speed['inference'], metrics.results_dict['metrics/precision(B)'], backend_change

def loadData(dataDir, imageCount):
  # Load Images from Numpy Files in DataDir
  imageDir = os.path.join(dataDir,'val2017')
  labelDir = os.path.join(dataDir,'labels/val2017')
  listOfImages = []
  listOfImagesRaw = []
  listOfLabels = []
  for root, dirs, files in os.walk(imageDir):
    for file in files:
      if len(listOfImages) < imageCount: # das hier weg
        try:
          img_opened = Image.open(os.path.join(imageDir,file))
          img = np.asarray(img_opened.resize((640,640), Image.LANCZOS))
          img_opened.close()
        except Exception as e: 
          print(e)
          print(f'couldnt open {os.path.join(imageDir,file)}')
    

        labelName = file[:-3]+'txt'
        labelPath = os.path.join(labelDir,labelName)
        try:
          with open(labelPath) as f:
            contents = f.read()
            if len(contents)>0 and img.shape == (640, 640, 3):
              listOfLabels.append(contents)
              listOfImages.append(img)
        except:
          print(f'couldnt load {labelPath}')
    print(len(listOfImages))
    print(len(listOfLabels))

  randomIndices = random.sample(range(0, len(listOfImages)), min(imageCount, len(listOfImages)) )
  drawImages = [listOfImages[i]  for i in randomIndices]
  drawLabels = []
  #drawLabels = [listOfLabels[i]  for i in randomIndices]
  return drawImages, drawLabels

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mn','--modelname', default='yolov8n-seg', help='Model to view')
  parser.add_argument('-b',"--backend", default="tflite_edgetpu", type=str, choices=["tflite_edgetpu","tf_gpu","tf_cpu"], help="machine learning software to use")
  parser.add_argument('-md', '--monitoringdir' , default = os.path.join(os.getcwd(),'mnt_data/staay/eval_seg_test') )
  parser.add_argument('-d',"--dataset", default="COCO128", type=str, choices=["COCO","COCO128"], help="dataset to use")

  args = parser.parse_args()


  model_name = args.modelname
  dataset = args.dataset
  imageCount = 5000 if dataset == "COCO" else 128

  monitoringDir = args.monitoringdir
  targetDir = create_output_dir(dir = monitoringDir,prefix = 'infer', config =args.__dict__)
  #targetDir = os.path.join(monitoringDir,model_name, args.backend) #Where to save the monitoring summary
  
  backend = args.backend

  if backend == 'tflite_edgetpu':
    duration, accuracy = edgetpu_inference(model_name, dataset, targetDir)     
  elif backend == 'tf_gpu': # No Multi GPU
    duration, accuracy, backend_change = tf_inference_gpu(model_name, dataset, targetDir)
    if backend_change:
      backend = 'tf_cpu'
  elif backend == 'tf_cpu':
    duration, accuracy = tf_inference_cpu(model_name, dataset, targetDir)

   
  
  
  
  results = {
                    'duration_ms':duration*imageCount,
                    'dataset':dataset,
                    'avg_duration_ms': duration,
                    'output_dir': monitoringDir,
                    'datadir': os.getcwd()+('/mnt_data/staay/coco' if dataset == "COCO" else '/mnt_data/staay/coco128-seg') ,
                    'model': model_name,
                    'backend': backend,
                    'accuracy': accuracy,
                    'validation_size': imageCount,
                    'batch_size': 1 if backend == 'tf_gpu' or backend == 'tf_cpu' else 1,
                    'task': 'segmentation'
                }
  print(results)

  with open(os.path.join(targetDir, 'validation_results.json'), 'w') as rf:
      json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


