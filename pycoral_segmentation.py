    

#import numpy as np
#import pycoral
#from PIL import Image
#import cv2


import argparse
import random
import os
import json
from helper_scripts.util import PatchedJSONEncoder
from helper_scripts.util import create_output_dir
random.seed(21)



import argparse
import random
import os
import json
from helper_scripts.util import PatchedJSONEncoder
from codecarbon import OfflineEmissionsTracker
from helper_scripts.util import create_output_dir
from ultralytics import YOLO
random.seed(21)



def edgetpu_inference(model_name, dataset, targetDir):
  from pycoral.utils import edgetpu # NECESSARY!

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
  duration = accuracy = 0
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
  


