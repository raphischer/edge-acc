    

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
  if "8" in model_name: 
    this_task = 'segment'
  else:
    this_task = 'detect'

  model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_full_integer_quant_edgetpu.tflite'), task = this_task)
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
    print(metrics)

  return  metrics.speed['inference'], metrics.results_dict['metrics/precision(B)'], metrics.results_dict['metrics/recall(B)'], metrics.results_dict['metrics/mAP50(B)'],  metrics.results_dict['metrics/mAP50-95(B)'],metrics.results_dict['metrics/precision(M)'],metrics.results_dict['metrics/recall(M)'],metrics.results_dict['metrics/mAP50(M)'],metrics.results_dict['metrics/mAP50-95(M)']

def tf_inference_cpu(model_name, dataset, targetDir):
  if "8" in model_name: 
    this_task = 'segment'
  else:
    this_task = 'detect'
  os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
  model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/saved_models/'+model_name+'_saved_model'), task = this_task)
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

  return  metrics.speed['inference'], metrics.results_dict['metrics/precision(B)'], metrics.results_dict['metrics/recall(B)'], metrics.results_dict['metrics/mAP50(B)'],  metrics.results_dict['metrics/mAP50-95(B)'],metrics.results_dict['metrics/precision(M)'],metrics.results_dict['metrics/recall(M)'],metrics.results_dict['metrics/mAP50(M)'],metrics.results_dict['metrics/mAP50-95(M)']

def tf_inference_gpu(model_name, dataset, targetDir):
  if "8" in model_name: 
    this_task = 'segment'
  else:
    this_task = 'detect'
  os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  model = YOLO(os.path.join(os.getcwd(),'mnt_data/staay/models/saved_models/'+model_name+'_saved_model'), task = this_task)
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




  return  backend_change, metrics.speed['inference'], metrics.results_dict['metrics/precision(B)'], metrics.results_dict['metrics/recall(B)'], metrics.results_dict['metrics/mAP50(B)'],  metrics.results_dict['metrics/mAP50-95(B)'],metrics.results_dict['metrics/precision(M)'],metrics.results_dict['metrics/recall(M)'],metrics.results_dict['metrics/mAP50(M)'],metrics.results_dict['metrics/mAP50-95(M)']



           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mn','--modelname', default='yolov8s-seg', help='Model to view')
  parser.add_argument('-b',"--backend", default="tflite_edgetpu", type=str, choices=["tflite_edgetpu","tf_gpu","tf_cpu"], help="machine learning software to use")
  parser.add_argument('-md', '--monitoringdir' , default = os.path.join(os.getcwd(),'mnt_data/staay/eval_seg_test') )
  parser.add_argument('-d',"--dataset", default="COCO128", type=str, choices=["COCO","COCO128"], help="dataset to use")

  args = parser.parse_args()


  model_name = args.modelname
  dataset = args.dataset
  imageCount = 5000 if dataset == "COCO" else 128

  monitoringDir = args.monitoringdir
  targetDir = create_output_dir(dir = monitoringDir,prefix = 'infer_segmentation', config =args.__dict__)
  #targetDir = os.path.join(monitoringDir,model_name, args.backend) #Where to save the monitoring summary
  
  backend = args.backend
  duration = accuracy = precision_B = recall_B = mAP50_B = mAP50_95_B = precision_M = recall_M = mAP50_M =  mAP50_95_M = 0
  if backend == 'tflite_edgetpu':
    duration, precision_B, recall_B, mAP50_B, mAP50_95_B, precision_M, recall_M, mAP50_M,  mAP50_95_M  = edgetpu_inference(model_name, dataset, targetDir)     
  elif backend == 'tf_gpu': # No Multi GPU
    backend_change, duration, precision_B, recall_B, mAP50_B, mAP50_95_B, precision_M, recall_M, mAP50_M,  mAP50_95_M  = tf_inference_gpu(model_name, dataset, targetDir)
    if backend_change:
      backend = 'tf_cpu'
  elif backend == 'tf_cpu':
    duration, precision_B, recall_B, mAP50_B, mAP50_95_B, precision_M, recall_M, mAP50_M,  mAP50_95_M = tf_inference_cpu(model_name, dataset, targetDir)

   
  
  
  
  results = {
                    'duration_ms':duration*imageCount,
                    'dataset':dataset,
                    'avg_duration_ms': duration,
                    'output_dir': monitoringDir,
                    'datadir': os.getcwd()+('/mnt_data/staay/coco' if dataset == "COCO" else '/mnt_data/staay/coco128-seg') ,
                    'model': model_name,
                    'backend': backend,
                    'precision_B': precision_B,
                    'recall_B' : recall_B,
                    'mAP50_B' : mAP50_B,
                    'mAP50_95_B' :mAP50_95_B,
                    'precision_M' : precision_M,
                    'recall_M' : recall_M,
                    'mAP50_M' :  mAP50_M,
                    'mAP50_95_M' :mAP50_95_M,
                    'validation_size': imageCount,
                    'batch_size': 1 if backend == 'tf_gpu' or backend == 'tf_cpu' else 1,
                    'task': 'segmentation'
                }
  print(results)

  with open(os.path.join(targetDir, 'validation_results.json'), 'w') as rf:
      json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


