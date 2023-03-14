
import argparse
import time
import numpy as np
from PIL import Image
import os
import json
from util import PatchedJSONEncoder
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.metrics import Accuracy
import tflite_runtime.interpreter as tflite #Needed on EdgeTPU for delegations

from load_models import prepare_model, load_preprocessing
from load_data import load_data

#from pycoral.pybind._pywrap_coral import SetVerbosity as set_verbosity
#set_verbosity(10)
from monitoring import Monitoring

#print(edgetpu.list_edge_tpus())

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

# Dict that translates codes to label names of imagenet
def create_labeldict(path_to_file):
  labelnames = load_labels(args.label_map)
  labelnamedict = {}
  for labelname in labelnames:
    number = labelname.split()[0]
    name = labelname.split()[2]
    labelnamedict[number] = name
  return labelnamedict


           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-mn',
      '--modelname',
      default='ResNet50',
      help='Model to view')
  parser.add_argument(
      '-i',
      '--image_directory',
      default='/home/mendel/Imagenet_Inference_Tests/image_data',
      help='image to be classified')
  parser.add_argument(
      '-md',
      '--monitoring_dir',
      default='/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/monitoring',
      help='output of monitoring')
  parser.add_argument('-b',"--backend", 
      default="tflite", 
      type=str, 
      choices=["tflite_edgetpu","tflite","tf_gpu","tf_cpu"],  #TODO Implement Tensorflow
      help="machine learning software to use")
  parser.add_argument(
     '-ic',
     '--imageCount',
     default = 320,
    help="Size of validation dataset")
  parser.add_argument(
      '--gpu_monitor_interval',
      default=.01, type=float,
      help=' gpu interval')
  parser.add_argument(
      '--cpu_monitor_interval',
      default=.01, type=float,
      help='cpu interval')
  args = parser.parse_args()


  model_name = args.modelname
  imageCount = int(args.imageCount)
  assert imageCount % 32 == 0, f"pick imagecount that is a multiple of 32, got {imageCount}"
  targetDir = os.path.join(args.monitoring_dir,model_name, args.backend) #Where to save the monitoring summary
  if not os.path.exists(targetDir):
    os.makedirs(targetDir)


  if args.backend == 'tflite_edgetpu':
    #interpreter = edgetpu.make_interpreter('/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/models/edgetpu_models/'+model_name+'_edgetpu.tflite')
    interpreter = tflite.Interpreter( model_path='/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/models/tflite_models/'+model_name+'.tflite')
    tflite.load_delegate('libedgetpu.so.1')
    batchsize = 1
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    output_size = output_details[0]['shape'][1]
    input_scale, input_zero_point = input_details[0]['quantization']


  elif args.backend == 'tflite':
    interpreter = tflite.Interpreter(
        model_path='/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/models/tflite_models/'+model_name+'.tflite')
    batchsize = 1
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']
    output_details = interpreter.get_output_details()
    output_size = output_details[0]['shape'][1]

    input_scale, input_zero_point = input_details[0]['quantization']
  elif args.backend == 'tf_gpu':
    batchsize = 1
    model = prepare_model(model_name) 
  elif args.backend == 'tf_cpu':
    batchsize = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = prepare_model(model_name) 


  batchNumber = round(imageCount/batchsize)
  


  preprocess = load_preprocessing(model_name)
  ds, _ = load_data(preprocess=preprocess, n_batches= batchNumber, batch_size=batchsize)

  y = np.concatenate([y for x, y in ds], axis=0)
  x = np.concatenate([x for x, y in ds], axis=0)


  
  print('X-SHAPE' + str(x.shape))
  durations = []
  accuracy = Accuracy()

  tflite_start_time = time.time() 

  if args.backend == 'tflite_edgetpu' or args.backend == 'tflite':
    x_quant = (x / input_scale) + input_zero_point
    x_quant = np.around(x_quant) 
    x_quant = x_quant.astype(input_dtype)
    classification_result = np.empty((imageCount,output_size),dtype=input_dtype)
    print(classification_result.shape)
    tflite_monitoring = Monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval,output_dir = targetDir)

    tflite_start_time = time.time()
    for i in range(0,imageCount):
            input_data = x_quant[i][None,:,:,:]
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()            
            classification_result[i] = interpreter.get_tensor(output_details[0]['index'])
    tflite_end_time = time.time()
    tflite_monitoring.stop()
    highest_pred_list = []
    for i in  range(0,imageCount):
       highest_pred_list.append(np.argmax(classification_result[i]))
    #tf_results = np.squeeze(output_data)
    accuracy.update_state(highest_pred_list,y.tolist())

  elif args.backend == 'tf_gpu' or args.backend == 'tf_cpu':
    tflite_monitoring = Monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval,output_dir = targetDir)
    tflite_start_time = time.time()
    for image in x
    prediction = model.predict(x, batch_size=1)
    tflite_end_time = time.time()
    tflite_monitoring.stop()
    final_predictions = []
    for i in range(0, imageCount):
       final_predictions.append(np.argmax(prediction[i]))
    accuracy.update_state(final_predictions,y.tolist())
    

  results = {
                    'duration_ms':(tflite_end_time - tflite_start_time)*1000,
                    'model': model_name,
                    'backend': args.backend,
                    'accuracy': accuracy.result().numpy(),
                    'validation_size': imageCount
                }
  print(results)

  with open(os.path.join(targetDir, 'valitation_results.json'), 'w') as rf:
      json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


