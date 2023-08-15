
import argparse
import time
import numpy as np
from PIL import Image
import os
import json
from helper_scripts.util import PatchedJSONEncoder
import tflite_runtime.interpreter as tflite


from monitoring import Monitoring

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
      default='/home/mendel/Imagenet_Inference_Tests/monitoring',
      help='output of monitoring')
  parser.add_argument('-b',"--backend", 
      default="tflite_edgetpu", 
      type=str, 
      choices=["tflite_edgetpu","tflite"],  #TODO Implement Tensorflow
      help="machine learning software to use")
  parser.add_argument(
     '-ic',
     '--imageCount',
     default = 32,
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
  targetDir = os.path.join(args.monitoring_dir,model_name, args.backend) #Where to save the summary
  if not os.path.exists(targetDir):
     os.makedirs(targetDir)


  #load interpreter from tflite
  if args.backend == 'tflite_edgetpu':
    interpreter = tflite.Interpreter(
        model_path='/home/mendel/Imagenet_Inference_Tests/models/edgetpu_models/'+model_name+'_edgetpu.tflite', experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
  elif args.backend == 'tflite':
    interpreter = tflite.Interpreter(
        model_path='/home/mendel/Imagenet_Inference_Tests/models/tflite_models/'+model_name+'.tflite')
  
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_scale, input_zero_point = input_details[0]['quantization']

  y = []
  imageList = []
  #Get Images from File System
  images_dir = os.path.join(args.image_directory,model_name)
  for class_number in os.listdir(images_dir):
    #print(class_number)
    for saved_array in os.listdir(os.path.join(images_dir,class_number)): #DirNames
      image_file = os.path.join(images_dir,class_number,saved_array) #FileNames
      imageList.append(np.load(image_file))
      y.append(int(class_number))
      
     
  x = np.stack(imageList, axis = 0 )
     
     

  #print(x.shape)
  #Quantized input for TFLITE
  x_quant = (x / input_scale) + input_zero_point
  x_quant = np.around(x_quant) 
  x_quant = x_quant.astype(np.int8)

  durations = []
  accuracy = []


  for i in range(0,imageCount):
        if args.backend == 'tflite_edgetpu' or args.backend == 'tflite':
          input_data = x_quant[i][None,:,:,:]

          #print(input_data.shape)
          interpreter.set_tensor(input_details[0]['index'], input_data)

          tflite_start_time = time.time()
          #print('START MONITORING')
          tflite_monitoring = Monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval,output_dir = targetDir)
          interpreter.invoke()
          # Time Stop
          tflite_end_time = time.time()
          tflite_monitoring.stop()
          #print('STOP MONITORING')

          durations.append(tflite_end_time - tflite_start_time)
          
          output_data = interpreter.get_tensor(output_details[0]['index'])
          tf_results = np.squeeze(output_data)
          highest_pred = np.argmax(tf_results)

          if int(highest_pred) == int(y[i]):
             accuracy.append(1)
          else:
             accuracy.append(0)
      
        
  results = {
                    'average_duration_ms':(sum(durations) / len(durations))*1000,
                    'model': model_name,
                    'backend': args.backend,
                    'accuracy': sum(accuracy) / len(accuracy),
                    'validation_size': imageCount
                }
  print(results)

  with open(os.path.join(targetDir, 'valitation_results.json'), 'w') as rf:
      json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


