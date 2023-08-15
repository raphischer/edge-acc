

import argparse
from helper_scripts.load_segmentation_models import load_segmentation_model
import json

import numpy as np
from PIL import Image
import os

from helper_scripts.util import PatchedJSONEncoder
import pathlib
from codecarbon import OfflineEmissionsTracker
from helper_scripts.util import create_output_dir


def edgetpu_segmentation(model, images, labels):
  from pycoral.adapters import common
  from pycoral.adapters import segment
  from pycoral.utils.edgetpu import make_interpreter
  print('HI')
  
#   #interpreter = make_interpreter(os.path.join(os.getcwd(),'mnt_data/staay/models/segmentation_edgetpu_models/'+model_name+'.tflite'))
#   interpreter = make_interpreter('/home/staay/Git/imagenet-on-the-edge/keras_post_training_unet_mv2_128_quant_edgetpu.tflite')

#   interpreter.allocate_tensors()
#   width, height = common.input_size(interpreter)

#   #img = Image.open(args.input)
#   resized_images = []
#   resized_labels= []
#   result_list = []
#   for img in images:
#     resized_images.append( img.resize((width, height), Image.LANCZOS))
#   for label in labels:
#     resized_labels.append(np.asarray(label.resize((width, height), Image.LANCZOS)))
#   for i in range(0, len(resized_images)):
#     result_list.append(np.empty((width, height)))

#   emissions_tracker = OfflineEmissionsTracker( log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir = targetDir)
#   emissions_tracker.start()
#   for i in range(0,len(resized_images)):
#       try:
#         common.set_input(interpreter, resized_images[i])
#         interpreter.invoke()
#         result = segment.get_output(interpreter)
#         result_list[i]=result
#       except:
#         pass

#   emissions_tracker.stop()
#   for i in range(0,len(result_list)):
#     if len(result_list[i].shape) == 3:
#       result_list[i] = np.argmax(result_list[i], axis=-1)+1
#     else:
#       new_width, new_height = resized_images[i].size
#       result_list[i] = result_list[i][:new_height, :new_width]+1

  
#   return(result_list, resized_labels)
#   print('Done. Results saved at')

def tf_segmentation(model, images, labels):
    pedictions = []
    for image in images:
      model.predict(images)
  

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-mn','--modelname', default='keras_post_training_unet_mv2_256_quant_edgetpu', help='Model to view')
  parser.add_argument('-b',"--backend", default="tflite_edgetpu", type=str, choices=["tflite_edgetpu","tflite","tf_gpu","tf_cpu"], help="machine learning software to use")
  #parser.add_argument('-ic','--imageCount', default = 320, help="Size of validation dataset")
  parser.add_argument('-md', '--monitoringdir' , default = os.path.join(os.getcwd(),'mnt_data/staay/eval_seg_1') )
  args = parser.parse_args()


  model_name = args.modelname
 
  monitoringDir = args.monitoringdir
  targetDir = create_output_dir(dir = monitoringDir,prefix = 'segment', config =args.__dict__)
  
  if not os.path.exists(targetDir):
    os.makedirs(targetDir)

  images = []
  labels = []
  if model_name == 'keras_post_training_unet_mv2_256_quant_edgetpu':
    #image_dir = os.path.join('./mnt_data/staay/oxford-iiit-pet_data/images')
    #label_dir = os.path.join('./mnt_data/staay/oxford-iiit-pet_data/annotations/trimaps')
    image_dir = '/home/staay/Git/imagenet-on-the-edge/images'
    label_dir = '/home/staay/Git/imagenet-on-the-edge/trimaps'
    for root, dirs, files in os.walk(image_dir):
      for file in files:
            try:
              img = Image.open(image_dir+'/'+file)
              lbl = Image.open(label_dir+'/'+file[:-3]+'png')
              images.append(img)
              labels.append(lbl)
            except:
              pass 
  print(len(images))

  resized_images = []
  resized_label_list= []
  result_list = []
  for img in images:
    resized_images.append( np.asarray(img.resize((256, 256), Image.LANCZOS)))
  for label in labels:
     resized_label_list.append(np.asarray(label.resize((256, 256), Image.LANCZOS)))


  if args.backend == 'tflite_edgetpu':
    highest_pred_list, resized_label_list = edgetpu_segmentation(model_name,images,labels)
  elif args.backend == 'tf_gpu' or args.backend == 'tf_cpu':

    if args.backend == 'tf_cpu':
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = load_segmentation_model('MobileNetV2', (256,256,3))
    model.summary()
    #images_array = np.asarray(resized_images)
    #print(images_array.shape)
    predictions = model.predict(resized_images)
  

  print(predictions[3])



  ioU_list = []
  for i in range(0,len(resized_label_list)):
    intersection_score = 0
    union_score = 0
    for row in range(resized_label_list[i].shape[0]):
      for col in range(resized_label_list[i].shape[1]):
        if resized_label_list[i][row,col] == highest_pred_list[i][row,col]:
          intersection_score += 1
        union_score += 1

    iou_score = intersection_score/union_score
    ioU_list.append(iou_score)
  IoU = sum(ioU_list)/len(ioU_list) #mean IoU
 
  results = {
                    'dataset':'oxford-iiit-pet_data',
                    'output_dir': monitoringDir,
                    'datadir': image_dir,
                    'model': model_name,
                    'backend': args.backend,
                    'IoU': IoU,
                    'validation_size': len(resized_label_list),
                    'batch_size': 32 if args.backend == 'tf_gpu' or args.backend == 'tf_cpu' else 1
                }
  print(results)

  with open(os.path.join(targetDir, 'validation_results.json'), 'w') as rf:
    json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)
  


