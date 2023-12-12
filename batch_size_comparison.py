import argparse
import os
import numpy as np
import random
from helper_scripts.load_models import prepare_model
import time
import platform
import pandas as pd


def load_everything(datadir, model_name, imageCount):
    dataDir = os.path.join(datadir, model_name)
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
    print('Finished Loading Data')
    return drawImages, drawLabels


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-ic','--imageCount', default = 128, help="Size of validation dataset")
  parser.add_argument('-dd', '--datadir' , default = 'mnt_data/staay/imagenet_data' )
  args = parser.parse_args()

  MODEL_NAMES = ['ResNet50', 'DenseNet169',  'ResNet101V2', 'ResNet50V2', 'VGG16', 'Xception',  'DenseNet169',  'DenseNet121', 'DenseNet201', 'EfficientNetB0', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetMobile', 'ResNet101' ]

  model_results_dict = []

  drawImages, drawLabels = load_everything(args.datadir, 'ResNet50', args.imageCount)

  for model_name in MODEL_NAMES:
      model = prepare_model(model_name)
      model_results_dict.append( {'model':model_name} )
      x_to_predict = np.stack( drawImages, axis=0 ).squeeze()
      for batch_size in [1,2,4,8,16,32,64]:
        try:
          tf_start_time = time.time()
          prediction = model.predict(x_to_predict, batch_size=batch_size)
          tf_end_time = time.time()
        except:
          drawImages_alt, drawLabels_alt = load_everything(args.datadir, model_name, args.imageCount)
          x_to_predict_alt = np.stack( drawImages_alt, axis=0 ).squeeze()
          tf_start_time = time.time()
          prediction = model.predict(x_to_predict_alt, batch_size=batch_size)
          tf_end_time = time.time()

        model_results_dict[-1][str(batch_size)] = (tf_end_time - tf_start_time) * 1000
        print(model_name +' '+ str(batch_size)+' '+str((tf_end_time - tf_start_time) * 1000))
      print(model_results_dict)
      df = pd.DataFrame(model_results_dict)
      df.to_csv(f'batch_comparison_results_{platform.node()}_{args.imageCount}.csv')