import os
import numpy as np
import random
from helper_scripts.load_models import prepare_model
import time
import pandas as pd
MODEL_NAMES = ['ResNet50', 'DenseNet169',  'ResNet101V2', 'ResNet50V2', 'VGG16', 'Xception',  'DenseNet169',  'DenseNet121', 'DenseNet201', 'EfficientNetB0', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetMobile', 'ResNet101' ]
imageCount = 128

def load_everything(model_name):
    dataDir = os.path.join('mnt_data/staay/imagenet_data', model_name)
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

df = pd.DataFrame(columns=['model','1','2','4','8','16','32','64'])

drawImages, drawLabels = load_everything('ResNet50')


for model_name in MODEL_NAMES:
    model = prepare_model(model_name)
    model_results_dict = {'model':model_name}
    x_to_predict = np.stack( drawImages, axis=0 ).squeeze()
    for batch_size in [1,2,4,8,16,32,64]:
      try:
        tf_start_time = time.time()
        prediction = model.predict(x_to_predict, batch_size=batch_size)
        tf_end_time = time.time()
      except:
        drawImages_alt, drawLabels_alt = load_everything(model_name)
        x_to_predict_alt = np.stack( drawImages_alt, axis=0 ).squeeze()
        tf_start_time = time.time()
        prediction = model.predict(x_to_predict_alt, batch_size=batch_size)
        tf_end_time = time.time()

      model_results_dict[str(batch_size)]=(tf_end_time - tf_start_time) * 1000
      print(model_name +' '+ str(batch_size)+' '+str((tf_end_time - tf_start_time) * 1000)) 
    print(model_results_dict)
    df = df.append(model_results_dict, ignore_index = True)

    
    df.to_csv('batch_comparison_results_' + str(imageCount)+'.csv')
