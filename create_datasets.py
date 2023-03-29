import os 
import numpy as np
def createDataset(model_name):
    imageDataDir =  os.path.join(os.getcwd(),'mnt_data/staay/image_data/',model_name)
    
    if not os.path.exists(imageDataDir):
        os.makedirs(imageDataDir)
    from load_models import load_preprocessing
    from load_data import load_data
    preprocess = load_preprocessing(model_name)
    ds, _ = load_data(preprocess=preprocess, n_batches= 3200, batch_size=1)
    y = np.concatenate([y for x, y in ds], axis=0)
    x = np.concatenate([x for x, y in ds], axis=0)

    for i in range(0,3200):
        labeldir = os.path.join(imageDataDir,str(y[i]))
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)
        print(x[i].shape)
        np.save(os.path.join(labeldir,str(i)+'.npy'),np.asarray(x[i]))
   
   



MODELS = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet152', 'ResNet50', 'ResNet101V2', 'ResNet152V2', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'MobileNetV3Large', 'MobileNetV3Small']
for model in MODELS:
    createDataset(model)