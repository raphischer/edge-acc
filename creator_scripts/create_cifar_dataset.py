from tensorflow import keras
import tensorflow as tf
import os
import numpy as np



def createDataset(model_name,x,y):
    print('start '+model_name)
    labelMap10 = {0:'airplane', 1:'automobile', 2:'bird',3:'cat', 4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}


    imageDir = os.path.join(os.getcwd(),'mnt_data/staay/cifar10_data')
    imageDataDir =  os.path.join(imageDir,model_name)

    
    print('start preprocessing '+model_name)
    from helper_scripts.load_models import load_preprocessing
    if model_name != 'None':
        preprocess = load_preprocessing(model_name)
    else:
        preprocess = None

   
    if preprocess is not None:
        for i in range(0,len(x)):
            x[i] = preprocess(x[i],y[i])[0]


    print('finished preprocessing '+model_name)

    #for i in range(0,len(x)):
    for i in range(0,3200):
        labeldir = os.path.join(imageDataDir,str(labelMap10[y[i][0]]))
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)
        arrayToSave = np.asarray(x[i][None,:,:,:]).astype('float32')

        np.save(os.path.join(labeldir,str(i)+'.npy'),arrayToSave)
        if i % 100 == 0:
            print(arrayToSave.shape)
    print('finished saving '+model_name)

#MODELS = ['None', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet152', 'ResNet50', 'ResNet101V2', 'ResNet152V2', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'MobileNetV3Large', 'MobileNetV3Small']
MODELS = [ 'DenseNet121','None']#, 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'MobileNet', 'MobileNetV2', 'NASNetMobile', 'ResNet101', 'ResNet50', 'ResNet101V2', 'ResNet50V2', 'VGG16','MobileNetV3Small']
#MODELS = ['None']
print('load dataset from keras') 
(x_train10, y_train10), (x_test10, y_test10) = keras.datasets.cifar10.load_data()
print('loaded dataset') 
x = list(x_train10) + list(x_test10)
y = list(y_train10) + list(y_test10)
for i in range(len(x)):
    x[i] = tf.image.resize(x[i],(224,224))
print('finished resizing :)')
    
for model in MODELS:
    createDataset(model,x,y)
