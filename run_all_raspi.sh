#!/bin/bash

#ALL 'DenseNet121' 'DenseNet169'  'DenseNet201'  'EfficientNetB0'  'EfficientNetB1'  'EfficientNetB2'  'EfficientNetB3'  'EfficientNetB4'  'EfficientNetB5'  'EfficientNetB6'  'EfficientNetB7'  'EfficientNetV2B0'  'EfficientNetV2B1'  'EfficientNetV2B2'  'EfficientNetV2B3'  'EfficientNetV2L'  'EfficientNetV2M'  'EfficientNetV2S'  'InceptionResNetV2'  'InceptionV3'  'MobileNet'  'MobileNetV2'  'NASNetLarge'  'NASNetMobile'  'ResNet101'  'ResNet152'  'ResNet50'  'ResNet101V2'  'ResNet152V2'  'ResNet50V2'  'VGG16'  'VGG19'  'Xception'  'MobileNetV3Large'  'MobileNetV3Small'
source ./l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64/setupvars.sh
monitoring_dir = './mnt_data/staay/workstation_final'

# Nets running with all forms of acceleration
for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'Xception' 'DenseNet121' 'DenseNet201' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do 
    echo $n
    python3 pycoral_classification_raspi.py -b tf_cpu -mn $n -ic 12000 -md  $monitoring_dir -bz 16
    python3 pycoral_classification_raspi.py -b NCS2  -mn $n -ic 12000 -md $monitoring_dir
    python3 pycoral_classification_raspi.py -b tflite_edgetpu  -mn $n -ic 12000 -md  $monitoring_dir 

done
# ONLY NCS and TensorFlow2

for n in 'ResNet152' 'ResNet152V2' 'VGG16' 'VGG19' 'MobileNetV3Small' 'MobileNetV3Large' 
do
    echo $n
    python3 pycoral_classification_raspi.py -b tf_cpu  -mn $n -ic 12000 -md  $monitoring_dir -bz 16
    python3 pycoral_classification_raspi.py -b NCS2  -mn $n -ic 12000 -md  $monitoring_dir
done

# EfficientNetB0
python3 pycoral_classification_raspi.py -b tf_cpu  -mn EfficientNetB0 -ic 12000 -md  $monitoring_dir -bz 16
python3 pycoral_classification_raspi.py -b tflite_edgetpu  -mn EfficientNetB0 -ic 12000 -md  $monitoring_dir -bz 16
