#!/bin/bash

#ALL 'DenseNet121' 'DenseNet169'  'DenseNet201'  'EfficientNetB0'  'EfficientNetB1'  'EfficientNetB2'  'EfficientNetB3'  'EfficientNetB4'  'EfficientNetB5'  'EfficientNetB6'  'EfficientNetB7'  'EfficientNetV2B0'  'EfficientNetV2B1'  'EfficientNetV2B2'  'EfficientNetV2B3'  'EfficientNetV2L'  'EfficientNetV2M'  'EfficientNetV2S'  'InceptionResNetV2'  'InceptionV3'  'MobileNet'  'MobileNetV2'  'NASNetLarge'  'NASNetMobile'  'ResNet101'  'ResNet152'  'ResNet50'  'ResNet101V2'  'ResNet152V2'  'ResNet50V2'  'VGG16'  'VGG19'  'Xception'  'MobileNetV3Large'  'MobileNetV3Small'
source ./l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64/setupvars.sh

# Nets running with all forms of acceleration
for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do
    for b in  'tf_cpu' 'NCS2' 'tflite_edgetpu'
    do
    echo $n $b
   python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
    done
done
# ONLY NCS and TensorFlow2

for n in 'NASNetLarge' 'ResNet152' 'ResNet152V2' 'VGG16' 'VGG19' 'MobileNetV3Small' 'MobileNetV3Large' 
do
    for b in  'tf_cpu' 'NCS2' 
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
    done
done
# ONLY TensorFlow 2 is working
for n in 'EfficientNetB1' 'EfficientNetB2' 'EfficientNetB3' 
do
    for b in  'tf_cpu'
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
    done
done



for n in 'yolov8n-seg' 'yolov8m-seg' 'yolov8s-seg'
do
   for b in   'tflite_edgetpu' 'NCS2' 'tf_cpu'
    do
        for d in  'COCO' 
         do
        python3 pycoral_segmentation.py -b $b  -mn $n -d $d -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
       echo $n $b
         done
    done
done    
# 2nc Run (Flipped order)
for n in 'ResNet101'  'NASNetMobile'  'MobileNetV2'  'MobileNet'  'InceptionV3'  'EfficientNetB0' 'DenseNet201' 'DenseNet121'  'DenseNet169'  'Xception'  'ResNet50V2' 'ResNet101V2'  'DenseNet169' 'ResNet50'
do
    for b in  'tf_cpu' 'NCS2' 'tflite_edgetpu'
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
    done
done


for n in 'VGG16' 'VGG19'  'MobileNetV3Small' 'MobileNetV3Large' 'NASNetLarge' 'ResNet152' 'ResNet152V2'
do
    for b in   'tf_cpu' 'NCS2' 
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
    done
done

for n in 'EfficientNetB1' 'EfficientNetB2' 'EfficientNetB3' 
do
    for b in  'tf_cpu'
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
    done
done



for n in 'yolov8s-seg' 'yolov8m-seg' 'yolov8n-seg'

do
    for b in  'tflite_edgetpu' 'NCS2' 'tf_cpu'
    do
        for d in  'COCO' 
         do
        python3 pycoral_segmentation.py -b $b  -mn $n -d $d -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/workstation_final
        echo $n $b
        done
    done
done   