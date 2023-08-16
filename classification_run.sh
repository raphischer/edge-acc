#!/bin/bash

#for n in 'DenseNet121' 'DenseNet169'  'DenseNet201'  'EfficientNetB0'  'EfficientNetB1'  'EfficientNetB2'  'EfficientNetB3'  'EfficientNetB4'  'EfficientNetB5'  'EfficientNetB6'  'EfficientNetB7'  'EfficientNetV2B0'  'EfficientNetV2B1'  'EfficientNetV2B2'  'EfficientNetV2B3'  'EfficientNetV2L'  'EfficientNetV2M'  'EfficientNetV2S'  'InceptionResNetV2'  'InceptionV3'  'MobileNet'  'MobileNetV2'  'NASNetLarge'  'NASNetMobile'  'ResNet101'  'ResNet152'  'ResNet50'  'ResNet101V2'  'ResNet152V2'  'ResNet50V2'  'VGG16'  'VGG19'  'Xception'  'MobileNetV3Large'  'MobileNetV3Small'

for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'VGG16' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do
    for b in  'tf_cpu' 'tf_gpu' 'tflite_edgetpu'
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/eval1200_16_08_2023_pyrapl_1
    done
done

for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'VGG16' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do
    for b in  'tf_cpu' 'tf_gpu' 'tflite_edgetpu'
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/eval1200_16_08_2023_pyrapl_2
    done
done

for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'VGG16' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do
    for b in  'tf_cpu' 'tf_gpu' 'tflite_edgetpu'
    do
    echo $n $b
    python3 pycoral_classification.py -b $b  -mn $n -ic 12000 -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/eval1200_16_08_2023_pyrapl_3
    done
done