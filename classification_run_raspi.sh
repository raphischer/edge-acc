#!/bin/bash
#source ./l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64/setupvars.sh

#for n in 'DenseNet121' 'DenseNet169'  'DenseNet201'  'EfficientNetB0'  'EfficientNetB1'  'EfficientNetB2'  'EfficientNetB3'  'EfficientNetB4'  'EfficientNetB5'  'EfficientNetB6'  'EfficientNetB7'  'EfficientNetV2B0'  'EfficientNetV2B1'  'EfficientNetV2B2'  'EfficientNetV2B3'  'EfficientNetV2L'  'EfficientNetV2M'  'EfficientNetV2S'  'InceptionResNetV2'  'InceptionV3'  'MobileNet'  'MobileNetV2'  'NASNetLarge'  'NASNetMobile'  'ResNet101'  'ResNet152'  'ResNet50'  'ResNet101V2'  'ResNet152V2'  'ResNet50V2'  'VGG16'  'VGG19'  'Xception'  'MobileNetV3Large'  'MobileNetV3Small'

for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'VGG16' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do
    for b in   'tf_cpu' # 'NCS2' 'tflite_edgetpu'
    do
    echo $n $b
    python3 pycoral_classification_raspi.py -b $b  -mn $n -ic 12000 -md  ~/Git/imagenet-on-the-edge/mnt_data/staay/final_raspi
    done
done

for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'VGG16' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do
    for b in 'tf_cpu' #'NCS2' 'tflite_edgetpu'
    do
    echo $n $b
    python3 pycoral_classification_raspi.py -b $b  -mn $n -ic 12000 -md  ~/Git/imagenet-on-the-edge/mnt_data/staay/final_raspi
    done
done

for n in 'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'VGG16' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
do
    for b in  'tf_cpu' #'NCS2' 'tflite_edgetpu'
    do
    echo $n $b
    python3 pycoral_classification_raspi.py -b $b  -mn $n -ic 12000 -md  ~/Git/imagenet-on-the-edge/mnt_data/staay/final_raspi
    done
done

python3 helper_scripts/raspi_merge.py -dir /mnt_data/staay/final_raspi