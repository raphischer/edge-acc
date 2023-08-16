#!/bin/bash

for n in 'yolov8n-seg' 'yolov8m-seg' 'yolov8s-seg'

do
    for b in   'tf_cpu' 'tflite_edgetpu' # 'tf_gpu'
    do
        for d in  'COCO'  'COCO128'
         do
        python3 pycoral_segmentation.py -b $b  -mn $n -d $d -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/segmentation_5000_16_08_2023
        echo $n $b
        done
    done
done    