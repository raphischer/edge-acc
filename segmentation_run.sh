#!/bin/bash

for n in 'yolov8n-seg' 'yolov8m-seg' 'yolov8s-seg'

do
    for b in   'tf_cpu' # 'tflite_edgetpu' 'tf_gpu'
    do
        for d in  'COCO'  #'COCO128'
         do
        python3 pycoral_segmentation.py -b $b  -mn $n -d $d -md  /home/staay/Git/imagenet-on-the-edge/mnt_data/staay/first_segmentation_128_5000
        echo $n $b
        done
    done
done    