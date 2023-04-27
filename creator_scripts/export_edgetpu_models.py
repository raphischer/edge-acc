import os
import time
models_dir='/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/models'
tflite_models_dir = os.path.join(models_dir, 'tflite_models')
tflite_edgetpu_models_dir = os.path.join(models_dir, 'edgetpu_models')


for model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'MobileNet', 'MobileNetV2', 'NASNetLarge',  'MobileNetV3Large', 'MobileNetV3Small']:
    print(model_name)
    command = "edgetpu_compiler -s -d -a "+str(os.path.join(tflite_models_dir,model_name+'.tflite'))+" -o "+ str(tflite_edgetpu_models_dir)
    returned_value = os.system(command)
    time.sleep(10)
