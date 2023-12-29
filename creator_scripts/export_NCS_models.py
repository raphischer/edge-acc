import os
import time
models_dir=os.getcwd()+'/mnt_data/staay/models'
openVINO_models_dir = os.path.join(models_dir, 'openVINO')
saved_models_dir = os.path.join(models_dir, 'saved_models')
# ADJUST
for model_name in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'MobileNet', 'MobileNetV2', 'NASNetLarge',  'MobileNetV3Large', 'MobileNetV3Small']:
    print(model_name)
    command = " mo  --use_new_frontend --static_shape --saved_model_dir " +saved_models_dir+'/'+model_name
    returned_value = os.system(command)
    time.sleep(20)

