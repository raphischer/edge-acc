# View our experimental Results in an interactive Console




# Models that we considered:
### Nets compared for Imagenet-Classification Task:
'DenseNet121' 'DenseNet169'  'DenseNet201'  'EfficientNetB0'  'EfficientNetB1'  'EfficientNetB2'  'EfficientNetB3'  'EfficientNetB4'  'EfficientNetB5'  'EfficientNetB6'  'EfficientNetB7'  'EfficientNetV2B0'  'EfficientNetV2B1'  'EfficientNetV2B2'  'EfficientNetV2B3'  'EfficientNetV2L'  'EfficientNetV2M'  'EfficientNetV2S'  'InceptionResNetV2'  'InceptionV3'  'MobileNet'  'MobileNetV2'  'NASNetLarge'  'NASNetMobile'  'ResNet101'  'ResNet152'  'ResNet50'  'ResNet101V2'  'ResNet152V2'  'ResNet50V2'  'VGG16'  'VGG19'  'Xception'  'MobileNetV3Large'  'MobileNetV3Small'
## Nets compared for Imagenet-Segmentation Task:
'yolov8s', 'yolov8n', 'yolov8m', 'yolov8x', yolov8l'


## NCS Setup
- Follow guide: https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_from_archive_linux.html
- Step 2 (environment config) needs to be performed in each session!
- udev Regeln angepasst werden und dein Nutzer muss der users Gruppe hinzugefigt werden. https://docs.openvino.ai/2022.3/openvino_docs_install_guides_configurations_for_ncs2.html#ncs-guide
- pip install openvino


## ValueError: Failed to load delegate from libedgetpu.so.1
- de- and reconnect edgeTPU
- try different USB cable!