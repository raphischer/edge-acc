# imagenet-on-the-edge
Nets compared for Imagenet-Classification Task:
'ResNet50' 'DenseNet169'  'ResNet101V2' 'ResNet50V2' 'VGG16' 'Xception'  'DenseNet169'  'DenseNet121' 'DenseNet201' 'EfficientNetB0' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'NASNetMobile' 'ResNet101' 
Nets compared for Imagenet-Segmentation Task:
'yolov8s', 'yolov8n', 'yolov8m'


## NCS Setup
- Follow guide: https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_from_archive_linux.html
- Step 2 (environment config) needs to be performed in each session!
- udev Regeln angepasst werden und dein Nutzer muss der users Gruppe hinzugefigt werden. https://docs.openvino.ai/2022.3/openvino_docs_install_guides_configurations_for_ncs2.html#ncs-guide
- pip install openvino


## ValueError: Failed to load delegate from libedgetpu.so.1
- de- and reconnect edgeTPU
- try different USB cable!