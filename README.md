# How to view our experimental Results in an interactive Console

To view our experimental results, ymake sure you have the the following required python packages installed:
```numpy, pandas, pint, scipy, dash, dash_bootstrap_components, Pillow, reportlab, fitz, frontend, plotly```

Run our script ```main.py```. Interactive plotly graphs will open in your standard browser. PDF files of the corresponding graphs can be found in the ```paper_results``` directory. 

## Models that we considered:
**Models compared for Imagenet-Classification Task:**

'DenseNet121' 'DenseNet169'  'DenseNet201'  'EfficientNetB0'  'EfficientNetB1'  'EfficientNetB2'  'EfficientNetB3'  'EfficientNetB4'  'EfficientNetB5'  'EfficientNetB6'  'EfficientNetB7'  'EfficientNetV2B0'  'EfficientNetV2B1'  'EfficientNetV2B2'  'EfficientNetV2B3'  'EfficientNetV2L'  'EfficientNetV2M'  'EfficientNetV2S'  'InceptionResNetV2'  'InceptionV3'  'MobileNet'  'MobileNetV2'  'NASNetLarge'  'NASNetMobile'  'ResNet101'  'ResNet152'  'ResNet50'  'ResNet101V2'  'ResNet152V2'  'ResNet50V2'  'VGG16'  'VGG19'  'Xception'  'MobileNetV3Large'  'MobileNetV3Small'

**Models compared for Imagenet-Segmentation Task:**

'yolov8s', 'yolov8n', 'yolov8m', 'yolov8x', yolov8l'
# How the repository is structured:
## Directories
- ```batchsize_comparison_results``` contains the tables that we used to compare the performance of different batch sizes on our three host systems. 
- ```classification_database``` and ```segmentation_database``` contain the results and configurations of our experimental results. These two directories are accessed by ```main.py``` to create the paper results.
- ```creator_scripts``` contain the python scripts used to create the datasets used for the experiment as well as the model files for the different model formats required by the accelerators TPU and NCS.
- ```helper_scripts``` contain miscelaneous scripts that are used by other scripts or that can be used to clean up experimental result files.
- ```paper_results``` contain our result graphs as PDF files.
- ```result_databases``` contain the pickled pandas dataframes created by ```load_experiment_logs.py``` that are further merged with the ```merge_all_databases.py``` script.
- ```strep``` contains scripts used to create our interactive model results.
- Make sure to **include a directory that holds the model input data as well as all model files** using the scripts from ```creator_scripts```.

## Scripts
- ```main.py```can be executed to create the interactive model results based on the content of ```classification_database``` and ```segmentation_database```. It uses the ```paper_results.py``` script where the graph specifications are coded. 
-  ```load_experiment_logs.py``` merges an experiments monitoring directory into a pandas dataframe and then pickles it for further use.
- ```merge_all_databases.py``` merges all of the chosen dataframes form ```result_databases``` into the ```classification_database``` and ```segmentation_database``` directories for the final results. 
- ```pycoral_classification.py```, ```pycoral_segmentation.py``` and ```pycoral_classification_raspi.py``` are the scripts that run the actual experiments. They are run by the ```run_all``` scripts.
- The ```run_all``` scripts define one run of an experiment depending on the used environments. We ran these expermients multimple times for our paper results. 

# How to setup your own experimental runs:
## EdgeTPU Setup
- Follow guide: https://coral.ai/docs/accelerator/get-started/
- Setup EdgeTPU Compiler for TPU Conversion: https://coral.ai/docs/edgetpu/compiler/
- Setup pycoral: https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
## NCS Setup
- Follow guide: https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_from_archive_linux.html
- Step 2 (environment config) needs to be performed in each session!
- udev Regeln angepasst werden und dein Nutzer muss der users Gruppe hinzugefigt werden. https://docs.openvino.ai/2022.3/openvino_docs_install_guides_configurations_for_ncs2.html#ncs-guide
- pip install openvino


### ValueError: Failed to load delegate from libedgetpu.so.1
- de- and reconnect edgeTPU
- try different USB cable!