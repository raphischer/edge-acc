# Code for *Stress-Testing USB Accelerators for Efficient Edge Inference*

![Accelerator Experiment Setup](other/pic_wide.jpg?raw=true "Title")

Our corresponding research paper is currently under review and available as a preprint at https://www.researchsquare.com/article/rs-3793927/latest

## How to explore our experimental results interactively

To view our experimental results, make sure you have the the following required python packages installed:
```numpy, pandas, pint, scipy, dash, dash_bootstrap_components, Pillow, reportlab, fitz, frontend, plotly```

Then start our app via ```main.py``` and open the [webpage](http://localhost:8888/). Besides the interactive plots, you can also inspect the PDF files in our ```paper_results``` directory. 

## Models that we considered:
**Models compared for Imagenet Classification:**

'DenseNet121' 'DenseNet169'  'DenseNet201'  'EfficientNetB0'  'EfficientNetB1'  'EfficientNetB2'  'EfficientNetB3'  'EfficientNetB4'  'EfficientNetB5'  'EfficientNetB6'  'EfficientNetB7'  'EfficientNetV2B0'  'EfficientNetV2B1'  'EfficientNetV2B2'  'EfficientNetV2B3'  'EfficientNetV2L'  'EfficientNetV2M'  'EfficientNetV2S'  'InceptionResNetV2'  'InceptionV3'  'MobileNet'  'MobileNetV2'  'NASNetLarge'  'NASNetMobile'  'ResNet101'  'ResNet152'  'ResNet50'  'ResNet101V2'  'ResNet152V2'  'ResNet50V2'  'VGG16'  'VGG19'  'Xception'  'MobileNetV3Large'  'MobileNetV3Small'

**Models compared for Imagenet Segmentation:**

'yolov8s', 'yolov8n', 'yolov8m', 'yolov8x', 'yolov8l'

## Repository structure:

### Directories
- ```batchsize_comparison_results``` contains the tables that we used to compare the performance of different batch sizes on our three host systems. 
- ```classification_database``` and ```segmentation_database``` contain the results and configurations of our experimental results. These two directories are accessed by ```main.py``` to create the paper results.
- ```creator_scripts``` contain the python scripts used to create the datasets used for the experiment as well as the model files for the different model formats required by the accelerators TPU and NCS.
- ```helper_scripts``` contain miscelaneous scripts that are used by other scripts or that can be used to clean up experimental result files. Model metadata that we use for analysis is collected with the corresponding scripts. Our batch size comparison that determines the batch size of used in our CPU experiments is included here. 
- ```paper_results``` contain our result graphs as PDF files.
- ```result_databases``` contain the pickled pandas dataframes created by ```load_experiment_logs.py``` that are further merged with the ```merge_all_databases.py``` script.
- ```strep``` contains scripts used to create our interactive model results.
- Make sure to **include a directory that holds the model input data as well as all model files** using the scripts from ```creator_scripts```.

### Scripts
- ```main.py```can be executed to create the interactive model results based on the content of ```classification_database``` and ```segmentation_database```. It uses the ```paper_results.py``` script where the graph specifications are coded. 
-  ```load_experiment_logs.py``` merges an experiments monitoring directory into a pandas dataframe and then pickles it for further use.
- ```merge_all_databases.py``` merges all of the chosen dataframes form ```result_databases``` into the ```classification_database``` and ```segmentation_database``` directories for the final results. 
- ```pycoral_classification.py```, ```pycoral_segmentation.py``` and ```pycoral_classification_raspi.py``` are the scripts that run the actual experiments. They are run by the ```run_all``` scripts.
- The ```run_all``` scripts define one run of an experiment depending on the used environments. We ran these expermients multimple times for our paper results. 

## How to setup your own experimental runs:

### Google Coral EdgeTPU Setup
- Follow guide: https://coral.ai/docs/accelerator/get-started/
- Setup EdgeTPU Compiler for TPU Conversion: https://coral.ai/docs/edgetpu/compiler/
- Setup pycoral: https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu

### Intel NCS Setup
- Follow guide: https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_from_archive_linux.html
- Step 2 (environment config) needs to be performed in each session!
- udev Regeln angepasst werden und dein Nutzer muss der users Gruppe hinzugefigt werden. https://docs.openvino.ai/2022.3/openvino_docs_install_guides_configurations_for_ncs2.html#ncs-guide
- pip install openvino

### 1. Setup Directories and Download Raw Data
- Make sure you **created your model and data directory** in our scripts, we use the directory name ```mnt_data```. 
- In ```mnt_data```, we have the subfolders ```unpacked``` and ```staay```. In ```unpacked```, we downloaded the 1% imagenet database. (https://www.image-net.org/)
- In ```staay```, we have the ```imagenet_data``` directory where our preprocessed data is saved as well as the ```models``` directory where the models are saved in the corresponding subdirectories ```edgetpu_models,openVINO,saved_models,tflite_models```. These directories get filled by using the scripts in the ```creator_scripts``` directory.
- Also in ```staay``` the files ```coco.yaml``` and ```coco128-seg.yaml``` should be included for segmentation with YOLO models. (Follow : https://docs.ultralytics.com/datasets/detect/coco/#applications)
- If you choose to change the naming of your files, make sure to adjust all occurences of ```mnt_data``` in this repositories scripts. 

### 2. Fill Directories using Creator Scripts

- Using the ```create_imagenet_dataset.py``` script, create the preprocessed imagenet data of the models that you want to compare. You may choose from any of the Tensorflow2 models listed in this readme. It saves the model-individualized preprocessed datasets as numpy arrays in the ```imagenet_data``` file.
- Use the ```export_tflite_models.py``` scripts to export the models into Tensorflow Lite models. These can then converted into TPU-compatible models using the ```export_edgetpu_models.py``` script. The latter script uses the EdgeTPU Compiler.
- Export the saved_models into NCS compatible models using ```export_NCS_models.py```. This uses the model optimizer ```mo``` provided by openVINO2022.3. Make sure to save the created models in the correct directory (```models/openVINO```).
- Execute the ```create_YOLO_models.py``` script to create YOLO models as saved_model, edgeTPU compatible model and openVINO model. Save accordingly.

### 3. Run Experiments

- Now you can execute singular runs with the ```pycoral_``` scripts. Adjust all flags to your liking.
- You can even test all of your created models by using the ```run_all``` scripts adjusting the flags to your liking. 
- This will create a directory with the experiments logging. (Monitoring directory)

### 4. Create database from monitoring directories

- Adjust the path at the end of ```load_experiment_logs.py``` to your monitoring directory and execute the script. This will merge the directory into one dataframe that is saved in ```result_databases```
- In ```merge_all_databases.py```, adjust all of the databases you want to include from ```result_databases```. This will create the final databases in the ```classification_database``` and ```segmentation_database``` directories.
- Now ```main.py``` can be run with the newly included experiment results. 

## Troubleshooting:

### TPU ERROR: `ValueError: Failed to load delegate from libedgetpu.so.1`
- de- and reconnect edgeTPU
- try different USB cable!

Copyright (c) 2023 Raphael Fischer, Alexander van der Staay, Sebastian Buschjäger