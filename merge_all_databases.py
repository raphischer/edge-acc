# %%
import pickle
import numpy as np
import os
import plotly
import pandas as pd

def getFamilyName(model_name):
    if 'densenet' in model_name.lower():
        model_family='DenseNet'
    elif 'inceptionresnet' in model_name.lower():
        model_family='InceptionResNet'
    elif 'efficientnet' in model_name.lower():
        model_family='EfficientNet'
    elif 'inception' in model_name.lower():
        model_family='Inception'
    elif 'mobilenet' in model_name.lower():
        model_family =  'MobileNet'
    elif 'nasnet' in model_name.lower():
        model_family =  'NASNet'
    elif 'resnet' in model_name.lower():
        model_family =  'ResNet'
    elif 'vgg' in model_name.lower():
        model_family = 'VGG'
    elif 'xception' in  model_name.lower():
        model_family = 'Xception'
    elif 'yolo' in model_name.lower():
        model_family = 'YOLO8'
    else :
        model_family = ''
    return model_family

def approx_USB_power_draw(row):
    if row['backend'] == 'TPU':
        return row['power_draw'] + 1.15 * (row['running_time']) # assume constant 4.5 watt
    elif row['backend'] == 'NCS':
        return row['power_draw'] + 1 * (row['running_time']) # assume constant 4.5 watt
    else:
        print('exception)')
        return row['power_draw'] 
    
def raspi_power_draw(row):
    if row['architecture'] == 'RasPi' and (row['backend'] == 'TPU' or row['backend'] == 'NCS'):
        print('hit')
        return row['running_time'] * 15 # assume constant 4.5 watt
    elif row['architecture'] == 'RasPi':
        print('hit')
        return row['running_time'] * 11
    else:
        return row['power_draw'] 

# CHOOSE ALL DATAFRAMES YOU WANT TO INCLUDE FOR THE FINAL RESULTS:
with open(os.getcwd()+'/result_databases/workstation_classification_final_18_12_2023.pkl', 'rb') as f:
    tmp_data = pickle.load(f)
    tmp_data.drop(tmp_data.index[(tmp_data['architecture'] == 'Workstation') & (tmp_data['batch_size']==32)],inplace=True)
    data = tmp_data
with open(os.getcwd()+'/result_databases/raspi_classification.pkl', 'rb') as f:
    raspi_data =  pickle.load(f) 
    data = data.append(raspi_data)
with open(os.getcwd()+'/result_databases/workstation_segmentation.pkl', 'rb') as f:
    data = data.append( pickle.load(f) )
with open(os.getcwd()+'/result_databases/laptop_class_1.pkl', 'rb') as f:
    data = data.append( pickle.load(f).assign(architecture='Intel(R) Core(TM) i7-10610U') )
with open(os.getcwd()+'/result_databases/laptop_segm_1.pkl', 'rb') as f:
    data = data.append( pickle.load(f).assign(architecture='Intel(R) Core(TM) i7-10610U') )
with open(os.getcwd()+'/result_databases/laptop_segm_2.pkl', 'rb') as f:
    data = data.append( pickle.load(f).assign(architecture='Intel(R) Core(TM) i7-10610U') )





#Add Family of Architecture
data['family']= data['model'].apply(lambda a : getFamilyName(a))



easy_dict = {'Core i7-6700':'Desktop' ,'ARM Cortex-A72':'RasPi' ,'Intel(R) Core(TM) i7-10610U':'Laptop','tf_cpu':'CPU','tflite_edgetpu':'TPU','NCS2':'NCS','tflite':'CPU'}
color_dict = {'Workstation':{'TPU':'#c45f00','NCS':'#5494DA','':'#b10000','CPU':'#b10000'},'Laptop':{'TPU':'#ff6f00','NCS':'#86CEFA','CPU':'#ff0000','TFLITE':'#ff0000'},'RasPi':{'TPU':'#633a00','NCS':'#003396','CPU':'#300000'}}
fsize_dict = {'DenseNet121': 33.226576, 'DenseNet169': 58.593032, 'DenseNet201': 82.584656, 'EfficientNetB0': 21.856559999999998, 'EfficientNetB1': 32.186136, 'EfficientNetB2': 37.468056, 'EfficientNetB3': 50.136184, 'EfficientNetB4': 78.916208, 'EfficientNetB5': 123.524456, 'EfficientNetB6': 174.53435199999998, 'EfficientNetB7': 268.416968, 'InceptionResNetV2': 225.206864, 'InceptionV3': 108.98181699999999, 'MobileNetV2': 14.552912, 'MobileNetV3Large': 22.7354, 'MobileNetV3Small': 10.804839999999999, 'NASNetMobile': 23.037831999999998, 'QuickNet': 53.162167999999994, 'QuickNetLarge': 93.740032, 'QuickNetSmall': 50.845368, 'RegNetX32GF': 432.05466099999995, 'RegNetX400MF': 22.289129, 'RegNetX8GF': 158.813061, 'ResNet101': 179.68035999999998, 'ResNet152': 242.954816, 'ResNet50': 102.985688, 'ResNext101': 356.173021, 'ResNext50': 100.488385, 'VGG16': 553.491728, 'VGG19': 574.7381439999999, 'Xception': 91.973528,'yolov8s-seg':22544384,'yolov8n-seg':6501171.2,'yolov8m-seg':43725619.2}
# Make Backend and Architecture naming unique.
data['architecture'] = data['architecture'].map(easy_dict)
data['backend'] = data['backend'].map(easy_dict)

data['fsize'] = data['model'].map(fsize_dict)
# Adjust Power draw according to RasPi Wattage

data['power_draw'] = data.apply(lambda row: raspi_power_draw(row), axis = 1)

# Adjust Power draw for USB-accelerated runs
data['approx_USB_power_draw'] = data.apply(lambda row: approx_USB_power_draw(row), axis = 1)

# add missing values that are equal across Models
for index, row in data.iterrows():
    current_model = row['model']
    if data.loc[data.model == current_model,'number_of_operations'].isnull:
        data.loc[data.model == current_model,'number_of_operations']=row['number_of_operations'] 
    data.loc[data.model == current_model,'number_of_unmapped_operations']=row['number_of_unmapped_operations'] 
    data.loc[data.model == current_model,'input_shape']=row['input_shape'] 
    data.loc[data.model == current_model,'total_parameters']=row['total_parameters'] 
    data.loc[data.model == current_model,'trainable_parameters']=row['trainable_parameters'] 
    data.loc[data.model == current_model,'non_trainable_parameters']=row['non_trainable_parameters'] 

data["number_of_operations"]=pd.to_numeric(data["number_of_operations"])
data["number_of_unmapped_operations"]=pd.to_numeric(data["number_of_unmapped_operations"])


data["total_parameters"]=pd.to_numeric(data["total_parameters"])

data["trainable_parameters"]=pd.to_numeric(data["trainable_parameters"])
data["non_trainable_parameters"]=pd.to_numeric(data["non_trainable_parameters"])
data["power_draw"] = 0 #data["power_draw"]
data["approx_USB_power_draw"] =  data["approx_USB_power_draw"]



classification_data = data[data['log_name'].str.contains("classification")].reset_index()
segmentation_data = data[data['log_name'].str.contains("segmentation")].reset_index()
classification_data_grouped = classification_data.groupby(['model','backend','architecture']).mean().reset_index()
classification_data_grouped=classification_data_grouped[['model','backend','architecture','accuracy_k1']]
classification_data_grouped.dropna(subset=['accuracy_k1'], inplace=True)

segmentation_data = data[data['log_name'].str.contains("segmentation")].reset_index()
segmentation_data_grouped = segmentation_data.groupby(['model','backend','architecture']).mean().reset_index()
segmentation_data_grouped=segmentation_data_grouped[['model','backend','architecture','precision_B']]
segmentation_data_grouped.dropna(subset=['precision_B'], inplace=True)

   


export_class_df = classification_data.drop(['index','precision_B', 'recall_B', 'mAP50_B','mAP50_95_B', 'precision_M', 'recall_M', 'mAP50_M', 'mAP50_95_M','family'],axis=1)
export_class_df = export_class_df.groupby(['model','backend','architecture']).mean().reset_index()
export_class_df['task'] = 'infer'
export_class_df['dataset'] = 'imagenet'
export_class_df['environment'] = export_class_df['architecture'] +' ' +export_class_df['backend']
export_class_df.to_csv(os.path.join(os.getcwd(),'classification_database/database_HIGH.csv'))

export_class_df.to_pickle(os.path.join(os.getcwd(),'classification_database/database.pkl')) 
export_seg_df = segmentation_data.drop(['index','accuracy_k1', 'accuracy_k3',
       'accuracy_k5', 'accuracy_k10','family'],axis=1)

export_seg_df = export_seg_df.groupby(['model','backend','architecture']).mean().reset_index()
export_seg_df['task'] = 'infer'
export_seg_df['dataset'] = 'coco'
export_seg_df['environment'] = export_seg_df['architecture'] + ' ' + export_seg_df['backend']

export_seg_df.to_pickle(os.path.join(os.getcwd(),'segmentation_database/database.pkl'))


seg_pt_power_draw = pd.pivot_table(export_seg_df, values = 'approx_USB_power_draw',index = 'environment', columns = 'model')

