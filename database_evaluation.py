import pickle
import numpy as np
import os


def createColorSequence(df):
    colors = []
    for i in df['backend']:
        if i == 'tflite_edgetpu':
            colors.append('orange')
        elif i == 'tf_cpu':
            colors.append('red')
        else: # GPU case
            colors.append('blue')
    return colors


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
    elif 'yolo8' in model_name.lower():
        model_family = 'YOLO8'
    else :
        model_family = ''
    return model_family

def approx_USB_power_draw(row):
    if row['backend'] == 'tflite_edgetpu':
        return row['power_draw'] + 25 * (row['running_time']/3600) # assume constant 25 watt
    else:
        return row['power_draw'] 


with open(os.getcwd()+'/result_databases/lamarrws01_clean_classification_lauf_4.pkl', 'rb') as f:
    data = pickle.load(f)
with open(os.getcwd()+'/result_databases/lamarrws01_clean_classification_lauf_3.pkl', 'rb') as f:
    data = data.append( pickle.load(f) )
with open(os.getcwd()+'/result_databases/lamarrws01_clean_classification_lauf_2.pkl', 'rb') as f:
    data = data.append( pickle.load(f) )

data = data.groupby(['model','backend']).mean().reset_index()

#Add Family of 
data['family']= data['model'].apply(lambda a : getFamilyName(a))
data['approx_USB_power_draw'] = data.apply(lambda row: approx_USB_power_draw(row), axis = 1)
print(data)

analyze_running_time = data[['model','backend','running_time']].dropna()#data[['model','backend','total_parameters', 'running_time','power_draw','accuracy','number_of_unmapped_operations']]
analyze_running_time
analyze_running_time = analyze_running_time.pivot_table(index='model',columns = 'backend', values='running_time', aggfunc = 'mean')#set_index('model').wide_to_long()
print('EDGETPU :' )
print(np.mean(analyze_running_time['tflite_edgetpu']))
print('CPU')
print(np.mean(analyze_running_time['tf_cpu']))
print('GPU')
print(np.mean(analyze_running_time['tf_gpu']))