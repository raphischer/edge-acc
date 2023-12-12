

import argparse
import json
import os

PROPERTIES = {
    'meta': {
        'task': lambda log: 'infer',#lambda log: log['directory_name'].split('_')[0],
        'dataset': lambda log: log['validation_results']['dataset'],
        'model': lambda log: log['config']['modelname'],
        'backend': lambda log: log['validation_results']['backend'], #architecure/software +CORAL statt backend
        'software': lambda log: extract_software(log),
        'architecture': lambda log: extract_architecture(log),
        'number_of_operations':lambda log: extract_edgetpu_compiler_data(log)[0]  if log['validation_results']['backend'] == 'tflite_edgetpu' else '',
        'number_of_unmapped_operations':lambda log: extract_edgetpu_compiler_data(log)[1] if log['validation_results']['backend'] == 'tflite_edgetpu' else '',
        'input_shape': lambda log: extract_model_meta_data(log)[0],
        'total_parameters':lambda log: extract_model_meta_data(log)[1],
        'trainable_parameters':lambda log: extract_model_meta_data(log)[2],
        'non_trainable_parameters': lambda log: extract_model_meta_data(log)[3],
        'running_time': lambda log: log['emissions']['duration']['0'] / log['validation_results']['validation_size'],
        'power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6 / log['validation_results']['validation_size'],
        'validation_size':lambda log: log['validation_results']['validation_size'],
        'batch_size':lambda log: log['validation_results']['batch_size']  
        

    },

    
    'infer_classification': {
        'accuracy_k1': lambda log: log['validation_results']['accuracy_k1'],
        'accuracy_k3': lambda log: log['validation_results']['accuracy_k3'],
        'accuracy_k5': lambda log: log['validation_results']['accuracy_k5'],
        'accuracy_k10': lambda log: log['validation_results']['accuracy_k10']

    },
    'infer_segmentation': {
        'precision_B': lambda log:log['validation_results']['precision_B'],
        'recall_B': lambda log:log['validation_results']['recall_B'],
        'mAP50_B': lambda log:log['validation_results']['mAP50_B'],
        'mAP50_95_B': lambda log:log['validation_results']['mAP50_95_B'],
        'precision_M': lambda log:log['validation_results']['precision_M'],
        'recall_M': lambda log:log['validation_results']['recall_M'],
        'mAP50_M': lambda log:log['validation_results']['mAP50_M'],
        'mAP50_95_M': lambda log:log['validation_results']['mAP50_95_M']
          
    }
}

def extract_model_meta_data(log):

    with open(os.path.join(os.getcwd(),'helper_scripts','model_summaries.json'), 'r') as meta:
        meta_dict =  json.load(meta)
        inputShape = meta_dict[log['config']['modelname']]["Input_Shape"]
        total_parameters = meta_dict[log['config']['modelname']]["Total params"]
        trainable_parameters = meta_dict[log['config']['modelname']]["Trainable params"]
        non_trainable_parameters = meta_dict[log['config']['modelname']]["Non-trainable params"]
    return inputShape, total_parameters, trainable_parameters, non_trainable_parameters

def extract_edgetpu_compiler_data(log):
   
    with open(os.path.join(os.getcwd(),'helper_scripts','edgetpu_compiler_summaries.json'), 'r') as meta:
        meta_dict = json.load(meta)
        operationSum = meta_dict[log['config']['modelname']]['sum_of_operations']
        unmappedSum = meta_dict[log['config']['modelname']]['unmapped_operations']
    return operationSum, unmappedSum 

def extract_architecture(log):
  
    with open(os.path.join(os.getcwd(),'helper_scripts','meta_environment.json'), 'r') as meta:
        processor_shortforms = json.load(meta)['processor_shortforms']
    if 'GPU' in log['execution_platform']:
        n_gpus = len(log['execution_platform']['GPU'])
        gpu_name = processor_shortforms[log['execution_platform']['GPU']['0']['Name']]
        name = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
    else:
        name = processor_shortforms[log['execution_platform']['Processor']]
    #if log['config']['backend'] == "tflite_edgetpu":
        #name = name + ' + GOOGLE CORAL USB ACCELERATOR'
  
    return name


def extract_software(log):
    backend_name = log['config']['backend']
    with open(os.path.join(os.getcwd(),'helper_scripts','meta_environment.json'), 'r')  as meta:
        ml_backends = json.load(meta)['ml_backends']
        good_name = ml_backends[backend_name]['Good_Name'][0]

    #     backend_name = log['config']['backend']
    
    # backend_meta = ml_backends[backend_name]
    # backend_version = 'n.a.'
    # for package in backend_meta["Packages"]:
    #     for req in log['requirements']:
    #         if req.split('==')[0].replace('-', '_') == package.replace('-', '_'):
    #             backend_version = req.split('==')[1]
    #             break
    #     else:
    #         continue
    #     break
    #return f'{backend_name} {backend_version}'
    return good_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", default="mnt_data/results_merged/train_2023_02_27_16_31_23.json")
    args = parser.parse_args()

    with open(args.logfile, 'r') as f:
        log = json.load(f)
    
    print(args.logfile)
    for task, metrics in PROPERTIES.items():
        if task == 'meta' or log['directory_name'].startswith(task):
            for metric_key, func in metrics.items():
                print(f'{task:<10} - {metric_key:<30} - {func(log)}')
