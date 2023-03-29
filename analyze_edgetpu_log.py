import os
import json
def analyze_log(model_name):
    file_path = os.path.join(os.getcwd(),'mnt_data/staay/models/edgetpu_models/'+model_name+'_edgetpu.log')
    sum_of_operations = 0
    sum_of_unmapped_operations = 0
    with open(file_path) as f:
        contents = f.readlines()[6:]
        for line in contents:
            sum_of_operations = sum_of_operations + int(line.split()[1])
            if line.split()[2] != 'Mapped':
                sum_of_unmapped_operations = sum_of_unmapped_operations + int(line.split()[1])
    return sum_of_operations, sum_of_unmapped_operations

MODELS = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet152', 'ResNet50', 'ResNet101V2', 'ResNet152V2', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'MobileNetV3Large', 'MobileNetV3Small']
modelsdict = {}
for model in MODELS:
    try:
        totalsum, unmapped = analyze_log(model)
    except:
        print("model logs for "+ model+" not found.")
    modelsdict[model]= {'sum_of_operations':totalsum, 'unmapped_operations':unmapped}
with open('edgetpu_compiler_summaries.json', 'w') as f:
    json.dump(modelsdict, f)
     
    # f.write('{')
    # for model_name in MODELS[:-1]:
    #     try:
    #         totalsum, unmapped = analyze_log(model_name)
    #         f.write('"'+model_name+'": { \n')
    #         f.write('"sum_of_operations": '+str(totalsum)+', \n')
    #         f.write('"unmapped_operations": '+str(unmapped)+' \n')
    #         f.write('},')
    #     except:
    #         print("couldn't find log for "+model_name)
    #     # last Model in the list
    # model_name = MODELS[-1]
    # totalsum, unmapped = analyze_log(model_name)
    # f.write('"'+model_name+'": { \n')
    # f.write('"sum_of_operations": '+str(totalsum)+', \n')
    # f.write('"unmapped_operations": '+str(unmapped)+' \n')
    # f.write('}')
    # f.write('}')