from helper_scripts.load_models import prepare_model
MODELS = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet152', 'ResNet50', 'ResNet101V2', 'ResNet152V2', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'MobileNetV3Large', 'MobileNetV3Small']
with open('helper_scripts/model_summaries.json', 'w') as f:
    f.write('{')
    for model_name in MODELS[:-1]:
        model = prepare_model(model_name)
        f.write('"'+model_name+'": {')
        f.write('"Input_Shape": '+ '"'+str(model.get_config()["layers"][0]["config"]["batch_input_shape"])+'", \n')
        model.summary(print_fn = lambda x: f.write('"'+ x.split(': ')[0] + '": '+x.split(': ')[1].replace(',','')+''+','+'\n') if x.startswith('Total params:') or x.startswith('Trainable params:') else "")
        model.summary(print_fn = lambda x: f.write('"'+ x.split(': ')[0] + '": '+x.split(': ')[1].replace(',','')+''+'\n') if x.startswith('Non-trainable params:') else "") # no comma after last params entry
        f.write('},')
    # last Model in the list
    model_name = MODELS[-1]
    model = prepare_model(model_name)
    f.write('"'+model_name+'": {')
    f.write('"Input_Shape": '+ '"'+str(model.get_config()["layers"][0]["config"]["batch_input_shape"])+'", \n')
    model.summary(print_fn = lambda x: f.write('"'+ x.split(': ')[0] + '": '+x.split(': ')[1].replace(',','')+''+','+'\n') if x.startswith('Total params:') or x.startswith('Trainable params:') else "")
    model.summary(print_fn = lambda x: f.write('"'+ x.split(': ')[0] + '": '+x.split(': ')[1].replace(',','')+''+'\n') if x.startswith('Non-trainable params:') else "") # no comma after last params entry
    f.write('}')
    f.write('}')