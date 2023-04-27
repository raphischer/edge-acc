import tensorflow as tf
import tensorflow_datasets as tfds
import os
from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils
import tensorflow.keras 


from helper_scripts.load_data import load_data
from helper_scripts.load_models import load_preprocessing, prepare_model#, MODELS


models_dir='/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/models'
saved_models_dir = os.path.join(models_dir, 'saved_models')
tflite_models_dir = os.path.join(models_dir, 'tflite_models')
tflite_edgetpu_models_dir = os.path.join(models_dir, 'edgetpu_models')

#MODELS = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'MobileNet', 'MobileNetV2', 'NASNetLarge',  'MobileNetV3Large', 'MobileNetV3Small']
MODELS = ['EfficientNetB1']
for model_name in MODELS:
  path_to_saved_model = os.path.join(saved_models_dir,model_name)

  if True: #not os.path.isdir(path_to_saved_model):
    # Modell erstellen & Als SavedModel abspeichern
    model = prepare_model(model_name) #keras.engine.functional.Functional' / tf.keras.applications.MobileNetV3Small
    print('MODEL INPUT')
    print(model.input.shape)
    model.input.set_shape((1,) + model.input.shape[1:])
    print('MODEL INPUT')
    print(model.input.shape)
    print('SAVE SavedModel: '+model_name)
    tf.saved_model.save(model, path_to_saved_model)

  if True:
  # Build Representative Dataset that helps infering optimal quantization
    preprocess = load_preprocessing(model_name)
    ds, _ = load_data(preprocess=preprocess, batch_size=1, n_batches=100) #tf.data.Dataset (tensorflow.python.data.ops.dataset_ops.TakeDataset) , tfds.core.DatasetInfo
    #Generator function with PREPROCESSED images
    def representative_dataset_preprocessed():
      for input_value in ds.as_numpy_iterator():
          #dtype float32
          #print(input_value[0].shape)
          yield [input_value[0]]

    ds_no_prep, _ = load_data(preprocess=None, n_batches=10, batch_size=32) #tf.data.Dataset (tensorflow.python.data.ops.dataset_ops.TakeDataset) , tfds.core.DatasetInfo
    #Generator function with non-processed
    def representative_dataset_not_preprocessed():
      for input_value in ds_no_prep.as_numpy_iterator():
          yield [input_value[0]]


    # Load model from SavedModel, perform full integer quantization, convert to TFLite, add metadata, save
    print('LOAD SavedModel: '+model_name)
    saved_model = tf.saved_model.load(path_to_saved_model)
    try:
      converter = tf.lite.TFLiteConverter.from_saved_model(path_to_saved_model)
      # Convert the model to quantized TFLite model.
      converter.optimizations =  [tf.lite.Optimize.DEFAULT]
      converter.representative_dataset = representative_dataset_preprocessed
      converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
      converter.target_spec.supported_types = [tf.int8]
      #converter.inference_type = tf.int8
      converter.inference_input_type = tf.int8  
      converter.inference_output_type = tf.int8  
      #converter.allow_custom_ops = True
      tflite_model = converter.convert()

    # save TFLite model 
      print('SAVE TFLite Model: '+model_name)
      with open(os.path.join(tflite_models_dir,model_name+'.tflite'), 'wb') as f:
        f.write(tflite_model)
    except Exception as inst:
      print('COULD NOT CONVERT'+model_name)
      print(type(inst))    # the exception instance
      print(inst.args)     # arguments stored in .args
      print(inst)     
      
   
  # print('SAVE TFLite_EdgeTPU Model: '+model_name)

  #compile TFLite model with edgeTPU Compiler
  #command = "edgetpu_compiler "+str(os.path.join(tflite_models_dir,model_name+'.tflite'))+" -o "+ str(tflite_edgetpu_models_dir)
  #returned_value = os.system(command)