import tensorflow as tf
import numpy as np
import os

from load_data import load_data
from load_models import load_preprocessing, prepare_model

model_name = 'MobileNetV3Small'

# load data and model
preprocess = load_preprocessing(model_name)
ds, _ = load_data(preprocess=preprocess, n_batches=50) #tf.data.Dataset (tensorflow.python.data.ops.dataset_ops.TakeDataset) , tfds.core.DatasetInfo
y = np.concatenate([y for x, y in ds], axis=0)
x = np.concatenate([x for x, y in ds], axis=0)
print(x)

model = prepare_model(model_name) #keras.engine.functional.Functional' / tf.keras.applications.MobileNetV3Small

#print(model.summary()) 
#result = model.evaluate(ds, return_dict=True)
#print(result)



models_dir='models'
saved_models_dir = os.path.join(models_dir, 'saved_models')


tf.saved_model.save(model, os.path.join(saved_models_dir,model_name))
