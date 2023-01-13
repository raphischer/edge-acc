import tensorflow as tf

from load_data import load_data
from load_models import load_preprocessing, prepare_model

model_name = 'MobileNetV3Small'

# load data and model
preprocess = load_preprocessing(model_name)
ds, _ = load_data(preprocess=preprocess, n_batches=50)
model = prepare_model(model_name)

result = model.evaluate(ds, return_dict=True)
print(result)
