import os
import json

import numpy as np
from tqdm import tqdm

data_dir = '/home/lfischer/mnt_data/staay/imagenet_data'
test_images = ['n01440764/450.npy', 'n03485407/5015.npy', 'n04265275/7995.npy', 'n03131574/8836.npy', 'n01614925/9938.npy']

imgs = {}
for mod in tqdm(os.listdir(data_dir)):
    try:
        imgs[mod] = np.array([ np.load(os.path.join(data_dir, mod, img)) for img in test_images ])
    except FileNotFoundError:
        test_images_name_error = [img.split('/')[0] + '\n/' + img.split('/')[1] for img in test_images]
        imgs[mod] = np.array([ np.load(os.path.join(data_dir, mod, img)) for img in test_images_name_error ])

models = {}
known_prep = {}
for mod, data in imgs.items():
    for mod2, data2 in known_prep.items():
        if data.shape == data2.shape and np.allclose(data, data2):
            models[mod] = mod2
            break
    else:
        models[mod] = mod
        known_prep[mod] = data

with open('prep_map.json', 'w') as jf:
    json.dump(models, jf)
