from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True
import os

import tensorflow as tf
import tensorflow_datasets as tfds

def load_data(rootdir='mnt_data', raw_sub='raw', unpacked_sub='unpacked', preprocess=None, batch_size=32, n_batches=None):
    
    raw_dir = os.path.join(rootdir, raw_sub)
    write_dir = os.path.join(rootdir, unpacked_sub)
    download_and_prepare_kwargs = {
        'download_dir': write_dir,
        'download_config': tfds.download.DownloadConfig(extract_dir=write_dir, manual_dir=raw_dir),
    }

    ds, info = tfds.load('imagenet2012_subset',
        data_dir=write_dir,         
        split='validation',
        shuffle_files=False,
        download=True,
        as_supervised=True,
        with_info=True,
        download_and_prepare_kwargs=download_and_prepare_kwargs
    )

    if preprocess is not None:
        ds = ds.map(preprocess)

    n_gpus = max(len(tf.config.list_physical_devices('GPU')), 1) # if no GPU is available, just use given batch size
    ds = ds.batch(batch_size * n_gpus, drop_remainder=True)
    if n_batches is not None:
        ds = ds.take(n_batches)
    
    return ds, info

