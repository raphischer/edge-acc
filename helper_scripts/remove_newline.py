import os

path = 'mnt_data/staay/imagenet_data copy'

for root, dirs, files in os.walk(path):
    for dir1 in dirs:
        os.rename(dir1,'jes')
        print(path+'/'+dir1)