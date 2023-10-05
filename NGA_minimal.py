#source l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64/setupvars.sh
from openvino.runtime import Core
import numpy as np


ie = Core()

devices = ie.available_devices

for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

classification_model_xml = "/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/models/openVINO/ResNet50/saved_model.xml"

model = ie.read_model(model=classification_model_xml)
compiled_model = ie.compile_model(model=model, device_name="MYRIAD")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
import cv2

image_filename = "/home/staay/Git/imagenet-on-the-edge/n01484850_great_white_shark.JPEG"
image = cv2.imread(image_filename)
N, C, H, W = input_layer.shape
resized_image = cv2.resize(src=image, dsize=(W, H))
resized_image.shape

input_data = np.load("/home/staay/Git/imagenet-on-the-edge/mnt_data/staay/imagenet_data/DenseNet121/n01440764/450.npy")


result = compiled_model(input_data)[output_layer]

print(input_data.shape)
print(image.shape)
print(input_layer.shape)
print(output_layer.shape)

print(result.shape)