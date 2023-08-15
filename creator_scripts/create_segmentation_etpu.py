import tensorflow as tf
from PIL import Image
import requests
from transformers import TFSegformerForSemanticSegmentation, TFSegformerImageProcessor


# model = tf.keras.models.load_model('segformer_b0_cityscapes.h5')
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# model_checkpoint = "nvidia/mit-b0"
# id2label = {0: "outer", 1: "inner", 2: "border"}
# label2id = {label: id for id, label in id2label.items()}
# num_labels = len(id2label)

# model = TFSegformerForSemanticSegmentation.from_pretrained(
#     model_checkpoint,
#     num_labels=num_labels,
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True,
# )


#feature_extractor = TFSegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
processor = TFSegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
print(model.summary())
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

#inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)