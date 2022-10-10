from transformers import AutoFeatureExtractor, SwinForImageClassification, SwinModel
from PIL import Image
import requests
from kmeans_pytorch import kmeans
import torch
import json
import torchvision.transforms as transforms
import numpy as np
import matplotlib.cm
import wandb
import random

# import required module
import os
# assign directory
directory = 'val'

images = []
predictions = []
predicted_classes = []
 
# iterate over files in
# that directory

cm_jet = matplotlib.cm.get_cmap('jet')

feature_extractor = AutoFeatureExtractor.from_pretrained('aspis/swin-finetuned-food101')
model = SwinForImageClassification.from_pretrained('aspis/swin-finetuned-food101')
model = model.cpu()

for subdir, dirs, files in os.walk(directory):
    for file in files:
        filepath = os.path.join(subdir, file)
        random_number = random.randint(1, 100)
        if(random_number == 6):

                print(filepath)
                image = Image.open(filepath)

                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                outputs = outputs

                logits = outputs.logits.cpu()
                images.append(filepath)

                predictions.append(logits)
                predicted_class_idx = logits.argmax(-1).item()
                predicted_classes.append(model.config.id2label[predicted_class_idx])

predictions = torch.stack(predictions)

print(predictions.shape)

cluster_ids_x, cluster_centers = kmeans(
    X=predictions, num_clusters=50, distance='cosine', device=torch.device('cpu')
)

list = cluster_ids_x.cpu().numpy().tolist()

cluster_dict = {}
classes_dict = {}

for i in range(len(list)):
    elem = list[i]
    cl = predicted_classes[i]

    if elem not in cluster_dict:
        cluster_dict[elem] = []
    if cl not in classes_dict:
        classes_dict[cl] = []
    
    cluster_dict[elem].append(images[i])
    classes_dict[cl].append(images[i])

with open('cluster.txt', 'w') as convert_file:
     convert_file.write(json.dumps(cluster_dict, indent=2))

with open('classes.txt', 'w') as convert_file:
     convert_file.write(json.dumps(classes_dict, indent=2))
