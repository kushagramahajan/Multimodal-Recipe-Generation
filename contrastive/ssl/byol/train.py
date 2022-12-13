import torch
from byol_pytorch import BYOL
from torchvision import models
import json
from PIL import Image
import torchvision.transforms as transforms
import os
from datetime import datetime

to_tensor_transform = transforms.ToTensor()

transforms = torch.nn.Sequential(
    transforms.CenterCrop(256),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)

batch_size = 128
epochs = 5

recipe_ids = []
image_ids = []

with open('../data/data_w_ssl/layer1_splitted_val_w_ssl.json', 'r') as split_file:
    data = split_file.read()
    mappings = json.loads(data)
    recipe_ids = [mapping['id'] for mapping in mappings if mapping['partition']=='ssl']

print(len(recipe_ids))

with open('../data/data_w_ssl/layer2_val.json', 'r') as image_file:
    data = image_file.read()
    mappings = json.loads(data)
    for mapping in mappings:
        recipe_id = mapping['id']
        if recipe_id in recipe_ids:
            for image in mapping['images']:
                image_ids.append(image['id'])

# with open('../image_titles_train.json', 'r') as myfile:
#     data = myfile.read()

# # parse file
# mappings = json.loads(data)

#image_ids = [mapping['image_id'] for mapping in mappings]
print("total images: ", len(image_ids))

resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images(index):
    
    range_max = (index+1)*batch_size
    if(range_max > len(image_ids)):
        range_max = len(image_ids)

    images = image_ids[index*batch_size:range_max]
    images = [transforms(to_tensor_transform(Image.open(os.path.join('/home/ubuntu/data/data_w_ssl/images/ssl', *list(image_id[:4]), image_id)))) for image_id in images]
    images = torch.stack(images)

    return images

for epoch in range(epochs):

    for index in range(int(len(image_ids)/batch_size)):
        images = sample_unlabelled_images(index)
        loss = learner(images)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(loss.item(), " ", current_time)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        print("indexes done: ", batch_size * index)
    
    torch.save(resnet.state_dict(), './improved-net-' + str(epoch) + '.pt')