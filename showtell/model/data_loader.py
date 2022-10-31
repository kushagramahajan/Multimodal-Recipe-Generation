'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script loads the COCO dataset in batches to be used for training/testing
''' 

import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import json

class DataLoader(data.Dataset):
    def __init__(self, image_root, caption_path, image_map, vocab, transform=None):

        self.image_root = image_root

        caption_file = open(caption_path)
        self.recipes = json.load(caption_file)
        self.recipes = [ elem for elem in self.recipes if elem['partition'] == 'val']

        self.recipe_ids = set([recipe['id'] for recipe in self.recipes])

        image_mapping_file = open(image_map)
        data = json.load(image_mapping_file)

        self.recipe_to_image = {}

        not_present = 0

        for elem in data:
            recipe_id = elem['id']
            image_id = elem['images'][0]['id']
            self.recipe_to_image[recipe_id] = image_id
            if(recipe_id in self.recipe_ids):
                not_present += 1

        print(not_present)

        self.recipes = [elem for elem in self.recipes if elem['id'] in self.recipe_to_image]
        self.recipe_ids = set([recipe['id'] for recipe in self.recipes])


        #print(self.recipe_to_image)

        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):

        vocab = self.vocab
        recipe_id = self.recipes[index]['id']
        caption = ''
        for instruction in self.recipes[index]['instructions']:
            caption += instruction['text'] + ' '
        caption = caption.strip()

        path = self.recipe_to_image[recipe_id]

        image = Image.open(os.path.join(self.image_root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.recipe_ids)

def collate_fn(data):
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(method, vocab, batch_size):

    # train/validation paths
    if method == 'train':
        root = 'data/images_resized'
        json = 'data/annotations/layer1_splitted_val.json'
        image_map = 'data/annotations/layer2_val.json'
    elif method =='val':
        root = 'data/test_images_resized'
        json = 'data/annotations/layer1_splitted_val.json'
        image_map = 'data/annotations/layer2_val.json'

    # rasnet transformation/normalization
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

    loader = DataLoader(image_root=root, caption_path=json, image_map = image_map, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=loader,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader
