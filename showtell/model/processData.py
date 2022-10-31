'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script processes the COCO dataset
'''  

import os
import pickle
from collections import Counter
import nltk
from PIL import Image
import json
#from pycocotools.coco import COCO

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(caption_path, threshold):

    f = open(caption_path)
    data = json.load(f)

    counter = Counter()
    count = 0
    
    for item in data:
        if item['partition'] == 'val':
            count += 1
            for line in item['instructions']:
                tokens = nltk.tokenize.word_tokenize(line['text'].lower())
                counter.update(tokens)
            print(count)

    # ommit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main(caption_path,vocab_path,threshold):

    #vocab = build_vocab(caption_path=caption_path,threshold=threshold)
    #with open(vocab_path, 'wb') as f:
    #    pickle.dump(vocab, f)

    print("resizing images...")
    splits = ['val']

    folder = './data/annotations/images/test'
    resized_folder = './data/test_images_resized/'
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    print('fold')
    for folder, subs, files in os.walk(folder):
        
        for filename in files:
            with open(os.path.join(folder, filename), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, filename), image.format)

    print("done resizing images...")

caption_path = './data/annotations/layer1_splitted_val.json'
vocab_path = './data/vocab.pkl'
threshold = 5

#main(caption_path,vocab_path,threshold)
