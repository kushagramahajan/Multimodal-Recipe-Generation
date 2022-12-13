import json
import collections
import nltk
import string
from nltk.corpus import stopwords

title_map = {}

stop_words = set(stopwords.words('english'))
stop_words.add('recipe')

with open("layer1_splitted_val.json") as file:
    json_data = json.load(file)
    for elem in json_data:
        if(elem['partition'] == 'val'):
            title = elem['title'].lower()
            title = title.translate(str.maketrans('', '', string.punctuation))
            if( title != ''): 
                title_map[elem['id']] = title

delete_keys = []

for id in title_map.keys():

    title = title_map[id]
    tokens = nltk.word_tokenize(title)
    tokens = [token for token in tokens if (token not in stop_words) and (len(token) > 1)]

    if(len(tokens) < 1):
        delete_keys.append(id)
        continue

    if(len(tokens) < 2):
        title_map[id] = tokens[-1]
    elif(len(tokens) < 3):
        title_map[id] = (tokens[-2] + ' ' + tokens[-1])
    else:
        title_map[id] = (tokens[-3] + ' ' + tokens[-2] + ' ' + tokens[-1])

for id in delete_keys:
    del title_map[id]

with open("shortened_titles_val.json", "w") as outfile:
    json.dump(title_map, outfile)
