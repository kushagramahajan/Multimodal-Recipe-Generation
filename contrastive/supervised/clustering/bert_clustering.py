import spacy
from nltk.cluster import KMeansClusterer
import nltk
import spacy_sentence_bert
import numpy as np
import json

# load one of the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
nlp = spacy_sentence_bert.load_model('en_nli_bert_large')

titles = []

# Utility function for generating sentence embedding from the text
def get_embeddinngs(text):
    return nlp(text).vector

with open("layer1_splitted_val.json") as file:
    count = 0
    json_data = json.load(file)
    for elem in json_data:
        if(elem['partition'] == 'train'):
            titles.append(get_embeddinngs(elem['title'].lower()))
            count += 1
            print(count)

print(len(titles))

def clustering_question(NUM_CLUSTERS = 15):

    X = np.array(titles)

    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
        repeats=25,avoid_empty_clusters=True)

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    print(len(assigned_clusters))

    #data['cluster'] = pd.Series(assigned_clusters, index=data.index)
    #data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])

    #return data, assigned_clusters

# Generating sentence embedding from the text
clustering_question()