import spacy
import neuralcoref
import json


class CorefSpan:
    def __init__(self, m_, index_):
        self.start = m_.start_char
        self.end = m_.end_char
        self.cluster = index_

    def __lt__(self, span_2):
        return self.start < span_2.start

    def to_json(self, text_):
        return {
            'span': (self.start, self.end),
            'index': self.cluster,
            'text': text_,
            'paragraph': None,
            'sentence': None,
            'animate': None,
            'syn role': None
        }


with open('The_Ministers_Black_Veil_8', 'r') as f:
    text = f.read()

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)
text_coref = nlp(text)
coref_clusters, total_ref = [], 0

for i in range(len(text_coref._.coref_clusters)):
    d = text_coref._.coref_clusters[i].__dict__
    index = d['i']
    new_cluster = [CorefSpan(m, index) for m in d['mentions']]
    new_cluster.sort()
    cluster_json = [x.to_json(text[x.start: x.end]) for x in new_cluster]
    coref_clusters.append(cluster_json)
    total_ref += len(cluster_json)

with open('text_8_coref.json', 'w') as f:
    json.dump(coref_clusters, f)

print(total_ref)
print(len(coref_clusters))
