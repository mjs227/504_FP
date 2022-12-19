
import json
from tqdm import tqdm


text_index = 8

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_coref.json', 'r') as f:
    coref_data = json.load(f)

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_sents.json', 'r') as f:
    sent_data = json.load(f)

for i in tqdm(range(len(sent_data))):
    start, end = tuple(sent_data[i]['span'])
    sent_data[i].update({'span': range(start, end)})

for i in tqdm(range(len(coref_data))):
    k_ = 0

    for j in range(len(coref_data[i])):
        start = coref_data[i][j]['span'][0]
        sent_found = False

        for k in range(k_, len(sent_data)):
            if start in sent_data[k]['span']:
                sent_found, k_ = True, k
                coref_data[i][j].update({
                    'sentence': k,
                    'paragraph': sent_data[k]['paragraph'],
                    'ref form': None
                })
                break

        if not sent_found:
            print((i, j))

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_coref_sents.json', 'w') as f:
    json.dump(coref_data, f)
