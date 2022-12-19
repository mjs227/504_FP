
import numpy
import json
from scipy import optimize


ANIM_PRO = {'he', 'she', 'you', 'I', 'me', 'him', 'his', 'her', 'hers', 'theirs', 'their', 'we',
            'our', 'ours', 'your', 'yours'}
INANIM_PRO = {'it', 'its', 'that', 'this'}


class CorefInst:
    def __init__(self, cl, index):
        offset = s_offsets[cl['sentence']]
        self.cluster = cl['index']
        self.index = index
        self.span = (cl['span'][0] - offset, cl['span'][1] - offset)


class Quant:
    def __init__(self, q, index):
        self.index = index
        self.span = tuple(q['span'])


def pseudo_dist(a, b):
    if a is None or b is None:
        return -1

    (x1, y1), (x2, y2) = a.span, b.span

    return (((x1 - x2) ** 2) + abs(y1 - y2)) ** .5


def hungarian_matching(coref_list, quant_list):
    if len(coref_list) > len(quant_list):
        quant_list += [None for _ in range(len(coref_list) - len(quant_list))]
    elif len(quant_list) > len(coref_list):
        coref_list += [None for _ in range(len(quant_list) - len(coref_list))]

    cost_arr = [[pseudo_dist(d1, d2) for d1 in quant_list] for d2 in coref_list]
    max_cost = max(max(x) for x in cost_arr) + 10

    for i_ in range(len(cost_arr)):
        for j_ in range(len(cost_arr[i_])):
            if cost_arr[i_][j_] == -1:
                cost_arr[i_][j_] = max_cost

    cost_matrix = numpy.array(cost_arr)
    assign_matrix = optimize.linear_sum_assignment(cost_matrix)
    am_0, am_1 = assign_matrix[0].tolist(), assign_matrix[1].tolist()

    return [(coref_list[i_], quant_list[j_], not cost_matrix[i_][j_] == max_cost) for i_, j_ in zip(am_0, am_1)]


text_index = 8

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_hpsg_mrs.json', 'r') as f:
    hpsg_file = json.load(f)

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_coref_sents.json', 'r') as f:
    coref_file = json.load(f)

coref_sent_list = [[] for _ in range(len(hpsg_file))]
quant_sent_list = [[] for _ in range(len(hpsg_file))]
s_offsets = [0 for _ in range(len(hpsg_file))]

for i in range(len(hpsg_file)):
    s_offsets[i] = hpsg_file[i]['span'][0]

    if hpsg_file[i]['quants'] is not None:
        for j in range(len(hpsg_file[i]['quants'])):
            quant_sent_list[i].append(Quant(hpsg_file[i]['quants'][j], j))

for cluster in coref_file:
    for i in range(len(cluster)):
        coref_sent_list[cluster[i]['sentence']].append(CorefInst(cluster[i], i))

for i in range(len(hpsg_file)):
    if hpsg_file[i]['quants'] is not None:
        match_list = hungarian_matching(coref_sent_list[i], quant_sent_list[i])

        for coref, quant, valid in match_list:
            if valid:
                quant_d = hpsg_file[i]['quants'][quant.index]
                coref_file[coref.cluster][coref.index].update({
                    'syn role': 's' if quant_d['subj'] else ('p' if quant_d['poss'] else 'n'),
                    'ref form': quant_d['type']
                })

anim_clusters = set()

for i in range(len(coref_file)):
    for inst in coref_file[i]:
        if inst['text'] in ANIM_PRO:
            anim_clusters.add(i)
        elif inst['text'] in INANIM_PRO:
            anim_clusters = anim_clusters - {i}
            break

for i in range(len(coref_file)):
    anim = i in anim_clusters

    for j in range(len(coref_file[i])):
        coref_file[i][j].update({'animate': anim})

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_coref_sents.json', 'w') as f:
    json.dump(coref_file, f)
