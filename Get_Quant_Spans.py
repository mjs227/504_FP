
import json


class OrderedNode:
    def __init__(self, id_, value):
        self.id = id_
        self.value = value

    def __lt__(self, on):
        return self.value < on.value


def get_quant_type(q_):
    if q_ in def_q:
        return 'def'

    if q_ in indef_q:
        return 'indef'

    if q_ in proper_q:
        return 'prop'

    if q_ in pronoun_q:
        return 'pro'

    return None


def pseudo_dist(a, b):
    (x1, y1), (x2, y2) = a, b

    return (((x1 - x2) ** 2) + abs(y1 - y2)) ** .5


def_q = {'def_explicit_q', 'def_implicit_q', '_that_q_dem', '_this_q_dem', '_these_q_dem', '_the_q', 'every_q', '_each_q',
         '_those_q_dem', '_every_q', '_all_q', '_both_q'}
indef_q = {'_some_q', '_any_q', '_another_q', 'udef_q', '_a_q', '_no_q', '_half_q', '_some_q_indiv', '_such+a_q', '_such_q',
           '_which_q', 'which_q', '_what+a_q', '_a+little_q'}
proper_q = {'proper_q'}
pronoun_q = {'pronoun_q'}
ignore_q = {'number_q', 'idiom_q_i', 'free_relative_q', '_enough_q', 'free_relative_ever_q'}

text_index = 8

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_hpsg_mrs.json', 'r') as f:
    file = json.load(f)

for i in range(len(file)):
    if file[i]['MRS'] is None:
        file[i].update({'quants': None})
    else:
        mrs = file[i]['MRS']['relations']
        handle_dict, quants, poss = {}, [], set()

        for ep in mrs:
            if ep['predicate'] == 'poss':
                if {'ARG0', 'ARG1'} <= set(ep['arguments'].keys()) and ep['arguments']['ARG0'][0] == 'x':
                    poss.add(ep['arguments']['ARG0'])
                elif {'ARG1', 'ARG2'} <= set(ep['arguments'].keys()) and ep['arguments']['ARG1'][0] == 'x':
                    poss.add(ep['arguments']['ARG1'])
            else:
                quant_type = get_quant_type(ep['predicate'])

                if quant_type is not None:
                    quants.append({'ep': ep, 'type': quant_type})

            if ep['label'] in handle_dict.keys():
                handle_dict[ep['label']].append(ep)
            else:
                handle_dict.update({ep['label']: [ep]})

        min_subj_dist, min_subj_dist_index = float('inf'), -1
        subj_span = tuple(file[i]['SUBJ'])
        qeq_dict = file[i]['MRS']['qeq']

        for j in range(len(quants)):
            q = quants[j]
            bvar = q['ep']['arguments']['ARG0']
            rstr = q['ep']['arguments']['RSTR']
            start, end = tuple(q['ep']['lnk'])

            if rstr in handle_dict.keys():
                rstr_handle = handle_dict[rstr]
            elif rstr in qeq_dict.keys() and qeq_dict[rstr] in handle_dict.keys():
                rstr_handle = handle_dict[qeq_dict[rstr]]
            else:
                rstr_handle = []

            for ep in rstr_handle:
                start_, end_ = tuple(ep['lnk'])
                start = min(start, start_)
                end = max(end, end_)

            dist_to_subj = pseudo_dist((start, end), subj_span)

            if dist_to_subj < min_subj_dist:
                min_subj_dist = dist_to_subj
                min_subj_dist_index = j

            q.pop('ep')
            q.update({'span': (start, end), 'poss': bvar in poss, 'subj': False})

        if not min_subj_dist_index == -1:
            quants[min_subj_dist_index].update({'subj': True})

        file[i].update({'quants': quants if len(quants) > 0 else None})

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_hpsg_mrs.json', 'w') as f:
    json.dump(file, f)
