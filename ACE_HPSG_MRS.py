
import subprocess
import json
import re
from tqdm import tqdm


DIGITS = {str(x) for x in range(10)}


def to_node(hpsg_str):
    txt = hpsg_str.split('(', maxsplit=1)[0]

    if '+FROM' in txt:  # terminal node
        start_search, end_search = False, False
        start_str, end_str = [0, 0], [0, 0]

        for i in range(len(txt)):
            if txt[i:i+5] == '+FROM':
                start_search = True
                start_str[0] = i
            elif start_search and txt[i:i+2] == '\\\"' and txt[i+2] not in DIGITS:
                start_str[1] = i + 2
                start_search = False
            elif txt[i:i+3] == '+TO':
                end_search = True
                end_str[0] = i
            elif end_search and txt[i:i+2] == '\\\"' and txt[i+2] not in DIGITS:
                end_str[1] = i + 2
                break

        start_index = int(re.search(r'\d+', txt[start_str[0]: start_str[1] + 1]).group())
        end_index = int(re.search(r'\d+', txt[end_str[0]: end_str[1] + 1]).group())

        return {
            'text': txt.strip(),
            'dtrs': [],
            'span': (start_index, end_index)
        }
    else:
        p_cnt, dtrs = 0, []

        for i in range(len(hpsg_str)):
            if hpsg_str[i] == '(':
                if p_cnt == 0:
                    dtrs.append([i + 1])

                p_cnt += 1
            elif hpsg_str[i] == ')':
                p_cnt -= 1

                if p_cnt == 0:
                    dtrs[-1].append(i)

        dtr_nodes, start, end = [], set(), set()

        for d in dtrs:
            dtr_json = to_node(hpsg_str[d[0]:d[1]].strip())
            dtr_nodes.append(dtr_json)
            start.add(dtr_json['span'][0])
            end.add(dtr_json['span'][1])

        start_index, end_index = min(start), max(end)

        return {
            'text': txt.strip(),
            'dtrs': dtr_nodes,
            'span': (start_index, end_index)
        }


def result_to_ep(result):
    result = result[1:-1].strip()
    predicate = result.split('<')[0].replace('\"', '').strip()
    from_to = re.search(r'<\d+:\d+>', result).group()[1:-1].split(':')
    remove, count, offset = [], 0, 0

    for i in range(len(result)):
        if result[i] == '[':
            if count == 0:
                remove.append([i])

            count += 1
        elif result[i] == ']':
            count -= 1

            if count == 0:
                remove[-1].append(i)

    for pair in remove:
        result = result[:pair[0] - offset] + result[(pair[1] + 1) - offset:]
        offset += pair[1] - pair[0]

    label, arg_dict = '', {}
    tag_match = list(re.finditer(r'\S+: \S+', result))

    for pair in tag_match:
        tag, item = tuple(map(lambda s: s.replace('\"', '').strip(), pair.group().split(':')))

        if tag == 'LBL':
            label = item
        else:
            arg_dict.update({tag: item})

    return {
        'label': label,
        'predicate': predicate,
        'lnk': (int(from_to[0]), int(from_to[1])),
        'arguments': arg_dict
    }


def results_to_json(results):
    if results[0][:5] == 'SKIP:':
        return None

    rels_list, i = [], 0

    while not results[i][:4] == 'RELS':
        i += 1

    rels_list.append(result_to_ep(results[i][8:]))

    while not results[i][-1] == '>':
        i += 1

        if results[i][-1] == '>':
            rels_list.append(result_to_ep(results[i][:-2]))
        else:
            rels_list.append(result_to_ep(results[i]))

    while not results[i][:5] == 'HCONS':
        i += 1

    qeq_match = list(re.finditer(r'h\d+ qeq h\d+', results[i]))
    qeq_dict = {}

    for match in qeq_match:
        match_strs = match.group().split(' qeq ')
        qeq_dict.update({match_strs[0]: match_strs[1]})

    return {'relations': rels_list, 'qeq': qeq_dict}


text_index = 8

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_sents.json', 'r') as f:
    file = json.load(f)

outputs = []

for j in tqdm(range(len(file))):
    p = subprocess.Popen(
        ['./ace', '-g', 'erg-1214-x86-64-0.9.34.dat', '-1f'],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd='/home/mj/Desktop/ace-0.9.34'
    )
    p.stdin.write(bytes(file[j]['text'], encoding='utf-8'))
    p.stdin.write(b'\n')
    p.stdin.close()
    readlines = p.stdout.readlines()
    output = []

    for k in range(len(readlines)):
        line = readlines[k].decode('utf-8').strip()

        if len(line) > 0:
            output.append(line)

    outputs.append(output)

out_list = []

for j in tqdm(range(len(file))):
    out_list.append({'sent': file[j]['text'], 'MRS': None, 'SUBJ': None, 'span': file[j]['span']})

    if not (len(outputs[j]) == 0 or 'SKIP:' in {outputs[j][k][:5] for k in range(len(outputs[j]))}):
        hcons_line, hpsg_line = tuple(map(lambda z: z.strip(), outputs[j][-1].split(';', maxsplit=1)))
        outputs[j][-1] = hcons_line

        try:
            mrs = results_to_json(outputs[j])
            subj_span = to_node(hpsg_line[1:-1])['dtrs'][0]['span']
            out_list[-1].update({'MRS': mrs, 'SUBJ': subj_span})
        except:
            continue

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_hpsg_mrs.json', 'w') as f:
    json.dump(out_list, f)
