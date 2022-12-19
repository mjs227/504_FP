
import json


# ref. persistence: [0, 20] -> [0, 1]
# animacy: [0, 1]
# "protagonisthood" (num. mentions up to that point -- i.e. index in list): [0, max] -> [0, 1]
# paragraph distance: [0, 20] -> [0, 1]
# linear distance: [0, 20] -> [0, 1]
# antecedent syntactic role: 2D 1-hot

text_index = 8

target_dict = {
    'def': [1, 0, 0, 0],
    'indef': [0, 1, 0, 0],
    'pro': [0, 0, 1, 0],
    'prop': [0, 0, 0, 1]
}

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_coref_sents.json', 'r') as f:
    file = json.load(f)

out_x, out_y = [], []

for i in range(len(file)):
    rp_cnt, prev_para, prev_sent, prev_role = 0, -20, -20, 'n'

    for j in range(len(file[i])):
        curr_sent = file[i][j]['sentence']
        curr_para = file[i][j]['paragraph']
        ref_form = file[i][j]['ref form']

        if prev_sent == curr_sent - 1:
            rp_cnt = min(rp_cnt + 1, 20)
        elif not prev_sent == curr_sent:
            rp_cnt = 0

        if ref_form is not None:
            ref_per = rp_cnt / 20
            para_d = 1 - (min(curr_para - prev_para, 20) / 20)
            lin_d = 1 - (min(curr_sent - prev_sent, 20) / 20)
            lin_ant_role = [1 if prev_role == 's' else 0, 1 if prev_role == 'p' else 0]
            protag = j
            anim = 1 if file[i][j]['animate'] else 0

            out_x.append([protag, para_d, lin_d, anim] + lin_ant_role + [ref_per])
            out_y.append(target_dict[ref_form])

        prev_sent = curr_sent
        prev_para = curr_para
        prev_role = file[i][j]['syn role']

with open('/home/mj/Desktop/504_FP/text_' + str(text_index) + '_vectors.json', 'w') as f:
    json.dump({'x': out_x, 'y': out_y}, f)

print(len(out_x))
print(len(out_y))
