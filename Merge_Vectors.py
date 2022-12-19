
import json


text_indices = [1, 2, 3, 4, 5, 6, 7, 8]
all_forms = False

max_protag = 0
x_files, y_files = [], []

for i in text_indices:
    with open('/home/mj/Desktop/504_FP/JSON_Files/text_' + str(i) + '_vectors.json', 'r') as f:
        file = json.load(f)

    x, y = file['x'], file['y']
    max_protag = max(max_protag, max(z[0] for z in x))
    x_files += x
    y_files += y

for i in range(len(x_files)):
    x_files[i][0] = x_files[i][0] / max_protag

if all_forms:
    unique_items_rp = list({tuple(x_files[i] + y_files[i]) for i in range(len(x_files))})
    unique_items = list({tuple(x_files[i][:6] + y_files[i]) for i in range(len(x_files))})
    unique_y_rp = [x[-4:] for x in unique_items_rp]
    unique_y = [x[-4:] for x in unique_items_rp]
    unique_x_rp = [x[:7] for x in unique_items_rp]
    unique_x = [x[:6] for x in unique_items_rp]
else:
    unique_items_rp = list({tuple(x_files[i] + [1 if y_files[i][-2] == 1 else 0]) for i in range(len(x_files)) if y_files[i][-1] == 0})
    unique_items = list({tuple(x_files[i][:6] + [1 if y_files[i][-2] == 1 else 0]) for i in range(len(x_files)) if y_files[i][-1] == 0})
    unique_y_rp = [x[-1] for x in unique_items_rp]
    unique_y = [x[-1] for x in unique_items_rp]
    unique_x_rp = [x[:7] for x in unique_items_rp]
    unique_x = [x[:6] for x in unique_items_rp]


with open('/home/mj/Desktop/504_FP/vector_list_' + ('all' if all_forms else 'bin') + '_no_rp.json', 'w') as f:
    json.dump({'x': unique_x, 'y': unique_y}, f)

with open('/home/mj/Desktop/504_FP/vector_list_' + ('all' if all_forms else 'bin') + '_ref_per.json', 'w') as f:
    json.dump({'x': unique_x_rp, 'y': unique_y_rp}, f)

print(len(unique_x))
print(len(unique_y))
print(len(unique_x_rp))
print(len(unique_y_rp))
