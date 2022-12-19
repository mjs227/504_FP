import spacy
import json


with open('The_Ministers_Black_Veil_8', 'r') as f:
    text = f.read()

nlp = spacy.load('en_core_web_sm', exclude=['parser'])
nlp.enable_pipe('senter')
text_sents = nlp(text)
out_json, para_cnt, new_para = [], 0, False

for sent in text_sents.sents:
    sent_text = sent.text
    newline = '\n' in sent_text
    sent_text = sent_text.strip()

    if len(sent_text) > 0:
        if new_para:
            new_para = False
            para_cnt += 1

        out_json.append({
            'text': sent_text,
            'span': (sent.start_char, sent.end_char),
            'paragraph': para_cnt
        })

        if newline:
            para_cnt += 1
    else:
        new_para = True

with open('text_8_sents.json', 'w') as f:
    json.dump(out_json, f, indent=2)
    