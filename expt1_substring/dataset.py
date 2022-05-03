from transformers import AutoTokenizer
import re, random, string
bert_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

words = list(set([x for x in list(bert_tokenizer.vocab.keys())
                if (len(x) == 1 or x[1:].islower()) and x[0] not in string.ascii_uppercase
                ]))
regex_pattern = r'Ġ?[a-zA-Z]+$'

group_by_len = {x: [] for x in set([len(y) for y in words if re.match(regex_pattern, y)])}
uniq_lens = sorted(group_by_len.keys())

for w in words:
    if re.match(regex_pattern, w):
        group_by_len[len(w)].append(w)

map_superstring = {}

for l, ws in group_by_len.items():
    for w1 in ws:
        map_superstring[w1] = {'pos': [], 'neg': []}
        for l2 in group_by_len:
            if l2 < l:
                for w2 in group_by_len[l2]:
                    if w2 in w1:
                        map_superstring[w1]['pos'].append(w2)
                random_iter = list(range(len(group_by_len[l2])))
                random.shuffle(random_iter)
                for random_idx in random_iter:
                    if not (group_by_len[l2][random_idx] in w1):
                        map_superstring[w1]['neg'].append(group_by_len[l2][random_idx])
                    if len(map_superstring[w1]['neg']) >= len(map_superstring[w1]['pos']):
                        break
    print(l)

import pickle
pickle.dump(map_superstring, open("generated_data.json", 'wb'))

