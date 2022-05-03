import os, sys
import random
from transformers import AutoTokenizer

lines = [lin.strip().split() for lin in open(sys.argv[1]).readlines()]
random_split_prob = float(sys.argv[2])

do_rand_split = 'rand' in sys.argv[3].lower()
if not do_rand_split:
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    unique_keys = set([k[1:] if k[0] == 'Ġ' else k for k in tok.vocab.keys()])
    # unweighted = 'un' in sys.argv[4].lower()

def split_token(tok):
    if len(tok) == 1:
        return [tok]
    if random.random() < random_split_prob:
        if do_rand_split:
            index_choice = random.choice(range(1, len(tok)))
            return [tok[:index_choice], tok[index_choice:]]
        else:
            choices = [i for i in range(1, len(tok))
                if tok[:i].lower() in unique_keys and tok[i:].lower() in unique_keys]
            if len(choices) == 0:
                return [tok]
            index_choice = random.choice(choices)
            return [tok[:index_choice], tok[index_choice:]]
    else:
        return [tok]

print("Starting")
random_splits = []
for i, lin in enumerate(lines):
    if i % 10000 == 0 and i > 0:
        print(i)
    random_splits.append([t.strip() for tok in lin for t in split_token(tok)])

outfile = sys.argv[1].strip('.tx') + \
        '.' + sys.argv[2]
if do_rand_split:
    outfile += '.rand'
else:
    outfile += '.gpt'
# if not do_rand_split:
#     outfile += '.unwtd' if unweighted else '.wtd'
outfile += '.txt'

open(outfile, 'w+').write('\n'.join([' '.join(line) for line in random_splits]))
