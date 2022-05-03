import json
import sys
import os, string
import multiprocessing as mp
from fuzzysearch import find_near_matches
# from transformers import AutoTokenizer
from collections import Counter
from nltk.corpus import wordnet as wn
# tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

files = os.listdir('merged/')

print(files)

do_extend_left = set(string.ascii_lowercase + ' ')
do_extend_right = set(string.ascii_lowercase)

def get_toks(word, this_string):
    if this_string[1] == False:
        start = 11
        max_len = len(this_string[0])
        end = max_len - 11
        while this_string[0][start] != ' ' and start > 0 and this_string[0][start-1].lower() in do_extend_left:
            start -= 1
        while end < max_len and this_string[0][end].lower() in do_extend_right:
            end += 1
        return this_string[0][start:end]
    else:
        match = find_near_matches(word, this_string[0].lower(), max_l_dist=1)[0]
        start = match.start + 1
        max_len = len(this_string[0])
        end = match.end - 1
        while this_string[0][start] != ' ' and start > 0 and this_string[0][start-1].lower() in do_extend_left:
            start -= 1
        while end < max_len and this_string[0][end].lower() in do_extend_right:
            end += 1
        return this_string[0][start:end]

def single_word(word):
    j = json.load(open('merged/' + word + '.json'))
    matched_words = [get_toks(word, jj) for jj in j]
    ctr = Counter(matched_words)
    return ctr

def main():
    target_words = [w.split('/')[-1].split('.')[0] for w in files]

    pool = mp.Pool(mp.cpu_count()*2)
    print(mp.cpu_count()*2)

    result = pool.map(single_word, target_words)

    dict_format = {k: list(v.items()) for k,v in zip(target_words, result)}
    print({k: len(v) for k, v in dict_format.items()})

    json.dump(dict_format, open("quantified.json", 'w+'))

if __name__ == "__main__":
    main()
