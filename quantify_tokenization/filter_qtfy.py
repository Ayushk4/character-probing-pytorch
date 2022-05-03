import json
import sys
from fuzzysearch import find_near_matches
import multiprocessing as mp

foldername = sys.argv[1]
offset = 10
MAX_LEVEN_DIST = 1

MAX_WORDS = int(sys.argv[2])
words = [x.strip().lower() for x in open('long_words.txt').read().strip().split('\n')][:MAX_WORDS]
all_words_long = json.load(open("ascii_all_long_words.json"))

def get_pseudo_matches(target_word):
    # Example projection-protection
    possible_matches = []
    for any_word in all_words_long:
        if target_word == any_word:
            continue
        match = find_near_matches(target_word,
                            any_word,
                            max_l_dist=MAX_LEVEN_DIST
                        )
        if len(match) == 0 or match[0].start != 0 or match[0].end != len(any_word):
            continue
        possible_matches.append(any_word)
    return possible_matches

words_pseudo_matches = {w: get_pseudo_matches(w) for w in words}

path = str(foldername) + "/"
print("PATH:", path)

filepath = path + "0.jsonl"
fp = open(filepath)
i = 0
line = fp.readline()
sentences_matched = {w: [] for w in words}

while len(line) > 0:
    json_loaded = json.loads(line)
    txt = json_loaded['text'].encode("ascii", errors="ignore").decode()

    lowered_txt = txt.lower()
    max_len_txt = len(txt)
    for wrd in words:
        match = find_near_matches(wrd, lowered_txt, max_l_dist=MAX_LEVEN_DIST)
        for m in match:
            if " " in m.matched or "\n" in m.matched or '\t' in m.matched:
                continue
            if m.end != max_len_txt and txt[m.end].isalpha():
                continue
            if m.start != 0 and txt[m.start-1].isalpha():
                continue

            continue_flag = False
            for pseudo in words_pseudo_matches[wrd]:
                if m.matched == pseudo:
                    continue_flag = True
                    break
            if continue_flag:
                continue

            flag = False
            start = m.start - 10
            end = m.end + 10
            if start < 0:
                flag = True
                start = 0
            if end > max_len_txt:
                flag = True
            sentences_matched[wrd].append(
                    (txt[start:end], flag)
                )
    if i % 1000 == 0:
        print(i, [len(sentences_matched[w]) for w in words[:10]])
        # print(file_arg_idx, i, len(sentences_matched))

    if i > 10000:
        break

    i += 1
    line = fp.readline()

json.dump(sentences_matched, open(filepath+".qtfy.json", "w+"))
