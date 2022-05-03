import json
from fuzzysearch import find_near_matches
import multiprocessing as mp

MAX_LEVEN_DIST = 2

MAX_WORDS = 1000
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

def main():
    pool = mp.Pool(mp.cpu_count())
    print(mp.cpu_count())

    result = pool.map(get_pseudo_matches, words)

    json.dump({w: r for w, r in zip(words, result)}, open("pseudo_possible.json", 'w+'))

if __name__ == "__main__":
  main()
