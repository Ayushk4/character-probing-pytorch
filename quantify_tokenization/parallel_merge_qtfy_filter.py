import json
import sys
import os
import multiprocessing as mp

try:
    os.mkdir("merged/")
except:
    pass

folder_names = ['00', '01', '02', '03', '04']
nums = list(range(141))
files = [folder_name + '/' + str(n) + '.jsonl.qtfy.json'
         for folder_name in folder_names for n in nums]

print(files)

def single_word(word):
    occurrences = []
    for file_idx, file in enumerate(files):
        j = json.load(open(file))
        occurrences.extend(j[word])
        if file_idx % 50 == 0:
            print(word, len(occurrences), file)

    json.dump(occurrences, open('merged/' + word + '.json', 'w+'))

    non_matched = [jj for jj in occurrences if word not in jj[0].lower()]

    return len(occurrences), len(non_matched)

def main():
    target_words = list(json.load(open(files[0])).keys())

    pool = mp.Pool(mp.cpu_count())
    print(mp.cpu_count())

    result = pool.map(single_word, target_words)

    print(result)

if __name__ == "__main__":
  main()


