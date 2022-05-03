import json
import sys
from nltk.tokenize import sent_tokenize
import regex, re
from fuzzysearch import find_near_matches
import multiprocessing as mp

foldername = sys.argv[1]

MAX_WORDS = int(sys.argv[2])
words = [x.strip() for x in open('long_words.txt').read().strip().split('\n')][:MAX_WORDS]
regex_pattern = '|'.join(['(' + w + '){e<=2}' for w in words])

path = str(foldername) + "/"
print("PATH:", path)

def single_portion(file_arg_idx):
    filepath = path + str(file_arg_idx) + ".jsonl"
    fp = open(filepath)
    i = 0
    line = fp.readline()
    sentences_matched = []

    while len(line) > 0:
        json_loaded = json.loads(line)
        txt = json_loaded['text'].encode("ascii", errors="ignore").decode()

        sentences = sent_tokenize(txt)

        if len(sentences) != 0:
            sent = sentences[0]
            sent_idx = 1
            while sent_idx < len(sentences):
                if len(sent) < 1000:
                    sent += " " + sentences[sent_idx]
                else:
                    # match = regex.search(regex_pattern, sent.lower(),
                    #                 flags=regex.IGNORECASE)
                    # if match != None:
                    #     sentences_matched.append(sent)
                    for wrd in words:
                        # if wrd in sent.lower():
                        #     sentences_matched.append(sent)
                        #     break
                        match = find_near_matches(wrd, sent.lower(), max_l_dist=1)
                        if len(match) != 0:
                            sentences_matched.append(sent)
                            break
                    sent = sentences[sent_idx]
                sent_idx += 1

            if len(sent) > 5:
                match = regex.search(regex_pattern, txt.lower(),
                                        flags=regex.IGNORECASE)
                if match != None:
                    sentences_matched.append(sent)

        if i % 1000 == 0:
            print(file_arg_idx, i, len(sentences_matched))

        i += 1
        line = fp.readline()

    json.dump(sentences_matched, open(filepath+".out.json", "w+"))
    return len(sentences_matched)

def main():
  pool = mp.Pool(mp.cpu_count())
  print(mp.cpu_count())
  ip_args = [i for i in range(141)]
  result = pool.map(single_portion, ip_args)

  print(result)

if __name__ == "__main__":
  main()
