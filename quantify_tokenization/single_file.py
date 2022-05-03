import os, sys
import regex, json
from nltk.tokenize import sent_tokenize

MAX_WORDS = int(sys.argv[2])

filepath = sys.argv[1]
words = [x.strip() for x in open('long_words.txt').read().strip().split('\n')][:MAX_WORDS]
regex_pattern = '|'.join(['(' + w + '){e<=2}' for w in words])

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
            if len(sent) < 300:
                sent += " " + sentences[sent_idx]
            else:
                match = regex.search(regex_pattern, sent.lower(),
                                    flags=regex.IGNORECASE)
                if match != None:
                    sentences_matched.append(sent)
                sent = sentences[sent_idx]
            sent_idx += 1

        if len(sent) > 5:
            match = regex.search(regex_pattern, txt.lower(),
                                    flags=regex.IGNORECASE)
            if match != None:
                sentences_matched.append(sent)

    if i % 2000 == 0:
        print(i, len(sentences_matched))

    i += 1
    line = fp.readline()

json.dump(sentences_matched, open(filepath+".out.json", "w+"))
