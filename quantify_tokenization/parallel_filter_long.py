import json
from torch.utils import data
import sys
from nltk.tokenize import sent_tokenize
import regex

foldername = sys.argv[1]

MAX_WORDS = int(sys.argv[2])
words = [x.strip() for x in open('long_words.txt').read().strip().split('\n')][:MAX_WORDS]
regex_pattern = '|'.join(['(' + w + '){e<=2}' for w in words])


class ParallelDataset(data.Dataset):
    def __init__(self):
        self.path = str(foldername) + "/"
        print("PATH:", self.path)

    def __len__(self):
        return 141

    def __getitem__(self, file_arg_idx):
        filepath = self.path + str(file_arg_idx) + ".jsonl"
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

            if i % 5000 == 0:
                print(file_arg_idx, i, len(sentences_matched))

            i += 1
            line = fp.readline()

        json.dump(sentences_matched, open(filepath+".out.json", "w+"))
        return i

parallel_loader = data.DataLoader(ParallelDataset(), num_workers=50, batch_size=100)
for x in parallel_loader:
    print(x)
