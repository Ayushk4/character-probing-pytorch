import torch
import os, json, random, sys
from params import params
import numpy as np
from transformers import AutoTokenizer
from torch.utils import data
import re
import string
import pickle

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

do_lowercase = True
lowercase = lambda x: x.lower() if do_lowercase else x
MAX_LEN = 0

bert_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
clean_up = lambda x: x[1:].lower() if x[0] == 'Ġ' else x.lower()

def shuffle_and_return(x):
  random.shuffle(x)
  return x

flatten = lambda x: [y for xx in x for y in xx]

class SpellingDataset:
    def __init__(self):
        self.bert_tokenizer = bert_tokenizer
        self.data_raw = list(pickle.load(open('generated_data.json', 'rb')).items())

        train_set, test_set = self.lemma_wise_split(self.data_raw)

        self.train_set = self.process(train_set, False)
        random.shuffle(self.train_set)

        if params.dummy_run:
            self.test_set = self.train_set
        else:
            self.test_set = self.process(test_set, True)

    def lemma_wise_split(self, full_data):
        lemma_wise = {x: [] for x in set(lemmatizer.lemmatize(clean_up(xx[0]))
                        for xx in full_data)}
        for x in full_data:
            lemma_wise[lemmatizer.lemmatize(clean_up(x[0]))].append(x)

        full_dataset_lemmawise = list(lemma_wise.values())
        self.split = int(0.8 * len(full_dataset_lemmawise))
        train_portion = flatten(full_dataset_lemmawise[:self.split])
        test_portions = flatten(full_dataset_lemmawise[self.split:])

        return train_portion, test_portions

    def process(self, all_data, is_test):
        if params.dummy_run:
            all_data = all_data[:5]

        label_mapper = {'pos': 1, 'neg':0}
        return [((superstring, substring, label) if is_test else 0,
                 bert_tokenizer.convert_tokens_to_ids(superstring),
                 bert_tokenizer.convert_tokens_to_ids(substring),
                 label_mapper[label],
                 [1.0*(x in superstring.lower()) for x in string.ascii_lowercase + string.ascii_uppercase],
                 [1.0*(x in substring.lower()) for x in string.ascii_lowercase + string.ascii_uppercase],
                )
                    for superstring, pos_negs in all_data
                    for label, substrings in pos_negs.items()
                    for substring in substrings
            ]

def pad(batch):
    get_f = lambda x: [single_data[x] for single_data in batch]
    batch_tokens       = get_f(0)
    superstring_batch  = torch.LongTensor(get_f(1)).to(params.device)
    substring_batch    = torch.LongTensor(get_f(2)).to(params.device)
    label_batch        = torch.FloatTensor(get_f(3)).to(params.device)
    onehot_superstring = torch.FloatTensor(get_f(4)).to(params.device)
    onehot_substring   = torch.FloatTensor(get_f(5)).to(params.device)

    return batch_tokens, superstring_batch, substring_batch, label_batch, onehot_superstring, onehot_substring

if __name__ == "__main__":
    dataset = SpellingDataset()
    print("Num datapoints:", len(dataset.train_set))
    import sys, os
    os.system("ps -a -o pid,command,%mem")
    print("====")

    print(pad(dataset.train_set[:5]))
    print(pad(dataset.test_set[:5]))

