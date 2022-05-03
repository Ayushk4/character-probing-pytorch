import torch
import os, json, random, sys
from params import params
import numpy as np
from transformers import AutoTokenizer
import re
import string

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

from tagger import POS_NER_Tagger

pos_tagger = POS_NER_Tagger(params.pos_path)
ner_tagger = POS_NER_Tagger(params.ner_path)
ner_tagger.eval()
pos_tagger.eval()
print(ner_tagger.meta, pos_tagger.meta)

do_lowercase = True
lowercase = lambda x: x.lower() if do_lowercase else x
MAX_LEN = 0

basepath = "/".join(os.path.realpath(__file__).split('/')[:-1])

regex_pattern = r'Ġ?[a-zA-Z]+$'

bert_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
full_dataset = sorted(bert_tokenizer.vocab.items(), key = lambda x: x[1])

full_dataset = [x for x in full_dataset if re.match(regex_pattern, x[0]) and
                len(x[0]) > 1 and set(x[0]) != {'Ġ'}]

char_vocab = list(set([lowercase(x) for d in full_dataset for x in d[0]]))
print(char_vocab)
print("Len Char Vocab:", len(char_vocab))
char_to_id = {c:i for i,c in enumerate(char_vocab)}
id_to_char = {i:c for i,c in enumerate(char_vocab)}

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
clean_up = lambda x: x[1:] if x[0] == 'Ġ' else x

lemma_wise   = {x:[] for x in set(lemmatizer.lemmatize(clean_up(x[0])) for x in full_dataset)}
for x in full_dataset:
    lemma_wise[lemmatizer.lemmatize(clean_up(x[0]))].append(x)

flatten = lambda x: [y for xx in x for y in xx]
full_dataset = list(lemma_wise.values())

# Split the dataset
random.shuffle(full_dataset)

def shuffle_and_return(x):
  random.shuffle(x)
  return x

class SpellingDataset:
    def __init__(self):
        self.bert_tokenizer = bert_tokenizer

        self.pos_tags = {}
        self.ent_tags = {}
        self.tag_tags = {}

        self.mapper = {}
        try:
            self.mapper = json.load(open(
                                'cache.' + \
                                params.pos_path.split('/')[-1].replace("/", "_")+'.' + \
                                params.ner_path.split('/')[-1].replace("/", "_")+'.json'
                            ))
        except:
            for i, w in enumerate([x[0] for x in flatten(full_dataset)]):
                self.mapper[w] = self.convert_to_syntax_feats(w)
                if i % 1000 == 0:
                    print(i)
            json.dump((self.mapper), open(
                                'cache.' + \
                                params.pos_path.split('/')[-1].replace("/", "_")+'.' + \
                                params.ner_path.split('/')[-1].replace("/", "_")+'.json'
                            , 'w+'))
        print('done')
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        self.alphabets = string.ascii_lowercase
        
        # self.alphabet_wise_datasets = {c: (self.process(c, [x for x in full_dataset
        #                                                     if c in x[0]]),
        #                     shuffle_and_return(self.process(c, [x for x in full_dataset
        #                                                         if c not in x[0]]))
        #                 ) for c in self.alphabets}

        self.split = int(0.8 * len(full_dataset))
        random.shuffle(full_dataset)
        train_set = flatten(full_dataset[:self.split])
        test_set = flatten(full_dataset[self.split:])

        self.alphabet_wise_datasets = {c: self.split_and_process(c, train_set, test_set)
                                    for c in self.alphabets
                                }

        # import json
        # json.dump(self.alphabet_wise_datasets, open("alphabet_wise_datasets.json",'w+'))

    def split_and_process(self, c, train_data, test_data):
        train_data = self.balance_dataset(c, train_data)
        test_data = self.balance_dataset(c, test_data)
        if params.dummy_run:
            test_data = train_data
        return (self.process(c, train_data), self.process(c, test_data))

    def balance_dataset(self, c, train_set):
        splitted_set = ([x for x in train_set if c in x[0]],
                        [x for x in train_set if c not in x[0]])
        assert len(splitted_set[0]) + len(splitted_set[1]) == len(train_set)

        train_set = splitted_set[0] + splitted_set[1][:len(splitted_set[0])]
        random.shuffle(train_set)
        return train_set

    def convert_to_syntax_feats(self, word):
        p = pos_tagger(word)
        e = ner_tagger(word)
        return (p[0], e[0])

    def process(self, c, all_data):
        if params.dummy_run:
            all_data = all_data[:5]
        return [(x[0], self.mapper[x[0]],
                int(c in x[0]))
            for x in all_data]

def pad(batch):
    get_f = lambda x: [single_data[x] for single_data in batch]
    batch_tokens = get_f(0)

    pos_logits_tensor = torch.cat([torch.FloatTensor([x[0]]) for x in get_f(1)], dim=0).to(params.device)
    ner_logits_tensor = torch.cat([torch.FloatTensor([x[1]]) for x in get_f(1)], dim=0).to(params.device)
    token_ids_tensor  = (pos_logits_tensor, ner_logits_tensor)

    char_ids_tensor = torch.FloatTensor(get_f(2)).to(params.device)

    return batch_tokens, token_ids_tensor, char_ids_tensor

if __name__ == "__main__":
    dataset = SpellingDataset()
    print("Num chars:", len(dataset.alphabet_wise_datasets))
    # train_datasets = dataset.train_datasets
    # test_datasets = dataset.test_datasets

    # print("Train_dataset Size =", len(train_datasets),
    #         "Eval_dataset Size =", len(test_datasets))
    print({x[0]: len(x[1][0]) for x in dataset.alphabet_wise_datasets.items()})
    print('\n')
    print({x[0]: len(x[1][1]) for x in dataset.alphabet_wise_datasets.items()})

    print(dataset.alphabet_wise_datasets['a'][0][:5])
    print(dataset.alphabet_wise_datasets['a'][1][:5])
