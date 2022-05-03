import torch
import os, json, random, sys
from params import params
import numpy as np
from transformers import AutoTokenizer

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

MAX_LEN = 0

DATA_PATH = "CONLL2003/"

class DatasetClass:
    def __init__(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
        print("Loaded Bert Tokenizer")

        try:
            self.train_data, self.test_data = json.load(open("cache"+params.task+".json"))
        except:
            self.train_data = self.read_data(DATA_PATH + "train.txt") + \
                                self.read_data(DATA_PATH + "valid.txt")
            self.test_data = self.read_data(DATA_PATH + "test.txt")
            json.dump([self.train_data, self.test_data], open("cache"+params.task+".json", "w+"))

        self.target_names = {k:i for i, k in enumerate(sorted(list(set(
                    [xx[1][0] for x in self.train_data for xx in x]))))}
        print(self.target_names)

        if params.dummy_run == True:
            self.train_dataset = self.batched_dataset([self.train_data[0]] * 2)
            self.eval_dataset = self.batched_dataset([self.train_data[0]] * 2)
        else:
            print("Train_dataset:", end= " ")
            self.train_dataset = self.batched_dataset(self.train_data)
            print("Eval_dataset:", end= " ")
            self.eval_dataset = self.batched_dataset(self.test_data)

    def read_data(self, path):
        raw_data = open(path).readlines()[2:]
        loaded_data = []
        this_sent = []

        for line in raw_data:
            if len(line.strip()) < 2:
                if len(this_sent) != 0:
                    loaded_data.append(this_sent)
                    this_sent = []
            else:
                line_word, line_pos, _, line_ner = line.strip().split()
                line_ner = line_ner.split('-')[-1]
                if line_pos == '-X-':
                    continue
                if params.task == 'ner':
                    line_label = line_ner
                else:
                    line_label = line_pos
                toks = self.bert_tokenizer.tokenize(line_word)
                this_sent.append([toks, len(toks) * [line_label]])

        if len(this_sent) != 0:
            loaded_data.append(this_sent)
            this_sent = []

        return loaded_data

    def batched_dataset(self, unbatched): # For batching full or a part of dataset.
        dataset = []

        idx = 0
        num_data = len(unbatched)

        if 'bert' in params.bert_type.lower():
            cls_tok_id = self.bert_tokenizer.convert_tokens_to_ids(["CLS"])
            sep_tok_id = self.bert_tokenizer.convert_tokens_to_ids(["CLS"])
            pad_ids = [0]
        else:
            # We will pad these later anyways
            cls_tok_id = [100]
            sep_tok_id = [100]
            pad_ids = self.bert_tokenizer.convert_tokens_to_ids(['<|endoftext|>'])
        extra_tok_label = [0]

        while idx < num_data:
            batch_text = []
            batch_labels = []
            
            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                sent_labels = extra_tok_label + \
                    [self.target_names[xx] for x in single_tweet for xx in x[1]] + \
                        extra_tok_label

                sent_toks = cls_tok_id + \
                    [self.bert_tokenizer.convert_tokens_to_ids(t) for line in single_tweet for t in line[0]] + \
                        sep_tok_id
                batch_text.append(sent_toks)
                batch_labels.append(sent_labels)

            max_toks = max([len(x) for x in batch_text])

            att_masks_padded = [ ([1] * len(x)) + ([0] * (max_toks - len(x))) for x in batch_text]
            loss_masks_padded = [[0] + [1] * (len(x)-2) + ([0] * (max_toks - len(x) + 1)) for x in batch_text]
            texts_padded = [x + (pad_ids * (max_toks - len(x))) for x in batch_text]
            labels_padded = [x + (extra_tok_label * (max_toks - len(x))) for x in batch_labels]

            texts = torch.LongTensor(texts_padded).to(params.device)
            labels = torch.LongTensor(labels_padded).to(params.device)
            att_masks = torch.LongTensor(att_masks_padded).to(params.device)
            loss_masks = torch.LongTensor(loss_masks_padded).to(params.device)

            global MAX_LEN
            MAX_LEN = max(MAX_LEN, texts.shape[1])

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            l = texts.size(1)

            assert texts.size() == torch.Size([b, l])
            assert labels.size() == torch.Size([b, l])
            assert att_masks.size() == torch.Size([b, l])
            assert loss_masks.size() == torch.Size([b, l])

            if params.task_level == "token":
                dataset.append((torch.flatten(texts).unsqueeze(-1), torch.flatten(labels),
                                None, torch.flatten(loss_masks)))
            else:
                dataset.append((texts, labels, att_masks, loss_masks))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        print(MAX_LEN)
        return dataset

if __name__ == "__main__":
    dataset = DatasetClass()
    print(dataset.target_names)
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    print(len(dataset.train_dataset))#[0])
    print(dataset.train_dataset[-1])
    #print(len(dataset.hard_dataset))
    import os
    os.system("nvidia-smi")
    print(MAX_LEN)
