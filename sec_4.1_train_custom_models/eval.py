from eval_params import params
from transformers import AdamW, AutoModel, AutoConfig, AutoTokenizer

from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import numpy as np

def evaluate(model, dataset, criterion, target_names):
    model.eval()
    valid_losses = []
    if params.task == 'ner':
        num_tags = 5
    else:
        num_tags = 45
    predicts = [i for i in range(num_tags)]
    gnd_truths = [i for i in range(num_tags)]

    with torch.no_grad():
        for i, batch in enumerate(dataset):
            (texts, labels, att_masks, loss_masks) = batch
            preds = model(texts, att_masks)
            loss = criterion(preds, labels, loss_masks)

            if params.task_level == "sentence":
                for sent_pred, sent_pad, sent_labels in zip(preds, loss_masks, labels):
                    predicts.extend([x for x, p in
                                zip(torch.max(sent_pred, axis=1)[1].tolist(),
                                    sent_pad.tolist()
                                ) if p == 1])
                    gnd_truths.extend([x for x, p in
                                zip(sent_labels.tolist(),
                                    sent_pad.tolist()
                                ) if p == 1])
            else:
                predicts.extend([x for x, p in
                                zip(torch.max(preds, axis=1)[1].tolist(),
                                    loss_masks.tolist()
                                ) if p == 1])
                gnd_truths.extend([x for x, p in
                                zip(labels.tolist(),
                                    loss_masks.tolist()
                                ) if p == 1])
            valid_losses.append(loss.item())

            if i % 20 == 0:
                print(i)

    assert len(predicts) == len(gnd_truths)

    confuse_mat = confusion_matrix(gnd_truths, predicts)
    classify_report = classification_report(gnd_truths, predicts, target_names=target_names, output_dict=True)

    mean_valid_loss = np.average(valid_losses)
    print("Valid_loss", mean_valid_loss)
    print(confuse_mat)

    for labl in target_names:
        print(labl, "F1-score:", classify_report[labl]["f1-score"])
    print("Accu:", classify_report["accuracy"])
    print("F1-Weighted", classify_report["weighted avg"]["f1-score"])
    print("F1-Avg", classify_report["macro avg"]["f1-score"])

    return mean_valid_loss, confuse_mat ,classify_report

############# Load dataset #############

MAX_LEN = 0
DATA_PATH = "CONLL2003/"

class DatasetClass:
    def __init__(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
        print("Loaded Bert Tokenizer")

        self.valid_data = self.read_data(DATA_PATH + "valid.txt")
        self.test_data = self.read_data(DATA_PATH + "test.txt")

        self.target_names = {x.strip(): i for i, x in enumerate(params.target_names.split("-"))}

        print("Valid_dataset:", end= " ")
        self.valid_dataset = self.batched_dataset(self.valid_data)
        print("Test_dataset:", end= " ")
        self.test_dataset = self.batched_dataset(self.test_data)

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

dataset_object = DatasetClass()
valid_dataset = dataset_object.valid_dataset
test_dataset = dataset_object.test_dataset
target_names = [x.strip() for x in params.target_names.split("-")]
print(target_names)

############# Create model #############

class BERT_Tagger(nn.Module):
    def __init__(self, num_labels):
        super(BERT_Tagger, self).__init__()
        self.bert = AutoModel.from_pretrained(params.bert_type)
        self.drop = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, text, att_mask):
        output = self.bert(text, attention_mask=att_mask,
                            output_hidden_states=True)
        if params.task_level == "sentence":
            output = output.last_hidden_state
        else:
            output = output.pooler_output
        return self.classifier(self.drop(output))

class GPTJ_Tagger(nn.Module):
    def __init__(self, num_labels):
        super(GPTJ_Tagger, self).__init__()
        trained_embeddings = torch.load("gpt-j-6B.Embedding.pth")
        self.gptj_config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B')
        assert self.gptj_config.vocab_size == trained_embeddings.shape[0], (self.gptj_config.vocab_size, trained_embeddings.shape)

        self.frozen_embeddings = nn.Embedding.from_pretrained(trained_embeddings, freeze=True)
        print(self.frozen_embeddings.weight.shape)

        self.n_dims = trained_embeddings.shape[1]

        self.ff = nn.Sequential(nn.Linear(self.n_dims, self.n_dims),
                                nn.SELU(),
                                nn.Linear(self.n_dims, self.n_dims),
                                nn.Tanh(), nn.Dropout(0.1),
                                nn.Linear(self.n_dims, num_labels)
                            )

    def forward(self, text, att_mask):
        embeds = self.frozen_embeddings(text.squeeze(1))
        return self.ff(embeds)

if "EleutherAI" in params.bert_type:
    model = GPTJ_Tagger(len(target_names))
else:
    model = BERT_Tagger(len(target_names))

print(sum(p.numel() for p in model.parameters()))

model = model.to(params.device)
model.load_state_dict(torch.load(params.weight_path, map_location=torch.device(params.device)))
model.eval()
print("Detected", torch.cuda.device_count(), "GPUs!")

################# Optimizer & Loss ####################

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

if params.task_level == "sentence":
    criterion = lambda _preds, _labels, _att_mask: (loss_fn(_preds.permute(0,2,1), _labels) * _att_mask).sum()/_att_mask.sum()
else:
    criterion = lambda _preds, _labels, _att_mask: (loss_fn(_preds, _labels) * _att_mask).sum()/_att_mask.sum()

########## Performance on Valid & Test Split ###########

print("\n\nValid set\n")
valid_loss, confuse_mat, classify_report = evaluate(model, valid_dataset, criterion, target_names)

print("\n\nTest set\n")
test_loss, confuse_mat, classify_report = evaluate(model, test_dataset, criterion, target_names)

