# Load Packages and setup wandb
from params import params
from torch.utils import data
import loader
from loader import SpellingDataset
import json, os, random

import string
import torch
import torch.nn as nn
import numpy as np

from transformers import AutoConfig

from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

from params import params

from sklearn.metrics import confusion_matrix, classification_report

import wandb

if not params.dummy_run and params.wandb:
    run_name = "ctrl" if params.control else "pred"
    if params.pos:
        run_name += ".pos"
    if params.ner:
        run_name += ".ner"
    if params.tag:
        run_name += ".tag"

    run_name += "." + str(params.lr) + "." + str(params.batch_size) + "." + str(params.n_epochs)

    wandb.init(project="spacy_pos_second", name=run_name)
    wandb.config.update(params)

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for i in range(len(dataset))[::params.batch_size]:
        (batch_tokens, token_ids_tensor, char_label_tensor) = loader.pad(dataset[i:i+params.batch_size])
        preds = model(token_ids_tensor)
        loss = criterion(torch.flatten(preds), torch.flatten(char_label_tensor))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        if num_batch % 50 == 0:
            print("Train loss at {}:".format(num_batch), loss.item(), len(batch_tokens))

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion):
    model.eval()
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for i in range(len(dataset))[::params.batch_size]:
            (batch_tokens, token_ids_tensor, char_label_tensor) = loader.pad(dataset[i:i+params.batch_size])
            preds = model(token_ids_tensor)
            loss = criterion(torch.flatten(preds), torch.flatten(char_label_tensor))

            predicts.extend([int(x) for x in torch.flatten(preds > 0).tolist()])
            gnd_truths.extend([int(x) for x in char_label_tensor.tolist()])
            valid_losses.append(loss.item())
    print()
    assert len(predicts) == len(gnd_truths)
    assert type(predicts[0]) == int

    target_names = ["Neg", "Pos"]
    confuse_mat = confusion_matrix(gnd_truths, predicts)
    classify_report = classification_report(gnd_truths, predicts,
                                    target_names=target_names,
                                    output_dict=True)

    mean_valid_loss = np.average(valid_losses)
    print("Valid_loss", mean_valid_loss)
    print(confuse_mat)

    for labl in ["Pos"]:
        print(labl, classify_report[labl])
    print("F1-Avg", classify_report["macro avg"]["f1-score"])

    return mean_valid_loss, confuse_mat ,classify_report


########## Load dataset #############
dataset = SpellingDataset()
print(len(dataset.pos_tags), len(dataset.tag_tags), len(dataset.ent_tags))
dataset = dataset.alphabet_wise_datasets
print(len(dataset)," datasets of Sizes:", len(dataset['a'][0]), len(dataset['a'][1]))
# assert len(set(len(x[0]) for x in dataset.values())) == 1, set(len(x[0]) for x in dataset.values())
# assert len(set(len(x[1]) for x in dataset.values())) == 1, set(len(x[1]) for x in dataset.values())

print("Dataset created")
os.system("nvidia-smi")


########## Create model #############

class SpellingModel(nn.Module):
    def __init__(self):
        super(SpellingModel, self).__init__()
        self.n_dims = 100
        self.pos_embeddings = nn.Embedding(100, self.n_dims)
        self.ner_embeddings = nn.Embedding(100, self.n_dims)
        self.tag_embeddings = nn.Embedding(100, self.n_dims)

        self.num_feats = int(params.ner) + int(params.pos) + int(params.tag)
        self.ff = nn.Sequential(nn.Linear(self.num_feats*self.n_dims, self.n_dims),
                                nn.SELU(),
                                nn.Linear(self.n_dims, self.n_dims),
                                nn.Tanh(), nn.Dropout(0.1),
                                nn.Linear(self.n_dims, 1)
                            )

    def forward(self, vocab_ids):
        mlp_ip = torch.cat([self.pos_embeddings(vocab_ids[i])
                             for i, bool_value in enumerate([params.ner, params.pos, params.tag])
                             if bool_value
                ], dim=1)
        return self.ff(mlp_ip)


# valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

folder = 'control' if params.control else 'preds'
try:
    os.mkdir(folder)
except:
    pass

test_dicts = {}
dev_dicts = {}

import json
json.dump(dataset, open(folder + "/dataset.json", 'w+'))
for c in string.ascii_lowercase:
    # if c != 'z': continue
    print("###########")
    print("Starting:", c)
    print("###########")

    model = SpellingModel()

    # print(sum(p.numel() for p in model.parameters()))
    model = model.to(params.device)
    # print("Detected", torch.cuda.device_count(), "GPUs!")
    print("Model created")

    criterion = torch.nn.BCEWithLogitsLoss().to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    split = int(0.8 * len(dataset[c][0]))
    this_train_set = dataset[c][0][:split]
    this_val_set = dataset[c][0][split:]

    for epoch in range(params.n_epochs):
        print("\n\n========= Beginning", epoch+1, "epoch ==========")

        train_loss = train(model, this_train_set, criterion)

        print("EVALUATING on Train set:")
        valid_loss, confuse_mat, classify_report = evaluate(model,
                                                        this_train_set,
                                                        criterion)

        print("EVALUATING on Valid set:")
        valid_loss, confuse_mat, classify_report = evaluate(model,
                                                        this_val_set,
                                                        criterion)
        dev_dicts[c] = classify_report
        epoch_len = len(str(params.n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        if not params.dummy_run and params.wandb:
            wandb_dict = {}
            for labl in ["Pos"]:
                for metric, val in classify_report[labl].items():
                    if 'f1' in metric:
                        wandb_dict[c + "_Valid_" + labl + "_" + metric] = val

            wandb_dict[c+"_Valid_F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
            wandb_dict[c+"_Valid_F1-Avg"] = classify_report["macro avg"]["f1-score"]
            # wandb_dict[c+"_Valid_Accuracy"] = classify_report["accuracy"]

            wandb_dict[c+"_Train_Loss"] = train_loss
            wandb_dict[c+"_Valid_Loss"] = valid_loss

            wandb.log(wandb_dict)

    # Store preds
    print("EVALUATING:")
    valid_loss, confuse_mat, classify_report = evaluate(model, dataset[c][1], criterion)
    test_dicts[c] = classify_report

    if not params.dummy_run and params.wandb:
        wandb_dict = {}
        for labl in ["Pos"]:
            for metric, val in classify_report[labl].items():
                if 'f1' in metric:
                    wandb_dict[c + "_Test_" + labl + "_" + metric] = val

        wandb_dict[c+"_Test_F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
        wandb_dict[c+"_Test_F1-Avg"] = classify_report["macro avg"]["f1-score"]
        # wandb_dict[c+"_Test_Accuracy"] = classify_report["accuracy"]

        wandb.log(wandb_dict)

    json.dump([valid_loss, confuse_mat.tolist(), classify_report], open(folder + f"/values_{c}.json", 'w+'))

    model.eval()
    tokens = []
    predicts = []
    logits = []
    gnd_truths = []

    test_dataset = dataset[c][1]
    with torch.no_grad():
        for i in range(len(test_dataset))[::params.batch_size]:
            (batch_tokens, token_ids_tensor, char_label_tensor) = loader.pad(test_dataset[i:i+params.batch_size])
            preds = model(token_ids_tensor)
            logits.extend(preds.tolist())
            predicts.extend([int(x) for x in torch.flatten(preds > 0).tolist()])
            gnd_truths.extend([int(x) for x in char_label_tensor.tolist()])
            tokens.extend(batch_tokens)

    json.dump([predicts, gnd_truths, logits, tokens, loader.char_to_id],
            open(folder + f"/preds_{c}.json", 'w+'))
    # torch.save(model, "model.pt")

assert len(test_dicts) == 26

# Following 1-liner is the worst piece of code you will ever see.
test_dicts_aggr = {k1 :
                    {k2: np.mean([single_dict[k1][k2]
                            for single_dict in test_dicts.values()])
                        for k2 in test_dicts['a'][k1].keys()}
                    if type(test_dicts['a'][k1]) == dict
                    else np.mean([single_dict[k1] for single_dict in test_dicts.values()])
                for k1 in test_dicts['a'].keys()}

if not params.dummy_run and params.wandb:
    wandb_dict = {}
    for labl in ["Pos"]:
        for metric, val in classify_report[labl].items():
            if 'f1' in metric:
                wandb_dict["Aggr__Test" + labl + "_" + metric] = val

    wandb_dict["Aggr_Test_F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
    wandb_dict["Aggr_Test_F1-Avg"] = classify_report["macro avg"]["f1-score"]
    # wandb_dict[c+"_Test_Accuracy"] = classify_report["accuracy"]

    wandb.log(wandb_dict)

### Do the same for Dev

test_dicts = dev_dicts
# Following 1-liner is the worst piece of code you will ever see.
test_dicts_aggr = {k1 :
                    {k2: np.mean([single_dict[k1][k2]
                            for single_dict in test_dicts.values()])
                        for k2 in test_dicts['a'][k1].keys()}
                    if type(test_dicts['a'][k1]) == dict
                    else np.mean([single_dict[k1] for single_dict in test_dicts.values()])
                for k1 in test_dicts['a'].keys()}


if not params.dummy_run and params.wandb:
    wandb_dict = {}
    for labl in ["Pos"]:
        for metric, val in classify_report[labl].items():
            if 'f1' in metric:
                wandb_dict["Aggr_Dev" + labl + "_" + metric] = val

    wandb_dict["Aggr_Dev_F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
    wandb_dict["Aggr_Dev_F1-Avg"] = classify_report["macro avg"]["f1-score"]
    # wandb_dict[c+"_Test_Accuracy"] = classify_report["accuracy"]

    wandb.log(wandb_dict)

