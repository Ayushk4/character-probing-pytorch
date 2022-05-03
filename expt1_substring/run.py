# Load Packages and setup wandb
from params import params
import dataloader
from dataloader import SpellingDataset
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

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for i in range(len(dataset))[::params.batch_size]:
        (batch_tokens, superstring_batch, substring_batch, label_batch, onehot_superstring, onehot_substring) = dataloader.pad(dataset[i:i+params.batch_size])
        if params.have_char_embeds:
            preds = model(onehot_superstring, onehot_substring)
        else:
            preds = model(superstring_batch, substring_batch)
        loss  = criterion(torch.flatten(preds), torch.flatten(label_batch))

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
            (batch_tokens, superstring_batch, substring_batch, label_batch, onehot_superstring, onehot_substring) = dataloader.pad(dataset[i:i+params.batch_size])
            if params.have_char_embeds:
                preds = model(onehot_superstring, onehot_substring)
            else:
                preds = model(superstring_batch, substring_batch)
            loss = criterion(torch.flatten(preds), torch.flatten(label_batch))

            predicts.extend([int(x) for x in torch.flatten(preds > 0).tolist()])
            gnd_truths.extend([int(x) for x in label_batch.tolist()])
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
train_dataset = dataset.train_set
test_dataset = dataset.test_set
print(len(train_dataset), " datasets of Sizes:", len(test_dataset))

print("Dataset created")
os.system("nvidia-smi")


########## Create model #############
if not params.have_char_embeds:
    trained_embeddings = torch.load("gpt-j-6B.Embedding.pth")
    if params.control:
        trained_embeddings = torch.normal(0, 0.01, size=(trained_embeddings.shape[0], trained_embeddings.shape[1]))

class SpellingModel(nn.Module):
    def __init__(self):
        super(SpellingModel, self).__init__()
        if params.have_char_embeds:
            self.n_dims = 1000
            # Actually not frozen
            self.frozen_embeddings = nn.Sequential(nn.ReLU(),
                                    nn.Linear(52, self.n_dims)
                                )    
        else:
            self.gptj_config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B')
            assert self.gptj_config.vocab_size == trained_embeddings.shape[0], (self.gptj_config.vocab_size, trained_embeddings.shape)

            self.frozen_embeddings = nn.Embedding.from_pretrained(trained_embeddings, freeze=True)
            print(self.frozen_embeddings.weight.shape)

            self.n_dims = trained_embeddings.shape[1]

        self.ff = nn.Sequential(nn.Linear(2*self.n_dims, self.n_dims),
                                nn.SELU(),
                                nn.Linear(self.n_dims, self.n_dims),
                                nn.Tanh(), nn.Dropout(0.1),
                                nn.Linear(self.n_dims, 1)
                            )

    def forward(self, superstrings, substrings):

        mlp_ip = torch.cat([self.frozen_embeddings(superstrings),
                            self.frozen_embeddings(substrings),
                        ], dim=1)
        return self.ff(mlp_ip)


# model = torch.nn.DataParallel(model)
########## Optimizer & Loss ###########

# valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

try:
    os.mkdir('preds')
except:
    pass
folder = 'preds'

model = SpellingModel()
model = model.to(params.device)
print("Model created")

criterion = torch.nn.BCEWithLogitsLoss().to(params.device)
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)

    print("EVALUATING:")
    valid_loss, confuse_mat, classify_report = evaluate(model, train_dataset,
                                                        criterion)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

# Store preds
print("EVALUATING:")
valid_loss, confuse_mat, classify_report = evaluate(model, test_dataset, criterion)
json.dump([valid_loss, confuse_mat.tolist(), classify_report], open(folder + f"/values.json", 'w+'))

model.eval()
tokens = []
predicts = []
logits = []
gnd_truths = []

with torch.no_grad():
    for i in range(len(test_dataset))[::params.batch_size]:
        (batch_tokens, superstring_batch, substring_batch, label_batch, onehot_superstring, onehot_substring) = dataloader.pad(test_dataset[i:i+params.batch_size])
        if params.have_char_embeds:
            preds = model(onehot_superstring, onehot_substring)
        else:
            preds = model(superstring_batch, substring_batch)
        logits.extend(preds.tolist())
        predicts.extend([int(x) for x in torch.flatten(preds > 0).tolist()])
        gnd_truths.extend([int(x) for x in label_batch.tolist()])
        tokens.extend(batch_tokens)

json.dump([predicts, gnd_truths, logits, tokens], open(folder + f"/preds.json", 'w+'))
torch.save(model, "model.pt")
