# Load Packages and setup wandb
from params import params
import wandb
if params.wandb and not params.dummy_run:
    wandb.init(project="bert_gptj_pos_ner",
        name=params.task+'.'+params.task_level+'.'+\
            params.bert_type.split('/')[-1]+'.'+str(params.lr)+'.'+\
            str(params.batch_size))
    wandb.config.update(params)

from bertloader import DatasetClass
import json, os, random

import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW, AutoModel, AutoConfig

from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for batch in dataset:
        (texts, labels, att_masks, loss_masks) = batch
        preds = model(texts, att_masks)
        loss = criterion(preds, labels, loss_masks)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        if num_batch % 100 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

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
        for batch in dataset:
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

    assert len(predicts) == len(gnd_truths)

    confuse_mat = confusion_matrix(gnd_truths, predicts)
    if params.dummy_run:
        classify_report = {"hi": {"fake": 1.2}}
    else:
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
dataset_object = DatasetClass()
train_dataset = dataset_object.train_dataset
eval_dataset = dataset_object.eval_dataset
target_names = [x[0] for x in sorted(list(dataset_object.target_names.items()),key=lambda x: x[1])]
print(target_names)

if params.dummy_run:
    eval_dataset = train_dataset

print("Dataset created")
os.system("nvidia-smi")


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
print("Detected", torch.cuda.device_count(), "GPUs!")

import os
print("Model created")
os.system("nvidia-smi")

if params.wandb:
    wandb.watch(model)

########## Optimizer & Loss ###########

#criterion = torch.nn.CrossEntropyLoss(weight=dataset_object.criterion_weights, reduction='sum')
loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

if params.task_level == "sentence":
    criterion = lambda _preds, _labels, _att_mask: (loss_fn(_preds.permute(0,2,1), _labels) * _att_mask).sum()/_att_mask.sum()
else:
    criterion = lambda _preds, _labels, _att_mask: (loss_fn(_preds, _labels) * _att_mask).sum()/_att_mask.sum()
optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)

# valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)
    if not params.dummy_run:
        print("EVALUATING:")
        valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)
    else:
        valid_loss = 0.0

    if not params.dummy_run and params.wandb:
        wandb_dict = {}
        for labl in target_names:
            for metric, val in classify_report[labl].items():
                if metric != "support":
                    wandb_dict[labl + "_" + metric] = val

        wandb_dict["F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
        wandb_dict["F1-Avg"] = classify_report["macro avg"]["f1-score"]

        wandb_dict["Accuracy"] = classify_report["accuracy"]

        wandb_dict["Train_loss"] = train_loss
        wandb_dict["Valid_loss"] = valid_loss

        wandb.log(wandb_dict)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

if not params.dummy_run:
    folder_name = 'savefolder'
    print(folder_name)
    if os.path.isdir(folder_name):
        os.system("rm -rf " + folder_name)
    os.mkdir(folder_name)

    # Store params
    json.dump(vars(params), open(os.path.join(folder_name, "params.json"), 'w+'))

    # Save model
    torch.save(model.state_dict(), os.path.join(folder_name, "model.pt"))

    # Store logs (accuracy)
    logs = {"Accu:": classify_report["accuracy"],
            "F1-Weighted": classify_report["weighted avg"]["f1-score"],
            "F1-Avg": classify_report["macro avg"]["f1-score"]
        }
    json.dump(logs, open(os.path.join(folder_name, "logs.json"), 'w+'))

