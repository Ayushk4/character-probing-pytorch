import torch
from torch import nn
import json
from params import params
from transformers import AutoModel, AutoTokenizer, AutoConfig

class BERT_Tagger(nn.Module):
    def __init__(self, num_labels, task_level, bert_type):
        super(BERT_Tagger, self).__init__()
        self.task_level = task_level
        self.bert = AutoModel.from_pretrained(bert_type)
        self.drop = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, text, att_mask):
        if self.task_level == "sentence":
            output = self.bert(text, attention_mask=att_mask,
                            output_hidden_states=True)
            output = output.last_hidden_state
            return self.classifier(self.drop(output)).mean(1).squeeze(0)
        else:
            assert text.shape[0] == 1 and len(text.shape) == 2
            text = text.reshape(-1, 1)
            output = self.bert(text, attention_mask=None,
                                output_hidden_states=True)
            output = output.pooler_output
            return self.classifier(self.drop(output)).mean(0)


class GPTJ_Tagger(nn.Module):
    def __init__(self, num_labels, task_level, bert_type):
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


class POS_NER_Tagger(nn.Module):
    def __init__(self, pos_path):
        super(POS_NER_Tagger, self).__init__()
        meta_data = json.load(open(pos_path + "/meta.json"))
        self.meta = meta_data
        if meta_data['model'] == 'GPTJ_Tagger':
            self.pos_model = GPTJ_Tagger(meta_data['num_labels'],
                            meta_data['task_level'], meta_data['bert_type']).to(params.device)
            self.word_tokenize = lambda x: self.pos_tokenizer.convert_tokens_to_ids(x)
        else:
            self.pos_model = BERT_Tagger(meta_data['num_labels'],
                            meta_data['task_level'], meta_data['bert_type']).to(params.device)
            self.word_tokenize = lambda x: self.pos_tokenizer.convert_tokens_to_ids(self.pos_tokenizer.tokenize(x.replace(r'Ġ', '')))

        meta_data = json.load(open(pos_path + "/meta.json"))
        self.pos_model.load_state_dict(torch.load(pos_path + '/model.pt', map_location=torch.device(params.device)))
        self.pos_tokenizer = AutoTokenizer.from_pretrained(meta_data['bert_type'])
        
        # if meta_data['model'] == 'GPTJ_Tagger':
        #     self.ner_model = GPTJ_Tagger(meta_data['num_labels'],
        #                     meta_data['task_level'], meta_data['bert_type']) 
        # else:
        #     self.ner_model = BERT_Tagger(meta_data['num_labels'],
        #                     meta_data['task_level'], meta_data['bert_type']) 

        # meta_data = json.load(open(ner_path + "/meta.json"))
        # self.ner_model.load_state_dict(torch.load(ner_path + '/model.pt'))
        # self.ner_tokenizer = AutoTokenizer.from_pretrained(meta_data['bert_type'])

    def forward(self, input_string):
        if self.meta['model'] == 'GPTJ_Tagger':
            pos_ip = [self.word_tokenize(input_string)]
            # print(pos_ip, "\n")
            pos_op = self.pos_model(torch.LongTensor([pos_ip]).to(params.device), None)
            return pos_op.detach().cpu().tolist()

        else:
            pos_ip = [self.word_tokenize(input_string)]
            pos_ip = torch.LongTensor(pos_ip).to(params.device)
            pos_op = self.pos_model(pos_ip, None)
            return [pos_op.detach().cpu().tolist()]


