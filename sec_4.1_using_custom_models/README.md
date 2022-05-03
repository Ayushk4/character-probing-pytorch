# Sec 4.1 Predicting Characters using Custom Models

## Dependencies

- [PyTorch >= 1.7.1](https://pytorch.org/get-started/previous-versions/)
- [Transformers > 4.10](https://huggingface.co/docs/transformers/installation)
- [Sentencepiece >= 0.1.91](https://pypi.org/project/sentencepiece/)
- [Scikit-Learn >= 0.24.2](https://scikit-learn.org/stable/install.html)
- [Wandb >= 0.12.2](https://docs.wandb.ai/quickstart#1.-set-up-wandb)


## Setting up


1. Install above dependencies, you may also install cudatoolkit if needed.
1. Download and unzip [PoS](https://github.com/Anonymous-ARR/Releases/releases/download/v4.1/pos_models.zip) and [NER](https://github.com/Anonymous-ARR/Releases/releases/download/v4.1/ner_models.zip) trained models into folders `pos_models` and `ner_models` within this folder.
1. Make sure the folders for the 6 models are in their respective folder - `bert_token`, `bert_sentence`, `gpt6b_model` in `pos_models/` and `ner_bert_token`, `ner_bert_sentence`, `ner_gptj` in `ner_models/`
1. Download [GPT-J's Embeddings](https://github.com/Anonymous-ARR/Releases/releases/download/gptj/gpt-j-6B.Embedding.pth) into the current folder.

You may train the model using NER and PoS from these custom pretrained models using - `python3 train.py --pos --pos_path <PATH_TO_POS_MODEL> --ner --ner_path <PATH_TO_NER_MODEL> --batch <BATCH_SIZE> --wandb --lr=<LEARNING_RATE> --seed=<SEED>`. The Wandb command is optional and using it will allow you to use WANDB for logging.

For example for the gptj model you can use:
`python3 train.py --pos --pos_path pos_models/gpt6b_model --ner --ner_path ner_models/ner_gptj --batch 64 --wandb --lr=1e-4 --seed=2714`

If you are using `cpu`, set also include `--device=cpu`.

It takes about 5-10 minutes to compute the POS+NER label distribution over the entire vocab of GPT-J. Do not end the pos_path or ner_path with a '/'.

Once the POS+NER label for any model-pair is cached, it takes less than 5 minutes per run.

It is recommended to clear cache each time you change the `--pos_path` and `--ner_path` arguments. As it may use the previous models' cache instead of the new one.

## Results

Each time you run the experiment, all the results will be printed at the end of the run, and also on wandb if you include the flag.

Averaged across 5 seeds/train-test splits

| Measure | Bert-Sent | Bert-Token | GPTJ |
| ------- | --------- | ---------- | ---- |
| F1 Aggr.|  55.00 | 59.75 | 61.24 |
| ------- | ------ | ----- | ----- |
| s       |  60.72 | 70.33 | 66.82 |
| y       |  60.39 | 67.16 | 64.89 |
| e       |  57.75 | 64.62 | 62.32 |
| t       |  54.38 | 64.07 | 60.73 |
| p       |  55.24 | 63.97 | 60.51 |
| i       |  56.41 | 63.35 | 61.60 |
| ------- | ------ | ----- | ----- |
| w       |  52.72 | 57.69 | 58.27 |
| q       |  56.53 | 57.54 | 53.54 |
| k       |  49.38 | 57.31 | 55.96 |
| o       |  53.61 | 56.83 | 55.63 |
| b       |  56.74 | 56.39 | 55.13 |
| m       |  53.40 | 56.28 | 55.61 |

The exact command to get these are:
`python3 train.py --pos --pos_path pos_models/bert_sentence --ner --ner_path ner_models/ner_bert_sentence --batch 64 --wandb --lr=1e-3`

`python3 train.py --pos --pos_path pos_models/bert_token --ner --ner_path ner_models/ner_bert_token --batch 64 --wandb --lr=3e-3`

`python3 train.py --pos --pos_path pos_models/gpt6b_model --ner --ner_path ner_models/ner_gptj --batch 64 --wandb --lr=1e-4`

Run the above commands on multiple (5) seeds and average them.
