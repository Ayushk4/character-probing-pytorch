# Sec 4.1 Using SpaCy for Predicting Characters in Tokens

## Dependencies

- [PyTorch >= 1.7.1](https://pytorch.org/get-started/previous-versions/)
- [Transformers > 4.10](https://huggingface.co/docs/transformers/installation)
- [Sentencepiece >= 0.1.91](https://pypi.org/project/sentencepiece/)
- [Scikit-Learn >= 0.24.2](https://scikit-learn.org/stable/install.html)
- [Wandb >= 0.12.2](https://docs.wandb.ai/quickstart#1.-set-up-wandb)
- [SpaCy >= 3.x](https://spacy.io/usage)

## Setting up

Install above dependencies, you may also install cudatoolkit if needed and SpaCy en web core, if needed.

`python3 train.py --pos --ner --tag --batch=[BATCH_SIZE] --wandb --lr=[LEARNING_RATE] --seed=[SEED] --n_epoch=[NUM_EPOCHS]`

Wandb command is optional. You need to input at least one of {`--pos`, `--ner`, `--tag`}, which enables the consideration of fine-grained pos, ner and coarse-grained pos tags features respectively.

**Control task:** include the flag `--flag` in the previous command to perform the control task.

If you are using `cpu`, set also include `--device=cpu`.

It takes about 10 minutes to compute the SpaCy ner labels distribution over the entire vocab of GPT-J. Do not end the pos_path or ner_path with a '/'.

Once the POS+NER label for any model-pair is cached, it takes less than 5 minutes per run.

It is recommended to clear cache each time you switch from SpaCy to control task or vice versa, as it may use the previous cache instead of creating new one.

## Results

Each time you run the experiment, all the results will be printed at the end of the run, and also on wandb if you include the flag.

Averaged across 5 seeds/train-test splits

| Measure | SpaCy | Control |
| ------- | ----- | ------- |
| F1 Aggr.| 52.34 | 49.68   |
| ------- | ----- | ------- |
| s       | 64.60 | 40.32   |
| y       | 61.96 | 48.68   |
| e       | 62.05 | 47.27   |
| t       | 60.68 | 48.49   |
| p       | 50.24 | 46.56   |
| i       | 60.80 | 42.82   |
| ------- | ----- | ------- |
| w       | 45.75 | 48.69   |
| q       | 43.79 | 49.28   |
| k       | 47.79 | 46.24   |
| o       | 52.94 | 43.59   |
| b       | 48.92 | 48.25   |
| m       | 48.13 | 46.11   |

The exact command to get these are:
`python3 train.py --pos --ner --tag --batch=128 --wandb --lr=1e-3`

`python3 train.py --pos --ner --tag --batch=128 --wandb --lr=1e-2 --control`

Run the above commands on multiple (5) seeds and average them.
